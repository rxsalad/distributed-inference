# vLLM 数据并行（DP）部署

> 来源：<https://docs.vllm.ai/en/latest/serving/data_parallel_deployment>（`latest` 版本，flag 可能随版本变化，细节以原页面为准）。官方页面讲部署，不讲原理，也未提到具体通信原语（All-to-All / All-Gather / Reduce-Scatter）。
>
> 本文在官方文档整理的基础上，补充了在本环境（vLLM 0.19.2rc1，Qwen3-8B，B300 节点）的实测结果。**标注约定**：
> - **[实测]**：在本环境跑过并观察到的现象；
> - **[help]**：在本环境 `vllm serve --help=all` 中核对过参数存在；
> - **[源码]**：读过本环境 vLLM 源码；
> - 其余没有标注的内容来自官方文档，本环境未验证。
>
> 实验记录：
> - 单节点 DP=4、TP=2：`vllm-dp4-tp2-single-node.md`
> - 双节点 DP=4、TP=1：`vllm-dp4-tp1-two-node.md`

``` bash
kubectl get pods -o wide
NAME       READY   STATUS    RESTARTS   AGE    IP              NODE                        NOMINATED NODE   READINESS GATES
server-0   1/1     Running   0          142m   10.154.15.24    rs-b300x8-gpu-pool-3okygz   <none>           <none>
server-1   1/1     Running   0          139m   10.154.14.242   rs-b300x8-gpu-pool-3okyg9   <none>           <none>
server-2   1/1     Running   0          137m   10.154.13.252   rs-b300x8-gpu-pool-3okyg0   <none>           <none>
server-3   1/1     Running   0          134m   10.154.14.119   rs-b300x8-gpu-pool-3okyg1   <none>           <none>
```

## 0. 要点

- DP 是把权重在多个 GPU 或实例上各复制一份，每份独立处理不同批次的请求；稠密模型和 MoE 模型都适用。
- **稠密模型的 DP rank 之间互相独立** [实测]：双节点实验中每个 Worker 的通信组大小为 1，没有跨 rank 的集合通信，跨节点只有 ZMQ 小消息。**MoE 模型的 rank 才通过专家层耦合**，需要 dummy batch 同步（官方文档，未实测）。
- Internal 模式下，选 DP rank 的是 APIServer，不是 DPCoordinator [源码]；DPCoordinator 只发布各 rank 的队列长度和 wave 状态。
- 多个 APIServer 共用同一个监听 socket，连接由内核分配，并不均匀 [实测]。
- 多节点 Internal 模式：从节点用 `--headless`，只运行引擎；跨节点的 ZMQ 走 TCP，Pod 网络即可，稠密模型不需要 RDMA [实测]。
- 对稠密小模型，DP 没有特别的实际收益，直接起多个独立实例即可（官方文档也这样建议）；DP 的真正价值在 MoE 的 DP + EP。

## 1. 主旨

DP 指权重在多个 GPU 或实例上各复制一份，每份独立处理不同批次的请求。稠密模型和 MoE 模型都适用。官方页面聚焦**在线部署**（API server），DP+EP 也支持离线使用（`LLM` 类）。

## 2. 核心概念

### 2.1 MoE 模型的常见组合

对 DeepSeek 这类 MLA + MoE 模型，常见做法是 **Attention 层用 DP，专家层用 EP 或 TP**。

- 未开启 EP 时，专家层组成大小为 **DP × TP** 的张量并行组。
- 加上 `--enable-expert-parallel` 后，专家层改用专家并行。

### 2.2 DP rank 是否独立：稠密模型与 MoE 不同

**MoE 模型**（官方文档，本环境未验证）：
- 各 rank 只通过 MoE 层耦合，但专家层每次前向都要同步，所以各 rank 的前向必须对齐。
- 只要有一个 rank 有请求在处理，其余空闲 rank 也要执行空的前向（**dummy batch**），否则集合通信会卡住。
- **DP Coordinator** 进程负责协调，并周期性地做集合通信，判断是否所有 rank 都已空闲，空闲了才能暂停。

**稠密模型** [实测]：
- 双节点实验（TP=1）中，每个 Worker 的日志都是 `rank 0 in world size 1 is assigned as DP rank ...`，通信组大小为 1，rank 之间没有跨 rank、跨节点的集合通信。（TP>1 时，通信组只在同一个 rank 内的 TP 组之间建立，这一点在单节点 TP=2 的实验中没有单独核对日志。）
- 因此 dummy batch 同步对它没有实际意义；DPCoordinator 的主要价值只剩发布负载统计。
- 等价于 N 个独立的 vLLM 实例，外加一个统一入口。

### 2.3 进程架构

- 每个 DP rank 是一个独立的 **core engine** 进程，通过 ZMQ 与前端进程通信 [实测、源码]。
- DP 可与 TP 组合：每个 DP 引擎拥有的 GPU worker 进程数等于 TP 大小，总 GPU 数 = DP × TP [实测]。例如 DP=4、TP=2 需要 8 张卡；DP=4、TP=1 需要 4 张卡。
- 单节点 DP=4、TP=2 实测共 19 个进程：4 个 APIServer、1 个 DPCoordinator、4 个 EngineCore、8 个 Worker，外加主进程和 resource_tracker [实测]。
- Worker 是 EngineCore 的子进程，数量等于 TP 大小 [实测]。

### 2.4 KV Cache 与前缀缓存

每个引擎的 KV Cache 相互独立 [实测：各 rank 单独分配 KV cache]。把前缀相似的请求导向同一引擎，能更好地利用前缀缓存，因此负载均衡策略会影响缓存命中率。例如，同一用户的多轮对话若被分到不同 rank，前缀缓存就用不上。

## 3. 三种在线负载均衡模式

| 模式 | 端点 | 谁来调度 | 适用 | 本环境 |
|---|---|---|---|---|
| Internal（内部） | 单一端点 | API server 进程内，依据各引擎的运行队列和等待队列 | 中小规模，最省事 | 已实测 |
| Hybrid（混合） | 每节点一个端点 | 节点内自己调度，上游负载均衡器再分发到各节点 | 大规模，避免单节点瓶颈 | 未实测 |
| External（外部） | 每个 rank 一个端点 | 外部路由器结合实时遥测分发 | 超大规模，由编排系统统一管理 | 未实测 |

### 3.1 Internal Load Balancing

- 对外只暴露一个 API 端点，DP 调度在 API server 进程内完成。
  - 具体做法 [源码]：每个 APIServer 的 client 根据 DPCoordinator 发布的各 rank 队列长度，选出最空闲的 EngineCore，再通过 ZMQ 直接把请求发给它，请求不经过 Coordinator。
- 多节点：每个节点各自启动 `vllm serve` 并指定本节点负责的 rank。API server 可以只运行在一个节点上，不必与 DP rank 同节点。
  - [实测] 两节点配置：Node 0 运行 API server、DPCoordinator 和 rank 0、1；Node 1 用 `--headless --data-parallel-start-rank 2` 运行 rank 2、3；两个节点都指向 Node 0 的地址和 RPC 端口。
  - [实测] Node 1 没有 APIServer 和 DPCoordinator，所有 HTTP 入口、分词和负载统计都在 Node 0。
  - [实测] `--data-parallel-rpc-port`（本例 13345）稳定运行后没有监听，看起来只用于启动握手；运行期的跨节点通信是 Node 1 的 EngineCore 主动连到 Node 0 的 APIServer / DPCoordinator 的 ZMQ TCP 端口（共 20 条连接）。
  - [实测] 跨节点走 Pod 网络（VPC）上的 TCP；稠密模型的 DP rank 之间没有集合通信，不需要 RDMA。
- 首节点只跑 API server：把首节点的本地 rank 数设为 0，所有引擎放在其他节点（`--headless`）。（官方文档，未实测）
- Ray 后端：`--data-parallel-backend=ray`，单条命令启动所有本地和远程 rank，无需指定地址和 RPC 端口。若单个模型副本需要跨多个节点，需设置环境变量 `VLLM_RAY_DP_PACK_STRATEGY=span`，此时 `--data-parallel-size-local` 会被忽略并自动推断。（官方文档，未实测；[help] 中该参数取值为 `mp` 或 `ray`，默认 `mp`）
- API server 扩展：DP 规模较大时 API server 可能成为瓶颈，可用 `--api-server-count` 横向扩展。对用户透明，仍是单一端口，但扩展仅限于首节点内部。
  - [help] 默认等于 `data_parallel_size`；[实测] 启动日志 `Defaulting api_server_count to data_parallel_size (4)`，双节点（每节点 2 个 rank）下仍是 4 个，即默认值取 DP 总规模，不是本节点的 rank 数。
  - [实测] 多个 APIServer 共用主进程创建的同一个监听 socket（被 5 个进程同时持有同一个 fd），连接由内核分配；201 个新建连接分到 4 个 APIServer 为 86 / 33 / 45 / 37，并不均匀。保持长连接的客户端只会落在同一个 APIServer 上。
- 准入控制（官方文档，本环境未验证）：
  - `--max-num-seqs` **按 DP rank** 生效 [help 中存在]。
  - `--max-num-queued-reqs`、`--max-num-queued-tokens` **按整个服务器**生效；**这两个参数在本环境 0.19.2rc1 的 `--help` 中没有找到**，可能是较新版本才有，使用前需确认版本。
  - 页面建议上限设为约 `data-parallel-size × max-num-seqs` 加上期望的队列深度，否则 rank 可能跑不满。

### 3.2 Hybrid Load Balancing（未实测）

- 每个节点运行自己的 API server，只把请求排入本节点的 DP 引擎，上游负载均衡器（如 Ingress）负责把请求分发到各节点。
- 启用：`--data-parallel-hybrid-lb`，每个节点仍使用全局 DP 大小启动。
- 与内部模式的区别：
  - 必须提供 `--data-parallel-size-local` 和 `--data-parallel-start-rank`，让节点知道自己负责哪些 rank。
  - 与 `--headless` 不兼容，因为每个节点都要暴露 API 端点。
  - `--api-server-count` 应按本节点的本地 rank 数设置。
- 优势：调度决策在节点内完成，减少跨节点流量，避免大 DP 规模下的单节点瓶颈。这正是 Internal 模式在实测中暴露的局限（入口和 DPCoordinator 都只在 Node 0）。
- [help] 参数说明：vLLM 在节点内的本地 DP rank 之间做负载均衡，节点之间由外部负载均衡器分发，需与 `--data-parallel-start-rank` 配合设置。

### 3.3 External Load Balancing（未实测）

- 把每个 DP rank 当作独立的 vLLM 部署，各自拥有端点，由外部路由器分发 HTTP 请求。
- **非 MoE 模型**：各实例本来就完全独立，直接启动多个不带 `--data-parallel-*` 参数的实例即可。外部 DP 参数仅支持 MoE 部署。（这与实测一致：稠密模型的 rank 之间本来就没有耦合）
- 同机部署：使用默认 RPC 端口，每个 rank 指定不同的 HTTP 端口。
- 多节点部署：需指定 rank 0 的地址和端口。
- rank 推断：配合 `--nnodes` 使用时，只有 `--nnodes` 能被 `--data-parallel-size` 整除，才能由 `--node-rank` 推断 rank，否则必须显式设置 `--data-parallel-rank`。
- 设备隔离：同一节点上多个 rank 需用 `CUDA_VISIBLE_DEVICES` 或 `--device-ids` 分配不重叠的设备，vLLM 不会自动划分。**`--device-ids` 在本环境 0.19.2rc1 的 `--help` 中没有找到**，可改用 `CUDA_VISIBLE_DEVICES`。
- DP Coordinator 与 rank 0 引擎同机运行。
- [help] `--data-parallel-external-lb` 的说明：适用于 Kubernetes 中“一个 Pod 一个 rank”的 wide-EP 部署；**显式提供 `--data-parallel-rank` 时会隐式开启该模式**。

## 4. 关键参数

“本环境”一列：**实测**表示实验中用过；**help**表示在 `--help` 中存在但未使用；**未找到**表示 0.19.2rc1 的 `--help` 中没有。

| 参数 | 作用 | 本环境 |
|---|---|---|
| `--data-parallel-size` | DP 总规模 | 实测 |
| `--tensor-parallel-size` | 每个 DP 引擎内的 TP 大小 | 实测 |
| `--data-parallel-size-local` | 本节点承载的 DP rank 数（0 表示本节点只运行 API server） | 实测（=2） |
| `--data-parallel-start-rank` | 本节点起始 rank 编号 | 实测（=2） |
| `--data-parallel-address` | rank 0 所在节点的 IP 地址 | 实测 |
| `--data-parallel-rpc-port` | 内部 RPC 通信端口 | 实测（13345，仅启动握手） |
| `--data-parallel-rank` | 外部 LB 模式下指定本进程的 rank；提供时隐式开启外部 LB | help |
| `--data-parallel-backend=ray` | 使用 Ray 作为 DP 后端（取值 `mp` 或 `ray`，默认 `mp`） | help |
| `--data-parallel-hybrid-lb` | 启用混合负载均衡 | help |
| `--data-parallel-external-lb` | 启用外部负载均衡 | help |
| `--enable-expert-parallel` | 专家层使用 EP 而非默认 TP | help |
| `--headless` | 不启动 API server，仅运行引擎 | 实测 |
| `--api-server-count` | 横向扩展 API server 进程数，默认等于 DP size | 实测（默认值） |
| `--max-num-seqs` | 每个 DP rank 的最大并发序列数 | help |
| `--max-num-queued-reqs` / `--max-num-queued-tokens` | 全服务器范围的排队准入上限 | 未找到 |
| `--nnodes` / `--node-rank` | 多节点外部 LB 中用于推断 rank | help |
| `--device-ids` | 为 rank 分配设备（官方文档） | 未找到 |
| `VLLM_RAY_DP_PACK_STRATEGY=span` | Ray 下单副本跨节点时使用 | 未验证 |

## 5. 部署场景

| # | 场景 | 要点 | 本环境 |
|---|---|---|---|
| 1 | 单节点 DP + TP | DP=4、TP=2，共 8 张 GPU，在 `vllm serve` 中加上对应的 DP 和 TP 参数 | 已实测，见 `vllm-dp4-tp2-single-node.md` |
| 2 | 多节点内部 LB | 节点 0 运行 API server 和 rank 0、1；节点 1 以 `--headless` 运行 rank 2、3；两节点指向同一个 rank 0 地址和 RPC 端口 | 已实测（TP=1，每节点 2 个 rank），见 `vllm-dp4-tp1-two-node.md` |
| 3 | API server 独占首节点 | 首节点本地 rank 数设为 0，全部引擎放在第二节点（`--headless`） | 未实测 |
| 4 | Ray 后端 | `--data-parallel-backend=ray`，在任一节点发起单条命令拉起全部 rank | 未实测 |
| 5 | 外部 LB 同机 | 每个 rank 用 `CUDA_VISIBLE_DEVICES` 绑定不同 GPU，用 `--data-parallel-rank` 区分，分别监听不同端口（如 8000、8001） | 未实测 |
| 6 | 外部 LB 多节点 | rank 0 指定地址和 RPC 端口，其余 rank 指向同一地址 | 未实测 |

## 6. 选型建议

以下是结合官方文档和本环境实测得出的推论，未对各方案做对照测试。

| 模型 | 建议 | 依据 |
|---|---|---|
| 稠密模型 | 直接启动多个独立实例，前面放负载均衡器；不必使用 DP 参数 | 官方文档对非 MoE 的建议；实测稠密模型的 DP rank 之间互相独立，DP 只带来“单一端点 + 按队列长度调度” |
| MoE，规模较小 | Internal | 最省事，单一端点 |
| MoE，规模较大 | Hybrid | Internal 的入口和 DPCoordinator 集中在首节点，大规模下可能成为瓶颈（实测看到了这种集中，瓶颈本身未测） |
| MoE，超大规模 | External，由编排系统管理 | 官方文档 |

关于网络：
- 稠密模型的 DP 只有 ZMQ 小消息跨节点，Pod 网络（VPC）即可，不需要 RDMA [实测]。
- MoE 的跨节点 EP（专家层 all-to-all 每步都要跨节点）、跨节点 TP、PD 分离传 KV cache，才需要 RDMA / RoCE（推论，未实测）。

## 7. 未验证项

- Hybrid、External、Ray 后端、首节点只跑 API server 这几种部署均未实测。
- MoE 模型的 rank 耦合、dummy batch、跨节点专家并行通信未实测；后续可用较小的 MoE 模型加 `--enable-expert-parallel` 验证。
- `--max-num-queued-reqs`、`--max-num-queued-tokens`、`--device-ids` 在 0.19.2rc1 中未找到，其来源版本未确认。
- 准入控制的“按 rank / 按服务器”生效范围，以及页面建议的队列上限，未验证。
- 本地 rank 与远端 rank 的端到端延迟、Node 0 故障行为未测。
