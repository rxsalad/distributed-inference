# vLLM 双节点数据并行：DP=4、TP=1，每节点 2 个 rank（Qwen3-8B）

把一个 DP=4 的 vLLM 部署跨两个节点运行：Node 0 对外提供 API，承载 DP0、DP1；Node 1 以 headless 模式承载 DP2、DP3。外部请求只访问 Node 0 的 8000 端口，vLLM 再把请求分配到两个节点上的 4 个 DP rank。

**要点**

- 配置：DP=4，TP=1（启动脚本未指定），两个节点各承载 2 个 rank，各用 2 张 GPU，共 4 张。
- Node 1 是 headless：没有 APIServer 和 DPCoordinator，只有 2 个 EngineCore 和各自的 Worker；入口、分词、负载统计都在 Node 0。
- 跨节点通信是 Pod 网络（VPC）上的 TCP（ZMQ），共 20 条连接，由 Node 1 的 EngineCore 主动连到 Node 0；不需要 RDMA。
- 稠密模型的 DP rank 之间没有集合通信（Worker 的通信组大小为 1）。
- 实测 501 个请求在 4 个 rank 间分配均匀（127 / 124 / 125 / 125），两个节点的 GPU 利用率都达到 97% 以上；APIServer 之间的连接分配则不均匀（86 / 33 / 45 / 37）。
- 对稠密小模型没有特别的实际收益，本实验的价值是验证机制，为 MoE 的 DP + EP 做预演（第 7 节）。

**参考**

- 负载均衡模式：Internal（单一端点，调度在 Node 0 的 API server 内完成）。
- 官方文档：<https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/#internal-load-balancing>
- 单节点 DP=4、TP=2 的详细分析见 `vllm-dp4-tp2-single-node.md`，本文只讲双节点相关的差异。

## 1. 实验环境

| | Node 0 | Node 1 |
|---|---|---|
| Pod | server-2 | server-1 |
| IP（Pod IP） | 10.154.13.252 | 10.154.14.242 |
| 角色 | API server + DPCoordinator + DP0、DP1 | headless，DP2、DP3 |
| GPU | 8×B300（本实验只用 GPU0、GPU1） | 8×B300（本实验只用 GPU0、GPU1） |

- 镜像 `vllm/vllm-openai:cu130-nightly`，vLLM 0.19.2rc1，两节点版本相同；`MODEL=Qwen/Qwen3-8B`。
- 两个 Pod 之间网络连通（ping 无丢包，延迟约 1.2 ms）。
- 与原始脚本的差异：
  - IP：示例中的 `10.99.48.128` 替换为 Node 0 的 Pod IP `10.154.13.252`。
  - 节点：原计划用 server-0 作为 Node 0，但它上一轮单节点实验遗留了孤儿 Worker 进程、占满 GPU，所以改用空闲的 server-2。
  - TP：脚本没有 `--tensor-parallel-size`，所以 TP=1，每个 DP rank 只用 1 张 GPU。

```
                      Client
                        │ HTTP :8000
                        ▼
  Node 0 (server-2, 10.154.13.252)
  ├─ APIServer_0..3   共用 8000 端口
  ├─ DPCoordinator
  ├─ EngineCore_DP0 ── Worker (GPU0)
  └─ EngineCore_DP1 ── Worker (GPU1)
        ▲
        │ TCP（ZMQ，Node 1 的引擎主动连过来）
        ▼
  Node 1 (server-1, 10.154.14.242)   headless，无 API server
  ├─ EngineCore_DP2 ── Worker (GPU0)
  └─ EngineCore_DP3 ── Worker (GPU1)
```

## 2. 启动

```
export MODEL=Qwen/Qwen3-8B

# Node 0（server-2，IP 10.154.13.252）
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-address 10.154.13.252 --data-parallel-rpc-port 13345

# Node 1（server-1）
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-start-rank 2 \
                  --data-parallel-address 10.154.13.252 --data-parallel-rpc-port 13345
```

| 参数 | 含义 |
|---|---|
| `--data-parallel-size 4` | DP 总规模，两个节点相同 |
| `--data-parallel-size-local 2` | 本节点承载的 rank 数 |
| `--data-parallel-start-rank 2` | Node 1 的起始 rank，所以它承载 DP2、DP3；Node 0 缺省从 0 开始，承载 DP0、DP1 |
| `--data-parallel-address` / `--data-parallel-rpc-port` | Node 0 的地址和 RPC 端口，两个节点都指向它 |
| `--headless` | Node 1 不启动 API server，只运行引擎 |

启动顺序：先启动 Node 0，再启动 Node 1。实验中两个节点都用 `setsid nohup ... &` 放到后台，日志写到各 Pod 的 `/tmp/vllm-node0.log`、`/tmp/vllm-node1.log`。

启动耗时：Node 0 在 22:13:24 启动，22:16:14 API server 开始监听 8000，约 3 分钟，其中包括下载模型（两个 Pod 之前都没有缓存）；引擎初始化 72 秒，其中编译约 31 秒。每个 rank 的 KV cache 为 1,659,776 个 token。

日志中的关键步骤：
- Node 0：启动 DP Coordinator → 启动 4 个 API server 进程 → 引擎就绪后 `All engine subscriptions received by DP coordinator` → API server 开始监听。
- Node 1：`Launching 2 data parallel engine(s) in headless mode, with head node address tcp://10.154.13.252:13345`。

## 3. 进程结构（实测）

**Node 0（server-2）**：

```
vllm serve (PID 1425)
├─ resource_tracker
├─ DPCoordinator
├─ EngineCore_DP0 ── Worker   (GPU0)
├─ EngineCore_DP1 ── Worker   (GPU1)
└─ APIServer_0..3             （共用 8000 端口）
```

**Node 1（server-1）**：

```
vllm serve --headless (PID 1449)
├─ resource_tracker
├─ EngineCore_DP2 ── Worker   (GPU0)
└─ EngineCore_DP3 ── Worker   (GPU1)
```

| 节点 | 进程 | 数量 | GPU |
|---|---|---|---|
| Node 0 | 主进程、resource_tracker | 各 1 | |
| Node 0 | DPCoordinator | 1 | |
| Node 0 | APIServer | 4 | |
| Node 0 | EngineCore_DP0 / DP1 | 2 | |
| Node 0 | Worker（DP0、DP1 各 1 个） | 2 | GPU0、GPU1 |
| Node 1 | 主进程（headless）、resource_tracker | 各 1 | |
| Node 1 | EngineCore_DP2 / DP3 | 2 | |
| Node 1 | Worker（DP2、DP3 各 1 个） | 2 | GPU0、GPU1 |

与单节点（见 `vllm-dp4-tp2-single-node.md`）相比：
- 因为 TP=1，每个 EngineCore 下只有 1 个 Worker（单节点 TP=2 时是 2 个）。
- APIServer 数量仍是 4：默认等于 DP size，而不是本节点的 rank 数（启动日志：`Defaulting api_server_count to data_parallel_size (4)`）。
- Node 1 没有 APIServer，也没有 DPCoordinator，只有引擎；所有 HTTP 入口、分词、负载统计都在 Node 0。
- 每个节点只有 GPU0、GPU1 被占用（约 253 GiB，`--gpu-memory-utilization` 默认 0.9），其余 6 张 GPU 空闲。

## 4. 跨节点通信（实测）

### 4.1 13345 端口只用于启动握手

稳定运行后，在 Node 0 上执行 `ss -ltnp` 看不到 13345 的监听，也没有该端口的已建立连接，只有 8000 在监听。所以 `--data-parallel-rpc-port` 看起来只用于启动阶段的握手，运行期间的通信不再使用它（根据观察到的现象下的结论，没有读源码核对）。

### 4.2 运行期的跨节点连接

在两个节点上各执行 `ss -tnp`，Node 0 和 Node 1 之间共有 **20 条 TCP 连接**（两端数量一致）。连接由 Node 1 的 EngineCore 作为客户端发起，连到 Node 0 的 APIServer 或 DPCoordinator 监听的端口：

| Node 0 侧进程 | 监听的端口数 | 每个端口的连接数（来自 Node 1 的 EngineCore_DP2、DP3） | 连接数 |
|---|---|---|---|
| APIServer × 4 | 每个 2 个 | 2 | 4 × 2 × 2 = 16 |
| DPCoordinator | 2 个 | 2 | 2 × 2 = 4 |
| 合计 | | | 20 |

- 方向的依据：Node 0 一侧只有 10 个不同的端口，每个端口上各有 2 条连接；Node 1 一侧的 20 个端口各不相同。所以 Node 0 的这 10 个端口是被连接的一方（监听端），Node 1 的引擎是客户端。
- 每个 APIServer 有 2 个端口，被 DP2、DP3 的 EngineCore 各连一次。结合单节点文档中 ZMQ 的通道设计（请求下发用 ROUTER/DEALER，结果回传用 PUSH/PULL），这 2 个端口应该分别对应请求下发和结果回传。这是推断，没有逐个端口核对。
- DPCoordinator 的 2 个端口被 DP2、DP3 各连一次，应对应统计上报和统计 / 状态广播，同样是推断。
- 这说明跨节点时 ZMQ 使用 TCP（同机时是 IPC，见单节点文档）。

### 4.3 稠密模型下 rank 之间没有集合通信

Node 1 的日志显示 Worker 是 `rank 0 in world size 1 is assigned as DP rank 0 ...`，`DP group leader: ... world_size=1`：每个 Worker 的通信组大小都是 1，没有跨 rank、跨节点的 NCCL 组。这与单节点文档里“稠密模型的 DP rank 之间互相独立”的结论一致。跨节点的流量只有 ZMQ 的小消息（请求、token 结果、统计）。

### 4.4 网络路径：走 Pod 网络（VPC），不需要 RDMA

本实验的跨节点通信走 Pod 网络（VPC）上的 TCP，没有用到 RDMA。

实测依据：
- 两个 Pod 都只有一个带 IPv4 地址的网卡 `eth0`（server-2 为 `10.154.13.252/32`，server-1 为 `10.154.14.242/32`）。
- 到对端的路由都是 `via 网关 dev eth0`，即走 Pod 网络。
- 4.2 中统计的 20 条跨节点连接，两端都是这两个 10.154.x 地址之间的 TCP 连接（ZMQ 的 TCP 传输）。
- Pod 内没有设置 NCCL、UCX 等与 RDMA 相关的环境变量。

不需要 RDMA 的原因：
- 稠密模型的 DP rank 之间没有集合通信（4.3），跨节点只有 ZMQ 小消息：请求、token 结果、负载统计，量小且对延迟不敏感，普通 TCP 足够。
- 本实验 TP=1，连节点内部的 all-reduce 都没有，更不存在跨节点的 all-reduce。

补充说明：Pod 内能看到 `mlx5_0` 到 `mlx5_19` 共 20 个 RDMA 设备（`n-server.yaml` 里申请了 RoCE fabric 资源），但本实验没有使用。因此本实验只验证了 Pod 网络可用，没有验证 RoCE 网络的状态。

RDMA 在以下场景才会成为关键：
- MoE 的跨节点 EP：专家层的 all-to-all 每步都要跨节点；
- 跨节点的 TP：每层的 all-reduce 都要跨节点；
- PD 分离中传输 KV cache（nixl、mooncake 等连接器）。

## 5. 功能与分发验证

### 5.1 单个请求

向 Node 0 的 8000 端口发送 `Who are you?`（`max_tokens=48`）返回正常，`prompt_tokens=12`、`completion_tokens=48`，请求由 engine 0（DP0）处理。

### 5.2 并发测试：请求在 4 个 rank 间的分配

用 `xargs -P` 并发调用 curl，每个请求新建一个连接。各 rank 处理的请求数取自 Node 0 的 `/metrics` 中 `vllm:request_success_total{engine=...}`，这个指标包含 Node 1 的 DP2、DP3。

| 测试 | 请求数 / 并发 / max_tokens | DP0 | DP1 | DP2（Node 1） | DP3（Node 1） |
|---|---|---|---|---|---|
| 单个请求 | 1 | 1 | 0 | 0 | 0 |
| 第 1 轮 | 200 / 32 / 128 | +51 | +50 | +50 | +49 |
| 第 2 轮 | 300 / 48 / 256 | +75 | +74 | +75 | +76 |
| 累计（共 501） | | 127 | 124 | 125 | 125 |

请求在 4 个 rank 之间分得很均匀，跨节点的 DP2、DP3 与本地的 DP0、DP1 没有差别。

### 5.3 两个节点的 GPU 都在工作

第 2 轮测试时每 2 秒采样一次 GPU 利用率：

| 采样 | Node 0 GPU0 / GPU1 | Node 1 GPU0（DP2） / GPU1（DP3） |
|---|---|---|
| 1 | 71% / 56% | 73% / 78% |
| 2 | 87% / 83% | 97% / 82% |
| 3 | 98% / 90% | 99% / 87% |
| 4 | 98% / 98% | 97% / 99% |

其余 GPU（GPU2 到 GPU7）全程 0%。

远端 rank 的运行中请求数无法直接观察：第 1 轮测试时每秒采样 Node 0 的 `/metrics`，DP0、DP1 的 `vllm:num_requests_running` 一直是 8（32 并发 ÷ 4 个 rank，符合均分），但其中没有 DP2、DP3 的这个指标，而 Node 1 是 headless，自己没有 `/metrics` 端点。因此远端 rank 在工作这一点，依据的是上表的 GPU 利用率和 5.2 的累计请求数。

### 5.4 APIServer 之间的连接分配并不均匀

第 1 轮测试后，统计 Node 0 日志里各 APIServer 处理的 `POST /v1/chat/completions` 次数（累计 201 个请求，每个请求都是新连接）：

| APIServer_0 | APIServer_1 | APIServer_2 | APIServer_3 |
|---|---|---|---|
| 86 | 33 | 45 | 37 |

这印证了单节点文档里的判断：4 个 APIServer 共用同一个监听 socket，由内核决定谁拿到连接，分配并不均匀；本次最多的 APIServer_0 是最少的 APIServer_1 的 2.6 倍。这是两层负载均衡的第一层，第二层（请求到 DP rank）是均匀的，见 5.2。

## 6. 与单节点（DP=4、TP=2）的对比

| | 单节点 | 双节点（本文） |
|---|---|---|
| 节点数 | 1 | 2 |
| TP | 2 | 1（脚本未指定） |
| GPU 使用 | 一个节点 8 张 | 每个节点 2 张，共 4 张 |
| API server / DPCoordinator | 同一节点 | 都在 Node 0；Node 1 是 headless |
| ZMQ 传输 | IPC（Unix 域 socket） | Node 0 到 Node 1 为 TCP（实测）；Node 0 本地 rank 的传输方式未核对 |
| 稳定运行时的 13345 端口 | 不涉及 | 没有监听，只用于启动握手 |
| rank 之间通信 | 无（稠密模型） | 无（稠密模型），只有 ZMQ 小消息跨节点 |
| 网络 | 节点内（IPC、NVLink） | 节点间走 Pod 网络（VPC）的 TCP，不需要 RDMA |
| 请求在 rank 间分配 | 均匀 | 均匀（含跨节点的 rank） |

## 7. 这种架构的意义与局限

结论：对 Qwen3-8B 这样的稠密小模型，这个架构没有特别的实际收益，本实验的主要价值是验证多节点 DP 的机制；它真正有用的场景是 MoE 大模型的 DP + EP。以下比较来自架构分析和本实验的观察，没有对“独立实例 + 外部 LB”方案做对照测试。

### 7.1 对稠密模型：没有特别的收益

稠密模型的 DP rank 之间互相独立（实测每个 Worker 的通信组大小为 1），4 个 rank 等价于 4 个独立的 vLLM 实例。与“每个节点各起独立实例，前面放一个负载均衡器”相比：

| | 双节点 DP（本实验） | 独立实例 + 外部 LB |
|---|---|---|
| 入口 | Node 0 单点，故障则整体不可用（架构推论，未做故障实验） | 各实例相互独立，一个挂了不影响其他 |
| 扩缩容 | 要同时调整所有节点的 DP 参数和 rank 编号 | 增减实例即可 |
| 负载均衡依据 | 各 rank 的队列长度，信息更准 | 取决于 LB，通常只能看连接数或轮询 |
| 额外组件 | DPCoordinator、跨节点 TCP | 无 |
| 效果 | 实测 4 个 rank 均衡 | 同样可以做到 |

唯一的实际好处是“单一端点 + 按队列长度调度”。vLLM 官方文档也说明，非 MoE 模型直接启动多个独立实例即可，外部 DP 参数只用于 MoE 部署（见 `data-parallel.md`）。

### 7.2 真正的意义：MoE 的 DP + EP

这个架构为 MoE 模型（如 DeepSeek）设计：

- Attention 用 DP，专家层用 EP 或 TP。MLA 的 KV cache 很小，适合 DP；专家参数太大，要分散到多张卡上。
- rank 之间通过专家层耦合：每次前向所有 rank 都要一起做专家层的集合通信，所以需要跨节点的统一调度、dummy batch（有一个 rank 在跑，其余空闲 rank 也要做空的前向），以及 DPCoordinator 协调“全部空闲”的状态。这时必须有一个统一的 DP 组，不能拆成独立实例。

本实验用到的 `--data-parallel-size-local`、`--data-parallel-start-rank`、`--headless`、`--data-parallel-address` 就是搭建这种 DP 组的基础参数；稠密模型下能跑通，但 MoE 才真正需要这些机制。

### 7.3 本实验的价值与局限

价值：
1. 验证了机制：多节点 DP 的组网、握手、跨节点 ZMQ 连接、请求分配都按预期工作，是做 MoE 之前的预演（可先确认网络和镜像没有问题）。
2. 看清了架构形态：入口集中在 Node 0、Node 1 是 headless、跨节点只有 ZMQ 小消息；这些在 MoE 场景下同样成立，差别是多了专家层的 NCCL 通信。

局限：
- 入口单点：APIServer 和 DPCoordinator 都只在 Node 0；DP 规模很大时 Node 0 可能成为瓶颈，`data-parallel.md` 里的 Hybrid / External 模式就是为解决这个问题而设计。
- 资源利用率低：TP=1、每节点 2 个 rank，每个节点只用了 8 张 GPU 中的 2 张。若要用满，可把 `--data-parallel-size-local` 提高到 8（`--data-parallel-size 16`，Node 1 的起始 rank 为 8）；该配置没有实测。
- 请求路径多了跨节点的一跳：被分到 DP2、DP3 的请求要经 TCP 往返于 Node 0 和 Node 1，网络往返约 1.2 ms，远小于生成耗时；但没有对比本地 rank 与远端 rank 的端到端延迟，不能下结论。

### 7.4 后续可做的实验（建议，未执行）

- 换成较小的 MoE 模型（例如 Qwen3-30B-A3B，未核对其在本环境能否运行），加上 `--enable-expert-parallel`，观察 rank 耦合、dummy batch 和跨节点的集合通信；集群的 Pod 配置了 RoCE 网络（见 `n-server.yaml`），可借此测试跨节点专家并行的通信。
- 测试 Hybrid 和 External 负载均衡模式，解决 Node 0 单点问题。

## 8. 未验证项

- 本地 rank 与远端 rank 的端到端延迟对比（未测）。
- 4.2 中各 ZMQ 端口的具体用途（推断，未逐个核对）；4.1 中 13345 只用于启动握手（根据 `ss` 的现象，未读源码）。
- 单个 APIServer 为何拿到更多连接（内核唤醒顺序 / uvicorn 的 epoll 行为），未深入分析。
- 满配置（每节点 8 个 rank）和 Node 0 故障行为未测试。
- 没有读取 RDMA 网卡的流量计数器来证明“零 RDMA 流量”；4.4 的结论依据路由、连接和环境变量。
- 远端 rank（DP2、DP3）的运行中请求数没有直接指标。

## 9. 运维备注

以下为实验结束时的状态：

- 双节点服务仍在运行：Node 0 在 server-2，Node 1 在 server-1，日志在各 Pod 的 `/tmp/vllm-node0.log`、`/tmp/vllm-node1.log`。
- server-0 上仍有 8 个孤儿 Worker 进程（PID 3498、3499、3515 到 3520）和 1 个 resource_tracker（PID 1742）占着 8 张 GPU，需要手动清理。
- 停止服务时要注意：只终止 `vllm serve` 主进程时，Worker 子进程可能变成孤儿进程继续占用 GPU（server-0 就是这样留下的）。停止后用 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 确认 GPU 已释放，必要时再清理残留的 `VLLM::Worker` 进程。
