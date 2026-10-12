# 策略三：Multi-Node Data + Expert Parallel（DP16 + EP16，hybrid-lb）实测

DeepSeek-V3（FP8）在 2 个 B300 节点（每节点 8 卡，共 16 张 GPU）上以 **DP Attention + EP** 方式运行，对应 recipes 页面的 Multi-Node Data + Expert Parallel（DEP）方案。本次与策略一（TP16）**同时**部署在同一集群的另外两个节点上，便于对照。

- 实验日期：2026-10-11；镜像 `vllm/vllm-openai:latest`（实际是 vLLM 0.31.0），NCCL 2.30.7；节点 `rs-b300x8-gpu-pool-3o1v8b`、`3o1v8e`（策略一占用 `3o1v8g`、`3o1v8w`）。
- 相关文件：`dep-server.yaml`（部署文件）、`recipe.md` 第 6 节（页面内容整理）、`strategy1.md`（TP16，用于对比）、`vllm-dp16-ep-two-node.md`（更早的 DP16 实验）。
- 标注约定：**实测**＝本集群的命令输出或日志；**源码核对**＝读了镜像内 vLLM 0.31.0 源码；**推断**＝由日志、机制或一般知识得出，未直接验证。
- **范围说明**：按要求不做性能测试，本文没有吞吐 / 延迟数据；测试只验证功能、请求分布、GPU 利用率和网络计数器。

## 1. 摘要

| 方面 | 结论 |
|---|---|
| 能否部署 | 能，页面命令（加上本集群需要的环境变量）可直接使用；**不需要** `--disable-custom-all-reduce`（TP=1，没有 TP 组的 custom all-reduce，第 4 节） |
| 启动耗时 | 从 vLLM 启动到服务就绪约 22 分钟（22:46:58 → 约 23:08:50）；`init engine` 1138 秒，比 TP16 的 912 秒长（第 4.1 节） |
| 进程 | 每节点：1 个主进程 + 8 个 API server 进程 + 8 个 EngineCore + 8 个 Worker；DPCoordinator 只在节点 0（第 5.1 节） |
| 对外入口 | **两个节点都对外服务**，各 8 个 API server 进程；用 Service 在两个 Pod 间分流（第 3 节） |
| Attention | 每个 DP rank 在自己的 GPU 上独立计算，**无通信**；每卡一份完整 Attention 权重（第 2 节） |
| KV cache | **每个 rank 存自己请求的 KV，互不重复**：每卡 128.39 GiB、1,961,792 tokens，16 个 rank 合计 31,388,672 tokens（TP16 全实例只有 2,959,680）（第 6.2 节） |
| 每卡显存 | 权重 58.02 GiB（TP16：40.47）；**峰值激活预留 58.9 GiB**（TP16：9.46），所以 KV 比 TP16 每卡少（第 6.3 节） |
| MoE 通信 | 每个 MoE 层：DP 组 `all_gatherv`（dispatch）→ 本卡专家计算 → `reduce_scatterv`（combine）；all2all 后端是默认的 `allgather_reducescatter`，**确实被用到**（DP>1）（第 7 节） |
| 控制面 | DP 步同步在异步调度下走 **CPU 上的 Gloo all-reduce**（源码），在每个步的路径上；这与 TP16 不同（第 5.2 节） |
| 请求分布 | 单个请求只由一个 DP rank 处理，但**两个节点的 16 张卡利用率都升到 90%+**；节点内 64 个请求在 8 个 rank 间分布为 7–10 个（第 8 节） |
| 数据面 | 跨节点 GPU 数据走 NCCL over RoCE（实测 RDMA write 计数器）；eth0 上只有几 MB（第 8 节） |

## 2. 方案总览

`--data-parallel-size 16 --data-parallel-size-local 8 --enable-expert-parallel`（TP 取默认值 1）：16 个 DP rank，每个 rank 占一张 GPU，EP = TP × DP = 1 × 16 = 16。

| 部分 | 并行方式 | 每张卡上的内容 |
|---|---|---|
| Attention（MLA） | DP：每个 rank 独立处理自己的请求，rank 之间**不通信** | 完整的 Attention 权重（TP=1，不切分） |
| KV cache | 每个 rank 只存分配给自己的请求 | 自己请求的 KV，互不重复 |
| 前 3 层稠密 FFN | 每个 rank 独立计算 | 完整副本（TP=1，推断） |
| 路由门控 | 每个 rank 对自己的 token 先算一遍 | 门控权重 |
| 256 个路由专家（后 58 层） | EP16 | 16 个完整专家，linear 放置（实测日志） |
| 共享专家 | 每个 rank 独立计算 | 完整副本（TP=1，推断） |

每个 MoE 层的通信：DP 组 `all_gatherv` + `reduce_scatterv`（NCCL）。Attention 和稠密层没有通信。

与 TP16 的区别：TP16 里 Attention 由 16 张卡共同计算、每层 all-reduce，KV 每卡一份完整拷贝；这里 Attention 各算各的、没有通信，KV 不重复，代价是权重在每张卡上多存一份完整的 Attention（第 6 节）。

## 3. 部署

**文件**：`dep-server.yaml`
- StatefulSet `dep`，2 副本，`podManagementPolicy: Parallel`，反亲和保证两个 Pod 落在不同节点；`tp16` 占用的两个节点 GPU 已满，调度器自然落到另外两个节点。
- headless Service `dep-headless`：稳定 DNS（`dep-0.dep-headless`），`publishNotReadyAddresses: true`。
- Service `dep-api`：8000 端口选中**两个** Pod（hybrid-lb 下两个节点都对外服务），在 Pod 之间做连接级分流。这与策略一不同（策略一只有 `tp16-0` 提供 API）。
- 每个 Pod 占 8 张 GPU 和 8 个 RoCE 网卡，其余沿用 `tp16-server.yaml`。

**角色怎么分**：静态约定，脚本取主机名最后一段作序号：`dep-0` → DP rank 0–7（也是 DP 协调者所在节点），`dep-1` → `--data-parallel-start-rank 8`（DP rank 8–15）。`--data-parallel-address` 是 `dep-0` 的 Pod IP（脚本解析 `dep-0.dep-headless` 得到）。序号不为 0 的节点会先等 `dep-0` 的 RPC 端口（13345）可连，再启动 vLLM。

**启动命令**（Pod 内实际执行，节点 0；节点 1 另加 `--data-parallel-start-rank 8`）：
```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --trust-remote-code --enable-expert-parallel \
  --data-parallel-hybrid-lb \
  --data-parallel-size 16 --data-parallel-size-local 8 \
  --data-parallel-address 10.154.2.110 --data-parallel-rpc-port 13345 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v3 --reasoning-parser deepseek_v3
```
**没有 `--tensor-parallel-size`**：缺省值是 1，即不做张量并行，每个 DP rank 独占一张 GPU（日志里 Worker 名为 `Worker_DPn_EPn`，没有 TP 后缀）。总 GPU 数 = DP × TP = 16 × 1 = 16；`--enable-expert-parallel` 下 EP 大小 = DP × TP = 16，每卡 256 / 16 = 16 个路由专家。若显式设 `--tensor-parallel-size 2`，需配 `--data-parallel-size 8` 才用同样的 16 张卡，即页面描述里的“TEP”形态（本文没有测）。

环境变量：`GLOO_SOCKET_IFNAME=eth0`、`NCCL_SOCKET_IFNAME=eth0`、`NCCL_IB_GID_INDEX=3`、`VLLM_ENGINE_READY_TIMEOUT_S=3600`。

**与页面命令的差异**：
- 加了 `NCCL_IB_GID_INDEX=3` 和 `VLLM_ENGINE_READY_TIMEOUT_S`（来自 `vllm-dp16-ep-two-node.md` 的实测经验）；显式指定了 `--data-parallel-rpc-port 13345`。
- 去掉了 `--chat-template`（镜像里的路径未验证），与策略一一致；工具调用没有测。
- 页面没有 `--disable-custom-all-reduce`，本次也**不需要**：TP=1，没有 TP 通信组需要 custom all-reduce；没有加这个参数也顺利启动。

**实际落点**：

| Pod | 节点 | Pod IP | 角色 |
|---|---|---|---|
| dep-0 | rs-b300x8-gpu-pool-3o1v8b | 10.154.2.110 | DP rank 0–7，DPCoordinator，API server |
| dep-1 | rs-b300x8-gpu-pool-3o1v8e | 10.154.2.184 | DP rank 8–15，API server |

（部署过两次，第二次 Pod 重建后 IP 变了；表中是当前值。）

## 4. 启动过程与故障

### 4.1 启动时间线（第二次部署）

| 时间 | 事件 |
|---|---|
| 22:46:58 | vLLM 启动；日志 `Defaulting api_server_count to data_parallel_size_local (8) for hybrid LB mode.` |
| 22:48:03 | EngineCore 启动（每个 DP rank 一个，`world_size=1`） |
| 22:48:36–22:48:42 | 建立 `dp:0`、`ep:0` 通信组；`Using AgRsAll2AllManager all2all manager.` |
| 22:48:53–22:48:58 | 选择注意力后端；`[EP Rank 0/16] ... Local/global number of experts: 16/256` |
| 22:49:31–22:49:48 | 加载权重 28.2 秒；"Model loading took 58.02 GiB and 64.6 seconds" |
| 22:50:34 | torch.compile 共 45.3 秒 |
| 22:51:00 | 初始 profiling / warmup 25.9 秒 |
| 22:59:25 | 确定 KV cache：每卡 128.39 GiB，每 rank 1,961,792 tokens |
| 23:08:46 | `init engine` 完成，共 1137.96 秒（其中编译 45.3 秒） |
| 约 23:08:50 | 两个节点的 API server 就绪 |

- profiling 到确定 KV cache 之间用了约 8.4 分钟，TP16 同一阶段约 5 分钟；原因没有深挖（可能与每步要处理 16 个 rank 的 token 有关，推断）。
- 该节点的 HF 缓存里已有权重，所以没有再下载。

### 4.2 第一次失败：dep-1 的 `HEAD_IP` 为空（DNS 竞态）

**现象**：第一次部署后 `dep-0` 正常启动并等待，`dep-1` 一直循环打印 `waiting for DP master :13345`，日志里命令是 `</dev/tcp//13345`（地址为空）。

**原因**：脚本先用 `until getent ahostsv4 <主节点域名>` 等待解析成功，再单独执行一次 `getent ... | awk` 取 IP。`dep-1` 上第一次调用返回成功，紧接着取 IP 却是空，是 DNS 的瞬时竞态；事后再查已经能解析到 `10.154.2.65`。

**处理**：改成循环直到取到**非空** IP，然后删除并重新部署。`tp16-server.yaml` 里有同样的潜在问题（策略一部署时没有触发），也一并修改了文件，但正在运行的 TP16 实例没有重启。

**没有出现策略一的卡死**：策略一在 `latest` 镜像上因为 TP16 组的 custom all-reduce 初始化而卡死（见 `strategy1.md` 第 4 节）；本策略 TP=1，没有这个问题。

## 5. 运行时结构

### 5.1 进程（实测）

```
节点 0 (dep-0, 10.154.2.110)                       节点 1 (dep-1, 10.154.2.184)
──────────────────────────                         ──────────────────────────
PID 1   vllm serve（MainProcess）                  PID 1   vllm serve（MainProcess）
 ├ DPCoordinator          （只有节点 0 有）         ├ APIServer_0..7      （8 个，进程名带序号）
 ├ APIServer_0..7         （8 个）                  ├ EngineCore_DP8..15  （8 个）
 ├ EngineCore_DP0..7      （8 个）                  │   └ Worker_DP8_EP8 .. DP15_EP15（每个 EngineCore 一个）
 │   └ Worker_DP0_EP0 .. DP7_EP7（每个 EngineCore 一个）
 + resource_tracker                                 + resource_tracker
```

- **8 个 API server 进程**：日志 `Defaulting api_server_count to data_parallel_size_local (8) for hybrid LB mode.`。它们共享 8000 端口的同一个监听 socket（`ss` 显示一个 LISTEN 条目下挂着多个 pid，fd 相同）。这与策略一不同：策略一的 API server 就是主进程本身，进程列表里找不到单独的 APIServer。
- **EngineCore**：**每个 DP rank 一个**（节点上共 8 个），各自有独立的调度器和 KV cache；每个 EngineCore 下面只有 1 个 Worker（`world_size=1`），因为 TP=1。TP16 里则是整个实例只有 1 个 EngineCore 管 16 个 Worker。
- **Worker**：进程名 `Worker_DPn_EPn`，DP rank 与 EP rank 相同；DP rank n 对应本节点 GPU `n % 8`（日志 `rank=n local_rank=n%8`）。
- **DPCoordinator**：只在节点 0 上有一个，监听 3 个随机端口；它负责协调所有 DP rank 的运行状态（推断，没有读它的实现）。
- 两个节点都有自己的 API server，但 EngineCore 与 Worker 数量相同，节点 1 并非 headless。

### 5.2 控制面

- **DP 初始化**：`--data-parallel-address 10.154.2.110 --data-parallel-rpc-port 13345`。13345 端口在启动阶段可连（`dep-1` 在等它），稳态下 `ss` 里已经没有监听者。`torch.distributed` 的会合地址是 `tcp://10.154.2.110:38989`（`world_size=16`，随机端口）。
- 两个 Pod 之间稳态有约 398 条已建立的 TCP 连接（具体用途没有逐条确认）。
- **DP 步同步走 CPU 上的 Gloo（源码核对）**：`config/vllm.py` 里，当异步调度开启、DP>1 且模型是 MoE 时，`disable_nccl_for_dp_synchronization` 被置为真，日志里有 `Disabling NCCL for DP synchronization when using async scheduling.`；`v1/worker/dp_utils.py` 的 `_get_device_and_group` 在该开关为真时改用 `get_dp_group().cpu_group` 在 CPU 上做 all-reduce，用来在 16 个 rank 之间对齐每步的 padding token 数和 cudagraph 模式（注释写的是 "Using CPU all reduce to synchronize DP padding between ranks"）。**日志里没有看到这行 "Using CPU all reduce"，所以没有直接确认该路径在运行时被执行**。
  - 如果成立，这意味着每个调度步都有一次 16 个 rank 的 CPU all-reduce（数据只有 4 × 16 个 int32），经 eth0 的 TCP 走；TP16 里 Gloo 不在步的路径上。对延迟的影响没有测。
- **所有 DP rank 步调一致**：DP + EP 的 MoE 层需要全部 16 个 rank 参与集合通信，所以没有请求的 rank 也要陪跑（第 8 节实测，机制见第 5.6 节）。

### 5.3 数据面

- 日志（实测）：`dp:0` 与 `ep:0` 通信组的 all-reduce 后端都是 `['PYNCCL']`；`Using AgRsAll2AllManager all2all manager.`。
- `--all2all-backend` 没有显式指定，取默认的 `allgather_reducescatter`。DP>1 时 `use_all2all` 为真，所以**这次确实被用到**（策略一里没有用到，见 `strategy1.md` 5.3）。
- MoE 的准备/收尾组件：`MoEPrepareAndFinalizeNaiveDPEPMonolithic`（策略一是 `NoDPEPMonolithic`）。
- 通信走的组（源码核对，`all2all.py` 的 `_get_comm_group`）：非 sequence-parallel、DP world size > 1、没有 prefill context parallel，所以用 **DP 组**（16 个 rank）。
- 跨节点流量（实测，第 8 节）：RDMA write 请求计数大幅增加，eth0 只有几 MB，即 GPU 数据走 RoCE。

### 5.4 对外入口：hybrid-lb 与内部 LB 的区别

与 `vllm-dp16-ep-two-node.md`（内部 LB）相比，并行结构相同（DP16 + EP16、TP=1、默认 `allgather_reducescatter`），主要区别在对外入口。

| | 内部 LB（旧文档） | hybrid-lb（本文） |
|---|---|---|
| 对外节点 | 只有 Node 1 | 两个节点都对外 |
| 节点 2 | `--headless`，无 API server | 非 headless，自带 8 个 APIServer |
| API server 数 | 显式 `--api-server-count=8`，只在节点 1 | 不写，默认等于本节点 DP 数（8），共 16 个 |
| 请求分配 | Node 1 的 API server 在全部 16 个 rank 间分配 | 外部 Service 先选节点，节点内再在 8 个 rank 间分配 |
| 客户端入口 | 固定访问 Node 1:8000 | 访问 `dep-api`（Service） |
| DPCoordinator | 节点 1 | 仍只在节点 0 |

实测（本文）：经 `dep-api` 发 24 个请求，两个节点各收到 12 个；64 个并发请求下节点内 8 个 rank 的分布为 7–10。两个节点之间的负载比没有单独测。

影响：
- **入口：** 内部 LB 下 Node 1 是唯一入口。hybrid-lb 下两个节点都能接请求，但 16 个 rank 在同一个 EP 组内，任一节点故障整个 EP 组都无法继续工作，所以并没有获得真正的高可用（推断）。
- **均衡粒度：** hybrid-lb 的跨节点均衡由外部 Service 完成，只到节点级，不感知节点内 rank 的负载。
- **不可直接比较：** 两份文档的镜像（vLLM 0.19.2rc1 与 0.31.0）、模型（V3-0324 与 V3）、LB 模式都不同，吞吐不能直接对比。同环境下对比两种 LB 没有做过。

### 5.5 APIServer 与 EngineCore 的对应关系（实测 + 源码核对）

8 个 APIServer 与 EngineCore 不是一一对应，而是**每个 APIServer 都管理一组 EngineCore，组内全连接**；管理哪一组，由 LB 模式决定。

**实测（dep-0，用 `ss -xnp` / `ss -tnp` 按进程统计）**：
- 8 个 APIServer 中的每一个，与本节点 8 个 EngineCore（DP0–DP7）的每一个都有 **2 条 unix socket**（请求下发、结果回传各一），共 16 条，是 8 × 8 全连接。
- 每个 APIServer 只有 **1 条 TCP 连接**，连到 DPCoordinator（dep-0 的 `:44361`）。dep-1 的 APIServer 也连到 dep-0 的这个端口（跨节点）。
- APIServer 与 EngineCore 之间**没有 TCP 连接**，本机通信走 unix socket（ZMQ `ipc://`）。这更正了旧文档里“APIServer ↔ EngineCore 走 TCP”的说法（旧文档对本机部分标的是“没有核对”）。
- 本节点 8 个 APIServer 看不到 dep-1 的 EngineCore（DP8–15）：没有任何 APIServer 到远端 EngineCore 的连接。

**源码核对（`vllm/v1/engine/core_client.py`，vLLM 0.31.0）**：
- 客户端类型：外部 LB（`--data-parallel-external-lb`）用 `DPAsyncMPClient`；**内部 LB 和 hybrid-lb 都用 `DPLBAsyncMPClient`**，由客户端在它管理的 EngineCore 之间做负载均衡。
- 管理范围：`num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size`，源码注释写明 “Client manages local+remote EngineCores in pure internal LB case. Client manages local EngineCores in hybrid and external LB case.”。即：

| | 每个 APIServer 管理的 EngineCore | 数量 |
|---|---|---|
| 内部 LB（`vllm-dp16-ep-two-node.md`） | 全部 16 个（本机 8 + 远端 8） | 16 |
| hybrid-lb（本文） | 仅本节点 8 个（dep-1 管理的是 DP8–15，按 `engine_ranks_managed` 从 `start-rank` 起算） | 8 |

- 选 rank（`get_core_engine_for_request`）：
  1. 请求里显式带了 `data_parallel_rank` 就直接用它。
  2. 否则对管理范围内每个 EngineCore 算分：`score = max(client_count × 本 APIServer 的 in-flight 数, waiting + running)`；若有 waiting，再加 `waiting × 6 × max(0, KV 使用率 − 0.5)`。取**分最低**的。
  3. 并列时轮转起点，避免固定偏向；选定后本地先把该引擎的 waiting 计数加一，在下一次统计到达之前也能把突发请求摊开。
- 统计来源：DPCoordinator 发布**全部 16 个引擎**的 waiting / running 计数（最短间隔 100 ms），APIServer 按自己管理的 rank 范围切片使用（`counts[ranks[0]:ranks[-1]+1]`）。所以 hybrid-lb 下 dep-0 的 APIServer 只看 DP0–7 的负载，看不到 DP8–15。
- DPCoordinator 对前端的发布地址：`local_only = not local_engines_only`。内部 LB 下前端与协调者同机，用本机地址；hybrid-lb 下 `local_engines_only` 为真，用 TCP 地址，这与实测一致（dep-1 的 APIServer 跨节点连到 dep-0 的协调者）。

**含义**：
- hybrid-lb 下“请求落到哪个节点”完全由外部（`dep-api` Service）决定，节点内才是 vLLM 按负载选 rank；节点之间不会因负载不均而重新分配。
- 结果回传给哪个 APIServer：按源码结构推断是发起请求的那个（EngineCore 对每个客户端有独立的输出通道），没有逐行核对。
- 每个 EngineCore 在 TCP 上还有来自本机和 dep-1 各 8 条左右的连接，指向它自己的一个监听端口，具体用途没有核对（应与 EngineCore 之间的 DP 组同步有关，推断）。

### 5.6 没有请求的 rank 为什么要陪跑：DP wave 与 dummy batch（源码核对）

**需要。** 16 个 rank 在同一个 EP 组里，每个 MoE 层都有 16 个 rank 参与的 all-gather / reduce-scatter（第 7 节），任何一个 rank 缺席，其他 rank 都会卡在集合通信上。所以只要有一个 rank 在处理请求，其余 15 个即使没有请求，也要和它每一步都同步地跑一次前向，这个空前向就是 dummy batch。

**机制（`vllm/v1/engine/core.py`，`run_busy_loop`）**：
1. **全空闲时**：`engines_running = False`，引擎循环阻塞在输入队列上，不跑前向，GPU 利用率 0%（实测，第 8 节）。
2. **来了第一个请求**：被选中的 EngineCore 发 `FIRST_REQ` 给 DPCoordinator，协调者向**全部 16 个**引擎广播 `START_DP_WAVE`（跨节点，走 TCP），各引擎把 `engines_running` 置为 True，进入同一个 wave。
3. **每一步**：每个引擎先 `_process_engine_step()`；如果本步没有要执行的请求（`not executed`）而 wave 仍在进行（`engines_running` 为真），就 `execute_dummy_batch()`，跑一次空前向，保证 MoE 层的集合通信凑齐 16 个 rank。
4. **何时结束**：每隔 `dp_sync_interval` 步（`step_counter`）做一次 16 个 rank 的 all-reduce，判断是否还有任何 rank 有未完成请求（`_has_global_unfinished_reqs`）。全部完成后 `engines_running = False`，wave 计数加一，所有引擎同时暂停，GPU 回到 0%。
5. **暂停时的协议**：两阶段（`pending_pause`）也靠陪跑实现：先在步进中等所有 rank 都置位，再一起停。

**和实测的对应**：单个 400 token 的请求只由 engine 4 处理，但两个节点 16 张卡都升到 90%+，请求结束后一起回到 0%（第 8 节）。与上面的机制一致：15 个 rank 在陪跑。dummy batch 的具体开销（空前向里的 token 数、耗时）没有测，也没有在日志里直接看到 dummy 的记录。

**hybrid-lb 的特点**：协调者只有一个（在 dep-0），dep-1 的 EngineCore 也连到它，所以 wave 是**跨两个节点**的。一个请求落到 dep-0，dep-1 的 8 个 rank 同样陪跑，反之亦然。外部 Service 把请求分流到哪个节点，并不能让另一个节点“休息”。

## 6. 模型切分与显存

### 6.1 每卡权重

- 日志：`Model loading took 58.02 GiB`（TP16 是 40.47 GiB）。
- 估算：专家 16 个 × 44.04M 参数 × 58 层 ≈ 38.1 GiB；非专家部分（Attention 11.4 GB、前 3 层稠密 FFN 1.19 GB、共享专家 2.55 GB、embedding 与 lm_head 约 3.7 GB、门控等）合计约 19 GB ≈ 17.7 GiB，DP 下每卡一份完整副本；两者相加约 55.8 GiB，加上其他（MLA 相关等）与 58.02 GiB 在同一量级（拆分是估算，不是实测）。
- 所以 DP+EP 每卡权重比 TP16 多约 17.5 GiB，这是"Attention 不切分"的代价。

### 6.2 KV cache：每个 rank 存自己的请求，互不重复

- 日志：每卡 `Available KV cache memory: 128.39 GiB`，每 rank `GPU KV cache size: 1,961,792 tokens`，对 163,840 token 的单请求最大并发 11.97 倍。
- 每 token KV 为 70,272 字节（见 `strategy1.md` 6.2）：128.39 GiB ÷ 70,272 B ≈ 1.962M tokens，与日志一致。
- **16 个 rank 各存各的请求**，所以全实例总容量是 16 × 1,961,792 = **31,388,672 tokens**，约为 TP16（2,959,680，每卡一份重复拷贝）的 10.6 倍。
- 代价：一个请求的 KV 只能放在它所在的那个 rank 上，单请求最长受一个 rank 的容量限制（1.96M tokens，仍远大于 163,840 的上限）；各 rank 之间的容量不能互相借用。

### 6.3 为什么每卡 KV 比 TP16 少：峰值激活预留大

日志（`gpu_worker.py:955`，两个策略对照）：

| 项 | TP16（策略一） | DEP（本策略） |
|---|---|---|
| 期望使用（0.92 × 267.69 GiB） | 246.27 GiB | 246.27 GiB |
| 权重 + 非 torch 内存 | 43.11 GiB | 58.98 GiB |
| **峰值激活** | **9.46 GiB** | **58.9 GiB** |
| CUDA graph | 3.21 GiB | 6.04 GiB |
| 当前 KV cache | 193.7 GiB | 128.39 GiB |

- DEP 的峰值激活约是 TP16 的 6.2 倍。原因（推断）：MoE 层的 dispatch 把全部 16 个 DP rank 的 token gather 到每张卡上，profiling 按最大批量（16384 token × 16 个 rank）估算激活，需要的临时缓冲远大于 TP16。
- 稳态 `nvidia-smi` 显存约 204,400 MiB ≈ 199.6 GiB / 卡（只有 TP16 的约 246 GiB 的 81%）：为峰值激活预留的 58.9 GiB 是预算出来的，并没有常驻占用（推断：权重 58.0 + KV 128.4 + CUDA graph 6.0 + 其他约 7 ≈ 199.6）。日志还提示可用 `--kv-cache-memory` 把 KV 调到约 141 GiB 来用满显存（没有尝试）。

### 6.4 EP：专家放置

与策略一相同：256 个路由专家按序号均分，**每卡 16 个完整专家**，`linear` 放置：`[EP Rank 0/16] ... 0->0, 1->1, ...`、`[EP Rank 8/16] ... 0->128, 1->129, ...`（只核对了 rank 0 和 rank 8，其余按规律推）。EP rank 等于 DP rank。

## 7. 一层 MoE 的计算流程（源码核对）

依据：`fused_moe/prepare_finalize/naive_dp_ep.py`、`distributed/device_communicators/all2all.py`（`AgRsAll2AllManager`）、`fused_moe/runner/moe_runner.py`。

```
每个 DP rank 手里只有自己的 N_r 个 token（Attention 在本地独立完成，无通信）
          │
          ▼  MoE 层
   ① 本地算门控：router_logits_r = gate(自己的 hidden_states)
   ② dispatch：对 DP 组（16 个 rank）all_gatherv(hidden_states, router_logits)   ← NCCL，通信 1
        → 每个 rank 拿到全部 16 个 rank 的 ΣN 个 token
   ③ 每个 rank 的 FlashInfer TRTLLM kernel 对这 ΣN 个 token 做 top-k 路由，
        只执行"全局专家号 ∈ [16·r, 16·r+16)"的本地专家              ← 无通信
   ④ combine：reduce_scatterv 到 DP 组：把 16 份部分结果求和，
        每个 rank 只拿回自己那 N_r 个 token 的结果                  ← NCCL，通信 2
   ⑤ 共享专家在本地算，并入结果（细节没读）
          │
          ▼
   继续下一层（Attention 仍然无通信）
```

- **② dispatch**：`NaiveDPEPMonolithic.prepare()` 调用 `get_ep_group().dispatch_router_logits(a1q, router_logits, ...)`，实际由 `AgRsAll2AllManager.dispatch_router_logits` 对 DP 组做 `all_gatherv`。docstring 写明 "the MoE kernel runs the router internally"：门控在 kernel 之前先在本地算好，所有 token 与它们的 router_logits 一起 gather，top-k 路由在 kernel 里对 gather 后的全部 token 做。
- **量化**：`prepare()` 先调用 `_quantize_and_setup_dispatch`，是否在 gather 前量化取决于 `defer_input_quant`（FlashInfer 等单 kernel 后端可能推迟量化到 kernel 内）。本配置下取值没有核对，所以 gather 的数据类型（FP8 还是 bf16）不确定。
- **③**：参数 `local_expert_offset = ep_rank × 16`、`local_num_experts = 16`，与策略一的 kernel 调用相同。
- **④ combine**：`finalize()` → `get_ep_group().combine(...)` → `AgRsAll2AllManager.combine`：`dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)`。`sizes` 是各 rank 当前步的 token 数（`get_chunk_sizes_across_dp_rank`）。
- **`output_is_reduced()` 返回 `False`**（`NaiveDPEPMonolithic`），所以 runner 里 `_maybe_reduce_final_output` 的条件（`tp_size>1 或 ep_size>1`，这里 `ep_size=16` 成立）仍会调用 `tensor_model_parallel_all_reduce`；但 TP 组大小为 1，应当是空操作（推断，没有核对 world_size=1 时的短路）。

**与策略一（TP16）的对比**：

| | TP16 | DEP（本策略） |
|---|---|---|
| Attention 通信 | 每层一次 TP16 all-reduce | **无** |
| MoE 通信 | 每层一次 TP16 all-reduce | 每个 MoE 层 all_gatherv + reduce_scatterv（DP 组，16 个 rank） |
| 每步集合通信次数 | 122 次 all-reduce（61 层 × 2） | 116 次（58 个 MoE 层 × 2），前 3 层稠密层无通信 |
| 路由 | 每个 rank 对**整批** token 算门控和 top-k | 每个 rank 只对**自己的** token 算门控，top-k 在 gather 后的全部 token 上做 |
| 每步 Gloo 同步 | 无 | 有（源码；日志未直接确认） |

两者的通信数据量不同：TP16 每次 all-reduce 的大小是整批 token × hidden；DEP 的 all_gatherv / reduce_scatterv 每个 rank 发送自己的 token、接收全部 token，量级相近但没有量化；本文不做性能分析。

## 8. 功能与行为测试（实测）

测试在策略一空闲时进行，所以没有两个实例之间的干扰。

1. **两个节点的 API**：`dep-0`、`dep-1` 的 `/v1/models` 都返回 `deepseek-ai/DeepSeek-V3`，`max_model_len=163840`。
2. **对话**：两个节点各一次，`finish_reason=stop`，25 个 completion token，文本正常（回答自称 OpenAI 的模型，是模型自己的内容），没有出现 BPE 乱码。工具调用没有测。
3. **经 Service `dep-api` 分流**：从 `dep-0` 内部访问 `http://dep-api:8000` 发 24 个小请求，两个节点各收到 **12 个**；节点内请求分散在各个 DP rank 上。
4. **空闲**：两个节点 16 张 GPU 利用率都是 **0%**。
5. **单个请求（400 token）**：由 engine 4（一个 DP rank）独立完成（生成的 400 个 token 全部记在 engine 4 上）；请求期间两个节点的 GPU 利用率都升到 90%+（`dep-0` 峰值各卡 98–100%，`dep-1` 峰值各卡 99–100%），其余 15 个没有请求的 rank 同样在忙。说明没有请求的 rank 陪跑（dummy batch，机制已由源码核对，见第 5.6 节；与 `vllm-dp8-ep-single-node.md` 6.1 记录一致）。请求前后利用率回到 0%。
6. **64 个请求、并发 32（每个 64 token）到 `dep-0` 的本机 8000 端口**：全部成功（64/64）；节点内 8 个 DP rank 完成的请求数为 {0:7, 1:7, 2:8, 3:9, 4:7, 5:8, 6:8, 7:10}，生成 token {448, 448, 512, 576, 448, 512, 512, 640}，分布比较均匀。这个负载只发给 `dep-0`，所以只有节点 0 上的 8 个 rank 分到了请求。
   - 网络计数器增量（这轮负载，约 15 秒）：两个节点的 RDMA `rx_write_requests` 都是 **+7,755,180**；`packet_seq_err` +912 / +474，`out_of_sequence` +480 / +915；eth0 每个节点收发各约 3.4 MB。数据面走 RoCE，eth0 上是小消息，与策略一一致。`packet_seq_err` / `out_of_sequence` 仍有增长，原因没有排查（见 `vllm-dp16-ep-two-node.md` 4.4）。
   - 计数器是自启动以来的累计值，只看增量；这只是一个短负载，不能与策略一的数据直接比较。

## 9. 与策略一（TP16）的实测对比

| 项 | 策略一：TP16 + EP16 | 策略三：DP16 + EP16（DEP） |
|---|---|---|
| 对外入口 | 1 个（节点 0 主进程内的 API server） | 16 个 API server 进程（每节点 8 个），两个节点都服务 |
| EngineCore | 整个实例 1 个 | 每个 DP rank 1 个，共 16 个 |
| 每卡权重 | 40.47 GiB | 58.02 GiB |
| 峰值激活预留 | 9.46 GiB | 58.9 GiB |
| 每卡 KV cache | 193.7 GiB | 128.39 GiB |
| KV 容量 | 2,959,680 tokens（16 份重复拷贝，全实例共享） | 每 rank 1,961,792 tokens，16 个 rank 合计 31,388,672（各存各的请求） |
| 稳态显存 | 约 246 GiB / 卡 | 约 199.6 GiB / 卡 |
| `init engine` | 912.6 秒 | 1137.96 秒 |
| Attention 通信 | 每层 1 次 all-reduce | 无 |
| MoE 通信 | 每层 1 次 all-reduce | 每个 MoE 层 all_gatherv + reduce_scatterv |
| all2all 后端 | 默认值，未被用到 | 默认值（`allgather_reducescatter`），被用到 |
| 每步 CPU 同步 | 无 | DP 同步走 Gloo（源码；日志未直接确认） |
| 空闲 / 单请求 GPU 利用率 | 0% / 16 卡都 90%+ | 0% / 16 卡都 90%+ |
| 需要 `--disable-custom-all-reduce` | 需要（TP16 组） | 不需要 |

## 10. 结论

- DEP 方案按页面命令加本集群所需的环境变量即可部署，不需要 `--disable-custom-all-reduce`；唯一遇到的问题是我写的部署脚本里的 DNS 竞态（已修）。
- 与 TP16 相比，DEP 用更多的每卡权重（58 vs 40 GiB）和大得多的峰值激活预留（58.9 vs 9.5 GiB）换来了：Attention 无通信、KV 不重复（全实例 KV 容量约 10.6 倍）、两个节点都能对外服务。
- 没有请求的 rank 仍要陪跑：只发 1 个请求，16 张卡的利用率也都升到 90%+；这与 TP16 的现象相同，但原因不同（TP16 是真的在共同计算，DEP 是 15 个 rank 空跑）。
- 性能没有测，本文不做性能结论；`strategy1.md` 第 9 节对 TP16 与 2×TP8 的理论分析也不适用于 DEP。

## 11. 未验证项

- **性能**：按要求没有做，吞吐、TTFT、TPOT 等都没有数据。
- DPCoordinator 的具体职责、两个 Pod 之间约 398 条 TCP 连接的用途、`--data-parallel-rpc-port` 在稳态下为什么没有监听者，都没有深挖。
- DP 步同步走 Gloo 是源码结论，日志里没有看到 "Using CPU all reduce" 那一行，没有在运行时验证；对每步延迟的影响没有测。
- gather 前激活的量化（`defer_input_quant`）在本配置下的取值没有核对，gather 的数据类型不确定；`output_is_reduced()=False` 后 TP 组 all-reduce 是否在 world_size=1 时短路没有核对。
- 峰值激活为什么达到 58.9 GiB（推断与 16 个 rank 的 token gather 有关）、profiling 阶段为什么比 TP16 长约 3 分钟，都没有验证。
- 共享专家和前 3 层稠密 FFN 在 DP 下是每卡完整副本，是根据 TP=1 的推断，没有单独核对；共享专家输出并入结果的位置没有读。
- 专家放置只核对了 rank 0 和 rank 8；Service 分流只测了 24 个来自同一个客户端的请求，更多客户端下的分布没有测。
- 数据面是否走 GPUDirect RDMA、具体 NCCL 传输没有打开 `NCCL_DEBUG` 确认；`packet_seq_err` 的原因没有排查。
- 工具调用（function calling）没有测。

## 12. 运维备注

- 服务当前仍在运行：`dep-0`、`dep-1`，占满节点 `3o1v8b`、`3o1v8e` 的 16 张 GPU；策略一的 `tp16-0`、`tp16-1` 同时在运行（占 `3o1v8g`、`3o1v8w`），四个节点的 GPU 现在全部被占用。
- 查看：`kubectl get pods`；日志：`kubectl logs -f dep-0`；验证：`kubectl exec dep-0 -- curl -s localhost:8000/v1/models`；经 Service：`kubectl exec dep-0 -- curl -s http://dep-api:8000/v1/models`。
- 停止：`kubectl delete -f dep-server.yaml`（HF 缓存保留）。
- 后续可做：在相同条件下做性能对比（TP16、DEP、2×TP8）；用 `--api-server-count`、`--kv-cache-memory` 调优；对比 hybrid-lb 与 internal LB（`--data-parallel-size-local` + headless）两种模式；排查 DP 步同步的 Gloo 路径。
