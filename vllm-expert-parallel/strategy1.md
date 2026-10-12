# 策略一：Multi-Node TEP（TP16 Attention + EP16 MoE）实测

DeepSeek-V3（FP8）在 2 个 B300 节点（每节点 8 卡，共 16 张 GPU）上作为**一个** vLLM 实例运行，对应 recipes 页面的 Multi-Node Tensor Parallel 方案。

**关于"TEP"**：本方案的启动参数是 `--tensor-parallel-size 16 --enable-expert-parallel`，即 Attention 用 TP16、MoE 用 EP16，按 `../concepts/` 概念文档的命名是 **TEP**（不是纯 TP：路由专家按 EP 布局，每卡持有 16 个完整专家，见第 6.3 节）。正文用"TEP"指整个方案，"TP16"只指它的 TP 通信组。对应 recipes 页面的 Multi-Node Tensor Parallel 方案。

- 实验日期：2026-10-11；镜像 `vllm/vllm-openai:latest`（实际是 vLLM 0.31.0），NCCL 2.30.7。
- 相关文件：`recipe.md`（页面内容整理，本方案见第 4 节）、`tep-server.yaml`（部署文件）；之前的 DP+EP 实验见 `vllm-dp8-ep-single-node.md`、`vllm-dp16-ep-two-node.md`。对照见 `strategy3.md`（DEP：DP16 + EP16，hybrid-lb）。
- 标注约定：**实测**＝本集群的命令输出或日志；**源码核对**＝读了镜像内 vLLM 0.31.0 源码；**推断**＝由日志、机制或一般知识得出，未直接验证。

## 1. 摘要

| 方面 | 结论 |
|---|---|
| 能否部署 | 能，但**页面命令不能直接用**：必须加 `--disable-custom-all-reduce`，否则启动时 16 个 rank 卡死在 custom all-reduce 的 MNNVL 初始化（第 4 节） |
| 启动耗时 | 权重已缓存时约 17 分钟（19:13:03 → 19:30:25）；大头是 KV cache 分配后的预热（autotune、CUDA graph），不是加载权重（第 4.1 节） |
| 进程 | 节点 0：主进程（内含 API server）+ 1 个 EngineCore + 8 个 Worker；节点 1：headless 主进程 + 8 个 Worker，**没有 EngineCore 和 API server**（第 5.1 节） |
| Attention | TP16，16 卡共同算同一个 token；**KV cache 每卡一份完整拷贝**（16 份重复，全实例只有 2,959,680 tokens 容量）（第 6.2 节） |
| MoE | 256 个路由专家分到 16 卡，每卡 16 个完整专家；共享专家和前 3 层稠密 FFN 按 TP16 切分（第 6.3 节） |
| 通信 | **每层两次覆盖 16 卡的 all-reduce**（Attention 一次 + MoE 一次），**没有 all-to-all**：输入在 16 卡上本来就相同，各卡自己算路由、只算自己的专家，最后一次 all-reduce 汇总；`--all2all-backend` 取默认值但未被用到（第 7 节）。标准 NCCL，**跨节点走 RoCE**（实测 RDMA write 计数器 +7.78M，对 eth0 仅 25 MB）；控制类通信用 Gloo 和 TCP（第 5.2、5.3 节） |
| 与 2×TEP8 对比（理论） | 理论上 2×TEP8 在低并发延迟、高并发吞吐、预填充、KV 容量（约 1.6 倍）、可靠性上占优；TEP16 只在中等批量下可能持平，窗口窄（第 9 节，**理论估算，未实测**） |
| 速度（粗测，**不可靠**） | 受前缀缓存命中和未预热影响，不建议引用，数据与说明见第 8.2 节 |
| GPU 利用率 | 空闲时 16 卡都是 0%；一个请求期间两个节点的 16 卡都在忙（第 8.3 节） |

## 2. 方案总览

`--tensor-parallel-size 16 --enable-expert-parallel`：16 个 Worker 组成一个 TP16 组，同时也是一个 EP16 组（EP = TP × DP = 16 × 1）。

| 部分 | 并行方式 | 每张卡上的内容 |
|---|---|---|
| Attention 的压缩投影（`q_a_proj`、`kv_a_proj_with_mqa`） | 不切分 | 完整副本（源码核对：`ReplicatedLinear`） |
| Attention 的其余部分 | TP16，按注意力头切分 | 128 个头中的 8 个（推断） |
| KV cache（MLA 的 latent） | 不切分 | 全部 token 的完整拷贝（实测数字核对，第 6.2 节） |
| 前 3 层稠密 FFN | TP16 | 1/16 |
| 路由门控 | 每卡各算一遍 | 门控权重（每卡是否一份副本未核对） |
| 256 个路由专家（后 58 层） | EP16 | 16 个完整专家，linear 放置（实测日志） |
| 共享专家 | TP16 | 1/16（源码核对，运行时未验证） |

每层通信：Attention 的 o_proj 之后一次 TP16 all-reduce；MoE 层结尾一次 TP16 all-reduce。

## 3. 部署

**文件**：`tep-server.yaml`
- StatefulSet `tp16`，2 副本，`podManagementPolicy: Parallel`（TP16 要求所有 rank 同时到齐），反亲和保证两个 Pod 落在不同节点。
- headless Service `tp16-headless`：给 Pod 稳定 DNS（`tp16-0.tp16-headless`），并设 `publishNotReadyAddresses`，Pod 未 Ready 时也能解析。
- Service `tp16-api`：8000 端口只指向 `tp16-0`。
- 每个 Pod 占 8 张 GPU 和 8 个 RoCE 网卡（`rdma/fabric0/2/…/14`），挂载 hostPath 的 HF 缓存和 128Gi `/dev/shm`；沿用 `n-server.yaml` 的网络与资源写法。

**角色怎么分**：没有选举，完全是静态约定。脚本取主机名最后一段作为 `--node-rank`：`tp16-0` → 0（对外服务），`tp16-1` → 1（`--headless`）。`--master-addr` 由脚本解析 `tp16-0.tp16-headless` 得到。rank 跟 Pod 名走，不跟物理节点走。启动顺序不影响角色：worker 先等主节点的 DNS 能解析，之后靠 vLLM 自己的重试连接（后一点没验证）。

**实际落点**：

| Pod | 节点 | Pod IP | 角色 |
|---|---|---|---|
| tp16-0 | rs-b300x8-gpu-pool-3o1v8g | 10.154.1.225 | 主节点 |
| tp16-1 | rs-b300x8-gpu-pool-3o1v8w | 10.154.1.88 | headless worker |

（部署过两次，落在相同节点；第二次 Pod 重建后 IP 变了。本文数据除特别说明外来自第二次。）

**启动命令**（Pod 内实际执行，节点 0；节点 1 改为 `--node-rank 1` 并加 `--headless`）：
```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --trust-remote-code --enable-expert-parallel \
  --tensor-parallel-size 16 --nnodes 2 --node-rank 0 --master-addr 10.154.1.225 \
  --disable-custom-all-reduce \
  --enable-auto-tool-choice --tool-call-parser deepseek_v3 --reasoning-parser deepseek_v3
```
环境变量：`GLOO_SOCKET_IFNAME=eth0`、`NCCL_SOCKET_IFNAME=eth0`、`NCCL_IB_GID_INDEX=3`、`VLLM_ENGINE_READY_TIMEOUT_S=3600`。

**与页面命令的差异**：
- 加了 `--disable-custom-all-reduce`（第 4.2 节）。
- 去掉了 `--chat-template examples/tool_chat_template_deepseekv3.jinja`：这是 vLLM 仓库里的相对路径，没有验证镜像里是否存在。工具调用因此没有测。
- 加了 `NCCL_IB_GID_INDEX=3` 和 `VLLM_ENGINE_READY_TIMEOUT_S`，来自 `vllm-dp16-ep-two-node.md` 的实测经验：不设 GID 时 NCCL 会选到链路本地 RoCE GID（`fe80::`），跨节点静默卡死；默认 600 秒的就绪超时不够。

## 4. 启动过程与故障

### 4.1 启动时间线（第二次部署，节点缓存里已有权重）

| 时间 | 事件 |
|---|---|
| 19:13:03 | vLLM 启动 |
| 19:13:30 | EngineCore 启动，`world_size=16, local_world_size=8` |
| 19:13:45–19:14:07 | 16 个 rank 初始化 NCCL，建立 TP 组和 EP 组 |
| 19:14:26–19:14:49 | 加载权重 23 秒（checkpoint 641.30 GiB；日志 "Model loading took 40.47 GiB and 59.8 seconds"） |
| 19:15:28–19:16:34 | torch.compile，共 84.8 秒 |
| 19:17:06 | 初始 profiling / warmup 32.3 秒 |
| 19:22:18 | 确定 KV cache：每卡 193.7 GiB |
| 19:25:04 起 | FlashInfer autotune：22 个 token bucket，每个 rank 约 1 分 47 秒 |
| 之后 | CUDA graph 捕获（PIECEWISE 83 个 + FULL 83 个） |
| 19:30:21 | `init engine` 完成，共 912.62 秒（其中编译 85.57 秒） |
| 19:30:25 | API server 启动，监听 8000 |

- 首次部署（权重未缓存）时，`hf download` 在两个节点各跑约 6 分钟（18:57 → 19:03），之后才启动 vLLM；hostPath 缓存在节点间不共享。
- 启动期大头不是加载权重，而是 KV cache 分配前后的预热：profiling 到 KV cache 确定之间约 5 分钟，autotune 加 CUDA graph 约 8 分钟（表中时间点之间的细分没有逐项计时）。

### 4.2 第一次失败：custom all-reduce 的 MNNVL 初始化卡死

**现象**：按页面命令（无 `--disable-custom-all-reduce`）启动，日志停在 `vLLM is using nccl==2.30.7` 之后，超过 5 分钟没有进展；GPU 显存只有约 1.9 GB，权重没有开始加载。日志里反复出现：
```
SymmMemCommunicator: World size 16 not supported, communicator is not available.
Custom collectives are disabled because this multi-node group does not support MNNVL multicast.
```

**定位**：
- 用 `py-spy dump` 看 16 个 Worker 的栈：14 个卡在 `torch/distributed/_symmetric_memory/__init__.py:2004 rendezvous`，调用方是 `vllm/distributed/device_communicators/custom_all_reduce.py:362 _init_mnnvl_buffer`；每个节点各有 1 个 Worker（local rank 7）已走到下一步的 `broadcast`。即 16 个 rank 在这个 collective 上没有对齐，互相等待。
- 与 `vllm-dp16-ep-two-node.md` 6.1 那次不同：这次 `NCCL_IB_GID_INDEX=3` 已设置，GID 表也正常（index 3 是 `fd02:...` 的 RoCE v2 地址），卡点不在 NCCL，而在 vLLM 的 custom all-reduce 初始化。
- 源码核对（`custom_all_reduce.py`）：构造函数在 `not same_node` 且没有 MNNVL multicast 时本来就会禁用自己，但**禁用发生在 `_init_mnnvl_buffer` 的 rendezvous 之后**，所以 rendezvous 必须先通过。

**处理**：加 `--disable-custom-all-reduce`。该参数让 `CustomAllreduce` 根本不会被构造，跳过这个 rendezvous；加上后启动成功。（`strategy3.md` 的 DEP 方案 TP=1，没有 TP 组，不需要这个参数。）

**未验证**：vLLM / PyTorch symmetric memory 在"没有跨节点 NVLink"的机器上做 rendezvous 时为什么没能对齐，没有深挖；是否所有 v0.31.0 的跨节点 TP 都会遇到，也不清楚。

### 4.3 背景：MNNVL 是什么，这批 B300 为什么不支持

- **MNNVL = Multi-Node NVLink**：把多台机器的 GPU 用 NVLink（经 NVLink Switch）连成一个域，跨机器的 GPU 之间也能走 NVLink，典型是 GB200 / GB300 NVL72 这类整机柜系统。
- **multicast**：NVLink / NVSwitch 的多播能力，一次写由交换机复制到多张 GPU，或在交换机内做归约；要求参与的 GPU 在同一个 NVLink 域内。（以上是一般了解，没有逐条对照 NVIDIA 文档。）
- **本集群的 B300 不支持 Multi-Node NVLink**（用户确认），实测与此一致：
  - `nvidia-smi topo -m`：同节点 GPU 之间是 `NV18`；GPU 与网卡之间是 `PXB / NODE / SYS`；没有指向其他节点的 NVLink。
  - `nvidia-smi -q` 的 Fabric 部分：`State: Completed`、`Status: Success`、`Health: Healthy`，但 `ClusterUUID` 全为 0、`CliqueId` 为 0；与"不属于任何跨节点 NVLink 域"一致（推断，没有对比 NVL72 机器）。
  - 这是 HGX 形态的 B300：节点内 NVLink，节点间走 8 个 RoCE 网卡。
- 所以日志里 "does not support MNNVL multicast" 是**预期内的结论**，真正的问题只是**顺序**：先做 rendezvous，后检查能力（见 4.2）。
- 对性能的含义：TP16 里跨节点的那部分通信只能走 RoCE，这是页面强调该方案需要 InfiniBand 的原因；在 NVL72 之类的机器上同样的 TP 组可全程走 NVLink，通信成本会低得多（推断，本集群没有这类机器）。

### 4.4 `--disable-custom-all-reduce` 是什么意思

帮助文字：*Disable the custom all-reduce kernel and fall back to NCCL.*

- custom all-reduce 是 vLLM 自带的 CUDA all-reduce 实现，与 NCCL 是两条不同的通信路径：用预先注册的 IPC 缓冲区让同一台机器内的 GPU 直接互相读写显存，且只处理小张量（`max_size = 8192 × 1024` 字节，约 8 MiB，更大的 `should_custom_ar` 返回假）。设计目的是降低解码阶段小张量 all-reduce 的延迟（一般了解，源码没有写明）。
- 默认开启时，每个通信组都会构造一个 `CustomAllreduce` 并做初始化；初始化后若不满足条件（不同节点、不支持 MNNVL multicast）会自己禁用。
- 加上参数后，不构造 `CustomAllreduce`，all-reduce 走剩下的后端：本次 `tp:0` 的后端列表是 `['FLASHINFER', 'PYNCCL']`，FlashInfer 初始化失败，实际落在 pynccl。
- **本次基本没有性能损失**：不加参数时，这个跨两节点的 TP16 组本来也会自己禁用 custom all-reduce，最终通信路径相同，区别只是前者会先卡死在初始化里。
- 单节点 TP8 下 custom all-reduce 可正常使用，禁用它会让小张量 all-reduce 改走 NCCL，延迟可能略高（推断，没有测）。
- 与 `--disable-nccl-for-dp-synchronization` 无关。

## 5. 运行时结构

### 5.1 进程（实测）

```
节点 0 (tp16-0, 10.154.1.225)                    节点 1 (tp16-1, 10.154.1.88)
─────────────────────────────                    ─────────────────────────────
PID 1   vllm serve（主进程，内含 API server，    PID 1   vllm serve --headless（主进程）
        :8000，日志标签 APIServer）               ├ Worker_TP8_EP8   → GPU0
 └ EngineCore（调度器）                           ├ Worker_TP9_EP9   → GPU1
    ├ Worker_TP0_EP0 → GPU0                      ├ ...
    ├ Worker_TP1_EP1 → GPU1                      └ Worker_TP15_EP15 → GPU7
    ├ ...
    └ Worker_TP7_EP7 → GPU7
 + resource_tracker                              + resource_tracker
```

- **API server 不是单独的进程，它就是节点 0 的 PID 1（`vllm serve` 主进程）本身**：监听 0.0.0.0:8000，处理 HTTP、分词、结果流式返回，所以进程列表里找不到名为 APIServer 的进程。依据：PID 1 的命令行就是 `vllm serve ...`；日志中所有 `(APIServer ...)` 行（共 73 行）的进程号都是 `pid=1`；8000 端口的监听者是 `vllm`，`pid=1`。
  - 之前的 DP16 实验里有 `APIServer_0..7` 独立进程，是因为用了 `--api-server-count=8`。该参数的帮助文字是 "How many API server processes to run. Defaults to data_parallel_size if not specified."；本次 DP=1 且没有指定，按帮助文字推断为 1 个，此时 API server 在主进程内运行（没有读代码确认默认值的取值逻辑）。
- **EngineCore**（节点 0）：整个实例只有一个，负责调度（continuous batching、KV cache 管理），16 个 Worker 共用。
- **Worker**（每 GPU 一个）：进程名里的 `TPn_EPn` 序号相同，即 TP16 组与 EP16 组由同一批 16 个进程构成。节点 0 的 Worker 是 EngineCore 的子进程；节点 1 的 Worker 直接是 PID 1 的子进程。
- **节点 1 没有 API server 和 EngineCore**：这就是 `--headless` 的含义，只提供 GPU 算力。
- Worker 的 CPU 占用约 50%。

### 5.2 控制面

- **API server → EngineCore**：同一 Pod 内的本地通道（具体机制没验证）。
- **EngineCore → 16 个 Worker**：每个调度步把调度结果发给全部 16 个 Worker。日志里 `DP group leader ... mq_connect_ip=10.154.1.225 (local)` 和反复出现的 `shm_broadcast` 提示，说明用的是共享内存广播队列；节点 1 的 Worker 不在同一台机器上，推断它们通过 TCP 连到节点 0（跨节点这一段没有抓包验证）。
- **`--master-addr` 的 29501 端口**：`torch.distributed` 的 TCPStore，由节点 0 的 `Worker_TP0_EP0`（rank 0）监听。实测该端口有 16 个已建立连接：8 个来自本节点，8 个来自节点 1，即 16 个 rank 都连到 rank 0。
- 节点 0 上到节点 1 的 TCP 连接约 800 条、本机内约 330 条；对应 NCCL bootstrap / 代理线程、消息队列等，各条用途没有逐一确认。

**Gloo：每个通信组配套的 CPU 侧后端**

vLLM 的 `parallel_state.py` 为每个 GPU 通信组（比如 TP16 组）同时建两个进程组：`device_group`（NCCL，传 GPU 张量）和 `cpu_group`（Gloo，传 CPU 上的小信息，不传 GPU 张量）；默认进程组的后端是 `cpu:gloo,cuda:nccl`。Gloo 走普通 TCP socket，本次通过 `GLOO_SOCKET_IFNAME=eth0` 指定走 eth0。

| 用途 | 源码位置（`parallel_state.py`） | 说明 |
|---|---|---|
| 建立设备通信器 | 546–555 | 创建 `device_communicator` 时传入 `cpu_group`，NCCL 自己的初始化信息等要靠它先交换 |
| 创建共享内存消息队列 | 561–564、612–632 | `MessageQueue.create_from_process_group(self.cpu_group, ...)` 用 Gloo 在各 rank 之间交换队列句柄，之后调度结果通过这个队列发给各 Worker |
| 广播 Python 对象 | 842–860 | 没有消息队列时用 `broadcast_object_list`（Gloo）；有的话走消息队列 |
| 广播张量字典的元数据 | 978–990 | 元数据走 `cpu_group`，张量本体走 `device_group`（NCCL） |
| 同步 | 1356–1363 | `barrier` 特意用 CPU 组；源码注释：NCCL 的 barrier 会悄悄创建 GPU 张量，容易搞乱当前设备 |
| 拓扑判断 | 1881–1884、1913、1962–1970 | 用 `cpu_group` 判断各 rank 是否在同一节点、节点数是多少 |

- 与本次实验直接相关：判断"这个 TP 组是否跨节点"依赖 `cpu_group`（`in_the_same_node_as`）；4.2 里 "这个组不是同节点，所以禁用 custom all-reduce" 的判断，来源应当就是这类基于 Gloo 的拓扑信息（根据调用关系推断，没有逐行追踪那条分支）。
- 为什么要设 `GLOO_SOCKET_IFNAME`：Gloo 需要指定网卡；在多网卡机器上（eth0 加 8 个 RoCE 网卡）不指定可能选到不可达的接口而卡住。`expert-parallel.md` 也记录了这个用途：InfiniBand 集群上强制 torch distributed 用以太网做发现，避免初始化卡住。
- 对运行时的影响（推断，没有测）：Gloo 不在每个 token 的关键路径上，每层的 all-reduce 是 NCCL；Gloo 主要用于启动阶段交换信息和低频的同步、对象广播。实测的 eth0 流量里 Gloo 的占比没有单独统计。
- **与 `strategy3.md` 里 Gloo 的区别**：这里的 Gloo 是 TP16 组的 `cpu_group`，用于启动阶段、对象广播和拓扑判断，不在每步的关键路径上；DEP 方案（DP>1）里还有每步的 DP 元数据同步（Worker 级）和 wave 共识（EngineCore 级）走 Gloo，见 `strategy3.md` 第 5.2 节。TEP 的 DP=1，没有后两类。

### 5.3 数据面

日志里各通信组选用的 all-reduce 后端（实测，按优先级顺序）：

| 组 | 成员 | 后端 |
|---|---|---|
| `tp:0`（TP16） | 16 个 rank | `['FLASHINFER', 'PYNCCL']` |
| `ep:0`（EP16） | 同样 16 个 rank | `['PYNCCL']` |

- 通信库是 NCCL 2.30.7（pynccl 封装）。
- FlashInfer all-reduce 的 workspace 初始化失败（`[fd_exchange] timed out connecting to /tmp/cuda_fd_xchg_...`，`backend=mnnvl`），因此 "AllReduce fusion pass is disabled"；`tp:0` 的 all-reduce 实际落在 PYNCCL。
- MoE 层结尾的汇总调用的是 `tensor_model_parallel_all_reduce`（源码核对），即 **TP 组**；`ep:0` 组在这条路径上是否被用到，没有查。
- **`--all2all-backend`** 没有显式指定，取默认值 `allgather_reducescatter`（`config/parallel.py:202`），但**本配置下没有被用到**：
  - `use_sequence_parallel_moe` = 后端在列表内 **且** EP 开启 **且** `tensor_parallel_size > 1` **且** `data_parallel_size > 1`；本配置 DP=1，为假。
  - `use_all2all` = `data_parallel_size > 1` 或 `use_sequence_parallel_moe` 或（EP + prefill context parallel）；本配置为假。
  - 所以 MoE 走的是 `MoEPrepareAndFinalizeNoDPEPMonolithic`（`prepare_finalize/no_dp_ep.py`），没有 all-to-all，也不涉及 DeepEP 等后端；换 `--all2all-backend` 对 TP16 方案没有意义（推断）。
  - 对比 DP16 + EP：DP>1，`use_all2all` 为真，实际用的就是默认的 `allgather_reducescatter`（见 `vllm-dp8-ep-single-node.md` 第 5 节）。

**跨节点通信栈总览：GPU 数据只走标准 NCCL（RoCE），但整个系统不只有 NCCL**

| 通信 | 用途 | 走的路径 |
|---|---|---|
| NCCL（`libnccl.so.2`，2.30.7，经 pynccl 封装） | 每层 Attention 和 MoE 的 all-reduce | 节点内 NVLink，跨节点 RoCE |
| Gloo | torch.distributed 的 CPU 进程组（见 5.2） | eth0 TCP |
| TCPStore | rank 会合（29501 端口，16 个连接） | eth0 TCP |
| vLLM 消息队列 | 调度结果从 EngineCore 发给 16 个 Worker | 本机共享内存；跨节点推断为 TCP |

- **库（实测，`/proc/<pid>/maps`）**：Worker 进程加载了 `libnccl.so.2`、`libibverbs`、`libmlx5`。vLLM 通过 pynccl 封装调用 NCCL，没有自定义的跨节点实现。
- **流量走哪里（实测）**：在 256 个请求（并发 64，随机 512 in / 128 out，约 15 秒）前后读计数器：

| 计数器 | 负载期间的增量 |
|---|---|
| RoCE `rx_write_requests`（`rdma statistic show link mlx5_N/1`，8 个 fabric 网卡合计） | 节点 0、节点 1 各 **+7,778,016** |
| eth0 | 节点 0 发送 +25.7 MB，节点 1 接收 +25.7 MB |

  大块数据走 RoCE（RDMA write），eth0 上只有几十 MB 的小消息，量级相差很大。说明：本集群这些网卡的 sysfs 里没有 `counters` / `hw_counters` 目录，所以用 `rdma statistic` 读取；计数器是自启动以来的累计值，只看了增量。
- **没有验证**：没有打开 `NCCL_DEBUG`，所以具体用了哪种 NCCL 传输（`NET/IB`、是否 GPUDirect RDMA）是从计数器推断的，没有在 NCCL 日志里直接确认。
- **RoCE 序列错误仍然存在**：同一轮负载里，`packet_seq_err` 节点 0 增加 16,407、节点 1 增加 11,504；`out_of_sequence` 分别增加 11,846 和 16,845；`local_ack_timeout_err`、`req_cqe_error`、`resp_cqe_error` 等其他错误计数为 0。与 `vllm-dp16-ep-two-node.md` 4.4 的遗留问题是同一类现象，原因仍未排查。
- **加载了但没有用到的库**：Worker 进程里还加载了 `libnvshmem_host`、`libtorch_nvshmem`、`libnixl*`、`libucp/ucs/uct`（UCX）。它们是 torch / vLLM 导入时带进来的，加载不等于使用：本配置没有 all-to-all（DP=1，见上），所以没有 DeepEP、`nixl_ep` 等后端；没有配置 KV transfer，所以没有 NIXL/UCX 的 KV 传输；custom all-reduce 已禁用，FlashInfer all-reduce 初始化失败。以上是根据配置和日志得出的，没有对这些库做运行时追踪。
- **环境变量核对**：Worker 和 EngineCore 进程的 `/proc/<pid>/environ` 里看不到 `NCCL_*` / `GLOO_*`，但 PID 1 和没有改过进程名的子进程（resource_tracker）能看到全部变量（`NCCL_SOCKET_IFNAME`、`GLOO_SOCKET_IFNAME`、`NCCL_IB_GID_INDEX`、`VLLM_ENGINE_READY_TIMEOUT_S`）。推断是被 `setproctitle` 改名的进程覆盖了那块内存，所以 `/proc/<pid>/environ` 对它们不可靠；没有直接证明，也没有用别的方法确认 Worker 内的 NCCL 实际读到了这些变量（NCCL 能跨节点正常工作是间接证据）。

## 6. 模型切分与显存

### 6.1 权重与显存（实测）

- Checkpoint 641.30 GiB（日志），16 卡均分约 40.1 GiB / 卡；实际每卡权重 40.47 GiB（日志）。
- 稳态每卡显存约 246 GiB / 268.6 GiB（`nvidia-smi` 报 251,960 MiB / 275,040 MiB；`--gpu-memory-utilization` 默认 0.92，0.92 × 268.6 ≈ 247 GiB）。其中权重 40.47 GiB、KV cache 193.7 GiB，其余约 11.9 GiB（CUDA graph、激活等，由差值得出）。
- 每卡 KV cache 193.7 GiB；全实例 KV cache 2,959,680 tokens；按 163,840 token 的单请求最大长度算，最大并发 18.06 倍（日志）。
- 后端（日志）：Attention 用 `FLASHINFER_MLA`，prefill 用 `FLASH_ATTN MLA`；MoE 用 `FLASHINFER_TRTLLM Fp8 MoE backend`；MoE 准备/收尾组件为 `MoEPrepareAndFinalizeNoDPEPMonolithic`。
- 每卡权重的构成（估算）：每个专家 3 × 7168 × 2048 ≈ 44.04M 参数，FP8 约 44 MB（`hidden_size=7168` 是 DeepSeek-V3 的已知值；6.2 节读过 `config.json`，但这一项没有单独核对）；16 个专家 × 58 层 ≈ 38.1 GiB。与 40.47 GiB 相差约 2.4 GiB，是不属于 EP 的部分：Attention 的 TP 分片和 MLA 压缩投影、前 3 层稠密 FFN、共享专家、embedding、lm_head。数量级吻合，拆分是估算，不是实测。

### 6.2 KV cache：每张卡一份完整拷贝

**结论**：每张 GPU 都保存**全部 token** 的完整 KV cache，16 张卡里是 16 份相同的拷贝，没有切分。

**原因**：普通多头注意力有多个 KV 头，TP 可以按头切分；MLA 缓存的不是各头的 K/V，而是压缩后的**一个**潜在向量（latent），没有"头"这个维度可分，各个头是计算时才从 latent 展开的。所以每张卡每一步都需要完整的 latent。MLA 的压缩投影在源码里是 `ReplicatedLinear`（源码核对），与此一致；"KV 为什么不能切"这个机制本身是一般了解，下面的数字核对支持它。

**用实测数据核对**（`config.json` 与日志）：

| 项 | 值 |
|---|---|
| `num_hidden_layers` | 61 |
| `kv_lora_rank` + `qk_rope_head_dim` | 512 + 64 = 576 个元素 / 层 / token |
| 数据类型 | `torch_dtype` = bfloat16（假设 KV cache 跟随模型类型，日志里没有直接确认 KV dtype） |
| 每 token 的 KV | 576 × 2 字节 × 61 层 = 70,272 字节 |
| 日志：全实例 KV cache 容量 | 2,959,680 tokens |
| 2,959,680 × 70,272 字节 | = 193.7 GiB |
| 日志：每卡可用 KV cache 显存 | 193.7 GiB |

两者吻合：每张卡的 193.7 GiB 刚好装下全部 2.96M 个 token 的完整 KV。其他日志：KV cache block size 32（`FLASHINFER_MLA` 后端），布局 `LBNHC`。

**代价**：
- 容量：全实例只能缓存 2,959,680 tokens，不是 16 倍；16 卡的 KV 显存里 15/16 是重复的。
- 带宽：每一步每张卡都要读自己那份完整 latent，KV 读流量也是 16 份重复。
- 与 DP 对比（推断，没测）：DP 下每个 rank 只缓存分给自己的请求，各 rank 的 KV 装的是不同请求，总容量约为 DP 数的倍数；这是页面把 DP 方案标为高吞吐的原因之一。
- 长上下文并发受限于单卡 KV 容量：最大并发约 18.06 倍的 163,840 token 请求（日志）。

### 6.3 EP：256 个路由专家怎么分到 16 张卡

**结论**：256 个路由专家按序号均分成 16 组，每张 GPU 持有一组，**每组 16 个完整的专家**；单个专家不会再被切碎。

- 配置（`config.json`）：`n_routed_experts=256`，`n_shared_experts=1`，`num_experts_per_tok=8`，`first_k_dense_replace=3`（前 3 层稠密 FFN，后 58 层 MoE），`moe_intermediate_size=2048`，`n_group=8`，`topk_group=4`。
- 日志（实测）：`[EP Rank 0/16] ... placement strategy: linear. Local/global number of experts: 16/256`，rank 0 的本地专家 0–15 对应全局 0–15；`[EP Rank 8/16] ... 0->128`，rank 8 的本地第 0 个专家是全局 128 号。规律：rank r 持有全局专家 `16r … 16r+15`（只核对了 rank 0 和 rank 8，其余按规律推）。
- 不开 EP 时，每个专家会被切成 16 份，16 卡各存每个专家的 1/16；开 EP 后每张卡存 16 个完整的专家。
- **共享专家不在 EP 里**：源码（`deepseek_v2.py`）中它是普通的 `DeepseekV2MLP`，`reduce_results=False`；只有 sequence-parallel MoE 才会 `disable_tp`，而该模式要求 DP>1（`config/parallel.py` 的 `use_sequence_parallel_moe`），本配置不满足，所以它按 **TP16** 切分。前 3 层稠密 FFN 同样按 TP16 切分。日志没有确认是否走了 fused shared expert 的路径。

## 7. 一层 MoE 的计算流程（源码核对）

**Attention 之后没有"路由通信"：16 张卡各自独立算出一样的路由结果，只处理落在自己那 16 个专家上的 token，最后做一次 all-reduce 汇总。** 依据：`deepseek_v2.py`、`fused_moe/runner/moe_runner.py`、`prepare_finalize/no_dp_ep.py`、`experts/trtllm_fp8_moe.py`。

```
Attention 的 o_proj 之后（TP16 all-reduce）
   └→ 16 张卡的 hidden_states 完全相同，都是整批 token
          │
          ▼  DeepseekV2MoE.forward → self.experts(...)
   ① 每张卡各自算门控：router_logits = gate(hidden_states)   ← 无通信
   ② 每张卡各自选 top-8 专家（分组 top-k）                    ← 无通信，结果相同
   ③ 每张卡只执行"全局专家号 ∈ [16·r, 16·r+16)"的那部分      ← 无通信
   ④ 每张卡得到部分和：命中自己专家的 token 有值，其余为 0
   ⑤ 加上共享专家（TP 分片）的部分和
   ⑥ 对 TP 组（16 卡）做一次 all-reduce                      ← 唯一的通信
          │
          ▼
   16 张卡得到相同的 MoE 层输出，进入下一层
```

- **前提**：`config/parallel.py` 的注释写明 "The all_reduce at the end of attention (during o_proj) means that inputs are replicated across each rank of the tensor parallel group."。
- **不切分 token**：`DeepseekV2MoE.forward` 里只有 `is_sequence_parallel` 为真才会 `sequence_parallel_chunk`；它要求 DP>1，本配置不满足，所以每张卡处理整批 token。
- **① 门控**：`FusedMoE` 构造时传入 `gate=self.gate`（`GateLinear`），`moe_runner.py:903–910` 在每个 rank 上执行 `self.gate(hidden_states)`；输入相同所以 router_logits 相同，不需要广播。
- **②③ 选专家与执行本地专家**：`NoDPEPMonolithic.prepare()` 只做输入量化，把 `router_logits` 原样交给 kernel；调用 `flashinfer.fused_moe.trtllm_fp8_block_scale_moe`，参数里有 `local_expert_offset = ep_rank × local_num_experts`、`local_num_experts = 16`（`trtllm_fp8_moe.py:323, 512–524`）。即每个 rank 的 kernel 按全部 256 个专家得出 top-8，但只执行落在 `[16·ep_rank, 16·ep_rank+16)` 内的专家。kernel 内部的分组 top-k 与 token 排序没有读（FlashInfer 的 CUDA kernel），以上来自调用参数。
- **④ 部分和**：`finalize()` 直接返回 kernel 输出，不做通信；`output_is_reduced()` 返回 `False`，表示输出还没有跨卡汇总。
- **⑤⑥ 共享专家与汇总**：共享专家输出也是部分和；`moe_runner.py` 中 `result = shared_output + fused_output`，再进 `_maybe_reduce_final_output`，满足条件时调用 `tensor_model_parallel_all_reduce`。条件：非 sequence-parallel（成立）、`skip_final_all_reduce` 为假（它要求 `reduce_results=False`，而 `DeepseekV2MoE` 的默认是 `True`，没有确认调用处是否传了 `False`；日志里 allreduce 融合已被禁用，与此一致）、`tp_size>1 或 ep_size>1`（成立）、输出未汇总（成立）。
- 数据量：每次 all-reduce 是 token 数 × 7168 × 2 字节（bf16 hidden state，按 `hidden_size=7168` 估）。

**为什么不需要额外的路由机制**：
- 输入已在 16 张卡上完全相同，专家又是按卡固定分好的，所以每张卡只要自己判断"哪些 token 选中了我的专家"，算完自己的部分即可；没有 token 需要发送到别的卡，不需要 dispatch。
- **路由的计算仍然存在，只是不需要通信**：门控和 top-8 选择在每张卡上都完整算一遍（16 份重复、结果一致）。它是 7168 → 256 的一个线性层加选 top-8，相对专家计算应当可以忽略（推断，没有测）。
- **最后的 all-reduce 兼做两件事**：① 相当于 all-to-all 方案里的 combine：一个 token 选中的 8 个专家分散在不同的卡上，各卡算出的是部分和，all-reduce 把它们加起来；② 汇总按 TP16 切分的共享专家的部分和。

**每层通信**：Attention 一次 + MoE 一次，共两次覆盖全部 16 张卡的 all-reduce，其中一部分必须跨节点；61 层累积（前 3 层稠密 FFN 按 TP 切分，预计也各有一次 all-reduce，推断，未核对），网络延迟直接进入每个 token 的时间。全程没有 all-to-all。

**代价（推断，均未测）**：
- 通信量：all-reduce 传的是整批 token 的完整 hidden state，与 top-k 无关；all-to-all 方案按每个 token 实际选中的专家分发，批量越大差别可能越明显。
- 负载不均：每张卡处理落在自己专家上的 token，热门专家所在的卡更忙，其他卡要等它才能进入 all-reduce；vLLM 的 EPLB（`--enable-eplb`）就是为此设计的，本次没有启用。
- Attention 与 KV cache 的 16 倍重复（第 6.2 节）。

**对比 DEP（DP16 + EP）**：完整对比表见 `strategy3.md` 第 9 节。要点：DP 下每个 rank 的 token 不同，专家要处理别的 rank 的 token，必须先 all-gather（或 all-to-all 分发），MoE 后再 reduce-scatter 回去；TEP 因为输入本来就是完整副本，跳过这一步，代价是 Attention 与 KV 的重复。

## 8. 测试（实测，粗测，非严格基准）

测试方法：对话 + 单流速度，以及 `vllm bench serve`。

### 8.1 功能
- `/v1/models` 返回 `deepseek-ai/DeepSeek-V3`，`max_model_len` 163840。
- 对话请求 `"Who are you? Answer in one sentence."`：`finish_reason: stop`，25 个 completion token，耗时 2.49 秒，文本正常可读（回答自称 OpenAI 的模型，是模型自己的内容，与部署无关）；没有出现 `vllm-dp8-ep-single-node.md` 第 7 节记录的 BPE 符号乱码。
- 工具调用（function calling）没有测。

### 8.2 速度

| 场景 | 输出 tok/s | TTFT | TPOT |
|---|---|---|---|
| 单流 256 token，`ignore_eos`，prompt 约 10 token（脚本内计时） | 68.2 | — | — |
| `vllm bench`，并发 1（8 请求，随机 512 in / 128 out） | 57.8 | 525 ms | 13.3 ms |
| 并发 64（256 请求） | 879 | 4182 ms | 40.4 ms |
| 并发 512（2048 请求） | 2014 | 6340 ms | 205.6 ms |

- **以下速度数据不可靠，仅供参考**：服务端开着前缀缓存（日志里 `enable_prefix_caching=True`，与 recipes 页面"默认关闭"的说法不一致），而这几次 `vllm bench` 都用了默认随机种子，重复运行会使用相同的 prompt，导致部分请求命中缓存（日志中的命中率在 0%–50% 之间波动）；同时没有预热。后来在同一实例上重跑并发 64 时输出吞吐为 2122 tok/s，而第一次是 879 tok/s，差异很可能主要来自这两点。这里的数字不能用来与其他策略比较，已决定不再做性能测试。
- 所有请求都成功（2048 / 2048，无失败）。
- 并发从 64 升到 512，总吞吐只从 4397 增到 10072 tok/s（约 2.3 倍），TPOT 从 40 ms 涨到 206 ms；原因没有分析（推断可能与每步的跨节点集合通信和调度有关）。
- **与已有数据对比要谨慎**：`vllm-dp16-ep-two-node.md` 的 DP16 数据用的是 `DeepSeek-V3-0324`、vLLM 0.19.2、另一套请求脚本（prompt 很短），本次是 `DeepSeek-V3`、0.31.0、512 token prompt。
  - 单流：TEP 约 68 tok/s，DP16 当时约 34.6 tok/s、单节点 DP8 约 60 tok/s；方向上与"TP 单请求延迟更低"一致，但条件不同。
  - 高并发：本次并发 512 输出 2014 tok/s，DP16 当时 512 并发 8203 tok/s，但负载形态不同；需要同一镜像、同一脚本重跑才能下结论。

### 8.3 GPU 利用率
- 服务空闲：两个节点的 16 张 GPU 都是 **0%**。
- 一个请求（400 token，5.5 秒）期间，节点 1 的 8 张 GPU 利用率为 90–99%（节点 0 同样为 94–99%）；请求前后都是 0%。
- 符合 TP 的语义：每个 token 都由全部 16 张卡共同计算，所以一个请求就会让两个节点的 GPU 都参与。这与 DP+EP 不同（`vllm-dp16-ep-two-node.md` 5.1：没有请求的 rank 要陪跑 dummy batch）；TP 下不存在"没有请求的 rank"。

## 9. 理论分析（估算，非实测）：TEP16（1 个实例）与 2 × TEP8（2 个实例）的对比

本节是**理论估算，没有做 TEP8 实测**；全部数字都是下界估算，不是实测或预测。目的是回答：在这套硬件上，为什么要把 16 张卡合成一个 TEP16 实例，而不是每个节点各跑一个 TEP8 实例？

### 9.1 比较对象与前提

- **方案 A**：本次部署，1 个实例，TEP16（TP16 + EP16），跨 2 个节点。
- **方案 B**：2 个实例，每个 TEP8（TP8 + EP8），各占 1 个节点，前面由外部负载均衡分发请求。
- 同样是 16 张 GPU、同一模型（DeepSeek-V3 FP8）。模型一个节点放得下：权重 641 GiB，一个节点 8 × 268.6 GiB ≈ 2149 GiB。所以跨节点不是为了"放得下"（页面的多节点方案主要面向放不下的硬件，比如 8×H100）。
- 硬件事实（实测）：节点内 NVLink，每卡 18 条链路 × 53.125 GB/s ≈ 956 GB/s（`nvidia-smi nvlink -s`）；节点间 8 个 RoCE 网卡，每个 400 Gb/s（`/sys/class/infiniband/mlx5_N/ports/1/rate`），合计线速 400 GB/s / 节点 / 方向；没有跨节点 NVLink（见 4.3）。
- 记号：B = 一个解码步里实例内的 token 数（约等于并发请求数）。对比时取**同样的总并发**：A 的 B 等于 B 方案两个实例的 B 之和，即每个 TP8 实例各 B/2。
- 本节所有数字均为**下界估算**，用于比较两个方案的相对关系，不是性能预测。通用假设：HBM 带宽约 8 TB/s（B300 的一般了解，没有核对这批机器）；路由均匀；通信与计算不重叠；忽略注意力、KV 读取、采样、CPU 调度等开销；集合通信延迟为经验假设（跨节点每次 30–60 µs，节点内 NVLink 每次 10–20 µs），没有测。

### 9.2 总览

| 维度 | A：TEP16 | B：2 × TEP8 | 理论上倾向 |
|---|---|---|---|
| 每卡权重 | 40.5 GiB（实测） | 约 80 GiB（估算） | A（但 B 的显存也够用） |
| KV cache 总容量 | 2.96M tokens（实测） | 约 4.7M tokens（估算） | **B（约 1.6 倍）** |
| 每层 all-reduce | 跨节点，经 RoCE | 全在节点内 NVLink | **B** |
| MoE 权重读取效率 | 每步读一遍，服务全部 token | 每个实例各读一遍 | **A（最高 2 倍）** |
| 低并发单步延迟 | 通信延迟大 | 通信延迟小 | **B** |
| 高并发吞吐 | 受跨节点带宽限制 | 受限较少 | **B** |
| 预填充（TTFT） | 通信量大 | 通信量较小 | **B** |
| 故障域 | 整个实例 | 剩一半容量 | **B** |
| 负载均衡 | 内置（单调度器） | 需外部负载均衡 | A |

### 9.3 显存与 KV cache

- **A（实测）**：每卡权重 40.47 GiB，KV cache 193.7 GiB，其余约 11.9 GiB（246.05 GiB 稳态占用减去前两项）；全实例 2,959,680 tokens。
- **B（估算）**：
  - 专家：32 个 × 44.04M 参数 × 58 层 ≈ 81.7 GB ≈ 76.1 GiB。
  - 非专家部分：A 里每卡约 2.4 GiB（实测权重 40.47 减去专家 38.06）；其中 MLA 压缩投影是每卡完整副本，不随 TP 变小，其余随 TP 分片、TP8 下翻倍，估算约 3–5 GiB。
  - 权重合计约 79–81 GiB；KV ≈ 246 − 80 − 11.9 ≈ 154 GiB；每 token 70,272 字节（见 6.2）→ 每实例约 2.35M tokens，**两个实例合计约 4.7M，约为 A 的 1.6 倍**。
- 原因：MLA 的 KV 每个 TP rank 都存完整一份（见 6.2），TP 越大，重复越多；权重省下来的显存（A 每卡少约 40 GiB）被这种重复吃掉了大半。
- 影响：同时能容纳的长序列并发数、前缀缓存容量，B 更大；对 163,840 的单请求上限没有影响（两者都远大于它）。

### 9.4 通信

- 每个解码步有 **122 次 all-reduce**（61 层 × 2：Attention 一次、MoE 或稠密 FFN 一次），两个方案都一样；每次的大小 S = B × 7168 × 2 字节 = 14,336 × B 字节，每步总量约 **1.75 MB × B**。区别只在：A 的每一次都要跨节点，B 的每一次都在节点内。
- **延迟项**（与 B 无关，对低并发影响最大）：
  - A：122 × (30–60 µs) ≈ **3.7–7.3 ms / 步**
  - B：122 × (10–20 µs) ≈ **1.2–2.4 ms / 步**
- **带宽项**（随 B 线性增长）：
  - A 跨节点：每步约 1.75 MB × B 经过每个节点的网卡，线速 400 GB/s 下需要 ≥ 4.4 µs × B：B = 512 时 ≥ 2.2 ms，B = 2048 时 ≥ 9 ms（线速下界，实际更慢）；节点内的 reduce-scatter / all-gather 段在 NVLink 上，量级略小（B = 2048 时约 6.6 ms）。
  - B：每个实例只有 B/2 个 token、全在 NVLink，B = 2048 时约 3.3 ms（环形 all-reduce 每卡流量按 2(N−1)/N × S 估）。
- 两个项合起来：低并发时 A 被延迟项拖累，高并发时 A 被带宽项拖累。

### 9.5 MoE 权重读取：A 唯一的"计算侧"优势

- 解码阶段的 MoE 是**权重读取带宽受限**，不是算力受限。每张 GPU 每步读的专家权重：
  - A：16 个专家 × 44.04 MB × 58 层 ≈ 40.9 GB（本地专家全部被命中时）→ 8 TB/s 下约 **5.1 ms**
  - B：32 个专家 → 81.7 GB → 约 **10.2 ms**
- 本地专家不一定全被命中：均匀路由（top-8 / 256）下，一个专家在含 B 个 token 的步里被选中的概率是 1 − (1 − 8/256)^B。把两个方案按同样总并发比较（B 方案每实例 B/2）：

| 总并发 B | A 命中比例 | B 方案命中比例 | 仅权重读取的吞吐比 A : B |
|---|---|---|---|
| 8 | 0.22 | 0.12 | 1.06 |
| 32 | 0.64 | 0.40 | 1.25 |
| 128 | 0.98 | 0.87 | 1.77 |
| 512 | 1.00 | 1.00 | 2.00 |

- 直观解释：A 把两个实例的批量合成一个，每个专家的权重每步只读一次却服务 2 倍的 token；B 的每个实例各读一遍。这正是大规模 EP 部署（专家数多、批量大）的收益来源。
- 局限：路由实际是倾斜的，命中的专家更少，差距会缩小；该优势只在内存带宽受限区间成立，算力成为瓶颈后两者趋同（FP8 峰值算力与 HBM 带宽之比决定转折点，粗估每个专家需要数百个 token，对应 B 约万级，没有核对 B300 规格）。

### 9.6 综合：单步时间下界

把 9.4 的通信和 9.5 的权重读取合起来（假设不重叠，取延迟假设的下限–上限）：

| 总并发 B | A：TEP16 单步（ms） | A 吞吐（k tok/s） | B：2×TEP8 单步（ms，每实例 B/2） | B 吞吐（k tok/s） | 倾向 |
|---|---|---|---|---|---|
| 8 | 4.8–8.5 | 0.9–1.7 | 2.5–3.7 | 2.2–3.3 | B（约 2 倍） |
| 32 | 7.1–10.7 | 3.0–4.5 | 5.3–6.6 | 4.9–6.0 | B |
| 128 | 9.2–12.9 | 9.9–13.9 | 10.3–11.5 | 11.1–12.4 | 基本持平 |
| 512 | 11.0–14.7 | 34.9–46.5 | 12.3–13.5 | 38.0–41.8 | 基本持平 |
| 2048 | 17.7–21.4 | 95.8–115.6 | 14.7–15.9 | 128.5–139.2 | B（约 1.2–1.3 倍） |

- 单请求（B = 1）：A 的下界 3.8–7.5 ms，B 方案的一个实例 1.5–2.8 ms。
- 解读：
  - 低并发：A 的跨节点延迟项主导，B 更好。
  - 中等并发（约 100–500）：A 的权重读取优势和通信开销大致抵消，两者在估算误差内持平；如果实际通信延迟比假设低，A 会略占优。
  - 高并发：A 的通信带宽项（线性增长）超过权重读取项（已饱和），B 更好。
- **与实测对照**：A 在 c=1 时的实测 TPOT 是 13.4 ms（只含解码，不受第 8.2 节说的缓存问题影响），高于下界 3.8–7.5 ms，说明还有约 6–10 ms 来自未估计的开销（kernel 启动、注意力、采样、CPU 调度等）。所以下界只能用来做相对比较，不能当作预测；这些固定开销对两个方案都存在，会让两者的差距比表中显示的小。

### 9.7 预填充（TTFT）

- 预填充受算力限制，但每个 all-reduce 的大小与 token 数成正比。按实测配置 `max_num_batched_tokens=16384`：一步 16,384 个 token，每次 all-reduce 235 MB，122 次共 **28.7 GB**。
  - A：跨节点线速下约 **72 ms** / 步；若节点内 NVLink 段（约 52 ms）不与跨节点段重叠，则约 125 ms。
  - B：每个实例节点内约 52 ms / 步，两个实例并行，折算到每 16,384 个 token 约 26 ms。
- 所以预填充的通信时间 A 约为 B 的 **3–5 倍**（72/26 ≈ 2.8，125/26 ≈ 4.8）。预填充的计算时间没有估（没有核对 B300 的 FP8 峰值），所以无法给出通信占总时间的比例，但通信项明显更大：**长 prompt 场景下 B 的 TTFT 更有优势**。

### 9.8 非性能因素

| 因素 | A：TEP16 | B：2 × TEP8 |
|---|---|---|
| 故障域 | 一个节点出问题，整个实例不可用 | 剩一半容量继续服务 |
| 网络依赖 | 依赖 RoCE 质量（实测 `packet_seq_err` 持续增长，原因未排查） | 不依赖节点间网络 |
| 启动与运维 | 约 17 分钟；需要 `--disable-custom-all-reduce` 才能起来（实测） | 单节点没有跨节点初始化问题；custom all-reduce 可正常使用（推断，没测） |
| 负载均衡 | 内置，一个调度器管全部 | 需要外部负载均衡；可做前缀缓存感知路由，但缓存分散在两个实例里 |
| 扩缩容粒度 | 以 2 节点为一组 | 以 1 节点为一个实例，滚动升级更灵活 |
| 调度瓶颈 | 一个 EngineCore 管 16 个 Worker，高并发下 CPU 调度可能成为瓶颈（推断，没测） | 每个实例的调度器只管 8 个 Worker |

### 9.9 结论与可验证的预测

- **在这套 HGX B300（节点间无 NVLink）上，理论上 2 × TEP8 在以下方面占优**：低并发延迟、高并发吞吐、预填充 / TTFT、KV 容量（约 1.6 倍）、可靠性；TEP16 只有在中等批量（总并发约 100–500）下才可能持平或略占优，窗口很窄。
- TEP16 的核心收益是"每个专家权重每步只读一次，服务更多 token"（最高 2 倍的权重读取效率）。要兑现它，跨节点通信必须足够便宜，比如有 MNNVL（NVL72 之类），或者不用"每层对全部 token 做 all-reduce"的方式，而是 DP attention + 只把 token 分发到选中的专家（即页面的 DEP，也是大规模 EP 部署的做法）。后者是之后对比其他策略的出发点。
- 页面把多节点 TP 标为"低延迟"，是在多节点方案之间比较，并没有与单节点 TEP8 实例比较，所以与上面的结论不矛盾。
- **可验证的预测（未实测）**：
  1. 单个 TEP8 实例的 c=1 TPOT 比 A 的 13.4 ms 低，量级约 2–5 ms（节点间通信延迟之差）。
  2. 总并发约 128–512 时，两个方案的吞吐差在 ±10% 以内。
  3. 总并发 ≥ 2048 或 8000 token 的长 prompt 时，2 × TEP8 明显更高（吞吐或 TTFT）。
  4. 单个 TEP8 实例的 KV cache 约 2.35M tokens。

## 10. 结论

- 在 B300 双节点上，TEP（TP16 + EP16）可以部署 DeepSeek-V3（FP8），16 张卡作为一个实例，节点 0 对外服务，节点 1 是 headless worker。
- 用 `latest`（0.31.0）镜像必须加 `--disable-custom-all-reduce`，原因是这批 B300 没有 MNNVL，vLLM 的初始化顺序使它卡死；加上后没有实际性能损失。
- 这个方案不需要路由通信（细节见第 7 节）：每层只有 Attention 和 MoE 各一次 16 卡 all-reduce，没有 all-to-all。代价是 Attention 与 KV cache 的 16 倍重复（第 6.2 节）。
- 单流速度高于之前的 DP16（68 vs 34.6 tok/s，条件不同）；高并发是否更优需要同条件对比。
- 启动慢（约 17 分钟），主要是预热而不是加载权重。
- 与 2 个 TEP8 实例相比，理论上 TEP16 跨节点在这套硬件上没有明显优势（第 9 节，估算）；是否值得，取决于是否有廉价的跨节点互联，或是否需要用 DEP 等方式来兑现 EP 的权重读取收益（对照 `strategy3.md`）。

## 11. 未验证项

- `--disable-custom-all-reduce` 对应的卡死根因没有定位；其他 vLLM 版本是否也会卡没有测。
- 5.2 中 EngineCore 到跨节点 Worker 的广播机制、各 TCP 连接的用途只是推断。
- 第 7 节的流程来自源码，没有在运行时（NCCL 层）抓取调用验证；FlashInfer TRTLLM kernel 内部的分组 top-k 与 token 排序没有读；`reduce_results` 在 `DeepseekV2MoE` 调用处是否被改为 `False`、`GateLinear` 是否每 rank 复制、是否走了 fused shared expert 路径、`ep:0` 组在这条路径上是否被用到，都没有确认；专家放置只核对了 rank 0 和 rank 8。
- 没有打开 `NCCL_DEBUG`，具体 NCCL 传输（`NET/IB`、GPUDirect RDMA）只是从 RDMA 计数器推断；`packet_seq_err` / `out_of_sequence` 在负载下持续增长，原因没有排查（丢包、乱序、PFC / ECN 配置等）。
- Gloo 在 eth0 流量里的占比没有单独统计；"加载了但没有用到"的 NVSHMEM、NIXL、UCX 库没有做运行时追踪；Worker 内 NCCL 是否读到 `NCCL_*` 环境变量没有直接确认（`/proc/<pid>/environ` 对改过进程名的进程不可靠）。
- FlashInfer all-reduce workspace 初始化失败（`fd_exchange` 超时）的原因没有查，它导致 allreduce-norm 融合被禁用，对性能的影响没有量化。
- 第 8 节的吞吐没有预热、单次运行；与 DP16 的对比因条件不同不能当作结论；也没有测更长 prompt、更多并发。
- 启动时间线中 19:17 → 19:22 → 19:25 → 19:30 之间各阶段的耗时没有逐项计时。
- `tool_chat_template_deepseekv3.jinja` 在镜像里的位置没有确认，工具调用没有测。
- KV cache 的数据类型是假设跟随模型（bfloat16），没有在日志里直接确认。
- 第 9 节全部是理论估算：没有部署 TEP8 做实测对比；HBM 带宽（约 8 TB/s）、集合通信延迟、均匀路由等都是假设，没有在这批机器上测量。

## 12. 运维备注

- 本文测试时（2026-10-11）`tp16-0`、`tp16-1` 占满节点 `rs-b300x8-gpu-pool-3o1v8g` 和 `3o1v8w` 的 16 张 GPU。**2026-10-12 查看集群时**，`tp16-*` 已不存在，这两个节点现在运行的是 `tppp-0`、`tppp-1`（`tppp-server.yaml`，见 `strategy2.md`），另外两个节点运行 `dep-0`、`dep-1`（`strategy3.md`）；四个 GPU 节点都被占用。
- 若要重新部署本方案：`kubectl apply -f tep-server.yaml`；日志：`kubectl logs -f tp16-0`；验证：`kubectl exec tp16-0 -- curl -s localhost:8000/v1/models`。
- 停止：`kubectl delete -f tep-server.yaml`（节点上的 HF 缓存保留，下次启动不用重新下载）。
- 后续可做：在同一镜像、同一脚本下与 TP8 × PP2、DEP（hybrid-lb）对比；排查 FlashInfer all-reduce 初始化失败；用 `NCCL_DEBUG=INFO` 确认传输路径；启用 `--enable-eplb` 看专家负载。
