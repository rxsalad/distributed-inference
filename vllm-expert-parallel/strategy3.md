# 策略三：Multi-Node Data + Expert Parallel（DP16 + EP16，hybrid-lb）实测

DeepSeek-V3（FP8）在 2 个 B300 节点（每节点 8 卡，共 16 张 GPU）上以 **DP Attention + EP** 方式运行，对应 recipes 页面的 Multi-Node Data + Expert Parallel（DEP）方案（在 `../concepts/` 概念文档中同样称为 DEP，命名见其 `parallelism-basics.md` 第 6 节）。本次与策略一同时部署在同一集群的另外两个节点上，便于对照。

**关于"策略一"的叫法**：策略一（`strategy1.md`，标题写作 TP16 + EP16）实际是 `--tensor-parallel-size 16 --enable-expert-parallel`，即 Attention 用 TP16、MoE 用 EP16，按概念文档的命名是 **TEP**，不是纯 TP。本文统一称其为"策略一（TEP）"；"TP16"只在指它的 TP 通信组时使用（如 custom all-reduce、all-reduce 的范围）。

- 实验日期：2026-10-11；镜像 `vllm/vllm-openai:latest`（实际是 vLLM 0.31.0），NCCL 2.30.7；节点 `rs-b300x8-gpu-pool-3o1v8b`、`3o1v8e`（策略一占用 `3o1v8g`、`3o1v8w`）。
- 相关文件：`dep-server.yaml`（部署文件）、`recipe.md` 第 6 节（页面内容整理）、`strategy1.md`（用于对比）、`vllm-dp16-ep-two-node.md`（更早的 DP16 实验）、`dummy-batch-why.md`（dummy batch 专题）。
- 标注约定：**实测**＝本集群的命令输出或日志；**源码核对**＝读了镜像内 vLLM 0.31.0 源码（只读，2026-10-11、10-12 在 `dep-0` 内查看）；**推断**＝由日志、机制或一般知识得出，未直接验证。
- **范围说明**：按要求不做性能测试，本文没有吞吐 / 延迟数据；测试只验证功能、请求分布、GPU 利用率和网络计数器。

**阅读指引**：想看结论读第 1、10 节；想看"进程与连接怎么组织"读第 5.1、5.5、5.7 节；想看"没有请求的 rank 为什么要陪跑"读第 5.6 节；想看"Gloo 与 NCCL 的分工"读第 5.2 节；想看显存与 KV 读第 6 节；想看一层 MoE 的计算与通信读第 7 节。

## 1. 摘要

| 方面 | 结论 |
|---|---|
| 能否部署 | 能，页面命令（加上本集群需要的环境变量）可直接使用；**不需要** `--disable-custom-all-reduce`（TP=1，没有 TP 组的 custom all-reduce，第 3 节） |
| 启动耗时 | 从 vLLM 启动到服务就绪约 22 分钟（22:46:58 → 约 23:08:50）；`init engine` 1138 秒，比策略一（TEP）的 912 秒长（第 4.1 节） |
| 进程 | 每节点：1 个主进程 + 8 个 API server + 8 个 EngineCore + 8 个 Worker；DPCoordinator 只在节点 0（第 5.1 节） |
| 对外入口 | **两个节点都对外服务**，各 8 个 API server 进程；用 Service 在两个 Pod 间分流（第 3、5.4 节） |
| 连接关系 | 每个 APIServer 只与**本 Pod** 的 8 个 EngineCore 全连接（8 × 8）；请求不跨 Pod 转发，跨 Pod 的只有到 DPCoordinator 的连接和 Worker 间的集合通信（第 5.5 节） |
| Attention | 每个 DP rank 在自己的 GPU 上独立计算，**无通信**；每卡一份完整 Attention 权重（第 2 节） |
| KV cache | **每个 rank 存自己请求的 KV，互不重复**：每卡 128.39 GiB、1,961,792 tokens，16 个 rank 合计 31,388,672 tokens（策略一 TEP 全实例只有 2,959,680）（第 6.2 节） |
| 每卡显存 | 权重 58.02 GiB（TEP：40.47）；**峰值激活预留 58.9 GiB**（TEP：9.46），所以 KV 比 TEP 每卡少（第 6.3 节） |
| MoE 通信 | 每个 MoE 层：DP 组 `all_gatherv`（dispatch）→ 本卡专家计算 → `reduce_scatterv`（combine）；all2all 后端是默认的 `allgather_reducescatter`，**确实被用到**（DP>1）（第 7 节） |
| 控制面 | DPCoordinator 负责负载统计与 DP wave 协调（第 5.7 节）；DP 同步在异步调度下走 **CPU 上的 Gloo**（源码），与 NCCL 数据面分工不同（第 5.2 节） |
| 请求分布 | 单个请求只由一个 DP rank 处理，但**两个节点的 16 张卡利用率都升到 90%+**（其余 rank 陪跑 dummy batch，第 5.6 节）；节点内 64 个请求在 8 个 rank 间分布为 7–10 个（第 8 节） |
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

与策略一（TEP）的区别：TEP 里 Attention 是 TP16，由 16 张卡共同计算、每层 all-reduce，KV 每卡一份完整拷贝；这里 Attention 各算各的、没有通信，KV 不重复，代价是权重在每张卡上多存一份完整的 Attention（第 6 节）。

## 3. 部署

**文件**：`dep-server.yaml`
- StatefulSet `dep`，2 副本，`podManagementPolicy: Parallel`，反亲和保证两个 Pod 落在不同节点；`tp16` 占用的两个节点 GPU 已满，调度器自然落到另外两个节点。
- headless Service `dep-headless`：稳定 DNS（`dep-0.dep-headless`），`publishNotReadyAddresses: true`。
- Service `dep-api`：8000 端口选中**两个** Pod（hybrid-lb 下两个节点都对外服务），在 Pod 之间做连接级分流。这与策略一不同（策略一只有 `tp16-0` 提供 API）。
- 每个 Pod 占 8 张 GPU 和 8 个 RoCE 网卡，其余沿用 `tep-server.yaml`。

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
**没有 `--tensor-parallel-size`**：缺省值是 1，即不做张量并行，每个 DP rank 独占一张 GPU（日志里 Worker 名为 `Worker_DPn_EPn`，没有 TP 后缀）。总 GPU 数 = DP × TP = 16 × 1 = 16；`--enable-expert-parallel` 下 EP 大小 = DP × TP = 16，每卡 256 / 16 = 16 个路由专家。若显式设 `--tensor-parallel-size 2`，需配 `--data-parallel-size 8` 才用同样的 16 张卡，即页面描述里的"TEP"形态（本文没有测）。

**环境变量**：`GLOO_SOCKET_IFNAME=eth0`、`NCCL_SOCKET_IFNAME=eth0`、`NCCL_IB_GID_INDEX=3`、`VLLM_ENGINE_READY_TIMEOUT_S=3600`。

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

- profiling 到确定 KV cache 之间用了约 8.4 分钟，TEP 同一阶段约 5 分钟；原因没有深挖（可能与每步要处理 16 个 rank 的 token 有关，推断）。
- 该节点的 HF 缓存里已有权重，所以没有再下载。

### 4.2 第一次失败：dep-1 的 `HEAD_IP` 为空（DNS 竞态）

- **现象**：第一次部署后 `dep-0` 正常启动并等待，`dep-1` 一直循环打印 `waiting for DP master :13345`，日志里命令是 `</dev/tcp//13345`（地址为空）。
- **原因**：脚本先用 `until getent ahostsv4 <主节点域名>` 等待解析成功，再单独执行一次 `getent ... | awk` 取 IP。`dep-1` 上第一次调用返回成功，紧接着取 IP 却是空，是 DNS 的瞬时竞态；事后再查已经能解析到 `10.154.2.65`。
- **处理**：改成循环直到取到**非空** IP，然后删除并重新部署。`tep-server.yaml` 里有同样的潜在问题（策略一部署时没有触发），也一并修改了文件，但正在运行的 TP16 实例没有重启。
- **没有出现策略一的卡死**：策略一在 `latest` 镜像上因为 TP16 组的 custom all-reduce 初始化而卡死（见 `strategy1.md` 第 4 节）；本策略 TP=1，没有这个问题。

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
- **EngineCore**：**每个 DP rank 一个**（节点上共 8 个），各自有独立的调度器和 KV cache；每个 EngineCore 下面只有 1 个 Worker（`world_size=1`），因为 TP=1。TEP 里则是整个实例只有 1 个 EngineCore 管 16 个 Worker。
- **Worker**：进程名 `Worker_DPn_EPn`，DP rank 与 EP rank 相同；DP rank n 对应本节点 GPU `n % 8`（日志 `rank=n local_rank=n%8`）。
- **DPCoordinator**：只在节点 0 上有一个，监听 3 个随机端口；它是 APIServer 与 EngineCore 之间的**控制面协调者**，职责见第 5.7 节。
- 两个节点都有自己的 API server，EngineCore 与 Worker 数量相同，节点 1 并非 headless。

### 5.2 控制面与 Gloo / NCCL 的分工

- **DP 初始化**：`--data-parallel-address 10.154.2.110 --data-parallel-rpc-port 13345`。13345 端口在启动阶段可连（`dep-1` 在等它），稳态下 `ss` 里已经没有监听者。`torch.distributed` 的会合地址是 `tcp://10.154.2.110:38989`（`world_size=16`，随机端口）。
- 两个 Pod 之间稳态有约 398 条已建立的 TCP 连接（具体用途没有逐条确认）。

**Gloo 与 NCCL 的分工（源码核对）**：NCCL 在 GPU 上跑，负责大张量的数据面（如第 7 节 MoE 层的 `all_gatherv` / `reduce_scatterv`）；Gloo 在 CPU 上跑，只负责很小的**控制信息**（几个 int32 或一个布尔值）。DP 组里有两类这样的 Gloo 同步，层级不同：

| | Worker 级：每步的 DP 元数据同步 | EngineCore 级：wave 结束与暂停共识 |
|---|---|---|
| 源码 | `v1/worker/dp_utils.py`：`coordinate_batch_across_dp` → `_run_ar` | `v1/engine/core.py`：`_has_global_unfinished_reqs` → `ParallelConfig.sync_dp_state` |
| 频率 | 每个调度步一次（由源码结构推断，没有逐行核对调用点） | 第 1 步，之后每隔 `dp_sync_interval` 步一次 |
| 内容 | 每个 rank 填一个 4 × 16 的 int32 张量：本 rank 的 token 数、补齐后的 token 数、是否做 ubatch、cudagraph 模式；all-reduce 后每个 rank 看到全部 16 个 rank 的值 | 是否还有 rank 有未完成请求；暂停（`pending_pause`）是否达成共识 |
| 用途 | ① 启用 cudagraph（同步后的模式非 NONE）或 ubatch 时，把各 rank 补齐到同一个最大 token 数；② cudagraph 模式取各 rank 的最小值（任一 rank 为 NONE 则全体 NONE）；③ ubatch 只有全体 rank 都同意才做 | 决定 DP wave 是否结束、各 rank 是否一起暂停（第 5.6 节） |
| 通信后端 | 默认是 DP 组的设备组（NCCL）；异步调度下 `disable_nccl_for_dp_synchronization` 默认置为真，改用 `get_dp_group().cpu_group`（Gloo） | 固定是 Gloo：`stateless_init_dp_group` 用 `backend="gloo"` 建组 |

**为什么已经有 NCCL 还要 Gloo**：
- EngineCore 级：EngineCore 是独立于 Worker 的进程，源码注释写明 "use gloo since the engine process might not have cuda device"，它没有 GPU 上下文，没法用 NCCL。
- Worker 级：Worker 有 GPU，用 NCCL 也能做，但要把这个小张量从 GPU 搬回 CPU 读取，会引入一个 GPU 同步点，损害异步调度的性能（源码注释）。改用 CPU 上的 all-reduce，张量本来就在 CPU，读取不需要等 GPU。这是异步调度下的**默认选择**，由参数 `disable_nccl_for_dp_synchronization` 控制（命令行形式没有核对）。

**走哪条网络**：Gloo 走 TCP，本部署设置了 `GLOO_SOCKET_IFNAME=eth0`，所以跨节点的这类同步走 **eth0**，而 MoE 的 NCCL 数据走 RoCE（第 8 节：RDMA 计数器大幅增长、eth0 只有几 MB，与此一致）。

**配置确认与开销**：
- 日志里有 `Disabling NCCL for DP synchronization when using async scheduling.`，说明异步调度下开关已置为真（该开关在异步调度开启时默认置真，DP>1 且为 MoE 时才打这行日志）。
- **日志里没有看到 `_get_device_and_group` 的 "Using CPU all reduce to synchronize DP padding between ranks"，所以没有直接确认 Worker 级这条路径在运行时被执行**，原因没有查。
- Worker 级的数据只有 4 × 16 个 int32（256 字节），是延迟型的小 all-reduce，需要 16 个 rank 经 TCP 同步一次；TEP 的 DP=1，没有这类同步。对每步延迟的实际影响没有测。

### 5.3 数据面

- 日志（实测）：`dp:0` 与 `ep:0` 通信组的 all-reduce 后端都是 `['PYNCCL']`；`Using AgRsAll2AllManager all2all manager.`。
- `--all2all-backend` 没有显式指定，取默认的 `allgather_reducescatter`。DP>1 时 `use_all2all` 为真，所以**这次确实被用到**（策略一里没有用到，见 `strategy1.md` 5.3）。
- MoE 的准备/收尾组件：`MoEPrepareAndFinalizeNaiveDPEPMonolithic`（策略一是 `NoDPEPMonolithic`）。
- 通信走的组（源码核对，`all2all.py` 的 `_get_comm_group`）：非 sequence-parallel、DP world size > 1、没有 prefill context parallel，所以用 **DP 组**（16 个 rank）。
- 跨节点流量（实测，第 8 节）：RDMA write 请求计数大幅增加，eth0 只有几 MB，即 GPU 数据走 RoCE。

### 5.4 对外入口：hybrid-lb 与内部 LB 的区别

与 `vllm-dp16-ep-two-node.md`（内部 LB）相比，并行结构相同（DP16 + EP16、TP=1、默认 `allgather_reducescatter`），主要区别在对外入口：

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
- **入口**：内部 LB 下 Node 1 是唯一入口。hybrid-lb 下两个节点都能接请求，但 16 个 rank 在同一个 EP 组内，任一节点故障整个 EP 组都无法继续工作，所以并没有获得真正的高可用（推断）。
- **均衡粒度**：hybrid-lb 的跨节点均衡由外部 Service 完成，只到节点级，不感知节点内 rank 的负载。
- **不可直接比较**：两份文档的镜像（vLLM 0.19.2rc1 与 0.31.0）、模型（V3-0324 与 V3）、LB 模式都不同，吞吐不能直接对比。同环境下对比两种 LB 没有做过。

### 5.5 APIServer 与 EngineCore 的对应关系（实测 + 源码核对）

8 个 APIServer 与 EngineCore 不是一一对应，而是**每个 APIServer 都管理一组 EngineCore，组内全连接**；管理哪一组，由 LB 模式决定。

**连接关系一览**（每个 Pod 内 APIServer × 8、EngineCore × 8，hybrid-lb）：

| 连接 | 方式 | 范围 |
|---|---|---|
| APIServer ↔ EngineCore | unix socket（ZMQ `ipc://`），**8 × 8 全连接**，每对 2 条（下发请求、回传结果） | **只限本 Pod**：dep-0 的 APIServer 只连 DP0–7，dep-1 的只连 DP8–15 |
| APIServer ↔ DPCoordinator | TCP，每个 APIServer 1 条 | **跨 Pod**：协调者只在 dep-0，dep-1 的 8 个 APIServer 也连到它 |
| EngineCore ↔ EngineCore（另一个 Pod） | NCCL 集合通信、DP 步同步（Gloo） | 跨 Pod，**不经过 APIServer**（第 5.2、5.3 节） |

所以**请求本身不会跨 Pod 转发**：落到哪个 Pod 由外部 Service 决定，之后只在该 Pod 的 8 个 EngineCore 里选 rank。跨 Pod 的只有两类：协调者发布的负载统计（APIServer 只取自己管理范围内的切片），以及 EngineCore 之间的集合通信。

**实测（dep-0，用 `ss -xnp` / `ss -tnp` 按进程统计）**：
- 8 个 APIServer 中的每一个，与本节点 8 个 EngineCore（DP0–DP7）的每一个都有 **2 条 unix socket**，共 16 条，是 8 × 8 全连接。
- 每个 APIServer 只有 **1 条 TCP 连接**，连到 DPCoordinator（dep-0 的 `:44361`）。dep-1 的 APIServer 也连到 dep-0 的这个端口（跨节点）。
- APIServer 与 EngineCore 之间**没有 TCP 连接**，本机通信走 unix socket。这更正了旧文档里"APIServer ↔ EngineCore 走 TCP"的说法（旧文档对本机部分标的是"没有核对"）。
- 本节点 8 个 APIServer 看不到 dep-1 的 EngineCore（DP8–15）：没有任何 APIServer 到远端 EngineCore 的连接。

**源码核对（`vllm/v1/engine/core_client.py`，vLLM 0.31.0）**：
- 客户端类型：外部 LB（`--data-parallel-external-lb`）用 `DPAsyncMPClient`；**内部 LB 和 hybrid-lb 都用 `DPLBAsyncMPClient`**，由客户端在它管理的 EngineCore 之间做负载均衡。
- 管理范围：`num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size`，源码注释写明 "Client manages local+remote EngineCores in pure internal LB case. Client manages local EngineCores in hybrid and external LB case."：

| | 每个 APIServer 管理的 EngineCore | 数量 |
|---|---|---|
| 内部 LB（`vllm-dp16-ep-two-node.md`） | 全部 16 个（本机 8 + 远端 8） | 16 |
| hybrid-lb（本文） | 仅本节点 8 个（dep-1 管理的是 DP8–15，按 `engine_ranks_managed` 从 `start-rank` 起算） | 8 |

- **选 rank**（`get_core_engine_for_request`）：
  1. 请求里显式带了 `data_parallel_rank` 就直接用它。
  2. 否则对管理范围内每个 EngineCore 算分：`score = max(client_count × 本 APIServer 的 in-flight 数, waiting + running)`；若有 waiting，再加 `waiting × 6 × max(0, KV 使用率 − 0.5)`。取**分最低**的。
  3. 并列时轮转起点，避免固定偏向；选定后本地先把该引擎的 waiting 计数加一，在下一次统计到达之前也能把突发请求摊开。
- **统计来源**：DPCoordinator 发布**全部 16 个引擎**的 waiting / running 计数（最短间隔 100 ms），APIServer 按自己管理的 rank 范围切片使用（`counts[ranks[0]:ranks[-1]+1]`）。所以 hybrid-lb 下 dep-0 的 APIServer 只看 DP0–7 的负载，看不到 DP8–15。
- **协调者的发布地址**：`local_only = not local_engines_only`。内部 LB 下前端与协调者同机，用本机地址；hybrid-lb 下 `local_engines_only` 为真，用 TCP 地址，这与实测一致。

**含义**：
- hybrid-lb 下"请求落到哪个节点"完全由外部（`dep-api` Service）决定，节点内才是 vLLM 按负载选 rank；节点之间不会因负载不均而重新分配。
- 结果回传给哪个 APIServer：按源码结构推断是发起请求的那个（EngineCore 对每个客户端有独立的输出通道），没有逐行核对。
- 每个 EngineCore 在 TCP 上还有来自本机和 dep-1 各 8 条左右的连接，指向它自己的一个监听端口，具体用途没有核对（应与 EngineCore 之间的 DP 组同步有关，推断）。

### 5.6 没有请求的 rank 为什么要陪跑：DP wave 与 dummy batch（源码核对）

**先分清两件事**：MoE 层的 `all_gatherv` / `reduce_scatterv` 是**通信**，是所有 rank 都有请求时的正常流程（第 7 节）；dummy batch 是没有请求的 rank 为了**参与这些通信**而做的**空前向**，是由前者引出的结果，不是同一个东西。空前向跑的是整个模型的一次前向，不只是 MoE 层。

**为什么需要**：16 个 rank 在同一个 EP 组里，每个 MoE 层都有 16 个 rank 参与的集合通信，任何一个 rank 缺席，其他 rank 都会卡住。所以只要有一个 rank 在处理请求，其余 15 个即使没有请求，也要和它每一步同步地跑一次前向，这个空前向就是 dummy batch。

**dummy batch 具体是什么（源码核对）**：它**不是一个假请求**，而是让 GPU 直接跑一次**没有请求对象的假前向**。

- 调用链：`EngineCore.execute_dummy_batch()` → `Worker.execute_dummy_batch()` → `model_runner._dummy_run(num_tokens, uniform_decode=True, randomize_inputs=...)`，其中 `num_tokens` 取 `uniform_decode_query_len`，缺省为 1，即名义上约 **1 个 token**。
- 不经过调度器，没有 Request 对象，不产生输出，也不返回给客户端；输入由 `_dummy_run` 现造：假的 token 输入（`randomize_dummy_inputs` 打开时用随机值填充）、清零的假块表和假的 slot mapping。
- GPU 真的在算：逐层跑完整个模型，包括 MoE 层的 `all_gatherv` / `reduce_scatterv`，这正是它让其他 rank 不卡住的原因，也是实测里没有请求的卡利用率仍升到 90%+ 的原因。
- 它还会参与 Worker 级的 DP 同步（取 `num_tokens_across_dp`，第 5.2 节）：启用 cudagraph 或 ubatch 时，空跑 rank 的 token 数会被补齐到各 rank 的最大值，所以**实际 token 数可能大于名义上的 1**，取决于同一步其他 rank 有多少 token。
- 没有确认的：空前向是否过 LM Head 或采样（在这段源码里没有 grep 到采样相关调用，但没有通读）。

**机制（`vllm/v1/engine/core.py`，`run_busy_loop`；协调者的角色见第 5.7 节）**：
1. **全空闲时**：`engines_running = False`，引擎循环阻塞在输入队列上，不跑前向，GPU 利用率 0%（实测，第 8 节）。
2. **来了第一个请求**：收到请求的 APIServer 在把请求发给选中的 EngineCore 的同时，向 DPCoordinator 发 `FIRST_REQ`（源码 `core_client.py` 的 `add_request_async`；发送方是 APIServer 的客户端，不是 EngineCore），协调者向其余**全部**引擎（共 16 个，已收到请求的那个除外）广播 `START_DP_WAVE`（跨节点，走 TCP），各引擎把 `engines_running` 置为 True，进入同一个 wave。
3. **每一步**：每个引擎先 `_process_engine_step()`；如果本步没有要执行的请求（`not executed`）而 wave 仍在进行（`engines_running` 为真），就 `execute_dummy_batch()`，跑一次空前向，保证 MoE 层的集合通信凑齐 16 个 rank。
4. **何时结束**：每隔 `dp_sync_interval` 步（`step_counter`）做一次 16 个 rank 的 all-reduce（EngineCore 级 Gloo，第 5.2 节），判断是否还有任何 rank 有未完成请求（`_has_global_unfinished_reqs`）。全部完成后 `engines_running = False`，wave 计数加一，所有引擎同时暂停，GPU 回到 0%。
5. **暂停时的协议**：两阶段（`pending_pause`）也靠陪跑实现：先在步进中等所有 rank 都置位，再一起停。

**和实测的对应**：单个 400 token 的请求只由 engine 4 处理，但两个节点 16 张卡都升到 90%+，请求结束后一起回到 0%（第 8 节）。与上面的机制一致：15 个 rank 在陪跑。空前向的 token 数在源码里名义上是 1（可能被补齐，见上），实际耗时没有测，也没有在日志里直接看到 dummy 的记录。

**hybrid-lb 的特点**：协调者只有一个（在 dep-0），dep-1 的 EngineCore 也连到它，所以 wave 是**跨两个节点**的。一个请求落到 dep-0，dep-1 的 8 个 rank 同样陪跑，反之亦然。外部 Service 把请求分流到哪个节点，并不能让另一个节点"休息"。

### 5.7 DPCoordinator：控制面协调者（源码核对）

依据：镜像内 vLLM 0.31.0 的 `vllm/v1/engine/coordinator.py`，以及 `core_client.py`、`core.py` 中与它交互的部分（只读源码，没有在运行时抓包验证消息内容）。

**定位**：只在 DP > 1 时存在，整个部署只有 **1 个进程**（进程名 `DPCoordinator`），运行在 DP master 所在节点（这里是 dep-0）。它夹在前端（APIServer）与后端（EngineCore）之间，属于**控制面，不在数据路径上**：不转发请求，也不处理 token；源码里对引擎消息断言 `outputs.outputs` 为空，只收统计和 wave 通知。MoE 层的 `all_gatherv` / `reduce_scatterv` 是 Worker 之间的 NCCL 通信，与它无关。

**三个 ZMQ socket**（对应第 5.1 节看到的 3 个随机端口）：

| socket | 类型 | 对端 | 用途 |
|---|---|---|---|
| 前端发布 | XPUB | 全部 APIServer（两个节点共 16 个） | 发布 `(各引擎计数, wave 号, engines_running)`；接收前端的 `FIRST_REQ` 通知 |
| 后端输出 | PULL | 全部 EngineCore（16 个） | 接收引擎上报的调度统计、`wave_complete`、`start_wave` |
| 后端发布 | XPUB | 全部 EngineCore | 发 `READY`、`START_DP_WAVE` |

地址类型（源码）：前端侧 `local_only = not local_engines_only`，后端侧 `local_only_eng = (dp_size == dp_size_local)`。hybrid-lb 下 `local_engines_only` 为真、16 ≠ 8，所以三个 socket 都是 TCP，这与 dep-1 的 APIServer 和 EngineCore 能跨节点连到它一致。

**具体职责**：

1. **启动同步**：等全部 16 个引擎订阅后，向它们发 `READY`。
2. **汇总并发布负载统计**：每个引擎上报 `[waiting, running, kv_cache_usage]`，协调者保存各引擎最新值，发布给所有前端，前端据此选 rank（第 5.5 节）。有变化时最短间隔 100 ms 发布一次，没有变化时每 5 秒发布一次；MoE（lockstep）DP 下还会等至少 50 ms，按步边界取快照，以收齐同一步各引擎的统计。
3. **记录 DP wave 与运行状态**：wave 号是"所有 rank 集体从运行态转为暂停态"的次数。全局暂停时，rank 0 的引擎上报 `wave_complete`，协调者把 wave 号加一、`engines_running` 置为假，并把新状态通知前端。
4. **唤醒暂停的引擎**：向引擎广播 `START_DP_WAVE`（带 wave 号和"已收到请求、无需再通知"的引擎序号）。两种触发：
   - 前端在引擎暂停时发新请求：APIServer 的客户端把请求发给选中的 EngineCore，同时向协调者发 `FIRST_REQ`；
   - 引擎在暂停态收到过期 wave 的请求：引擎上报 `start_wave`，协调者据此让其他引擎也进入该 wave。
5. **弹性 EP 扩缩容**：处理 `SCALE_ELASTIC_EP` 通知并调整引擎数。本部署没有用到。

**它不做的事**：
- **不判定 wave 结束**：由引擎自己每隔 `dp_sync_interval` 步做一次 16 个 rank 的 all-reduce 判断，rank 0 上报结果，协调者只记录（第 5.6 节）。
- **不决定请求去哪个 rank**：选 rank 由各 APIServer 的客户端按协调者发布的统计打分完成（第 5.5 节），协调者只提供数据。
- **`engines_running` 只由引擎自己的通知更新**：暂停态的引擎会丢弃 `START_DP_WAVE`，所以协调者发出该消息并不说明引擎已经在跑。

**其他说明**：
- 前端本地也维护一份 `engines_running` 和 `current_wave`，收到协调者的发布后覆盖；发请求时若本地认为引擎暂停，会先置为运行并发 `FIRST_REQ`。
- 外部 LB 模式下引擎不上报统计，只在 wave 或运行状态变化时通知前端（源码 docstring）。
- 协调者只有一个进程、位于 dep-0；它故障会怎样（失去负载统计和唤醒）是由职责推断的，没有验证。

## 6. 模型切分与显存

### 6.1 每卡权重

- 日志：`Model loading took 58.02 GiB`（TEP 是 40.47 GiB）。
- 估算：专家 16 个 × 44.04M 参数 × 58 层 ≈ 38.1 GiB；非专家部分（Attention 11.4 GB、前 3 层稠密 FFN 1.19 GB、共享专家 2.55 GB、embedding 与 lm_head 约 3.7 GB、门控等）合计约 19 GB ≈ 17.7 GiB，DP 下每卡一份完整副本；两者相加约 55.8 GiB，加上其他（MLA 相关等）与 58.02 GiB 在同一量级（拆分是估算，不是实测）。
- 所以 DP+EP 每卡权重比 TEP 多约 17.5 GiB，这是"Attention 不切分"的代价。

### 6.2 KV cache：每个 rank 存自己的请求，互不重复

- 日志：每卡 `Available KV cache memory: 128.39 GiB`，每 rank `GPU KV cache size: 1,961,792 tokens`，对 163,840 token 的单请求最大并发 11.97 倍。
- 每 token KV 为 70,272 字节（见 `strategy1.md` 6.2）：128.39 GiB ÷ 70,272 B ≈ 1.962M tokens，与日志一致。
- **16 个 rank 各存各的请求**，所以全实例总容量是 16 × 1,961,792 = **31,388,672 tokens**，约为 TEP（2,959,680，每卡一份重复拷贝）的 10.6 倍。
- 代价：一个请求的 KV 只能放在它所在的那个 rank 上，单请求最长受一个 rank 的容量限制（1.96M tokens，仍远大于 163,840 的上限）；各 rank 之间的容量不能互相借用。

### 6.3 为什么每卡 KV 比 TEP 少：峰值激活预留大

日志（`gpu_worker.py:955`，两个策略对照）：

| 项 | TEP（策略一） | DEP（本策略） |
|---|---|---|
| 期望使用（0.92 × 267.69 GiB） | 246.27 GiB | 246.27 GiB |
| 权重 + 非 torch 内存 | 43.11 GiB | 58.98 GiB |
| **峰值激活** | **9.46 GiB** | **58.9 GiB** |
| CUDA graph | 3.21 GiB | 6.04 GiB |
| 当前 KV cache | 193.7 GiB | 128.39 GiB |

- DEP 的峰值激活约是 TEP 的 6.2 倍。原因（推断）：MoE 层的 dispatch 把全部 16 个 DP rank 的 token gather 到每张卡上，profiling 按最大批量（16384 token × 16 个 rank）估算激活，需要的临时缓冲远大于 TP16。
- 稳态 `nvidia-smi` 显存约 204,400 MiB ≈ 199.6 GiB / 卡（只有 TEP 的约 246 GiB 的 81%）：为峰值激活预留的 58.9 GiB 是预算出来的，并没有常驻占用（推断：权重 58.0 + KV 128.4 + CUDA graph 6.0 + 其他约 7 ≈ 199.6）。日志还提示可用 `--kv-cache-memory` 把 KV 调到约 141 GiB 来用满显存（没有尝试）。

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

**与策略一（TEP）的对比**（通信形式、每步通信次数、路由方式等）统一放在第 9 节的对比表里，这里不重复。

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

## 9. 与策略一（TEP）的实测对比

策略一是 `--tensor-parallel-size 16 --enable-expert-parallel`：Attention 用 TP16，MoE 用 EP16（TEP）。两者并行结构不同，下表把实测数据和第 7 节的机制对比放在一起。

**入口与进程**

| 项 | 策略一：TEP（TP16 + EP16） | 策略三：DP16 + EP16（DEP） |
|---|---|---|
| 对外入口 | 1 个（节点 0 主进程内的 API server） | 16 个 API server 进程（每节点 8 个），两个节点都服务 |
| EngineCore | 整个实例 1 个 | 每个 DP rank 1 个，共 16 个 |

**并行与通信**

| 项 | 策略一：TEP（TP16 + EP16） | 策略三：DP16 + EP16（DEP） |
|---|---|---|
| Attention 通信 | 每层一次 TP16 all-reduce | **无** |
| MoE 通信 | 每层一次 TP16 all-reduce（权重是 EP 布局，见下） | 每个 MoE 层 all_gatherv + reduce_scatterv（DP 组，16 个 rank） |
| 每步集合通信次数 | 122 次 all-reduce（61 层 × 2） | 116 次（58 个 MoE 层 × 2），前 3 层稠密层无通信 |
| 路由 | 每个 rank 对**整批** token 算门控和 top-k | 每个 rank 只对**自己的** token 算门控，top-k 在 gather 后的全部 token 上做 |
| all2all 后端 | 默认值，未被用到 | 默认值（`allgather_reducescatter`），被用到 |
| 每步 CPU 同步 | 无 | DP 同步走 Gloo（源码；日志未直接确认，第 5.2 节） |
| 需要 `--disable-custom-all-reduce` | 需要（TP16 组） | 不需要 |

**显存与容量**

| 项 | 策略一：TEP（TP16 + EP16） | 策略三：DP16 + EP16（DEP） |
|---|---|---|
| 每卡权重 | 40.47 GiB | 58.02 GiB |
| 峰值激活预留 | 9.46 GiB | 58.9 GiB |
| 每卡 KV cache | 193.7 GiB | 128.39 GiB |
| KV 容量 | 2,959,680 tokens（16 份重复拷贝，全实例共享） | 每 rank 1,961,792 tokens，16 个 rank 合计 31,388,672（各存各的请求） |
| 稳态显存 | 约 246 GiB / 卡 | 约 199.6 GiB / 卡 |

**启动与行为**

| 项 | 策略一：TEP（TP16 + EP16） | 策略三：DP16 + EP16（DEP） |
|---|---|---|
| `init engine` | 912.6 秒 | 1137.96 秒 |
| 空闲 / 单请求 GPU 利用率 | 0% / 16 卡都 90%+ | 0% / 16 卡都 90%+ |

**注意：TEP 的 MoE 权重是 EP 布局，不是 TP 切片**。策略一每卡持有 16 个**完整**的专家（`strategy1.md` 6.3 节），只是通信形式看起来像纯 TP（每层两次 all-reduce、没有 all-to-all）：Attention 用 TP，其 all-reduce 之后每张卡手里已经有**全部 token**，所以各卡自己算路由、只执行落在本卡 16 个专家上的 token，最后一次 all-reduce 汇总，不需要把 token 发到别的卡。纯 TP 则是连 MoE 的每个专家也切成 16 份，与此不同。

**通信数据量**：TEP（TP16 组）每次 all-reduce 的大小是整批 token × hidden；DEP 的 all_gatherv / reduce_scatterv 每个 rank 发送自己的 token、接收全部 token，量级相近但没有量化；本文不做性能分析。

## 10. 结论

- DEP 方案按页面命令加本集群所需的环境变量即可部署，不需要 `--disable-custom-all-reduce`；唯一遇到的问题是我写的部署脚本里的 DNS 竞态（已修）。
- 与策略一 TEP 相比，DEP 用更多的每卡权重（58 vs 40 GiB）和大得多的峰值激活预留（58.9 vs 9.5 GiB）换来了：Attention 无通信、KV 不重复（全实例 KV 容量约 10.6 倍）、两个节点都能对外服务。
- 没有请求的 rank 仍要陪跑：只发 1 个请求，16 张卡的利用率也都升到 90%+；这与 TEP 的现象相同，但原因不同（TEP 是真的在共同计算，DEP 是 15 个 rank 空跑）。
- 性能没有测，本文不做性能结论；`strategy1.md` 第 9 节对 TEP16 与 2×TEP8 的理论分析（该文写作 TP16 + EP16 与 2 × (TP8 + EP8)）也不适用于 DEP。

## 11. 未验证项

- **性能**：按要求没有做，吞吐、TTFT、TPOT 等都没有数据。
- 两个 Pod 之间约 398 条 TCP 连接的用途、`--data-parallel-rpc-port` 在稳态下为什么没有监听者，都没有深挖。DPCoordinator 的职责已读源码（第 5.7 节），但没有在运行时抓包验证消息内容。
- DP 步同步走 Gloo 是源码结论，日志里没有看到 "Using CPU all reduce" 那一行，没有在运行时验证；Worker 级同步的调用频率是由结构推断的；对每步延迟的影响没有测。
- gather 前激活的量化（`defer_input_quant`）在本配置下的取值没有核对，gather 的数据类型不确定；`output_is_reduced()=False` 后 TP 组 all-reduce 是否在 world_size=1 时短路没有核对。
- 峰值激活为什么达到 58.9 GiB（推断与 16 个 rank 的 token gather 有关）、profiling 阶段为什么比 TEP 长约 3 分钟，都没有验证。
- 共享专家和前 3 层稠密 FFN 在 DP 下是每卡完整副本，是根据 TP=1 的推断，没有单独核对；共享专家输出并入结果的位置没有读。
- 专家放置只核对了 rank 0 和 rank 8；Service 分流只测了 24 个来自同一个客户端的请求，更多客户端下的分布没有测。
- 数据面是否走 GPUDirect RDMA、具体 NCCL 传输没有打开 `NCCL_DEBUG` 确认；`packet_seq_err` 的原因没有排查。
- 工具调用（function calling）没有测。

## 12. 运维备注

- 本文测试时（2026-10-11）`dep-0`、`dep-1` 占满节点 `3o1v8b`、`3o1v8e` 的 16 张 GPU，策略一的 `tp16-0`、`tp16-1` 占用另外两个节点。**2026-10-12 查看集群时**：`dep-0`、`dep-1` 仍在运行（Running，0 次重启），另外两个节点上现在运行的是 `tppp-0`、`tppp-1`（对应 `tppp-server.yaml` 与策略二），不再是 `tp16-*`；四个 GPU 节点的卡仍全部被占用。
- 查看：`kubectl get pods`；日志：`kubectl logs -f dep-0`；验证：`kubectl exec dep-0 -- curl -s localhost:8000/v1/models`；经 Service：`kubectl exec dep-0 -- curl -s http://dep-api:8000/v1/models`。
- 停止：`kubectl delete -f dep-server.yaml`（HF 缓存保留）。
- 后续可做：在相同条件下做性能对比（TEP16、DEP、2×TEP8）；用 `--api-server-count`、`--kv-cache-memory` 调优；对比 hybrid-lb 与 internal LB（`--data-parallel-size-local` + headless）两种模式；排查 DP 步同步的 Gloo 路径。


## 附录: 进程

``` bash
root@dep-0:/vllm-workspace# ps -ef
UID          PID    PPID  C STIME TTY          TIME CMD
root           1       0  0 Oct11 ?        00:00:42 /usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --enable-expert-pa
root        1412       1  0 Oct11 ?        00:00:00 /usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(40)
root        1413       1  0 Oct11 ?        00:00:31 VLLM::DPCoordinator
root        1613       1  0 Oct11 ?        00:01:31 VLLM::EngineCore_DP0
root        1614       1  0 Oct11 ?        00:01:30 VLLM::EngineCore_DP1
root        1615       1  0 Oct11 ?        00:01:24 VLLM::EngineCore_DP2
root        1616       1  0 Oct11 ?        00:01:19 VLLM::EngineCore_DP3
root        1617       1  0 Oct11 ?        00:01:26 VLLM::EngineCore_DP4
root        1618       1  0 Oct11 ?        00:01:31 VLLM::EngineCore_DP5
root        1619       1  0 Oct11 ?        00:01:30 VLLM::EngineCore_DP6
root        1620       1  0 Oct11 ?        00:01:20 VLLM::EngineCore_DP7
root        1621       1  0 Oct11 ?        00:00:39 VLLM::APIServer_0
root        1622       1  0 Oct11 ?        00:00:45 VLLM::APIServer_1
root        1623       1  0 Oct11 ?        00:00:40 VLLM::APIServer_2
root        1624       1  0 Oct11 ?        00:00:48 VLLM::APIServer_3
root        1625       1  0 Oct11 ?        00:00:37 VLLM::APIServer_4
root        1626       1  0 Oct11 ?        00:00:39 VLLM::APIServer_5
root        1627       1  0 Oct11 ?        00:00:40 VLLM::APIServer_6
root        1628       1  0 Oct11 ?        00:00:45 VLLM::APIServer_7
root        7051    1619  1 Oct11 ?        00:17:13 VLLM::Worker_DP6_EP6
root        7052    1620  1 Oct11 ?        00:17:03 VLLM::Worker_DP7_EP7
root        7053    1613  1 Oct11 ?        00:17:47 VLLM::Worker_DP0_EP0
root        7145    1618  1 Oct11 ?        00:17:12 VLLM::Worker_DP5_EP5
root        7181    1616  1 Oct11 ?        00:17:10 VLLM::Worker_DP3_EP3
root        7182    1614  1 Oct11 ?        00:17:32 VLLM::Worker_DP1_EP1
root        7183    1615  1 Oct11 ?        00:17:34 VLLM::Worker_DP2_EP2
root        7213    1617  1 Oct11 ?        00:17:17 VLLM::Worker_DP4_EP4
root       25476       0  0 00:07 pts/0    00:00:00 /bin/bash
root       25743       0  0 17:22 pts/1    00:00:00 /bin/bash
root       26015   25743  0 18:43 pts/1    00:00:00 ps -ef

root@dep-1:/vllm-workspace# ps -ef
UID          PID    PPID  C STIME TTY          TIME CMD
root           1       0  0 Oct11 ?        00:00:42 /usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --enable-expert-pa
root        1418       1  0 Oct11 ?        00:00:00 /usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(45)
root        1419       1  0 Oct11 ?        00:01:33 VLLM::EngineCore_DP8
root        1420       1  0 Oct11 ?        00:01:22 VLLM::EngineCore_DP9
root        1421       1  0 Oct11 ?        00:01:28 VLLM::EngineCore_DP10
root        1422       1  0 Oct11 ?        00:01:28 VLLM::EngineCore_DP11
root        1423       1  0 Oct11 ?        00:01:32 VLLM::EngineCore_DP12
root        1424       1  0 Oct11 ?        00:01:27 VLLM::EngineCore_DP13
root        1425       1  0 Oct11 ?        00:01:26 VLLM::EngineCore_DP14
root        1426       1  0 Oct11 ?        00:01:23 VLLM::EngineCore_DP15
root        1427       1  0 Oct11 ?        00:00:37 VLLM::APIServer_0
root        1428       1  0 Oct11 ?        00:00:45 VLLM::APIServer_1
root        1429       1  0 Oct11 ?        00:00:38 VLLM::APIServer_2
root        1430       1  0 Oct11 ?        00:00:47 VLLM::APIServer_3
root        1431       1  0 Oct11 ?        00:00:43 VLLM::APIServer_4
root        1432       1  0 Oct11 ?        00:00:43 VLLM::APIServer_5
root        1433       1  0 Oct11 ?        00:00:43 VLLM::APIServer_6
root        1434       1  0 Oct11 ?        00:00:46 VLLM::APIServer_7
root        6789    1426  1 Oct11 ?        00:17:11 VLLM::Worker_DP15_EP15
root        7007    1422  1 Oct11 ?        00:17:26 VLLM::Worker_DP11_EP11
root        7008    1421  1 Oct11 ?        00:17:32 VLLM::Worker_DP10_EP10
root        7009    1419  1 Oct11 ?        00:17:34 VLLM::Worker_DP8_EP8
root        7010    1424  1 Oct11 ?        00:17:11 VLLM::Worker_DP13_EP13
root        7011    1425  1 Oct11 ?        00:17:12 VLLM::Worker_DP14_EP14
root        7012    1420  1 Oct11 ?        00:17:37 VLLM::Worker_DP9_EP9
root        7013    1423  1 Oct11 ?        00:17:07 VLLM::Worker_DP12_EP12
root       24991       0  0 00:07 pts/0    00:00:00 /bin/bash
root       25030       0  0 17:22 pts/1    00:00:00 /bin/bash
root       25039   25030  0 18:43 pts/1    00:00:00 ps -ef
```