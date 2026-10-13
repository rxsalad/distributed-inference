<a id="top"></a>
# 1P1D：Prefill 与 Decode 都用 DEP（DP Attention + EP），NVIDIA B300x8 上的 PD 分离

DeepSeek-V3（FP8）在 B300x8 集群 (Rail-Optimized) 上做 Prefill/Decode 分离：**1 个 Prefill 节点 + 1 个 Decode 节点**，各是一个 8 GPU 的 DEP 组（DP8 + EP8）；KV cache 用 NIXL 从 Prefill 传到 Decode；前面由一个 CPU Pod 上的 vllm-router 接入。


本文分三部分：**原理**（第 2 章）、**部署与配置**（第 3 章）、**观测与实测**（第 4、5 章）。第 6、7 章汇总推断的核对结果和尚未验证的内容。

- 命令来源：[recipes.vllm.ai DeepSeek-V3，Prefill/Decode Disaggregation，prefill_mode=dep、decode_mode=dep](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V3?hardware=b300&prefill_mode=dep&prefill_nodes=2&decode_mode=dep&decode_nodes=2)。
- 标注约定：**页面**＝recipes 页面原文；**观察**＝在本集群上看到过的现象；**源码核对**＝读了镜像内 vLLM 0.31.0 源码；**推断**＝由机制得出，未验证。第 5 章的所有数据都是 2026-10-13 在运行中的 1P1D 上采集的。

<a id="toc"></a>
## 目录

- [1. 摘要](#s1)
- [2. 原理](#s2)
  - [2.1 总体结构与请求路径](#s2-1)
  - [2.2 DEP：8 个 DP rank × 8 个 EP rank](#s2-2)
  - [2.3 一个 MoE 层怎么算；为什么要 dummy batch；同步机制](#s2-3)
  - [2.4 Prefill 与 Decode 在 DEP 下的差别；与 TEP 对比](#s2-4)
  - [2.5 KV 传输原理（NIXL + UCX）](#s2-5)
- [3. 部署与配置](#s3)
  - [3.1 文件](#s3-1)
  - [3.2 前提](#s3-2)
  - [3.3 使用与开关](#s3-3)
  - [3.4 Prefill / Decode 池](#s3-4)
  - [3.5 环境变量与理由](#s3-5)
  - [3.6 Router](#s3-6)
  - [3.7 运行环境记录](#s3-7)
- [4. 观测方法](#s4)
  - [4.1 `./pdd.sh metrics [窗口秒数]`](#s4-1)
  - [4.2 日志的局限](#s4-2)
  - [4.3 网卡、内存区和 GPU](#s4-3)
  - [4.4 进程与线程](#s4-4)
- [5. 实测](#s5)
  - [5.1 进程结构](#s5-1)
  - [5.2 空闲时：GPU 为 0%，但 Worker 占满一个 CPU 核](#s5-2)
  - [5.3 单请求：GPU 全体参与、没有唤醒延迟](#s5-3)
  - [5.4 KV 传输：131072 token 的大 KV（P → D）](#s5-4)
  - [5.5 拓扑与模型配置](#s5-5)
  - [5.6 EP 的集合通信走 NVLink](#s5-6)
  - [5.7 并发下的 KV 传输](#s5-7)
- [6. 推断核对一览](#s6)
- [7. 未验证与注意](#s7)
- [8. 附录：原始输出与统计脚本](#s8)
  - [A.1 `pdd-prefill-0`](#a1)
  - [A.2 `pdd-decode-0`](#a2)
  - [A.3 PID 1 的完整命令行（`/proc/1/cmdline`）](#a3)
  - [A.4 NVLink / RDMA 流量统计脚本](#a4)

<a id="s1"></a>
## 1. 摘要

[↑ 返回顶部](#top)

**这是什么**：Prefill 池和 Decode 池各自是一个 8 卡的 DEP 组（每张 GPU 同时是一个 DP rank 和一个 EP rank），两个池之间只有 NIXL 点对点的 KV 通道。

**关键结论（均为 1P1D 上的实测，细节见第 5 章）**

| 主题 | 结论 |
|---|---|
| 一个请求的 GPU 行为 | 只发 1 个请求，所在池的 8 张 GPU 都升到 99%～100%（没有请求的 rank 陪跑 dummy batch）；空闲时全是 0% |
| EP 集合通信走哪里 | **NVLink**：空闲时 NVLink 流量为 0，一个请求下 Prefill 的 8 张 GPU 之间约 278 GiB（32768 token）；RoCE 上没有集合通信流量（5.6） |
| KV 传输方式 | **RDMA READ，由 Decode 发起**，从 Prefill 的显存直接拉取；Prefill 的 CPU/GPU 不参与搬运 |
| KV 传输用哪块网卡 | 每次传输两侧各只用**一块**网卡，且是该 rank 自己 GPU 最近的那块（rank r ↔ `mlx5_{2r}`） |
| 是否跨轨道 | Router 给出的 Prefill/Decode 配对在一次启动内固定为 8 种（不是 8×8 = 64 种组合），配对随 Router 启动而变：最初那次启动里序号从不相同、全部跨轨道，后来重新部署的这次有 1 对同轨（P6→D6）。跨轨和同轨的传输速度没有看到差别（约 48 GB/s，5.7） |
| 传输性能 | 9.21 GB（131072 token）的 KV 约 0.19 s，约 46～48 GB/s，线上峰值约 50 GB/s，接近单卡 400 Gb/s 线速 |
| 传输在 TTFT 中的占比 | 约 0.5%（TTFT 约 36 s，其中 Prefill 计算 34～35 s） |
| 并发时的 KV 传输 | 24 个 32768 token 请求、并发度 8：NIXL 记录的传输耗时从单请求的约 49 ms 升到 57～65 ms（+16%～34%），线上速度不变（突发中位数 60 ms、约 39 GB/s）；**1P1D 的 RoCE 上没有集合通信，仍然变长，所以变长不是集合通信造成的**（5.7） |
| 并发吞吐 | 同样的 24 个请求，1P1D 28.6 s（0.84 req/s，TTFT 平均 6.85 s，TPOT 17.4 ms；5.7） |
| 内存区 | 每个 Worker 在自己的网卡上注册 **2 个**大小相同的大内存区，等于整个 KV cache；一次传输由若干 descriptor（一层里一段连续的块）组成，数量从 61 到 249,856 不等 |
| 空闲时 CPU | 每个 Worker 空闲时占满一个核，**不是** dummy batch，更可能是 NIXL 的 UCX progress 线程在忙轮询 |
| 首请求延迟 | 空闲约 28 分钟后的第一个请求没有额外延迟 |

<a id="s2"></a>
## 2. 原理

[↑ 返回顶部](#top)

<a id="s2-1"></a>
### 2.1 总体结构与请求路径

```
                    客户端
                      │  :30000
               ┌──────▼──────┐
               │ pdd-router  │  CPU Pod，vllm-router（PD 模式，round_robin）
               └──┬───────┬──┘
        请求 / 首 token │       │ 请求 / 解码
        ┌───────────┘       └────────────┐
┌───────▼──────────────┐        ┌────────▼─────────────┐
│ pdd-prefill-0        │  NIXL  │ pdd-decode-0         │
│ 1 个 GPU 节点，8 GPU   │ ◄───── │ 1 个 GPU 节点，8 GPU   │
│ kv_producer，:8001    │  READ  │ kv_consumer，:8002    │
│ DP8 + EP8            │        │ DP8 + EP8            │
└──────────────────────┘        └──────────────────────┘
```

- 共用 **2 个 GPU 节点 + 1 个 CPU 节点**。
- 一个请求的路径：客户端 → Router → Prefill（算 prompt、生成首 token）→ Decode 通过 NIXL 从 Prefill **拉取**这个请求的 KV → Decode 继续生成。
- Prefill 与 Decode 是**两个独立的 DP 组**，各有自己的 DP 协调者（`DPCoordinator`，在 Pod 内），互不相连。
- 每个池是 hybrid-lb：Pod 内有 8 个 API server，在 8 个本地 DP rank 间分流。Router 把每个 Pod 当作一个后端，用 `--intra-node-data-parallel-size 8` 告诉它每个后端有 8 个 DP rank，并**以 DP rank 为单位**选后端（日志里形如 `prefill=http://<ip>:8001@N`）。

<a id="s2-2"></a>
### 2.2 DEP：8 个 DP rank × 8 个 EP rank

`--data-parallel-size 8 --enable-expert-parallel`，TP 取默认值 1：8 张 GPU，每张同时承担两个角色。

| | Attention 部分（DP） | MoE 部分（EP） |
|---|---|---|
| 单位 | 8 个 DP rank，一张 GPU 一个 | 8 个 EP rank，一张 GPU 一个，8 个 rank 构成**一个 EP 组** |
| 权重 | **每张卡一份完整的 Attention 权重**（MLA、前 3 层稠密 FFN、门控、共享专家等，TP=1 不切分；其中稠密 FFN 和共享专家是推断） | **256 个路由专家分到 8 张卡，每卡 32 个**，不重复 |
| 处理什么 | 各 rank **独立**处理分配给自己的请求，有自己的 EngineCore（调度器）和 KV cache | 每个 rank 处理**所有 DP rank** 发来的、选中自己那 32 个专家的 token |
| rank 之间通信 | **没有** | 有：每个 MoE 层都要集合通信（2.3） |
| 进程 | `EngineCore_DPn`（每 rank 一个）及其 `Worker_DPn_EPn` | 同一个 `Worker_DPn_EPn`（DP、EP 序号相同） |

要点：
- **「完整权重」只指 Attention 一侧**：整个模型的权重不是每卡一份，专家按 EP 切开，每卡只有 1/8。
- **KV cache 不重复**：每个 rank 只存自己请求的 KV，8 个 rank 的容量加起来是总容量；TP 方案则是每卡一份完整 KV。
- 每张卡的显存 = 完整 Attention 权重 + 32 个专家 + 自己的 KV + 激活预留。
- **EP 组完全在节点内，集合通信走 NVLink**（已验证）：8 张 GPU 两两 `NV18`（NVLink 全互联）；NVLink 计数在空闲时为 0、有请求时大量增加；一个请求期间 RoCE 网卡上没有集合通信流量，只有 NIXL 传 KV 的 READ（第 5.3、5.4、5.6 节）。RoCE 在 1P1D 里只用于 KV 传输。

<a id="s2-3"></a>
### 2.3 一个 MoE 层怎么算；为什么要 dummy batch；同步机制

DeepSeek-V3 共 61 层，前 3 层稠密 FFN，后 58 层是 MoE（`config.json` 核对）。一个 MoE 层的流程（all2all 后端是默认的 `allgather_reducescatter`，源码核对）：

```
每个 DP rank 上：  Attention（独立，无通信）
                   → 路由门控：本 rank 的 token 各选 8 个专家（共 256 个）
                   → all_gatherv：8 个 rank 的 token 汇总，每个 rank 都拿到全部 token（dispatch）
                   → 每个 rank 只算自己那 32 个专家对「全部 token」的贡献
                   → reduce_scatterv：各 rank 的结果求和，每个 rank 只取回自己那批 token 的输出（combine）
                   → 共享专家（本地算）、下一层
```

- 通信是 **NCCL 集合通信**，每个 MoE 层一次 `all_gatherv` + 一次 `reduce_scatterv`，一个前向步里 58 层各做一遍。Attention 和稠密层没有 rank 间通信，所以 DEP 的通信开销只出现在 MoE 层。
- 这个后端把**所有 token 广播给所有 rank**，通信量随 rank 数增长，不像按路由只发给目标专家的 all2all 那么省；B300（sm_103）上 DeepEP 不可用，所以只能用默认后端。

**为什么没有请求的 rank 也要「陪跑」（dummy batch）**

EP 把每个 MoE 层拆到所有 rank 上：某个 rank 上的专家要替**别的 rank 的 token** 做计算，每个 MoE 层都是全组参与的集合操作，组内所有 rank 必须执行**相同次数的前向步**。而 DP 让每个 rank 的请求流彼此独立，没有请求的 rank 本来不需要前向。所以只要组里有一个 rank 在处理请求，其余 rank 就要跑 **dummy batch**（一次只有 1 个 token 的空前向，但要走完所有层，包括 58 层的集合通信；源码里有 `execute_dummy_batch`）。

- **范围**：陪跑的原因是 MoE 的跨 rank 依赖，范围是整个 EP 组：1P1D 里 Prefill 池有一个请求，Prefill 的 8 张卡都忙；Decode 池同理（已验证，第 5.3 节）。
- **与后端无关、稠密模型没有陪跑**：这两点按机制推断，没有验证。
- **对观测的影响**：`nvidia-smi` 的 GPU 利用率会失真，一个请求就能让 8 张卡都在 99% 以上，不能用它判断有效负载。
- **注意区分**：请求期间 8 卡陪跑（GPU 利用率）已验证；而空闲时 Worker 占满一个 CPU 核**不是** dummy batch 在空转（空闲时 GPU 是 0%，第 5.2 节）。

**同步机制**（源码核对，vLLM 0.31.0）

| 层次 | 频率 | 做什么 | 通道 |
|---|---|---|---|
| MoE 层集合通信 | 每步 × 58 层 × 2（gather + scatter） | 交换 token，各 rank 算自己的专家 | NCCL |
| 每个前向步的 DP 元数据同步 | 每步 1 次 | 小的 int32 all-reduce：各 rank 的 token 数、CUDA graph 模式、是否做 DP padding；开了 CUDA graph 时所有 rank 的 token 数 pad 到同一个最大值 | 异步调度下走 **CPU 上的 Gloo**（经 eth0 的 TCP）。日志有 `Disabling NCCL for DP synchronization when using async scheduling.`（已验证）；走 Gloo 是源码结论 |
| wave 的启停共识 | 每 `dp_sync_interval` 步，默认 16（`config/parallel.py` 核对） | 全局 all-reduce，判断是否还有 rank 有未完成请求 | 同上 |

`DPCoordinator`（每个池一个）负责 wave 的协调和负载统计：全空闲后的第一个请求先通知它，它广播启动，各引擎才开始步进。这也是 Prefill 池和 Decode 池互不影响的原因：各有协调者、各自的 wave。

**wave 是什么**（源码 `v1/engine/coordinator.py` 的说明，vLLM 0.31.0）

- 一个池里的所有 DP 引擎同进同退，在全局的 **running（步进）** 与 **paused（暂停）** 两个状态间交替。一次 running → paused 的整体转换算完成一个 **wave**，wave 编号就是这种转换发生的次数（从 0 开始计数）。
- **启动**：引擎暂停时来了新请求，前端会同时通知 `DPCoordinator`，它向池内所有引擎广播 `START_DP_WAVE`，全组一起进入 running；某个引擎收到的请求带着过期的 wave 编号时，也会通知它。
- **结束**：running 期间每隔 `dp_sync_interval` 步做一次全局 all-reduce，所有 rank 都没有未完成请求时，全组一起回到 paused，wave 编号加 1。
- **和 dummy batch 的关系**：dummy batch 只在 running 状态出现，wave 内只要有一个 rank 有请求，其余 rank 就陪跑；paused 时所有 rank 都不步进（空闲 GPU 为 0%，第 5.2 节）。
- **没有验证的部分**：以上来自源码说明，没有在运行中观察 wave 编号的变化；`dp_sync_interval` 对延迟的影响（最多多跑若干个空步）也只是推断。

<a id="s2-4"></a>
### 2.4 Prefill 与 Decode 在 DEP 下的差别；与 TEP 对比

| | Prefill 池（`kv_producer`） | Decode 池（`kv_consumer`） |
|---|---|---|
| 每个 DP rank 做什么 | 对分到的请求做 prompt 的前向，生成首 token，把 KV 留给 NIXL 被拉走 | 拉取 KV，逐 token 生成 |
| 单步的特点 | token 多（长 prompt），算力受限；一个 rank 上的长 prefill 让其他 7 个 rank 在每个 MoE 层等它 | 每步 token 少，每个 MoE 层集合通信占比高，对每步延迟敏感；用 `FULL_DECODE_ONLY` CUDA graph |
| 负载不均的影响 | 最慢的 rank 决定整组步调（推断，未测） | 同左 |

**与 TEP 对比**

- **DEP**（Data-parallel attention + Expert Parallel，本方案）：Attention 按 DP 切，每个 rank 处理各自的请求；MoE 按 EP 切，专家分布在所有 rank 上。
- **TEP**（Tensor-parallel attention + Expert Parallel）：Attention 按 TP 切，一个请求由全部 8 张卡共同计算；MoE 同样按 EP 切。两者的 MoE 部分相同，差别在 Attention 和请求的组织方式。

| | DEP（本方案） | TEP |
|---|---|---|
| Attention | 8 个 rank 各算各的请求，无通信 | 8 张卡共同算一个请求，每层 all-reduce |
| KV cache | 每个 rank 只存自己请求的 KV，不重复 | MLA 模型只有一个 latent 头，TP 切不开，每卡一份完整拷贝 |
| MoE | EP8，每层 all-gather + reduce-scatter | EP，TP 组内另有 all-reduce |
| 空闲 rank | 要陪跑 dummy batch | 只有一个 EngineCore，没有「陪跑」 |
| 适合 | 高并发、追求吞吐 | 低并发、追求单请求延迟 |

纯 TP（MoE 的专家权重也按 TP 切）与 TEP 的区别只在 MoE，这里不展开。

**验证状态**：这个对比表来自机制与源码推断，本环境只部署了 DEP，**没有跑过 TEP/TP，没有实测对比**；「适合」一行是通行的经验判断，本文没有数据支持。

<a id="s2-5"></a>
### 2.5 KV 传输原理（NIXL + UCX）

- **拉取式（pull）**：Prefill 算完后，Decode 的 Worker 通过 NIXL 发起 `READ`（源码 `kv_connector/v1/nixl/pull_worker.py` 里 `make_prepped_xfer("READ", ...)`），从 Prefill 的显存直接读，完成后用通知消息告知对端。底层是 RDMA READ，Prefill 的 CPU 和 GPU 不参与搬运（已验证，第 5.4 节）。
- **rank 对 rank**：一个请求落在 Prefill 的某个 rank 和 Decode 的某个 rank，KV 在传输之前一直留在 Prefill 的那张卡上，两个 rank 序号不必相同。
- **内存注册**：每个 Worker 把自己整个 KV cache（一整块连续的显存）注册为 RDMA 内存区，只注册在**自己那块网卡**上，日志是 `Registering KV_Caches. use_mla: True, kv_buffer_device: cuda, use_host_buffer: False`。所以传输自然落在这块网卡上。
- **侧信道端口（握手）**：Decode 第一次要从某个 Prefill 引擎拉 KV 时，要先拿到对方的 NIXL agent 元数据（内存区注册信息、UCX 地址等），这一步走**侧信道**，与 KV 数据路径分开。每个引擎（每个 DP rank）起一个 ZMQ ROUTER 监听线程，绑定 `VLLM_NIXL_SIDE_CHANNEL_HOST:PORT`；Decode 用 ZMQ REQ 连过去，发 `GET_META_MSG`（带目标 pp rank、tp rank），对方回复元数据，Decode 据此调用 `add_remote_agent`，之后才能发 RDMA READ。回复里还附带对方的 `perf_counter`，用来估算两端时钟偏移（KV 块过期时间要用）。握手只在首次连接时发生，KV 数据走 UCX/RoCE，不再经过这个端口（源码 `kv_connector/v1/nixl/base_scheduler.py`、`base_worker.py`，vLLM 0.31.0）。
  - **端口 = 基础端口 + `data_parallel_index`**：每个 DP rank 一个端口。基础端口 Prefill 为 5557，Decode 为 5558。已验证：`pdd-prefill-0` 里 `ss -ltn` 在 Pod IP 上监听 5557～5564 共 8 个端口；Decode 池的端口范围按源码推算为 5558～5565，未逐一查看。
  - **地址必须是别的 Pod 能到达的本 Pod IP**（本部署用 `POD_IP`）；填 `localhost` 会让对端连回它自己。
- **descriptor**：一次传输由若干 descriptor 组成，一个 descriptor 是「一层里一段连续的块」；两侧块分配越连续，descriptor 越少。块大小 32 token，每块每层 36,864 B（32 × 576 × 2 B）。
- **每 token 的 KV**：MLA，(512 + 64) × 2 B × 61 层 = 70,272 B ≈ 70.3 KB（`kv_lora_rank=512`、`qk_rope_head_dim=64`）；32768 token ≈ 2.30 GB，131072 token ≈ 9.21 GB。可以用它核对 `metrics` 里的传输字节数。

<a id="s3"></a>
## 3. 部署与配置

[↑ 返回顶部](#top)

<a id="s3-1"></a>
### 3.1 文件

| 文件 | 作用 |
|---|---|
| `pdd-dep.yaml` | envsubst 模板：Prefill/Decode 各一个 StatefulSet + headless Service；Router 的 ServiceAccount/Role/RoleBinding、Service、StatefulSet |
| `pdd-launch.sh` | Prefill/Decode 共用的启动脚本，经 ConfigMap `pdd-scripts` 挂载到 `/scripts` |
| `pdd-router.py` | Router 的启动器，同样经 ConfigMap 挂载 |
| `pdd.sh` | 控制脚本：`up` / `down` / `status` / `render` / `test` / `bench` / `metrics` |
| `pdd-dap-ep-nvidia(1p1d).md` | 本文 |

除本文外所有文件都用英文。脚本放在 ConfigMap 里：改了脚本后重新 `up`，已运行的 Pod 要重启才会用新脚本。

<a id="s3-2"></a>
### 3.2 前提

- 至少 2 个 B300×8 的 GPU 节点（`gpu-brand=nvidia`，带 `nvidia.com/gpu` taint）、multus + `roce-net-fabricN` 网络附件、`rdma/fabricN` 设备插件资源；vLLM 镜像 0.31.0（`latest` 解析的版本）；至少 1 个非 GPU 节点给 Router。
- 模型 `deepseek-ai/DeepSeek-V3`，每个节点首次启动会下载到本节点的 hostPath 缓存。
- `pdd.sh up` 会检查 `PREFILL_NODES + DECODE_NODES` 不超过集群的 GPU 节点数。

<a id="s3-3"></a>
### 3.3 使用与开关

```bash
PREFILL_NODES=1 DECODE_NODES=1 ./pdd.sh up      # 部署 1P1D（不写这两个变量时默认是 2、2）
./pdd.sh status                                  # Pod 和 Service
./pdd.sh test                                    # 经 Router 发一个请求验证端到端
./pdd.sh bench [额外的 vllm bench serve 参数]    # 经 Router 压测，输出同时写入 benchmark-<时间>.log
./pdd.sh metrics [窗口秒数]                      # 汇总 P/D 的 /metrics（第 4.1 节）
./pdd.sh render                                  # 只渲染 yaml
./pdd.sh down                                    # 删除（HF 缓存保留）
```

Pod 名固定：`pdd-prefill-0`、`pdd-decode-0`、`pdd-router-0`。可选环境变量（调整后需要 `down` 再 `up`）：

| 变量 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `PREFILL_NODES`、`DECODE_NODES` | 正整数 | 2、2 | 1P1D 要显式设为 1、1 |
| `IMAGE` | 镜像 | `vllm/vllm-openai:latest` | 三个 workload 共用；`latest` 在不同时间拉取可能拿到不同版本，要一致请用固定 tag |
| `MODEL` | 模型名 | `deepseek-ai/DeepSeek-V3` | |
| `PREFIX_CACHING` | `on` / `off` | `off` | `off` 时给 P、D 加 `--no-enable-prefix-caching`。vLLM 0.31.0 默认开启 prefix caching，测 PD 分离时命中缓存会让 TTFT 和 KV 传输量失真；P、D 必须一致 |
| `HIDE_HEALTH_LOG` | `on` / `off` | `off` | `on` 时给 P、D 加 `--disable-access-log-for-endpoints /health,/metrics,/ping`，保留 `POST /v1/...` 日志 |
| `UCX_LOG_LEVEL` | `fatal\|error\|warn\|info\|debug\|trace` | `info` | 只作用于 P、D；`info` 能看到 UCX 选择的传输、设备和 rail，嫌日志多设 `warn` |

`bench` 默认：32768 输入 / 128 输出 / 1 个请求 / request-rate 1 / seed 1，可用 `BENCH_INPUT_LEN`、`BENCH_OUTPUT_LEN`、`BENCH_NUM_PROMPTS`、`BENCH_REQUEST_RATE`、`BENCH_SEED`、`BENCH_LOG` 覆盖，`bench` 之后的参数原样追加给 `vllm bench serve`。

**测试前必须预热**：每个 DP rank 第一次处理长 prefill 时有一次性的额外开销，第一次的数据没有意义，不纳入分析。做任何性能测试之前，先经 Router 发一批长请求（输入不短于正式测试），让全部 8 个 Prefill rank 和 8 个 Decode rank 都处理过，例如 `BENCH_NUM_PROMPTS=16 BENCH_REQUEST_RATE=inf ./pdd.sh bench --max-concurrency 8`，并用 `kubectl logs pdd-router-0 | grep "Selected prefill"` 确认每个 Prefill rank（`@0`..`@7`）都出现过；Router 的 P→D 配对是固定的（第 5.4 节），覆盖了全部 Prefill rank 也就覆盖了全部 Decode rank。预热之前采集的耗时数据不要用，NVLink、网卡的字节数不受影响。本文第 5 章的耗时数据采集时，所涉及的 rank 之前都处理过请求（Router 日志可查），但 1P1D 上没有逐 rank 做过冷热对比。

**测试都在 `pdd-router-0` 里跑**：`test`、`bench`、`metrics` 都是 `kubectl exec pdd-router-0`，不进 Prefill/Decode Pod，避免测试客户端占用 GPU 节点的 CPU、干扰 P/D。Router Pod 与 P/D 用同一个镜像，里面有 `vllm bench`；tokenizer 第一次会从 HF 下载，缓存在 CPU 节点的 hostPath。读硬件计数器（网卡、`nvidia-smi`）必须进 P/D Pod，保持只读、轻量。

<a id="s3-4"></a>
### 3.4 Prefill / Decode 池

**启动（`pdd-launch.sh`）**：两个角色共用同一个脚本，差别只由环境变量决定。

| | Prefill | Decode |
|---|---|---|
| StatefulSet / headless Service | `pdd-prefill` / `pdd-prefill-headless` | `pdd-decode` / `pdd-decode-headless` |
| `KV_ROLE` | `kv_producer` | `kv_consumer` |
| API 端口 | 8001 | 8002 |
| NIXL 侧信道端口 | 5557 | 5558 |
| 额外参数 | — | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`（页面：Decode 池额外使用仅 decode 的 CUDA graph） |

1P1D 时 `NODES=1`，所以 `DP_SIZE=8`、`START_RANK=0`，Pod 自己就是 Head（`-0` 号）。步骤：

1. 解析 Head 地址 `<ROLE_APP>-0.<ROLE_APP>-headless`，循环到解析出非空 IP 为止（DNS 有瞬时竞态：`getent` 可能先成功、紧接着取到的 IP 却为空）。headless Service 设了 `publishNotReadyAddresses: true`，Pod 还没 Ready 时域名也能解析。
2. `hf download ${MODEL}`：hostPath 缓存按节点各自一份，已有缓存时立即返回。
3. 设 NIXL 侧信道：`VLLM_NIXL_SIDE_CHANNEL_HOST=${POD_IP}`（握手用，必须是别的 Pod 能到达的本 Pod IP）、`VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_PORT}`。
4. 按开关追加参数：`PREFIX_CACHING=off` → `--no-enable-prefix-caching`；`HIDE_HEALTH_LOG=on` → `--disable-access-log-for-endpoints ...`；非法取值直接报错退出。
5. `exec vllm serve`：

```
vllm serve ${MODEL} --trust-remote-code --host 0.0.0.0 --port ${HTTP_PORT}
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"<kv_producer|kv_consumer>","kv_load_failure_policy":"fail"}'
  --data-parallel-size 8 --data-parallel-size-local 8 --data-parallel-start-rank 0
  --data-parallel-address ${HEAD_IP} --data-parallel-rpc-port 13345 --data-parallel-hybrid-lb
  --enable-expert-parallel
  --enable-auto-tool-choice --tool-call-parser deepseek_v3 --reasoning-parser deepseek_v3
```

**Pod 规格**
- 调度到带 `doks.digitalocean.com/gpu-brand: nvidia` 的节点，容忍 `nvidia.com/gpu` 的 `NoSchedule` taint；`podManagementPolicy: Parallel`。
- 每个 Pod 请求 8 个 `nvidia.com/gpu` 和 8 个 RoCE 设备资源 `rdma/fabric{0,2,…,14}`（每个 10），并通过 multus 注解挂上 `roce-net-fabric{0,2,…,14}`。
- `IPC_LOCK` 能力；`/dev/shm` 用 128Gi 的内存卷；hostPath `/root/.cache/huggingface` 作为模型缓存。
- 探针：startupProbe 周期 60 秒、失败阈值 60（最多等约 1 小时，启动本身要 15～25 分钟）；readinessProbe 周期 60 秒、超时 5 秒、失败阈值 3。周期取 60 秒（常见写法是 10 秒）是为了减少 vLLM 里的 `GET /health` 访问日志；代价是 Pod 变 Ready 最多晚 1 分钟，故障最多 3 分钟才被发现。

<a id="s3-5"></a>
### 3.5 环境变量与理由

| 变量 | 值 | 理由 / 状态 |
|---|---|---|
| `GLOO_SOCKET_IFNAME`、`NCCL_SOCKET_IFNAME` | `eth0` | 控制面/引导走 Pod 网卡 |
| `NCCL_IB_GID_INDEX` | `3` | **观察**（此前在本集群的多节点 NCCL 部署中）：否则 NCCL 选到链路本地 RoCE GID，跨节点静默卡死 |
| `UCX_IB_GID_INDEX` | `3` | 与 `NCCL_IB_GID_INDEX` 同理，让 UCX 选对 RoCE GID。**未验证**这个变量本身的作用：UCX 能建立 RDMA 后端、传输正常，但没有做去掉它的对比 |
| `UCX_NET_DEVICES` | `mlx5_0:1,mlx5_2:1,…,mlx5_14:1` | **观察**：`fabricN` 网卡挂在 VRF 下，UCX 默认会枚举 `tcp/fabricN` 并 bind 失败（`bind ... Cannot assign requested address` → `NIXL_ERR_BACKEND`）。只列 8 个 RoCE 设备（GPU i 对应 `mlx5_{2i}`，与它同属一个 PCIe 桥），**不留 `eth0` 的 TCP 回退**，否则 RDMA 不可用时 KV 传输会悄悄走 TCP |
| `UCX_MAX_RNDV_RAILS` | `1` | 每次 rendezvous 传输最多用 1 条 rail（UCX 默认 2，最大 4）；每个 worker 用自己的 NIC，8 个 worker 合起来覆盖 8 块网卡。**观察**：每次传输每侧只用 1 块网卡（第 5.4 节），与此一致，但没有对比去掉它的情况 |
| `UCX_IB_TRAFFIC_CLASS` | `104` | RoCE 流量的 DSCP / traffic class，沿用 AMD GPU 上 NIXL PD 部署的取值；交换机是否按它分类、有没有生效，在本环境里无法验证 |
| `UCX_LOG_LEVEL` | 见 3.3 | 默认 `info`，用于核对 KV 传输走哪条路径 |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `3600` | 初始化慢，放宽引擎就绪超时 |

**与页面命令的差异**：节点数 1P+1D（DP8+EP8）；增加 `NCCL_IB_GID_INDEX`、`VLLM_ENGINE_READY_TIMEOUT_S` 和上表的 `UCX_*`；去掉 `--chat-template`（镜像里的相对路径未验证）；NIXL 侧信道地址用 Pod IP（downward API），不用页面里的 `$PREFILL_NODE_n`；Router 不用页面里写死的 URL，而是由 `pdd-router.py` 从 K8S 发现 Pod；增加 `PREFIX_CACHING`（默认 `off`，与 vLLM 0.31.0 的默认行为相反）、`HIDE_HEALTH_LOG` 两个开关。

<a id="s3-6"></a>
### 3.6 Router

CPU Pod（StatefulSet `pdd-router`，1 副本，Pod 名 `pdd-router-0`），用 nodeAffinity 调度到没有 `doks.digitalocean.com/gpu-brand` 标签的非 GPU 节点（不写死 CPU 池名，因为池名在不同集群/时间可能不同）。Service `pdd-router:30000` 是对外入口。

- **镜像**与 P/D 相同（`${IMAGE}`），里面有 `python3`（**没有 `python`**）、curl 和 vllm，所以 `vllm bench` 可以在 Router Pod 里跑。启动命令：`pip install --no-cache-dir vllm-router kubernetes requests`（只新增包，不改动已有的包）后 `exec python3 -u /scripts/pdd-router.py`。CPU 节点第一次拉这个约 9 GB 的镜像要几分钟。
- **RBAC**：ServiceAccount `pdd-router-sa` + Role（`pods` 的 get/list/watch，`apps/statefulsets` 的 get/list）。
- **就绪探针**：`GET /health`:30000，周期 10 秒。这是 vllm-router 自己的端点，不会在 P/D 的日志里留下访问日志。

**`pdd-router.py` 的行为**

1. 从 K8S API 读 `pdd-prefill`、`pdd-decode` StatefulSet 的 `replicas`。
2. 等 `pdd-prefill-0`、`pdd-decode-0` 都 Running + Ready + 有 IP，且 `GET http://<ip>:<port>/health` 返回 200；等待期间每 5 秒轮询，同一条状态最多每 30 秒打印一次。所以 P/D 启动期间 `pdd-router-0` 不会 Ready，这是预期行为。
3. 两者就绪后启动：

```
vllm-router --policy round_robin --vllm-pd-disaggregation --host 0.0.0.0 --port 30000
  --intra-node-data-parallel-size 8
  --prefill http://<prefill-pod-ip>:8001  --decode http://<decode-pod-ip>:8002
```

4. 之后持续监控：每 5 秒检查 router 子进程是否退出（退出则脚本也退出，让 K8S 重启 Pod；这一步不产生请求）；每 `WATCH_INTERVAL`（默认 60 秒）检查 Pod IP 和 `/health`：IP 变化（例如 P 或 D 的 Pod 重启换了 IP）时重启 router，某个 Pod 暂时不 Ready 时只记录日志（router 自己有健康检查与熔断），K8S API 偶发失败不影响正在服务的 router。
5. `INTRA_NODE_DP=8` 必须等于每个 `vllm serve` 的 `--data-parallel-size-local`（`pdd-launch.sh` 里的 `LOCAL_DP`）。
6. 不带 `--api-key`，`vllm-router` 也能启动并路由；它自己的健康检查间隔用默认值（60 秒），脚本不显式设置。

> 后端是启动时一次性发现的，P/D 换 IP 要靠第 4 步的监控重启 router 才能感知；这是脚本的逻辑，不是 vllm-router 的服务发现。

<a id="s3-7"></a>
### 3.7 运行环境记录

第 5.7 节的实验（以及 3.7 本节）是在**重新部署**的一次 1P1D 上完成的，配置、脚本和镜像与前面章节相同；第 5.1～5.6 节是在更早的一次 1P1D 部署上完成的（当时 `pdd-prefill-0` 在 `3o1v8w`、`pdd-decode-0` 在 `3o1v8e`，那次部署的镜像 digest 没有记录）。采集时间 2026-10-13 05:08 UTC，Pod 已运行 55 分钟。

**`kubectl get pods -o wide`**

```
NAME            READY   STATUS    RESTARTS   AGE   IP             NODE                        NOMINATED NODE   READINESS GATES
pdd-decode-0    1/1     Running   0          55m   10.154.2.102   rs-b300x8-gpu-pool-3o1v8b   <none>           <none>
pdd-prefill-0   1/1     Running   0          55m   10.154.1.73    rs-b300x8-gpu-pool-3o1v8w   <none>           <none>
pdd-router-0    1/1     Running   0          55m   10.154.3.202   rs-power-cpu-pool-3jmm1a    <none>           <none>
```

**`kubectl get nodes -o wide`**（省略了外网 IP 列；同集群里还有未使用的 `3o1v8e`、`3o1v8g` 两个 GPU 节点）

```
NAME                        STATUS   ROLES    AGE     VERSION   INTERNAL-IP   OS-IMAGE                       KERNEL-VERSION                CONTAINER-RUNTIME
rs-b300x8-gpu-pool-3o1v8b   Ready    <none>   36h     v1.37.0   10.132.0.20   Debian GNU/Linux 13 (trixie)   6.12.96+deb13-amd64 (amd64)   containerd://2.3.3
rs-b300x8-gpu-pool-3o1v8e   Ready    <none>   36h     v1.37.0   10.132.0.19   Debian GNU/Linux 13 (trixie)   6.12.96+deb13-amd64 (amd64)   containerd://2.3.3
rs-b300x8-gpu-pool-3o1v8g   Ready    <none>   36h     v1.37.0   10.132.0.23   Debian GNU/Linux 13 (trixie)   6.12.96+deb13-amd64 (amd64)   containerd://2.3.3
rs-b300x8-gpu-pool-3o1v8w   Ready    <none>   36h     v1.37.0   10.132.0.22   Debian GNU/Linux 13 (trixie)   6.12.96+deb13-amd64 (amd64)   containerd://2.3.3
rs-power-cpu-pool-3jmm1a    Ready    <none>   7h12m   v1.37.0   10.132.0.17   Debian GNU/Linux 13 (trixie)   6.12.96+deb13-amd64 (amd64)   containerd://2.3.3
rs-power-cpu-pool-3jmm1e    Ready    <none>   7h12m   v1.37.0   10.132.0.13   Debian GNU/Linux 13 (trixie)   6.12.96+deb13-amd64 (amd64)   containerd://2.3.3
```

**镜像与软件版本**
- 3 个 Pod 用的是同一个镜像：`vllm/vllm-openai:latest`，digest 都是 `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`。
- 镜像内：vLLM 0.31.0、NIXL 1.4.1（`nixl-cu13`）、NCCL 2.30.7、PyTorch 2.13.0+cu130；NVIDIA B300 SXM6，驱动 580.178.04；每个 GPU 节点 8 块用到的 RoCE 网卡（400 Gb/s，4X NDR，MTU 4200，单卡线速约 50 GB/s）。

**启动**
- 这次部署的启动：`init engine` Prefill 约 1098 s、Decode 约 514 s（编译都约 43 s），**Prefill 约是 Decode 的 2 倍**；Router 在 Pod 全部就绪后启动，命令里是 1 个 `--prefill`、1 个 `--decode`。两个 GPU Pod 日志里 `ERROR`/`Traceback`/`NIXL_ERR` 为 0 行，各有 8 个 `Backend UCX was instantiated`。
- 这次部署的 Router 配对（预热时统计，8 对）：P0→D2、P1→D4、P2→D1、P3→D5、P4→D3、P5→D7、P6→D6、P7→D0，其中 **P6→D6 是同轨**（网卡序号 6→6）。

<a id="s4"></a>
## 4. 观测方法

[↑ 返回顶部](#top)

<a id="s4-1"></a>
### 4.1 `./pdd.sh metrics [窗口秒数]`

在 `pdd-router-0` 里经 headless Service 的 Pod 域名读 Prefill、Decode 的 `/metrics`，只读、很轻，汇总打印：请求运行/排队/成功/抢占数；KV cache 占用（8 个 engine 的最大和平均）；prompt/生成 token 数（带窗口时有速率）；TTFT、TPOT、ITL、排队、prefill、decode、e2e 的平均值；本地与外部（NIXL）prefix cache 命中；NIXL 传输次数、总量、平均大小、平均耗时、带宽（总字节 ÷ 各次传输耗时之和，只作粗略参考）、失败/通知失败/过期计数。

- 不带参数是引擎启动以来的累计值；`./pdd.sh metrics 30` 采样两次，显示 30 秒内的增量和速率。一次读取得到的是该 Pod 全部 API server 汇总后的值（已验证）。
- NIXL 传输指标只在 Decode（consumer）一侧有数值，Prefill 侧计数为 0 是预期；Prefill 的 TPOT 为 0 也是预期（它只生成 1 个 token）。
- Decode 的外部 prefix cache 命中数等于查询数，说明 Decode 的 prompt KV 全部来自 Prefill 的传输。
- 指标带 `engine="0".."7"` 标签，对应 8 个 DP rank；`metrics` 子命令把它们相加。需要按 rank 区分时（例如看某次传输落在哪个 Decode rank），直接读 `/metrics` 的 `engine` 标签。

<a id="s4-2"></a>
### 4.2 日志的局限

- **没有周期性统计**：hybrid-lb 下 `api_server_count` 默认等于 `data-parallel-size-local`（8），而 `api_server_count > 1` 时 vLLM 会关闭周期性 stats 日志（日志里有 8 行 `AsyncLLM created with api_server_count more than 1; disabling stats logging to avoid incomplete stats.`，源码 `v1/metrics/loggers.py` 核对）。`Avg prompt throughput`、`Running: N reqs` 以及一起输出的 `KV Transfer metrics: ...` 因此都不会出现；Prometheus 的 `/metrics` 不受影响，所以用 4.1 的 `metrics`。如果一定要在日志里看，可给 `vllm serve` 加 `--api-server-count 1`：每个 Pod 只剩 1 个 API 进程服务 8 个引擎，高并发时前端可能成为瓶颈（**推断**，未测），`vllm-router` 与单个 API 进程的配合也未验证；当前脚本没有提供这个开关。
- **`GET /health` 的访问日志**有三个来源，周期现在都是 60 秒：K8S 的 startup/readiness 探针、`pdd-router.py` 的检查（`WATCH_INTERVAL`）、vllm-router 自己的健康检查（默认值）。`HIDE_HEALTH_LOG=on` 可以屏蔽（只屏蔽 `/health,/metrics,/ping`，保留 `POST` 请求日志）。
- UCX 日志（`UCX_LOG_LEVEL=info`）里每个 worker 的配置行显示传输和设备，例如 `rma(rc_mlx5/mlx5_N:1)`；P、D 的每个 worker 都有 `Backend UCX was instantiated`，没有 `NIXL_ERR_BACKEND`。

<a id="s4-3"></a>
### 4.3 网卡、内存区和 GPU

- **哪块网卡有 RDMA 流量**：Pod 里的 `rdma statistic show link mlx5_N/1`（镜像自带），给出 `rx_write_requests`、`rx_read_requests` 等**请求计数，不是字节数**；它们是目标端的计数（被读/被写的一方才增加）。
- **网线上的字节数和速率**：`fabricN` 的 `tx/rx_vport_rdma_unicast_bytes`（`fabricN`↔`mlx5_N`）。镜像里没有 `ethtool` 命令，但 Python 可以直接用 `SIOCETHTOOL` ioctl（`ETHTOOL_GSSET_INFO`/`GSTRINGS`/`GSTATS`）读，只读、不需要安装；读 8 块网卡一次约 14 ms，所以采样频率可达 70 Hz；也可以临时安装 `ethtool`（只装在运行中的 Pod 里，重启即丢）。
- **已注册的内存区**：`rdma resource show mr`，每行含设备、`mrlen`、`pdn`、`pid`（注意 `pid` 可能是线程号）；可按进程和设备汇总。
- **网卡速率**：`/sys/class/net/fabricN/speed`（本集群 400000 Mb/s）、`/sys/class/infiniband/mlx5_0/ports/1/rate`（`400 Gb/sec (4X NDR)`），MTU 4200，单卡线速约 50 GB/s。
- **GPU**：`nvidia-smi --query-gpu=utilization.gpu,memory.used`；拓扑 `nvidia-smi topo -m`；NVLink 流量 `nvidia-smi nvlink -gt d`（每条链路累计 Tx/Rx，KiB；链路状态 `nvidia-smi nvlink -s`，每条链路 53.125 GB/s，`NV18` 即 18 条）。
- 这些读取都要进 P/D Pod，保持只读、轻量；Pod 里没有 sysfs 的 `hw_counters`。

<a id="s4-4"></a>
### 4.4 进程与线程

- `ps -ef`、`/proc/<pid>/cmdline`、`/proc/<pid>/stat`（CPU 时间）、`/proc/<pid>/task/<tid>/`（逐线程的 CPU、上下文切换、`io` 的 `syscr`）。
- Python 栈：`py-spy dump --nonblocking --pid <pid>`（镜像里没有，临时 `pip install --target /tmp/pyspy` 装在容器 `/tmp`，Pod 重启即丢）；容器里没有 `gdb`/`strace`/`perf`，`/proc/<tid>/stack` 无权限，所以拿不到原生线程的栈。

<a id="s5"></a>
## 5. 实测

[↑ 返回顶部](#top)

环境：2026-10-13，`pdd-prefill-0`、`pdd-decode-0`、`pdd-router-0` 已运行 70～90 分钟，没有重启，`PREFIX_CACHING=off`；请求都经 `pdd-router-0`。

<a id="s5-1"></a>
### 5.1 进程结构

两个 Pod 的进程结构相同，每个 Pod 共 **27 个服务进程**（原始 `ps -ef` 输出见附录）：

| 进程 | 数量 | 说明 |
|---|---|---|
| `vllm serve`（PID 1） | 1 | 主进程，启动并管理其余进程 |
| `multiprocessing.resource_tracker` | 1 | Python multiprocessing 的资源跟踪进程 |
| `VLLM::DPCoordinator` | 1 | 本池的 DP 协调者 |
| `VLLM::APIServer_0..7` | 8 | 8 个 API server（见 4.2） |
| `VLLM::EngineCore_DP0..7` | 8 | 每个 DP rank 一个引擎核心（调度器），父进程是 PID 1 |
| `VLLM::Worker_DPn_EPn` | 8 | 每个 DP rank 一个 GPU worker，**父进程是对应的 `EngineCore_DPn`**；DP、EP 序号相同，对应 DP8 + EP8 |

- Prefill 与 Decode 的进程种类和数量完全一样，差别只在 PID 1 的命令行（`kv_role`、端口、Decode 多 `--compilation-config`，与 3.4 的命令一致）。两个 Pod 里 `nproc` 都是 224，线程数（`ps -eLf`）分别是 6871 和 6890。
- 没有 `vllm-router`；NIXL 传输在 Worker 进程内完成（每个 Worker 加载了 `libnixl.so`、`libplugin_UCX.so`、`libuct_ib.so`，已核对）。
- PID 14250（Prefill）和 21098（Decode）是挂在 `pts/0` 上的 `/bin/bash`，启动于 23:42:51 和 23:43:24，`TERM=xterm`，是带 TTY 的交互 `kubectl exec` 会话；是谁开的无法从 Pod 里判断。

<a id="s5-2"></a>
### 5.2 空闲时：GPU 为 0%，但 Worker 占满一个 CPU 核

- **GPU**：`nvidia-smi` 连续采样，P、D 的 8 张 GPU 利用率全是 **0%**。空闲时每卡已用显存：Prefill 242,286～247,088 MiB，Decode 229,310 MiB。
- **CPU**：`ps` 的 `C` 列显示 Worker 约 97（Prefill）和 99（Decode），累计 CPU 时间 57～61 分钟，与 Pod 运行时长相当，即每个 Worker 一直占满一个核。5 秒窗口的瞬时占用（`Worker_DP0_EP0`）：

| 进程 | Prefill | Decode |
|---|---|---|
| `Worker_DP0_EP0` | 100%（用户态 31 + 内核态 69） | 101%（用户态 23 + 内核态 78） |
| `EngineCore_DP0`、`APIServer_0`、`DPCoordinator` | 0% | 0% |

- **按线程拆分**（Decode 的 Worker 0，共 153 个线程）：只有 **1 个线程**占满一个核（tid 11409，75%～78% 为内核态），其余都低于 3%；它 2 秒内主动让出 CPU 0 次，read 系统调用约 **725,000 次/秒**、write 为 0，即一直在忙轮询。
- **Python 栈**（`py-spy`）：主线程在 `worker_busy_loop → dequeue → acquire_read → wait → zmq poll`，状态 idle（等任务）；热点线程不在 Python 线程列表里，是原生线程。
- **这个线程是什么**：
  - 创建时间：4140 秒前，换算为 23:39:51；日志里 NIXL 在 `2026-10-12 23:39:51` 初始化 agent 和 `Backend UCX was instantiated`，精确到秒吻合（Worker 进程本身创建于 4348 秒前）。
  - `libplugin_UCX.so` 里有 `nixlUcxEngine::progressLoop`、`nixlUcxWorker::progressLoop`、`ucp_worker_progress` 和 `Couldn't create progress thread control pipe`。
  - `rdma resource show mr` 里，这个线程（Decode tid 11409、Prefill tid 19084）名下有自己的 RDMA 内存区，只在 `mlx5_0`（rank 0 的网卡）上：Decode 是 2 个 20 MiB + 2 个 10 MiB，Prefill 是 2 个 20 MiB。
- **结论**：空闲时 Worker 占满一个核**不是** dummy batch 在空转（GPU 是 0%，主线程在等任务）；证据指向 NIXL 的 UCX progress 线程在忙轮询。这是强证据而不是定论——没有拿到该线程的栈；要确证需要改 NIXL 相关设置重新部署后对比。

<a id="s5-3"></a>
### 5.3 单请求：GPU 全体参与、没有唤醒延迟

依次发 3 个 32768 token 的请求（seed 不同）：请求 1 在空闲约 28 分钟后（上一个请求是 00:21，本次在 00:49），请求 2 紧接请求 1，请求 3 在请求 2 结束 30 秒后；同时在 P、D 里每 0.4 秒采一次 8 张 GPU 的利用率，并在每个请求前后读 `rdma statistic`。

| | 请求 1 | 请求 2 | 请求 3 |
|---|---|---|---|
| Benchmark duration | 6.59 s | 6.55 s | 6.54 s |
| Mean TTFT | 3531.8 ms | 3518.6 ms | 3500.5 ms |
| Mean TPOT | 16.17 ms | 15.96 ms | 16.03 ms |
| Router 选的 Prefill rank | `@0` | `@5` | `@7` |
| Prefill 侧 `rx_read_requests` 增量 | `mlx5_0`：8784 | `mlx5_10`：124928 | `mlx5_14`：124928 |
| 其余 7 块网卡 / 所有网卡的 `rx_write_requests` | 0 / 0 | 0 / 0 | 0 / 0 |
| 请求期间 8 张 Prefill GPU 最大利用率 | 100 × 8 | 100 × 8 | 100 × 8 |
| 请求期间 8 张 Decode GPU 最大利用率 | 99 × 8 | 99 × 8 | 99 × 8 |

- **8 卡陪跑已验证**：Router 只把请求分给 Prefill 的一个 DP rank，但 P、D 的 8 张 GPU 都升到 99%～100%，请求结束后回到 0%。窗口内的平均利用率只有 12%～20%，因为窗口是整个 `pdd.sh bench` 命令的耗时（约 17.7 s，其中真正的请求约 6.6 s，其余是 `vllm bench` 的启动开销），不能当作请求期间的利用率。
- **没有唤醒延迟**：空闲 28 分钟后的第一个请求 TTFT 3532 ms，与紧接着的（3519 ms）、间隔 30 秒后的（3501 ms）没有差别。预热要求见第 3.3 节。
- **EP 通信不走 RoCE**：所有 8 块网卡的 `rx_write_requests` 全程为 0。
- Prefill 侧 `rx_read_requests` 在请求 1（8784）与请求 2、3（124928）之间差别很大：这个计数取决于 NIXL 传输的 descriptor 的数量和大小，不是网卡的行为。RDMA 请求按 256 KiB 切分：32768 token 的 KV 每层一个连续段是 36 MiB，所以每个 descriptor 对应 144 个请求，61 个 descriptor（每层一个连续段）就是 8784（= 61 × 144）；完全碎片化时是 62,464 个 descriptor（61 层 × 1024 块），每个 36,864 B，对应 2 个请求，共 124,928（= 62,464 × 2）。这里的 8784 和 124928 应是这两种情形（**推断**：这几次的 descriptor 数当时没有记录；131072 token 的 35,136 = 61 × 576、499,712 = 249,856 × 2 见 5.4）。

<a id="s5-4"></a>
### 5.4 KV 传输：131072 token 的大 KV（P → D）

**目的**：用比平时大得多的 KV，看性能、尺寸、内存区、是否跨轨道、GPU 是否选就近的网卡、底层是 READ 还是 WRITE。

**方法**
- 6 个请求依次发出，每个 131072 token 输入、16 token 输出（`BENCH_INPUT_LEN=131072 BENCH_OUTPUT_LEN=16 ./pdd.sh bench --num-prompts 1`，seed 201～206），每个请求的 KV 约 9.21 GB（131072 × 70,272 B）。
- 每个请求前后：读 P、D 的 `/metrics`（按 `engine` 标签取 NIXL 字节数、传输耗时、发起耗时、descriptor 数，以及 prefill 耗时），读两侧 8 块网卡（`mlx5_0,2,…,14`）的 `rdma statistic`。
- 整个实验期间：在 P、D 里每约 15～20 ms 读一次 8 块网卡的 `tx/rx_vport_rdma_unicast_bytes`（4.3 的 ioctl 方法）。一次传输只持续约 0.2 秒，所以每次只有 10～11 个采样点，速率是粗略的。实验前读了 `rdma resource show mr`。

**结果**

| 请求 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Prefill rank → Decode rank（Router 日志，与 `engine` 标签一致） | 1 → 2 | 0 → 4 | 5 → 6 | 7 → 0 | 1 → 2 | 0 → 4 |
| Prefill 侧用的网卡 | `mlx5_2` | `mlx5_0` | `mlx5_10` | `mlx5_14` | `mlx5_2` | `mlx5_0` |
| Decode 侧用的网卡 | `mlx5_4` | `mlx5_8` | `mlx5_12` | `mlx5_0` | `mlx5_4` | `mlx5_8` |
| 其余 7 块网卡的字节数 | 0 | 0 | 0 | 0 | 0 | 0 |
| NIXL 传输字节数 | 9,210,691,584 | 同左 | 同左 | 同左 | 同左 | 同左 |
| NIXL descriptor 数 | 61 | 62,769 | 183 | 183 | 249,856 | 187,453 |
| 平均每个 descriptor 的大小 | 150,994,944 B（144 MiB） | 146,739 B | 50,331,648 B（48 MiB） | 50,331,648 B | 36,864 B | 49,136 B |
| NIXL 传输耗时（`xfer_time`） | 191.3 ms | 192.6 ms | 191.4 ms | 192.1 ms | 198.3 ms | 196.4 ms |
| 字节 ÷ `xfer_time` | 48.1 GB/s | 47.8 GB/s | 48.1 GB/s | 48.0 GB/s | 46.4 GB/s | 46.9 GB/s |
| NIXL 发起耗时（`post_time`） | 0.1 ms | 2.7 ms | 0.2 ms | 0.2 ms | 13.4 ms | 9.6 ms |
| Prefill 网卡发送的线上字节数 | 9,386 MB | 9,392 MB | 9,386 MB | 9,386 MB | 9,410 MB | 9,404 MB |
| 线速峰值（Prefill 发送 / Decode 接收） | 53.2 / 50.7 GB/s | 49.8 / 50.3 | 50.1 / 50.3 | 50.0 / 50.5 | 49.7 / 50.0 | 49.9 / 49.7 |
| 网卡上有流量的持续时间 | 约 0.18～0.20 s | 0.18 | 0.18 | 0.18～0.20 | 0.20 | 0.18～0.20 |
| Prefill 侧 `rx_read_requests` 增量（`rx_write_requests` 全为 0） | 35,136 | 151,768 | 35,380 | 35,380 | 499,712 | 383,568 |
| Decode 侧 READ/WRITE 请求增量 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Prefill 的 prefill 耗时 | 35.2 s | 34.6 s | 34.5 s | 34.1 s | 35.2 s | 34.6 s |
| Decode 记录的排队时间 | 0.207 s | 0.209 s | 0.206 s | 0.207 s | 0.224 s | 0.215 s |
| Decode 记录的 prefill 阶段耗时 | 0.520 s | 0.552 s | 0.565 s | 0.534 s | 0.539 s | 0.571 s |
| 请求端测得的 TTFT | 36,992 ms | 36,389 ms | 36,305 ms | 35,807 ms | 37,042 ms | 36,515 ms |

**性能**
- 一次 9.21 GB 的传输，NIXL 记录的耗时约 191～198 ms，折合 **46～48 GB/s（约 370～385 Gb/s）**；网线上采到的峰值约 50 GB/s，接近单卡 400 Gb/s 的线速。每次传输只用**一块网卡**。
- 网线上的字节数比 NIXL 的负载多约 1.9%～2.2%（9,386～9,410 MB 对 9,211 MB），是协议头开销；descriptor 越碎略高（请求 5 最高）。
- 传输只占 TTFT 的约 0.5%：TTFT 约 36 s，其中 Prefill 的 prefill 计算 34～35 s（约 3.7k token/s），Decode 侧排队约 0.21 s，Decode 记录的 prefill 阶段约 0.52～0.57 s（含 KV 拉取，大于 `xfer_time` 的 0.19 s，差额是握手、通知和调度，各部分没有进一步拆开）。所以在这个长度下 KV 传输不是瓶颈，瓶颈是 Prefill 计算。
- descriptor 从 61 个变成 249,856 个，`xfer_time` 只从 191 ms 变到 198 ms（约慢 3.6%），`post_time` 从 0.1 ms 变到 13.4 ms。样本只有 6 个，只能说「碎片多时稍慢」。

**尺寸与 descriptor**
- 一次传输的字节数固定：131072 token × 70,272 B = 9,210,691,584 B，与 `config.json` 的算法一致；每层 144 MiB，共 61 层。
- descriptor 是「一层里一段连续的块」，块大小 32 token，最小 36,864 B；请求 5 的 249,856 = 61 层 × 4096 块，正好每层每块一个；请求 1 只有 61 个，即每层 1 个连续的 144 MiB 段；请求 3、4 是每层 3 段、每段 48 MiB。**descriptor 的数量取决于块分配的连续程度**，同样的字节数可以是 61 个也可以是 249,856 个；这个数是 Decode 侧记录的，两侧各自的影响没有拆开验证。
- RDMA 请求计数：请求 1 的 35,136 正好是 61 × 576（每个 144 MiB 的 descriptor 对应 576 个请求，即每个请求 256 KiB）；请求 5 的 499,712 正好是 249,856 × 2。更小的 descriptor（36,864 B）是 2 个请求；请求 2 和 6 的计数如何拆分，没有逐个核对，这里只记录计数。每个计数都由 descriptor 的数量和大小决定，与哪块网卡无关。

**内存区**（`rdma resource show mr`，P、D 各 8 个 Worker）
- 每个 Worker 在**自己的那块网卡**（rank r ↔ `mlx5_{2r}`）上注册了 **2 个大小完全相同的大内存区**，大小就是该 rank 的整个 KV cache：Prefill 124,301,611,008 B（115.77 GiB，日志里 `GPU KV cache size: 1,768,864 tokens`，1,768,864 × 70,272 B 正好吻合），Decode 128,311,050,240 B（119.5 GiB，对应 1,825,920 tokens）。两个内存区在同一个 protection domain，为什么是 2 个没有查清。
- 其他 7 块网卡上，这个 Worker 没有注册 KV 内存区，只有一些小内存区（8 B、544,768 B、6.3 MB、39.8 MB 等），用途没有逐个核对。
- 所以 KV cache 是**一整块连续注册的显存**，不是每层单独注册；一次传输是把这块大内存区里的若干段拉到 Decode 那边另一块大内存区的对应位置。

**是否跨轨道**
- 6 次传输的 Prefill rank → Decode rank 是 1→2、0→4、5→6、7→0、1→2、0→4，序号全部不同，即**全部跨轨道**（`mlx5_{2r}` 对 `mlx5_{2d}`，r ≠ d），仍有约 48 GB/s，没有看到速度惩罚。
- 看 Router 的全部日志（27 个请求，含本实验的 6 个和之前的测试）：配对是**固定的 8 种**（8×8 = 64 种组合里只出现这 8 种）：P0→D4、P1→D2、P2→D7、P3→D5、P4→D3、P5→D6、P6→D1、P7→D0，从未出现 r = d。所以在**那一次 Router 启动**里同轨道配对没有出现。但配对随 Router 启动而变：后来重新部署的这次，8 对里就有 1 对同轨（P6→D6），同轨传输的对比见 5.7。配对的规律也查清了一部分：Decode rank 按固定顺序轮转、Prefill rank 按另一个固定顺序轮转，周期都是 8，所以组成固定的一一对应（**推断**，没有读 `vllm-router` 的源码）。

**GPU 是否选就近的网卡**
- 是。Prefill rank r 总是用 `mlx5_{2r}`，Decode rank d 总是用 `mlx5_{2d}`；`nvidia-smi topo -m` 显示 GPU i 与 `mlx5_{2i}`、`mlx5_{2i+1}` 同属一个 PCIe 桥（`PXB`），用的是偶数那块，`UCX_NET_DEVICES` 里也只列了偶数设备。结合「KV 内存区只注册在自己那块网卡上」，传输自然落在它上面。6 次传输覆盖了 4 个不同的 Prefill rank（0、1、5、7）和 4 个不同的 Decode rank（0、2、4、6），没有一次用到别的网卡。

**底层是 RDMA READ，由 Decode 发起**
- Prefill 侧一块网卡的 `rx_read_requests` 增加，`rx_write_requests` 全程为 0；Decode 侧两类计数都不增加（这两个计数是目标端的，被读的一方才增加）；
- 线上的方向是 Prefill 发、Decode 收：Prefill 的 `tx_vport_rdma_unicast_bytes` 和 Decode 的 `rx_vport_rdma_unicast_bytes` 都增加约 9.4 GB，且只在这一对网卡上；
- 源码里用 `make_prepped_xfer("READ", ...)`，完成通过通知消息告知对端。

因此是 Decode 主动从 Prefill 的显存拉取，Prefill 的 CPU 和 GPU 不参与数据搬运。

<a id="s5-5"></a>
### 5.5 拓扑与模型配置

- `nvidia-smi topo -m`（P、D 输出相同）：8 张 GPU 两两 `NV18`；GPU 0～3 在 NUMA 0（CPU 0-111），GPU 4～7 在 NUMA 1（CPU 112-223）。GPU i 与 NIC{2i}、NIC{2i+1} 是 `PXB`，与同 NUMA 的其他网卡是 `NODE`，与另一个 NUMA 的是 `SYS`；NIC{2i} 与 NIC{2i+1} 之间是 `PIX`。（`NICn` 对应 `mlx5_n`。）
- `rdma link show`：`mlx5_0,2,…,14` 都是 `ACTIVE / LINK_UP`，对应 `fabric0,2,…,14`；`mlx5_1,3,…` 没有 netdev。
- 模型 `config.json`：`num_hidden_layers=61`、`first_k_dense_replace=3`、`n_routed_experts=256`、`n_shared_experts=1`、`num_experts_per_tok=8`、`kv_lora_rank=512`、`qk_rope_head_dim=64`；所以 58 个 MoE 层，EP8 下每卡 32 个路由专家。

<a id="s5-6"></a>
### 5.6 EP 的集合通信走 NVLink

问题：`all_gatherv` + `reduce_scatterv` 是不是在节点内走 NVLink（因为 EP 组在单节点内）？前面用拓扑（`NV18` 全互联）和 RoCE 无流量做了间接判断（5.3、5.5），这里直接读 NVLink 的流量计数。

**方法**：`nvidia-smi nvlink -gt d`（每条 NVLink 的累计 Tx/Rx 数据量，KiB），在 P、D 两个 Pod 里把每张 GPU 的所有链路求和；先空闲 10 秒取一次差值，再发 1 个 32768 token 的请求（`./pdd.sh bench --num-prompts 1`，seed 301，TTFT 3531 ms），前后各取一次。每次读取需要约 5 秒，所以窗口是 21.9 秒，包含 `vllm bench` 的启动开销（请求本身约 6.6 秒）。

**结果**

| | 空闲 10 秒 | 1 个 32768 token 的请求 |
|---|---|---|
| Prefill，8 张 GPU 的 NVLink Tx 合计 | 0 | 277.9 GiB（Rx 合计相同） |
| Prefill，各 GPU 的 Tx | 全为 0 | 39.66 × 6，25.64（GPU 4），14.29（GPU 5） |
| Decode，8 张 GPU 的 NVLink Tx 合计 | 0 | 19.49 GiB（Rx 合计相同） |
| Decode，各 GPU 的 Tx | 全为 0 | 2.44 × 8 |

**解读**
- 空闲时 NVLink 上没有任何流量；有请求时 P、D 的 8 张 GPU 之间出现大量 NVLink 流量，而同一类请求在 RoCE 网卡上只有 NIXL 的 READ（5.3），所以**节点内的 MoE 集合通信走 NVLink，不走 RoCE**，已验证。
- Prefill 的流量远大于 Decode（约 278 GiB 对 19.5 GiB）：Prefill 一次处理 32768 个 token，每个 MoE 层的 `all_gatherv` 要把这些 token 的激活广播给所有 rank，而 Decode 每步只有几个 token。
- Decode 的 8 张 GPU 完全均匀（2.44 GiB）；Prefill 里有 6 张均匀（39.66 GiB），GPU 4 和 GPU 5 较少（25.64、14.29 GiB），原因没有查清（可能与这个请求落在哪个 rank 有关，但没有验证）。
- 这个计数只说明字节走了 NVLink，**不能**区分 NCCL 具体用了哪种算法或协议（例如 P2P、NVLS），也不能把字节数拆成 `all_gatherv` 和 `reduce_scatterv` 各自的份额。
- 计数是该 GPU 上全部 NVLink 流量，包括 NCCL 之外可能存在的点对点拷贝；NIXL 传 KV 走的是 RDMA 网卡，不经过 NVLink。

<a id="s5-7"></a>
### 5.7 并发下的 KV 传输

**目的**：看并发时 KV 传输是否变慢，慢在网卡还是别处。1P1D 的 EP 集合通信全在节点内，RoCE 上只有 KV，所以这里的数字是不含集合通信干扰的基线。

**方法**：重新部署的 1P1D，先预热（24 个 32768 token 请求、并发度 8，覆盖了全部 8 个 Prefill rank 和 8 个 Decode rank），再做：(1) 1 个 32768 token 的单请求；(2) 4 个顺序的 32768 token 请求；(3) 4 个 131072 token 的请求；(4) 24 个 32768 / 128 token 的请求、并发度 8，做 2 次。测量方式见第 4 章和附录 A.4。

**单请求与顺序请求**（1P1D，预热后）

| | 单请求 | 4 个顺序请求 |
|---|---|---|
| TTFT | 3.55 s | 3.51～3.55 s |
| TPOT | 16.3 ms | 16.1～16.3 ms |
| Prefill 耗时 | 3.06 s | 2.99～3.05 s |
| NIXL 传输耗时 | 48.2 ms（62,464 个 descriptor，`post_time` 2.4 ms） | 平均 49.3 ms（4 次，descriptor 平均 54,671 个） |
| 线上突发（Decode 侧接收） | 60 ms，39.1 GB/s，峰值 49.8 | 中位数 60 ms（40～60），39.1 GB/s |
| NVLink 流量 | Prefill 278.1 GiB，Decode 19.2 GiB | — |
| RoCE | 只有 KV：两侧各一块网卡 2352 MB，其余网卡 ≤ 12 MB | 同样只有 KV |

**4 个 131072 token 的请求**（1P1D，预热后）：P3→D5、P5→D7、P2→D1、P4→D3，网卡序号 3→5、5→7、2→1、4→3，**全部跨轨**；TTFT 36.3～36.9 s，Prefill 耗时 34.46～35.14 s；NIXL 传输耗时 189.8～192.6 ms（47.8～48.5 GB/s），`post_time` 0.2～4.3 ms；descriptor 数 183、61、61、93,757；每次只有一块 Prefill 网卡和一块 Decode 网卡有 >10 GB/s 的流量（线上突发 0.201 s，峰值 49.7～51.9 GB/s）；Prefill 侧 `rx_read_requests` 35,136（descriptor 为 183、61、61 个的三次）、209,352（93,757 个的一次），写请求计数为 0。

**并发 24 个请求，并发度 8**（做 2 次）

| | 第 1 次 | 第 2 次 |
|---|---|---|
| 耗时 | 28.7 s | 28.6 s |
| 请求吞吐 / token 吞吐 | 0.84 req/s / 27.5k tok/s | 0.84 req/s / 27.6k tok/s |
| TTFT 平均 / 中位数 / P99 | 6.85 / 6.72 / 8.63 s | 6.86 / 6.70 / 8.59 s |
| TPOT 平均 / P99 | 17.4 / 17.9 ms | 17.3 / 17.7 ms |
| NIXL 传输耗时平均 | **65.5 ms** | **57.0 ms** |
| NIXL `post_time` 平均 | 2.1 ms | 1.6 ms |
| descriptor 平均 / 次 | 48,172 | 33,880 |
| `xfer_time` 分布 | 9 次 ≤ 50 ms，21 次 ≤ 75 ms，24 次 ≤ 200 ms | 13 次 ≤ 50 ms，23 次 ≤ 75 ms，24 次 ≤ 200 ms |
| 线上突发（Decode 侧接收）中位数 | 60 ms，39.0 GB/s（40～64 ms） | 60 ms，39.0 GB/s（40～61 ms） |
| 每个请求的 NVLink（Prefill / Decode） | 275.8 / 5.2 GiB | 275.8 / 5.4 GiB |
| RoCE 上的集合通信 | 无（每块网卡 7051～7057 MB = 3 次 KV 传输） | 无 |

**结论**
- **NIXL 记录的传输耗时在并发时变长**：单请求约 49 ms，并发时 57.0 ms 和 65.5 ms（+16%～34%），两次运行之间就差 8.5 ms。
- **线上速度没有变**：并发时 KV 突发的中位数是 60 ms、约 39 GB/s，与单请求相同。RoCE 上只有 KV，没有集合通信，所以**变长不是网卡或集合通信造成的**；更可能是并发本身（Decode 引擎在持续 decode，传输完成通知要等引擎处理，**推断**，没有验证）。
- **并发吞吐**：24 个请求、并发度 8，0.84 req/s，TTFT 平均 6.85 s，TPOT 17.3～17.4 ms。更高并发没有测。
- **每个请求的 NVLink 流量**：Prefill 在并发下基本不变（275.8 GiB，与单请求的 278.1 GiB 相同）；Decode 从单请求的 19.2 GiB 降到 5.2～5.4 GiB（约 1/3.6），因为并发的请求共享 decode 步，一步的集合通信由同一步里的所有请求分摊（**推断**，由数字得出）。

**同轨与跨轨**（这次部署的配对里 P6→D6 是同轨）：并发实验里 8 个 Decode engine 各收到 3 次传输，engine 6 对应同轨的 P6→D6。两次运行里，每个 engine 的平均传输耗时（ms）：

| Decode engine | 0 | 1 | 2 | 3 | 4 | 5 | **6（同轨）** | 7 |
|---|---|---|---|---|---|---|---|---|
| 第 1 次 | 55.4 | 57.5 | 82.4 | 49.4 | 52.7 | 87.1 | **50.4** | 89.2 |
| 第 2 次 | 58.3 | 51.3 | 52.6 | 50.1 | 56.7 | 53.0 | **85.1** | 48.9 |
| 两次合计（每个 6 次） | 56.9 | 54.4 | 67.5 | 49.8 | 54.7 | 70.0 | **67.8** | 69.1 |

- 同轨（engine 6，6 次）平均 67.8 ms，跨轨（其余 42 次）平均 60.3 ms，没有稳定的差别：同一个 engine 在两次运行里可以从 50 ms 变到 85 ms（engine 6 本身是 50.4 → 85.1，engine 2、5、7 也是），波动远大于同轨与跨轨的差别。同轨的线上突发（NIC 6 的 3 次传输）也与跨轨相同（60 ms 左右、约 39 GB/s）。
- **结论：同轨和跨轨的 KV 传输速度没有看到差别**（6 次同轨对 42 次跨轨，且有较大的运行间波动）。

<a id="s6"></a>
## 6. 推断核对一览

[↑ 返回顶部](#top)

| 说法 | 结果 |
|---|---|
| 空闲时 Worker 占满 CPU 是 dummy batch 在空转 | **被推翻**：空闲时 GPU 是 0%，主线程在等任务；热点是一个原生线程，更可能是 NIXL 的 UCX progress 线程（5.2） |
| 只发 1 个请求，Prefill、Decode 的 8 张卡都会忙 | **已证实**，P、D 各 3 次（5.3） |
| 空闲一段时间后的第一个请求会更慢（唤醒整组） | **未观察到**（5.3） |
| NIXL 传输在 worker 进程内完成 | **已证实**（5.1） |
| EP 的集合通信在节点内、走 NVLink、不走 RoCE | **已证实**：`NV18` 全互联；请求期间无 RDMA WRITE 计数（5.3、5.5）；NVLink 计数空闲为 0、有请求时大幅增加（5.6）。具体用哪种 NCCL 算法/协议没有核对 |
| 每个 worker 用离自己 GPU 最近的网卡（rank r ↔ `mlx5_{2r}`） | **已证实**（5.4） |
| KV 传输走 RDMA 而不是 TCP | **已证实**：只有 1 块 RoCE 网卡有 RDMA READ 计数；UCX 配置行是 `rc_mlx5`（5.4、4.2） |
| 传输方向和方式 | **RDMA READ，Decode 发起**（5.4） |
| DP 同步在异步调度下走 CPU 上的 Gloo | 日志确认 NCCL 被禁用；走 Gloo 是源码结论 |
| `dp_sync_interval` 默认 16；`api_server_count > 1` 时关闭 stats 日志 | **已在镜像源码里确认** |
| 每 token KV ≈ 70.3 KB；61 层、58 个 MoE 层、每卡 32 个专家 | **已用 `config.json` 和传输字节数核对** |
| 一个 Pod 的 `/metrics` 汇总了全部 API server | **已证实**（传输计数与请求数吻合） |
| `UCX_IB_GID_INDEX`、`UCX_IB_TRAFFIC_CLASS` 的作用 | **仍未验证**：需要改值重新部署做对比 |
| `UCX_MAX_RNDV_RAILS=1` 的作用 | **仍未验证**：观察到每次只用 1 块网卡，与它一致，但没有对比 |
| `--api-server-count 1` 的性能影响 | **仍未验证**：需要重新部署 |
| 最慢的 rank 决定整组步调；陪跑与 all2all 后端无关；稠密模型 DP 没有陪跑 | **仍未验证** |
| 多节点池（每个池多于 1 个节点） | **仍未验证**：需要重新部署 |

<a id="s7"></a>
## 7. 未验证与注意

[↑ 返回顶部](#top)

**未验证**
- **同轨传输**的速度：只有重新部署后的那一次有 1 对同轨配对（P6→D6），收到 6 次传输，没有看到与跨轨的差别（5.7），样本少，且运行间波动大。
- 其他 KV 长度下的带宽（只测了 131072 和 32768 token）；多个并发传输是否共享同一块网卡、互相影响；单次传输内部的速率曲线（采样点太少）。
- 两个相同大小的 KV 内存区为什么是 2 个；小内存区的用途；RDMA 请求计数里更小的 descriptor 为什么是 2 个请求；Router 的配对顺序为什么在重启或重新部署后改变（只观察到 P、D 各一个固定顺序的轮询，没有读 `vllm-router` 源码）。
- 空闲 CPU 热点线程的确切身份（5.2）。
- NCCL 在 NVLink 上具体用的算法/协议，`all_gatherv` 与 `reduce_scatterv` 各自的流量份额，以及 Prefill 里 GPU 4、GPU 5 的 NVLink 流量偏低的原因（5.6）。
- 多节点池（每个池多于 1 个节点，如 DP16+EP16）、EP 不能整除专家数的节点数（如 3、6）；非 0 号节点等 Head RPC 端口的启动逻辑只在设计上存在。
- 并发压测、P/D 配比（如 1P+3D）、性能基准。
- 第 6 章里标「仍未验证」的各项。

**注意**
- `IMAGE` 默认 `:latest`，三个 workload 可能拿到不同版本，要保证一致请用固定 tag。
- `PREFIX_CACHING` 与 `HIDE_HEALTH_LOG` 只影响 P、D；Router 与 vllm-router 不受它们控制。
- 部署的创建、重建和删除由使用者决定；调整节点数或任何开关都要 `down` 再 `up`。

<a id="s8"></a>
## 8. 附录：原始输出与统计脚本

[↑ 返回顶部](#top)

用 `kubectl exec <pod> -- ps -ef --width 250` 读取，2026-10-13 00:35 UTC，Pod 已运行约 60 分钟，1P1D 空闲。输出没有删行，只有两处处理：命令行被截到 250 列（PID 1 的完整命令行见下），以及末尾 3 行（`bash -c date ...`、`ps -ef`、`cut`）是执行这次采集本身产生的进程，不属于服务。`C` 列是进程生命周期内的平均 CPU 占比，不是瞬时值。

<a id="a1"></a>
### A.1 `pdd-prefill-0`

```
2026-10-13T00:35:27Z
UID          PID    PPID  C STIME TTY          TIME CMD
root           1       0  1 Oct12 ?        00:00:39 /usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --host 0.0.0.0 --port 8001 --kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_loa
root        1409       1  0 Oct12 ?        00:00:00 /usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(40)
root        1410       1  0 Oct12 ?        00:00:26 VLLM::DPCoordinator
root        1610       1  1 Oct12 ?        00:00:53 VLLM::EngineCore_DP0
root        1611       1  1 Oct12 ?        00:00:48 VLLM::EngineCore_DP1
root        1612       1  1 Oct12 ?        00:00:49 VLLM::EngineCore_DP2
root        1613       1  1 Oct12 ?        00:00:53 VLLM::EngineCore_DP3
root        1614       1  1 Oct12 ?        00:00:46 VLLM::EngineCore_DP4
root        1615       1  1 Oct12 ?        00:00:54 VLLM::EngineCore_DP5
root        1616       1  1 Oct12 ?        00:00:51 VLLM::EngineCore_DP6
root        1617       1  1 Oct12 ?        00:00:52 VLLM::EngineCore_DP7
root        1618       1  0 Oct12 ?        00:00:17 VLLM::APIServer_0
root        1619       1  0 Oct12 ?        00:00:15 VLLM::APIServer_1
root        1620       1  0 Oct12 ?        00:00:22 VLLM::APIServer_2
root        1621       1  0 Oct12 ?        00:00:24 VLLM::APIServer_3
root        1622       1  0 Oct12 ?        00:00:24 VLLM::APIServer_4
root        1623       1  0 Oct12 ?        00:00:20 VLLM::APIServer_5
root        1624       1  0 Oct12 ?        00:00:21 VLLM::APIServer_6
root        1625       1  0 Oct12 ?        00:00:21 VLLM::APIServer_7
root        7198    1617 97 Oct12 ?        00:57:21 VLLM::Worker_DP7_EP7
root        7199    1616 97 Oct12 ?        00:57:23 VLLM::Worker_DP6_EP6
root        7200    1610 97 Oct12 ?        00:57:27 VLLM::Worker_DP0_EP0
root        7202    1612 97 Oct12 ?        00:57:27 VLLM::Worker_DP2_EP2
root        7207    1613 97 Oct12 ?        00:57:32 VLLM::Worker_DP3_EP3
root        7208    1614 97 Oct12 ?        00:57:27 VLLM::Worker_DP4_EP4
root        7209    1615 97 Oct12 ?        00:57:20 VLLM::Worker_DP5_EP5
root        7210    1611 97 Oct12 ?        00:57:28 VLLM::Worker_DP1_EP1
root       14250       0  0 Oct12 pts/0    00:00:00 /bin/bash
root       24999       0  0 00:35 ?        00:00:00 bash -c date -u +%FT%TZ; ps -ef --width 250 | cut -c1-250
root       25006   24999  0 00:35 ?        00:00:00 ps -ef --width 250
root       25007   24999  0 00:35 ?        00:00:00 cut -c1-250
```

<a id="a2"></a>
### A.2 `pdd-decode-0`

```
2026-10-13T00:35:28Z
UID          PID    PPID  C STIME TTY          TIME CMD
root           1       0  1 Oct12 ?        00:00:39 /usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --host 0.0.0.0 --port 8002 --kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_loa
root        1406       1  0 Oct12 ?        00:00:00 /usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(40)
root        1407       1  0 Oct12 ?        00:00:26 VLLM::DPCoordinator
root        1607       1  1 Oct12 ?        00:00:55 VLLM::EngineCore_DP0
root        1608       1  1 Oct12 ?        00:00:54 VLLM::EngineCore_DP1
root        1609       1  1 Oct12 ?        00:01:00 VLLM::EngineCore_DP2
root        1610       1  1 Oct12 ?        00:00:58 VLLM::EngineCore_DP3
root        1611       1  1 Oct12 ?        00:00:57 VLLM::EngineCore_DP4
root        1612       1  1 Oct12 ?        00:00:50 VLLM::EngineCore_DP5
root        1613       1  1 Oct12 ?        00:00:44 VLLM::EngineCore_DP6
root        1614       1  1 Oct12 ?        00:00:53 VLLM::EngineCore_DP7
root        1615       1  0 Oct12 ?        00:00:19 VLLM::APIServer_0
root        1616       1  0 Oct12 ?        00:00:18 VLLM::APIServer_1
root        1617       1  0 Oct12 ?        00:00:19 VLLM::APIServer_2
root        1618       1  0 Oct12 ?        00:00:20 VLLM::APIServer_3
root        1619       1  0 Oct12 ?        00:00:16 VLLM::APIServer_4
root        1620       1  0 Oct12 ?        00:00:22 VLLM::APIServer_5
root        1621       1  0 Oct12 ?        00:00:17 VLLM::APIServer_6
root        1622       1  0 Oct12 ?        00:00:20 VLLM::APIServer_7
root        7157    1607 99 Oct12 ?        01:00:38 VLLM::Worker_DP0_EP0
root        7159    1613 99 Oct12 ?        01:00:40 VLLM::Worker_DP6_EP6
root        7189    1614 99 Oct12 ?        01:00:39 VLLM::Worker_DP7_EP7
root        7192    1610 99 Oct12 ?        01:00:49 VLLM::Worker_DP3_EP3
root        7193    1608 99 Oct12 ?        01:00:45 VLLM::Worker_DP1_EP1
root        7197    1612 99 Oct12 ?        01:00:42 VLLM::Worker_DP5_EP5
root        7198    1609 99 Oct12 ?        01:00:43 VLLM::Worker_DP2_EP2
root        7199    1611 99 Oct12 ?        01:00:37 VLLM::Worker_DP4_EP4
root       21098       0  0 Oct12 pts/0    00:00:00 /bin/bash
root       25288       0  0 00:35 ?        00:00:00 bash -c date -u +%FT%TZ; ps -ef --width 250 | cut -c1-250
root       25295   25288  0 00:35 ?        00:00:00 ps -ef --width 250
root       25296   25288  0 00:35 ?        00:00:00 cut -c1-250
```

<a id="a3"></a>
### A.3 PID 1 的完整命令行（`/proc/1/cmdline`）

Prefill：

```
/usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --host 0.0.0.0 --port 8001 --kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"} --no-enable-prefix-caching --data-parallel-size 8 --data-parallel-size-local 8 --data-parallel-start-rank 0 --data-parallel-address 10.154.1.123 --data-parallel-rpc-port 13345 --data-parallel-hybrid-lb --enable-expert-parallel --enable-auto-tool-choice --tool-call-parser deepseek_v3 --reasoning-parser deepseek_v3
```

Decode：

```
/usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --host 0.0.0.0 --port 8002 --kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"} --no-enable-prefix-caching --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"} --data-parallel-size 8 --data-parallel-size-local 8 --data-parallel-start-rank 0 --data-parallel-address 10.154.2.224 --data-parallel-rpc-port 13345 --data-parallel-hybrid-lb --enable-expert-parallel --enable-auto-tool-choice --tool-call-parser deepseek_v3 --reasoning-parser deepseek_v3
```

<a id="a4"></a>
### A.4 NVLink / RDMA 流量统计脚本

本文的流量数字（NVLink 的 GiB、RoCE 的 MB 和线上速度、RDMA 请求数、NIXL 传输耗时）都由下面几个小脚本得到。它们放在 `traffic-tools/` 目录，全部**只读**：只读计数器，不改任何配置，Pod 里不需要安装任何东西。

**采集脚本**

| 脚本 | 运行位置 | 读什么 | 怎么读 |
|---|---|---|---|
| `nvlink_counters.py` | P/D Pod 内 | 每张 GPU 的 NVLink 累计 Tx/Rx（KiB） | 解析 `nvidia-smi nvlink -gt d`，把一张 GPU 的所有链路相加 |
| `roce_bytes_sampler.py` | P/D Pod 内 | 8 块网卡的 `tx/rx_vport_rdma_unicast_bytes`（累计字节） | 用 ethtool 的 ioctl（`SIOCETHTOOL`）读 `fabric0,2,…,14`，对应 `mlx5_0,2,…,14`；默认每 20 ms 采样一次 |
| `rdma_request_counters.sh` | P/D Pod 内（`kubectl exec`） | 每块网卡收到的 RDMA WRITE / READ 请求数 | `rdma statistic show link mlx5_N/1` |
| `nixl_metrics.py` | Router Pod 内 | NIXL 传输字节、传输耗时、post 耗时、descriptor 数、prefill 时间等，按引擎（DP rank）分开 | 抓 P/D 的 `/metrics`，保留所需的几条 |

**实验与分析脚本**

| 脚本 | 作用 |
|---|---|
| `exp_window.sh` | 跑一条 bench 命令，测整个窗口的流量（前后快照 + RoCE 采样），用于并发实验 |
| `exp_per_request.sh` | 逐个请求测，每个请求前后各取一次快照，用于 131072 token 的大 KV |
| `analyze_window.py` | 汇总窗口内每 GPU 的 NVLink GiB、每块网卡的 RoCE MB 和峰值速率、READ/WRITE 请求数、NIXL 耗时（含 `xfer_time` 直方图分位数）、KV 线上突发的时长和速率、Router 配对 |
| `analyze_per_request.py` | 每个请求一块输出：Prefill/Decode rank 对、是否同轨、TTFT、NIXL 耗时和带宽、服务的网卡、线上突发的字节数/时长/峰值 |
| `traffic_common.py` | 两个分析脚本共用的函数（解析 Router 日志、读采样数据、找突发等） |

**其他只读小脚本**

| 脚本 | 运行位置 | 作用 |
|---|---|---|
| `router_pairs.py` | 能用 `kubectl` 的机器 | 统计 Router 日志里的 P→D 配对：每一对出现的次数、两端网卡序号、是否同轨、选择顺序（5.4） |
| `idle_cpu.py` | P/D Pod 内 | 空闲时每个 Worker 和其他进程 3 秒内的 CPU 占用（5.2） |
| `rdma_mr.py` | P/D Pod 内 | 每个 Worker 在哪些网卡上注册了哪些内存区、多大（5.4） |

**怎么算**

- **计数器都是累计值**：前后各取一次快照相减。NVLink：`(after − before) / 2²⁰` 得 GiB（每张 GPU），8 张相加是整个节点；每个字节有一张 GPU 发、另一张收，所以全节点 Tx 总和等于 Rx 总和。
- **线上速度**：RoCE 采样的相邻两个值相减再除以时间间隔，速率超过 10 GB/s 的连续区间算一次 KV 突发（空闲时的心跳和其他网卡上的集合通信远低于这个值，KV 传输是 40～50 GB/s）。时间分辨率就是采样间隔 20 ms。
- **READ 请求数**：Decode 发起 RDMA READ，所以 READ 计数增加的是**服务数据的 Prefill 网卡**；这个计数按请求/descriptor 数量增加，**不是字节数**（8784 对 124928 的解释见第 5.4 节）。
- **平均传输耗时和带宽**：`nixl_xfer_time_seconds_sum / nixl_xfer_time_seconds_count`；`nixl_bytes_transferred_sum / nixl_xfer_time_seconds_sum`。

**使用时的注意事项**

- 测量前必须预热，让每个 DP rank 都处理过长请求（第 3.3 节），否则首个请求的一次性开销会混进数据。
- 请求从 Router Pod 发出（`pdd.sh bench` 就是这样做的）；P/D Pod 只做只读的计数器读取。
- NVLink 计数器包含所有 NVLink 流量，不区分来源，要在池空闲的状态下围着一个已知的负载取前后快照；一次读取约 0.4～1 s，快照不是原子的。
- RoCE 计数器只统计 RDMA 流量（NIXL/UCX 的 KV 传输，以及 NCCL 走 RoCE 的部分），eth0 上的 TCP 不在内。
- `exp_*.sh` 默认测 1P1D，测多节点池要设置环境变量 `PODS`、`PN`、`DN`；分析脚本从目录里的文件自动识别 Pod 数量，Router 日志里的 IP 靠实验开始时保存的 `ipmap.json` 换算成 Pod。

**验证情况**：两个分析脚本用本文 1P1D 的实测数据重跑过，结果与文中一致（4×131072：传输 189.8～192.6 ms、47.8～48.5 GB/s；并发 24 个请求：65.5 ms，`xfer_time` 分布 9/21/24 次落在 50/75/200 ms 以内）；采集脚本和 `router_pairs.py`、`idle_cpu.py`、`rdma_mr.py` 在运行中的 1P1D 上做过只读调用，输出正常；`exp_*.sh` 两个驱动脚本只做了语法检查，整理成现在的形式后没有重新跑过完整实验（整理前的原始版本是当时实际使用的）。

[↑ 返回顶部](#top)
