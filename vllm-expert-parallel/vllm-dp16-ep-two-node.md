# vLLM 双节点 DP=16 + EP：DeepSeek-V3-0324（TP=1，每节点 8 卡）

把一个 DP=16、EP=16 的 vLLM 部署跨两个 8 卡节点运行：Node 1（server-1）对外提供 API，承载 DP0–DP7；Node 2（server-2）以 headless 模式承载 DP8–DP15。外部请求只访问 Node 1 的 8000 端口。每张 GPU 是一个 DP rank，同时也是一个 EP rank，共 16 张。

本文是 EP 系列实验的第二篇，单节点（DP=8 + EP=8）见 `vllm-dp8-ep-single-node.md`，概念笔记见 `expert-parallel.md`。

标注约定：**实测**＝本机日志或命令输出；**源码核对**＝读了镜像内 vLLM 源码；**推断**＝由机制或现象得出，未直接验证。

## 摘要

| 问题 | 结论 |
|---|---|
| 原定的 DeepEP 能跑吗 | 不能。`--all2all-backend deepep_low_latency` 在第一次 MoE 前向就报 `internode_ll.cu:553 'named symbol not found'`。原因（推断，未重编译验证）：B300 算力 10.3（sm_103），镜像里的 DeepEP 只含 `sm_100a` 和 `sm_90a` 的 cubin，没有覆盖 sm_103（6.3） |
| 实际怎么跑的 | 默认后端 `allgather_reducescatter`，其余参数与原脚本一致，只改了 IP。成功启动，请求在 16 个 DP rank 间分配均匀（161–168） |
| 必须额外设置的环境变量 | `NCCL_IB_GID_INDEX=3`：不设时 NCCL 选到链路本地的 RoCE GID（`fe80::`），跨节点不可达，16 个 rank 的第一次集合通信静默卡死（6.1）。用 DeepEP 还需要 NVSHMEM 自己的 `NVSHMEM_IB_GID_INDEX=3` 或 `NVSHMEM_IB_ADDR_FAMILY=AF_INET6`（6.2） |
| 通信分层 | 请求入口、DPCoordinator、DP 同步这类控制面走 TCP（eth0）；各 DP rank 的 Attention 互不通信；MoE 层集合通信的数据面，节点内走 NVLink、跨节点走 RDMA（后两者是推断）（4.1） |
| 跨节点数据面走 RDMA | 一个请求 43 秒内 eth0 只多了约 34–35 MB，RDMA 收到的 write 请求增加了约 2080 万次（4.3） |
| 空闲 rank | 只发 1 个请求，两个节点的 16 张 GPU 都在忙（92–100%），与单节点一致（5.1） |
| 吞吐（粗测） | 单流约 34.6 tok/s（单节点约 60）；512 并发 8203 tok/s（单节点 8070）；2048 并发 17808 tok/s（单节点未测）（5.3） |
| 遗留问题 | 负载下 RDMA 的 `packet_seq_err` / `out_of_sequence` 持续增长，会触发重传，原因未排查（4.4） |

## 1. 环境与拓扑

| | Node 1 | Node 2 |
|---|---|---|
| Pod | server-1 | server-2 |
| 所在节点 | rs-b300x8-gpu-pool-3okyg9 | rs-b300x8-gpu-pool-3okygz |
| Pod IP（eth0） | 10.154.14.180 | 10.154.15.93 |
| 角色 | API server + DPCoordinator + DP0–DP7 | headless，DP8–DP15 |
| GPU | 8×B300（算力 10.3） | 8×B300（算力 10.3） |

- 镜像 `vllm/vllm-openai:cu130-nightly`，vLLM 0.19.2rc1.dev134，nvidia-nccl-cu13 2.28.9，nvidia-nvshmem-cu13 3.4.5，deep_ep 1.2.1+73b6ea4，驱动 590.48.01，两个节点相同。
- 实验开始前 server-1 / 2 / 3 被重建过，Pod IP 与之前不同，本文用的是重建后的 IP。两个 Pod 之间 ping 通（eth0 延迟约 0.4–1.8 ms）。
- 模型 `deepseek-ai/DeepSeek-V3-0324`，缓存目录 642 GB。两个节点上都已有完整缓存（server-1 / server-2 上 `hf download` 立即返回，没有重新下载；下载命令和耗时见 `vllm-dp8-ep-single-node.md` 第 2.1 节，在 server-0 上执行过）；实测 hostPath 缓存在节点间不共享（在 server-0 上建的探针文件在另两个节点看不到），所以每个节点各自需要一份。
- 网络：每个 Pod 有 `eth0`（IPv4，Pod 网络）和 8 个 RoCE 网卡 `fabric0/2/…/14`（对应 `mlx5_0/2/…/14`），每个网卡有 IPv6 地址 `fd02:0:<rail>:<节点号>::1/64`。同一条 rail 在两个 Pod 之间 `ping6` 通，延迟约 0.1 ms。
- 参考：<https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/>；稠密模型双节点 DP 的实验见 `../data-parallel/vllm-dp4-tp1-two-node.md`。

```
                        Client
                          │ HTTP :8000
                          ▼
  Node 1 (server-1, 10.154.14.180)
  ├─ APIServer_0..7   共用 8000 端口
  ├─ DPCoordinator
  └─ EngineCore_DP0..7 ── Worker_DP0_EP0 .. DP7_EP7   (GPU0..7)
        ▲                       ▲
        │ TCP (ZMQ, eth0)       │ NCCL all-gather / reduce-scatter（跨节点部分走 RDMA/RoCE）
        ▼                       ▼
  Node 2 (server-2, 10.154.15.93)   headless，无 API server
  └─ EngineCore_DP8..15 ── Worker_DP8_EP8 .. DP15_EP15  (GPU0..7)
```

## 2. 部署与启动

### 2.1 启动命令

相对原脚本的改动：
- IP：`192.168.1.100` 替换为 Node 1 的 Pod IP `10.154.14.180`。
- 去掉 `--all2all-backend deepep_low_latency`（原因见 6.3），改用默认的 `allgather_reducescatter`。
- 增加环境变量 `NCCL_IB_GID_INDEX=3`（原因见 6.1）和 `VLLM_ENGINE_READY_TIMEOUT_S=3600`（单节点实验已知默认的 600 秒超时不够，见 `vllm-dp8-ep-single-node.md` 第 2.4 节）。

```
# Node 1（server-1，主节点，处理请求）
NCCL_IB_GID_INDEX=3 VLLM_ENGINE_READY_TIMEOUT_S=3600 \
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-address 10.154.14.180 \
    --data-parallel-rpc-port 13345 \
    --api-server-count=8

# Node 2（server-2，headless）
NCCL_IB_GID_INDEX=3 VLLM_ENGINE_READY_TIMEOUT_S=3600 \
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-start-rank 8 \
    --data-parallel-address 10.154.14.180 \
    --data-parallel-rpc-port 13345 \
    --headless

# 验证（在 server-1 上）
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-ai/DeepSeek-V3-0324",
       "messages": [{"role": "user", "content": "Who are you?"}],
       "max_tokens": 64}'
```

- 启动顺序：先起 Node 1，约 20 秒后再起 Node 2。
- 验证：`/v1/completions` 和 `/v1/chat/completions` 在 server-1 上成功返回。

### 2.2 启动时间线（最终成功的那次）

| 阶段 | 时间 | 说明 |
|---|---|---|
| Node 1 启动 | 01:59:27 | |
| all2all 选定 | 02:00:30 | `Using AgRsAll2AllManager all2all manager` |
| 开启 EP | 02:00:32 | `[EP Rank 0/16] ... Local/global number of experts: 16/256`；Node 2 上 `[EP Rank 8/16] ... map: 0->128, 1->129, ...` |
| MoE 后端 | 02:00:32 | `Using FLASHINFER_TRTLLM Fp8 MoE backend` |
| 权重加载完成 | 02:03:45 | `Model loading took 58.16 GiB memory and 193.0 seconds`（Node 2：195.9 秒） |
| KV cache 就绪 | 02:05:40 | `Available KV cache memory: 158.58 GiB`；`GPU KV cache size: 2,423,008 tokens`；`Maximum concurrency for 163,840 tokens per request: 14.79x`（两个节点相同） |
| CUDA graph 抓取完成 | 02:12:28 | `Graph capturing finished in 254 secs`；`init engine ... took 526.06 s (compilation: 50.11 s)` |

从启动到可以服务约 13 分钟。

### 2.3 与单节点（DP=8 + EP=8）的对比

| | 单节点 DP=8 / EP=8 | 双节点 DP=16 / EP=16 |
|---|---|---|
| 每个 rank 的专家数 | 32 / 256 | 16 / 256 |
| 每卡权重 | 96.23 GiB | 58.16 GiB |
| 每卡 KV cache | 133.02 GiB | 158.58 GiB |
| 每个 rank 的 KV cache 容量 | 2,032,480 token | 2,423,008 token |
| 163,840 token 请求的最大并发 | 12.41× | 14.79× |
| 空闲时各 rank | 0% | 0% |

每卡权重的估算（按 `config.json`，FP8 按 1 字节/参数，embedding 和 lm_head 按 bf16，不含量化 scale、norm、router gate 和 MTP 层）：路由专家 653.9 B ÷ 16 ≈ 38.1 GiB，非专家部分（Attention 10.6 + shared expert 与前 3 层稠密 FFN 3.5 + embedding 与 lm_head 3.5）约 17.6 GiB，合计约 55.6 GiB，实测 58.16 GiB，差约 2.5 GiB，与单节点实验中的差距一致（差的部分没有逐项核对）。EP 变大后，专家部分每卡减半，复制的非专家部分不变，所以每卡省下的显存全部来自专家。

## 3. 进程结构（实测）

| | Node 1（server-1） | Node 2（server-2） |
|---|---|---|
| 主进程 `vllm serve` | 1 | 1 |
| APIServer | 8 | 0 |
| DPCoordinator | 1 | 0 |
| EngineCore | 8（DP0–DP7） | 8（DP8–DP15） |
| Worker | 8（`Worker_DP0_EP0` … `Worker_DP7_EP7`） | 8（`Worker_DP8_EP8` … `Worker_DP15_EP15`） |

（不含 `resource_tracker`。）

- Node 2 是 headless：没有 APIServer 和 DPCoordinator，只有引擎和 Worker；入口、分词、负载统计都在 Node 1。
- Node 2 的 Worker 的 `local_rank` 是 0–7，全局 rank 是 8–15（日志 `world_size=16 rank=8 local_rank=0`）。
- 与单节点一样，Worker 的 PPID 是对应的 EngineCore。`Worker_DPn_EPn` 的命名规则和 DP/EP 权重的区别见 `vllm-dp8-ep-single-node.md` 第 1.2 节。

## 4. 跨节点通信

### 4.1 通信分层总览：控制面走 TCP，MoE 数据面走 NVLink + RDMA

按通信的用途分层：

| 层次 | 通信内容 | 传输 | 证据 |
|---|---|---|---|
| 请求入口 | APIServer ↔ EngineCore（请求下发、token 回传） | ZMQ。Node 2 的 8 个 EngineCore 必须跨节点连回 Node 1，走 TCP（eth0）；Node 1 的 8 个 EngineCore 与 APIServer 同机，走 TCP 还是本机 IPC 没有核对 | 实测：Node 1 到 Node 2 有 518 条 TCP 连接，其中 APIServer 128、DPCoordinator 16（4.2） |
| 负载统计与 DP 协调 | EngineCore ↔ DPCoordinator（队列长度、wave 状态） | TCP（eth0） | 实测，同上 |
| 各 rank 之间的 DP 同步 | 判断是否还有未完成请求（约每 32 步一次，源码核对） | 日志写着异步调度下“DP 同步不走 NCCL”（`Disabling NCCL for DP synchronization when using async scheduling.`），具体通道没有核对，应是 CPU 侧的 TCP 通道，是推断 | 日志；通道推断 |
| Attention 计算 | 各 DP rank 只处理自己的请求，rank 之间不交换数据 | 无通信 | 机制（`vllm-dp8-ep-single-node.md` 第 1 节） |
| MoE 层 | all-gather / reduce-scatter（每步每个 MoE 层一次） | NCCL。同一节点内的 8 张卡之间应走 NVLink，跨节点部分走 RDMA（RoCE，NCCL 的 IB 传输） | 跨节点走 RDMA：推断，依据 RDMA write 请求计数与 eth0 字节数的对比（4.3），独立 NCCL 测试里选的是 `NET/IB` + `GDRDMA`（6.1）；节点内走 NVLink：按 NCCL 一般行为推断，没有抓取 |

要点：
- **“DP rank 之间独立”只限于 Attention 计算**。每一步的 MoE 层都要所有 rank 一起参与，没有请求的 rank 也要跑 dummy batch（5.1），所以 rank 之间在时间上是耦合的。
- **MoE 层的通信不是纯 RDMA**：16 个 rank 的集合通信里，节点内部分走 NVLink，只有跨节点部分走 RDMA。两者的分工是推断，没有用 `NCCL_DEBUG` 或抓包验证。
- **这也解释了为什么 `NCCL_IB_GID_INDEX` 必须设**：跨节点那部分 RDMA 如果选到了不可达的链路本地 GID，第一次集合通信就卡死（6.1）；而 TCP 控制面不受影响，所以卡死时 vLLM 不报错。最后一句是推断，没有专门验证。

### 4.2 控制面的 TCP 连接（实测）

- 主节点监听：`ss -ltnp` 显示 `0.0.0.0:8000` 只有一个 LISTEN socket，被 9 个进程同时持有（8 个 APIServer 加主进程），与单节点和稠密 DP 实验一致。`13345`（`--data-parallel-rpc-port`）没有处于监听状态，与稠密 DP 双节点实验一致，它只用于启动阶段的握手（现象观察，没有读源码核对）。
- 稳定运行后，从 Node 1 看，与 Node 2 的 IP 之间共有 518 条 ESTABLISHED 的 TCP 连接，按 Node 1 一侧的进程分：APIServer 128、DPCoordinator 16、EngineCore 72、Worker 302；从 Node 2 看同样是 518 条：EngineCore 216（= 128 + 16 + 72）、Worker 302。
  - 128 = 8 个 APIServer × 2 个端口 × 8 个远端引擎，16 = DPCoordinator 的 2 个端口 × 8 个远端引擎，与稠密 DP 双节点实验里“每个远端引擎对每个 APIServer / DPCoordinator 的 2 个端口各连一次”的结构一致。
  - EngineCore 之间的 72 条连接和 Worker 之间的 302 条连接的具体用途没有核对（可能是 torch.distributed / gloo / NCCL 的 TCP 通道等，不确定）。
- 这些 TCP 连接用的是 eth0（Pod 网络）；集合通信的大块数据走 RDMA（4.3）。

### 4.2.1 补充：APIServer 与 EngineCore 的对应关系（2026-10-11 补充，源码核对 + 在另一套 hybrid-lb 部署上实测）

本文的部署（内部 LB）当时没有直接统计过每个 APIServer 的连接，下面是补充。这套内部 LB 部署已不在运行，所以**内部 LB 部分只有源码核对，没有在本文的部署上重新实测**；本机通信的传输方式来自同镜像 hybrid-lb 部署的实测。

- **每个 APIServer 都管理全部 16 个 EngineCore**（本机 8 + 远端 8），是全连接，不是一一对应。依据（vLLM 0.31.0 `v1/engine/core_client.py`）：`num_ranks = dp_local_size if local_engines_only else dp_size`，注释写明纯内部 LB 下 “Client manages local+remote EngineCores”。这与 4.2 的数字吻合：128 = 8 个 APIServer × 2 个端口 × 8 个**远端**引擎；本机的 8 个引擎不出现在跨节点连接里。
- **APIServer 数量（8）与 EngineCore 数量（16）没有对应关系**：APIServer 只做分词、协议转换和流式输出（CPU 工作），数量按 CPU 吞吐定。
- **本机通信大概率是 unix socket 而不是 TCP**：在同镜像的 hybrid-lb 部署上实测，APIServer 与本机每个 EngineCore 之间是 2 条 unix socket（`ipc://`），没有 TCP。内部 LB 下本机部分应相同（推断，没有在本文部署上验证）。这回答了第 4.1 表里“Node 1 的 8 个 EngineCore 与 APIServer 同机，走 TCP 还是本机 IPC 没有核对”：应为 IPC。
- **选 rank 的逻辑**：由接到请求的 APIServer 自己决定，不经过协调者转发。对 16 个引擎各算一个分：`max(客户端数 × 本 APIServer 的 in-flight 数, waiting + running)`，有排队时再按 KV 使用率加罚；取最低分，并列时轮转（`DPLBAsyncMPClient.get_core_engine_for_request`）。
- **DPCoordinator 的作用**：收集 16 个引擎的 waiting / running 计数，最短每 100 ms 向所有 APIServer 广播一次；它不转发请求。请求仍由 APIServer 直接发给选中的引擎。
- **传输分工**：请求下发和结果回传，跨节点那一半走 eth0 的 ZMQ TCP（4.2 的 518 条连接），本机那一半走 unix socket；负载统计走协调者的 TCP。
- 对比：hybrid-lb 下每个 APIServer 只管理本节点的 8 个引擎，节点内 8 × 8 全连接，跨节点只剩到 DPCoordinator 的 1 条 TCP，见 `strategy3.md` 第 5.5 节。

### 4.3 数据面走 RDMA，eth0 上只有小消息（推断）

Pod 里没有 RDMA 的字节计数器（`/sys/class/infiniband/*/ports/1/counters` 不存在，也没有 `ethtool`）。改用 `rdma statistic show link mlx5_N/1` 里的 `rx_write_requests`（收到的 RDMA write 请求次数）作为 RDMA 流量的代理指标，同时记录 `eth0` 的 tx/rx 字节数。

| 场景（耗时） | eth0 tx 增量 | eth0 rx 增量 | RDMA 收到的 write 请求增量 |
|---|---|---|---|
| 单个请求 1500 token（43.4 s） | Node 1 34.4 MB / Node 2 35.0 MB | Node 1 34.7 MB / Node 2 34.7 MB | Node 1 20.79 M / Node 2 20.98 M |
| 64 并发 × 256 token（20.5 s） | Node 1 7.1 MB / Node 2 8.5 MB | Node 1 8.5 MB / Node 2 7.1 MB | 两个节点都是 15.65 M |
| 512 并发 × 256 token（16.0 s） | Node 1 7.8 MB / Node 2 12.2 MB | Node 1 12.2 MB / Node 2 7.8 MB | 两个节点都是 15.79 M |
| 2048 并发 × 256 token（29.4 s） | Node 1 7.9 MB / Node 2 21.9 MB | Node 1 21.9 MB / Node 2 7.9 MB | 两个节点都是 22.84 M |

- eth0 上的字节数很小（MB 量级），随生成的 token 数增长（2048 并发时 Node 2 发出 21.9 MB，Node 1 收到同样多，应是 Node 2 的引擎把生成的 token 回传给 Node 1 的 APIServer，是 ZMQ 小消息）；RDMA 的 write 请求次数则是千万量级，随运行时间和负载增长。所以集合通信的大块数据走 RDMA（RoCE），不走 eth0。
- 这是依据计数器的推断；没有直接测到 RDMA 的字节数，也没有对 vLLM 运行时的 NCCL 打开 `NCCL_DEBUG`。在独立的最小 NCCL 测试里，NCCL 选的传输是 `NET/IB` + `GDRDMA`（6.1）。
- 8 个 RoCE 设备上共有 152 个 QP（每个设备 19 个），运行期间不变。

### 4.4 RDMA 错误计数（遗留问题）

`packet_seq_err` 和 `out_of_sequence` 在负载下持续增长（两个节点的累计值：Node 1 为 18572 / 13371，Node 2 为 13267 / 18657）。这两项是序列错误，会触发重传，不是连接失败；`req_transport_retries_exceeded`（5）和 `local_ack_timeout_err`（50）是 6.1 的失败测试留下的，运行期间没有变化。

按汇总指标（`packet_seq_err`、`local_ack_timeout_err`、`req_transport_retries_exceeded` 之和），Node 1 在三轮并发测试期间共增加约 4.2 千（14421 → 18627，含单个请求之后的部分），同期收到约 7500 万次 RDMA write 请求，占比很小；但说明跨节点链路上存在少量丢包或乱序，没有进一步排查（例如是否与 PFC / ECN 配置有关）。

## 5. 运行行为（实测）

### 5.1 一个请求让两个节点的 16 张 GPU 都忙起来

| 状态 | Node 1 的 8 张 GPU | Node 2 的 8 张 GPU |
|---|---|---|
| 空闲（无请求） | 全部 0% | 全部 0% |
| 只有 1 个请求在生成（max_tokens=1500，耗时 43.4 s），3 次采样 | 93–100% | 92–100%（第一次采样时 GPU0 为 0%，之后恢复） |
| 请求结束后 | 全部 0% | 全部 0% |

这与单节点 DP+EP 的现象一致：各 rank 通过 MoE 层的集合通信耦合，没有请求的 rank 用 dummy batch 陪跑（机制见 `vllm-dp8-ep-single-node.md` 第 6.1 节和本文 5.1.1，与 all2all 后端无关）；现在这些空闲 rank 在另一台机器上。GPU 利用率不能反映有效负载。

### 5.1.1 补充：陪跑的机制（2026-10-11 补充，源码核对）

5.1 里“没有请求的 rank 也在忙”此前只引用了单节点文档的结论。下面是读源码得到的机制。**注意：源码来自后来的 `vllm/vllm-openai:latest`（vLLM 0.31.0）镜像，本文实验用的是 0.19.2rc1，两者的细节可能不同，没有对本文的版本核对。** 本文的实测现象（16 张卡同时升到 92–100%）与该机制一致。

**为什么要陪跑**：16 个 rank 在同一个 EP 组里，每个 MoE 层都有 16 个 rank 参与的集合通信，任何一个 rank 缺席，其余 rank 都会卡在集合通信上。所以只要有一个 rank 在处理请求，其余 15 个即使没有请求，也要每一步都同步地跑一次前向，这个空前向就是 dummy batch。

**机制（`vllm/v1/engine/core.py`，`run_busy_loop`）**：
1. **全空闲时**：`engines_running = False`，引擎循环阻塞在输入队列上，不跑前向，GPU 利用率 0%（与 5.1 的实测一致）。
2. **来了第一个请求**：被选中的 EngineCore 通知 DPCoordinator，协调者向**全部 16 个**引擎广播 `START_DP_WAVE`，各引擎把 `engines_running` 置为 True，进入同一个 wave。本文是内部 LB，协调者在 Node 1，Node 2（headless）的 8 个引擎也连到它（4.2 里的 16 条 DPCoordinator 连接），所以 wave 跨两个节点。
3. **每一步**：每个引擎先 `_process_engine_step()`；本步没有要执行的请求、而 wave 还在进行时，就 `execute_dummy_batch()`，跑一次空前向，让 MoE 层的集合通信凑齐 16 个 rank。
4. **何时结束**：每隔 `dp_sync_interval` 步（`step_counter`）做一次 16 个 rank 的 all-reduce，判断是否还有 rank 有未完成请求（`_has_global_unfinished_reqs`）。全部完成后 `engines_running = False`，wave 计数加一，所有引擎同时暂停，GPU 回到 0%。这与 4.1 表里的“约每 32 步一次”的 DP 同步是同一件事（具体间隔以 `dp_sync_interval` 为准，本文没有核对 0.19.2rc1 的值）。
5. **暂停协议**：暂停也是两阶段（`pending_pause`）：先在步进中等所有 rank 都置位，再一起停。

**与 all2all 后端无关**：陪跑由 DP 的 wave 机制保证，不管用 `allgather_reducescatter` 还是 `deepep_low_latency`，空闲 rank 都要参与。

**没有验证的**：dummy batch 的开销（空前向里的 token 数、耗时）没有测，日志里也没有看到 dummy 的直接记录；GPU 利用率不能反映有效负载。hybrid-lb 下的同一机制见 `strategy3.md` 第 5.6 节。

### 5.2 请求在 16 个 DP rank 间的分配

读取 Node 1 `/metrics` 中按 `engine` 标签区分的 `vllm:request_success_total{finished_reason="length"}`：

| engine | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 完成请求数 | 164 | 163 | 167 | 163 | 164 | 163 | 164 | 162 | 163 | 161 | 163 | 162 | 165 | 168 | 168 | 166 | 2626 |

2626 = 之前的 2 个请求 + 64 + 512 + 2048，没有丢失。分配比较均匀（161–168），engine 0–7 在 Node 1，8–15 在 Node 2，两个节点之间也均匀。

### 5.3 吞吐（粗测，不是基准测试）

| 并发 | 每请求生成 | 总生成 token | 耗时 | 吞吐 | 单节点 DP=8 对照 |
|---|---|---|---|---|---|
| 1 | 1500 | 1500 | 43.4 s | 约 34.6 tok/s | 约 60 tok/s |
| 64 | 256 | 16384 | 20.5 s | 799 tok/s | 1102 tok/s |
| 512 | 256 | 131072 | 16.0 s | 8203 tok/s | 8070 tok/s |
| 2048 | 256 | 524288 | 29.4 s | 17808 tok/s | 没有测 |

条件：`temperature=0`、`ignore_eos=true`、prompt 约 10 个 token（极短）、每档只跑了一次、没有预热对比；不代表真实负载下的性能，也没有测延迟分布。

观察：
- 低并发时，双节点比单节点慢（单流约 34.6 vs 60 tok/s，即单步约 29 ms vs 17 ms）。每个 MoE 层的 all-gather 和 reduce-scatter 都要跨节点，延迟比节点内高，低并发时单步时间被通信占满（推断）。
- 512 并发时两者接近（8203 vs 8070 tok/s），没有因为 GPU 翻倍而提升；2048 并发时达到 17808 tok/s，但单节点没有测这个并发，所以不能据此说“扩展效率”。
- 没有分析 512 并发时没有增益的具体原因（可能是通信开销，也可能是负载不够大，没有验证）。

## 6. 排障过程（本实验遇到的三次失败）

按发生顺序记录，方便以后复现和排查。

### 6.1 第一次：16 个 rank 的 NCCL 初始化卡死（缺少 `NCCL_IB_GID_INDEX`）

**现象**：按原脚本（仅改 IP）启动，两个节点都组成了 16 个 rank（日志 `world_size=16 rank=0..15`），但日志停在 `vLLM is using nccl==2.28.9` 之后，10 多分钟没有任何新输出，也没有报错；GPU 显存只有约 1.8 GB，权重没有开始加载；Worker 的 CPU 占用约 100%。单节点同一阶段只需十几秒。

**定位**：
- 用 `py-spy dump` 看 Worker 的 Python 栈，卡在 `vllm/distributed/device_communicators/pynccl.py:143` 的 `torch.cuda.synchronize()`，即创建 NCCL communicator 后的预热 all-reduce 一直没有完成。
- 在 vLLM 之外写了一个最小的 2 进程 NCCL 测试（每个节点 1 个进程，`all_reduce`）复现：设置 `NCCL_DEBUG=INFO` 后，NCCL 选的传输是 `NET/IB`（8 个 RoCE 设备，`via NET/IB/0/GDRDMA`），随后报错：

  ```
  NCCL WARN NET/IB: Got completion from peer 10.154.15.93<53470> with status=IBV_WC_RETRY_EXC_ERR(12) ...
  localGid fe80::ac3a:e2ff:fecf:d75e remoteGids fe80::ac3a:e2ff:fed1:cfea ...
  ```

  本端和对端用的都是 `fe80::...` 的链路本地 GID，跨节点不可达，所以重传超限。
- 查 RoCE 设备的 GID 表（`/sys/class/infiniband/mlx5_0/ports/1/gids/`）：index 0 / 1 是 `fe80::...`（分别为 RoCE v1 / v2），index 2 / 3 是 `fd02:0:400:527::1`（分别为 RoCE v1 / v2），只有 index 3 是可路由的 RoCE v2 地址。

**处理**：设置 `NCCL_IB_GID_INDEX=3`。同一个最小测试变为成功（`all_reduce ok value=3.0`，即 1+2）。之后 vLLM 的 16 个 rank 初始化通过。

注意：这类问题表现为静默卡死，不是报错；如果只看 vLLM 日志，很难判断原因。

### 6.2 第二次：`deepep_low_latency` 在 NVSHMEM 初始化时卡死

**背景：NVSHMEM 是什么**：NVIDIA 的 GPU 版 OpenSHMEM 通信库，提供跨 GPU 的对称内存和单边（one-sided）通信。
- 每个 PE（通常对应一张 GPU）在初始化时分配同样布局的一块显存（对称堆）；一个 PE 可以直接对另一个 PE 的对称内存做 put / get / 原子操作，对方不需要调用对应的 recv。
- 通信可以直接在 GPU 内核里发起，不必回到 CPU，因此延迟低，也可以把计算和通信写进同一个 kernel。
- 节点内走 NVLink / P2P；跨节点走 InfiniBand / RoCE，可以由 GPU 直接发起 RDMA（IBGDA），也可以由 CPU 上的代理线程代发。
- 与 NCCL 的区别：NCCL 是集合通信库（all-reduce、all-gather、all-to-all 等），由 CPU 侧发起，各 rank 要同时调用；NVSHMEM 是单边、细粒度的，适合 token 级别的不规则通信，例如 MoE 里按路由把 token 发给指定专家。
- 与本实验的关系：DeepEP 的节点间通信（`deepep_low_latency`）建在 NVSHMEM 之上，所以它的初始化（`deep_ep/buffer.py` 里的 NVSHMEM init）会卡在这里。这一点依据的是卡住的位置和一般资料，没有读 DeepEP 源码确认。默认的 `allgather_reducescatter` 后端只用 NCCL，不涉及 NVSHMEM。
- 运行日志里有两条警告：GDRCopy 不可用，IBGDA 回退为 CPU 代理，说明 GPU 直接发起 RDMA 的路径没有启用，原因没有排查。

**现象**：加上 `NCCL_IB_GID_INDEX=3` 后，按原脚本（含 `--all2all-backend deepep_low_latency`）启动，权重加载成功（约 178 秒，日志 `Using DeepEPLLAll2AllManager all2all manager`），之后 Worker 卡在 `deep_ep/buffer.py:133`（`self.runtime.sync(...)`，即 NVSHMEM 初始化），17 分钟没有进展。

**定位**：写了独立的 2 进程 DeepEP 测试（每个节点 1 个进程，创建 `Buffer(low_latency_mode=True)`），设置 `NVSHMEM_DEBUG=INFO`：
- 日志里有 `NVSHMEM_IB_ADDR_FAMILY set by environment to AF_INET`：NVSHMEM 默认按 IPv4 地址族找 GID，而这些 RoCE 网卡上只有 IPv6 的 GID，找不到可用的。

四组对照（每组跑一次）：

| 组 | 环境变量 | 结果 |
|---|---|---|
| A | 只有 `NCCL_IB_GID_INDEX=3` | 超时卡死（`Buffer` 没有建成） |
| B | 另加 `NVSHMEM_IB_ADDR_FAMILY=AF_INET6` 和 `NVSHMEM_IB_GID_INDEX=3` | 成功，`DeepEP Buffer OK in 3.4s` |
| C | 另加 `NVSHMEM_IB_GID_INDEX=3` | 成功，3.3 s |
| D | 另加 `NVSHMEM_IB_ADDR_FAMILY=AF_INET6` | 成功，3.3 s |

即 `NVSHMEM_IB_GID_INDEX=3` 或 `NVSHMEM_IB_ADDR_FAMILY=AF_INET6` 任一个就足够。

**NVSHMEM 不读 NCCL 的环境变量，要单独设置**：NVSHMEM 有自己独立的传输层和配置，与 NCCL 各自初始化、各自建立 RDMA 连接，所以同类问题要各修一次。依据：
- 6.1 已设 `NCCL_IB_GID_INDEX=3`、NCCL 部分通过之后，NVSHMEM 初始化仍然卡死（组 A）；加上 NVSHMEM 自己的变量才成功（组 B / C / D）。说明 NCCL 的设置对 NVSHMEM 没有影响。
- NVSHMEM 日志写着 `NVSHMEM_IB_ADDR_FAMILY set by environment to AF_INET`：这是它自己的默认地址族，与 NCCL 无关。

| 作用 | NCCL | NVSHMEM |
|---|---|---|
| 选 RoCE 的 GID 索引 | `NCCL_IB_GID_INDEX=3` | `NVSHMEM_IB_GID_INDEX=3` |
| 地址族 | 无此项 | `NVSHMEM_IB_ADDR_FAMILY=AF_INET6` |

- `deepep_low_latency` 这条路径同时用到两者：进程组和部分集合通信走 NCCL，DeepEP 的 dispatch / combine 走 NVSHMEM，所以用它时两组变量都要设。各自具体管哪部分，没有读 DeepEP 和 vLLM 的代码确认，是依据现象和一般了解。
- 默认的 `allgather_reducescatter` 后端只用 NCCL，只需要 `NCCL_IB_GID_INDEX=3`。

成功的测试里有若干警告，说明 IBGDA 走的是回退路径，但不影响建立：
- `GDRCopy open call failed, disabling GDRCopy`（Pod 里没有 `/dev/gdrdrv`）；
- `cudaHostRegister with IoMemory failed with error=800`、`ibgda_nic_mem_gpu_map failed. We may need to use the CPU fallback path`、`ibgda_alloc_and_map_qp_uar with GPU as handler failed`。

### 6.3 第三次：`deepep_low_latency` 的内核在 B300 上不可用（最终放弃该后端）

**现象**：加上 `NCCL_IB_GID_INDEX=3` 和 `NVSHMEM_IB_GID_INDEX=3` 重新启动，NVSHMEM 初始化通过，但在内存 profile 阶段的第一次 MoE 前向，所有 Worker 都报错退出：

```
RuntimeError: Failed: CUDA error /tmp/ep_kernels_workspace/DeepEP/csrc/kernels/internode_ll.cu:553 'named symbol not found'
```

调用栈：`gpu_worker.determine_available_memory → profile_run → _dummy_run → deepseek_v2.forward → fp8.apply → modular_kernel._prepare`（DeepEP 的 dispatch）。

**原因（有强证据，没有通过重新编译验证）**：
- GPU：`NVIDIA B300 SXM6`，算力 10.3（`torch.cuda.get_device_capability(0)` 返回 `(10, 3)`）。
- `cuobjdump --list-elf deep_ep_cpp...so` 只有两个 cubin：`sm_100a` 和 `sm_90a`；`cuobjdump --list-ptx` 显示没有 PTX。
- 推断：二进制里没有覆盖 sm_103 的内核，所以 DeepEP 的低延迟内核找不到可用的符号。镜像里 PyTorch 的 arch 列表是 `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120`，也没有 `sm_103`。
- “没有 PTX”不是全部原因：按我对 CUDA 的一般了解（本机没有验证），带 `a` 后缀的架构特定目标（`sm_100a`）不向前兼容，即使附带了对应的 PTX 也不一定能在 sm_103 上 JIT 运行；要覆盖 sm_103，需要直接编译 `sm_103`，或编译家族级目标（如 `sm_100f`）。
- 与 GPU 驱动版本的关系：驱动 590.48.01，`nvidia-smi` 显示支持的 CUDA 为 13.1；镜像里的 `nvcc` 和 torch 是 CUDA 13.0，驱动不低于镜像运行时。驱动能识别这批 B300，torch、NCCL、FlashInfer 的 MoE 内核都正常运行。torch 没有 `sm_103` 却能工作，说明普通的 `sm_100` cubin 可以在 10.3 上运行（家族内兼容），而 DeepEP 里带 `a` 后缀的 `sm_100a` 不行（与上一条的一般了解一致，没有单独验证）。报错 `named symbol not found` 指向二进制没有与当前 GPU 匹配的内核，驱动版本无法补上；驱动太旧通常是别的错误，也会影响整个镜像。因此证据不支持驱动是原因，但没有换驱动版本做对照，不能说证明了无关。
- 影响范围：只有镜像里的 DeepEP 这一个组件。同一个镜像里的 vLLM、NCCL、FlashInfer 的 MoE 内核在 B300 上都正常跑通了（本实验成功运行），所以不能说镜像整体有问题。
- 性质：这更像镜像里 DeepEP 的构建配置缺口（大概按 B200 的 sm_100 和 H100 的 sm_90 编译，没有覆盖 B300 的 sm_103），不是部署配置的错误。这一点是推测，没有查镜像的构建记录。
- 镜像里没有 DeepEP 源码（`/tmp/ep_kernels_workspace` 不存在）；`nvcc` 存在，重新编译需要先取得源码。

**处理**：本实验改用默认后端 `allgather_reducescatter`。如果要在这批机器上用 DeepEP，有两个办法，都没有尝试：
- 用能生成 sm_103（或家族级目标）内核的参数重新编译 DeepEP，每个节点都要有；需要先取得源码。
- 换一个已经包含 sm_103 版 DeepEP 的镜像（如果有的话）。

### 6.4 清理现场

- Worker 进程名为 `VLLM::Worker_DPn_EPn` 等，`pkill -f "vllm serve"` 只会杀掉主进程，Worker 会继续占着 GPU 和端口（实测 Worker 以约 100% CPU 空转）。可用 `ps -eo pid,cmd | grep "[V]LLM::" | awk '{print $1}' | xargs kill -9`（方括号写法避免匹配到自己），并确认 `nvidia-smi` 显存释放。pod 内会留下 `<defunct>` 僵尸进程（PID 1 是 `sleep infinity`），不影响重启。
- 旧日志保留在各 Pod 的 `/root/work/` 下：`vllm-node{1,2}.nccl-gid-hang.log`（6.1）、`vllm-node{1,2}.nvshmem-hang.log`（6.2）、`vllm-node{1,2}.deepep-ll-sm103-fail.log`（6.3）；最终成功的日志是 `vllm-node{1,2}.log`。

## 7. 与单节点 / 稠密 DP 双节点的区别

| | 稠密 DP 双节点（Qwen3-8B） | 单节点 DP=8 + EP=8 | 本实验：双节点 DP=16 + EP=16 |
|---|---|---|---|
| rank 之间的集合通信 | 无（组大小为 1） | 节点内 NCCL（NVLink，推论） | 跨节点 NCCL，走 RDMA（RoCE） |
| 跨节点流量 | 只有 ZMQ 小消息，TCP | 无 | TCP 小消息 + RDMA 集合通信 |
| 是否需要 RDMA | 不需要 | 不需要 | 需要；且必须正确设置 GID（6.1） |
| 一个请求使用的 GPU | 1 个 rank 对应的 GPU | 全部 8 张 | 全部 16 张（两个节点） |
| 空闲 rank | 真正空闲 | dummy batch 陪跑 | dummy batch 陪跑，且跨节点 |

## 8. 结论

- 双节点 DP=16 + EP=16 可以部署 DeepSeek-V3-0324（FP8），请求在 16 个 rank 间分配均匀，两个节点的 GPU 都参与。
- 在这批 B300 机器上，`deepep_low_latency` 因为镜像里的 DeepEP 没有 sm_103 内核而不能用（推断），要用需要重新编译或换镜像。默认的 `allgather_reducescatter` 可以工作，但它把全部 token 广播给所有 rank，通信量随 rank 数增长（机制见 `vllm-dp8-ep-single-node.md` 第 5 节），这里每层的集合通信都要跨节点。
- 这批机器上跨节点 NCCL / NVSHMEM 必须显式设置 GID（`NCCL_IB_GID_INDEX=3`、`NVSHMEM_IB_GID_INDEX=3`），否则会静默卡死；两个库的环境变量互不影响。
- 通信分层：控制面走 TCP，MoE 数据面节点内走 NVLink、跨节点走 RDMA（后者是推断）。
- 低并发下，双节点比单节点慢；本实验没有证明双节点在高并发下的扩展效率。

## 9. 未验证项

**DeepEP 与镜像**
- DeepEP 在 sm_103 上不可用的结论，没有通过重新编译 DeepEP 来验证，只是依据算力 10.3、cubin 列表（`sm_100a` / `sm_90a`）和没有 PTX 做出的判断。
- “`a` 后缀目标不向前兼容、需要 `sm_103` 或家族级目标”只是对 CUDA 的一般了解，没有在本机验证；镜像里 DeepEP 为什么没有覆盖 sm_103（构建配置），也没有查构建记录。
- 驱动版本与 DeepEP 内核问题无关，只是依据上述现象的判断，没有换驱动对照；重新编译 sm_103 目标所需的最低 CUDA 版本（我记得是 12.9 以上，镜像里的 13.0 应该满足）没有查证。
- `deepep_low_latency` 在重新编译后的性能，以及与 `allgather_reducescatter` 的对比。
- DeepEP 内部如何调用 NVSHMEM（6.2 的背景说明依据报错位置和一般资料）；GDRCopy 不可用、IBGDA 回退为 CPU 代理的原因，以及它对 `deepep_low_latency` 性能的影响。
- 6.1 的 NCCL 对照（默认 vs `NCCL_IB_GID_INDEX=3`）和 6.2 的四组 NVSHMEM 对照都只各跑了一次。

**通信**
- 集合通信的大块数据走 RDMA，是依据 RDMA write 请求计数和 eth0 字节数的推断，没有直接测到 RDMA 字节数，也没有在 vLLM 运行时打开 `NCCL_DEBUG` 确认。
- 4.1：节点内的 MoE 通信走 NVLink、跨节点走 RDMA 的分工（推断）；Node 1 上 EngineCore 与 APIServer 之间走 TCP 还是本机 IPC；异步调度下 DP 同步实际使用的通道。
- 4.2 中 EngineCore 之间 72 条、Worker 之间 302 条 TCP 连接的用途。
- 跨节点链路上 `packet_seq_err` / `out_of_sequence` 的原因（丢包、乱序、PFC / ECN 配置等）。

**运行与配置**
- 5.3 的吞吐是未经预热、单次、极短 prompt 的粗测；512 并发时没有比单节点更快的原因没有分析；2048 并发没有单节点对照。
- 每个节点 Worker 的 EP rank 与专家区间：只核对了 EP0（0->0）和 EP8（0->128）的映射，其余是按 `linear` 推的。
- `vllm-dp8-ep-single-node.md` 里的未验证项（monolithic 与 modular 路径、dummy 前向内部实现等）在本实验里同样没有验证。
- 返回文本的字节级 BPE 乱码问题（见 `vllm-dp8-ep-single-node.md` 第 7 节）：本实验的验证请求只核对了 `finish_reason` 和 token 数，没有检查返回文本的内容，所以不知道在这个环境里是否同样出现。

## 10. 运维备注

- 服务当前仍在 server-1 / server-2 里运行（占满两个节点共 16 张 GPU）；server-0 上的单节点实验服务也仍在运行（占满它自己的 8 张 GPU，与本实验互不影响）。
- 日志、脚本在各 Pod 内（Pod 重建后会丢失）：`/root/work/vllm-node{1,2}*.log`、`/root/work/conc.py`（`python3 conc.py <并发数> <max_tokens>`）、`/root/work/netcnt.sh`（读取 eth0 字节和 RDMA 计数）、`/root/work/nccl_test.py`、`/root/work/deepep_test.py`（6.1、6.2 里的最小测试）。
- 停止服务的方法见 6.4。
- 后续可做：重新编译带 sm_103 的 DeepEP 并与默认后端对比；用 `--enable-eplb` 的 `log_balancedness` 看专家负载；排查 RDMA 序列错误；用 `NCCL_DEBUG=INFO` 跑 vLLM 确认传输路径。
