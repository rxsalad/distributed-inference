# 策略二：Multi-Node TP + Pipeline Parallel（TP8 × PP2，每阶段内 EP8）实测

DeepSeek-V3（FP8）在 2 个 B300 节点（每节点 8 卡，共 16 张 GPU）上作为**一个** vLLM 实例运行，对应 recipes 页面的 Multi-Node TP + Pipeline Parallel 方案：节点内 TP8，节点之间 PP2。

- 实验日期：2026-10-11/12；镜像 `vllm/vllm-openai:latest`（vLLM 0.31.0），NCCL 2.30.7。
- 相关文件：`recipe.md`（页面内容整理，本方案见第 5 节）、`tppp-server.yaml`（部署文件）；对照见 `strategy1.md`（TP16）、`strategy3.md`（DEP）。
- 标注约定：**实测**＝本集群命令输出或日志；**源码核对**＝读了镜像内 vLLM 0.31.0 源码；**推断**＝由日志、机制得出，未直接验证。
- 本文**不含性能数据**（按要求不做性能测试）。

## 1. 摘要

| 方面 | 结论 |
|---|---|
| 能否部署 | 能，页面命令（加本集群需要的环境变量）直接可用；**不需要** `--disable-custom-all-reduce`（实测没有加也顺利启动；TP 组都在节点内，没有策略一那种跨节点 MNNVL rendezvous，原因为推断） |
| 启动耗时 | vLLM 启动 01:02:15 → 就绪约 01:16:30，约 14 分钟；`init engine` 737.67 s（TP16 912.6 s，DEP 1137.96 s） |
| 层怎么分 | DeepSeek-V3 的 61 个隐藏层不均匀切成 **[31, 30]**：节点 0（PP 阶段 0）31 层，节点 1（阶段 1）30 层（日志 `Hidden layers were unevenly partitioned: [31,30]`） |
| EP 范围 | **只在每个阶段内的 8 张卡之间**：`[EP Rank 0/8]`，每卡 32/256 个专家，两个阶段各自有完整的 256 个专家（对应各自的层）。MoE 通信全在节点内 |
| 每卡权重 | 阶段 0：38.71 GiB；阶段 1：41.17 GiB（TP16：40.47；DEP：58.02） |
| KV cache | 阶段 0 每卡 199.71 GiB，阶段 1 每卡 195.69 GiB；整个实例 **6,004,640 tokens**（TP16：2,959,680，约 2.03 倍），最大并发（16.4 万 token/请求）36.65×。每个阶段只存自己那些层的 KV，所以同样的 KV 显存能存更多 token（第 6.2 节） |
| 进程 | 节点 0：主进程（含 API server）+ 1 个 EngineCore + 8 个 Worker；节点 1（headless）：主进程 + 8 个 Worker，没有 EngineCore |
| 跨节点数据面 | 只有阶段之间的 NCCL 点对点（RDMA）：**每个 decode 步每个节点收到 32 个 RDMA write 请求，8 块网卡各 4 个**，与输出 token 数严格成正比，空闲为 0（第 5.3 节） |
| 空闲 rank | **没有陪跑**：只有一个调度器、一个批次；两个阶段轮流处理同一批（第 5.4 节） |
| 测试 | 对话、单请求、16 个并发、20k 和 49k token 的长 prompt 全部成功（第 8 节） |
| 副作用 | 流水线气泡：单请求时两个阶段是串行的；GPU 利用率看不出来（第 8.2、9 节） |

## 2. 方案总览

命令：`--tensor-parallel-size 8 --pipeline-parallel-size 2 --enable-expert-parallel --nnodes 2`。总 GPU 数 = TP × PP = 16。

| | 怎么分 | 通信 |
|---|---|---|
| PP（跨节点） | 61 层切成 [31, 30]：节点 0 做前 31 层，节点 1 做后 30 层 | 阶段交界处传一次激活值，NCCL 点对点，走 RoCE |
| TP8（节点内） | 阶段内 8 张卡共同算同一个 token 的 Attention 和稠密 FFN，按 TP8 切 | 每层 all-reduce，节点内 NVLink（推断） |
| EP8（节点内） | `--enable-expert-parallel`：阶段内 256 个专家分到 8 张卡，每卡 32 个（`linear` 放置） | MoE 使用 `MoEPrepareAndFinalizeNoDPEPMonolithic`，与策略一同一条路径，无 all-to-all（日志 + 策略一的源码核对） |

和另外两个策略比：PP 方案里一个 token 只在**一个阶段的 8 张卡**内做 TP/EP 通信，跨节点只传阶段之间的激活值；TP16（策略一）每层要两次跨节点 all-reduce；DEP（策略三）每个 MoE 层要 16 个 rank 的 all-gather / reduce-scatter。

## 3. 部署

**文件**：`tppp-server.yaml`，基于 `tep-server.yaml` 修改。
- StatefulSet `tppp`，2 副本，`podManagementPolicy: Parallel`，反亲和保证两个 Pod 落在不同节点；headless Service `tppp-headless`（`publishNotReadyAddresses: true`）；Service `tppp-api` 只指向 `tppp-0`（`tppp-1` 是 headless，不提供 API）。
- 每个 Pod 占 8 张 GPU 和 8 个 RoCE 网卡。
- 取主节点 IP 用“循环直到非空”的写法（策略三修复的 DNS 竞态）。

**启动命令**（节点 1 另加 `--node-rank 1 --headless`）：
```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --trust-remote-code --enable-expert-parallel \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 0 --master-addr 10.154.1.190 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v3 --reasoning-parser deepseek_v3
```
环境变量：`GLOO_SOCKET_IFNAME=eth0`、`NCCL_SOCKET_IFNAME=eth0`、`NCCL_IB_GID_INDEX=3`、`VLLM_ENGINE_READY_TIMEOUT_S=3600`。

**与页面命令的差异**：加了上述环境变量；去掉了 `--chat-template`（镜像里的路径未验证，工具调用没有测）；**没有** `--disable-custom-all-reduce`。

**与策略一 yaml 的差异**：`--tensor-parallel-size 8 --pipeline-parallel-size 2`，去掉 `--disable-custom-all-reduce`。

**实际落点**：

| Pod | 节点 | Pod IP | 角色 |
|---|---|---|---|
| tppp-0 | rs-b300x8-gpu-pool-3o1v8g | 10.154.1.190 | 阶段 0（31 层），API server，EngineCore |
| tppp-1 | rs-b300x8-gpu-pool-3o1v8w | 10.154.1.14 | 阶段 1（30 层），headless |

部署前先 `kubectl delete -f tep-server.yaml`（释放策略一占用的这两个节点）；`dep`（策略三）不受影响。

## 4. 启动过程

### 4.1 时间线（HF 权重已在节点缓存里）

| 时间 | 事件 |
|---|---|
| 01:02:15 | vLLM 启动，解析参数 |
| 01:02:42 | EngineCore 初始化，`world_size=16, local_world_size=8` |
| 01:02:54–59 | 16 个 Worker 建立 NCCL（`distributed_init_method=tcp://10.154.1.190:29501`）；日志 `rank 0 … PP rank 0 … EP rank 0`、`rank 8 … PP rank 1 … EP rank 0` |
| 01:03:18 | 日志：隐藏层不均匀切分 `[31,30]` |
| 01:03:54 | 加载权重 19.55 s；模型加载共 52 s |
| 01:04:20–50 | torch.compile 38.8 s（编译范围 (1,292)、(293,16384)） |
| 01:04:50–01:09:56 | 内存 profile 与第一轮 CUDA graph 捕获：阶段 0 用 278 s，阶段 1 用 216 s；其间日志反复出现 `No available shared memory broadcast block found in 60 seconds`（计算耗时导致的提示，不是错误） |
| 01:09:56 | 阶段 0 `Available KV cache memory: 199.71 GiB`，`GPU KV cache size: 6,004,640 tokens` |
| 01:16:26 | 第二轮 CUDA graph 捕获结束（阶段 0 124 s） |
| 01:16:28 | `init engine … took 737.67 s (compilation: 39.65 s)`；随后 `Application startup complete` |

### 4.2 故障

本次没有遇到故障：没有 MNNVL 卡死，没有 DNS 竞态。策略一的卡死发生在跨节点的 16 卡 TP 组里；这里 TP 组只有 8 张卡且都在一个节点内（推断是没有触发的原因，没有对照实验）。

## 5. 运行时结构

### 5.1 进程（实测）

| | tppp-0（阶段 0） | tppp-1（阶段 1，headless） |
|---|---|---|
| 主进程 `vllm` | 1（里面是 API server，日志 `APIServer pid=1`） | 1 |
| EngineCore | **1** | **0** |
| Worker | 8：`Worker_PP0_TP0_EP0` … `Worker_PP0_TP7_EP7` | 8：`Worker_PP1_TP0_EP0` … `Worker_PP1_TP7_EP7` |

- 只有一个调度器（节点 0 的 EngineCore）管 16 个 Worker，和策略一相同；策略三是每个 DP rank 一个 EngineCore。
- Worker 名称里的 `TPn_EPn` 是**阶段内**的编号（0–7），`PP0/PP1` 区分阶段；EP 大小是 8，不是 16。

### 5.2 控制面

- 主节点用 `--master-addr` 做 NCCL 和 Gloo 的初始化地址（`tcp://10.154.1.190:29501`）。
- 调度结果从节点 0 的 EngineCore 广播给两个节点的 Worker；跨节点部分走 eth0 的 TCP。实测一个 256 token 的请求，tppp-0 的 eth0 发出 5.6 MB、收到 1.2 MB（tppp-1 反过来），应是调度输出和采样结果（推断）。
- 引擎的批次队列长度等于 PP 大小：`max_concurrent_batches` 返回 `pp_size`（2），用来让两个阶段同时各处理一个批次（源码核对，`config/vllm.py`）。V1 模型运行器在 PP 下不完全支持异步调度（同一处源码注释）。

### 5.3 数据面（实测）

跨节点的 GPU 数据只有阶段之间的 NCCL 点对点。测量方法：`rdma statistic show link` 的 `rx_write_requests` 在请求前后的差值，两个节点各自统计。

| 条件 | 每个节点的增量 |
|---|---|
| 空闲 15 s | 0 |
| 单请求，输出 16 / 64 / 256 / 1024 token | 512 / 2,048 / 8,192 / 32,768 |

- **严格线性：每个 decode 步每个节点 32 个 write 请求**，与 token 数成正比，可重复。两个节点增量相同。
- 单请求 256 token 时按网卡拆：tppp-0 和 tppp-1 各自 `mlx5_0/2/4/6/8/10/12/14` 8 块网卡各 1,024（= 4 × 256 步）。即 **8 条并行的点对点流**，阶段 0 的第 i 个 TP rank 对应阶段 1 的第 i 个 TP rank，各用自己的网卡（推断：每对 rank 一条流，由 8 网卡均分得出）。
- 两个节点计数相同（不只是阶段 0 → 阶段 1 一个方向）的原因没有核对；NCCL 的点对点用反向的写入做流控（credit），可能是原因（推断）。
- 计数是**请求个数，不是字节数**，没有测每个请求的大小。
- eth0 上只有 MB 量级的小消息（5.2）。

### 5.4 没有陪跑

PP 只有一个调度器和一个批次，批次依次流过两个阶段：没有 DP rank，所以没有“没有请求的 rank 为了凑集合通信而空跑”的 dummy batch（对比 `dummy-batch-why.md`）。代价是另一种：**流水线气泡**（第 9 节）。

## 6. 模型切分与显存

### 6.1 每卡显存（日志）

| | 阶段 0 | 阶段 1 |
|---|---|---|
| 层数 | 31 | 30 |
| 模型加载 | 38.71 GiB | 41.17 GiB |
| 权重 + 非 torch | 39.74 GiB | 43.17 GiB |
| 峰值激活预留 | 6.82 GiB | 7.42 GiB |
| CUDA graph | 1.2 GiB | 1.5 GiB |
| KV cache | 199.71 GiB | 195.69 GiB |
| 稳态 `nvidia-smi` | 250,262 MiB | 247,500 MiB |

- 阶段 1 的权重更大，尽管层数少一层：阶段 0 的前 3 层是稠密 FFN（比 MoE 层小），阶段 1 还有 `lm_head`（推断，没有逐项核对）。
- 设备显存 267.69 GiB，`gpu_memory_utilization=0.92`（246.27 GiB 可用）。

### 6.2 KV cache：每个 TP rank 一份完整拷贝，但只存自己阶段的层

- MLA 的 KV 在 TP 内**不切分**，阶段内 8 张卡每张都存完整的 KV（与策略一相同）。
- 每个 token 每层的 KV 约 1,152 B（576 个元素 × 2 字节；由策略一的数字反推：193.7 GiB ÷ 2,959,680 tokens ÷ 61 层 = 1,152 B，推断）。
- 阶段 0 每个 token 要存 31 层 = 35,712 B：199.71 GiB ÷ 35,712 B ≈ 6,004,6xx，与日志 `6,004,640` 吻合。阶段 1 要存 30 层 = 34,560 B，按同样公式约 6.08M tokens；实例的容量取较小者，即阶段 0 的 6,004,640。
- 对比 TP16：每个 token 要在 16 张卡上各存 61 层；PP2 × TP8 只在 8 张卡上存（每阶段 8 张存一半的层），所以同样的每卡显存下，token 容量约为 2 倍（6,004,640 / 2,959,680 = 2.03）。
- 与 DEP 比：DEP 每个 rank 存自己请求的 KV，16 个 rank 合计 31,388,672 tokens（策略三第 6.2 节）；不同方案的“token 数”口径不同，不能直接当成吞吐上限。

### 6.3 EP：每阶段内 256 个专家分到 8 张卡

日志 `[EP Rank 0/8] … Expert placement strategy: linear. Local/global number of experts: 32/256`，两个阶段相同。即每个阶段都存**完整的 256 个专家**（对应各自的层），每张卡 32 个专家，共 8 个 rank 分担；对比 TP16 + EP16 每卡 16 个专家（EP = 16）。

## 7. 一个 token 怎么走（推断为主）

1. 请求到 tppp-0 的 API server，EngineCore 调度成一个批次，广播给两个节点的 Worker（5.2）。
2. 阶段 0（节点 0）做前 31 层：每层 Attention 按 TP8，MoE 每个 rank 只算本地 32 个专家，TP8 内 all-reduce 合并（与策略一同一路径，规模 8 而不是 16；推断）。
3. 31 层结束，隐藏状态通过 NCCL 点对点发给阶段 1 的对应 TP rank（5.3：8 条并行流）。
4. 阶段 1 做后 30 层，得到 logits，采样出下一个 token，传回调度器；下一个 decode 步重复。
5. 批次队列长度是 2：理论上两个阶段能各处理一个不同的批次；但一个请求只有一个 token 在途时，两个阶段是串行的（第 9 节）。

## 8. 功能与行为测试（实测；不是性能测试）

### 8.1 功能

| 测试 | 结果 |
|---|---|
| `/v1/models` | `deepseek-ai/DeepSeek-V3` |
| 对话（`Who are you?`，64 token） | 正常结束（finish=stop） |
| 16 个并发请求，各 128 token | 16/16 全部成功 |
| 约 20k token 的 prompt（超过 `max_num_batched_tokens=16384`，触发分块 prefill） | 成功，prompt_tokens=20010，输出符合内容 |
| 约 49k token 的随机文本 prompt（两次） | 成功（prompt_tokens=49284、49552） |

### 8.2 GPU 利用率（实测，0.5 s 采样）

- 空闲：16 张卡 0%。
- 单请求 decode（输出 1500 token，12.4 s）：**两个节点的 16 张卡都持续 100%**（tppp-0 28 个采样点中 25 个 ≥ 50%，tppp-1 29 个中 24 个）。
- **利用率看不出流水线气泡**：单请求时两个阶段其实是轮流算，但 `nvidia-smi` 显示两个阶段都是 100%。可能原因是等待数据的 NCCL 点对点 kernel 本身占着 GPU（推断，没有抓 kernel 验证）。所以不能用 GPU 利用率判断 PP 的有效负载。
- 长 prefill 时只有零星的几个采样点是高值；第一次 49k token 请求用了 11 s，第二次 1.5 s，日志里对应有 `CuTeDSL JIT compilation during inference: FlashPrepareScheduler`（01:20，两个阶段都有），说明第一次遇到新形状会有 JIT 开销（日志直接提示；与耗时的对应关系是推断）。

### 8.3 网络计数器

- 一个 256 token 的请求：每节点 `rx_write_requests` +8,192，`packet_seq_err` 和 `out_of_sequence` 为 0（5.3）。
- 16 个并发请求：+8,304，两项错误计数仍为 0。
- 20k token 长 prompt：+4,976，`packet_seq_err` +16（tppp-0）、`out_of_sequence` +16（tppp-1）。这是本方案唯一一次看到序列错误，且数量很小；策略三的一轮短负载是数百。原因没有排查。

## 9. 与策略一、策略三的对比（实测数字 + 机制）

| | 策略一 TP16 + EP16 | 策略二 TP8 × PP2（EP8/阶段） | 策略三 DP16 + EP16 |
|---|---|---|---|
| 调度器 / EngineCore | 1 | 1 | 16（每 DP rank 一个） |
| 每卡权重 | 40.47 GiB | 38.71 / 41.17 GiB | 58.02 GiB |
| 峰值激活预留 | 9.46 GiB | 6.82 / 7.42 GiB | 58.9 GiB |
| KV token 容量（整个实例） | 2,959,680 | 6,004,640 | 31,388,672（16 个 rank 合计，每 rank 自己的请求） |
| 每卡专家数 | 16（EP16） | 32（EP8） | 16（EP16） |
| `init engine` | 912.6 s | 737.67 s | 1137.96 s |
| 跨节点通信 | 每层 2 次 TP16 all-reduce | 阶段交界处的点对点：每步 32 个 RDMA write/节点 | 每个 MoE 层 all-gather + reduce-scatter（每步 116 次集合通信） |
| 需要 `--disable-custom-all-reduce` | 需要 | 不需要 | 不需要 |
| 空闲 rank 陪跑 | 无（一批共同计算） | 无 | 有（`dummy-batch-why.md`） |
| 主要副作用 | 跨节点 all-reduce 在每层关键路径上（推断） | 流水线气泡：单请求时阶段串行；并发不足时有一个阶段在等 | 陪跑、最慢 rank 定步调 |

说明：“每步 32 个 RDMA write”是请求个数，策略一、三的计数口径是同一个计数器，但负载不同，数值不直接可比；上表没有给性能结论。

## 10. 结论

- 策略二可以直接部署，页面命令没有做功能性修改，也没有遇到卡死。
- 它的结构最“简单”：一个调度器；所有重通信（TP8 all-reduce、EP8 MoE）都在节点内，跨节点只有阶段交界的点对点，实测每个 decode 步每节点 32 个 RDMA write，8 块网卡各 4 个。
- KV 容量约为 TP16 的 2 倍，因为每个阶段只存自己那些层的 KV；每卡权重与 TP16 相当，比 DEP 少约 17–19 GiB。
- 代价是流水线气泡；这里没有做性能测试，所以没有量化气泡对延迟和吞吐的影响。

## 11. 未验证项

- 流水线气泡的大小、单请求和并发下的延迟与吞吐（按要求没有做性能测试）。
- 8 条点对点流是否按 TP rank 一一对应、两个方向的 RDMA write 是否来自 credit 流控（5.3 的推断）。
- 阶段之间传的是完整隐藏状态还是切片（源码 `parallel_state.py` 有 `all_gather_group` 的切片优化，本次没有核对是否启用）。
- `[31, 30]` 的切分对两阶段耗时是否平衡；可用 `VLLM_PP_LAYER_PARTITION` 调整，没有试。
- GPU 利用率为什么看不出气泡（推断为 NCCL 点对点 kernel 占用，未抓 kernel）。
- 单请求时批次队列（长度 2）能否让两个阶段重叠，需要在并发下观察，没有做。
- 工具调用（去掉了 `--chat-template`）。
- 20k token 长 prompt 里出现的 `packet_seq_err` +16 的原因。

## 12. 运维备注

- 部署：`kubectl apply -f tppp-server.yaml`；日志 `kubectl logs -f tppp-0`；验证 `kubectl exec tppp-0 -- curl -s localhost:8000/v1/models`。
- 停止：`kubectl delete -f tppp-server.yaml`（HF 缓存在 hostPath，保留）。
- 从 dev 机访问 API：`kubectl port-forward svc/tppp-api 18000:8000`；RDMA 计数用 Pod 内的 `rdma statistic show link`（Pod 内 `/sys/class/infiniband/*/ports/1/hw_counters` 不存在）。
- 本集群当前 4 个节点：`tppp`（本方案）占 2 个，`dep`（策略三）占 2 个，GPU 全部用满。
- 现象：`pkill -f "nvidia-smi …"` 会杀掉自己的 `bash -c` 命令行（命令行里含该字符串），用 `pkill -x nvidia-smi`。
