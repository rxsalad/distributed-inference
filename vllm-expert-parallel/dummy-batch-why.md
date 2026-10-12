# 为什么 DP + EP 下没有请求的 rank 也要陪跑（dummy batch）

只发 1 个请求，DP16 + EP16 的 16 张 GPU 利用率都升到 90%+，请求结束后一起回到 0%。本文回答：为什么会这样、是不是 `allgather_reducescatter` 造成的、是不是因为只有一个 EP 组，以及它带来的副作用。

标注约定：**实测**＝本集群日志或命令输出；**源码核对**＝读了镜像内 vLLM 源码（`vllm/vllm-openai:latest`，vLLM 0.31.0）；**推断**＝由机制得出，未直接验证。本文不含性能数据（按要求不做性能测试）。

相关文档：`strategy3.md`（第 5.6、7、8 节）、`vllm-dp16-ep-two-node.md`（5.1、5.1.1、第 7 节）、`vllm-dp8-ep-single-node.md`（6.1）。

## 摘要

| 问题 | 结论 |
|---|---|
| 本质原因 | EP 把**每一个 MoE 层**拆到所有 rank 上：一个 token 要发到它选中的专家所在的 rank，专家所在的 rank 要替**别的 rank 的 token** 做计算。于是每个 MoE 层都是一次全组参与的集合操作，组内所有 rank 必须执行**相同次数的前向步**。DP 让各 rank 的请求流彼此独立，没有请求的 rank 步数本来就少，只能用 dummy batch 补齐 |
| 是 `allgather_reducescatter` 造成的吗 | **不是**。它决定的是通信量和开销，不是要不要陪跑。换成别的 all2all 后端，专家所在的 rank 仍要为别人的 token 算专家，仍要参与（推断；DeepEP 在本集群没跑起来，没有实测） |
| 是因为只有一个 EP 组吗 | **一半是**。陪跑的**原因**是 MoE 的跨 rank 依赖；“只有一个组”决定的是陪跑的**范围**：16 个 rank 在同一个集合通信域里，1 个请求就拖着 15 个 rank 陪跑。如果拆成两个独立的 EP 组，一个组有请求，另一个组不需要陪跑（推断） |
| 稠密模型的 DP 为什么没有陪跑 | rank 之间没有计算依赖（实测：Qwen3-8B 双节点 DP，空闲 rank 真正空闲，`vllm-dp16-ep-two-node.md` 第 7 节） |
| 主要副作用 | GPU 利用率失真；低负载时浪费；最慢的 rank 决定所有 rank 的步调；故障互相耦合；每步多一次跨节点的 CPU 同步；结束时有拖尾（第 5 节） |

## 1. 现象（实测）

- 全空闲：16 张 GPU 都是 0%。
- 只发 1 个 400 token 的请求：由 engine 4（一个 DP rank）处理，但两个节点的 16 张卡都升到 90%+（`dep-0` 各卡 98–100%，`dep-1` 各卡 99–100%）；请求结束后一起回到 0%（`strategy3.md` 第 8 节）。
- 旧文档 `vllm-dp16-ep-two-node.md` 在内部 LB 部署上同样观察到 16 张卡都在忙（92–100%）。
- 对照：稠密模型（Qwen3-8B）双节点 DP，空闲 rank 真正空闲（`vllm-dp16-ep-two-node.md` 第 7 节）。

## 2. 本质原因：MoE 把 rank 之间变成了“计算依赖”

1. **EP 切的是专家**。256 个路由专家分到 16 个 rank，每个 rank 只有 16 个专家。一个 token 选中的 8 个专家分散在不同 rank 上。
2. **所以 rank 的计算依赖别的 rank 的数据**。某个 rank 上的专家，要处理来自**所有 DP rank** 的 token。这不只是“通信要凑齐”：即使通信是单边的、不需要对方调用，专家所在的 rank 也必须**运行专家的矩阵乘**，否则别人的 token 没人算。
3. **DP 让各 rank 的步数不同**。每个 DP rank 有自己的调度器、自己的请求；没有请求的 rank，本来不需要前向。
4. **合起来**：每个前向步里，每个 MoE 层（DeepSeek-V3 有 58 个）都要 16 个 rank 同时参与。一个 rank 少跑一步，别的 rank 就会卡在这一步的集合通信上，直到超时。所以所有 rank 必须**步数一致**（SPMD：每个 rank 执行相同序列的集合操作），没有请求的 rank 就跑一个 **dummy batch** 来凑数。

dummy batch 是什么（源码核对，`v1/worker/gpu_worker.py`）：`execute_dummy_batch()` 调 `model_runner._dummy_run(num_tokens, uniform_decode=True)`，`num_tokens` 默认取 `uniform_decode_query_len`，没有投机解码时是 1，即一个只有 1 个 token 的空前向。它自己的计算量很小，但要走完所有层，包括 58 层的集合通信。

## 3. 三个常见误解

### 3.1 是 `allgather_reducescatter` 造成的吗？不是

- 本集群实测用的后端是默认的 `allgather_reducescatter`（`AgRsAll2AllManager`，`strategy3.md` 第 7 节）：每层先 `all_gatherv` 把所有 token 发给所有 rank，各 rank 算自己的专家，再 `reduce_scatterv` 汇总。这是集合通信，缺一个 rank 就卡。
- 但换成点对点或单边的 all2all（如 DeepEP），也解决不了第 2 节第 2 点：专家所在 rank 必须为别人的 token 做计算。所以**陪跑与后端无关**（旧文档 5.1.1 也写了这一点）。
- 后端影响的是**陪跑的代价**：`allgather_reducescatter` 把全部 token 广播给每个 rank，通信量随 rank 数增长（旧文档第 8 节的结论，机制见 `vllm-dp8-ep-single-node.md` 第 5 节），陪跑 rank 同样要收发这些数据。不像按路由只发送给目标专家的 all2all 那么省（推断）。
- DeepEP 在本集群 B300 上没跑起来（`vllm-dp16-ep-two-node.md` 6.3），所以“别的后端也要陪跑”没有实测，是机制推断。

### 3.2 是因为只有一个 EP 组吗？决定的是范围，不是原因

| 场景 | 有没有陪跑 | 说明 |
|---|---|---|
| 稠密模型 DP | 没有 | rank 之间没有计算依赖（实测） |
| DP16 + EP16，单实例（本文，内部 LB 或 hybrid-lb） | 有，范围是全部 16 个 rank | 16 个 rank 是同一个 EP 组；hybrid-lb 的协调者也只有一个，所以 wave 跨两个节点（`strategy3.md` 5.6） |
| 两个独立实例，各自 DP8 + EP8 | 组内有，组间没有 | 一个实例有请求，另一个实例不需要陪跑（推断） |
| TP16 + EP16（策略一） | 没有“陪跑”这个概念 | 只有一个 EngineCore 和一个调度器，16 个 rank 每步都处理**同一批** token，没有空闲 rank（`strategy1.md`） |
| DP + 专家按 TP 切（无 EP，页面上 MI355X 的 DPA+TP） | 仍然需要（推断） | MoE 仍是跨 rank 的集合操作 |

结论：只要 **MoE 在多个有独立请求流的 DP rank 之间协同计算**，就需要陪跑；“一个组”只是让陪跑的范围变成了全部 16 个。

### 3.3 `hybrid-lb` 能不能避免？不能

hybrid-lb 改变的是**请求怎么进来**（每个节点有自己的 API server，节点内 vLLM 做负载均衡，节点间由 Service 分流），不改变 EP 组：16 个 rank 仍是一个组、一个协调者、同一个 wave。请求只落在 dep-0，dep-1 的 8 个 rank 也要陪跑（`strategy3.md` 5.6）。

## 4. 机制：三层同步（源码核对）

| 层次 | 频率 | 做什么 | 通道 |
|---|---|---|---|
| MoE 层的集合通信 | 每步 × 58 层 × 2（gather + scatter） | 交换 token，各 rank 算自己的专家 | NCCL，跨节点走 RoCE |
| 每个前向步的 DP 元数据同步 | 每步 1 次 | 一个 `4 × 16` 的 int32 all-reduce：各 rank 的 token 数、CUDA graph 模式（取最小值）、是否做 DP padding。开了 CUDA graph 时，所有 rank 的 token 数 pad 到同一个最大值（`v1/worker/dp_utils.py`） | 本部署开了异步调度，日志有 `Disabling NCCL for DP synchronization when using async scheduling.`（实测），按源码走 **CPU Gloo**，经 eth0 的 TCP；日志里没有看到 `Using CPU all reduce...` 那一行，所以走 CPU 是源码结论，没有直接确认 |
| wave 的启停共识 | 每 `dp_sync_interval` 步（0.31.0 默认 16；第 1 步也同步） | 全局 all-reduce，判断是否还有 rank 有未完成请求（`EngineCore._has_global_unfinished_reqs`） | 同上 |

wave 的启动与结束、两阶段暂停见 `strategy3.md` 5.6。旧文档里写的“约每 32 步一次”是旧版本（0.19.2rc1）的值，0.31.0 的默认值是 16。

## 5. 副作用

| 副作用 | 说明 | 依据 |
|---|---|---|
| GPU 利用率失真 | 空闲 rank 也是 90%+，不能用 `nvidia-smi` 的利用率判断有效负载或做扩缩容 | 实测 |
| 低负载时浪费 | 1 个请求占用 16 张卡的算力、电力和 RDMA 带宽；hybrid-lb 下流量全落在一个节点，另一个节点也不能休眠或挪作他用 | 现象实测；成本推断 |
| 最慢的 rank 决定步调 | 每层都要等所有 rank，一个 rank 的长 prefill、专家负载倾斜（热点专家所在的 rank）会让其余 15 个 rank 一起等 | 推断（由集合通信结构得出，没有测） |
| 跨 rank 的相互干扰 | 一个 rank 上的长 prefill 会拖慢其他 rank 上正在 decode 的请求，尽管 Attention 本身互不通信 | 推断 |
| 故障互相耦合 | 一个 rank 卡住或崩溃，整组卡住，没有隔离。旧实验里缺少 `NCCL_IB_GID_INDEX` 时，16 个 rank 在第一次集合通信上静默卡死（`vllm-dp16-ep-two-node.md` 6.1），说明这种耦合确实存在 | 耦合的现象实测，推广到运行期为推断 |
| 每步多一次 16 个 rank 的 CPU 同步 | 在关键路径上，且跨节点走 TCP；TP16 里 Gloo 不在每步路径上。延迟影响没有测 | 源码；开销未测 |
| 结束时有拖尾 | wave 的结束需要全局共识，最后一个请求完成后，最多还要空跑到下一个同步点（约 `dp_sync_interval` 步） | 源码推断，未测 |
| 起步多一次通知 | 全空闲后的第一个请求，要先通知协调者，协调者向 16 个引擎（含跨节点）广播 `START_DP_WAVE`，各引擎才开始步进 | 源码；耗时未测 |
| 高并发时成本被摊薄 | 陪跑的 rank 本来就在跑，请求越多，真正有活的 rank 越多，空转的比例越低。这也是 DEP 面向高吞吐的原因 | 推断 |

## 6. 缓解思路（推断，均未验证）

- **缩小 EP 组**：例如两个独立的 DP8 + EP8 实例，陪跑范围从 16 降到 8，且互不影响；代价是每卡专家和权重更多（单节点 DP8 + EP8 的每卡权重比 DP16 + EP16 大）。
- **提高并发，让所有 rank 同时有活**：陪跑变成真实计算；在前面加外部 LB，让各节点负载均匀。
- **低负载场景改用 TP**（策略一）：没有陪跑概念，但每层要两次 TP16 all-reduce（`strategy1.md`）。
- **用 EPLB 缓解专家负载倾斜**：不能消除陪跑，只能减少最慢 rank 的拖累。

## 7. 未验证项

- dummy batch 的实际耗时和占用（只知道默认 1 个 token，没有测）。
- 别的 all2all 后端（DeepEP、pplx 等）下是否同样必须陪跑：按机制推断，本集群没有跑起来。
- 两个独立 EP 组之间不互相陪跑：按机制推断，没有同时部署两组验证。
- 各副作用都没有性能测量（按要求不做性能测试）。
- 本文的源码结论来自 0.31.0；旧文档的 0.19.2rc1 细节可能不同。
- 日志里没有直接看到 dummy batch 的记录，“15 个 rank 在陪跑”是由利用率和源码两方面推出来的。
