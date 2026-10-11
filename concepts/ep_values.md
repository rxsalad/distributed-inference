# EP 的价值：把 Expert 分布到多机多卡

> 前置：`moe.md`（MoE 结构、总参数与激活参数）、`parallelism-basics.md`（TP / EP 与集合通信）、`dpa_ep.md`（DPA + EP 一层怎么执行——本文聚焦价值与边界，不复述流程）。本文回答：MoE 模型把 Expert 分布到多机多卡（Expert Parallel，EP）能换来什么、通信模式与 TP 有何不同、收益来自哪里、边界在哪里。文中参数量与权重体积由 DeepSeek-V3 的 `config.json`（Hugging Face）按配置计算得出，非实测权重统计。

## 1. 核心：容量随卡数扩展，单 token 计算量不变

MoE 模型有两个独立的旋钮（细节见 `moe.md` 第 3 节）：

| 旋钮 | 由什么决定 | 作用 |
| --- | --- | --- |
| 总参数 | Expert 的数量 | 模型容量：能存多少知识 |
| 激活参数 | 每个 token 被 top-k 选中的 Expert 数 | 每个 token 的计算量 |

以 DeepSeek-V3 为例：总参数约 6710 亿，其中约 97.4% 在 Experts 里；每个 token 只激活 256 个路由 Expert 中的 top-8，激活参数约 376 亿。

EP 把"总参数"这个旋钮接到集群规模上：**每张 GPU 持有一部分完整的 Expert，集群的模型容量等于各卡持有量之和**。卡越多，能服务的 MoE 模型越大，而单 token 的计算量（激活参数）不变。这是 MoE"参数多、激活少"特性最直接的兑现，稠密模型做不到——稠密模型扩容只能靠 TP / PP 切分权重，其中 TP 的并行度越大，单卡计算越碎、通信占比越高（见 `parallelism-basics.md` 第 5 节）；MoE 多了 EP 这条"切容量、不切单 token 计算"的路径。

## 2. 显存：97% 的参数要切分，而且是"切分"不是"复制"

DeepSeek-V3 的权重以 bf16 计约 1.3 TB：

- 远超单卡 HBM（几百 GB 量级）；
- 放在单台 8 卡机（HBM 约 2 TB 量级）上，光权重就占去大半 HBM，留给 KV Cache 和激活的空间很有限。

EP 按 **Expert 粒度**切分：每张 GPU 持有一部分完整的 Expert，一个 Expert 不被劈到两张卡上（"EP + 机内 TP"组合下 Expert 还会在节点内再做 TP 切分，见第 3 节），同一份 Expert 权重在集群里只存一份——这是**切分，不是复制**。集群可容纳的权重容量随 GPU 数线性增长，没有多存任何冗余。

对照 DPA + EP 组合里的 **Attention 权重是复制的**：每个 DP rank 各存一整份（代价分析见 `dp_attention.md` 第 5 节）。Attention 在 DeepSeek-V3 中只占总参数约 1.7%，复制得起；Expert 占大头，只有靠切分才能落到多机上。

## 3. 通信模式：为什么是"机内 TP + 跨机 EP"

TP 和 EP 都切权重，但通信模式差别很大：

| | TP | EP |
| --- | --- | --- |
| 通信发生在哪 | 每一层：Attention 后一次、FFN 后一次 | 只有 MoE 层：Expert 计算前后各一次（Dispatch / Combine） |
| 操作 | AllReduce（各卡部分和求和，人人拿到完整结果） | All-to-All（token → 目标 Expert 所在 GPU，结果返回） |
| 通信搬的是什么 | 每层末尾全量 `[tokens, d_model]` 激活的部分和 | token 的 hidden state，Router 路由决定去向，只搬"该搬的"token；每卡通信量见下文 |

**TP 为什么通常限制在机内**（推导见 `parallelism-basics.md` 第 4、5 节）：

- AllReduce 在**每一层**的关键路径上，L 层共 2L 次，跨机网络延迟会直接叠加到每层延迟里；
- 通信量与 token 数成正比，而机内互联（NVLink / Infinity Fabric）与跨机网络的带宽差一到两个数量级，这种每层高频通信只能放在机内高带宽链路上。

**EP 为什么适合跨机**：

- 通信只出现在 MoE 层：DeepSeek-V3 共 61 层，前 3 层是稠密 FFN（没有逐 token 路由，由引擎单独处理），其余 58 个 MoE 层每层 2 次 All-to-All；
- 切分单位是**完整的 Expert**：Router 的路由结果本身就决定了每个 token 的去向，All-to-All 把 token 恰好搬到该 Expert 所在的 GPU，通信模式与数据归属天然对齐；
- 与 DPA 配合后 Attention 侧零通信、KV 留在本地（见 `dp_attention.md`），GPU 间互联带宽全部留给 MoE 层的 All-to-All。

所以大规模 MoE 的常见做法是**组合**：TP 在机内（每层高频通信走高带宽互联），EP 跨机（All-to-All 走 RDMA），总 GPU 数 = 各并行度之积（见 `parallelism-basics.md` 第 6 节）。

一个需要点明的细节：MLA 模型的标准形态常常是 **TP = 1**——Attention 反正没法按 head 切 KV，而权重复制又付得起（见 `dp_attention.md` 第 5 节），于是直接"每张 GPU 一个独立 DP rank（DPA）+ Expert 用 EP"，全模型没有 TP；"机内 TP + 跨机 EP"是 TP > 1 时的形态（GQA 模型，或 Attention 权重复制付不起的情况）。以 vLLM 为例：EP 度由 TP × DP 推导（见 `parallelism-basics.md` 第 6 节），其 DeepSeek 部署示例正是 TP = 1 + DP = 8（单机）/ 16（双机）。

跨机 EP 的两个前提：

1. 高带宽、低延迟的 RDMA（RoCE / InfiniBand）；
2. 流量模式是 **any-to-any**（详见下小节），多轨（multi-rail）并行网络能为它提供多条独立路径，与这种流量模式的匹配度优于单链路。

### EP 的通信量：稀疏的 All-to-All 与 1/P 因子

直觉上"All-to-All"是每张卡都给每张卡发（full 模式，每卡收发量 ≈ N，量级系数见 `parallelism-basics.md` 第 3 节），但 MoE 的 Dispatch 不是 full All-to-All，而是**数据依赖的 scatter/gather**：

- **单 token 视角**：一个 token 只去持有它被选中 k 个 Expert 的 GPU（P ≫ k 时约 k 个不同的 GPU），不去全部 P 张卡；每个目的地收到多少 token 事先不知道，由路由决定（若选中的多个 Expert 恰在同一张卡，该卡只收一份，实际量略小于估算）；
- **batch 视角**：整个 batch 的路由结果把 token 打散到全集群，卡间流量矩阵被填得基本稠密——任意两张卡最终都会有数据互发，所以网络必须按 any-to-any（高 bisection 带宽）建设，而不是只看邻居链路；
- **每卡通信量**：每次 Dispatch 每 GPU 发出 ≈ $k \cdot N_{local} \cdot d = \frac{kN}{P}d$（N 为本层总 token 数，d 为 hidden 维度），Combine 返回时同量级，每 MoE 层两次 ≈ $\frac{2kN}{P}d$，比 full All-to-All 的 $2Nd$ 小 **P/k 倍**。

这也把 `dpa_ep.md` 第 6 节两种搭配的通信差别说精确了（每层每卡）：

| 方案 | 每层每卡通信量 | 随 P 的变化 |
| --- | --- | --- |
| DPA + EP（Dispatch + Combine） | $\approx \frac{2kN}{P}d$ | ∝ 1/P，**集群越大越薄** |
| DPA + TP（All-Gather + Reduce-Scatter） | $\approx 2Nd$ | 常数，**不随 P 变薄** |

比值即 **P/k**（k = 8、P = 64 时，TP 方案每卡每层多搬 8 倍数据）。All-Gather + Reduce-Scatter 是广播/归约模式而非置换模式，能避开 any-to-any——代价是每卡都要聚集**全部** token，通信量不随集群规模摊薄。这就是"为什么机内 TP、跨机 EP"的定量形态：EP 的每卡通信负担随集群摊薄，TP 的不摊薄。

两点补充：

- **共享 Expert**（DeepSeek 等模型设有）在全部卡上做 TP 切分，各卡本地算完自己那份，不进入 Dispatch；
- RDMA 实现把该模式拆成**单边 put/get**（NIC 卸载、无集合 barrier、收发可与计算重叠），机内部分走机内互联（NVLink / Infinity Fabric），只有跨节点卡对走网络——DeepEP 这类专用 MoE 通信库的通用做法，通信库一览见第 4 节。

### 对照 PDD 的 KV 传输：同一张网络上的两种流量模式

Prefill–Decode 分离（PDD，Prefill 与 Decode 跑在独立实例，Prefill 完成后把 KV 移交给 Decode）的 KV 传输是**固定的 1:1 rail 对 rail 双射**：Prefill 实例的 TP rank i 发给对端 Decode 实例的 TP rank i（如 TP8 下 8 路分片并发传输），模式确定、与请求内容无关，可批量、可与计算重叠。EP 的 Dispatch 则相反：**动态（路由结果）、逐层、延迟敏感**的 any-to-any。同一张 RDMA 网络承载这两种模式——这也是 PDD 的"KV 传输"用通用传输库即可工程化，而 EP 的"Dispatch/Combine"需要专用通信库、并构成大 MoE 推理通信层核心的原因。

## 4. 计算侧：token 被 Router 散到各卡

路由是逐 token 独立的（`moe.md` 第 2 节），token 的 Expert 计算天然发生在持有该 Expert 的 GPU 上：

- **Prefill（token 多）**：一个 batch 的成百上千个 token 被 Router 散到各机的 Expert 上，每张 GPU 对收到的 token 做完整的 FFN GEMM。纯 EP 形态下单卡的计算单位是"一个完整 Expert"而不是"矩阵的一个切片"，只要每个 Expert 分到的 token 足够多，GEMM 效率就高；吞吐随卡数扩展，直到被网络带宽和负载均衡卡住；
- **Decode（token 少）**：每个 request 每步只有 1 个新 token，路由到每个 Expert 的 token 很少，GEMM 退化为 GEMV（见 `inference.md` 第 1.2 节），Expert 计算利用率低。所以 Decode 端通常 EP 度取得较小，或改用 TP 切 Expert（见 `moe.md` 第 4 节）。

这是 EP 与 TP 在计算侧的本质差别：TP 的并行度由"权重怎么放下、延迟怎么降"决定，EP 的并行度由"Expert 有多少、token 去哪儿"决定——前者把计算切碎，后者把计算分派。

### 执行这次散落的通信库

上面的 scatter/gather 由三层库执行（实现生态以 vLLM 为例，入口是 `--all2all-backend`，文档口径 2026-10）：

| 层 | 代表 | 做法 |
| --- | --- | --- |
| 通用集合通信 | NCCL（NVIDIA）/ RCCL（AMD） | alltoall 内部是调度的 p2p 收发序列；通用、零依赖，但等份 buffer 假设、集合 barrier 语义 |
| MoE dispatch 专用 | DeepEP（DeepSeek）、FlashInfer MoE-EP、pplx-kernels | 懂路由语义：变长 scatter/gather、收发与计算重叠、层级化（机内卡对走机内互联，跨节点卡对才走 RDMA） |
| 传输底座 | UCX / libibverbs、GPUDirect RDMA、IBGDA | 网卡直读直写 GPU 显存、不经主机内存；IBGDA 让 GPU 的 SM 直接下发 RDMA 指令、绕开 CPU 提交路径 |

**DeepEP** 的两种模式对应上文的两类负载：

| 模式 | 面向负载 | 做法 |
| --- | --- | --- |
| 高吞吐（high_throughput） | Prefill | RDMA 批量搬运、带宽优先，配合 grouped GEMM 的连续布局 |
| 低延迟（low_latency） | Decode | SM 驱动 RDMA（IBGDA）、延迟优先；跨机的 hidden state 可量化（INT8/FP8）后再发 |

vLLM 的 backend 选型：

| Backend | 定位 |
| --- | --- |
| `allgather_reducescatter`（默认） | 用 All-Gather + Reduce-Scatter 执行 dispatch/combine，即第 3 节的 DPA + TP 流量形态，每卡流量是真 a2a 的 P/k 倍；好处是零额外依赖 |
| `deepep_high_throughput` / `deepep_low_latency` | 真 a2a，分别面向 Prefill / Decode |
| `flashinfer_nvlink_one_sided` / `two_sided` | 多机 NVLink（MNNVL）系统 |
| `passthrough` | FlashInfer MoE-EP megakernel 自己完成 dispatch/combine，框架层不跑 all-to-all |

官方口径的两个 caveat：DeepEP 两种模式为 Prefill–Decode 分离部署优化，混合负载（同实例 chunked prefill + decode）下性能可能变差；通信与计算的重叠是一等开关（`--enable-dbo`，Dual Batch Overlap）。

PDD 侧（另一种流量模式，对照见第 3 节末）是另一族库：NIXL（NVIDIA）及其 ROCm 移植 rixl、Mooncake Transfer Engine、MoRI-IO（AMD）——底层 GDR / RDMA 手段与第二层相同，差别在流量模式（固定 1:1 对 rail vs 动态 any-to-any）。

## 5. 与 DPA 配合：权重和 KV 都随集群扩展

EP 和 DPA 分别解决"放哪里"问题的两半：

| 资源 | EP + DPA 下的放置 | 随卡数如何变化 |
| --- | --- | --- |
| Expert 权重（参数大头） | 每张 GPU 持有一部分完整 Expert | 线性增长，集群容量 ↑ |
| Attention 权重（占比小） | 每张 GPU 复制一整份 | 复制，代价见 `dp_attention.md` 第 5 节 |
| KV Cache | 留在 request 所在 GPU 本地 | 线性增长，并发容量 ↑ |
| 单 token 计算 | top-k 个 Expert，可能散在几张 GPU 上 | 不变 |

集群规模变大时，模型容量和并发 KV 容量同时变大，GPU 间通信集中在 MoE 层的 All-to-All 上——这就是"通信资源要用在刀刃上"（推理见 `dpa_ep.md` 第 1 节）。

## 6. 代价与边界

EP 不是免费的，代价集中在 MoE 层的集合通信与负载上（末两条为 DPA + EP 组合特有）：

| 代价 | 说明 |
| --- | --- |
| All-to-All 开销 | 每 MoE 层 2 次，位于关键路径；每卡量 ≈ $\frac{2kN}{P}d$、∝ 1/P（见第 3 节）；跨机时受 RDMA 带宽与延迟影响；token 少（Decode）时偏延迟型，token 多（Prefill）时偏带宽型，见 `parallelism-basics.md` 第 5 节 |
| Expert 负载不均 | Router 可能让某些 Expert 成为热点，持有它的 GPU 成为瓶颈，其他卡空等，见 `moe.md` 第 4 节 |
| Decode 小 batch 利用率低 | 每个 Expert 分到的 token 少，GEMM 效率低，见 `moe.md` 第 4 节 |
| EP 度过大收益递减 | EP 越大，每卡 Expert 越少、每个 Expert 分到的 token 越少，通信占比上升，扩展效率下降，见 `parallelism-basics.md` 第 5 节 |
| DP rank 同步 | 各 DP rank 的 request 数与序列长度不同，进出 MoE 的集合通信要所有 rank 同步参与，快的等慢的，见 `dpa_ep.md` 第 7 节 |
| request 粘性 | request 固定在所属 DP rank 上，路由策略和 prefix cache 命中受此限制，见 `dpa_ep.md` 第 3 节 |

### 热点 Expert 的工程缓解：EPLB

上述代价中，Expert 负载不均在生产里有最成熟的缓解手段：vLLM 的 **EPLB（Expert Parallel Load Balancer）**（`--enable-eplb`），两个措施：

1. **周期性重映射**：每次前向收集 token → Expert 分布，滑窗跟踪（默认 1000 步），周期性（默认每 3000 步）把 Expert → rank 的分配重新映射，让热点 Expert 摊到更多 rank 上；
2. **冗余 Expert**：把热点 Expert 多复制一份放到不同 rank（`num_redundant_experts` 参数，官方建议大规模场景取 32）。代价是额外显存——DeepSeek-V3 上每个冗余 Expert 每 rank 约 2.4 GB——显存紧张时不划算。

两个措施本质相同：**不改 Router，只改物理放置**——让热点 Expert 的放置变冗余、变均匀。重映射的权重搬移走通信后端（NIXL、pynccl 等），平衡度指标为每 Expert 平均 token 数 ÷ 最大 token 数。

用不用 EP、EP 度取多大，受三个条件同时约束：

- **权重容量**：模型放不放得下（下界）；
- **网络**：RDMA 带宽够不够撑住每层 MoE 的 All-to-All（上界）；
- **负载类型**：Prefill 为主还是 Decode 为主（决定合理的 EP 度，见第 4 节）。

## 7. 小结

- EP 把 MoE"参数多、激活少"兑现为集群能力：**总参数（容量）随卡数扩展，单 token 计算量不变**；
- 按 Expert 粒度切分，权重容量线性增长且无复制；配合"机内 TP + 跨机 EP"，把高频通信留在机内高带宽链路，All-to-All 走多轨 RDMA，每卡通信量 ∝ 1/P、随集群摊薄；
- 代价集中在 MoE 层的 All-to-All 与负载均衡，EP 度是权重容量、网络带宽、负载类型三者的平衡点。
