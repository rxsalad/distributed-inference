# 并行与集合通信基础

> 前置知识：本系列后续文章（`moe.md`、`dp_attention.md`、`dpa_ep.md`）反复用到 TP / DP / EP 等并行方式，以及 AllReduce、All-Gather、Reduce-Scatter、All-to-All 等集合通信。本文集中交代这些概念。

## 1. 为什么需要多 GPU 并行

- **显存**：大模型的权重加 KV Cache，单卡放不下；
- **吞吐 / 延迟**：把计算分给多张卡，可以同时服务更多请求，或缩短单个请求的延迟。

代价是 GPU 之间要交换数据（通信）。不同的并行方式，本质上是在"切什么"和"通信什么"之间做不同的取舍。

## 2. 四种基本并行方式

| | 切什么 | 每个 GPU 持有 | 主要通信 | 典型用途 |
| --- | --- | --- | --- | --- |
| **DP**（Data Parallel） | 切**数据**（请求） | 完整的模型（或模型的一个完整部分） | 推理时基本无需通信 | 提高吞吐；Attention 侧的 DP 见 `dp_attention.md` |
| **TP**（Tensor Parallel） | 切**权重矩阵** | 每个权重矩阵的 1/P | 每层 AllReduce | 模型单卡放不下、又希望降低单请求延迟 |
| **PP**（Pipeline Parallel） | 切**层** | 连续的若干层 | 相邻 stage 间点对点传递激活 | 跨机放下超大模型 |
| **EP**（Expert Parallel） | 切 **Expert**（MoE） | 一部分 Expert | All-to-All（Dispatch / Combine） | MoE 模型，见 `moe.md` |

几点说明：

- **推理中的 DP 与训练不同**：训练的 DP 要在各副本之间同步梯度（AllReduce）；推理没有梯度，各 DP rank 独立处理各自的请求，不需要同步参数；
- **PP** 把相邻 stage 之间传递的是 hidden state（点对点 send / recv），需要用 micro-batch 让各 stage 同时有活干，否则存在空闲（bubble）；本系列后续文章主要不涉及 PP；
- 实际系统常把几种方式**组合**使用（见第 6 节）。

## 3. 集合通信原语

集合通信指一组 GPU 共同参与的通信操作。下面用 4 个 GPU（GPU0~GPU3）说明，设每个 GPU 上的数据被切成 4 块。

### AllReduce：求和，且人人拿到完整结果

对各 GPU 上**同位置的元素**做归约（推理中是求和），**每个 GPU 都得到完整的归约结果**。

```
GPU0: [a0, a1, a2]
GPU1: [b0, b1, b2]       AllReduce(sum)       每个 GPU 都得到：
GPU2: [c0, c1, c2]       ───────────────►     [a0+b0+c0, a1+b1+c1, a2+b2+c2]
```

### Reduce-Scatter：求和，但每人只拿一份

先归约，再把结果**切成 P 份，每个 GPU 只拿其中 1 份**。

```
GPU0: [A0 A1 A2 A3]                      GPU0: [A0+B0+C0+D0]
GPU1: [B0 B1 B2 B3]   Reduce-Scatter     GPU1: [A1+B1+C1+D1]
GPU2: [C0 C1 C2 C3]   ─────────────►     GPU2: [A2+B2+C2+D2]
GPU3: [D0 D1 D2 D3]                      GPU3: [A3+B3+C3+D3]
```

### All-Gather：拼接，人人拿到完整结果

每个 GPU 持有 1 份，**把所有人的份拼起来，每个 GPU 都得到完整数据**（不做求和）。

```
GPU0: [A]                                GPU0: [A B C D]
GPU1: [B]             All-Gather         GPU1: [A B C D]
GPU2: [C]             ─────────────►     GPU2: [A B C D]
GPU3: [D]                                GPU3: [A B C D]
```

### All-to-All：每人给每个人发不同的数据

每个 GPU 把自己的数据切成 P 块，**第 j 块发给 GPU j**，效果相当于对"GPU × 数据块"这个矩阵做转置。

```
GPU0: [a0 a1 a2 a3]                      GPU0: [a0 b0 c0 d0]
GPU1: [b0 b1 b2 b3]   All-to-All         GPU1: [a1 b1 c1 d1]
GPU2: [c0 c1 c2 c3]   ─────────────►     GPU2: [a2 b2 c2 d2]
GPU3: [d0 d1 d2 d3]                      GPU3: [a3 b3 c3 d3]
```

MoE 的 Dispatch / Combine 用的就是它：每个 GPU 上的 token 各自要去不同的 Expert（不同的 GPU），发给每个目标 GPU 的数据量不同。

### 它们之间的关系与通信量

- **AllReduce = Reduce-Scatter + All-Gather**。实际实现（如 ring 算法）正是把 AllReduce 拆成这两步，所有 GPU 同时收发，避免某个节点成为带宽瓶颈；
- 设每个 GPU 上的数据大小为 N、GPU 数为 P，ring 算法下每个 GPU 的收发量约为：

| 操作 | 每个 GPU 的收发量 |
| --- | --- |
| AllReduce | $2\frac{P-1}{P} N$ |
| Reduce-Scatter / All-Gather | $\frac{P-1}{P} N$ |
| All-to-All | $\frac{P-1}{P} N$（发出的数据量，每个目标 GPU 一份） |

P 较大时，这些系数都接近常数，与 GPU 数基本无关。以上是标准算法下的结论；实际库（如 NCCL）会根据拓扑选用不同算法，系数可能不同。

## 4. TP 为什么每层都要 AllReduce

TP 对每个子层采用"先列切、后行切"的配对，中间不用通信，只在子层末尾通信一次。以 FFN 为例（TP=8）：

```
x [tokens, d_model]            ← 每个 GPU 都有完整的 x
 │
 ▼  W1 按列切：每个 GPU 持有 d_ff/8 列
h_i = act(x @ W1_i)            ← 各 GPU 独立算，h_i: [tokens, d_ff/8]
 │
 ▼  W2 按行切：每个 GPU 持有 d_ff/8 行
y_i = h_i @ W2_i               ← 每个 GPU 只得到"部分和"，y_i: [tokens, d_model]
 │
 ▼
y = y_0 + y_1 + ... + y_7      ← 需要把 8 份部分和加起来
```

$y = hW_2$ 中对 $d_{ff}$ 维的求和被拆到了 8 个 GPU 上，每个 GPU 只有 1/8 的加数。要得到正确的 $y$，必须把 8 份**同形状**的 `[tokens, d_model]` 结果逐元素相加，并让**每个 GPU 都拿到完整的和**（下一层每个 GPU 都需要完整输入）。这就是 AllReduce。

Attention 同理：各 GPU 负责一部分 head，输出投影 $W_O$ 按行切，输出也是部分和。因此**每层 2 次 AllReduce**（Attention 后一次、FFN 后一次），L 层共 2L 次。

## 5. 通信开销怎么看

- **在关键路径上**：下一层要等通信完成才能开始，通信时间直接叠加到延迟里，很难完全被计算掩盖；
- **通信量与 token 数成正比**：每次传的是 `[tokens, d_model]` 量级的激活，batch 越大数据量越大；
- **延迟型与带宽型**：token 很少时（如 Decode），单次通信的数据量小，开销主要是固定的启动延迟；token 很多时（如 Prefill），开销主要由带宽决定；
- **机内与跨机**：同一台机器内的 GPU 通常由 NVLink 等高带宽互联，跨机则走网络，带宽低得多。因此 TP 这类每层都通信的方式通常限制在机内；
- **并行度越大，计算越碎**：每个 GPU 分到的计算变少，通信占比上升，扩展效率下降。

## 6. 混合并行

实际系统常对模型的不同部分采用不同的并行方式，总 GPU 数等于各并行度的乘积。MoE 模型里，Attention 与 FFN / MoE 可以各选一种，形成下面几种搭配。**本系列统一使用这套命名**：简称由"Attention 的并行方式 + FFN / MoE 的并行方式"组成，D 表示 Attention 用 DP（即 DPA，Data Parallel Attention），T 表示用 TP，E 表示 MoE 用 EP。

| 方案 | 简称 | Attention | FFN / MoE | 典型适用 | 通信细节 |
| --- | --- | --- | --- | --- | --- |
| 纯 TP | TP | TP | TP | 稠密模型 | 每层 2 次 AllReduce |
| DPA + TP | DTP | DP | TP | 稠密 FFN 或小规模 MoE，Attention 需要 DP | `dpa_ep.md` 第 6 节 |
| DPA + EP | DEP | DP | EP | MLA + MoE 大模型（如 DeepSeek-V3） | `dpa_ep.md` 第 2~5 节 |
| TP + EP | TEP | TP | EP | GQA 等 KV 可按 head 切分的 MoE 模型 | `dpa_ep.md` 第 6 节 |

正文中全称与简称并用；通信方式的完整对比表只在 `dpa_ep.md` 第 6 节维护。**Wide EP** 不是新的搭配，而是在 EP 侧把 EP 度做得很大并配合负载均衡（见 `ep_values.md` 第 6 节"Wide EP"）。Attention 与 FFN / MoE 之间需要按后者的并行方式做数据重排：EP 用 All-to-All，TP 用 All-Gather + Reduce-Scatter（见 `dp_attention.md` 第 4 节）。

**几个常用术语**：

- **rank**：一个并行组内的一个参与者，通常对应一张 GPU。DP rank、EP rank 指同一张 GPU 在不同阶段扮演的角色。
- **EP 度**：参与 EP 的 GPU 数，即 EP Group 的大小；**EP Group** 是共同持有全部 Expert、彼此做 All-to-All 的一组 GPU（详见 `dpa_ep.md` 第 2 节）。
- **EP = TP × DP**：以 vLLM 的 EP 部署为例，EP 度不是独立配置的，而是由配置推导：Attention 侧的 DP 度为 DP（TP = 1 时 Attention 权重在所有 DP rank 上复制，即 DPA 形态），Expert 侧的 EP 度为 TP × DP，两侧共享同一组 GPU。所以开头的"乘积"是 TP × DP，EP 是推导量，不是再乘一次的因子。

为什么 Attention 用 DP、MoE 用 EP，见 `dp_attention.md` 和 `dpa_ep.md`。
