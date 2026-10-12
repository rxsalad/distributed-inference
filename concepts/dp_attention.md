# DP Attention

> 多 GPU 推理下，Attention 的并行方式与 KV Cache 的放置方式紧密相关。MLA 把 KV 压缩成所有 head 共享的 latent，使传统 TP Attention 失去"按 head 切 KV"的天然对齐，DP Attention 由此成为 MLA + MoE 大模型推理的常用方案。本文按"TP Attention → MLA 的引入 → MLA + TP 的矛盾 → DP Attention → 代价与适用场景"展开。TP、AllReduce 等并行与集合通信概念见 `parallelism-basics.md`。

## 1. 背景：多 GPU 下 Attention 的 TP 切分

模型单卡放不下时，常用 **TP（Tensor Parallel）** 把每层的权重切到多张 GPU。对 MHA，Attention 按 head 切分：

- Q/K/V 投影按 head（输出维度，即列）切到各 GPU，每个 GPU 只计算自己负责的那一组 head；
- **KV Cache 随 head 一起切**：GPU0 保存 Head 0 ~ N/8 的 K/V，GPU1 保存下一组，以此类推；
- 输出投影 $W_O$ 按输入维度（行）切分，各 GPU 得到形状仍是 `[N, d_model]` 的**部分和**，再做一次 **AllReduce** 汇总。

因此在 MHA + TP=8 下，**Head 与 KV Cache 天然对齐**：每个 GPU 用本地 KV 就能完成自己那部分 Attention，KV 占用和读取量都随 TP 均摊为 1/8。

**TP 的通信代价：每层 2 次 AllReduce**（推导见 `parallelism-basics.md` 第 4、5 节）：$W_O$ 与 FFN 的 $W_{down}$ 按行切分，每个 GPU 只得到部分和，AllReduce 把它们按元素相加，并让每个 GPU 都拿到完整的 `[N, d_model]`，作为下一个子层的输入（下一个子层的第一个矩阵乘按列切分，但输入必须完整）。Attention 后一次、FFN 后一次，共 2 次；每次通信量为 `N × d_model`，夹在两个子层之间，位于关键路径上。

Attention 参数相对 MoE 很小，却要每层为它付一次 AllReduce，在 MLA 下 KV 还没法切，这是引入 DP Attention 的动机之一（见第 4 节）。

## 2. MLA 的引入

**动机**：长 context、大 batch 下，KV Cache 占满 HBM，并且 Decode 每步都要把它读一遍，KV 成为容量和带宽的双重瓶颈。

**做法**：MLA（Multi-head Latent Attention）用低秩压缩。不再为每个 head 保存完整的 K 和 V，而是把它们压缩成一个较小的 **latent 向量**（再加一小段携带位置信息的 RoPE 部分），KV Cache 里只存这份 latent，需要时再通过上投影还原出各 head 的 K、V。

**效果**，以 DeepSeek-V3 为例（61 层、128 个 head；每个 head 的 K 为 192 维 = 128 + 64 的 RoPE 部分，V 为 128 维）：

- 若用 MHA 缓存每个 head 的 K 和 V，每个 token 每层要缓存 $128 \times (192 + 128) = 40960$ 个元素；
- MLA 只缓存 latent（512 维）+ RoPE 部分（64 维）= 576 个元素，约为前者的 1/71；
- 折算到全部 61 层：MLA 每 token 约 35136 个元素（bf16 约 69 KiB），MHA 方式约 4.8 MiB。

（论文中常见的口径是只按 128 维的 head 计算：$2 \times 128 \times 128 = 32768$，对应约 1/57。）

**关键特性**：这份 latent 是**所有 Attention head 共同依赖**的，而不是像 MHA 那样每个 head 有自己的一份。推理时借助矩阵吸收（把上投影合并进 Q 和输出投影），Attention 可以直接在 latent 上进行，效果上相当于**所有 Q head 共享同一份 KV**（类似 MQA，只有 1 个"KV head"）。

## 3. MLA + TP 的矛盾

MLA 大幅缩小了 KV，却破坏了 TP 依赖的"head 与 KV 对齐"：KV 只有一份、被所有 head 共享，**没法按 head 切**。TP=8 时，每个 GPU 负责 1/8 的 head，但每个 GPU 的 Attention 都需要整份 latent KV。于是只剩两种选择：

- **Replication（复制）**：每个 GPU 保存完整的 latent KV。通信少，但：
  - 显存：KV 被复制 TP 份，集群能容纳的 token 总数**不再随 GPU 数增长**（MHA + TP 下是随 GPU 数线性增长的）；
  - 带宽：Decode 每步，每个 GPU 都要读完整的 latent KV，总 HBM 读取量放大 TP 倍。
- **Sharding（分片）**：latent KV 沿序列维度分到各 GPU。显存省了，但每个 GPU 只持有一部分 token 的 KV，Attention 需要各 GPU 分别算部分结果再合并，或跨 GPU 获取数据，因此产生**跨 GPU 通信**。

```
TP Attention + MLA
────────────────────────
GPU0 ── Head 组 ──┐
GPU1 ── Head 组 ──┤
GPU2 ── Head 组 ──┤
...               ├── 共享的 Latent KV（只有一份）
GPU7 ── Head 组 ──┘

复制 KV → 显存 ×TP、HBM 读取 ×TP
分片 KV → GPU 间通信增加
```

Decode 阶段这个问题尤其明显：计算量本身小，主要开销是每步读 KV。如果 8 个 GPU 都为各自的 Attention 去读或同步同一份 KV，MLA 省下的 KV 容量并不意味着 KV 的读取量和通信也按同样比例下降。TP 规模越大，Attention 越容易从 compute-bound 变成 memory / communication-bound。

**根因**：通信或复制开销增加，不是因为"KV 被压缩了"，而是因为**共享的 latent KV 与按 head 分散在各 GPU 上的 Attention 计算之间出现了数据 locality 问题**。

> 对比 GQA：GQA 有 $n_{kv}$ 组 KV，TP 度数不超过 $n_{kv}$ 时仍可按组切分，超过后同样需要复制。MLA 相当于只有 1 组 KV，所以 TP > 1 就会遇到复制问题（MHA / GQA / MQA / MLA 的 KV 大小与可切分性对比见 `inference.md` 第 2 节）。

## 4. DP Attention：让 Attention 脱离 TP

**思路**：不让一个 request 的 Attention 被多个 GPU 共同计算，而是把 Attention 从 TP 中解耦，改为 **Data Parallel**：

- 不同 DP rank（通常对应一张 GPU）处理**不同的 request**；
- 每个 request 绑定一个 DP rank，它的 Attention 和 **KV Cache 都留在该 GPU 本地**（request affinity）；
- 每个 DP rank 持有完整的 Attention 权重（MoE 模型中 Attention 占总参数比例很小，复制代价可以接受，详见第 5 节）。

```
DP Attention
────────────────────────
Request 0 → GPU0 → 本地 KV → Attention
Request 1 → GPU1 → 本地 KV → Attention
Request 2 → GPU2 → 本地 KV → Attention
...
```

**解决了什么**：

| 问题 | TP Attention + MLA | DP Attention |
| --- | --- | --- |
| KV 放置 | 复制 TP 份，或分片后跨卡访问 | 每个 request 的 KV 只存一份，在本地 |
| KV 总容量 | 不随 GPU 数增长 | 随 GPU 数线性增长 |
| Decode 的 KV 读取 | 每个 GPU 都读全量，总读取量 ×TP | 每个 GPU 只读自己 request 的 KV |
| Attention 通信 | 每层 AllReduce，或分片带来的跨卡访问 | 无 |

### 与 FFN / MoE 的衔接

Attention 按 request 切分后，每个 DP rank 手里只有**自己 request 的 token**。而 FFN / MoE 需要什么样的数据分布，取决于它自己的并行方式。因此每层 Attention 与 FFN / MoE 之间，要按 FFN / MoE 的实现方式（TP 或 EP）做对应的**数据编排**：进入前把 token 重排成它需要的分布，算完后再还原给各 request 的 owner GPU，继续下一层 Attention。

| FFN / MoE 的并行方式 | 它需要的数据 | 进入前的编排 | 离开后的编排 |
| --- | --- | --- | --- |
| TP（切分权重） | **全部 token**：每个 GPU 持有权重的一部分，要对所有 token 算部分和 | All-Gather：每个 GPU 拿到所有 DP rank 的 token | Reduce-Scatter：部分和求和，同时把结果按 token 归属切回各 DP rank |
| EP（切分 Expert） | **被路由到本机 Expert 的 token** | Dispatch（All-to-All）：Router 选好 Expert 后，把 token 发往拥有该 Expert 的 GPU | Combine（All-to-All）：结果送回 token 原来的 GPU，按路由权重加权求和 |

- **共同点**：编排的对象是 token 的 hidden state，只是临时搬运；request 的 Attention owner 和 KV Cache 始终不动。
- **取代了什么**：这两次集合通信取代了 TP Attention 内的 AllReduce。
- 两种搭配（DPA + EP、DPA + TP）的完整流程和三种方案的通信对比，见 `dpa_ep.md` 第 5、6 节。

## 5. 代价与适用场景

DP Attention 不是免费的，它把 Attention 的通信问题换成了下面这些代价：

| 代价 | 说明 |
| --- | --- |
| Attention 权重复制 | 每个 DP rank 存一整份 Attention 权重，集群多占 $(P-1) \times A$ 的显存（详见下文） |
| 与 FFN / MoE 之间的集合通信与同步 | 取代了 Attention 内的 AllReduce；各 DP rank 需同步参与，快的要等慢的（详见 `dpa_ep.md` 第 7 节） |
| request 粘性 | request 必须固定在某个 DP rank 上；prefix cache 的命中也受限于 request 被路由到哪个 rank |
| 低并发无收益 | batch 很小时 KV 读取压力不大，DP 切分反而让部分 GPU 空转 |

### 5.1 Attention 权重复制的代价

设 Attention 总权重为 $A$，DP 度数为 $P$：

- TP 方式：每个 GPU 存 $A/P$，集群一共存 $A$；
- DP 方式：每个 GPU 存 $A$，集群一共存 $P \times A$，多占 $(P-1) \times A$ 显存。

这个代价是否划算，取决于 $A$ 在模型里占多大，以及复制换来了什么（以下参数量由模型的 `config.json` 按层数、维度等配置计算得出，非实测权重统计）：

| | 稠密模型（以 Llama-3-70B 为例） | MoE 模型（以 DeepSeek-V3 为例） |
| --- | --- | --- |
| Attention 参数 | 约 121 亿，占总参数约 17% | 约 114 亿，占总参数 6710 亿的约 1.7% |
| DP=8 的复制开销 | 每卡约 121 亿参数；TP=8 切分时每卡仅约 15 亿 | 每卡约 114 亿参数；Experts 是切分而非复制，不受 DP 影响 |
| 换来的收益 | 几乎没有：MHA/GQA 的 KV 本就能按 head 切，TP Attention 不存在"KV 无法切分"的问题 | KV 容量随 GPU 数线性增长，省掉 Attention 的 AllReduce |
| 结论 | 不划算 | 相对整个模型代价小，通常值得 |

**关键是乘上复制份数**：Attention 参数占比是按**一份**算的，而 DP 要复制 P 份，多占的显存相对整个模型的比例就是 $(P-1) \times \text{占比}$。取 $P=8$：

| | Attention 占比 | 多占的显存 | 相当于整个模型的 |
| --- | --- | --- | --- |
| 稠密（Llama-3-70B） | 约 17% | 约 $7 \times 121$ 亿 ≈ 846 亿 | **约 120%** |
| MoE（DeepSeek-V3） | 约 1.7% | 约 $7 \times 114$ 亿 ≈ 800 亿 | 约 12% |

所以差别不在"Attention 是否比 FFN 小"（稠密模型里它也只占总参数的约 1/6，MHA 时可达约 1/3），而在**占总参数的比例乘以复制份数后能否接受**：稠密模型复制 8 份后多占的显存比整个模型还大，MoE 模型复制 8 份也只多占一成多。

**"占比小"的含义**：上面的"小"是相对**整个模型的总参数**而言。从**单个 GPU** 看，每张卡都要放一整份 Attention，而它分到的 Experts 份额是总 Experts ÷ GPU 数。集群规模越大，每张卡分到的 Experts 越少，复制的 Attention 在单卡显存里的占比就越高。所以更准确的说法是：DP Attention 的复制代价相对整个模型小，但随集群规模变大，在单卡显存中的占比会上升。

### 5.2 适用场景

是否使用 DP Attention，要同时看"用不用得起"和"值不值得用"：

| 条件 | 回答的问题 |
| --- | --- |
| Attention 参数占比小 | **用得起吗**：复制 Attention 权重的显存代价能否接受 |
| KV 难以按 head 切分，或 KV 的容量 / 带宽成为瓶颈 | **值得用吗**：DP Attention 能带来什么好处 |
| 高并发 Decode | **收益有多大**：batch 大时 KV 读取压力才大；低并发时 DP 切分反而让部分 GPU 空转 |

占比小只是"用得起"的前提，代价低但没有收益，仍然没必要用。几种典型情况：

| 情况 | 判断 |
| --- | --- |
| **MoE + MLA**（如 DeepSeek-V3） | Attention 占比小，且 KV 是共享的 latent，TP 下只能复制或跨卡通信。代价低、收益大，最适合 |
| **MoE + GQA** | Attention 占比同样可能很小，但 KV 有多组，TP 度数不超过 KV 组数时仍可按组切分，没有严重的复制问题，收益没那么明显。是否使用看 TP 度数是否超过 KV 组数，以及并发和 KV 容量是否吃紧 |
| **稠密模型** | Attention 占总参数约 1/6 到 1/3，复制 P 份后多占的显存可能超过整个模型，收益又有限，通常不划算 |

## 6. 小结

MLA 与 DP Attention 解决的是两个不同层次的问题：

- **MLA**：解决"KV Cache 太大"，通过 latent 压缩大幅减少 KV 的容量与读取量；
- **DP Attention**：解决"MLA + TP Attention 下 KV 无法按 head 切分，导致复制或跨卡通信"，让 Attention 与其 KV 在同一 GPU 上闭环。

所以在 DeepSeek-V3 这类 MLA + MoE + 大规模多卡的系统里，不能只看"KV 压缩了多少"，还要看 KV 在哪些 GPU 上、是否被复制、每个 token 的 Attention 需要多少跨 GPU 流量。最终目标不是单纯让 KV 更小，而是让 KV 尽可能本地，同时让 Attention 尽可能少做 GPU 间通信。
