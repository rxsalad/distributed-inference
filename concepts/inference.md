# Decoder-only 模型的推理

> 承接 `decoder-only.md`（模型结构与训练）：本文讲推理阶段，按"Prefill 与 Decode → KV Cache → 一个推理步中的 GPU 与 CPU 分工 → 与分布式推理并行策略的对应"展开。并行与集合通信的概念见 `parallelism-basics.md`。

## 1. Prefill 与 Decode

推理分两个阶段：先用 **Prefill** 处理整个 prompt，再用 **Decode** 逐 token 续写。

### 1.1 Prefill：处理 prompt

1. **每层一个大矩阵**：N 个 prompt token 拼成 `[N, d_model]` 一次送入，每层的每个操作只调用一次，同一层内 N 个 token 并发；层间串行（第 L+1 层要等第 L 层的输出）。
2. **Mask 只管 Attention 的可见性**：在 $QK^\top$ 的分数矩阵上，softmax 之前把未来位置设为 $-\infty$，使位置 i 只能看到 ≤ i 的 token。这样 N 个位置可以同时算，结果又与逐个生成一致。Q/K/V 投影、FFN、Norm 逐 token 独立，不需要 Mask。
3. **写入 KV Cache**：每层把这 N 个 token 的 K、V 存下来，供 Decode 复用。
4. **只取最后一个位置出结果**：每层 N 个位置都产出 hidden state，但最后只有**最后一个位置**的 hidden state 过 LM Head，得到 logits 并采样出第一个新 token；其余 N-1 个位置的输出只用于推进后续层的计算。

关于第 4 步：Mask 管的是"谁能看见谁"，输出哪个位置由这一步决定。主流推理引擎在进入 LM Head 之前就先切出最后一个位置，不会对全部 N 个位置算完 logits 再丢弃（否则 `[N, d_model] × [d_model, vocab]` 的大 GEMM 和 `[N, vocab]` 的 logits 显存都是浪费）。例外：请求要求返回 prompt 每个 token 的概率（如 `prompt_logprobs` / `echo`）或做 perplexity 评估时，才需要对所有位置算 LM Head。

#### "并发"到底指什么

Prefill 的并发**不是**"每个 token 一个进程/线程"，而是**一次矩阵运算包含全部 token**：

- **软件层面**：每个操作只调用一次，没有"for 每个 token"的循环。权重（如 $W_Q$）只读一次，就服务全部 N 个 token，这是 Prefill 算力利用率高的原因。Attention 虽有 token 间依赖，但依赖的是 K/V；K/V 经投影后一次性全部算出，所以 $QK^\top$ 也是一次矩阵乘。
- **硬件层面**：GPU 执行大 GEMM 时，把输出矩阵切成许多 tile（如 128×128），分给几十到上百个 SM，每个 SM 内再由成千上万个线程并行计算。这种切分**不与 token 一一对应**：一个 tile 可能覆盖多个 token 的部分特征维度，一个 token 的计算也可能分散在多个 tile，所以不能说"一个 SM 处理一个 token"。
- **长 prompt**：N 很大时会做 chunked prefill，把 prompt 切成若干块依次处理，以控制显存和延迟，并便于与 Decode 请求混合调度；每一块内部仍是并发的。

**与多卡并行的区别（容易混淆）**：

| | 范围 | 切分方式 |
| --- | --- | --- |
| 单卡内 | 一个大矩阵运算 | 硬件自动切 tile，与 token 不对应 |
| 多卡间 | EP：不同 token 被 Router 发往不同 GPU 的 expert；DPA：不同 request 由不同 GPU 负责 | 人为按 token / request 切分 |

`decoder-only.md` 2.2 所说"Position-Wise 部分可以按 token 任意切分"指的是多卡间这一层：这类计算**允许**被切分到不同 GPU，并不是说单卡上每个 token 被单独计算。

### 1.2 Decode：逐 token 续写

Decode 无法像训练那样一次算完，因为后面的 token 还没生成：必须自回归、逐 token 串行。每一步只处理上一步刚生成的那 1 个 token：

1. Embedding 得到该 token 的向量；
2. 每一层：只为这个新 token 计算 Q、K、V，把新 K、V **追加**到 KV Cache，再用新 Q 与**全部**历史 K、V 做 Attention，然后过 FFN；
3. 最后一层输出过 LM Head，采样出下一个 token；
4. 遇到 EOS 或达到最大长度则停止，否则回到第 1 步。

### 1.3 对照

**Prefill 与训练的前向几乎一样**（训练的细节见 `decoder-only.md` 第 3 节）：都是整条序列一次送入的大 GEMM（算力型），用同一套 kernel（如 FlashAttention）。可以把 Prefill 看成"只做前向、只取最后一个位置输出、顺手把 K/V 存下来的训练前向"。

| | 训练 | 推理 Prefill | 推理 Decode |
| --- | --- | --- | --- |
| 每步处理 | 整条序列 | 整个 prompt | 1 个新 token |
| LM Head | **每个位置**都算 | 只算**最后一个位置** | 只算这 1 个位置 |
| 损失 / 反向 | 有 | 无 | 无 |
| KV Cache | 不需要 | 写入 | 追加并读取全部历史 |
| 瓶颈 | 算力 | 算力 | 带宽（memory-bound） |

LM Head 的差别在词表很大时很可观：训练要算 `[N, d_model] × [d_model, vocab]`，Prefill 只需其中 1 行。推理因此是"Prefill（算力型）+ Decode（带宽型）"两类负载，训练只有前者。

**Decode 的两类瓶颈**：
- Attention：Q 只有 1 个 token，却要读该序列全部历史 KV Cache，是 memory-bound（MLA 压缩这份 KV，DP Attention 让它保持本地）；
- FFN：单序列时退化为 GEMV（矩阵 × 向量）；多序列 batch 后也只是小 batch GEMM，受权重读取带宽限制，算力喂不满。

## 2. KV Cache

**为什么能缓存**：Causal Mask 保证已有 token 看不到后面的 token，所以历史 token 的 K、V 一旦算出就不会再变。Decode 时直接复用，只需为新 token 计算增量，否则每步都要重算整个前缀。

**大小**（MHA / GQA）：每个 token 每层存一份 K 和一份 V，

$$\text{KV 大小/token} = 2 \times L \times n_{kv} \times d_{head} \times \text{bytes}$$

例如 Llama-3-8B（$L=32$，$n_{kv}=8$，$d_{head}=128$，bf16 每个元素 2 字节）：$2 \times 32 \times 8 \times 128 \times 2 = 131072$ 字节，即每 token 128 KB；一条 4096 token 的序列约 512 MB。
**不同 Attention 方案的 KV 大小**：KV 的大小由"每 token 每层缓存多少元素"决定，$\text{KV 大小/token} = L \times (\text{每层元素数}) \times \text{bytes}$。几种方案的区别在于 KV head 的数量，也决定了 TP 能否按 head 切分 KV：

| 方案 | KV head 数 | 每 token 每层缓存元素数 | TP 下按 head 切分 KV | 例子（`config.json`） |
| --- | --- | --- | --- | --- |
| MHA | $n_h$（与 Q head 数相同） | $2 n_h d_h$ | 可以，TP ≤ $n_h$ | Llama-2-7B（32 个 KV head） |
| GQA | $n_{kv} < n_h$ | $2 n_{kv} d_h$ | 可以切到 TP ≤ $n_{kv}$，更大的 TP 需要复制 | Llama-3-8B / 70B（8 个 KV head） |
| MQA | 1 | $2 d_h$ | 只有 1 组，TP > 1 就要复制 | Falcon-7B（`multi_query: true`） |
| MLA | 1 份被所有 head 共享的 latent | $d_c + d_{rope}$ | 无法按 head 切分，只能复制或沿序列分片 | DeepSeek-V3（$512 + 64 = 576$） |

以 Llama-3-8B 的形状（$L=32$，$d_h=128$，bf16）为基准，只改变 KV head 数，每 token 的 KV 大小如下；MLA 用 DeepSeek-V3 的实际配置（61 层）对照，是不同模型，仅作量级参考：

| 方案 | 每 token 每层元素数 | 每 token（全部层） | 4096 token 的序列 |
| --- | --- | --- | --- |
| MHA（32 个 KV head） | 8192 | 512 KB | 约 2 GB |
| GQA（8 个 KV head） | 2048 | 128 KB | 约 512 MB |
| MQA（1 个 KV head） | 256 | 16 KB | 约 64 MB |
| MLA（DeepSeek-V3，61 层） | 576 | 约 69 KB | 约 275 MB |

KV 越小，容量和读取压力越小；但 KV head 越少，TP 按 head 切分 KV 就越受限。MLA 把这个矛盾推到了极致，为什么它与 TP 不匹配、DP Attention 如何解决，见 `dp_attention.md`。

**对性能的影响**：Decode 每步都要读取全部历史 KV 以及模型权重，而计算量很小，因此是 memory-bound。序列越长、并发 request 越多，KV Cache 越占显存和带宽，这也是 MLA、GQA 和 DP Attention（让 KV 留在本地）要解决的问题。

## 3. 一个推理步中的 GPU 与 CPU

**一个推理步 = 一次前向 + 采样**（Decode 时每个 request 每步生成 1 个 token）。CPU 的参与粒度是"每个推理步一次"，不是每层一次：

1. **步前（CPU）**：调度器决定本步哪些 request 进 batch（continuous batching），为新增 token 分配 KV Cache 块，准备输入（token id、位置、block table 等）；
2. **启动（CPU）**：把本步的 GPU 工作提交出去；
3. **前向 + 采样（GPU）**：各层由 GPU 异步顺序执行，层间衔接不需要 CPU 逐层调度（MoE 的 Router 选 expert 也在 GPU 上完成）；
4. **步后（CPU）**：拿到 token id，做停止判断（EOS、最大长度、stop 字符串）、detokenize（token id → 文本）、流式返回、更新 request 状态，再回到第 1 步。

一个 request 生成 N 个 token，CPU 就参与 N 次这样的循环（Prefill 算第 1 次）。

**采样在 GPU 上完成**：logits（`[batch, vocab]`，词表可达十几万）本来就在 GPU 显存里，temperature 缩放、top-k / top-p 过滤、softmax 和随机采样都是并行友好的向量操作，直接在 GPU 上做；搬到 CPU 会让每步 Decode 多出一次大数据拷贝。采样完只需把选出的 **token id**（每个序列几个字节）拷回 CPU。例外：带结构化输出约束（如 JSON schema）时，"哪些 token 合法"的 mask 常由 CPU 生成，再交给 GPU 参与采样。

**性能影响**：Decode 每步 GPU 计算很短，CPU 提交 kernel 的开销和步前步后的串行处理都可能让 GPU 空等。因此引擎常用 CUDA Graph（把一整步的 kernel 序列录制后一次重放）减少提交开销，并让 CPU 调度与 GPU 计算重叠。

## 4. 与分布式推理并行策略的对应

TP / DP / EP 等并行方式以及 AllReduce、All-to-All 等集合通信的含义，见 `parallelism-basics.md`。

| 部分 | 特点 | 并行粒度 | 对应方案 |
| --- | --- | --- | --- |
| Attention | token 间依赖，依赖 KV Cache | request | DP Attention（KV 本地）或 TP（按 head 切分） |
| FFN / MoE | Position-Wise | token | TP 切分权重，或 MoE 用 Expert Parallel（All-to-All） |
| LM Head | 大 GEMM，每步都执行 | 可按词表维度切分 | TP |
| Embedding | 查表 | — | 通常随 TP 切分词表 |

整个 Decoder-only 模型的推理，就是把 Prefill（算力型）和 Decode（带宽型）两类负载，在上述 Attention / FFN / MoE 的不同切分方式上组合调度。DeepSeek 一类的 MLA + MoE 模型采用 **Attention 用 DP、MoE 用 EP** 的组合，原因见 `dpa_ep.md`。
