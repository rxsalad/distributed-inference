# Decoder-only 架构

> Llama、DeepSeek、Qwen 等主流 LLM 都是 Decoder-only。本文讲模型结构与训练，按"核心思路 → 模型结构 → 训练"展开；推理流程（Prefill / Decode、KV Cache、GPU 与 CPU 分工、并行策略）见 `inference.md`。

## 1. 核心思路

Decoder-only 把所有任务统一成一件事：**给定前缀，预测下一个 token**。

- prompt 只是已知的前缀，生成就是不断往后续写；prompt 与生成内容是**同一条序列**；
- 注意力始终是因果的（Masked Self-Attention）：位置 i 只能看到 ≤ i 的 token；
- 结构简单、易于扩大规模，推理时只需维护一份 KV Cache。

## 2. 模型结构

<img src="images/gpt-decoder-only-architecture.png" alt="GPT Decoder-only 架构图" width="560" />

*图：GPT（Decoder-only）完整架构。左：整体结构；右：单个 Block 的展开，Pre-Norm（LayerNorm 在子层之前），Attention 子层内每个 head 做 Matmul → Mask → Softmax → Matmul，随后是 Linear 输出投影（W_O），FFN 子层为 Linear → GELU → Linear，两个子层各有残差连接。来源：[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Full_GPT_architecture.svg)（CC0）。该图是 GPT-2 风格（LayerNorm、GELU、Dropout），现代 Llama 类模型的差异见 2.2。*

### 2.1 整体流程

```
采样（temperature / top-k / top-p）→ 下一个 token
   ▲
   │
LM Head (Linear)               d_model → vocab，得到 logits
   ▲
   │
Final Norm
   ▲
   │
┌─────────────────────┐
│  Decoder Block × L   │      L 层结构相同、参数各自独立
└─────────────────────┘
   ▲
   │
Embedding                      查表：token id → d_model 维向量
   ▲
   │
token ids
```

- **Embedding**：把离散的 token id 映射成连续向量，是模型的入口。
- **位置信息**：早期模型把 Positional Encoding 加在输入上（如图）；现代模型多用 RoPE，在每一层 Attention 内对 Q、K 做旋转，输入端不再加位置编码。
- **Final Norm + LM Head**：只在最后出现一次。LM Head 是一个大线性层，把 hidden state 从 $d_{model}$ 维映射到词表维度（vocab），得到每个候选 token 的分数 logits。softmax 通常不对全词表计算：先经 temperature / top-k / top-p 过滤，再对候选归一化采样。
  LM Head **不属于 FFN**。词表大小因模型而异：经典 Transformer 约 2 万，现代 LLM 常见 3 万到 15 万以上。每个 Decode 步都要执行 `[batch, d_model] × [d_model, vocab]` 这个大 GEMM，是采样阶段的主要开销。

### 2.2 单个 Decoder Block

现代模型通常使用 **Pre-Norm**（常用 RMSNorm），一个 Block 的计算为：

```
h   = x + Attention(Norm(x))      # Attention 子层 + 残差
out = h + FFN(Norm(h))            # FFN 子层 + 残差
```

两个子层的经典形态与现代常见形态：

| 子层 | 经典形态 | 现代常见形态 |
| --- | --- | --- |
| Attention | Multi-Head Attention（MHA） | GQA（多个 Q head 共享一组 KV）、MLA（共享压缩的 latent KV） |
| FFN | 两层线性 + ReLU/GELU | SwiGLU；大模型常把 FFN 替换成 MoE（多套 Expert + Router） |

Attention 的变体直接决定 KV Cache 的大小和并行方式（MLA 与 TP 的矛盾见 `dp_attention.md`），FFN 换成 MoE 则引出 Expert Parallel（见 `dpa_ep.md`、`moe.md`）。

**哪些操作 token 间有依赖**：Block 内只有 Attention 核心（$QK^\top$ 与对 V 的加权求和）让 token 之间互相依赖；其余操作（Q/K/V 投影、输出投影 W_O、FFN、Norm）都是 Position-Wise，即每个 token 用相同参数独立计算，可以按 token 任意切分。

### 2.3 Attention 详解

**计算过程**：每个 token 经 Q、K、V 三个线性变换得到自己的 Q、K、V；用自己的 Q 与所有**可见**（受 Causal Mask 限制）的 K 计算 attention 权重，再用权重对相应的 V 加权求和，得到该 token 的输出。

**多头与输出投影 W_O**：以 $d_{model}=512$、8 个 head、每个 head 64 维为例：

1. 8 个 head 并行计算，每个 head 为每个 token 输出一个 64 维向量；
2. 在特征维上拼接（concat）为 $8 \times 64 = 512$ 维；
3. 经输出投影 $W_O$（`nn.Linear(n_heads * d_v, d_model)`）映射回 $d_{model}$。

拼接只是把各 head 的输出"堆"在一起，$W_O$ 才让每个输出维度综合利用所有 head 的特征，起到**融合**作用。$W_O$ 属于 Attention 模块，**不属于 FFN**。

**Causal Mask 与 FlashAttention**：Causal Mask 把未来位置设为不可见，只限制可见性，不限制计算并发。理论上标准实现会算出约一半最终被 Mask 掉的 $QK^\top$ 结果；FlashAttention 等现代实现通过 kernel fusion、分块（blocking）和 online softmax，既不显式生成完整的 Attention Matrix，也按 block 跳过上三角区域，实际浪费远小于理论值。

### 2.4 Position-Wise FFN

对每个 token 位置，使用同一组 FFN 参数独立变换，token 之间没有信息交换；参数量与序列长度无关。每个 Block 里有一个 FFN（Llama 等为 SwiGLU），L 层就有 L 个 FFN。

**通向 MoE**：当模型规模增长、参数主要堆在 FFN 上时，一个自然的扩展是准备多套 FFN 参数（Experts），由 Router 根据每个 token 的 hidden state 选择其中几套计算。这就是 MoE 的出发点（见 `moe.md`）。

## 3. 训练

**数据组织**：把输入和输出拼成**同一条序列**，一次前向送入（Teacher Forcing）；标签就是这条序列错开一位，位置 i 预测第 i+1 个 token：

```
序列:    x1   x2   x3   x4   x5
输入:    x1   x2   x3   x4          ← 取前 n-1 个
标签:         x2   x3   x4   x5     ← 取后 n-1 个
```

- **并发**：每层把整条序列作为一个大矩阵 `[N, d_model]` 一次算完；Causal Mask 保证位置 i 看不到 i 之后的 token，不会偷看答案。一次前向得到 n-1 个训练信号。
- **损失**：每个位置的 logits 与下一个 token 求交叉熵，所以**每个位置**都要过 LM Head。预训练时每个位置都算 loss；指令微调（SFT）时通常用 loss mask 屏蔽 prompt 部分，只对回答部分算 loss（prompt 位置只当上下文）。
- **反向传播**：需要保存激活值；不需要 KV Cache（整条序列一次算完，用完即丢）。

## 4. 从训练到推理

训练时整条序列（输入加输出）一次送入，所有位置并发算完；推理时后面的 token 还没生成，必须先用 **Prefill** 处理 prompt，再用 **Decode** 逐 token 续写，并借助 KV Cache 避免重复计算。这部分见 `inference.md`。
