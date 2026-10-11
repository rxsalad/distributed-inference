# MoE（Mixture of Experts）

> 先区分稠密模型与 MoE 模型，再讲 MoE 层的结构、总参数与激活参数的区别、MoE 带来的收益与代价，最后落到与分布式推理的关系。文中模型参数量由 `config.json`（Hugging Face 上的 DeepSeek-V3、Llama-3）按层数、维度等配置计算得出。

## 1. 稠密模型与 MoE 模型

**稠密模型（Dense）**：处理每个 token 时，**模型的全部参数都参与计算**。每层 FFN 只有一套参数，所有 token 都走这一套。例如 Llama-3 系列：70B 就是 700 亿参数，每个 token 都要用到这 700 亿。

**MoE 模型**：把每层的 FFN 换成**多套并列的 FFN（称为 Expert）加一个 Router**，每个 token 只经过其中被选中的几个 Expert：

```
token 的 hidden state
        │
        ▼
     Router            ← 一个小的线性层，给每个 Expert 打分
        │  选 top-k（如 8 个里选 2 个）
        ▼
 ┌────┬────┬────┬────┐
 │ E1 │ E2 │ E3 │ ... │   ← 多套 FFN，每个 token 只经过被选中的几个
 └────┴────┴────┴────┘
        │ 按路由权重加权求和
        ▼
      输出
```

- **Attention 部分与稠密模型一样**，所有 token 都经过；
- **FFN 部分是稀疏的**：每个 token 只激活少数几个 Expert，其余 Expert 不参与这个 token 的计算；
- 不同 token 选的 Expert 不同。

因此 MoE 也叫**稀疏模型**，与"稠密"相对。

## 2. MoE 层的结构

| 组件 | 作用 |
| --- | --- |
| Router（门控） | 一个线性层，把 token 的 hidden state 映射成对每个 Expert 的分数，取分数最高的 top-k 个 Expert，并得到各自的路由权重 |
| Expert | 一套独立的 FFN（结构与稠密模型的 FFN 相同，参数各自独立） |
| 加权求和 | 把被选中的 k 个 Expert 的输出，按路由权重加权求和，作为这一层 FFN 位置的输出 |

要点：

- 路由是**逐 token 独立**的：每个 token 根据自己的 hidden state 选 Expert，token 之间互不影响。这与 FFN 本身"每个 token 独立计算"的性质一致，也是后面可以把不同 token 送到不同 GPU 的基础；
- 部分模型（如 DeepSeek 系列）除了被路由选择的 Expert，还设有**共享 Expert**，对所有 token 都激活。

## 3. 总参数与激活参数

| | 稠密模型 | MoE 模型 |
| --- | --- | --- |
| 总参数 | 全部参与每个 token 的计算 | 很大，但每个 token 只用其中一小部分 |
| 例子 | Llama-3-70B：总约 706 亿，每 token 全部参与 | DeepSeek-V3：总约 6710 亿，每 token 激活约 376 亿（官方口径约 370 亿） |
| 参数主要在哪 | Attention + FFN（FFN 占大头） | **绝大部分在 Experts 里** |

Experts 的数量让总参数暴涨，Attention 参数占比被稀释：DeepSeek-V3 中 Attention 只占总参数约 1.7%（稠密的 Llama-3-70B 约 17%）。这一点直接影响并行策略的选择，见 `dp_attention.md`。

**以 DeepSeek-V3 为例**（来自 `config.json`）：

| 配置项 | 数值 |
| --- | --- |
| 层数 | 61 层，其中前 3 层是稠密 FFN，其余 58 层是 MoE 层 |
| 每个 MoE 层 | 256 个路由 Expert + 1 个共享 Expert，每个 token 选 top-8 个路由 Expert |
| 参数占比 | Experts 约 6540 亿，占总参数的 97.4%；Attention 约 114 亿，占 1.7% |

也就是说，MoE 模型并不一定每一层都是 MoE，前几层可能保留稠密 FFN。

## 4. 收益与代价

**收益**：用较低的每 token 计算量，换来很大的模型容量。激活参数决定计算量，总参数决定模型能存多少知识。

**代价**：

- **显存**：虽然每个 token 只用部分 Expert，但所有 Expert 的权重都要放在显存里，总参数多，单卡放不下，必须把 Experts 分到多张 GPU 上；
- **通信**：不同 token 要去不同 Expert，而 Expert 分布在不同 GPU 上，需要在 GPU 之间搬运 token（All-to-All）；
- **负载均衡**：Router 可能让某些 Expert 被大量 token 选中（热点），拥有它的 GPU 成为瓶颈，其他 GPU 空等（工程缓解见 `ep_values.md` 第 6 节"EPLB"）；
- **小 batch 下利用率低**：batch 小时每个 Expert 分到的 token 很少，计算利用率低，Decode 阶段尤其明显。

## 5. 与分布式推理的关系

MoE 层的参数占绝大多数，需要多卡切分，最典型的是 **Expert Parallel（EP）**：把不同 Expert 放到不同 GPU，Router 选定后通过 All-to-All 把 token 发到对应 Expert 所在的 GPU，计算完成后再发回、恢复原来的 token 顺序。

因为 Attention 参数占比很小，通常对 Attention 和 MoE 采用不同的并行方式：Attention 用 DP（见 `dp_attention.md`），MoE 用 EP（两者如何衔接见 `dpa_ep.md`）。
