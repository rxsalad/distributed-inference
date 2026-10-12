# Concepts：LLM 分布式推理核心概念

围绕"**MLA + MoE 大模型如何在多 GPU 上推理**"这条主线，整理的一组概念文章。从 Decoder-only 模型的结构讲起，逐步引出 KV Cache、并行方式、MoE，最后落到 **DP Attention + Expert Parallel** 的组合方案，以及把 EP 度做大的 Wide EP。

## 阅读顺序

```
decoder-only ──► inference ──► parallelism-basics ──► moe ──► dp_attention ──► dpa_ep ──► ep_values
 模型结构与训练    推理流程        并行与集合通信          MoE     Attention 的 DP     DPA + EP 组合     EP 多机价值
```

按顺序读最顺畅；已有相关背景的话，可以按下表的"前置"挑着读。

## 方案总览

Attention 与 FFN / MoE 可以各选一种并行方式，命名约定为"Attention 并行 + FFN / MoE 并行"（D = Attention 用 DP，即 DPA；T = TP；E = EP），详见 `parallelism-basics.md` 第 6 节：

| 方案 | 简称 | Attention | FFN / MoE | 典型适用 | 在哪讲 |
| --- | --- | --- | --- | --- | --- |
| 纯 TP | TP | TP | TP | 稠密模型 | `parallelism-basics.md` 第 4 节 |
| DPA + TP | DTP | DP | TP | Attention 需要 DP、FFN 仍用 TP | `dpa_ep.md` 第 6 节 |
| DPA + EP | DEP | DP | EP | MLA + MoE 大模型 | `dp_attention.md`、`dpa_ep.md` |
| TP + EP | TEP | TP | EP | KV 可按 head 切的 MoE 模型 | `dpa_ep.md` 第 6 节 |
| Wide EP | — | 视搭配而定 | EP 度做得很大，带负载均衡 | 超大 MoE、高速互联 | `ep_values.md` 第 6 节 |

演进的主线是：

```
TP（每层 AllReduce）
  │  MLA 把 KV 压缩成共享 latent，TP 无法按 head 切 KV
  ▼
DP Attention（KV 留在本地）  ──►  MoE 侧用 EP（All-to-All）= DEP
  │  Expert 数量与规模继续增长，热点与通信成为瓶颈
  ▼
Wide EP（EP 度做大 + 负载均衡 + 专用通信 kernel）
```

## 文章一览

| 文章 | 内容 | 建议前置 |
| --- | --- | --- |
| [`decoder-only.md`](decoder-only.md) | Decoder-only 的核心思路、模型结构（Block、Attention、FFN、LM Head）、张量形状与 batch、训练（Teacher Forcing、损失） | — |
| [`inference.md`](inference.md) | Prefill 与 Decode、KV Cache（含 MHA / GQA / MQA / MLA 对比）、一个推理步中的 GPU 与 CPU 分工、Prefill / Decode 分离（PDD） | `decoder-only.md` |
| [`parallelism-basics.md`](parallelism-basics.md) | DP / TP / PP / EP 对照；AllReduce、Reduce-Scatter、All-Gather、All-to-All；TP 为什么每层都要 AllReduce；混合并行的方案命名（TP / DTP / DEP / TEP） | `inference.md` |
| [`moe.md`](moe.md) | 稠密模型与 MoE 模型、Router 与 Expert、总参数与激活参数、收益与代价 | `decoder-only.md` |
| [`dp_attention.md`](dp_attention.md) | MLA 的引入、MLA + TP 的矛盾、DP Attention 的思路、与 FFN / MoE 的衔接、复制代价与适用场景 | `inference.md`、`parallelism-basics.md` |
| [`dpa_ep.md`](dpa_ep.md) | Attention 用 DP、MoE 用 EP 的完整流程；EP Group；DTP 与 TEP 的其他搭配及通信对比；通信与负载挑战 | `moe.md`、`dp_attention.md` |
| [`ep_values.md`](ep_values.md) | MoE 的 Expert 分布到多机多卡（EP）的价值：容量与显存扩展、"机内 TP + 跨机 EP"、与 DPA 配合、稀疏 All-to-All 的通信量（1/P 因子）、执行 dispatch 的通信库、代价与边界（EPLB、Wide EP） | `moe.md`、`dpa_ep.md` |

## 术语表

| 术语 | 含义 | 详见 |
| --- | --- | --- |
| hidden state | 层与层之间流动的 `[N, d_model]` 张量（残差流），通常指每个 Block 的输出 | `decoder-only.md` 2.2 |
| Position-Wise | 每个 token 用相同参数独立计算的操作（FFN、Norm、投影、LM Head），可按 token 任意切分 | `decoder-only.md` 2.2 |
| KV Cache | 历史 token 的 K、V，跨推理步缓存，Decode 时复用 | `inference.md` 第 2 节 |
| MLA | 把 KV 压缩成所有 head 共享的 latent，KV 极小但无法按 head 切 | `dp_attention.md` 第 2 节 |
| rank | 并行组内的一个参与者，通常对应一张 GPU | `parallelism-basics.md` 第 6 节 |
| DPA / DP rank | Attention 用 DP；每个 DP rank 独立负责一部分 request，KV 留在本地 | `dp_attention.md` 第 4 节 |
| Router / Expert | MoE 层里给每个 token 选 Expert 的小网络 / 一套独立的 FFN | `moe.md` 第 2 节 |
| EP 度 / EP Group | 参与 EP 的 GPU 数 / 共同持有全部 Expert、彼此做 All-to-All 的 GPU 组 | `dpa_ep.md` 第 2 节 |
| Dispatch / Combine | MoE 前后的两次 All-to-All：把 token 发到 Expert 所在 GPU / 把结果送回并加权求和 | `dpa_ep.md` 第 4 节 |
| EPLB / Wide EP | 缓解热点 Expert 的负载均衡措施 / 把 EP 度做大并配合负载均衡 | `ep_values.md` 第 6 节 |
| PDD | Prefill 与 Decode 放到独立实例，KV 在实例间传输 | `inference.md` 第 5 节 |

## 按问题找文章

| 想了解 | 去哪看 |
| --- | --- |
| Attention 的作用、FFN 为什么能存知识 | `decoder-only.md` 2.3、2.4 |
| N 与 batch：训练、Prefill、Decode 的张量形状 | `decoder-only.md` 2.5 |
| Prefill 时 N 个 token 到底怎么"并发"，多个 prompt 如何一起处理 | `inference.md` 1.1 |
| 训练和 Prefill 有什么区别 | `inference.md` 1.3、`decoder-only.md` 第 3 节 |
| KV Cache 有多大，不同 Attention 方案差多少 | `inference.md` 第 2 节 |
| 采样在 CPU 还是 GPU，CPU 在一个推理步里做什么 | `inference.md` 第 3 节 |
| Prefill / Decode 分离是什么 | `inference.md` 第 5 节 |
| AllReduce、All-to-All 等集合通信是什么 | `parallelism-basics.md` 第 3 节 |
| TP 为什么每层都要 AllReduce，代价在哪 | `parallelism-basics.md` 第 4、5 节 |
| TP / DTP / DEP / TEP 分别是什么 | `parallelism-basics.md` 第 6 节 |
| 稠密模型和 MoE 模型有什么区别 | `moe.md` 第 1、3 节 |
| MLA 是什么，为什么和 TP 不匹配 | `dp_attention.md` 第 2、3 节 |
| DP Attention 的代价是什么，什么情况下适用 | `dp_attention.md` 第 5 节 |
| DPA 如何与 MoE / FFN 配合 | `dp_attention.md` 第 4 节、`dpa_ep.md` 第 4、5、6 节 |
| EP Group 是什么 | `dpa_ep.md` 第 2 节 |
| TEP 与 DEP 怎么选 | `dpa_ep.md` 第 6 节 |
| DPA + EP 有哪些挑战 | `dpa_ep.md` 第 7 节（流程级）；`ep_values.md` 第 6 节（完整代价表） |
| MoE 的 Expert 分到多台机器有什么好处 | `ep_values.md` |
| 为什么是"机内 TP + 跨机 EP" | `ep_values.md` 第 3 节 |
| EP 的 All-to-All 通信量怎么算，为什么比 DPA + TP 小 P/k 倍 | `ep_values.md` 第 3 节"EP 的通信量" |
| MoE 的 token 散落（Dispatch/Combine）用哪些通信库 | `ep_values.md` 第 4 节"执行这次散落的通信库" |
| Expert 热点在生产上怎么缓解（冗余 Expert / 周期性重映射） | `ep_values.md` 第 6 节"EPLB" |
| Wide EP 是什么 | `ep_values.md` 第 6 节"Wide EP" |

## 数据来源与说明

- **模型参数量、KV 大小**：由 Hugging Face 上各模型的 `config.json`（DeepSeek-V3、Llama-2-7B、Llama-3-8B / 70B、Falcon-7B）按层数、维度等配置计算得出，不是对实际权重文件的统计；
- **通用机制类说法**（如 CUDA Graph、chunked prefill、continuous batching 的实现方式、各集合通信的实现算法与通信量系数、引擎如何选择通信形式、TEP 的通信形式）是通用情况，未对照具体推理引擎的源码，不同实现可能有差异；个别引擎相关陈述（vLLM 的 EP 部署口径、all2all backend、EPLB 等）已在正文标注"以 vLLM 为例"或文档口径；
- **Wide EP**：来自 NVIDIA TensorRT-LLM 的文档与技术博客（链接见 `ep_values.md` 第 6 节），不是 `config.json` 计算，也未对照引擎源码；
- **单位**：KV 大小用二进制单位（KiB / MiB / GiB），权重体积与显存容量用十进制（GB / TB）；
- **简称**：TP / DTP / DEP / TEP 是本系列的命名约定（见 `parallelism-basics.md` 第 6 节），不是业界统一术语；
- 图片放在 [`images/`](images/)。其中 GPT 架构图来自 [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Full_GPT_architecture.svg)（CC0）。
