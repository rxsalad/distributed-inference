# Concepts：LLM 分布式推理核心概念

围绕"**MLA + MoE 大模型如何在多 GPU 上推理**"这条主线，整理的一组概念文章。从 Decoder-only 模型的结构讲起，逐步引出 KV Cache、并行方式、MoE，最后落到 **DP Attention + Expert Parallel** 的组合方案。

## 阅读顺序

```
decoder-only ──► inference ──► parallelism-basics ──► moe ──► dp_attention ──► dpa_ep ──► ep_values
 模型结构与训练    推理流程        并行与集合通信          MoE     Attention 的 DP     DPA + EP 组合     EP 多机价值
```

按顺序读最顺畅；已有相关背景的话，可以按下表的"前置"挑着读。

## 文章一览

| 文章 | 内容 | 建议前置 |
| --- | --- | --- |
| [`decoder-only.md`](decoder-only.md) | Decoder-only 的核心思路、模型结构（Block、Attention、FFN）、训练（Teacher Forcing、损失） | — |
| [`inference.md`](inference.md) | Prefill 与 Decode、KV Cache（含 MHA / GQA / MQA / MLA 对比）、一个推理步中的 GPU 与 CPU 分工 | `decoder-only.md` |
| [`parallelism-basics.md`](parallelism-basics.md) | DP / TP / PP / EP 对照；AllReduce、Reduce-Scatter、All-Gather、All-to-All；TP 为什么每层都要 AllReduce | `inference.md` |
| [`moe.md`](moe.md) | 稠密模型与 MoE 模型、Router 与 Expert、总参数与激活参数、收益与代价 | `decoder-only.md` |
| [`dp_attention.md`](dp_attention.md) | MLA 的引入、MLA + TP 的矛盾、DP Attention 的思路、复制代价与适用场景 | `inference.md`、`parallelism-basics.md` |
| [`dpa_ep.md`](dpa_ep.md) | Attention 用 DP、MoE 用 EP 的完整流程；DPA + TP 的另一种搭配；通信与负载挑战 | `moe.md`、`dp_attention.md` |
| [`ep_values.md`](ep_values.md) | MoE 的 Expert 分布到多机多卡（EP）的价值：容量与显存扩展、"机内 TP + 跨机 EP"、与 DPA 配合、稀疏 All-to-All 的通信量（1/P 因子）、执行 dispatch 的通信库、代价与边界 | `moe.md`、`dpa_ep.md` |

## 按问题找文章

| 想了解 | 去哪看 |
| --- | --- |
| Prefill 时 N 个 token 到底怎么"并发" | `inference.md` 1.1 |
| 训练和 Prefill 有什么区别 | `inference.md` 1.3、`decoder-only.md` 第 3 节 |
| KV Cache 有多大，不同 Attention 方案差多少 | `inference.md` 第 2 节 |
| 采样在 CPU 还是 GPU，CPU 在一个推理步里做什么 | `inference.md` 第 3 节 |
| AllReduce、All-to-All 等集合通信是什么 | `parallelism-basics.md` 第 3 节 |
| TP 为什么每层都要 AllReduce，代价在哪 | `parallelism-basics.md` 第 4、5 节 |
| 稠密模型和 MoE 模型有什么区别 | `moe.md` 第 1、3 节 |
| MLA 是什么，为什么和 TP 不匹配 | `dp_attention.md` 第 2、3 节 |
| DP Attention 的代价是什么，什么情况下适用 | `dp_attention.md` 第 5 节 |
| DPA 如何与 MoE / FFN 配合 | `dpa_ep.md` 第 4、5、6 节 |
| DPA + EP 有哪些挑战 | `dpa_ep.md` 第 7 节（流程级）；`ep_values.md` 第 6 节（完整代价表） |
| MoE 的 Expert 分到多台机器有什么好处 | `ep_values.md` |
| 为什么是"机内 TP + 跨机 EP" | `ep_values.md` 第 3 节 |
| EP 的 All-to-All 通信量怎么算，为什么比 DPA + TP 小 P/k 倍 | `ep_values.md` 第 3 节"EP 的通信量" |
| MoE 的 token 散落（Dispatch/Combine）用哪些通信库 | `ep_values.md` 第 4 节"执行这次散落的通信库" |
| Expert 热点在生产上怎么缓解（冗余 Expert / 周期性重映射） | `ep_values.md` 第 6 节"EPLB" |

## 数据来源与说明

- **模型参数量、KV 大小**：由 Hugging Face 上各模型的 `config.json`（DeepSeek-V3、Llama-2-7B、Llama-3-8B / 70B、Falcon-7B）按层数、维度等配置计算得出，不是对实际权重文件的统计；
- **通用机制类说法**（如 CUDA Graph、chunked prefill、各集合通信的实现算法与通信量系数、引擎如何选择通信形式）是通用情况，未对照具体推理引擎的源码，不同实现可能有差异；个别引擎相关陈述（vLLM 的 EP 部署口径、all2all backend、EPLB 等）已在正文标注"以 vLLM 为例"或文档口径；
- 图片放在 [`images/`](images/)。其中 GPT 架构图来自 [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Full_GPT_architecture.svg)（CC0）。
