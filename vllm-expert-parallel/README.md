# vLLM Expert Parallel 实验：DeepSeek-V3 在 B300 集群上的多节点部署

在 4 个 B300×8 GPU 节点（共 32 张 GPU）上，对 DeepSeek-V3（FP8）做多种并行方式的部署与行为实验。这些方案来自 [recipes.vllm.ai 的 DeepSeek-V3 B300 页面](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V3?hardware=b300)（2 节点方案的四个变体）。

- **做了什么**：部署、监控、分析进程与通信、功能与行为测试、显存与 KV 拆解、源码核对。
- **没做什么**：性能基准测试（按要求忽略），所以文档里没有吞吐和延迟的结论。
- **标注约定**（各文档通用）：**实测**＝本集群命令输出或日志；**源码核对**＝读了镜像内 vLLM 源码；**推断**＝由机制得出，未直接验证。

## 环境

| 项 | 值 |
|---|---|
| 集群 | `do-mkc1-rs-mkc1-unification`，4 个 GPU 节点 `rs-b300x8-gpu-pool-3o1v8{b,e,g,w}` |
| 每节点 | 8×B300（算力 10.3，每卡 267.69 GiB），节点内 NV18 全互联，8 块 400G RoCE 网卡（GPU i 对应 `mlx5_{2i}`） |
| 节点间 | RoCE（IPv6 ULA，每个 rail 一个 /64），需要 `NCCL_IB_GID_INDEX=3`；**不支持 Multi-Node NVLink** |
| 镜像 | `vllm/vllm-openai:latest`（vLLM 0.31.0，NCCL 2.30.7）；早期实验用 `cu130-nightly`（0.19.2rc1） |
| 模型 | `deepseek-ai/DeepSeek-V3`（早期实验用 `DeepSeek-V3-0324`） |

## 文档

### 主线：四种策略

| 文档 | 内容 | 方案 |
|---|---|---|
| [`recipe.md`](recipe.md) | recipes 页面内容整理：概念（TP/DP/EP/PP/DEP/TEP）、五个策略的命令、页面描述与命令不一致之处 | 总览 |
| [`strategy1.md`](strategy1.md) | **策略一 Multi-Node TEP**：TP16 Attention + EP16 MoE，一个实例。启动故障（MNNVL、`--disable-custom-all-reduce`）、进程、通信栈、显存与 KV、一层 MoE 的流程、TEP16 与 2×TEP8 的理论对比 | `tep-server.yaml` |
| [`strategy2.md`](strategy2.md) | **策略二 Multi-Node TP + PP**：TP8 × PP2，每阶段内 EP8。层切分 [31, 30]、两阶段显存、KV 容量约为 TP16 的 2 倍、跨节点只有点对点（每步每节点 32 个 RDMA write） | `tppp-server.yaml` |
| [`strategy3.md`](strategy3.md) | **策略三 DEP**：DP16 + EP16，hybrid-lb，两个节点都对外。进程（每节点 8 个 APIServer / EngineCore / Worker）、APIServer 与 EngineCore 的对应关系、DP wave 与陪跑、显存与 KV | `dep-server.yaml` |
| （见 `recipe.md` 第 7 节） | **策略四 TEP**：页面上它的命令与策略一逐字相同（TP16 + EP16），所以策略一实际就是 TEP；页面描述与命令对不上的情况已记录 | — |
| （见 `recipe.md` 第 8 节） | **策略五 PD 分离**：只整理了页面内容，没有部署 | — |

### 专题分析

| 文档 | 回答的问题 |
|---|---|
| [`dummy-batch-why.md`](dummy-batch-why.md) | DP + EP 下为什么没有请求的 rank 也要陪跑（dummy batch）？是 `allgather_reducescatter` 造成的吗？是因为只有一个 EP 组吗？副作用是什么？ |
| [`fabric-rail-analysis.md`](fabric-rail-analysis.md) | 这些实验对 GPU Fabric 的要求是什么？rail-only 有什么问题，是否一定要 rail-optimized？含本集群的跨 rail 连通性实测 |

### 背景与早期实验

| 文档 | 内容 |
|---|---|
| [`expert-parallel.md`](expert-parallel.md) | vLLM 官方 Expert Parallel 文档的学习笔记（EP 大小 = TP × DP 等） |
| [`vllm-dp8-ep-single-node.md`](vllm-dp8-ep-single-node.md) | 单节点 DP8 + EP8（早期实验，`DeepSeek-V3-0324`） |
| [`vllm-dp16-ep-two-node.md`](vllm-dp16-ep-two-node.md) | 双节点 DP16 + EP16，内部 LB、节点 2 headless（早期实验）。DeepEP 在 B300（sm_103）上不可用、`NCCL_IB_GID_INDEX` 的必要性、吞吐粗测。第 4.2.1、5.1.1 节是后来补充的 |

## 部署文件

| 文件 | 说明 | 资源名 |
|---|---|---|
| `tep-server.yaml` | 策略一：StatefulSet 2 副本，TP16 + EP16 | `tp16`、`tp16-headless`、`tp16-api`（只指向 `tp16-0`） |
| `tppp-server.yaml` | 策略二：StatefulSet 2 副本，TP8 × PP2 | `tppp`、`tppp-headless`、`tppp-api`（只指向 `tppp-0`） |
| `dep-server.yaml` | 策略三：StatefulSet 2 副本，DP16 + EP16，hybrid-lb | `dep`、`dep-headless`、`dep-api`（指向两个 Pod） |
| `n-server.yaml`、`n-server1.yaml` | 早期实验用的 4 个 Pod（`sleep infinity`，在 Pod 里手动起 vLLM）；两者只差镜像（`cu130-nightly` 对 `v0.31.0`）和是否预下载模型 | `server` |

每个方案一个 yaml：每个 Pod 占 8 张 GPU 与 8 块 RoCE 网卡，两个 Pod 反亲和到不同节点，HF 缓存用 hostPath 保留。主节点 IP 用“循环直到非空”的写法解析（修过一次 DNS 竞态）。

**常用命令**：

```bash
kubectl apply -f <yaml>                       # 部署
kubectl logs -f <pod>                         # 日志，启动约 14–22 分钟
kubectl exec <pod> -- curl -s localhost:8000/v1/models   # 验证（策略一、二是 *-0）
kubectl delete -f <yaml>                      # 停止，HF 缓存保留
```

## 关键结论速览

| | 策略一 TP16+EP16 | 策略二 TP8×PP2 | 策略三 DP16+EP16 |
|---|---|---|---|
| 一个实例的 EngineCore 数 | 1 | 1 | 16（每 DP rank 一个） |
| 每卡权重 | 40.47 GiB | 38.71 / 41.17 GiB | 58.02 GiB |
| KV 容量（整个实例） | 2,959,680 tokens | 6,004,640 tokens | 31,388,672 tokens（16 个 rank 合计，各存自己的请求） |
| `init engine` | 912.6 s | 737.67 s | 1137.96 s |
| 需要 `--disable-custom-all-reduce` | 需要（否则 16 个 rank 卡在 MNNVL rendezvous） | 不需要 | 不需要 |
| 跨节点通信 | 每层 2 次 TP16 all-reduce | 阶段交界处的点对点 | 每个 MoE 层 all-gather + reduce-scatter |
| 没有请求的 rank | 无这个概念 | 无陪跑 | **陪跑**（dummy batch） |

其他共同发现：
- 跨节点 GPU 数据只走标准 NCCL over RoCE；NVSHMEM、NIXL、UCX 库被加载但没有被使用。
- 必须设置 `NCCL_IB_GID_INDEX=3`，否则 NCCL 选到链路本地 GID，跨节点第一次集合通信静默卡死。
- 这个集群的 fabric 跨 rail 可达（实测），不是 rail-only。
- 各方案的 RoCE 序列错误计数（`packet_seq_err`）都有不同程度的增长，原因没有排查。

## 目录约定与现状

- 实验在 `dev.rshue.com` 上进行，包括通过 K8S 访问 GPU 节点；本目录路径 `/root/work/data/distributed-inference/vllm-expert-parallel`。
- 当前运行的实例：`dep`（策略三，节点 3o1v8b、3o1v8e）和 `tppp`（策略二，节点 3o1v8g、3o1v8w），4 个节点的 GPU 全部占用。策略一（`tp16`）已停止。
- 没有保留测试脚本，文档里只写了测试方法。
- 备份文件在远端 `/tmp`（如 `*.bak`），重启会丢。

## 未做与后续

- 性能对比：TP16、2×TP8、TP8×PP2、DEP（按要求没做）。
- 去掉 `--enable-expert-parallel` 的纯 TP16（页面没有给）；页面描述的“节点内 TP8 + 节点间 DP2”形态。
- hybrid-lb 与内部 LB 在相同环境下的对比。
- PD 分离（策略五）的部署。
- 真正 rail-only 环境、NCCL 拓扑日志（`NCCL_DEBUG=INFO`）的验证。
