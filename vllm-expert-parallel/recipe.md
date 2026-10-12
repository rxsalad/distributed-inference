# vLLM Recipes：DeepSeek-V3 在 B300 上的官方部署方案

来源（2026-10-11 抓取，页面标注 Updated 2026-09-10）：
- <https://recipes.vllm.ai/deepseek-ai/DeepSeek-V3?hardware=b300>
- 以及 `&nodes=2`、`&nodes=2&strategy=multi_node_tp_pp`、`multi_node_dep`、`multi_node_tep` 四个变体

页面由 JS 动态渲染，用浏览器读取各 tab 的命令。页面里的 "2× B300" 指 2 个节点（16 张 GPU）。
概念背景见 `expert-parallel.md`，本集群上的实测见 `vllm-dp8-ep-single-node.md`、`vllm-dp16-ep-two-node.md`。

标注约定：**页面**＝recipes 页面原文；**实测**＝本集群实验结果（来自上述两份报告）；**推断**＝由 vLLM / DeepSeek 架构的一般知识得出，未在本集群验证。

## 目录
1. 模型与硬件
2. 所有多节点方案的共同设置
3. 基本概念：TP / DP / EP / PP
4. 策略一：Multi-Node Tensor Parallel（详细）
5. 策略二：Multi-Node TP + Pipeline Parallel
6. 策略三：Multi-Node Data + Expert Parallel（DEP）
7. 策略四：Multi-Node Tensor + Expert Parallel（TEP）
8. 策略五：Prefill/Decode 分离
9. 策略对比与本集群的关系
10. Blackwell FP4（B200 方案）
11. 基准测试
12. 页面没有说明的内容

## 1. 模型与硬件

| 项 | 内容 |
|---|---|
| 模型 | `deepseek-ai/DeepSeek-V3`，671B MoE，每 token 激活 37B，上下文 163,840 |
| 结构 | 61 层；Attention 为 MLA；MoE 层有 256 个路由专家，每 token 选 8 个，另有 1 个共享专家 |
| 其他检查点 | `DeepSeek-V3-0324`；Blackwell 的 FP4 版 `nvidia/DeepSeek-V3-FP4`（R1 对应 `nvidia/DeepSeek-R1-FP4`） |
| 最低 vLLM | 0.12.0+；v0.28.0+ 在 Blackwell 上自动选 FlashInfer MoE kernel（FP4/FP8），可用 `--moe-backend flashinfer_trtllm` 固定 |
| B300 显存 | 268 GB / GPU |
| 权重大小 | FP8 805 GB；FP4 403 GB |
| 页面基础要求 | FP8：8×H200；FP4：4×B200（页面没有给 B300 的单节点专用命令） |
| 安装 | `uv venv && source .venv/bin/activate && uv pip install -U vllm --torch-backend auto` |

16 张 B300 共约 4.2 TB 显存，权重不是瓶颈（推断）。

## 2. 所有多节点方案的共同设置

环境变量（`$IFACE_NAME` 为节点间通信用的网卡名）：
```bash
export GLOO_SOCKET_IFNAME=$IFACE_NAME
export NCCL_SOCKET_IFNAME=$IFACE_NAME
```

每条命令共有的参数：
```
--trust-remote-code
--enable-expert-parallel
--enable-auto-tool-choice
--tool-call-parser deepseek_v3
--chat-template examples/tool_chat_template_deepseekv3.jinja
--reasoning-parser deepseek_v3
```

使用方法：先把 `$HEAD_IP` 设为 rank-0 节点的 IP，然后每个节点各运行自己那条命令。两条命令只有 `--node-rank`（TP / TEP / TP+PP）或 `--data-parallel-start-rank`（DEP）不同。
页面对 TP 和 TEP 都注明需要 InfiniBand 互联。

## 3. 基本概念：TP / DP / EP / PP

DEP、TEP 是 recipes 页面的叫法（URL 里的 `multi_node_dep` / `multi_node_tep`），不是 vLLM 的命令行参数。

- **TP（张量并行）**：每层权重切成几份，几张卡合力算同一个 token，每层都要通信。
- **DP（数据并行）**：每张卡（或每组卡）一份完整副本，各自处理不同请求，互不通信。
- **EP（专家并行）**：只作用于 MoE 层，256 个专家整个地分到不同卡上，每卡只存、只算自己那部分。
- **PP（流水线并行）**：模型按层分成几段，每段放在不同的卡（或节点）上，前一段的输出传给后一段。
- **DEP** = Data + Expert Parallel：Attention 用 DP，MoE 用 EP。
- **TEP** = Tensor + Expert Parallel：Attention 用 TP，MoE 用 EP。

EP 的大小 = TP × DP（见 `expert-parallel.md` 第 1 节）。

### EP 到底是什么意思（以 TP16 + `--enable-expert-parallel` 为例）

EP 只改变 **MoE 层**的切分方式，Attention 层不受影响。

| | 每个专家 | 每卡存放 | MoE 通信方式 |
|---|---|---|---|
| 不开 EP（MoE 也用 TP） | 切成 16 份，16 卡各一份 | 256 个专家各 1/16 | all-reduce |
| 开 EP | 完整放在 1 张卡上 | 16 个完整专家（256 ÷ 16） | token 按路由分发到专家所在卡，再汇总 |

- 不开 EP：每个专家的权重矩阵被切碎，处理一个 token 时，被选中的 8 个专家要在 16 张卡上各算一小块，再合并。
- 开 EP：处理一个 token 时，token 被路由到它选中的专家所在的卡，在那里完整地算完这个专家，再把结果送回来合并。
- 为什么这么做（推断）：专家矩阵本身不大，切成 16 份后每块很小，GPU 算起来效率低；整个放在一张卡上，矩阵乘更大，效率更高。

所以 TP16 + EP 不是"TP 之外再加一层并行"，而是同一组 16 张卡，在 Attention 里按 TP 分工，在 MoE 里按 EP 分工。此时 EP = 16 × 1 = 16。

### TEP 与 DEP 的区别：只在 Attention 层

| | Attention 层 | MoE 层 | 命令 |
|---|---|---|---|
| TEP | TP：16 卡共同算一个 token | EP16 | `--tensor-parallel-size 16 --enable-expert-parallel` |
| DEP | DP：每卡一份完整 Attention，各算各的请求 | EP16 | `--data-parallel-size 16 --enable-expert-parallel`（TP 默认 1） |

## 4. 策略一：Multi-Node Tensor Parallel

### 4.1 是什么

**页面**：Splits the model across GPUs spanning multiple nodes. TP is set to total GPU count across all nodes. Head node serves requests, worker nodes run with `--headless`. Requires InfiniBand interconnect. 定位：Latency oriented（低延迟）。

**注意**：页面描述只说 TP，没提 EP，但下面的命令带了 `--enable-expert-parallel`，实际是 TP16 + EP16（即 TEP）。详见第 7 节末尾的对比。

即把 TP 的范围从一台机器扩展到多台机器：TP 组里的 GPU 分布在不同节点上。页面的例子是 2 节点 × 8 GPU、`--tensor-parallel-size 16`：
- GPU 0–7 在节点 0，GPU 8–15 在节点 1；
- 它们是**同一个 TP 组**，共同处理同一个 token；
- 每一层的集合通信，一部分走节点内 NVLink，一部分走节点间网络（InfiniBand / RoCE）。

**用 16 张 GPU 作为一个 vLLM 实例**：
- 一个模型副本：16 张卡合起来才是一份完整模型，每卡只持有每层权重的约 1/16；
- 一个对外入口：只有主节点（`--node-rank 0`）启动 API server（8000 端口）；
- 从节点只出算力：从节点 `--headless`，只跑 worker，不接收请求，所有请求走主节点；
- 每个请求由 16 张卡一起算，没有"某张卡负责某个请求"的概念。

### 4.2 命令（页面，2 节点 × 8 GPU，FP8）

```bash
# 两个节点都先设置
export GLOO_SOCKET_IFNAME=$IFACE_NAME
export NCCL_SOCKET_IFNAME=$IFACE_NAME

# Node 0（主节点，对外提供 API）
vllm serve deepseek-ai/DeepSeek-V3 \
  --trust-remote-code \
  --enable-expert-parallel \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr $HEAD_IP \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v3 \
  --chat-template examples/tool_chat_template_deepseekv3.jinja \
  --reasoning-parser deepseek_v3

# Node 1（从节点）：与上面完全相同，只改两处
#   --node-rank 1
#   增加 --headless
```

| 参数 | 作用 |
|---|---|
| `--tensor-parallel-size 16` | TP 组大小 = 所有节点的 GPU 总数 |
| `--nnodes 2` | 节点总数 |
| `--node-rank N` | 本节点编号（0 为主节点） |
| `--master-addr $HEAD_IP` | 0 号节点的 IP，用来建立分布式通信 |
| `--headless` | 只在从节点加：只跑 worker，不启动 API server |
| `--enable-expert-parallel` | MoE 层用 EP（见第 3 节）；页面所有多节点命令都带这个参数 |

### 4.3 每一层各部分怎么分工（本命令下）

- **Attention 层：** TP16。DeepSeek-V3 有 128 个注意力头，每卡负责 8 个（推断）。
- **MoE 层：** EP16。256 个专家分到 16 卡，每卡 16 个完整专家。
- **权重占用（推断）：** FP8 约 805 GB ÷ 16 ≈ 50 GB / 卡，B300 的 268 GB 显存还有大量余量用于 KV cache 和激活。

### 4.4 通信特点（推断）

- 每个 token 的每一层都有跨卡集合通信：Attention 输出合并一次，MoE 层再有一轮 EP 通信。61 层累积下来，每个解码步要做上百次集合通信。
- 这 16 卡里有一部分通信要跨节点，所以速度取决于节点间网络的带宽和延迟；页面强调需要 InfiniBand 的原因就在这里。
- 所有 GPU 合力算同一个 token，单请求延迟最低，这也是页面标注 "Latency oriented" 的原因。
- MLA 的 KV cache 在 TP 下每个 TP rank 各存一份，总 KV 容量不如 DP 方案。
- 只有一个 API 入口，没有跨实例负载均衡的问题，部署最简单。

### 4.5 和其他策略的区别

| | 谁和谁一起算同一个 token |
|---|---|
| 单节点 TP8 | 一个节点内的 8 张卡 |
| **多节点 TP16（本节）** | **跨两个节点的 16 张卡** |
| TP8 × PP2 | 节点内 8 卡 TP；两个节点按层分段，串行传递 |
| DP16 | 没有谁和谁一起算，每张卡独立处理各自的请求 |

### 4.6 映射到本集群的做法

已在本集群部署并测试，完整记录见 `strategy1.md`，部署文件为 `tp16-server.yaml`。关键发现：用 `vllm/vllm-openai:latest`（vLLM 0.31.0）时，页面命令会在启动阶段卡在 custom all-reduce 的 MNNVL 共享内存 rendezvous，需要额外加 `--disable-custom-all-reduce`。下面是部署前的草案要点：


- 需要两个 Pod，分别占一个节点的 8 张 GPU（可用 `n-server.yaml` 的 StatefulSet，`replicas: 2` 即可）；`$HEAD_IP` 取 server-0 的 Pod IP。
- `$IFACE_NAME` 取 Pod 的 `eth0`（`vllm-dp16-ep-two-node.md` 里控制面走 eth0）。
- 之前的实测经验（见 `vllm-dp16-ep-two-node.md` 第 6 节）：需要额外设置 `NCCL_IB_GID_INDEX=3`，否则跨节点 NCCL 初始化会静默卡死；还要设 `VLLM_ENGINE_READY_TIMEOUT_S`（默认 600 秒不够）。页面没有提到这些，需要自行加上。
- 页面没有说明 TP16 在这批 B300 机器上是否会遇到 RDMA 序列错误（`packet_seq_err`，见 `vllm-dp16-ep-two-node.md` 第 4.4 节），需要观察。

## 5. 策略二：Multi-Node TP + Pipeline Parallel

已在本集群部署并测试（功能与行为，未做性能测试），完整记录见 `strategy2.md`，部署文件为 `tppp-server.yaml`。关键发现：页面命令不需要 `--disable-custom-all-reduce`；61 层切成 [31, 30]；EP 只在每个阶段内（EP8，每卡 32 个专家）；只有 1 个 EngineCore（在节点 0），节点 1 headless；跨节点只有阶段之间的 NCCL 点对点，每个 decode 步每节点 32 个 RDMA write（8 块网卡各 4 个）；没有陪跑；KV 容量 6,004,640 tokens（TP16 的约 2.03 倍）。下面是部署前整理的页面内容：

**页面**：Tensor parallel within each node, pipeline parallel across nodes. TP stays inside the high-bandwidth intra-node NVLink/IB domain; PP is what crosses the slower cross-node link. 适合跨节点带宽成为瓶颈的超大模型（例子：16×H800 上的 Kimi-K2 1T）。定位：Throughput oriented（高吞吐）。

```bash
# Node 0
vllm serve deepseek-ai/DeepSeek-V3 \
  --trust-remote-code --enable-expert-parallel \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 0 --master-addr $HEAD_IP \
  （其余工具调用 / 推理参数同第 2 节）

# Node 1：--node-rank 1，并加 --headless
```

推断：
- 节点内 TP8，节点之间是流水线：节点 0 跑前一半层，节点 1 跑后一半层。
- 跨节点只传相邻阶段之间的激活值（点对点），通信量远小于跨节点 TP。
- 代价是流水线气泡，并发不够时单请求延迟较高。
- 带了 `--enable-expert-parallel`，EP 只在每个阶段内的 8 卡之间，MoE 的 all-to-all 全在 NVLink 内。

## 6. 策略三：Multi-Node Data + Expert Parallel（DEP）

已在本集群部署并测试（功能与行为，未做性能测试），完整记录见 `strategy3.md`，部署文件为 `dep-server.yaml`。关键发现：命令中无 TP 参数，实际是 DP16 + EP16；hybrid-lb 下每个节点各启动 8 个 API server 进程、两个节点都对外服务；该方案不需要 `--disable-custom-all-reduce`。下面是部署前整理的页面内容：


**页面**：Each node runs TEP over its local GPUs, with DP across nodes. Best throughput for MoE models across multiple nodes. Pass `-dp <total_workers>` at deploy time. Uses hybrid load balancing across DP workers. 定位：Throughput oriented（最高吞吐）。

```bash
# Node 0
vllm serve deepseek-ai/DeepSeek-V3 \
  --trust-remote-code --enable-expert-parallel \
  --data-parallel-hybrid-lb \
  --data-parallel-size 16 --data-parallel-size-local 8 \
  --data-parallel-address $HEAD_IP \
  （其余工具调用 / 推理参数同第 2 节）

# Node 1：同上，另加 --data-parallel-start-rank 8
```

要点：
- 两条命令都没有 `--headless`，每个节点各自对外提供 API；也没有 `--api-server-count`。
- 与 `vllm-dp16-ep-two-node.md` 的区别：之前是单 API 入口 + headless，请求先到节点 1 再由 DPCoordinator 分发；hybrid-lb 下每个节点有自己的 API server，节点内在本地 8 个 DP rank 间均衡，节点之间需要外部负载均衡（K8S Service 或 router）（推断）。
- 页面描述与命令的出入：页面说 "每个节点内部是 TEP"，但命令里没有 TP 参数（TP 默认 1），按命令更接近纯 DP16，没有节点内 TP。以实测为准。
- 实测（`vllm-dp16-ep-two-node.md`，默认 all2all 后端）：单流约 34.6 tok/s，512 并发 8203 tok/s，2048 并发 17808 tok/s。
- 扩到本集群 4 节点：`--data-parallel-size 32`，`--data-parallel-start-rank` 依次为 0 / 8 / 16 / 24。

## 7. 策略四：Multi-Node Tensor + Expert Parallel（TEP）

（待展开，下面是当前整理的内容。）

**页面**：Combines cross-node tensor parallel with expert parallel for MoE models. All GPUs across nodes participate in TP, with EP splitting expert layers. Requires InfiniBand interconnect. 定位：Balanced（均衡）。

页面上 TEP 的命令与第 4 节（Multi-Node Tensor Parallel）**完全相同**：因为页面所有多节点命令都带 `--enable-expert-parallel`，所以第 4 节的 “多节点 TP” 实际就是 TEP。但两页的**描述和定位标签不同**，见下表。

### 页面上 TP 与 TEP 的描述和命令对不上（2026-10-11 重新读取页面核对）

| | Multi-Node Tensor Parallel | Multi-Node Tensor + Expert Parallel |
|---|---|---|
| 页面描述 | Multi-node TP. Splits the model across GPUs spanning multiple nodes. TP is set to total GPU count across all nodes. Head node serves requests, worker nodes run with `--headless`. Requires InfiniBand. | Multi-node TEP. Combines cross-node tensor parallel with expert parallel for MoE models. All GPUs across nodes participate in TP, with EP splitting expert layers. Requires InfiniBand. |
| 定位标签 | Latency oriented | Balanced |
| Node 0 命令 | `--enable-expert-parallel --tensor-parallel-size 16 --nnodes 2 --node-rank 0 --master-addr $HEAD_IP …` | 同左，逐字相同 |
| Node 1 命令 | 同上，`--node-rank 1 --headless` | 同左，逐字相同 |

- **描述不同**：TP 的描述是纯 TP，没有提到专家并行；TEP 的描述才明确有 EP。
- **命令相同**：两页的命令都带 `--enable-expert-parallel`，所以 TP 页面的命令实际是 TP16 + EP16，也就是 TEP。
- **可能的原因（推测，未确认）**：
  - TP 页面的命令可能是笔误／漏改，把 EP 打开了。
  - 但命令下面有共用注释 “they differ only by --node-rank (TP/TEP) or --data-parallel-start-rank (DEP)”，说明页面把 TP 和 TEP 当作同一条命令模板，更像是模板没按策略区分，而不是个别笔误。
  - DEP 页面也有同类问题：描述写 “Each node runs TEP over its local GPUs, with DP across nodes”，命令里却没有 TP 参数（TP 默认 1，实际是纯 DP16）。三处描述与命令不一致，更像系统性问题。没有查 recipes 仓库的源数据，无法确认哪边是对的。
- **本文的处理**：以**命令**为准。
  - 策略一部署的 `tp16-server.yaml` 带了 `--enable-expert-parallel`，实测每卡 16 个专家（EP16），所以它实际是 TEP（TP16 + EP16），`strategy1.md` 就是它的记录。
  - 页面没有给出不带 EP 的纯 TP16。要测真正的纯 TP，需要自己去掉 `--enable-expert-parallel`：MoE 层也按 TP16 切，每个专家切 16 份，MoE 通信从 EP 变成 all-reduce。这个版本没有部署过，没有数据。

## 8. 策略五：Prefill/Decode 分离

（待展开，下面是当前整理的内容。）

页面只抓到 Prefill 的命令；Decode 和 Router 两个 tab 没有读取。

```bash
# Prefill：1 节点 × 8 GPU，TP8
export VLLM_NIXL_SIDE_CHANNEL_PORT=5557
export VLLM_NIXL_SIDE_CHANNEL_HOST=$PREFILL_NODE_1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
vllm serve deepseek-ai/DeepSeek-V3 \
  --trust-remote-code --enable-expert-parallel --port 8001 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
  --tensor-parallel-size 8 \
  （其余工具调用 / 推理参数同第 2 节）
```

`expert-parallel.md` 里的做法是 Prefill 用 `deepep_high_throughput`、Decode 用 `deepep_low_latency`，但 DeepEP 在 B300（sm_103）上已知不可用（见 `vllm-dp16-ep-two-node.md` 第 6.3 节）。

## 9. 策略对比与本集群的关系

| 策略 | 跨节点通信 | 单请求延迟 | 高并发吞吐 | 对网络要求 | 页面定位 |
|---|---|---|---|---|---|
| TP16 / TEP | 每层集合通信 | 最低（推断） | 中（推断） | 最高 | 低延迟 / 均衡 |
| TP8 × PP2 | 仅阶段间激活 | 较高（推断） | 高（推断） | 低 | 高吞吐 |
| DEP | 仅 MoE 层 | 较高（实测偏慢） | 最高 | 中 | 最高吞吐 |
| PD 分离 | KV cache 传输 | 可控 | 取决于配比 | 高 | 面向 TTFT / ITL SLA |

- 本集群有 4 个 B300 节点（32 卡），页面只讲 2 节点。
- 四种策略里只有 DEP 在本集群测过；TP16、TP+PP 没有测过。
- 建议后续测 TEP / TP16 和 TP+PP，与已有 DEP 数据在相同并发下对比。

## 10. Blackwell FP4（页面的 B200 方案，B300 未单独给出）
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve nvidia/DeepSeek-V3-FP4 \
  --trust-remote-code --tensor-parallel-size 4 --enable-expert-parallel
# DP 版：把 --tensor-parallel-size 4 换成 --data-parallel-size 4
```

## 11. 基准测试
```bash
vllm bench serve --model deepseek-ai/DeepSeek-V3 --dataset-name random \
  --random-input-len 8000 --random-output-len 1000 \
  --request-rate 10000 --num-prompts 16 --ignore-eos
```
三种形态：prefill 为主 8000/1000，decode 为主 1000/8000，均衡 1000/1000。vLLM 默认不开 prefix caching，无需额外参数。

## 12. 页面没有说明的内容，以及与本目录实验的差异

- 页面没有 `--all2all-backend`、DeepEP、NVSHMEM、`NCCL_IB_GID_INDEX` 的任何设置。`vllm-dp16-ep-two-node.md` 里在本集群上确认需要 `NCCL_IB_GID_INDEX=3`，否则 NCCL 初始化静默卡死；页面的 `*_SOCKET_IFNAME` 不能替代它。
- 页面的 DEP 用 `--data-parallel-hybrid-lb` 且两个节点都对外服务；之前的实验是单 API 入口 + headless。两种模式尚未对比。
- 页面没有给 B300 的 FP4（NVFP4）命令，`temp.txt` 里的 B300 + NVFP4 实验需要参照第 10 节自行调整，并实测 FlashInfer 自动选择是否在 B300 上生效。
- 页面没有写 K8S、RDMA 网卡或 `ulimit -l` 等容器环境要求；这部分以 `n-server.yaml` 和 `expert-parallel.md` 第 8 节为准。
- 页面标注的最低版本 0.12.0+，而 `n-server1.yaml` 使用 `v0.31.0`，满足 FlashInfer 自动选择的 0.28.0+ 要求。
- 命令里的 `examples/tool_chat_template_deepseekv3.jinja` 是 vLLM 仓库里的相对路径，镜像里不一定在当前目录下，运行前需确认路径。
