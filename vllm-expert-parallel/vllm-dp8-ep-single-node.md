# vLLM 单节点 DP=8 + EP：DeepSeek-V3-0324（TP=1）

在一台 8 卡节点（server-0）上，用 `--data-parallel-size 8 --enable-expert-parallel` 部署 DeepSeek-V3-0324（MoE，256 个路由专家，FP8），对外只有一个 8000 端口。每张 GPU 既是一个 DP rank，也是一个 EP rank，EP size = TP × DP = 1 × 8 = 8。

本文是 EP 系列实验的第一篇。双节点（DP=16 + EP=16）见 `vllm-dp16-ep-two-node.md`，概念笔记见 `expert-parallel.md`。

标注约定：**实测**＝本机日志或命令输出；**源码核对**＝读了镜像内 vLLM 0.19.2rc1 源码；**推断**＝由机制或估算得出，未直接验证。

## 摘要

| 问题 | 结论 |
|---|---|
| 并行结构 | Attention 用 DP，MoE 层用 EP（即 DP Attention）。vLLM 没有单独开关，由 TP、DP、`--enable-expert-parallel` 的组合自动形成（1.1） |
| 权重怎么放 | 256 个路由专家按 EP 分到 8 张卡，每卡 32 个；其余（MLA Attention、前 3 层稠密 FFN、shared expert、embedding、lm_head）每卡一份完整副本。每卡权重 96.23 GiB，KV cache 133.02 GiB（第 4 节） |
| MoE 层怎么通信 | 默认后端 `allgather_reducescatter`：每个 MoE 层先 all-gather 全部 token，各 rank 只算本卡专家，再 reduce-scatter 求和。通信库是 NCCL（pynccl 封装），没有用 DeepEP（第 5 节） |
| 通信粒度 | 每一步、每个 MoE 层一次，不是每个 token 一次（5.3） |
| 空闲 rank | 只发 1 个请求，8 张 GPU 利用率都约 100%：没有请求的 rank 要跑 dummy batch 陪跑。整个服务没有请求时全部为 0%。这与 all2all 后端无关（6.1） |
| 吞吐（粗测） | 单流约 60 tok/s；64 并发 1102 tok/s；512 并发 8070 tok/s（6.3） |
| 启动 | 冷启动约 11 分钟；**默认 600 秒就绪超时不够**，需设 `VLLM_ENGINE_READY_TIMEOUT_S`（2.4） |
| 已知问题 | 返回文本显示为字节级 BPE 符号（`Ġ`、乱码），模型输出的 token 本身是对的，问题在解码，根因未定（第 7 节） |

## 1. 并行结构：DP Attention + EP

```
              Client
                │ HTTP :8000
                ▼
     APIServer_0..7 （共用同一个监听 socket）
                │ 按各 rank 队列长度选 rank（统计来自 DPCoordinator）
                ▼ ZMQ
   EngineCore_DP0  DP1  DP2  DP3  DP4  DP5  DP6  DP7
        │           │    │    │    │    │    │    │
   Worker_DP0_EP0 ... Worker_DP7_EP7        每个 Worker 一张 GPU
        GPU0     ...    GPU7

   Attention / dense 层：各 rank 只处理自己的请求，互不通信（DP）
   MoE 层：每层 all-gather 全部 token → 各卡算本卡 32 个专家 → reduce-scatter（EP，NCCL）
```

### 1.1 并行方式由配置决定

**本实验的依据（实测）**
- 配置是 TP=1、DP=8、EP=8。日志里 rank 0 被标为 `DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank 0`，每个 DP rank 只占 1 张 GPU，Attention 没有被张量并行切分。
- 每个 DP rank 有自己的 EngineCore 和独立的 KV cache（每个 rank 都是 `GPU KV cache size: 2,032,480 tokens`），请求被分到不同 rank（每个约 72 个，见 6.2）。
- MoE 层是 EP：每个 rank 持有 32/256 个专家，8 个 rank 之间做集合通信（第 5 节）。

**不同配置下的行为**。下表来自 vLLM 的 EP 文档（见 `expert-parallel.md` 第 1 节），第 1、4 行有本机实测，第 2、3 行没有在本机运行：

| 配置 | Attention | MoE 层 |
|---|---|---|
| DP + EP，TP=1（本实验） | 每个 DP rank 独立计算，权重复制，KV cache 独立，即 DP Attention | 专家按 EP 分片到所有 rank |
| DP + EP，TP>1 | 每个 DP 组内按 TP 切分 | 同上，EP size = TP × DP |
| DP，不开 EP（MoE 模型） | 由 TP 决定 | 用 TP × DP 组成的 TP 组做张量并行 |
| DP，稠密模型 | 各 rank 独立（实测见 `../data-parallel/vllm-dp4-tp2-single-node.md`） | 没有专家层，rank 之间互不通信 |

即：Attention 怎么并行由 TP 决定，MoE 怎么并行由是否开 `--enable-expert-parallel` 决定，DP 负责把请求分给不同的副本。TP=1 时 Attention 退化为纯 DP，就是 DP Attention。

**“没有单独开关”的核对（源码核对）**
- `vllm serve --help=all` 中相关的参数只有 `--data-parallel-*` 系列、`--enable-expert-parallel`（`-ep`）和 `--all2all-backend`，没有名字里带 “attention” 的 DP 开关。
- 在 Python 源码中搜索 `dp attention` / `dp_attention`，只有 `v1/core/sched/interface.py` 的一处注释：`has_finished_requests()` 的说明里写着 “This information is useful for DP attention.”（其余命中是无关的类名 `InternSdpaAttention`）。vLLM 把 “DP attention” 当作概念使用，没有对应的配置项。
- 与 SGLang 的区别：SGLang 有显式的 `--enable-dp-attention`，vLLM 没有。在 vLLM 里要得到 DP Attention，就是 TP=1（或较小的 TP）加 DP 加 `--enable-expert-parallel`。

### 1.2 `Worker_DPn_EPn` 的含义

一个 Worker 同时属于一个 DP rank 和一个 EP rank。名字由 vLLM 按各并行维度的 rank 拼出来（源码核对，`v1/executor/multiproc_executor.py`）：`dp_size > 1` 时加 `_DP{dp_rank}`，`tp_size > 1` 时加 `_TP{tp_rank}`，`pp_size > 1` 时加 `_PP{pp_rank}`，开了 EP 就一定加 `_EP{ep_rank}`。

本例 TP=1，没有 `TP` 段，且 `ep_rank = dp_rank`（EP 组和 DP 组是同一批 8 个 rank），所以 `DPn` 与 `EPn` 数字相同。这是配置带来的结果，不是两者绑定。TP>1 时名字里会出现 `_TPm` 段，EP 与 DP 的编号也不再一一对应（只据源码判断，没有在本机跑过 TP>1）。

## 2. 部署与启动

### 2.1 环境

- server-0（StatefulSet `server` 的第 0 个 pod），8×NVIDIA B300（每张 275040 MiB），镜像 `vllm/vllm-openai:cu130-nightly`，vLLM 0.19.2rc1.dev134，torch 2.11.0+cu130，flashinfer 0.6.8，nvidia-nccl-cu13 2.28.9。镜像里装有 deep_ep 1.2.1，本实验没有使用。
- 模型：`deepseek-ai/DeepSeek-V3-0324`，缓存目录 642 GB。在 server-0 的 pod 里后台下载，未带 `HF_TOKEN`（模型公开，匿名下载），173 个文件约 6.5 分钟下完：

  ```
  hf download deepseek-ai/DeepSeek-V3-0324 > /root/.cache/huggingface/dl-v3.log 2>&1 &
  ```

  下载到默认的 HF 缓存目录 `/root/.cache/huggingface`（hostPath，落在节点上），vLLM 启动时直接复用。hostPath 在节点间不共享，每个节点要各自下载一份。
- 参考：<https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/>。

### 2.2 启动命令与验证

```
# 启动
VLLM_ENGINE_READY_TIMEOUT_S=3600 \
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel

# 验证
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-ai/DeepSeek-V3-0324",
       "messages": [{"role": "user", "content": "Who are you?"}],
       "max_tokens": 64}'
```

请求成功返回，`finish_reason: length`，prompt 7 个 token、生成 64 个 token，回答内容正确（自称 DeepSeek Chat）。返回文本的显示问题见第 7 节。

### 2.3 启动时间线（第二次启动，权重已缓存）

| 阶段 | 日志时间 | 说明 |
|---|---|---|
| 命令启动 | 00:24:40 | `Defaulting api_server_count to data_parallel_size (8)` |
| 初始化分布式 | 00:25:10 | 8 个 Worker，`world_size=8`，backend=nccl |
| 选定 all2all | 00:25:27 | `Using AgRsAll2AllManager all2all manager` |
| 开启 EP | 00:25:29 | `[EP Rank 0/8] Expert parallelism is enabled. Expert placement strategy: linear. Local/global number of experts: 32/256` |
| 权重加载完成 | 00:29:40 | `Model loading took 96.23 GiB memory and 251.3 seconds` |
| KV cache 就绪 | 00:30:56 | `Available KV cache memory: 133.02 GiB`；`GPU KV cache size: 2,032,480 tokens`；`Maximum concurrency for 163,840 tokens per request: 12.41x` |
| CUDA graph 抓取完成 | 00:36:13 | `Graph capturing finished in 241 secs, took 5.13 GiB`；`init engine ... took 393.37 s (compilation: 47.18 s)` |
| 对外服务 | 00:36 之后 | 8 个 APIServer 均 `Application startup complete` |

首次启动（权重冷读）加载权重用了 272 秒，其余阶段相近。

### 2.4 第一次启动失败：600 秒就绪超时

第一次启动没有设置环境变量，约 600 秒后失败：

```
TimeoutError: Timed out waiting for engine core processes to start. This is often caused by
slow weight loading for large models. Waited 600s (configured by VLLM_ENGINE_READY_TIMEOUT_S).
RuntimeError: Process ApiServer_0 (PID: 2532) died with exit code 1
```

- 8 个 APIServer 等待 EngineCore 就绪的默认上限是 600 秒，而权重加载（约 4.5 分钟）加上 autotune、DeepGEMM warmup 和 CUDA graph 抓取（约 4 分钟）超过了它。失败时引擎仍在正常执行（日志停在抓取 CUDA graph），不是模型或 EP 配置的问题。
- 处理：设置 `VLLM_ENGINE_READY_TIMEOUT_S=3600` 后重启，成功。旧日志保留为 pod 内 `/root/work/vllm-ep.first-timeout.log`。
- 清理现场：`pkill -f vllm` 会把执行它的 `bash -c` 也杀掉（命令行里含 vllm）。pod 内会留下若干 `<defunct>` 僵尸进程（PID 1 是 `sleep infinity`，不回收），但 GPU 显存已全部释放，不影响重启。

## 3. 进程结构（实测）

共 27 个进程：

```
vllm serve (主进程)
├─ resource_tracker                       Python multiprocessing 清理
├─ DPCoordinator                          DP 协调器
├─ EngineCore_DP0..7  (8 个)
│     └─ Worker_DPn_EPn  (每个 EngineCore 一个子进程)
└─ APIServer_0..7  (8 个)                 HTTP 前端，共用 8000 端口
```

| 进程 | 数量 | 作用 |
|---|---|---|
| 主进程 | 1 | 解析参数，创建 8000 端口监听 socket，启动并监控子进程 |
| resource_tracker | 1 | 资源清理 |
| DPCoordinator | 1 | 收集各 rank 队列长度并发布给 APIServer；维护 DP wave 状态 |
| APIServer | 8 | 默认等于 DP size |
| EngineCore_DPn | 8 | 每个 DP rank 一个，负责调度、continuous batching、KV 块管理 |
| Worker_DPn_EPn | 8 | 每个 EngineCore 一个，独占 1 张 GPU，做前向计算 |

- Worker 的 PPID 就是对应的 EngineCore，例如 `Worker_DP0_EP0`（PID 18327）的父进程是 `EngineCore_DP0`（15131），`Worker_DP6_EP6`（18322）的父进程是 `EngineCore_DP6`（15137）。PID 大小与 DP 序号不对应，判断归属要看 PPID。
- TP=1，所以一个 EngineCore 只有 1 个 Worker；稠密 DP4×TP2 实验里每个 EngineCore 有 2 个 Worker。
- 8000 端口：`ss -ltnp` 显示 `0.0.0.0:8000` 只有一个 LISTEN socket，被多个 APIServer 进程以同一个 fd（37）持有，与稠密 DP 实验的结论一致（输出被截断，只确认看到 3 个 APIServer 的 PID）。
- Worker 和 EngineCore 在 Pod IP（10.154.13.196）上监听了大量 TCP 端口，用途未逐个确认；`ss -x` 约 385 条 Unix socket 记录，说明进程间仍大量使用本机 IPC。

## 4. 权重与显存

| 项目 | 数值 | 说明 |
|---|---|---|
| 权重 | 96.23 GiB / 卡 | 日志 `Model loading took 96.23 GiB` |
| KV cache | 133.02 GiB / 卡 | 日志 `Available KV cache memory` |
| CUDA graph | 5.13 GiB / 卡 | 日志 |
| 实测显存占用 | 约 247652 MiB（约 241.8 GiB）/ 275040 MiB | `nvidia-smi`；`--gpu-memory-utilization` 实际为 0.92 |

### 4.1 哪些权重被复制、哪些被分片

同一个 Worker 的权重由两部分组成：
- **非专家部分**：与其他 7 个 Worker 完全相同（DP 复制）。TP=1 时包括 MLA Attention 的各投影、前 3 层稠密 FFN、每个 MoE 层的 shared expert、embedding、lm_head 和 norm。所以被复制的不只是 Attention。
- **路由专家部分**：与其他 7 个 Worker 完全不同（EP 分片），每卡 32 个。

因此不能说“同一 DP 组里的权重都一样”：这里 EP 的分片范围正好跨越所有 DP rank，DP rank 之间的专家权重并不相同。

### 4.2 每卡权重估算

用 `config.json` 估算（FP8 按 1 字节/参数，embedding 和 lm_head 按 bf16 计；不含量化 scale、norm、router gate 和 MTP 层；是算术估算，不是逐张量统计）：

| 部分 | 参数量 | 每卡大小 | 备注 |
|---|---|---|---|
| 路由专家 | 653.9 B | 76.1 GiB | 58 个 MoE 层 × 256 个专家 × 3 × 7168 × 2048，EP 8 路分片后每卡 1/8 |
| MLA Attention | 11.4 B | 10.6 GiB | 61 层，每个 rank 一份完整副本 |
| shared expert + 前 3 层稠密 FFN | 3.7 B | 3.5 GiB | 每个 rank 一份完整副本 |
| embedding + lm_head | 1.85 B | 3.5 GiB | bf16，每个 rank 一份完整副本 |
| **合计（非专家部分复制）** | | **约 93.7 GiB** | 实测 96.23 GiB，差约 2.5 GiB（约 2.7%），可由未计入的 scale、norm、router gate 等解释（未核对） |
| 对照：非专家部分也 8 路切分 | | 约 78.3 GiB | 与实测相差 18 GiB，不符 |

实测值接近“非专家部分复制”的估算，与“全部切分”差距很大，数字支持复制。

代价：Attention 每卡约 10.6 GiB，8 份合计约 85 GiB，其中 7 份（约 74 GiB）是冗余的，这是 DP Attention 的显存代价；MLA 的 KV cache 小，所以通常仍然划算。模型目录 642 GB，8 张卡合计权重约 770 GiB（8 × 96.23 GiB），差异来自复制的部分；目录里还含 MTP 层，估算未计入。

### 4.3 shared expert

- 每个 MoE 层有 1 个 shared expert（`n_shared_experts = 1`），不参与 EP 分片，每张卡各存一份完整副本。原因：它不经过路由，每个 token 都要过它，每个 DP rank 在本地就能算，不需要跨卡通信；本实验 TP=1，所以连层内切分也没有。
- 它很小：单个专家参数量 7168 × 2048 × 3 ≈ 44 M，58 个 MoE 层合计约 2.55 B 参数，FP8 约 2.4 GiB（`config.json` 算术估算，不是实测）。路由专家总量约为它的 256 倍，整个模型 671 B 参数里它约占 0.4%，复制它的显存代价可以忽略。4.2 表中“shared expert + 前 3 层稠密 FFN”的 3.5 GiB，其中 shared expert 约 2.4 GiB，前 3 层稠密 FFN 约 1.1 GiB。
- 它的执行方式（与路由专家融合、单独算后相加、或与通信重叠）取决于 MoE 后端；源码里只核对到 `_maybe_combine` 单独保存其输出（5.2），没有追踪不同后端的差异。

### 4.4 专家放置与路由配置

**专家放置**
- 256 个路由专家分到 8 个 EP rank，每张 GPU 持有 32 个，每个专家只存在于一张卡上。日志：`[EP Rank 0/8] Expert parallelism is enabled. Expert placement strategy: linear. Local/global number of experts: 32/256. Experts local to global index map: 0->0, 1->1, 2->2, ...`。
- 策略是 `linear`（连续分块）：EP0 放全局专家 0–31，EP1 放 32–63，依此类推。日志里只核对了 EP0 的映射，其余 rank 的区间是按 `linear` 推出的。

**路由配置**（`config.json`，已核对）

| 项 | 值 |
|---|---|
| 每个 token 激活的路由专家数 `num_experts_per_tok` | 8（top-8） |
| 路由专家总数 `n_routed_experts` | 256 |
| 专家分组 `n_group` / 每个 token 选的组数 `topk_group` | 8 / 4 |
| 路由方式 `topk_method` / 打分函数 `scoring_func` | `noaux_tc` / `sigmoid` |
| 稠密层数 `first_k_dense_replace` | 3（共 61 层，其中 58 层是 MoE 层） |

## 5. MoE 层的通信路径

### 5.1 后端与组件（日志）

- **all2all 后端**：`Using AgRsAll2AllManager`，即默认的 `allgather_reducescatter`。没有指定 `--all2all-backend`，没有用 DeepEP。
- **MoE 计算后端**：`Using FLASHINFER_TRTLLM Fp8 MoE backend out of potential backends: ['AITER', 'FLASHINFER_TRTLLM', 'FLASHINFER_CUTLASS', 'DEEPGEMM', 'TRITON', ...]`。
- **Attention 后端**：`FLASHINFER_MLA`，KV cache block size 为 32。
- **DP 同步**：日志 `Disabling NCCL for DP synchronization when using async scheduling.`，即异步调度下 DP 同步不走 NCCL；具体走什么通道没有核对。

### 5.2 一个 token 在 MoE 层里怎么走（源码核对）

`allgather_reducescatter` **不是**“把 token 定向发给专家所在的卡”，而是让每个 rank 拿到所有 rank 的全部 token，各自只算本卡的专家，最后求和。依据 vLLM 0.19.2rc1 源码（`device_communicators/all2all.py` 的 `AgRsAll2AllManager`、`fused_moe/runner/moe_runner.py`、`experts/trtllm_fp8_moe.py`、`models/deepseek_v2.py`）：

1. **打分**：Attention 之后，每个 rank 用 gate（`GateLinear`）对自己的 token 算出 256 维 `router_logits`。DeepSeek-V3 的 gate 带 `e_score_correction_bias`（`noaux_tc`），选专家用分组 top-k：`use_grouped_topk=True`、`num_expert_group=8`、`topk_group=4`、sigmoid 打分、`routed_scaling_factor=2.5`。
2. **dispatch（all-gather）**：`dispatch_router_logits` 把所有 rank 的 `hidden_states`（FP8 量化后）、量化 scale 和 `router_logits` 沿 token 维 `all_gatherv`。完成后每个 rank 都持有 8 个 rank 的全部 token。
3. **本地专家计算**：每个 rank 对**全部** token 只计算自己持有的 32 个专家（`local_expert_offset = ep_rank × 32`、`local_num_experts = 32`），没选中本卡专家的 token 贡献为 0。top-8 的选择有两种实现，源码里都存在：
   - monolithic：把 `routing_logits` 交给 FlashInfer 的 `trtllm_fp8_block_scale_moe`，在内核里按 DeepSeekV3 路由方法选专家并计算；
   - modular：由 vLLM 的路由器先算出 `topk_weights` / `topk_ids`，再一起 all-gather（`dispatch`）。

   本次运行走的是哪一条，没有从日志核对（日志只写了 `Using FLASHINFER_TRTLLM Fp8 MoE backend`）。
4. **combine（reduce-scatter）**：`reduce_scatterv` 把各 rank 上“本卡专家对全部 token 的部分输出”按 token 求和，并按每个 rank 原来的 token 切分散回去，每个 rank 拿回自己 token 的完整 MoE 输出。
5. **shared expert**：`_maybe_combine` 里 shared expert 的输出单独保存，不经过上面的 reduce-scatter，在各自 rank 上本地处理（代码层面的观察，没有进一步追踪）。

**汇总的是 MoE 层的输入，不是各 rank 的最终输出**：各 DP rank 对自己的请求做 Attention 时互不通信，汇总发生在 MoE 层入口，对象是 token 的隐状态（外加路由信息）。MoE 层结束后，每个 rank 通过 reduce-scatter 只拿回自己那批 token 的结果，继续下一层的 Attention，最终采样也在各自 rank 上完成。

**通信库**：`all_gatherv` 和 `reduce_scatterv` 都通过 vLLM 的 `pynccl_comm`（对 NCCL 的封装，本镜像为 nvidia-nccl-cu13 2.28.9），代码里有 `assert pynccl_comm is not None`。8 张 GPU 在同一台机器上，走的应是节点内 NVLink，这是推论，没有抓取流量。

### 5.3 通信粒度：每步、每层一次，不是每个 token 一次

- 调度器每一步把一批 token 凑成 batch，整批一起过整个模型。prefill 时一个 prompt 的 token 是并行处理的；长 prompt 会按 `max_num_batched_tokens` 切成多个 chunk（chunked prefill），每步处理一块，同一步里还可能混入其他请求的 decode token。
- 每个 MoE 层，每个 rank 做一次 all-gather（dispatch）和一次 reduce-scatter（combine），对象是本步本 rank 的整批 token。DeepSeek-V3 有 58 个 MoE 层，所以每一步是 58 对集合通信调用。这是按结构推断的，没有用 profiler 或 `NCCL_DEBUG` 数过实际调用次数。
- prefill 一步里 token 多，通信量大，偏带宽受限；decode 每个请求一步只有 1 个 token，每次通信的数据量小，更看延迟（这也是 DeepEP low-latency 模式主要针对的场景，该模式本实验未测）。
- **各 rank 的 token 数不同时怎么处理（源码核对）**：`AgRsAll2AllManager.dispatch*` 从 `dp_metadata.get_chunk_sizes_across_dp_rank()` 取出每个 rank 本步的 token 数 `sizes`，调用 `all_gatherv(..., dim=0, sizes=sizes)` 做变长 all-gather，并断言 `sizes[本 rank] == hidden_states.shape[0]`；`combine` 对应的是变长 reduce-scatter。所以不是 padding 到统一长度。没有请求的 rank 靠 dummy batch 参与集合通信（6.1）。MoE 内部是否还有按 `VLLM_MOE_DP_CHUNK_SIZE` 的再分块，没有核对。

### 5.4 与 token 级 all2all 的区别

`deepep_high_throughput` / `deepep_low_latency`（DeepEP）和 FlashInfer NVLink all2all 等后端才是“按路由结果把 token 定向发到专家所在的卡”，通信量只与被选中的专家有关。而 `allgather_reducescatter` 把全部 token 广播给所有 rank，每个 rank 都要接收 8 个 rank 的全部数据，通信量随 rank 数增长。它实现简单，适合单节点；多节点通常要换成 DeepEP 等后端（见 `expert-parallel.md` 第 3 节）。这个通信量的比较是从机制推的，没有实测对比。

## 6. DP+EP 的运行行为（实测）

### 6.1 一个请求让 8 张 GPU 全部忙起来

稠密 DP 实验中，一个请求只会让其中一个 DP rank 的 GPU 工作。这里：

| 状态 | 8 张 GPU 利用率（`nvidia-smi`） |
|---|---|
| 空闲（无请求，连续 3 次采样） | 全部 0%（功耗约 227–239 W） |
| 只有 1 个请求在生成（max_tokens=1500，耗时 25.2 s） | 全部 99–100% |
| 请求结束后 | 全部 0% |

另外，启动完成后的最初几秒曾看到约 95%，随后降到 0%，是启动收尾的残留，不是空转。

**原因（源码核对，`v1/engine/core.py` 的 `run_busy_loop`、`v1/worker/gpu_worker.py` 的 `execute_dummy_batch`）**

- **整个服务没有请求**：所有引擎空闲，源码在没有本地请求且 `engines_running` 为 False 时直接 `continue`，不做前向，GPU 为 0%。
- **只要有一个 rank 有请求**：DPCoordinator 广播 `START_DP_WAVE`，所有引擎进入运行状态（`engines_running = True`）。此后没有可执行请求的 rank 会调用 `execute_dummy_batch()`（源码注释：“We are in a running state and so must execute a dummy pass”），因为每个 MoE 层的集合通信需要所有 rank 一起参与（5.2）。稠密模型的 DP rank 之间没有这类通信，所以空闲 rank 可以真正空闲。
- **dummy 的规模**：`_dummy_run(num_tokens, uniform_decode=True)`，`num_tokens` 取 `uniform_decode_query_len`，默认是 1，即每步约 1 个 token 的前向。`_dummy_run` 内部实现没有读，是否包含完整的 Attention 计算没有核对。
- **何时停止**：`_has_global_unfinished_reqs` 每 32 步才做一次 all-reduce（`has_unfinished_dp`）来确认所有 rank 都没有未完成请求，之后这一轮 wave 结束，所有引擎一起暂停。所以最后一个请求完成后，空闲 rank 可能还会多陪跑最多约 32 步（按源码推断，没有测量）。

因此 DP+EP 下 GPU 利用率不能反映有效负载：只有 1 个请求时 8 张卡都显示约 100%，其中 7 张卡在做约 1 token 的 dummy 前向。另外每一层都有同步点，快的 rank 要等最慢的 rank，负载不均时一个 rank 的长 prefill 会拖慢其他 rank 的 decode（机制推断，没有实测）。

**与 all2all 后端无关**：`DPEngineCoreProc` 断言 `model_config.is_moe`，只要是 MoE 模型并开了 DP 就走这套逻辑；`run_busy_loop` 里“本 rank 没有可执行请求、但全局处于运行状态就跑 dummy”的判断没有任何 all2all 后端相关的条件，属于引擎层，不是通信后端层。换成 `deepep_high_throughput` / `deepep_low_latency` 也一样要陪跑：根本原因是专家分散在所有 rank 上，每个 rank 的专家都要替别的 rank 的 token 计算，且 dispatch / combine 是全体 rank 一起参与的集合操作（某 rank 本步发送 0 个 token 也要参与）。DeepEP 要求全员参与是从机制推的，没有逐个读这些后端的源码。不同后端的区别只在通信量和延迟：AgRs 把全部 token 广播给所有 rank，DeepEP 只发给被选中的专家所在的卡，所以 dummy rank 在 DeepEP 下收发的数据更少，但同步参与不变。

**什么情况下可以不陪跑**：只有把 Attention 与专家放在不同的 GPU 上（Attention-FFN 分离，AFD），且两者之间用点对点通信而不是全员集合通信，没有请求的 Attention 卡才可能真正闲着。代价是每层都要跨卡传输隐状态、两边存在流水线气泡（需要 micro-batch 交替来重叠）、两边的卡数需要配平。本实验 DP 与 EP 覆盖同一批 rank，Attention 与专家在同一张卡上；vLLM 是否支持 AFD 没有核对；上述 AFD 的优缺点来自一般了解，不是本次实验验证的内容。

### 6.2 请求在 8 个 DP rank 间的分配

用并发脚本发请求后，读取 `/metrics` 中按 `engine` 标签区分的 `vllm:request_success_total`：

| engine | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 合计 |
|---|---|---|---|---|---|---|---|---|---|
| 完成请求数 | 73 | 72 | 72 | 71 | 72 | 74 | 72 | 73 | 579 |

579 = 此前的 3 个请求 + 64 + 512，没有丢失，也没有 error / abort。分配比较均匀（71–74），与 APIServer 按各 rank 队列长度选 rank 的机制一致。

### 6.3 吞吐（粗测，不是基准测试）

| 场景 | 并发 | 每请求生成 | 总生成 token | 耗时 | 吞吐 |
|---|---|---|---|---|---|
| 单流 | 1 | 1500 | 1500 | 25.2 s | 约 60 tok/s |
| 64 并发 | 64 | 256 | 16384 | 14.9 s | 1102 tok/s |
| 512 并发 | 512 | 256 | 131072 | 16.2 s | 8070 tok/s |

条件：`temperature=0`、`ignore_eos=true`、prompt 约 10 个 token（极短）、每档只跑了一次、没有预热对比；不代表真实负载下的性能，也没有测延迟分布。趋势：并发上去后吞吐近似线性增长，单流 decode 只有约 60 tok/s。

## 7. 已知问题：返回文本显示为字节级 BPE 符号

`/v1/chat/completions` 和 `/v1/completions` 返回的文本形如：

```
"IâĢĻmĠ**DeepSeekĠChat**,ĠyourĠAIĠassistant..."
'Paris.æ³ķåĽ½çļĦé¦ĸéĥ½æĺ¯å·´é»İãĢĤ2'
```

空格显示成 `Ġ`，中文和 emoji 显示成乱码，但英文单词本身以及 "Paris." 是对的，说明模型输出的 token 正确，问题出在 token 到文本的解码这一步。

已核对：
- 镜像里的 `transformers` 是 5.6.0，`AutoTokenizer` 加载出来的类型是 `LlamaTokenizer`；用它对一句英文编码再解码，结果丢了所有空格（`'IamDeepSeekChat,yourassistanthi'`）。
- 直接用 `tokenizers` 库加载同目录的 `tokenizer.json` 编码再解码，结果正确（`'The capital of France is Paris. 法国的首都是巴黎。'`）。

没有核对：根因是否是 transformers 5.x 对 DeepSeek-V3 tokenizer 配置处理不当，vLLM 的 detokenizer 是否走同一条路径，以及如何修复（例如指定 tokenizer 类型或换 transformers 版本）。这不影响对 EP 本身的结论，但用输出内容做质量评估之前需要先解决。

## 8. 结论

- 单节点 8 卡上 DP=8 + EP=8 可以直接部署 DeepSeek-V3-0324（FP8），能正常响应 chat 请求。
- 并行结构是 Attention 用 DP、MoE 用 EP，由配置自动形成；每卡权重约 96 GiB，其中非专家部分被复制，剩 133 GiB 做 KV cache，单 rank 可容纳约 203 万 token。
- 默认后端 `allgather_reducescatter` 是全量 all-gather + 本地专家计算 + reduce-scatter，每步每个 MoE 层一次，通信走 NCCL，实现简单，适合单节点。
- DP+EP 下各 rank 必须同步：有请求时空闲 rank 用 dummy batch 陪跑，GPU 利用率不能反映有效负载，这是与稠密 DP 最大的区别，且与通信后端无关。
- 大模型首次启动耗时长，必须调大 `VLLM_ENGINE_READY_TIMEOUT_S`。

## 9. 未验证项

**并行与权重**
- “非专家部分在每个 rank 上完整复制”只有显存估算佐证（估算 93.7 GiB vs 实测 96.23 GiB），没有逐张量统计每张卡的实际权重，也没有读 vLLM 加载代码；2.5 GiB 的差距没有逐项拆解。
- TP>1 时 Worker 的实际命名和 rank 对应关系，只据 `multiproc_executor.py` 的命名代码判断，没有在本机运行。
- 1.1 表中 TP>1、不开 EP 两种配置，只来自 vLLM 文档，没有在本机运行。建议用 TP=2×DP=4 + EP 做对照（显存、KV cache 容量、dummy batch 行为）。
- EP1–EP7 的专家区间是按 `linear` 推的，只核对了 EP0。
- 源码核对只覆盖这个镜像里的 vLLM 版本。

**通信**
- 本次运行走 monolithic 还是 modular 路径没有确认。
- 通信走 NVLink 是推论，没有抓取 NCCL 流量；每步 58 对集合调用是按结构推断，没有数过；也没有对比 `deepep_high_throughput` / `deepep_low_latency` 后端，各后端 dispatch / combine 是否都要求全员参与也没有读源码。
- MoE 内部是否还有按 `VLLM_MOE_DP_CHUNK_SIZE` 的再分块。
- 各专家实际收到的 token 数（负载是否倾斜、EPLB 是否有必要）没有测，可用 `--enable-eplb` 的 `log_balancedness` 实测。
- 异步调度下 DP 同步实际走什么通道。
- `--enable-eplb`、`--enable-dbo`、`--async-scheduling` 的效果（异步调度从日志看默认已开，其余没有试）。
- vLLM 是否支持 Attention 与 FFN 分卡部署（AFD）。

**运行行为**
- `_dummy_run` 的内部实现（是否包含完整 Attention 计算）；最后一个请求结束后空闲 rank 实际多陪跑几步（源码上限约 32 步，没有实测）；负载不均时 rank 之间互相拖慢的程度。
- 6.3 的吞吐是未经预热、单次、极短 prompt 的粗测。

**进程与环境**
- 8000 端口是否被全部 8 个 APIServer 持有同一监听 fd（只看到 3 个），请求在各 APIServer 之间的分配是否均匀（只统计了 DP rank 一层）。
- Worker 在 Pod IP 上监听的大量 TCP 端口的用途。
- 镜像里 `import deep_gemm` 报 `ModuleNotFoundError`，但启动日志里出现了 `DeepGEMM warmup` 进度条，两者的关系没有查清。
- 第 7 节乱码的根因和修复方式。

## 10. 运维备注

- 服务当前仍在 server-0 里运行（`vllm serve`，PID 14722），日志在 pod 内 `/root/work/vllm-ep.log`，只在容器文件系统中，pod 重建后会丢失；模型缓存在 hostPath `/root/.cache/huggingface`，在节点上持久（每个节点各一份，不共享）。
- 并发测试脚本在 pod 内 `/root/work/conc.py`（`python3 conc.py <并发数> <max_tokens>`）。
- 停止服务不要用 `pkill -f vllm`（会误杀自己所在的 shell），用 `kill <主进程 PID>`，并确认 `nvidia-smi` 显存已释放。
- 多机 EP 的实验见 `vllm-dp16-ep-two-node.md`。
