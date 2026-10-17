# 单节点并行组合实验：Attention（TP/DP）× MoE（TP/EP）

DeepSeek-V3（FP8）单节点 vLLM 实例，Attention 并行（TP8 / DP8）× MoE 并行（TP8 切 / EP8）四种组合，每组合一个 Pod、独占一个 B300 节点的 8 卡，四者同时部署、同批实验：验证运行模式，统一方法测出性能、吞吐、能效、流量与场景差异。

- 环境：镜像 `vllm/vllm-openai:latest`（vLLM 0.31.0），K8s；B300×8 / 节点（每卡 267.69 GiB，节点内 NV18）。
- 标注约定：**实测**＝本集群命令输出或日志；**推断**＝由机制或估算得出，未直接验证。
- 部署文件：`sn-tp8.yaml`、`sn-tep8.yaml`、`sn-dtp8.yaml`、`sn-dep8.yaml`（本目录）。

**结论速览**（数据见第 5、7 章）：

| 问题 | 结论 |
|---|---|
| 四种组合运行模式与配置一致吗 | 一致——进程、日志、行为三方面证据互证（第 3 章） |
| 延迟敏感场景谁好 | TP 组（tp8/tep8）：单流 TPOT 7.7 ms（DP 组的 1.9 倍速度），开环三档延迟全部最低 |
| 吞吐场景谁好 | dtp8：饱和吞吐 1.32× tp8、TPOT 优 29%、P99 ITL 仅 1/3、能效 +40% |
| 交叉点在哪 | 随机均衡负载下吞吐/TPOT/能效三个交叉点重合在 **C=128–160**；prefill 为主左移到 64，真实短输出右移到 256 以外 |
| KV 容量的作用 | **最硬分水岭**：TP 组在 97% 容量即已劣化，超容后 preemption、TTFT 50–108 s；DP 组全程 0 抢占 |
| MoE 切法（TP 切 vs EP）重要吗 | 远小于 Attention 并行方式的影响（6–8% 对 2×）；本负载下 EP 无优势 |
| 一次 32k 请求实际产生多少流量 | TP 组 487 GiB/节点、DP 组 295 GiB（TP 1.65×）——但流量多的 TP 反而快 3.1×，NVLink 只用到 6%（第 7 章） |

## 1. 四种组合

四种组合 = **2×2**：Attention 并行（TP8 / DP8）× MoE 切法（TP 切：每个专家切 1/8 / EP：每卡 32 个完整专家）。四者用同样的 16 卡、同一模型、同一条命令模板，变量只有两个开关——并行方式选 `--tensor-parallel-size 8` 还是 `--data-parallel-size 8`、加不加 `--enable-expert-parallel`。

| Pod | yaml | Attention | MoE | vllm serve 关键参数 |
|---|---|---|---|---|
| tp8-0 | `sn-tp8.yaml` | TP8 | TP8 切（每专家 1/8） | `--tensor-parallel-size 8` |
| tep8-0 | `sn-tep8.yaml` | TP8 | EP8（每卡 32 个完整专家） | `--tensor-parallel-size 8 --enable-expert-parallel` |
| dtp8-0 | `sn-dtp8.yaml` | DP8 | TP8 切（每专家 1/8） | `--data-parallel-size 8` |
| dep8-0 | `sn-dep8.yaml` | DP8 | EP8（每卡 32 个完整专家） | `--data-parallel-size 8 --enable-expert-parallel` |

术语：下文称 tp8/tep8 为 **TP 组**（同为 TP Attention）、dtp8/dep8 为 **DP 组**；"**组内**"指同 Attention 模式下比较（变量只剩 MoE 切法），"**组间**"比较 Attention 并行方式。

四组共有参数：`--trust-remote-code --no-enable-prefix-caching --enable-auto-tool-choice --tool-call-parser deepseek_v3 --chat-template examples/tool_chat_template_deepseekv3.jinja --reasoning-parser deepseek_v3`；环境变量 `GLOO/NCCL_SOCKET_IFNAME=eth0`、`NCCL_IB_GID_INDEX=3`（单节点无害）、`VLLM_ENGINE_READY_TIMEOUT_S=3600`。单节点无需跨节点参数（`--nnodes/--master-addr/--disable-custom-all-reduce` 等）。

**通信结构总览**（日志证据见 3.1/3.2；传输均走节点内 NVLink）：

| 配置 | Attention 通信 | MoE 通信（每层） | 每 decode 步集合通信次数 |
|---|---|---|---|
| tp8 | 每层 1 次 TP8 all-reduce（o_proj 后） | 每层 1 次 TP8 all-reduce（MoE 结尾汇总部分和） | 122（61 层 × 2） |
| tep8 | 每层 1 次 TP8 all-reduce（o_proj 后） | 每层 1 次 TP8 all-reduce（MoE 结尾汇总部分和；EP 仅改每卡专家布局，通信不变） | 122（61 层 × 2） |
| dtp8 | **零通信**（各 rank 各算各的请求） | all_gatherv → 本地算 → reduce_scatterv（DP 组 8 rank） | 116（58 个 MoE 层 × 2；前 3 层稠密零通信） |
| dep8 | **零通信**（各 rank 各算各的请求） | all_gatherv → 本地算 → reduce_scatterv（DP 组 8 rank；与 dtp8 同一条 AgRs 路径，仅本地算的对象不同） | 116（58 个 MoE 层 × 2；前 3 层稠密零通信） |

**Attention 层**（tp8/tep8 与 dtp8/dep8 的分野）：前两组每层 1 次 all-reduce，数据量 = 批内 token × hidden（约 14 KB/token）；后两组**零通信**——**DP 组的通信优势全部来自 Attention 层**。

**MoE 层**：tp8 与 tep8 通信完全相同（EP 只改每卡专家布局）；dtp8 与 dep8 走**同一条 AgRs 路径**（这就是组内 MoE 切法差异小的原因），两段各传——

- dispatch（all_gatherv）：把各 rank token 的**输入**（hidden states + router logits）发给全部 8 卡，每卡收到全批（＝全组 8 rank 的 token 合计，单请求时含其余 rank 的 dummy token）；
- combine（reduce_scatterv）：把各卡对全批 token 的**部分和**求和后散回，每卡只收回自己 token 的最终输出；
- 通信上"收到全批"直接意味着计算上"要算全批"（每卡为别人的 token 算自己那份专家切片）——dummy batch 陪跑的根源。

**控制面（仅 dtp8/dep8）**：每步一次 DP 元数据 CPU 同步（Gloo，256 字节级小消息）与低频 wave 共识，不在 GPU 数据路径上；tp8/tep8 无此机制。

**层内位置（关键不对称，整体对比）**：**TP 组"输入同、输出合"**——o_proj 的 all-reduce 之后各卡持有整批 token，每层 2 次 all-reduce 都在层尾输出侧合并部分和；**DP 组"输入凑、输出还"**——各 rank 的 Attention 独立算自己的 token，每个 MoE 层入口 `all_gatherv` 先把全组 token 凑到每卡、层尾 `reduce_scatterv` 把结果按归属归还。字节数上一次 all-reduce ≈ 一次（AG+RS）（环形 all-reduce 本身就是 RS+AG 拼成的）。

**次数相同、字节不同（整体对比）**：四种配置在 MoE 层上**每层都是 2 次 collective**（TP 组：Attention + MoE 各一次 AR；DP 组：AG + RS 各一次，唯前 3 层稠密为 0 次）——但 TP 组的每次是整份（≈411 MB/次/卡），DP 组的每次是半份（AG、RS 各 ≈206 MB，合计才等于一次 all-reduce）。**通信次数几乎一样、TP 组流量是 DP 组的 2 倍**，差距全部来自 Attention 列（TP 交整批、DP 零通信，详见 7.4）。

## 2. 部署与启动

StatefulSet（`replicas: 1`）+ headless Service + api Service；每 Pod 占 8 GPU + 8 RoCE 网卡，按资源自动分散到 4 个 GPU 节点（`kubectl get pods -o wide`，每 Pod 独占一节点）：

```
NAME      READY   STATUS    RESTARTS   AGE    IP             NODE
tp8-0     1/1     Running   0          2d1h   10.154.5.180   rs-b300x8-gpu-pool-3jvd2z
tep8-0    1/1     Running   0          2d1h   10.154.5.99    rs-b300x8-gpu-pool-3jvdpn
dtp8-0    1/1     Running   0          2d1h   10.154.6.27   rs-b300x8-gpu-pool-3jv0ev
dep8-0    1/1     Running   0          2d4h   10.154.4.190   rs-b300x8-gpu-pool-3jvd29
```

集群共 6 个节点：4 个 B300×8 GPU 节点（上述落点）+ 2 个 CPU 节点（`kubectl get nodes -o wide`，K8s v1.37.0，Debian 13 / 内核 6.12 / containerd 2.3.3，四者相同）：

```
NAME                        STATUS   ROLES    AGE     VERSION   INTERNAL-IP   OS-IMAGE
rs-b300x8-gpu-pool-3jv0ev   Ready    <none>   2d      v1.37.0   10.132.0.38   Debian GNU/Linux 13 (trixie)
rs-b300x8-gpu-pool-3jvd29   Ready    <none>   2d      v1.37.0   10.132.0.32   Debian GNU/Linux 13 (trixie)
rs-b300x8-gpu-pool-3jvd2z   Ready    <none>   2d      v1.37.0   10.132.0.34   Debian GNU/Linux 13 (trixie)
rs-b300x8-gpu-pool-3jvdpn   Ready    <none>   2d      v1.37.0   10.132.0.33   Debian GNU/Linux 13 (trixie)
rs-power-cpu-pool-3jmm1a    Ready    <none>   4d19h   v1.37.0   10.132.0.17   Debian GNU/Linux 13 (trixie)
rs-power-cpu-pool-3jmm1e    Ready    <none>   4d19h   v1.37.0   10.132.0.13   Debian GNU/Linux 13 (trixie)
```

| Pod | 节点 | `init engine` | 权重加载 |
|---|---|---|---|
| tp8-0 | 3jvd2z | 916.69 s | 84.4 s |
| tep8-0 | 3jvdpn | 931.04 s | 69.9 s |
| dtp8-0 | 3jv0ev | 1082.57 s | 90.1 s |
| dep8-0 | 3jvd29 | 1092.86 s | 76.1 s |

启动无故障（单节点不触发跨节点初始化问题；HF 权重由启动脚本在各节点 hostPath 预热）。DP 组 `init engine` 比 TP 组长约 160 s，就绪总耗时 TP 组约 19 min、DP 组约 22 min。冒烟：四组对话请求均正常返回（`finish_reason=stop`，文本正常）。

## 3. 运行模式验证

### 3.1 Attention 运行模式（日志证据）

| 证据 | tp8 / tep8 | dtp8 / dep8 |
|---|---|---|
| `DP group leader ... world_size=?` | `world_size=8`（一个 TP 组） | 8 个 EngineCore 各自 `world_size=1` |
| `api_server_count` | 未打印（默认 1，API server 在主进程内） | `Defaulting api_server_count to data_parallel_size (8)` |
| DPCoordinator 进程 | 无 | 有 |

即：TP 组一个调度器管 8 个 Worker、单请求 8 卡共同计算；DP 组 8 个独立引擎（每 rank 一个）+ DPCoordinator + 8 个 APIServer，请求只落一个 rank，其余 rank 以 dummy batch 陪跑（MoE 集合通信要求全员参与，推断）。

### 3.2 MoE 运行模式（日志证据）

| 证据 | tp8 | tep8 | dtp8 | dep8 |
|---|---|---|---|---|
| `[EP Rank r/8] ... 32/256, placement: linear` | 无 | **有** | 无 | **有** |
| `Using AgRsAll2AllManager` | 无 | 无 | **有** | **有** |
| `MoEPrepareAndFinalizeNoDPEPMonolithic` | **有** | **有** | 无 | 无 |
| `MoEPrepareAndFinalizeNaiveDPEPMonolithic` | 无 | 无 | **有** | **有** |

四组 MoE 计算后端同为 `FLASHINFER_TRTLLM Fp8`，Attention 后端同为 `FLASHINFER_MLA` + prefill `FLASH_ATTN MLA`。

- TP 组（DP=1）：无 all2all——TP all-reduce 后输入在各卡本就是完整副本，各卡自算路由只算本地部分，每层结尾一次 TP 组 all-reduce；tp8 与 tep8 仅每卡专家形态不同（1/8 切片对 32 个完整专家）。
- DP 组：MoE 无论 TP 切还是 EP 都走 all_gatherv → 本地计算 → reduce_scatterv，组件同为 `NaiveDPEPMonolithic`；区别只在每 rank 本地算的对象（dtp8：每个专家的 1/8 切片；dep8：32 个完整专家——由 EP 日志行区分，字节上相同，推断）。

### 3.3 显存与 KV

每卡（日志实测）：

| 项 | tp8 | tep8 | dtp8 | dep8 |
|---|---|---|---|---|
| 权重（`Model loading took`） | 79.74 GiB | 79.74 GiB | 96.09 GiB | 96.09 GiB |
| consumed（权重 + 非 torch） | 81.32 | 81.32 | 96.99 | 96.99 |
| 峰值激活 | 6.17 | 7.54 | 27.13 | 33.52 |
| CUDA graph | 1.47 | 1.61 | 4.05 | 5.78 |
| KV cache | 158.78 | 157.41 | 122.16 | 115.77 |
| 每 rank KV 容量（tokens） | 2,426,144 | 2,405,184 | 1,866,496 | 1,768,864 |
| 163,840 单请求最大并发（每卡日志值¹） | 14.81× | 14.68× | 11.39× | 10.80× |

实例层面（由上表推算）：

| | tp8 | tep8 | dtp8 | dep8 |
|---|---|---|---|---|
| KV 显存总量 | 1,270 GiB | 1,259 GiB | 977 GiB | 926 GiB |
| 实例 KV 容量 | 2,426,144（= 单 rank，8 份拷贝） | 2,405,184 | 14,931,968（8 池合计） | 14,150,912 |
| 每 token 有效成本 | ~562 B | ~562 B | ~70 B | ~70 B |
| 满上下文（163,840）并发 | 14.8 | 14.7 | 91.1 | 86.4 |

¹ 每卡行 = 本 rank KV 池 ÷ 163,840（Worker 日志值，不含 `max_num_seqs` 等调度限制）；TP 组即实例值（8 份池为同一拷贝），DP 组实例值见下表（×8）。

分析：

- 权重只分两档，由 **Attention 并行方式**决定：TP 组 79.74（权重基本全按 8 切）、DP 组 96.09（专家 1/8 + 非专家完整复制），差 16.35 GiB 即 Attention DP 的显存代价。同档内 TP 切与 EP 字节相同，**显存无法区分 MoE 模式，只能靠 3.2 的日志行**。
- 峰值激活按 TP→DP、TP 切→EP 递增（DP 组的 MoE gather 收全部 rank 的 token；EP 又略大于 TP 切），DP 组每卡 KV 因此更小（122/116 对 158/157 GiB）。
- KV 的本质差异是 **MLA latent 复制**：TP 组 8 卡存同一份内容，每 token 有效成本 8×（562 对 70 B，恰为复制份数）；DP 组 8 个互不重复的池，实例容量合计差 6.2×（不是 8×——每卡池先被 +16.35 GiB 权重、+20~27 GiB 激活侵蚀）。
- DP 组容量是 **8 个独立池的算术合计**：池间不能互借，单请求仍受单 rank 上限约束，负载不均时按最闲的 rank 折算。

### 3.4 进程结构核对（实测，原始输出见附录）

| 核对项 | tp8 | tep8 | dtp8 | dep8 |
|---|---|---|---|---|
| 主进程命令行 | `--tensor-parallel-size 8` | `--tensor-parallel-size 8 --enable-expert-parallel` | `--data-parallel-size 8` | `--data-parallel-size 8 --enable-expert-parallel` |
| `--no-enable-prefix-caching`（命令行可见） | ✓ | ✓ | ✓ | ✓ |
| EngineCore 数 | 1 | 1 | 8（`EngineCore_DP0..7`） | 8（`EngineCore_DP0..7`） |
| DPCoordinator | 无 | 无 | 有 | 有 |
| APIServer 独立进程 | 无（主进程内） | 无（主进程内） | 8（`APIServer_0..7`） | 8（`APIServer_0..7`） |
| Worker 命名 | `Worker_TP0..7` | `Worker_TPn_EPn` | `Worker_DP0..7` | `Worker_DPn_EPn` |
| Worker 父子关系 | 8 Worker 同挂 1 EngineCore | 8 Worker 同挂 1 EngineCore | 每 EngineCore 恰 1 Worker（PPID 一一对应） | 每 EngineCore 恰 1 Worker（PPID 一一对应） |
| 进程数（含主进程、resource_tracker） | 11 | 11 | 27 | 27 |

**结论：4 个实例的进程结构与配置完全匹配**——Worker 命名的 `_TPn`/`_DPn`/`_EPn` 段与 TP/DP/EP 三个开关一一对应，与 3.1/3.2 的日志证据互证。

附带观察：**Worker 累计 CPU（TIME 列）稳定呈 2× 差异**——无 EP 组（tp8/dtp8）约为有 EP 组（tep8/dep8）的两倍；原因未查。

## 4. Benchmark 方法

**记号约定**：`in/out`＝每请求随机输入 / 强制输出的 token 数（`--ignore-eos` 保证输出足额）；`C`＝并发数（闭环压满时 C = `--num-prompts`）；`--max-concurrency`＝同时在途请求上限（仅阶段 4 使用）；"波次"＝num-prompts ÷ max-concurrency。指标：**TTFT**＝首 token 延迟（≈prefill 时间，闭环下含排队）；**TPOT**＝平均每输出 token 的生成时间（decode 步长，不含首 token）；**ITL**＝相邻 token 间隔（流式视角）；P99＝99 分位。

- 工具：`vllm bench serve`；除阶段 7（`--backend openai-chat`，含服务端分词）外均 `--backend vllm`（token id 直发，剔除 APIServer 数差异的 CPU 影响）；闭环压满（`--request-rate 10000`，C = `--num-prompts`），阶段 4 限并发 3 波次、阶段 5 开环 Poisson；`--ignore-eos`；每轮不同 `--seed`；每点 2 次重复（饱和 3 次、ShareGPT 2 次）取均值，方差绝大多数 <3%。
- 公平性：前缀缓存配置层关闭（`non-default args` 确认 `enable_prefix_caching: False`）；测前充分预热（三个 prefill 形状 1000/8000/32000 逐 engine 覆盖 + C=512 + 金丝雀两次计时一致）；Pod 内打 localhost；四 Pod 并行（各占一节点）；全部阶段 RDMA 计数增量均为 0（单节点无跨网卡流量）。
- 准入核查：C=1024 时四组峰值并发均恰为 1024（`max_num_seqs` 默认 ≥1024），无分波排队干扰。
- 阶段矩阵：①单流延迟 ②并发扫描（11 档）③负载形态（5 形状 + KV 压力 3 梯度）④饱和（3 波次 ×3）⑤开环 SLO（2/5/8 rps）⑥ShareGPT 真实数据 ⑦openai-chat 对照。
- 产物：`bench-results/<pod>/`（JSON + GPU 功耗采样 + per-engine 快照含 preemption 计数，每组 136 个文件）；脚本 `sn-warmup2.sh`、`sn-bench2.sh`；分析 `tab.py`、`energy.py`。

## 5. Benchmark 结果（2026-10-15/16，实测）

### 5.1 阶段 1：单流延迟（C=1）

**测试设置的含义**：单流＝系统空闲时一次只发 1 个请求（闭环 `--num-prompts 1`），完成后才发下一个——测的是"单个用户独占实例时的最快体感"。两个形状：短交互（100/100）、长回复（1000/2000），各 2 次取均值；测前已按第 4 章充分预热，数字为稳态值。C=1 时 TPOT = ITL（每步只生成 1 个 token），输出 tok/s 即单请求生成速度。

短交互（100/100）：

| | TTFT (ms) | TPOT (ms) | 输出 tok/s |
|---|---|---|---|
| tp8 | 33.2 | **7.67** | 126.0 |
| tep8 | 34.3 | 8.15 | 118.7 |
| dtp8 | 47.8 | 14.28 | 68.4 |
| dep8 | 89.8 | 15.44 | 61.7 |

长回复（1000/2000）：

| | TTFT (ms) | TPOT (ms) | 输出 tok/s |
|---|---|---|---|
| tp8 | 55.5 | **7.58** | **131.5** |
| tep8 | 59.3 | 8.03 | 124.2 |
| dtp8 | 132.1 | 14.23 | 70.0 |
| dep8 | 196.5 | 15.45 | 64.3 |

- 单流 TPOT：TP 组约为 DP 组的 **1.9 倍速度**（DP 单请求 1 rank 真算、7 个陪跑）。
- TTFT 差距更大：长 prompt 下 dep8 是 tp8 的 3.5 倍；DP 组内 EP 进一步拖慢 prefill（dep8 196 > dtp8 132），TP 组 TTFT 几乎不随 prompt 长度变化。

**从配置角度解释（机理推断）**——单请求延迟取决于"这个 token 由几张卡算"：

| | 单 token 由谁算 |
|---|---|
| tp8 | 8 卡各算 1/8（Attention 切 16 头；MoE 每卡算每专家的 1/8 切片） |
| tep8 | 8 卡各算 1/8（Attention 切 16 头；MoE 每卡算本地 32 专家中被路由命中的） |
| dtp8 | 1 卡算整份（Attention 完整 128 头；MoE 为全批算每专家的 1/8 切片） |
| dep8 | 1 卡算整份（Attention 完整 128 头；MoE 为全批算本地 32 专家） |

（每 step 通信见第 1 章通信总览：TP 组每层 2 次 all-reduce，DP 组仅 MoE 层 gather/scatter、dummy 也要参与。）

- **TP 组 7.7 ms** = 1/8 并行计算 + 每层 2 次 all-reduce（61 层累积约 1–2 ms）+ 调度/采样固定开销；**DP 组 14.3/15.4 ms** = 单卡算完整 Attention、计算不并行，但省掉大部分通信——两者差 1.9 倍而非 8 倍，是通信与固定开销对 TP 理想加速的折损。
- **组内（同 Attention 模式）MoE 两种切法的单流 TPOT 只差 6–8%**（tp8→tep8 +6.3%/+5.9%，dtp8→dep8 +8.1%/+8.6%）：通信路径完全相同，差异只在 kernel 布局（1/8 切片对 32 个完整专家的 grouped GEMM）——远小于 Attention 并行方式的 2 倍差。该方向全阶段一致（EP 均略慢），吞吐口径下差 5–13%。
- **TTFT 差距（3.5×）比 TPOT（1.9×）更大**：prefill 是算力活，TP 组 1000 token 的 prefill 同样被 8 卡分摊；DP 组单卡独算。dep8 额外慢到 196 ms 是 EP 路径下 MoE gather 数据量正比于 token 数，prefill 大批量时被放大——这也是 DP 组内 dep8 比 dtp8 慢的主因。

### 5.2 阶段 2：并发扫描（1000/1000，C ∈ 8–512 共 11 档）

**测试设置的含义**：闭环压满——C 个请求**同一瞬间全部发出**（`--request-rate 10000`），服务端并发即为 C（预检确认峰值并发＝C），全部完成后结束；输出吞吐＝总输出 token ÷ 总耗时，TPOT 为该并发下的 decode 步长。**闭环下 mean TTFT 含排队**：所有请求同时到达、先到者先 prefill，mean TTFT 随 C 增长是闭环的正常特征（真实到达场景由 5.5 开环回答）。档位选择：8–64 粗扫低并发区、128–256 每 32 一档精确定位交叉点、320–512 每 64 一档观察尾延迟悬崖。

输出吞吐（tok/s）：

| C | tp8 | tep8 | dtp8 | dep8 |
|---|---|---|---|---|
| 8 | **760** | 659 | 479 | 437 |
| 32 | **1,987** | 1,827 | 1,478 | 1,392 |
| 64 | **3,079** | 2,896 | 2,535 | 2,424 |
| 128 | **4,871** | 4,714 | 4,468 | 4,326 |
| 160 | 5,308 | 5,177 | **5,461** | 5,282 |
| 192 | 6,078 | 5,852 | **6,229** | 5,940 |
| 224 | 6,693 | 6,449 | **7,090** | 6,801 |
| 256 | 7,258 | 6,855 | **7,629** | 7,579 |
| 320 | 7,642 | 7,379 | **9,099** | 8,554 |
| 384 | 8,475 | 8,159 | **10,067** | 9,762 |
| 512 | 9,645 | 9,172 | **12,419** | 11,705 |

TPOT（ms）与 P99 ITL（ms，C ≥ 256 括号内）：

| C | tp8 | tep8 | dtp8 | dep8 |
|---|---|---|---|---|
| 8 | **10.3** | 11.9 | 16.4 | 18.0 |
| 64 | **19.8** | 20.9 | 24.4 | 25.2 |
| 128 | **24.5** | 25.2 | 27.0 | 27.3 |
| 160 | 28.1 | 28.5 | **27.4** | 27.6 |
| 256 | 32.2 (37.5) | 33.7 (38.6) | **30.9** (34.8) | 30.3 (33.6) |
| 320 | 38.0 (46.3) | 38.9 (48.5) | **32.1** (38.5) | 33.0 (36.2) |
| 384 | 40.8 (**320**) | 41.6 (**400**) | **34.3** (39.2) | 34.4 (40.6) |
| 512 | 47.1 (**331**) | 48.6 (**406**) | **36.6** (43.1) | 37.5 (44.3) |

- **吞吐与 TPOT 的交叉点都在 C=128–160**：128 时 TP 组领先 9%，160 时 DP 组反超 3%。
- **TP 组尾延迟悬崖在 C≈320–384**（本轮出现在 384，起点存在轮间浮动）：tp8 P99 ITL 46 ms（C=320）→ 320 ms（384）；DP 组全程 ≤45 ms。p99 TTFT 同向：C=512 时 tp8 10.7s、tep8 13.0s，对 dtp8 7.1s。

**从配置角度解释（机理推断）**——瓶颈随并发换位：低并发比"单 token 多快"，高并发比"每步总吞吐多少"：

| 每 decode 步的关键量（随 C 增长） | TP 组（tp8/tep8） | DP 组（dtp8/dep8） |
|---|---|---|
| 谁算一个 token | 8 卡各算 1/8 | 1 卡算整份 |
| 每卡批大小 | C（全批，切 1/8） | **C/8**（只算本 rank 的请求） |
| 每步集合通信 | 每层 2 次全批 all-reduce，**数据量 ∝ C** | 每 MoE 层全组 gather/scatter（量也 ∝ C），**Attention 零通信** |
| 调度器 | **1 个** EngineCore 管 C 个请求 | **8 个**各管 C/8 |
| TPOT 从 C=8 到 512 | 10.3 → 47.1 ms（**×4.6**） | 16.4 → 36.6 ms（**×2.2**） |

- **低并发（C ≤ 128）TP 优**：瓶颈是单 token 延迟，TP 把每个 token 切 8 份并行算；DP 每卡整算、每 rank 批只有 C/8（小批无摊薄收益），TP 快 1.6–1.9×。
- **高并发（C ≥ 160）DP 优**：TP 的两项成本随 C 线性变重——每层 2 次 all-reduce 的数据量 ∝ C、单调度器每步遍历 C 个请求的 CPU 开销 ∝ C；DP 的每卡批从 1 涨到 C/8，整卡 GEMM 与每步权重读取被摊得越薄，且 Attention 完全不通信——TPOT 斜率 4.6× 对 2.2×，曲线在 C=128–160 交叉。
- **尾延迟悬崖（C≈320–384）是 TP 的结构性代价**：单调度器 + 每层全批 all-reduce 在大批下的偶发停顿直接打到 P99 ITL（300–400 ms）；DP 的 8 个小调度器天然分摊，无此结构。
- MoE 切法（组内）不改变上述任何一项——通信事件数与数据量、调度结构都相同——所以组内差异全程保持小幅（见 5.1）。

### 5.3 阶段 3：负载形态

**测试设置的含义**：五个形状 + KV 压力三梯度，全部闭环、各 2 次取均值，形状设计：

| 形状 | 设计意图 |
|---|---|
| 纯 prefill（8000/1，C=64） | 输出只有 1 个 token——只做 prefill 不进入 decode 循环，隔离测 **prefill 容量**（总输入 token ÷ 总耗时） |
| prefill 为主（8000/1000，C=64） | 8:1 的输入输出比，模拟文档摘要类负载，看混合时 prefill 对 decode 的干扰 |
| decode 为主（1000/8000，C=64） | 95% 时间在 decode，看长输出的持续 decode 能力 |
| 长上下文（20000/1000，C=16） | 20k 超过 `max_num_batched_tokens`（16384），每请求分 2 块 prefill——测分块路径；C=16 使单请求 KV 占比更大 |
| KV 压力（32000/4096，C=64/80/104） | KV 需求 2.36M/2.95M/3.83M＝TP 组实例容量（2.41–2.43M）的 97%/121%/158%、DP 组合计容量（14.2–14.9M）的 16–27%——梯度设计使只有 TP 组触顶 |

C=64 取交叉点（128–160）之下，使形状效应不被高并发效应淹没。

纯 prefill（8000/1，C=64）：

| | prefill 容量 (tok/s) | TTFT (ms) |
|---|---|---|
| tp8 | 47,734 | 5,613 |
| tep8 | 39,483 | 6,791 |
| dtp8 | **61,853** | **5,143** |
| dep8 | 45,896 | 6,732 |

prefill 为主（8000/1000，C=64）：

| | 输出 tok/s | TTFT (ms) | TPOT (ms) | P99 ITL (ms) |
|---|---|---|---|---|
| tp8 | 1,771 | 5,694 | 30.2 | **345** |
| tep8 | 1,626 | 6,867 | 32.3 | **416** |
| dtp8 | **1,885** | **5,223** | **28.7** | 28 |
| dep8 | 1,717 | 6,665 | 30.6 | 29 |

decode 为主（1000/8000，C=64）：

| | 输出 tok/s | TTFT (ms) | TPOT (ms) |
|---|---|---|---|
| tp8 | **2,723** | 916 | **23.4** |
| tep8 | 2,650 | 1,193 | 24.0 |
| dtp8 | 2,473 | 847 | 25.8 |
| dep8 | 2,435 | 1,249 | 26.1 |

长上下文（20000/1000，C=16，分块 prefill）：

| | 输出 tok/s | TTFT (ms) | TPOT (ms) |
|---|---|---|---|
| tp8 | **674** | **4,275** | **19.4** |
| tep8 | 598 | 5,008 | 21.6 |
| dtp8 | 581 | 5,924 | 21.6 |
| dep8 | 530 | 6,980 | 23.2 |

KV 压力（32000/4096，三梯度；TP 组实例容量 2.41–2.43M、DP 组合计 14.2–14.9M tokens；preemption 列＝rep1 前后各 engine 的 `vllm:num_preemptions_total` 计数器差值合计，单位：次）：

| 梯度 | KV 需求 | 指标 | tp8 | tep8 | dtp8 | dep8 |
|---|---|---|---|---|---|---|
| **97%**（C=64） | 2.36M，装得下 | 输出 tok/s | 1,038 | 996 | **1,482** | 1,415 |
| | | TTFT mean/p99 (s) | 29.0 / 56.4 | 33.0 / 64.1 | 27.0 / 50.2 | 31.6 / 57.3 |
| | | TPOT (ms) / P99 ITL (ms) | 54.2 / **431** | 55.8 / **493** | **36.6 / 33** | 37.5 / 34 |
| | | preemption | **0** | **0** | 0 | 0 |
| **121%**（C=80） | 2.95M，超 21% | 输出 tok/s | 912 | 865 | **1,741** | 1,650 |
| | | TTFT mean/p99 (s) | 49.5 / **292** | 57.3 / **302** | **32.0 / 59** | 38.2 / 70 |
| | | TPOT / P99 ITL (ms) | 61.3 / 462 | 62.3 / 523 | **38.1 / 34** | 39.1 / 35 |
| | | preemption | **8** | **8** | 0 | 0 |
| **158%**（C=104） | 3.83M，超 58% | 输出 tok/s | 954 | 911 | **2,048** | 1,911 |
| | | TTFT mean/p99 (s) | **107.8 / 312** | **116.9 / 327** | **40.7 / 75** | 48.1 / 89 |
| | | TPOT / P99 ITL (ms) | 59.1 / 468 | 60.8 / 529 | **40.8 / 35** | 42.6 / 35 |
| | | preemption | **8** | **8** | 0 | 0 |

- 纯 prefill 与 prefill 为主：**DP 组在 C=64 即反超**（dtp8 prefill 容量 1.30× tp8），且 TP 组 P99 ITL 暴涨（分块 prefill 卡 decode 步，345–416 ms 对 28 ms）。
- decode 为主、中低并发：TP 组优（+10%）；长上下文低并发：TP 组大幅优（TTFT 快 39%）。
- **KV 压力梯度是两种形态最硬的分水岭**：
  - **97%（无抢占）TP 组就已显著劣化**——吞吐只有 dtp8 的 70%、TPOT 54 对 37 ms、P99 ITL 431 ms（KV 接近满载时单调度器的调度压力，机制未深挖，实测记录）；
  - 过载后退化非线性：TP 组 TTFT 从 49.5s（121%）涨到 107.8s（158%），吞吐比从 1.91× 扩大到 2.15×；
  - DP 组三梯度全程 **0 抢占、P99 ITL ≤36 ms**——容量优势在压力下全额兑现。

**抢占（preemption）机制——"请求之间互踢"**：

vLLM 不拒绝请求——池子满了也想接，调度器**驱逐**若干正在运行的请求（KV cache 释放、退回等待队列），腾出空间给新请求；被驱逐的请求重新排队后必须**从头重算整个 32k prompt 的 prefill**（本配置关了 prefix caching，一块都省不下来）。代价双重：被驱逐请求自身延迟爆炸（32k 重算），且重算的 prefill 占用算力、拖慢池内所有请求的 decode（P99 ITL 被拖到 431–529 ms）。

三梯度实为两种现象：**97% 装得下、0 次抢占，劣化来自近满池的调度压力**（非抢占，机制未深挖）；**121%/158% 超容、各 8 次抢占**——驱逐到刚好装得下为止（次数不随超容比例涨），但队列更深、等待更久（TTFT 49.5s → 107.8s）。

**DP 组为何 0 抢占**：同样需求摊到 8 个独立池，每池只装约 25%（10–13 个请求 × ~37k ≈ 0.4M 对单池 1.87M），任何池都远未触顶——TP 组是"一个池大家挤、挤满互踢"，DP 组是"8 个池各装各的、谁也挤不着谁"。

**从配置角度解释（机理推断）**：

- **prefill 密集时 DP 反超、且 TP 组 decode 被卡**：prefill 批大（64×8000 token），TP 组每层 2 次 all-reduce 的数据量 ∝ 批内 token（16,384-token 分块 × 14 KB ≈ 235 MB/次、每分块 122 次），通信开始吃掉切分收益；DP 组 Attention 零通信、每 rank 独立 prefill 自己的约 1/8 请求，只剩事件数减半的 MoE gather/scatter。P99 ITL 暴涨是同一结构：单调度器里 prefill 分块与 decode 共用一个串行队列，DP 的 8 个小队列天然错开。
- **EP 在 prefill 场景额外拖慢**（组内 TP 切快于 EP：tp8>tep8 21%、dtp8>dep8 35%，远大于 decode 口径的 6–8%）：整专家 grouped GEMM 的形状随路由波动（每专家 token 数不均），不如每专家 1/8 切片的固定形状 GEMM 高效。
- **decode 为主（C=64）与长上下文（C=16）都在 5.2 的低并发区**，按 5.1/5.2 的低并发机理 TP 占优；长上下文 TTFT 差距更大，因为 20k prompt 的 prefill 被 TP 8 卡分摊（2 个分块）、DP 单卡独算。
- **KV 压力是"单池单调度"的结构性代价**：TP 组容量是一个共享池（latent 8 份拷贝共 2.4M），全部请求挤一个调度器与一个池——近满（97%）时调度器在近乎无余量的池上做准入/驱逐决策（劣化但未深挖），超容后 preempt 即整个 32k prompt 重算；DP 组是 8 个独立池各装约 1/8 需求，任何梯度下都远未触顶。

### 5.4 阶段 4：饱和吞吐（500/512，num-prompts=3072 + max-concurrency 1024，3 波次，3 次重复）

**测试设置的含义**：饱和吞吐＝系统满负荷时的最大持续产出（tok/s），对应离线批处理场景（不关心单请求延迟，只关心总产能）。参数值：500/512＝每个请求 500 随机输入 token、强制生成 512 个输出；3072 个请求一次性发起，但 `--max-concurrency 1024` 限制同时在途最多 1024 个——完成一个放行一个，即**恒定 1024 并发连续 3 个波次**，总产出固定 157.3 万 token：

```
3072 个请求（总量）
 ├─ 波 1：1024 并发 ← 任意时刻保持 1024 在跑
 ├─ 波 2：1024
 └─ 波 3：1024
时间 ════════════════════════▶  ~85–115 s（吞吐 = 157.3 万 ÷ 总耗时）
```

数值选择的理由：**500/512** 小形状使 KV 占用约 1M tokens（1024 × ~1000），远低于 TP 组 2.4M 实例容量——排除 KV 干扰、纯测产能；**1024** 是实测的准入上限（`max_num_seqs` 默认 ≥1024，四组峰值并发均恰好 1024，更高只会转为服务端排队）；**3072 = 3×1024** 用后两个波次摊薄首波瞬态（全部请求同时涌入排队做 prefill），单波测的是"一批完成时间"、三波测的是稳态产能。

| | 输出 tok/s | TTFT mean/p99 (ms) | TPOT (ms) | P99 ITL (ms) |
|---|---|---|---|---|
| tp8 | 14,065 | 2,719 / 10,710 | 67.2 | 345 |
| tep8 | 12,775 | 3,513 / 13,362 | 73.0 | 422 |
| dtp8 | **18,504** | 3,706 / **7,988** | **48.0** | **111** |
| dep8 | 16,236 | 5,310 / 10,232 | 52.6 | 105 |

dtp8 饱和吞吐为 tp8 的 **1.32×**，TPOT 优 29%，P99 ITL 只有 TP 组的 1/3；TP 组 p99 TTFT 10.7–13.4 s（多波次排队尾延迟）。

**从配置角度解释（机理推断）**：饱和是 5.2 高并发机理的外推——C=1024 时 TP 组的两项成本同时到顶（单调度器管 1024 个请求、每层全批 all-reduce = 1024 token × 14 KB × 每步 122 次），DP 组每 rank 批恰为 128（落在交叉点附近的高效区）且 Attention 零通信；TP 组 p99 TTFT 10.7–13.4 s 是波次间排队在单队列上的体现，DP 的 8 个队列天然分摊。

### 5.5 阶段 5：开环 SLO（1000/1000，Poisson 到达 2/5/8 rps × 240 prompts）

**测试设置的含义**：前面各阶段均为闭环（请求压满、完成一个补一个），测"给定并发的最大能力"；本阶段为**开环**——请求按 Poisson 时间表发出（平均每秒 R 个、间隔随机），**与服务端状态无关**，服务端慢了请求照样到达、只在队列堆积。测的是"给定业务到达率，用户实际感受的延迟"（SLO 视角）：到达率 < 产能则队列有界、延迟 ≈ 天然服务延迟；到达率 > 产能则 TTFT 发散（闭环测不出这一点）。每请求 1000/1000，共 240 个（2 rps 到达期约 120 s、8 rps 约 30 s），三档速率的隐含并发（Little 定律 L=λW，W≈TTFT+999×TPOT）：

| 到达率 | 隐含并发 | 相对 tp8 产能（≈9.6 req/s） |
|---|---|---|
| 2 rps | ≈ 40 | 21%（轻载） |
| 5 rps | ≈ 170 | 52%（中载） |
| 8 rps | ≈ 290 | 83%（接近饱和） |

三档均未超产能——因此表中吞吐四组接近（完成率被到达率限定，开环的正常特征），**差异全部体现在延迟列**（TPOT 与闭环曲线吻合）：

| 到达率 | 指标 (mean/p99) | tp8 | tep8 | dtp8 | dep8 |
|---|---|---|---|---|---|
| 2 req/s | TTFT (ms) | **145 / 264** | 156 / 273 | 173 / 304 | 220 / 341 |
| | TPOT (ms) | **20.2** | 22.4 | 27.2 | 30.6 |
| 5 req/s | TTFT (ms) | **187 / 370** | 195 / 394 | 200 / 353 | 254 / 465 |
| | TPOT (ms) | **33.0** | 33.9 | 35.7 | 38.6 |
| 8 req/s | TTFT (ms) | **206 / 391** | 204 / 374 | 230 / 451 | 293 / 461 |
| | TPOT (ms) | **35.7** | 35.5 | 36.5 | 38.1 |

**开环下 TP 组延迟全面最优**（三档 TTFT/TPOT 全部最低，dep8 每项最差、TTFT 差 40–50%）：需求不超饱和时 TP 是延迟最优选择；DP 的吞吐优势只在需求超饱和时兑现。

**从配置角度解释（机理推断）**：三档隐含并发（40/170/290）大多低于交叉点，开环下系统大部分时间未饱和——处于"单 token 延迟主导"区（5.1 机理），TP 的切分优势直接兑现；DP 只有在排队积压进入大批摊薄区后才反超，而 SLO 场景不允许积压。dep8 每项最差是双重叠加：单 rank 整算（DP）+ EP 的 prefill 惩罚（5.3）都压在 TTFT/TPOT 上。

### 5.6 阶段 6：ShareGPT 真实数据（闭环，2 次重复均值）

**测试设置的含义**：数据集为 ShareGPT_V3 过滤子集 5,000 条（≥2 轮对话，pod 内 `/tmp/ShareGPT_subset.json`）；每请求取对话**第 1 轮为 prompt、第 2 轮为期望输出**——输入输出长度都随真实对话变化（区别于其他阶段的固定长度）。闭环 C=64/256，各 2 次；**同 rep 同 C 四组用同一 seed** → 数据集洗牌一致 → 样本集相同 → 同 C 跨配置可比；不同 C 用不同 seed → **跨档位不可比**（抽到的对话不同）。

| | C | 输出 tok/s | TTFT mean/p99 (ms) | TPOT (ms) |
|---|---|---|---|---|
| tp8 | 64 | **1,699** | 274 / 343 | **18.1** |
| tep8 | 64 | 1,556 | 319 / 339 | 18.5 |
| dtp8 | 64 | 1,063 | 2,527 / 2,662 | 23.3 |
| dep8 | 64 | 999 | 2,565 / 2,675 | 24.4 |
| tp8 | 256 | **4,286** | 1,870 / 2,107 | **26.3** |
| tep8 | 256 | 4,172 | 1,862 / 2,047 | 25.8 |
| dtp8 | 256 | 3,220 | 3,927 / 4,136 | 28.8 |
| dep8 | 256 | 3,451 | 1,873 / 2,058 | 28.3 |

**真实对话（输出偏短）下 TP 组两个档位都领先**：C=64 时 tp8 为 dtp8 的 1.60×；C=256 时 TP 组仍领先（tp8 4,286 最高、dtp8 最低 3,220）——与随机数据集的交叉点（128–160）相比，真实短输出把交叉点**右移到 256 以外**：输出越短，TP 的每层 all-reduce 开销越难被吞吐差异压过，TP 优势区间越大。

**从配置角度解释（机理推断）**：每档是固定 num-prompts 的一次性波（闭环压满、无补充）——输出短则波快速排空，全程平均并发远低于 C，等效于运行在更低的并发档，即落回 TP 优势区（5.2 低并发机理）；同时短输出使 TTFT 占请求时延的比例上升，TP 的 TTFT 优势（8 卡合力 prefill）被放大。交叉点的位置由"稳态 decode 批大小"决定——批大小由活跃请求数决定，输出越短同 C 下的持续批越小。

### 5.7 阶段 7：openai-chat 对照（含服务端分词）

**测试设置的含义**：`--backend openai-chat` + `--endpoint /v1/chat/completions`——请求以**文本**经真实 chat API 进入（服务端做分词 + chat 模板），区别于其他阶段 token id 直发 `/v1/completions`（绕过服务端分词）。目的：量化入口差异——TP 组只有 1 个 APIServer（主进程内）对 DP 组 8 个 APIServer，看分词与 HTTP 处理是否成为 TP 组瓶颈。两个形状各 1 次：1000/1000 C=256（与阶段 2 同档直接对照）、500/512 C=1024（入口高压）。

| | 用例 | 输出 tok/s | TPOT (ms) | P99 ITL (ms) |
|---|---|---|---|---|
| tp8 | 1000/1000 C=256 | 7,262 | 32.1 | 59 |
| tep8 | 1000/1000 C=256 | 7,094 | 32.4 | 59 |
| dtp8 | 1000/1000 C=256 | 7,549 | 31.2 | 36 |
| dep8 | 1000/1000 C=256 | 7,473 | 30.4 | 35 |
| tp8 | 500/512 C=1024 | 13,754 | 61.7 | 340 |
| tep8 | 500/512 C=1024 | 12,911 | 64.4 | 413 |
| dtp8 | 500/512 C=1024 | **18,984** | **45.1** | **118** |
| dep8 | 500/512 C=1024 | 16,891 | 48.2 | 86 |

与免分词的 `--backend vllm` 路径相比几乎无差（C=256 档与阶段 2 同档对比：tp8 7,262 对 7,258、dtp8 7,549 对 7,629，均 <1.5%）：**服务端分词与 TP 组单 APIServer 不构成瓶颈**。

**从配置角度解释（机理推断）**：分词是 CPU 端毫秒级工作、发生在请求入口，而 GPU 每步 30–60 ms 的批处理远慢于分词——TP 组单 APIServer 的处理能力在本负载的 QPS 下未触顶，两条路径的差距被 GPU 时间淹没。TP 组的 APIServer 瓶颈要在更高 QPS 或超长 prompt 的分词下才会显现（未测）。

### 5.8 能效（rep1 GPU 功耗采样，J/输出 token）

**测试设置的含义**：非独立测试——每用例第 1 次重复运行时，Pod 内后台采样器每 2 s 记录一次 8 卡功率/利用率（`*.gpu.log`）；能耗＝采样期平均合计功率 × 该轮时长，J/输出 token＝能耗 ÷ 该轮输出 token 数。工作点取自阶段 1（单流）、2（各 C）、4（饱和）。精度受 2 s 采样间隔与均值法限制——横向对比足够，非计费级精度。

| 工作点 | tp8 | tep8 | dtp8 | dep8 |
|---|---|---|---|---|
| 单流（1000/2000） | **19.0** | 19.8 | 46.7 | 49.3 |
| C=8 | **3.80** | 4.39 | 7.22 | 7.74 |
| C=128 | **1.13** | 1.12 | 1.18 | 1.17 |
| C=160 | 1.05 | 1.05 | **0.97** | 0.97 |
| C=256 | 0.80 | 0.84 | **0.73** | 0.72 |
| C=512 | 0.67 | 0.67 | **0.49** | 0.48 |
| 饱和（3 波次） | 0.48 | 0.49 | **0.34** | 0.34 |

- **能效交叉点与吞吐/TPOT 交叉点重合（C=128–160）**；低负载 TP 组能效优势巨大（C=8 时 1.9×、单流 2.5×——DP 组 1 个请求拖 8 卡陪跑，约 3.1–3.2 kW 白烧）；饱和时 dtp8 能效高 ~40%。
- 单流到饱和每 token 能耗差 40–60 倍：**负载率是能效的第一变量，并行形态是第二变量**。

**从配置角度解释（机理推断）**：GPU 功耗近似"档位"（低载约 1.9–3.2 kW、满载约 5.5–6.7 kW），与做的功是否有效无关——能效 = 功率 × 时间 ÷ 有效 token。低负载下 DP 组的结构性代价直接变成电费：1 个请求也要 8 卡跑 dummy 前向，功率照烧而有效 token 只有 1 份；饱和下 dtp8 每秒多产出 32% token 而各组功率相近（5.5–6.7 kW），每 token 能耗即低 40%。功率全程只变化 2–3 倍、吞吐变化 40–60 倍——所以负载率（决定分母）压过并行形态（决定倍率）。

### 5.9 汇总：场景选型

| 场景 | 推荐 | 依据 |
|---|---|---|
| 单流 / 低并发在线交互（延迟敏感） | **tp8** | TPOT 7.7 ms（DP 的 1.9 倍速度）、TTFT 最低、单流能效 2.5× |
| 长上下文、低并发 | **tp8** | 20k prompt TTFT 4.27 s（dep8 6.98 s） |
| decode 为主、中低并发（C ≤ 128） | **tp8 / tep8** | 吞吐与 TPOT 均优（随机与真实数据一致） |
| 真实对话负载（短输出） | **TP 组优势区间扩大** | ShareGPT C=64/256 均 TP 领先，交叉点右移到 256 以外 |
| prefill 为主的混合负载 | **dtp8** | C=64 即反超；P99 ITL 28 ms 对 TP 组 345–416 ms |
| 长上下文高并发（KV 压力） | **dtp8 / dep8** | TP 组即使不超容（97%）也已劣化，超容后 TTFT 50–108 s、抢占重算——实际不可用 |
| 高并发 / 饱和 / 离线批量（C ≥ 160，随机负载） | **dtp8** | 吞吐 1.32×、TPOT 优 29%、能效 +40%、P99 ITL 1/3 |
| 开环真实到达率（未超饱和） | **TP 组** | 三档速率 TTFT/TPOT 全部最优 |

- **并发（且 prefill 占比、输出长度）是选型的主变量**：随机均衡负载下三个交叉点（吞吐、TPOT、能效）重合在 C=128–160；prefill 为主左移（C=64 即 DP 优）、真实短输出右移（>256 仍 TP 优）。
- MoE 切法：随机数据下无 EP 的 TP 切全程略优（3–14%）；ShareGPT 下 TP 组同样领先。**本负载（均匀/随机路由）下 EP 无优势**；真实倾斜路由下 EP（无 EPLB）可能劣化，未测。
- **EP（dep8）为何在本环境无优势——Wide EP 的收益条件一个不占**：业界的 Wide EP（DeepSeek-V3/R1、Kimi K2 等巨型 MoE 的生产部署：Attention 用 DP、MoE 用 EP、几十~几百卡）在四个前提下才兑现优势——
  1. **token 级 all-to-all**（只把 token 发给选中专家所在的卡，通信 ∝ 有效流量；本测试用的 AgRs 是全量 all-gather，且 DeepEP 在 B300/sm_103 不可用）；
  2. **跨节点**（通信贵的网络才省得值钱；本测试单节点 NVLink）；
  3. **大批**（生产单步几万 token，整专家 GEMM 才高效；本测试每专家每步仅 ~4–32 个 token，实测整专家布局在 prefill 反而慢 21–35%）；
  4. **大 EP 度数**（每卡 1–2 个专家；本测试 EP8 每卡 32 个、与 TP 切每卡字节相同）。

  故本测试中 dep8 的优势全部来自 DP attention（与 dtp8 平分），EP 布局本身是纯劣势；本测试的均匀随机路由（无热点）已是对 dep8 最有利的条件。
- 运维约束：TP 组高并发尾延迟悬崖（C≈320–384，P99 ITL 300–400 ms）与 KV 容量硬限（2.4M tokens）是部署前必须预算的约束；DP 组则为每卡 +16.35 GiB 权重与低负载陪跑能耗买单。

## 6. 未验证项

- 97% KV 梯度下 TP 组劣化的机制（调度压力？未深挖）。
- TP 组尾延迟悬崖起点在 320–384 间浮动的成因（调度随机性，未深挖）。
- `ep:0` 通信组在无 EP 配置下的用途。
- dtp8 每 rank 算"每个专家 1/8 切片"为推断（日志+权重字节，未核对源码传参）。
- EPLB / 倾斜路由下的 MoE 切法对比（需另开部署轮次）。
- tep8/dep8 的 EP rank 1–7 专家区间按 linear 推断（只核对了 rank 0）。
- Worker 空转 CPU 2× 差异的原因。
- NCCL 集合通信的实际算法（环形/树形/多播混合，第 7 章实测/模型比 65–78% 的成因）。
- DP 组单请求下 NVLink 每 GPU 分布不均（26.5–41.8 GiB）的机制。
- 工具调用（`--tool-call-parser deepseek_v3` + chat template）在本环境的可用性未测。

## 7. 流量实测：32768/128 单请求（NVLink 字节级）

实测四种配置一次请求（32768 in / 128 out，单请求，闭环）产生的**实际 NVLink 流量**，并与通信模型对照。RoCE 不测：单节点全部集合通信走节点内 NVLink（第 4 章整役 RDMA 计数恒为 0）。

### 7.1 方法

- 形状：**32768 in / 128 out，单请求，闭环**——32768 = 2×16384，正好走 2 块分块 prefill；128 步 decode 的流量可用 `32768/128 − 32768/1` 分离。
- 工具：`sn-traffic.sh`（Pod 内执行）+ `traffic-tools/`（`nvlink_counters.py` 等 NVLink/RDMA 只读计数工具）。每轮请求前后各采一次 `nvidia-smi nvlink -gt d` 的每 GPU 累计 Tx（KiB），取差值；engine 归属取自 `/metrics`。
- 流程：空闲 10 s 基线（差值为 0）→ 预热 32768/8 至全 engine 覆盖（1 轮即达 1/1 或 8/8）→ **5 轮 32768/128 + 3 轮 32768/1**，每配置共 8 个请求、每请求独立前后快照。
- 轮间方差极小（TP 组节点合计 487.3–487.4 GiB，0.02%）；空闲基线四组均为 0（无背景 NVLink 流量）。原始输出存 `traffic-results/<pod>.txt`。

### 7.2 通信模型预测（环形算法，bf16，14.336 KB/token）

预测是一笔四步的"通信账"：

**① 基本单位：14.336 KB/token**。集合通信搬的是 token 的隐藏状态：7168 维 × 2 字节（bf16）＝14.3 KB——一次通信量 = 参与 token 数 × 14.3 KB。

**② 一次 collective：411 MB/卡**。分块 prefill 一步一个 chunk＝16,384 token → 整批隐藏状态 16,384 × 14.3 KB ≈ 235 MB；环形算法下数据沿环传递、每卡替别的卡转发，每卡实际发送 ≈ 2×(8−1)/8 ≈ 1.75× 批数据 → **235 × 1.75 ≈ 411 MB**。（all-reduce 与 AG+RS 字节数等价：一次 all-reduce 本身就是 reduce-scatter + all-gather 拼成的。）

**③ 每层交几次"411 MB"——TP 与 DP 的全部区别**：

| | 每层交几次 | 原因 |
|---|---|---|
| TP 组（tp8/tep8） | **2 次** | Attention 的 o_proj 后一次 + MoE 结尾一次，每层都交 |
| DP 组（dtp8/dep8） | **1 次** | 只有 MoE 层交（AG+RS 合计＝一次 all-reduce 的量）；Attention 层零通信；前 3 层稠密零通信 |

**④ 乘起来**（32768 token = 2 个 chunk，每 chunk 过全部层）：

| | 每 chunk 每卡 | 2 chunk prefill（每 GPU） | decode 128 步（每 GPU） |
|---|---|---|---|
| tp8/tep8 | 61 层 × 2 × 411 MB ≈ 50 GB | **≈100 GB（93 GiB）** | +0.4 GB（每步批=1 token） |
| dtp8/dep8 | 58 层 × 1 × 411 MB ≈ 24 GB | **≈48 GB（44 GiB）** | +1.5 GB（每步批=8 token：1 真+7 dummy） |

**TP ≈ 2.1× DP 的原因一句话**：TP 每层交两份（Attention 也得交），DP 只交一份（Attention 免单）。另两个预测：tp8≈tep8、dtp8≈dep8——字节上通信路径完全相同（MoE 切法只改本地计算布局）。

### 7.3 实测结果（每 GPU 平均 Tx，GiB）

| | tp8 | tep8 | dtp8 | dep8 |
|---|---|---|---|---|
| 32768/128（整请求） | 60.9 | 61.2 | 36.9 | 36.9 |
| 32768/1（纯 prefill） | 60.7 | 61.0 | 34.7 | 34.7 |
| decode 贡献（128−1） | +0.26 | +0.25 | +2.16 | +2.16 |
| 节点合计（整请求，GiB） | 487.3 | 489.9 | 295.2 | 295.2 |
| TTFT（ms） | **987** | 1,113 | 3,118 | 3,165 |
| 纯 prefill 时长（s） | **1.00** | 1.11 | 3.11 | 3.15 |
| 每 GPU 有效 NVLink 带宽 | 61 GB/s | 60 GB/s | 11 GB/s | 11 GB/s |
| 每 GPU 分布 | 均匀（60.9×8） | 均匀 | **不均：26.5–41.8** | **不均：28.9–41.8** |

### 7.4 解读（配置 → 通信模型）

- **TP 组流量是 DP 组的 1.65×**（487 对 295 GiB），方向与模型一致、幅度低于模型的 2.1×：实测/环形模型之比 TP 组 65%、DP 组 78%——**NCCL 实际用的不是纯环形算法**（多播/树形混合，具体协议未验证）。
- **差距 100% 来自 Attention，MoE 部分两组打平**：每层每卡（一个 16,384-token chunk）——Attention 列 TP 组交 411 MB、DP 组交 0；MoE 列 TP 的 1 次 all-reduce ≈ DP 的 AG+RS（字节等价）。即两种并行哲学的差别：TP 切"一个 token 的计算"（128 头分 8 卡），每层必须 all-reduce 拼回部分和——**只要切了单个 token，每层每个 token 都要付这次合并**；DP 不切单个 token、各 rank 独立算完整 Attention，**没有东西需要合并**（代价是单 token 无 8 卡加速，即 5.1 的 1.9× 速度差）——通信量和计算并行度是同一枚硬币的两面。
- **dtp8 与 dep8 的字节完全相同**（295.2 / 277.9 GiB，逐轮一致到 0.1 GiB）——MoE 两种切法走同一条 AgRs 路径的**字节级铁证**：性能差异（5.1–5.3 的 6–35%）全部来自 kernel 布局，与通信量无关。tp8 ≈ tep8（差 0.5%）同理。
- **流量多 ≠ 慢：TP 组多搬 1.65× 的字节却快 3.1×**（TTFT 987 对 3,118 ms）。有效 NVLink 带宽 tp8 仅 61 GB/s（NV18 956 GB/s 的 6%）、dtp8 仅 11 GB/s（1%）——**NVLink 远未跑满，单请求 prefill 是计算/组织受限，不是带宽受限**。TP 的优势来自切分计算（8 卡并行），DP 的劣势来自单卡串行——流量是结果，不是原因。
- **decode 贡献很小**：TP 组每 GPU 仅 +0.26 GiB（128 步 × 每步 122 次小 AR，延迟型）；DP 组 +2.16 GiB（每步 gather 含 8 个 rank 的 dummy token）。DP 组 decode 实测比模型高 ~55%（模型 1.4 GiB），可能与 dummy 补齐/同步有关（未验证）。
- **每 GPU 分布**：TP 组完全均匀（每个 rank 处理全批的 1/8）；DP 组不均（26.5–41.8 GiB）且随请求落点变化——单请求只在一个 rank 上，gather 的各 rank 发送量在算法内不对称（机制未深挖）。
- **TTFT 对照**（单请求 32k prefill）：TP 987/1,113 ms 对 DP 3,118/3,165 ms（3.2×，与 5.3 长上下文结论一致）。

## 运维备注

- 部署：`kubectl apply -f sn-*.yaml`（四文件）；日志：`kubectl logs -f <pod>-0`；验证：`kubectl exec <pod>-0 -- curl -s localhost:8000/v1/models`；删除：`kubectl delete -f sn-*.yaml`（HF 缓存保留）。
- 预热：`kubectl exec -i <pod>-0 -- bash -s < sn-warmup2.sh`——三个 prefill 形状逐 engine 覆盖 + C=512 + 金丝雀；**每次 Pod 重建后必须重跑**。
- Benchmark：`kubectl exec -i <pod>-0 -- bash -s < sn-bench2.sh <phase 1-7>`；结果 `bench-results/<pod>/`；汇总 `bench-results/tab.py <pattern>`；能效 `bench-results/energy.py <pattern>`。
- ShareGPT 数据集：pod 内 `/tmp/ShareGPT_subset.json`（dev 机 `/tmp/ShareGPT_subset.json` 为源；**Pod 重建后需重新 kubectl cp**）。
- 阶段 6 的 sharegpt 需 `--dataset-path`、阶段 7 的 openai-chat 需 `--endpoint /v1/chat/completions`（脚本已内置）。
- 流量实测：`sn-traffic.sh`（Pod 内执行，NVLink 计数器内嵌于脚本）；`traffic-tools/`（独立 NVLink/RDMA 只读计数工具，备将来扩展用）；原始输出存 `traffic-results/<pod>.txt`；汇总表 `traffic-analyze.py`（解析原始输出，输出第 7.3 节格式）。
- 当前（2026-10-16）四组全部运行中，4 个 GPU 节点占满。

## 附录：进程清单（实测，`ps -ef`）

### tp8-0

```bash
root           1       0  0 16:09 ?        00:01:01 /usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --tensor-parallel-size 8 --no-enable-prefix-caching --enable-auto-tool-choice --tool-call-parser deepseek_v3 --chat-template examples/tool_chat_template_deepseekv3.jinja --reasoning-parser deepseek_v3
root        4091       1  0 16:27 ?        00:00:00 /usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(102)
root        4092       1  1 16:27 ?        00:01:52 VLLM::EngineCore
root        4323    4092 29 16:28 ?        00:29:30 VLLM::Worker_TP0
root        4324    4092 29 16:28 ?        00:29:44 VLLM::Worker_TP1
root        4325    4092 29 16:28 ?        00:29:54 VLLM::Worker_TP2
root        4326    4092 29 16:28 ?        00:30:17 VLLM::Worker_TP3
root        4327    4092 29 16:28 ?        00:29:38 VLLM::Worker_TP4
root        4328    4092 28 16:28 ?        00:28:51 VLLM::Worker_TP5
root        4329    4092 28 16:28 ?        00:29:03 VLLM::Worker_TP6
root        4330    4092 28 16:28 ?        00:29:15 VLLM::Worker_TP7
```

### tep8-0

```bash
root           1       0  0 16:09 ?        00:00:58 /usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --enable-expert-parallel --tensor-parallel-size 8 --no-enable-prefix-caching --enable-auto-tool-choice --tool-call-parser deepseek_v3 --chat-template examples/tool_chat_template_deepseekv3.jinja --reasoning-parser deepseek_v3
root        3942       1  0 16:27 ?        00:00:00 /usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(102)
root        3943       1  1 16:27 ?        00:01:38 VLLM::EngineCore
root        4174    3943  8 16:27 ?        00:08:50 VLLM::Worker_TP0_EP0
root        4175    3943  8 16:27 ?        00:08:57 VLLM::Worker_TP1_EP1
root        4176    3943  8 16:27 ?        00:08:55 VLLM::Worker_TP2_EP2
root        4177    3943  8 16:27 ?        00:08:47 VLLM::Worker_TP3_EP3
root        4178    3943  8 16:27 ?        00:08:52 VLLM::Worker_TP4_EP4
root        4179    3943  8 16:27 ?        00:08:52 VLLM::Worker_TP5_EP5
root        4180    3943  8 16:27 ?        00:08:53 VLLM::Worker_TP6_EP6
root        4181    3943  8 16:27 ?        00:08:54 VLLM::Worker_TP7_EP7
```

### dtp8-0

```bash
root           1       0  0 16:15 ?        00:00:39 /usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --data-parallel-size 8 --no-enable-prefix-caching --enable-auto-tool-choice --tool-call-parser deepseek_v3 --chat-template examples/tool_chat_template_deepseekv3.jinja --reasoning-parser deepseek_v3
root        3728       1  0 16:27 ?        00:00:00 /usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(40)
root        3729       1  0 16:27 ?        00:00:26 VLLM::DPCoordinator
root        3929       1  1 16:27 ?        00:01:07 VLLM::EngineCore_DP0
root        3930       1  1 16:27 ?        00:01:16 VLLM::EngineCore_DP1
root        3931       1  1 16:27 ?        00:01:15 VLLM::EngineCore_DP2
root        3932       1  1 16:27 ?        00:01:15 VLLM::EngineCore_DP3
root        3933       1  1 16:27 ?        00:01:16 VLLM::EngineCore_DP4
root        3934       1  1 16:27 ?        00:01:17 VLLM::EngineCore_DP5
root        3935       1  1 16:27 ?        00:01:06 VLLM::EngineCore_DP6
root        3936       1  1 16:27 ?        00:01:15 VLLM::EngineCore_DP7
root        3937       1  0 16:27 ?        00:00:22 VLLM::APIServer_0
root        3938       1  0 16:27 ?        00:00:21 VLLM::APIServer_1
root        3939       1  0 16:27 ?        00:00:21 VLLM::APIServer_2
root        3940       1  0 16:27 ?        00:00:22 VLLM::APIServer_3
root        3941       1  0 16:27 ?        00:00:18 VLLM::APIServer_4
root        3942       1  0 16:27 ?        00:00:31 VLLM::APIServer_5
root        3943       1  0 16:27 ?        00:00:26 VLLM::APIServer_6
root        3944       1  0 16:27 ?        00:00:16 VLLM::APIServer_7
root        9517    3929 28 16:27 ?        00:28:57 VLLM::Worker_DP0
root        9527    3930 29 16:27 ?        00:29:46 VLLM::Worker_DP1
root        9525    3931 29 16:27 ?        00:29:41 VLLM::Worker_DP2
root        9526    3932 28 16:27 ?        00:29:29 VLLM::Worker_DP3
root        9529    3933 28 16:27 ?        00:28:47 VLLM::Worker_DP4
root        9528    3934 28 16:27 ?        00:28:52 VLLM::Worker_DP5
root        9518    3935 28 16:27 ?        00:29:20 VLLM::Worker_DP6
root        9519    3936 28 16:27 ?        00:28:58 VLLM::Worker_DP7
```

### dep8-0

```bash
root           1       0  0 16:09 ?        00:00:39 /usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --enable-expert-parallel --data-parallel-size 8 --no-enable-prefix-caching --enable-auto-tool-choice --tool-call-parser deepseek_v3 --chat-template examples/tool_chat_template_deepseekv3.jinja --reasoning-parser deepseek_v3
root        3953       1  0 16:27 ?        00:00:00 /usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(40)
root        3954       1  0 16:27 ?        00:00:26 VLLM::DPCoordinator
root        4154       1  1 16:27 ?        00:01:22 VLLM::EngineCore_DP0
root        4155       1  1 16:27 ?        00:01:18 VLLM::EngineCore_DP1
root        4156       1  1 16:27 ?        00:01:09 VLLM::EngineCore_DP2
root        4157       1  1 16:27 ?        00:01:17 VLLM::EngineCore_DP3
root        4158       1  1 16:27 ?        00:01:22 VLLM::EngineCore_DP4
root        4159       1  1 16:27 ?        00:01:18 VLLM::EngineCore_DP5
root        4160       1  1 16:27 ?        00:01:08 VLLM::EngineCore_DP6
root        4161       1  1 16:27 ?        00:01:19 VLLM::EngineCore_DP7
root        4162       1  0 16:27 ?        00:00:21 VLLM::APIServer_0
root        4163       1  0 16:27 ?        00:00:20 VLLM::APIServer_1
root        4164       1  0 16:27 ?        00:00:22 VLLM::APIServer_2
root        4165       1  0 16:27 ?        00:00:18 VLLM::APIServer_3
root        4166       1  0 16:27 ?        00:00:26 VLLM::APIServer_4
root        4167       1  0 16:27 ?        00:00:20 VLLM::APIServer_5
root        4168       1  0 16:27 ?        00:00:25 VLLM::APIServer_6
root        4169       1  0 16:27 ?        00:00:18 VLLM::APIServer_7
root        9730    4154  9 16:27 ?        00:09:53 VLLM::Worker_DP0_EP0
root        9743    4155  9 16:27 ?        00:09:47 VLLM::Worker_DP1_EP1
root        9744    4156  9 16:27 ?        00:09:42 VLLM::Worker_DP2_EP2
root        9745    4157  9 16:27 ?        00:09:43 VLLM::Worker_DP3_EP3
root        9746    4158  9 16:27 ?        00:09:43 VLLM::Worker_DP4_EP4
root        9738    4159  9 16:27 ?        00:09:40 VLLM::Worker_DP5_EP5
root        9736    4160  9 16:27 ?        00:09:41 VLLM::Worker_DP6_EP6
root        9735    4161  9 16:27 ?        00:09:44 VLLM::Worker_DP7_EP7
```
