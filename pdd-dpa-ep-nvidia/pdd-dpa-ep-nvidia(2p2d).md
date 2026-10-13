<a id="top"></a>
# 2P2D：Prefill 与 Decode 各 2 个节点（DP16 + EP16）——B300x8 上的 PD 分离实测

DeepSeek-V3（FP8）在 B300x8 集群 (Rail-Optimized) 上做 Prefill/Decode 分离：**2 个 Prefill 节点 + 2 个 Decode 节点**。每个池是一个跨 2 个节点、16 张 GPU 的 DEP 组（DP16 + EP16）；KV cache 用 NIXL 从 Prefill 传到 Decode；前面由一个 CPU Pod 上的 vllm-router 接入。

**建议先读 1P1D 篇**（原理、部署与配置、观测方法、单池实测）。本文是它的多节点扩展：部署用的 YAML 和脚本与 1P1D **完全相同**（`pdd-dep.yaml`、`pdd-launch.sh`、`pdd-router.py`、`pdd.sh`，一个字都没改），只是把两个池的节点数从 1 改成 2。1P1D 篇已经讲过的内容（DEP 与 dummy batch、wave、NIXL 与侧信道、部署参数、观测方法、预热规则、单池的实测）本文不再重复，只引用它的章节（写作「1P1D 篇 x.x」）；本文只讲多节点带来的差别，并在第 8 章汇总与 1P1D 的对比。

**本文回答四个问题**

| 问题 | 在哪一章 |
|---|---|
| P 和 D 各 2 个节点时，有哪些进程、怎么配合？ | 第 3 章 |
| EP 的集合通信同时走 NVLink 和 RDMA，各走多少、怎么用各条 rail？ | 第 5 章 |
| KV Cache 传输在跨节点、跨轨的情况下表现如何？ | 第 6 章 |
| EP 集合通信的 RoCE 流量会不会影响 NIXL/UCX 的 KV 传输？ | 第 7 章 |

**约定**
- **标注**：**观察**＝在本集群上直接看到的现象或读到的数据；**推断**＝由机制得出，没有验证；**源码核对**＝读了镜像内 vLLM 0.31.0 源码。
- **数据**：所有数据都是 2026-10-13 在运行中的 2P2D 上采集的，并且都是**预热之后**的数据（第 4.3 节）。
- **rail**：网卡序号。这里假设各节点上同序号的网卡（`mlx5_{2i}`，i = 0..7，下文称 NIC i）连在同一条 rail 上（NCCL 的跨节点流量全部是「同序号网卡对同序号网卡」，与此一致，但没有读交换机配置去证实）。**跨轨**＝两端用的网卡序号不同。
- **global rank**：池内 DP rank 0..15，节点 0 是 0..7，节点 1 是 8..15。P5 表示 Prefill 池的 global rank 5，D13 表示 Decode 池的 global rank 13。

<a id="toc"></a>
## 目录

- [1. 摘要](#s1)
- [2. 部署与环境](#s2)
  - [2.1 部署命令，以及为什么不用改文件](#s2-1)
  - [2.2 运行环境记录](#s2-2)
  - [2.3 启动时序](#s2-3)
- [3. 进程与关系](#s3)
  - [3.1 每个 Pod 里的进程](#s3-1)
  - [3.2 DP rank / EP rank 的分布](#s3-2)
  - [3.3 控制面：节点之间怎么连](#s3-3)
  - [3.4 Router 与 P→D 的配对](#s3-4)
  - [3.5 空闲状态](#s3-5)
- [4. 测量方法与预热](#s4)
  - [4.1 测什么、怎么测](#s4-1)
  - [4.2 如何把集合通信和 KV 流量分开](#s4-2)
  - [4.3 测试前必须预热](#s4-3)
- [5. EP 集合通信：NVLink 与 RDMA](#s5)
  - [5.1 EP16 的通信模式](#s5-1)
  - [5.2 一个 32768 token 请求的集合通信量](#s5-2)
  - [5.3 NVLink 与 RoCE 的分工](#s5-3)
  - [5.4 跨节点的集合通信按 rail 对齐](#s5-4)
  - [5.5 Prefill 用 8 条 rail，Decode 只用 3 条](#s5-5)
  - [5.6 流量在时间上的分布：RoCE 没有跑满](#s5-6)
  - [5.7 并发时的集合通信量](#s5-7)
- [6. KV 传输：跨节点、跨轨](#s6)
  - [6.1 大 KV（131072 token）的 4 次传输](#s6-1)
  - [6.2 是否跨轨、是否就近、READ 还是 WRITE](#s6-2)
  - [6.3 性能](#s6-3)
  - [6.4 Router 重启后：同轨与跨轨的对照](#s6-4)
- [7. 集合通信会影响 KV 传输吗](#s7)
- [8. 与 1P1D 的对比](#s8)
- [9. 推断核对与未验证](#s9)
- [10. 附录：原始输出与辅助脚本](#s10)
  - A.1～A.4：四个 Pod 的 `ps -ef` 原始输出
  - [A.5 辅助脚本](#a5)

<a id="s1"></a>
## 1. 摘要

[↑ 返回顶部](#top)

**部署与进程**

| 问题 | 结论 |
|---|---|
| 要不要改 YAML/脚本 | **不用**：`PREFILL_NODES=2 DECODE_NODES=2 ./pdd.sh up`（也是默认值）即可，`pdd.sh metrics`、Router 都按节点数自动适配，没有发现 bug |
| 进程 | 每个 Pod 的进程数和 1P1D 一样（27 个）；**`DPCoordinator` 只在节点 0**；节点 1 的引擎是 `EngineCore_DP8..15`、Worker 是 `Worker_DP8_EP8..DP15_EP15` |
| 池内与池间的连接 | 池内 2 个节点之间有 398 条 TCP 连接（控制面，eth0）；**P 池与 D 池之间没有任何 TCP 连接**，只有 NIXL 点对点 |
| P→D 的配对 | Router 的 16 个 Prefill rank → 16 个 Decode rank 是固定的一一对应（不是 16×16）；本文主要数据所在的那次 Router 启动里**没有一对**两端网卡序号相同，**Router 重启后配对会重排，出现过 3 对同轨**（3.4、6.4） |

**EP 集合通信**

| 问题 | 结论 |
|---|---|
| 走什么 | **同时走 NVLink 和 RoCE**：一个 32768 token 的请求，每个 Prefill 节点内 NVLink 约 293～331 GiB、跨节点 RoCE 约 42～44 GB（约占 NVLink 字节量的 12%） |
| 怎么用 rail | **rail 对齐**：节点 A 的 NIC i 只和节点 B 的 NIC i 通信；Prefill 池 8 条 rail 全用，**Decode 池只用 3 条（NIC 0、1、2）** |
| 带宽 | 突发，单块网卡 2～4 GB/s，不到线速的 10%，**RoCE 没有跑满** |

**KV 传输**

| 问题 | 结论 |
|---|---|
| 方式 | **RDMA READ，Decode 发起**，每次传输两侧各只用一块网卡，用的是各自 rank 最近的那块（rank r ↔ `mlx5_{2(r%8)}`） |
| 是否跨轨 | **全部跨轨**；跨轨传输仍能达到约 50 GB/s 的线速峰值 |
| 性能 | 9.21 GB（131072 token）约 191～223 ms，约 41～48 GB/s |
| 集合通信会不会影响它 | **线上速度不受影响**：24 个并发请求下 KV 突发的中位数仍是 60 ms、约 39 GB/s，与单请求相同；NIXL 记录的传输耗时从约 54 ms 升到 72 ms（约 +30%），不是网卡被占满；用 1P1D（RoCE 上没有集合通信）做对照，并发下同样变长 16%～34%，所以**主要不是集合通信造成的**（第 7 章） |

**与 1P1D 对比，以及测量要求**

| 问题 | 结论 |
|---|---|
| TTFT | 32768 token：2P2D **3.55～3.71 s**，1P1D 3.5 s，基本相同；131072 token 的 prefill 34.7～35.4 s，与 1P1D 的 34.1～35.2 s 相同 |
| TPOT | 2P2D 21.0 ms，1P1D 16.1 ms（Decode 集合通信要跨节点，慢约 5 ms） |
| **测试前必须预热** | 先让全部 16 个 Prefill rank 和 16 个 Decode rank 都处理过长请求，再开始计时（第 4.3 节） |

<a id="s2"></a>
## 2. 部署与环境

[↑ 返回顶部](#top)

<a id="s2-1"></a>
### 2.1 部署命令，以及为什么不用改文件

```bash
cd /root/work/data/distributed-inference/pdd-dpa-ep-nvidia/
PREFILL_NODES=2 DECODE_NODES=2 ./pdd.sh up     # 不写这两个变量时默认就是 2、2
```

`pdd.sh up` 检查 `PREFILL_NODES + DECODE_NODES = 4` 不超过集群的 GPU 节点数（4）。四个 Pod 各占一个 B300×8 节点，`pdd-router-0` 在 CPU 节点。各文件和参数的含义见 1P1D 篇第 3 章。

**为什么不用改任何文件**（已用实际部署验证）：
- `pdd-launch.sh` 按 Pod 名序号算 `DP_SIZE = NODES × 8 = 16` 和 `START_RANK = 序号 × 8`（节点 1 是 8）；非 0 号节点先等 Head（`-0`）的 RPC 端口 13345 可连再启动。
- `pdd-router.py` 读两个 StatefulSet 的 `replicas`，等所有 P/D Pod 就绪，然后用 2 个 `--prefill` 和 2 个 `--decode` 启动 `vllm-router`；日志里 Router 发现了 **32 个 worker**（每个 Pod 8 个 DP rank）。
- `pdd.sh metrics` 遍历每个池的所有节点（输出里是 `16 engines`）；`test`、`bench` 照常可用，`./pdd.sh test` 启动完成后一次通过（HTTP 200）。
- 部署后四个文件的修改时间没有变（都早于本次部署）。

<a id="s2-2"></a>
### 2.2 运行环境记录

采集时间 2026-10-13 03:27 UTC（Pod 已运行 112 分钟，本文的实验都在这套部署上完成）。

**`kubectl get pods -o wide`**

```
NAME            READY   STATUS    RESTARTS   AGE    IP             NODE                        NOMINATED NODE   READINESS GATES
pdd-decode-0    1/1     Running   0          112m   10.154.2.116   rs-b300x8-gpu-pool-3o1v8b   <none>           <none>
pdd-decode-1    1/1     Running   0          112m   10.154.1.150   rs-b300x8-gpu-pool-3o1v8g   <none>           <none>
pdd-prefill-0   1/1     Running   0          112m   10.154.1.124   rs-b300x8-gpu-pool-3o1v8w   <none>           <none>
pdd-prefill-1   1/1     Running   0          112m   10.154.2.207   rs-b300x8-gpu-pool-3o1v8e   <none>           <none>
pdd-router-0    1/1     Running   0          112m   10.154.3.203   rs-power-cpu-pool-3jmm1a    <none>           <none>
```

文中出现的 Pod IP（Router 的 `--prefill`/`--decode` 地址、`ss` 统计里的对端）就是上面的 IP；所有 Pod 都没有重启。

**`kubectl get nodes -o wide`**（省略了外网 IP 列）

```
NAME                        STATUS   ROLES    AGE     VERSION   INTERNAL-IP   OS-IMAGE                       KERNEL-VERSION                CONTAINER-RUNTIME
rs-b300x8-gpu-pool-3o1v8b   Ready    <none>   35h     v1.37.0   10.132.0.20   Debian GNU/Linux 13 (trixie)   6.12.96+deb13-amd64 (amd64)   containerd://2.3.3
rs-b300x8-gpu-pool-3o1v8e   Ready    <none>   35h     v1.37.0   10.132.0.19   Debian GNU/Linux 13 (trixie)   6.12.96+deb13-amd64 (amd64)   containerd://2.3.3
rs-b300x8-gpu-pool-3o1v8g   Ready    <none>   35h     v1.37.0   10.132.0.23   Debian GNU/Linux 13 (trixie)   6.12.96+deb13-amd64 (amd64)   containerd://2.3.3
rs-b300x8-gpu-pool-3o1v8w   Ready    <none>   35h     v1.37.0   10.132.0.22   Debian GNU/Linux 13 (trixie)   6.12.96+deb13-amd64 (amd64)   containerd://2.3.3
rs-power-cpu-pool-3jmm1a    Ready    <none>   5h31m   v1.37.0   10.132.0.17   Debian GNU/Linux 13 (trixie)   6.12.96+deb13-amd64 (amd64)   containerd://2.3.3
rs-power-cpu-pool-3jmm1e    Ready    <none>   5h31m   v1.37.0   10.132.0.13   Debian GNU/Linux 13 (trixie)   6.12.96+deb13-amd64 (amd64)   containerd://2.3.3
```

**镜像与软件版本**：5 个 Pod（4 个 GPU Pod + Router）用的是**同一个镜像** `vllm/vllm-openai:latest`，digest 是 `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`，与 1P1D 篇 3.7 记录的相同（所以 `latest` 在这次部署里没有造成版本不一致）。软件版本（vLLM 0.31.0、NIXL 1.4.1、NCCL 2.30.7、PyTorch 2.13.0+cu130）、GPU 与网卡硬件、驱动、节点系统与 1P1D 篇 3.7、5.5 相同。

<a id="s2-3"></a>
### 2.3 启动时序

观察（日志）：

| 事件 | 时间（UTC，2026-10-13） |
|---|---|
| 四个 Pod 创建、启动脚本开始 | 01:35 |
| 节点 1 等 Head：`waiting for pdd-prefill-0.pdd-prefill-headless ...`、`waiting for DP master <ip>:13345 ...` | 启动初期，随后继续 |
| 所有引擎开始初始化 | 01:36:31 |
| 每个 rank 的 `Expert parallelism is enabled ... [EP Rank k/16] ... Local/global number of experts: 16/256` | 01:37:26 |
| Decode 两个节点 `init engine` 完成 | 约 531 s，01:47:06 |
| Prefill 两个节点 `init engine` 完成 | 约 1144 s，01:57:19 |
| Router 发现 2+2 个 Pod 全部就绪并启动 | 01:57:56 |

- 每个 rank 16 个专家（256 / 16）；其中编译约 43～45 s，两个池相同。
- **Prefill 的初始化约是 Decode 的 2 倍**（1144 s 对 531 s），原因没有查清；Decode 用 `FULL_DECODE_ONLY` 的 CUDA graph，Prefill 用默认的 `FULL_AND_PIECEWISE`，这是可能相关的差别（**推断**，没有验证）。
- 全部 4 个 GPU Pod 的日志里 `ERROR`/`Traceback`/`NIXL_ERR` 为 0 行；每个 Pod 有 8 个 `Backend UCX was instantiated`。

<a id="s3"></a>
## 3. 进程与关系

[↑ 返回顶部](#top)

<a id="s3-1"></a>
### 3.1 每个 Pod 里的进程

每个 GPU Pod 的进程构成与 1P1D 篇 5.1 相同（`vllm serve`、`resource_tracker`、8 个 `APIServer_0..7`、8 个 `EngineCore_DPn`、8 个 `Worker_DPn_EPn`）；多节点的差别是 **`DPCoordinator` 只在节点 0**，节点 1 的 rank 序号接着排（原始输出见附录）：

| Pod | `DPCoordinator` | `EngineCore` | `Worker` |
|---|---|---|---|
| `pdd-prefill-0` / `pdd-decode-0`（节点 0） | 1 个 | `DP0..DP7` | `Worker_DP0_EP0 .. DP7_EP7` |
| `pdd-prefill-1` / `pdd-decode-1`（节点 1） | **没有** | `DP8..DP15` | `Worker_DP8_EP8 .. DP15_EP15` |

- 节点 0 有 27 个服务进程，节点 1 只有 26 个。
- 每个池内是一个 **DP16 + EP16 的大组**，协调者只有一个（节点 0 的 `DPCoordinator`，wave 的协调见 1P1D 篇 2.3）；P 池和 D 池各有一个，互不相连。
- 节点 1 的 8 个 EngineCore 通过 `--data-parallel-address`（节点 0 的 Pod IP）和 RPC 端口 13345 接入节点 0 的协调者。

<a id="s3-2"></a>
### 3.2 DP rank / EP rank 的分布

| | 节点 0 | 节点 1 |
|---|---|---|
| DP rank = EP rank | 0..7 | 8..15 |
| `--data-parallel-start-rank` | 0 | 8 |
| 每 rank 的专家数 | 16 个（共 256） | 16 个 |
| 每 rank 的 GPU | 本节点的 GPU 0..7 | 本节点的 GPU 0..7 |

日志（每个节点）：`rank k in world size 16 is assigned as DP rank k, PP rank 0, PCP rank 0, TP rank 0, ETP rank 0, EP rank k`，`[EP Rank k/16] ... Local/global number of experts: 16/256`。

<a id="s3-3"></a>
### 3.3 控制面：节点之间怎么连

在每个 GPU Pod 里用 `ss -tn` 统计已建立的 TCP 连接，按对端分类：

| 在哪个 Pod 里 | 到另一个同池 Pod | 到另一个池的 Pod |
|---|---|---|
| `pdd-prefill-0` | 398（382 + 16） | 0 |
| `pdd-prefill-1` | 398 | 0 |
| `pdd-decode-0` | 398 | 0 |
| `pdd-decode-1` | 398 | 0 |

- 池内两个节点之间有 398 条 TCP 连接（走 eth0，控制面：DP 协调的 RPC/ZMQ、Gloo、TCPStore 等；具体每条连接的用途没有逐一核对）。这个数在 4 个 Pod 里对称，与哪个节点是 Head 无关。
- **P 池和 D 池之间一条 TCP 连接都没有**：P 与 D 之间只有 NIXL 点对点（握手之后数据走 RDMA）。
- **Router 到各个 Pod 的连接是短连接，空闲时看不到**：Router 对 4 个 GPU Pod 一视同仁，与谁是 Head 无关。空闲时在 Router Pod 和 4 个 GPU Pod 里都数不到 Router 的连接；发 8 个并发请求时，在 Router Pod 里看到的同时存在的连接数最多是 `prefill-0` 4 条、`prefill-1` 6 条、`decode-0` 2 条、`decode-1` 7 条。

<a id="s3-4"></a>
### 3.4 Router 与 P→D 的配对

Router 启动命令：

```
vllm-router --policy round_robin --vllm-pd-disaggregation --host 0.0.0.0 --port 30000 --intra-node-data-parallel-size 8
  --prefill http://10.154.1.124:8001 --prefill http://10.154.2.207:8001
  --decode http://10.154.2.116:8002 --decode http://10.154.1.150:8002
```

Router 以 DP rank 为单位选后端（日志里形如 `prefill=http://<ip>:8001@r`，r 是节点内 rank），共 32 个 worker。对 Router 日志里截至目前的全部 91 个请求统计 **global rank 的配对**：是固定的一一对应（16 个 Prefill rank → 16 个互不相同的 Decode rank，**不是 16×16 的组合**），每个 Prefill rank 出现 3～8 次，所有配对都落在这张表里：

| P rank | D rank | P NIC 序号 | D NIC 序号 | 同序号？ |
|---|---|---|---|---|
| P0 | D5 | 0 | 5 | 否 |
| P1 | D15 | 1 | 7 | 否 |
| P2 | D0 | 2 | 0 | 否 |
| P3 | D4 | 3 | 4 | 否 |
| P4 | D7 | 4 | 7 | 否 |
| P5 | D11 | 5 | 3 | 否 |
| P6 | D3 | 6 | 3 | 否 |
| P7 | D10 | 7 | 2 | 否 |
| P8 | D14 | 0 | 6 | 否 |
| P9 | D12 | 1 | 4 | 否 |
| P10 | D6 | 2 | 6 | 否 |
| P11 | D1 | 3 | 1 | 否 |
| P12 | D8 | 4 | 0 | 否 |
| P13 | D9 | 5 | 1 | 否 |
| P14 | D2 | 6 | 2 | 否 |
| P15 | D13 | 7 | 5 | 否 |

（NIC 序号 = global rank % 8，对应 `mlx5_{2×序号}`。）**在这次 Router 启动里，16 对里没有一对两端网卡序号相同，所以本文主要数据里的 KV 传输全部是跨轨**。但这不是 Router 策略的必然结果：配对随 Router 的启动而变（下面的重启实验）。

**重启 Router 之后配对变了**（2026-10-13 03:54 UTC 重启 `pdd-router-0`，其余 Pod 不动，随后发 48 个 32768 token 的请求，并发度 8）：

| P rank | P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11 | P12 | P13 | P14 | P15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D rank（重启后） | D3 | D4 | D9 | D1 | D5 | D0 | D2 | D6 | D10 | D14 | D8 | **D11** | **D12** | D15 | D13 | **D7** |
| D rank（重启前，上表） | D5 | D15 | D0 | D4 | D7 | D11 | D3 | D10 | D14 | D12 | D6 | D1 | D8 | D9 | D2 | D13 |

- 重启后仍是固定的一一对应，48 个请求里每一对恰好出现 3 次；但配对整体重排了，其中 **3 对是同轨的**（加粗，网卡序号 P→D 为 3→3、4→4、7→7）。所以「Router 不会产生同轨配对」只对重启前的那次启动成立。
- **配对的规律**：Router 日志里选择的顺序周期为 16，Decode rank 严格按 D1、D2、D3、…、D15、D0 的顺序轮转，Prefill rank 按另一个固定的顺序轮转（P3、P6、P0、P1、P4、P7、P15、P10、P2、P8、P11、P12、P14、P9、P13、P5）。也就是说，P 和 D 各有一个 `round_robin` 轮询，每个请求各前进一步，两个列表长度都是 16、顺序在一次启动内固定，所以组成固定的一一对应；重启之后 Prefill 列表的顺序变了（**推断**：列表顺序来自 Router 启动时发现 worker 的顺序，没有读 `vllm-router` 的源码）。
- 注意：`vllm bench` 每次运行会先发一个「初始单请求测试」，它会让轮询前进一步，但在 Router 日志里不一定计入，用日志里的计数去估算下一个请求落在哪个配对会差一步。

<a id="s3-5"></a>
### 3.5 空闲状态

- 空闲时四个 Pod 的 8 块 RoCE 网卡上，8 秒内收发字节数全是 0；8 张 GPU 的利用率全是 0%。
- **每个 Worker 仍然占满一个 CPU 核**：全部 32 个 Worker（每池 16 个）3 秒窗口内 CPU 占用都是 100%～101%，其余进程（`EngineCore`、`APIServer`、`DPCoordinator`）都低于 1%。与 1P1D 篇 5.2 是同样的现象（那里查明是 Worker 里一个忙轮询的原生线程，很可能是 NIXL 的 UCX progress 线程，不是 dummy batch）；2P2D 上没有重新逐线程核对。

<a id="s4"></a>
## 4. 测量方法与预热

[↑ 返回顶部](#top)

<a id="s4-1"></a>
### 4.1 测什么、怎么测

测量方法与 1P1D 篇第 4 章和附录 A.4 相同：TTFT / TPOT 用 `./pdd.sh bench`（在 `pdd-router-0` 里运行，经 Router 发请求）；NIXL 的字节数、耗时、descriptor 数看 P、D 的 `/metrics`（只有 Decode 一侧有数值）；NVLink 用 `nvidia-smi nvlink -gt d` 的前后差；RoCE 字节数用 ioctl 读 `tx/rx_vport_rdma_unicast_bytes`（每 20 ms 一次）；RoCE 请求计数用 `rdma statistic`。2P2D 的差别和补充：

- **global rank**：池里有 16 个 rank，`/metrics` 的 `engine` 标签就是池内 global rank；Router 日志（`Selected prefill=…@r … decode=…@d`）里的 r、d 是节点内 rank，要按 Pod 换算成 global rank（第 3.4 节的配对表就是这样统计的）。
- **要同时读 4 个 GPU Pod**：NVLink 的读取不是原子的，窗口比请求本身略长；NVLink 的空闲流量在 1P1D 上测过是 0，2P2D 上没有单独测。
- **所有测试流量都经 `pdd-router-0` 发出**，不在 Prefill/Decode Pod 里发请求；读 NVLink、网卡计数器要进 GPU Pod，保持只读、轻量。
- **采样对被测系统没有影响**：每 20 ms 读一次 4 个 GPU Pod 的网卡计数，对 TTFT 没有影响（不采样 5 次 3.54～3.64 s，采样 3 次 3.58 s）。

<a id="s4-2"></a>
### 4.2 如何把集合通信和 KV 流量分开

集合通信和 KV 传输共用同样的 RoCE 网卡，需要把它们分开看：

- **RDMA 操作类型不同**：集合通信（NCCL）用 RDMA WRITE，KV 传输（NIXL）用 RDMA READ。所以 `rx_write_requests` 主要反映集合通信，`rx_read_requests` 反映 KV。
- **KV 的字节有固定的位置**：一次 KV 传输会在源节点的一块网卡上多发、在目的节点的一块网卡上多收（量等于 KV 大小，约 2.35 GB / 32768 token），这两块网卡由 Router 日志里的配对和网卡-GPU 对应关系决定。本文表格里这两处都单独标出，计算集合通信量时扣掉。
- **时间上也分开**：集合通信发生在 prefill 和 decode 期间，KV 传输在 prefill 结束之后，只持续约 0.2 s，是单块网卡约 50 GB/s 的突发。

<a id="s4-3"></a>
### 4.3 测试前必须预热

规则与 1P1D 篇 3.3 相同：**做任何性能测试之前，先让每个 DP rank 都处理过一遍长请求**。每个 rank 第一次处理长 prefill 时有一次性的额外开销（在这套 2P2D 上观察到约 8 s），第一次的数据不纳入本文。2P2D 上的做法：

- 经 Router 发一批长请求，覆盖全部 16 个 Prefill rank 和 16 个 Decode rank。P→D 的配对是固定的（第 3.4 节），覆盖了 16 个 Prefill rank，也就覆盖了 16 个 Decode rank。
- 推荐命令（32768 token，并发度 8）：`BENCH_NUM_PROMPTS=24 BENCH_REQUEST_RATE=inf ./pdd.sh bench --max-concurrency 8`。实测一次 24 个请求就覆盖了全部 16 对（每对 1～2 次）。
- **是否覆盖全部 rank，要看 Router 日志确认**：`kubectl logs pdd-router-0 | grep "Selected prefill"` 里两个 Prefill 节点的 `@0`..`@7` 都出现过。选 rank 的规律不保证一批请求一定覆盖全部，不够就再发一批。
- 预热请求的输入要不短于正式测试（≥ 32768 token）；几个很短的请求（例如 `./pdd.sh test` 的 10 token）不算预热。
- 预热后同样的请求连续发几次，TTFT 应当稳定（32768 token 的单请求约 3.6 s）。

预热之前采集的耗时数据（TTFT、prefill 耗时、传输耗时）都不要用；NVLink、RoCE 的字节数不受影响。

<a id="s5"></a>
## 5. EP 集合通信：NVLink 与 RDMA

[↑ 返回顶部](#top)

本章只讨论 **EP（专家并行）的集合通信**：它在节点内走什么、跨节点走什么、各走多少、怎么用各条 rail。KV 传输在第 6 章，P 池与 D 池之间的控制面连接在第 3.3 节，KV 流量与集合通信的相互影响在第 7 章。测量方法见第 4 章。

<a id="s5-1"></a>
### 5.1 EP16 的通信模式

每个池是 DP16 + EP16，每个 MoE 层（共 58 层）做一次 `all_gatherv`（每个 rank 把自己的 token 发给池里全部 16 个 rank）和一次 `reduce_scatterv`（汇总各 rank 对「我的 token」的专家输出）。这两个函数名是 vLLM 源码里的函数名（`v` 表示变长，各 rank 的 token 数可以不同）。对每个 rank 来说，另外 15 个 rank 里有 7 个在同一节点、8 个在另一个节点：

| 对端 | 数量 | 走什么 |
|---|---|---|
| 同节点的 rank | 7 | **NVLink**（`NV18` 全互联） |
| 另一个节点的 rank | 8 | **RoCE，RDMA WRITE**，用本 rank 最近的那块网卡（GPU i ↔ `mlx5_{2i}`） |

没有请求的 rank 也要陪跑 dummy batch（原因见 1P1D 篇 2.3），所以只要池里有一个请求，32 张 GPU 的集合通信流量都不为 0（5.2）。

<a id="s5-2"></a>
### 5.2 一个 32768 token 请求的集合通信量

在预热后的空闲 2P2D 上发 1 个 32768 / 128 token 的请求（`./pdd.sh bench`，经 `pdd-router-0`）。请求落在 `pdd-prefill-1` 的 rank 7（P15）和 `pdd-decode-1` 的 rank 5（D13）；TTFT 3.58 s，TPOT 21.0 ms，Prefill 耗时 3.1 s。

**NVLink**（每个 Pod 内 8 张 GPU 的 Tx 合计，GiB）

| Pod | 合计 | 各 GPU |
|---|---|---|
| `pdd-prefill-0` | 330.6 | 39.9, 41.4, 41.4, 41.5, 41.6, 41.6, 41.6, 41.6 |
| `pdd-prefill-1` | 293.1 | 29.4, 41.4, 41.4, 41.5, 41.6, 41.6, 41.6, 14.6 |
| `pdd-decode-0` | 36.5 | 5.2, 3.4, 3.3, 3.8, 5.2, 5.2, 5.2, 5.2 |
| `pdd-decode-1` | 36.5 | 5.2, 3.4, 3.3, 3.8, 5.2, 5.2, 5.2, 5.2 |

- **请求只落在一个 rank 上，但 4 个 Pod、32 张 GPU 都有 NVLink 流量**：其余 rank 陪跑，EP16 的集合通信让整个池都参与。
- Prefill 的流量约是 Decode 的 8～9 倍：prefill 一步处理上万个 token，decode 一步只有几个 token。
- `pdd-prefill-1` 的 GPU 0、7 流量偏低（29.4、14.6 GiB），原因没有查清。

**RoCE**（该请求期间每块网卡的收发字节数，MB；NIC 序号 0..7 对应 `mlx5_0,2,…,14`）

| Pod | 发送 tx | 接收 rx |
|---|---|---|
| `pdd-prefill-0` | 5808, 5879, 5816, 5649, 5654, 5649, 5644, 3725（合计 43.8 GB） | 5803, 5867, 5824, 5649, 5668, 5660, 1928, 5637 |
| `pdd-prefill-1` | 5803, 5867, 5826, 5649, 5676, 5667, 1928, **7990**（合计 44.4 GB） | 5808, 5877, 5816, 5648, 5653, 5649, 5644, 3737 |
| `pdd-decode-0` | 2289, 2243, 1707, 0, 0, 0, 0, 0（合计 6.2 GB） | 2290, 2244, 1707, 0, 0, 0, 0, 0 |
| `pdd-decode-1` | 2290, 2244, 1707, 0, 0, 0, 0, 0 | 2289, 2243, 1707, 0, 0, **2352**, 0, 0 |

- 加粗的两处是这个请求的 KV 传输（第 4.2 节）：`pdd-prefill-1` 的 NIC 7 发送 7990 MB，比同 Pod 其他 NIC（约 5650）多约 2.35 GB；`pdd-decode-1` 的 NIC 5 接收 2352 MB，而该 Pod 的 NIC 0～2 只有集合通信的 1707～2289 MB、其余为 0。**扣掉 KV 后**，每个 Prefill 节点的集合通信发送约 42～44 GB，每个 Decode 节点约 6.2 GB。
- `rx_write_requests` 的增量：Prefill 两个节点完全相同，NIC 0 约 13.6 万、NIC 1 和 2 约 7.5 万、NIC 3、4、5 约 2.6 万、NIC 6 约 2.0 万、NIC 7 约 2.3 万；Decode 两个节点只有 NIC 0、1、2（100 万、50 万、50 万）。

<a id="s5-3"></a>
### 5.3 NVLink 与 RoCE 的分工

| | NVLink（节点内） | RoCE（跨节点，扣掉 KV） | RoCE 占 NVLink 字节量 |
|---|---|---|---|
| 每个 Prefill 节点 | 293～331 GiB（约 315～355 GB） | 约 42～44 GB | 约 12% |
| 每个 Decode 节点 | 36.5 GiB（约 39 GB） | 约 6.2 GB | 约 16% |

- **同一个 MoE 层的通信同时用到 NVLink 和 RoCE**，NVLink 承担了大部分字节。
- RoCE 的字节量远小于「每个 rank 把数据发给 8 个远端 rank」的朴素估计，与「跨节点部分每份数据只过一次网络、之后在节点内用 NVLink 转发」的层次化算法一致（**推断**，没有读 NCCL 的算法选择）。

<a id="s5-4"></a>
### 5.4 跨节点的集合通信按 rail 对齐

对比两个 Prefill 节点同序号网卡的收发字节数（5.2 的表）：

| NIC 序号 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `prefill-0` 发送 | 5808 | 5879 | 5816 | 5649 | 5654 | 5649 | 5644 | 3725 |
| `prefill-1` 接收 | 5808 | 5877 | 5816 | 5648 | 5653 | 5649 | 5644 | 3737 |
| `prefill-1` 发送（扣掉 NIC 7 的 KV 约 2.35 GB） | 5803 | 5867 | 5826 | 5649 | 5676 | 5667 | 1928 | 约 5640 |
| `prefill-0` 接收 | 5803 | 5867 | 5824 | 5649 | 5668 | 5660 | 1928 | 5637 |

- 节点 0 的 NIC i 发送的字节数，**与节点 1 的 NIC i 接收的字节数几乎完全相同**（差别在十几 MB 以内），反方向同理；Decode 两个节点之间同样（`decode-0` 的 tx 与 `decode-1` 的 rx 逐 NIC 一致）。也就是说**跨节点的集合通信是 rail 对齐的：NIC i ↔ NIC i，没有跨轨**。
- 每个 GPU 用自己最近的那块网卡（GPU i ↔ `mlx5_{2i}`，同一 PCIe 桥）与对端节点同序号的 GPU 通信；集合通信的流量因此不需要经过 spine 层的跨轨路径（**推断**，没有读交换机拓扑）。
- 不对称：这个请求里 `prefill-1` 的 NIC 6 发送只有 1.9 GB，`prefill-0` 的 NIC 7 发送只有 3.7 GB（其他约 5.6 GB），各个方向的字节数不完全一样，原因没有查清（具体通道数、环/树结构没有读 NCCL 日志，没有核对）。

<a id="s5-5"></a>
### 5.5 Prefill 用 8 条 rail，Decode 只用 3 条

- Prefill 池：8 块网卡都有集合通信流量，每块约 5.6～5.9 GB（32768 token 一个请求）。
- Decode 池：**只有 NIC 0、1、2（`mlx5_0,2,4`）有集合通信流量**（发送 2.3、2.2、1.7 GB），NIC 3～7 完全没有（0 字节）。24 个请求的并发实验（5.7）里也是这样。
- 原因没有查清。**推断**：Decode 每步只有很少的 token，NCCL 对小消息只用较少的通道，因此只占用前几条 rail；Prefill 的消息大，用满 8 条。没有读 NCCL 的通道数。
- 对 KV 传输的含义：Decode 的 rank 序号落在 NIC 0～2（D0、D1、D2、D8、D9、D10）时，KV 会落在同时承载 Decode 集合通信的网卡上；这对 KV 传输速度有没有影响见第 7 章（没有看到影响）。

<a id="s5-6"></a>
### 5.6 流量在时间上的分布：RoCE 没有跑满

把 20 ms 的采样按 1 秒聚合，看这个单请求的发送速率（GB/s，括号内是单块网卡的最高值）：

| 阶段 | 每个 Prefill 节点（8 块网卡合计） | 每个 Decode 节点（8 块网卡合计） |
|---|---|---|
| prefill（约 3 s） | 最高 18.2（单块 ≤ 4.0）；其余几秒 10～11（≤ 2.8） | 0 |
| decode（128 个 token，约 2.7 s） | 0 | 约 2.0（单块 ≤ 0.8） |

- 集合通信是**突发**的，并且**单块网卡的速率只有 2～4 GB/s，不到 50 GB/s 线速的 10%**；Decode 阶段单块网卡最高 0.8 GB/s。
- 所以集合通信不是 RoCE 带宽受限的（**推断**：它受 16 个 rank 之间同步的次数和延迟约束，没有拆开验证）。
- 对比：KV 传输那一刻单块网卡能到约 50 GB/s。集合通信的流量和 KV 的突发在带宽上相差一个数量级，这是第 7 章结论的基础。

<a id="s5-7"></a>
### 5.7 并发时的集合通信量

24 个 32768 token 的请求，并发度 8（第 7 章的实验）：

| | 每个节点合计 | 折合每个请求 | 单请求（5.2）对照 |
|---|---|---|---|
| Prefill 节点 NVLink | 7094.9 / 7168.4 GiB | 295.6 / 298.7 GiB | 293～331 GiB |
| Prefill 节点 RoCE 发送（含 KV） | 1035.4 / 978.3 GB | 两个节点合计扣掉 24 次 KV（约 56 GB）后，每节点约 40.8 GB | 42～44 GB |
| Decode 节点 NVLink | 275.8 GiB（两个节点相同） | 11.5 GiB | 36.5 GiB |
| Decode 节点 RoCE 发送（NIC 0、1、2） | 17.3 + 17.0 + 12.9 = 47.2 GB | 1.97 GB | 6.2 GB |

- **Prefill 阶段每个请求的集合通信量基本不随并发变化**（NVLink 约 296～299 GiB，RoCE 约 41 GB / 节点），因为一个 prefill 的 token 数是固定的，每个 token 都要经过这些通信。
- **Decode 阶段每个请求分摊到的集合通信量随并发大幅下降**（NVLink 11.5 对 36.5 GiB，RoCE 1.97 对 6.2 GB，约 1/3）：并发的请求共享 decode 步，一步的集合通信由同一步里的所有请求分摊（**推断**，由数字得出，没有逐步验证）。
- 并发时 Decode 的集合通信仍只用 NIC 0、1、2，NIC 3～7 的发送为 0。
- 并发时 Prefill 节点 8 块网卡合计的发送速率峰值约 113～116 GB/s（含 KV 突发），KV 之外的集合通信仍是几 GB/s 的量级。

<a id="s6"></a>
## 6. KV 传输：跨节点、跨轨

[↑ 返回顶部](#top)

<a id="s6-1"></a>
### 6.1 大 KV（131072 token）的 4 次传输

预热后，依次发 4 个 131072 / 16 token 的请求（每个 KV 约 9.21 GB）。每个请求前后读 `/metrics` 和 `rdma statistic`，全程每 20 ms 采样 8 块网卡的收发字节数（第 4.1 节）。

| 请求 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Prefill rank → Decode rank（global） | P4 → D7 | P13 → D9 | P5 → D11 | P15 → D13 |
| 节点内位置 | `prefill-0` r4 → `decode-0` r7 | `prefill-1` r5 → `decode-1` r1 | `prefill-0` r5 → `decode-1` r3 | `prefill-1` r7 → `decode-1` r5 |
| Prefill 侧发送的网卡 | `mlx5_8` | `mlx5_10` | `mlx5_10` | `mlx5_14` |
| Decode 侧接收的网卡 | `mlx5_14` | `mlx5_2` | `mlx5_6` | `mlx5_10` |
| 网卡序号（P → D） | 4 → 7 | 5 → 1 | 5 → 3 | 7 → 5 |
| 其他网卡有无 >10 GB/s 的流量 | 无 | 无 | 无 | 无 |
| NIXL 传输字节数 | 9,210,691,584 | 同左 | 同左 | 同左 |
| NIXL descriptor 数 | 62,769 | 62,769 | 62,769 | 732 |
| NIXL 传输耗时（`xfer_time`） | 193.0 ms | 222.9 ms | 194.2 ms | 191.4 ms |
| 字节 ÷ `xfer_time` | 47.7 GB/s | 41.3 GB/s | 47.4 GB/s | 48.1 GB/s |
| 发起耗时（`post_time`） | 2.3 ms | 2.8 ms | 2.7 ms | 1.7 ms |
| Prefill 侧网卡上 >10 GB/s 的发送突发 | 9,353 MB / 0.201 s / 峰值 51.0 GB/s | 9,322 MB / 0.221 s / 49.3 | 9,376 MB / 0.201 s / 49.8 | 9,386 MB / 0.201 s / 53.1 |
| Decode 侧网卡上 >10 GB/s 的接收突发 | 9,392 MB / 0.201 s / 50.8 | 9,405 MB / 0.223 s / 52.4 | 9,026 MB / 0.181 s / 51.1 | 9,386 MB / 0.201 s / 50.8 |
| Prefill 侧 `rx_read_requests` 增量（写请求不增加） | 151,768（`mlx5_8`） | 151,768（`mlx5_10`） | 151,768（`mlx5_10`） | 36,234（`mlx5_14`） |
| Prefill 的 prefill 耗时 | 35.37 s | 34.70 s | 34.81 s | 35.14 s |
| Decode 记录的排队时间 | 0.219 s | 0.245 s | 0.218 s | 0.223 s |
| TTFT | 37.1 s | 36.4 s | 36.4 s | 36.9 s |

说明：
- 突发的持续时间由 20 ms 一次的采样决定，±20 ms；峰值超过 50 GB/s（最高 53.1）是采样边界造成的抖动，没有超过线速。
- 每次传输期间，**只有一块 Prefill 网卡和一块 Decode 网卡**出现 >10 GB/s 的流量，其余网卡没有。
- descriptor 数取决于 KV 块分配的连续程度：请求 1～3 是 62,769 个（块大小 32 token，131072 / 32 × 61 = 249,856 是完全碎片化的上限，62,769 约是它的 1/4，说明大约每 4 个块合并成一个 descriptor），请求 4 只有 732 个（几乎连续）。无论多碎，传输耗时都在 191～223 ms，发起耗时只有 1.7～2.8 ms。
- `rx_read_requests` 的计数取决于 descriptor 的数量和大小：732 个 descriptor 对应 36,234，62,769 个对应 151,768（同一个大小的 KV）。

<a id="s6-2"></a>
### 6.2 是否跨轨、是否就近、READ 还是 WRITE

- **跨轨**：4 次传输的网卡序号是 4→7、5→1、5→3、7→5，**全部不同**；第 3.4 节的完整配对表里 16 对也没有一对相同。KV 传输必然经过跨轨路径，仍然达到 49～53 GB/s 的峰值，所以跨轨本身没有可见的速度惩罚；同轨的样本见 6.4（Router 重启后只捕到 1 个）。
- **就近网卡**：Prefill rank r 总是用 `mlx5_{2(r%8)}`，Decode rank d 总是用 `mlx5_{2(d%8)}`，与 GPU 同一个 PCIe 桥（`PXB`）。4 次大 KV 传输加上单请求的 1 次，共 5 次传输、涉及 5 个不同的 Prefill rank 和 5 个不同的 Decode rank，都符合，没有一次用到别的网卡。KV cache 注册在 rank 自己的那块网卡上（1P1D 篇 5.4 用 `rdma resource show mr` 核对过；2P2D 没有重新读）。
- **READ**：Prefill 侧的 `rx_read_requests` 增加、`rx_write_requests` 不因 KV 增加（集合通信的 WRITE 另算，见第 5 章）；线上方向是 Prefill 发、Decode 收。与 1P1D 一样，是 Decode 通过 RDMA READ 从 Prefill 的显存拉取（1P1D 篇 2.5、5.4）。
- **跨节点**：P 池和 D 池永远在不同节点（不同 Pod），所以每次 KV 传输都是跨节点的。

<a id="s6-3"></a>
### 6.3 性能

- 一次 9.21 GB 的传输，`xfer_time` 约 191～223 ms（41～48 GB/s）；线上的发送突发持续约 0.18～0.22 s，峰值 49～53 GB/s，接近单卡 400 Gb/s（50 GB/s）的线速。
- **`xfer_time` ≈ `post_time` + 线上传输时间**：`post_time` 只有 1.7～2.8 ms（每个 descriptor 约 0.04 μs，与 1P1D 相同），线上传输约 190～220 ms。
- 传输只占 TTFT 的约 0.5%：131072 token 的 TTFT 36～37 s，其中 Prefill 计算 34.7～35.4 s，KV 传输约 0.2 s，Decode 排队约 0.22～0.25 s。32768 token（2.3 GB）时 `xfer_time` 约 49～64 ms（5 次：descriptor 61 个的两次是 54.3 和 63.7 ms，62,464 个的三次是 48.9、52.2、52.5 ms）。
- 请求 2 的 `xfer_time`（222.9 ms）比其他三次高约 30 ms，原因没有查清。

<a id="s6-4"></a>
### 6.4 Router 重启后：同轨与跨轨的对照

重启 Router 后出现了 3 对同轨的配对（P11→D11、P12→D12、P15→D7，3.4），于是试着用小请求（`max_tokens=1`）把每个 131072 / 16 token 的大请求引到预定的配对上，对比同轨与跨轨的 KV 传输。**引导基本失败**：`vllm bench` 先发的那个初始测试请求让轮询多前进了一步，而我在日志里的计数没有把它算进去，12 个大请求里只有 1 个落在同轨的配对上（P12→D12，是碰巧）。完成实验时 2P2D 已被删除，没有机会重做。下面是这 12 次传输的实测（预热后，Router 重启后的状态）：

| 请求 | P→D | 网卡序号 P→D | 同轨？ | descriptor 数 | `xfer_time` | 字节 ÷ `xfer_time` | `post_time` | Decode 侧接收突发持续 | Prefill 的 prefill 耗时 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | P10→D8 | 2→0 | 否 | 61 | 230.6 ms | 39.9 GB/s | 0.1 ms | 0.241 s | 35.55 s |
| 2 | P14→D13 | 6→5 | 否 | 249,551 | 201.2 ms | 45.8 GB/s | 13.4 ms | 0.201 s | 34.37 s |
| 3 | P13→D15 | 5→7 | 否 | 125,477 | 190.7 ms | 48.3 GB/s | 5.5 ms | 0.201 s | 34.70 s |
| 4 | P12→D12 | 4→4 | **是** | 249,856 | 205.3 ms | 44.9 GB/s | 11.6 ms | 0.201 s | 35.58 s |
| 5 | P13→D15 | 5→7 | 否 | 249,856 | 203.2 ms | 45.3 GB/s | 10.7 ms | 0.201 s | 34.72 s |
| 6 | P0→D3 | 0→3 | 否 | 62,525 | 190.3 ms | 48.4 GB/s | 3.0 ms | 0.201 s | 34.90 s |
| 7 | P9→D14 | 1→6 | 否 | 249,856 | 198.4 ms | 46.4 GB/s | 11.9 ms | 0.201 s | 35.40 s |
| 8 | P0→D3 | 0→3 | 否 | 249,856 | 197.7 ms | 46.6 GB/s | 11.7 ms | 0.201 s | 34.90 s |
| 9 | P4→D5 | 4→5 | 否 | 187,453 | 196.6 ms | 46.9 GB/s | 10.5 ms | 0.188 s | 35.37 s |
| 10 | P6→D2 | 6→2 | 否 | 187,270 | 206.7 ms | 44.6 GB/s | 10.8 ms | 0.201 s | 34.31 s |
| 11 | P4→D5 | 4→5 | 否 | 187,270 | 192.0 ms | 48.0 GB/s | 6.7 ms | 0.201 s | 35.40 s |
| 12 | P2→D9 | 2→1 | 否 | 249,856 | 229.1 ms | 40.2 GB/s | 13.5 ms | 0.241 s | 35.31 s |

**观察**
- **同轨的那 1 次（请求 4）与跨轨没有区别**：`xfer_time` 205.3 ms、线上突发 0.201 s、峰值 49.7（Prefill 发送）/ 52.5 GB/s（Decode 接收），落在跨轨的 190.3～206.7 ms、0.188～0.201 s 范围内。样本只有 1 个，只能说没有看到差别，不能说同轨和跨轨一样快。
- 其余 11 次都是跨轨，`xfer_time` 190.3～230.6 ms，Decode 侧接收突发 0.188～0.241 s。每个传输的发起耗时与 descriptor 数成正比（约 0.04～0.05 μs / 个）。
- **落在 `pdd-decode-1` 的 NIC 0、1（D8、D9）上的传输偏慢**：请求 1（D8）230.6 ms、请求 12（D9）229.1 ms，加上 6.1 里的请求 2（D9）222.9 ms，Decode 侧接收突发持续 0.223～0.241 s，比其他传输（190.3～206.7 ms、0.188～0.201 s）长约 15%（约 40 GB/s 对 47 GB/s）。这三次的 Prefill 侧网卡和 descriptor 数都各不相同，只有 Decode 侧网卡序号（0 或 1）相同。原因没有查清；NIC 0、1 恰好是 Decode 集合通信用得最多的 rail，但这些传输发生时 Decode 是空闲的，不是集合通信流量造成的。样本只有 3 个，且没有测 `pdd-decode-0` 的 NIC 0、1（D0、D1），不能下结论。
- **1P1D 上的同轨样本**：重新部署的 1P1D 配对里有 1 对同轨（P6→D6），并发实验里该配对收到 6 次传输（2 次运行各 3 次），平均 67.8 ms，其余 42 次跨轨传输平均 60.3 ms，没有稳定的差别（同一个 engine 在两次运行里可以从 50 ms 变到 85 ms）；线上突发也相同（60 ms 左右）。把 2P2D 的 1 次和 1P1D 的 6 次合起来看，**同轨和跨轨的 KV 传输速度没有看到差别**。

<a id="s7"></a>
## 7. 集合通信会影响 KV 传输吗

[↑ 返回顶部](#top)

**实验**：预热后发 24 个 32768 / 128 token 的请求，并发度 8（`BENCH_NUM_PROMPTS=24 BENCH_REQUEST_RATE=inf ./pdd.sh bench --max-concurrency 8`）。这时 Prefill 池的两个节点同时做多个请求的 prefill（RoCE 上有大量集合通信流量），Decode 池也在同时 decode，KV 传输与这些集合通信重叠。对照是前面单请求时的 KV 传输（空闲时只有 KV 在 RoCE 上）。

**整体结果**

| | 数值 |
|---|---|
| 请求 | 24 个全部成功，共 786,432 个输入 token |
| 耗时 / 吞吐 | 34.6 s，0.69 req/s，22.8k token/s（输入 + 输出） |
| TTFT | 平均 8.1 s，中位数 8.5 s，P99 9.1 s |
| TPOT | 平均 22.3 ms，P99 23.1 ms |
| Prefill 节点 RoCE | 每块网卡发送 118～132 GB、接收 115～129 GB（包含集合通信和 KV）；全节点 8 块网卡合计的发送速率峰值 116.3 / 112.8 GB/s |
| Decode 节点 RoCE | NIC 0、1、2 发送 17.3 / 17.0 / 12.9 GB；NIC 3～7 发送 0，只收到 KV（每块 2.3～4.7 GB） |
| 每块网卡的单向速率峰值 | Prefill 47.3～50.3 GB/s，Decode 43.7～51.2 GB/s（这是突发的 KV 传输） |

**NIXL 记录的传输**

| | 24 次传输，并发 | 单请求（32768 token）对照 |
|---|---|---|
| 平均 `xfer_time` | **71.9 ms**（`decode-0` 71.6，`decode-1` 72.4） | 48.9～63.7 ms（5 次，平均约 54 ms） |
| 平均 `post_time` | 1.5 ms | 0.1～3.0 ms |
| 平均 descriptor 数 / 次 | 33,343 / 41,710 | 61 或 62,464 |
| 按字节 ÷ 累计 `xfer_time` | 32.0 GB/s | 约 36～47 GB/s |
| `xfer_time` 分布 | 18 次 ≤ 75 ms，21 次 ≤ 100 ms，24 次 ≤ 200 ms | 5 次都 ≤ 75 ms |

**线上速度**（用 Decode 侧网卡的接收突发来量，那里的接收流量几乎只有 KV；每个突发 ≥ 0.5 GB）

| | 并发（n = 24） | 单请求（n = 5：1 个单请求实验 + 4 个顺序请求实验） |
|---|---|---|
| NIC 0～2（同时承载 Decode 集合通信） | n = 9，持续时间中位数 60 ms（60～80），平均线速中位数 39.1 GB/s（29.8～39.4） | n = 1：80 ms，29.8 GB/s |
| NIC 3～7（只收 KV） | n = 15，持续时间中位数 60 ms（60～60），平均线速中位数 39.0 GB/s（35.8～39.1） | n = 4：60 ms，38.4～39.1 GB/s |
| Prefill 侧网卡上的发送突发（n = 24） | 中位数 60 ms（60～80），平均线速 39.1 GB/s（31.7～42.0） | 同上，约 60 ms |

**结论**
- **线上速度几乎不受集合通信影响**：并发时每次 KV 突发的持续时间（中位数 60 ms）和平均线速（约 39 GB/s）与单请求时相同；即使 KV 落在同时承载 Decode 集合通信的 NIC 0～2 上（n = 9）也与落在干净的 NIC 3～7 上（n = 15）没有差别。原因：集合通信的单卡速率只有 2～4 GB/s（5.6），而 KV 突发占用的是 40～50 GB/s、持续不到 0.1 s，两者在时间和带宽上几乎不冲突。
- **NIXL 记录的传输耗时变长了约 30%**（72 ms 对约 54 ms），而线上时间没变（约 60 ms），`post_time` 也只有 1.5 ms，所以多出来的约 12～18 ms 不在发起阶段，也不在网卡上。**1P1D 的对照（下面）表明这不是集合通信造成的**：1P1D 的 RoCE 上没有集合通信，并发时同样变长。可能的原因是并发时 Decode 引擎在持续 decode，完成通知要等引擎的下一个步边界才被处理（**推断**，没有验证）。
- descriptor 数从 61 涨到 3 万多，`post_time` 仍只有 1.5 ms，说明 descriptor 碎片本身不是传输变慢的原因。
- **没有测的方向**：KV 的 READ 流量反过来会不会拖慢同一块网卡上的集合通信（例如延长 prefill 步的时间）没有测量。

**对照实验：1P1D（RoCE 上没有集合通信）**

在重新部署的 1P1D 上做同样的实验（详细数据见 1P1D 篇 5.7；预热后，24 个 32768 / 128 token 的请求，并发度 8，做 2 次；单请求作基线）：

| | 单请求（基线） | 并发 24 个（第 1 次） | 并发 24 个（第 2 次） |
|---|---|---|---|
| 1P1D：NIXL 平均传输耗时 | 48.2～49.3 ms | **65.5 ms** | **57.0 ms** |
| 1P1D：线上突发（Decode 侧）中位数 | 60 ms，39.1 GB/s | 60 ms，39.0 GB/s | 60 ms，39.0 GB/s |
| 1P1D：RoCE 上的集合通信 | 无 | 无（每块网卡只有 3 次 KV，7051～7057 MB） | 无 |
| 2P2D：NIXL 平均传输耗时 | 约 54 ms | **71.9 ms** | — |

- **1P1D 并发时 NIXL 传输耗时同样变长**（49 ms → 57～65 ms，+16%～34%），线上速度不变。所以并发下传输耗时变长**主要不是 RoCE 集合通信造成的**，是并发本身。
- 2P2D（71.9 ms）比 1P1D（57.0～65.5 ms）高 6～15 ms，但 1P1D 两次运行之间就差 8.5 ms，所以无法确定这一部分是不是集合通信造成的，也不能排除。
- 线上速度在 1P1D 和 2P2D 上完全相同，说明 RoCE 集合通信对 KV 的线上传输没有可测的影响。

<a id="s8"></a>
## 8. 与 1P1D 的对比

[↑ 返回顶部](#top)

**1P1D 是什么**：Prefill 和 Decode 各 1 个节点，每个池是 8 张 GPU 的 DEP 组（DP8 + EP8），EP 的集合通信全在节点内（NVLink），RoCE 上只有 P→D 的 KV 传输；部署、原理、观测方法和单池的实测见 1P1D 篇。2P2D 把每个池扩成 2 个节点（DP16 + EP16），多出来的是跨节点的 EP 集合通信，所以两者的差别主要在：集合通信走不走 RoCE、Decode 每步的延迟、并发下的吞吐。

| | 1P1D | 2P2D |
|---|---|---|
| 每个池的规模 | 8 张 GPU（DP8 + EP8） | 16 张 GPU（DP16 + EP16，跨 2 个节点） |
| 每 rank 的专家数 | 32 | 16 |
| `init engine` 耗时 | Prefill 约 1098 s，Decode 约 514 s | Prefill 约 1144 s，Decode 约 531 s |
| 32768 token 的 TTFT | 3.51～3.55 s | 3.55～3.71 s |
| 32768 token 的 TPOT | 16.1～16.3 ms（并发 8 时 17.3～17.4 ms） | 20.8～21.3 ms（并发 8 时 22.3 ms） |
| 131072 token 的 Prefill 耗时 | 34.1～35.2 s（两次部署） | 34.70～35.37 s |
| 单个 Prefill 节点 NVLink 流量（32768 token 的请求） | 约 278 GiB | 293～331 GiB |
| Decode 节点 NVLink 流量（同上） | 19.2～19.5 GiB | 36.5 GiB |
| 集合通信的 RoCE 流量 | 无 | Prefill 每节点约 42～44 GB，Decode 每节点约 6.2 GB |
| KV 传输 9.21 GB | 190～198 ms（46～48 GB/s） | 191～223 ms（41～48 GB/s） |
| KV 传输的发起耗时 | 0.1～13.4 ms（0.04 μs / descriptor） | 1.7～2.8 ms（0.04 μs / descriptor） |
| P 与 D 的 rank 配对 | 一次启动内固定 8 种；第一次部署全部跨轨，重新部署后有 1 对同轨 | 一次启动内固定 16 种；本文主要数据所在的启动全部跨轨，Router 重启后有 3 对同轨 |
| 24 个 32768 token 请求，并发度 8：耗时 / 吞吐 | 28.6～28.7 s，0.84 req/s，27.5～27.6k tok/s | 34.6 s，0.69 req/s，22.8k tok/s |
| 同上：TTFT 平均 | 6.85 s | 8.08 s |
| 同上：NIXL 平均传输耗时 | 57.0～65.5 ms | 71.9 ms |
| 同上：每个请求的 NVLink（Prefill / Decode） | 275.8 / 5.2～5.4 GiB | 296～299 / 11.5 GiB（每节点） |
| 空闲时 Worker CPU | 100% | 100% |

- **2P2D 与 1P1D 的 prefill 速度基本相同**：虽然 EP16 的集合通信要跨 2 个节点、RoCE 的字节量也不小（Prefill 每节点约 43 GB），但单请求 TTFT 只有 3.55～3.71 s 对 3.5 s。2P2D 的代价体现在 Decode：TPOT 约 21 ms，比 1P1D 的 16 ms 多约 5 ms，Decode 的 NVLink 流量也近 2 倍（每步的集合通信要跨节点）。
- **TPOT 为什么高约 5 ms（数据支持什么、不支持什么）**：
  - **观察**：单请求 TPOT 16.1～16.3 ms 对 20.8～21.3 ms，并发度 8 时 17.3～17.4 ms 对 22.3 ms，两种负载下差值都是约 5 ms；Decode 池的集合通信在 2P2D 里要经过 RoCE（NIC 0～2，单卡峰值只有 0.8 GB/s，远没有跑满），在 1P1D 里全在 NVLink 内；同样的 prefill（大消息）TTFT 却没有差别。
  - **与「跨节点通信时延使每步变长」一致**：差值不随并发度变化，说明它更像每个 decode 步固定多出来的时间，而不是随 token 数增长的计算量；RoCE 带宽远没有用满，说明不是带宽问题；prefill 的消息大、计算时间长，通信时延被掩盖，所以 TTFT 不受影响。粗略折算：每步 58 层 × 2 次集合通信 = 116 次，5 ms 分摊到每次约 40 μs（**推断**，这是除法，不是测量）。
  - **但不能据此断言**：没有测每一层集合通信的耗时，也没有把「跨节点」和 EP16 带来的其他变化分开，这些同样会让每步变慢——集合通信的参与 rank 从 8 个增加到 16 个、每个 rank 的专家数从 32 个减到 16 个，以及每步的 DP 元数据同步（Gloo，经 eth0 的 TCP，1P1D 篇 2.3）也要跨节点。「算力不是瓶颈」也只是由 Decode 每步 token 很少推出来的，没有测 GPU 的计算时间。要验证需要分层计时（例如在 NCCL 层或 profiler 里测每次 `all_gatherv` / `reduce_scatterv` 的耗时）或者用一个节点内的 DP16 对照，本环境没有做。
- **并发度 8 时，1P1D 比 2P2D 更快**：同样的 24 个请求，1P1D 28.6～28.7 s（0.84 req/s），2P2D 34.6 s（0.69 req/s），TTFT（6.85 对 8.08 s）、TPOT（17.4 对 22.3 ms）也是 1P1D 更好。2P2D 的两个池各有 16 个 rank，容量更大，但在这个并发度下没有体现出优势；更高并发（例如 16、32）下 2P2D 是否反超没有测。
- **KV 传输**：线上速度相同（并发时突发中位数都是 60 ms、约 39 GB/s）；并发下 NIXL 记录的传输耗时两边都变长（1P1D +16%～34%，2P2D 约 +33%），所以变长主要不是集合通信造成的（第 7 章）。

<a id="s9"></a>
## 9. 推断核对与未验证

[↑ 返回顶部](#top)

**本次已验证**

| 说法 | 结果 |
|---|---|
| 不用改 YAML / 脚本 | **已证实**（按默认节点数部署，文件未改，Router 自动发现 32 个 worker，`metrics` 按 16 engines 汇总） |
| `DPCoordinator` 只在节点 0 | **已证实**（`ps` 对比） |
| 池内节点之间有控制面连接，P 与 D 之间没有 TCP | **已证实**（398 对 0） |
| Router 只连 Head | **被推翻**：Router 对 4 个 GPU Pod 都有连接，空闲时看不到是因为短连接（3.3） |
| P→D 的配对是 16×16 | **被推翻**：只有固定的 16 对；配对随 Router 启动重排，重启后出现了同轨配对（3.4） |
| EP 集合通信同时走 NVLink 和 RoCE | **已证实**（NVLink 计数、RoCE 字节数与 `rx_write_requests`） |
| 跨节点集合通信 rail 对齐（NIC i ↔ NIC i） | **已证实**（两个 Prefill 节点、两个 Decode 节点的逐 NIC 字节数一致） |
| Decode 池只用 3 条 rail 做集合通信 | **已证实**（现象）；原因是推断 |
| KV 传输是 RDMA READ、一次一块网卡、就近网卡、全部跨轨 | **已证实**（4 次大 KV + 单请求 + 16 个配对） |
| 集合通信的 RoCE 流量会拖慢 KV 的线上传输 | **未观察到**（并发 24 次，线上速度与单请求相同） |
| 集合通信会让 NIXL 记录的传输耗时变长 | **主要不是**：并发下变长约 30%（72 ms 对约 54 ms），1P1D（RoCE 无集合通信）并发下同样变长 16%～34%；线上时间都没变；是否有 ≤ 10 ms 的额外影响无法确定（第 7 章） |

**未验证**
- **同轨传输**的速度：2P2D 上只捕到 1 次、1P1D 上 6 次，都没有看到与跨轨的差别（6.4）；样本仍然少，且 1P1D 的运行间波动较大。
- KV 的 READ 流量对集合通信（prefill 步的时间）的反向影响。
- NCCL 的算法、通道数和为什么 Decode 只用 3 条 rail（没有读 NCCL 日志；设置 `NCCL_DEBUG` 需要改 YAML，本次没有做）。
- 并发下 NIXL 记录的传输耗时多出约 12～18 ms 的来源（1P1D 对照表明主要不是集合通信，第 7 章）。
- 同序号网卡是否在交换机上属于同一条 rail（假设成立）。
- Router 配对的顺序为什么在重启后改变（只观察到 P、D 两个固定顺序的轮询，没有读 `vllm-router` 源码，3.4）。
- 落在 `pdd-decode-1` 的 NIC 0、1 上的 KV 传输偏慢约 15% 的原因，以及 `pdd-decode-0` 的 NIC 0、1 是否同样（6.4）。
- Prefill 初始化比 Decode 慢约 2 倍的原因；`pdd-prefill-1` 里个别 GPU 的 NVLink 流量偏低的原因；5.4 里个别网卡发送字节数不对称的原因；6.1 请求 2 的 `xfer_time` 偏高的原因。
- 2P2D 上空闲 Worker 占满 CPU 核的线程身份（只确认了 CPU 占用，没有逐线程核对）。
- 并发度和输入长度的系统扫描（只测了并发度 8；更高并发下 2P2D 是否超过 1P1D）、P/D 配比（如 1P+3D）。
- 部署的创建、重建和删除由使用者决定；调整节点数或任何开关都要 `down` 再 `up`。

<a id="s10"></a>
## 10. 附录：原始输出与辅助脚本

[↑ 返回顶部](#top)

用 `kubectl exec <pod> -- ps -ef --width 250` 读取，2026-10-13 02:14 UTC，Pod 已运行约 39 分钟，2P2D 空闲。输出没有删行；命令行被截到 250 列；末尾的 `bash -c date ...`、`ps -ef`、`cut` 是采集命令本身产生的进程；`pdd-prefill-0`、`pdd-prefill-1` 里挂在 `pts/0` 上的 `/bin/bash` 是带 TTY 的交互 `kubectl exec` 会话，不是这次采集创建的。

### A.1 `pdd-prefill-0`（节点 0，有 `DPCoordinator`）

```
2026-10-13T02:14:40Z
UID          PID    PPID  C STIME TTY          TIME CMD
root           1       0  1 01:35 ?        00:00:39 /usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --host 0.0.0.0 --port 8001 --kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_loa
root        1405       1  0 01:35 ?        00:00:00 /usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(40)
root        1406       1  1 01:35 ?        00:00:26 VLLM::DPCoordinator
root        1606       1  3 01:35 ?        00:01:28 VLLM::EngineCore_DP0
root        1607       1  3 01:35 ?        00:01:28 VLLM::EngineCore_DP1
root        1608       1  4 01:35 ?        00:01:35 VLLM::EngineCore_DP2
root        1609       1  3 01:35 ?        00:01:29 VLLM::EngineCore_DP3
root        1610       1  3 01:35 ?        00:01:28 VLLM::EngineCore_DP4
root        1611       1  3 01:35 ?        00:01:27 VLLM::EngineCore_DP5
root        1612       1  3 01:35 ?        00:01:28 VLLM::EngineCore_DP6
root        1613       1  3 01:35 ?        00:01:27 VLLM::EngineCore_DP7
root        1614       1  0 01:35 ?        00:00:22 VLLM::APIServer_0
root        1615       1  0 01:35 ?        00:00:19 VLLM::APIServer_1
root        1616       1  1 01:35 ?        00:00:27 VLLM::APIServer_2
root        1617       1  0 01:35 ?        00:00:22 VLLM::APIServer_3
root        1618       1  1 01:35 ?        00:00:31 VLLM::APIServer_4
root        1619       1  1 01:35 ?        00:00:30 VLLM::APIServer_5
root        1620       1  0 01:35 ?        00:00:17 VLLM::APIServer_6
root        1621       1  0 01:35 ?        00:00:21 VLLM::APIServer_7
root        7044    1606 99 01:36 ?        00:45:16 VLLM::Worker_DP0_EP0
root        7045    1613 99 01:36 ?        00:45:17 VLLM::Worker_DP7_EP7
root        7046    1612 99 01:36 ?        00:45:23 VLLM::Worker_DP6_EP6
root        7202    1608 99 01:36 ?        00:45:31 VLLM::Worker_DP2_EP2
root        7203    1609 99 01:36 ?        00:45:19 VLLM::Worker_DP3_EP3
root        7204    1607 99 01:36 ?        00:45:25 VLLM::Worker_DP1_EP1
root        7205    1610 99 01:36 ?        00:45:15 VLLM::Worker_DP4_EP4
root        7206    1611 99 01:36 ?        00:45:26 VLLM::Worker_DP5_EP5
root       25210       0  9 02:06 ?        00:00:51 python3 - 940 0.02
root       25393       0  8 02:10 ?        00:00:18 python3 - 330 0.02
root       25474       0  0 02:14 pts/0    00:00:00 /bin/bash
root       25489       0  0 02:14 ?        00:00:00 bash -c date -u +%FT%TZ; ps -ef --width 250 | cut -c1-250
root       25496   25489  0 02:14 ?        00:00:00 ps -ef --width 250
root       25497   25489  0 02:14 ?        00:00:00 cut -c1-250
```

### A.2 `pdd-prefill-1`（节点 1，没有 `DPCoordinator`）

```
2026-10-13T02:14:41Z
UID          PID    PPID  C STIME TTY          TIME CMD
root           1       0  1 01:35 ?        00:00:39 /usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --host 0.0.0.0 --port 8001 --kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_loa
root        1444       1  0 01:36 ?        00:00:00 /usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(45)
root        1445       1  4 01:36 ?        00:01:33 VLLM::EngineCore_DP8
root        1446       1  3 01:36 ?        00:01:27 VLLM::EngineCore_DP9
root        1447       1  3 01:36 ?        00:01:28 VLLM::EngineCore_DP10
root        1448       1  3 01:36 ?        00:01:30 VLLM::EngineCore_DP11
root        1449       1  3 01:36 ?        00:01:31 VLLM::EngineCore_DP12
root        1450       1  3 01:36 ?        00:01:24 VLLM::EngineCore_DP13
root        1451       1  3 01:36 ?        00:01:27 VLLM::EngineCore_DP14
root        1452       1  3 01:36 ?        00:01:25 VLLM::EngineCore_DP15
root        1453       1  0 01:36 ?        00:00:22 VLLM::APIServer_0
root        1454       1  1 01:36 ?        00:00:23 VLLM::APIServer_1
root        1455       1  0 01:36 ?        00:00:17 VLLM::APIServer_2
root        1456       1  0 01:36 ?        00:00:18 VLLM::APIServer_3
root        1457       1  0 01:36 ?        00:00:17 VLLM::APIServer_4
root        1458       1  0 01:36 ?        00:00:19 VLLM::APIServer_5
root        1459       1  0 01:36 ?        00:00:21 VLLM::APIServer_6
root        1460       1  0 01:36 ?        00:00:19 VLLM::APIServer_7
root        6873    1445 99 01:36 ?        00:45:37 VLLM::Worker_DP8_EP8
root        6874    1447 99 01:36 ?        00:45:34 VLLM::Worker_DP10_EP10
root        6877    1452 99 01:36 ?        00:45:26 VLLM::Worker_DP15_EP15
root        6930    1446 99 01:36 ?        00:45:34 VLLM::Worker_DP9_EP9
root        7020    1449 99 01:36 ?        00:45:31 VLLM::Worker_DP12_EP12
root        7027    1450 99 01:36 ?        00:45:27 VLLM::Worker_DP13_EP13
root        7033    1451 99 01:36 ?        00:45:42 VLLM::Worker_DP14_EP14
root        7037    1448 99 01:36 ?        00:45:29 VLLM::Worker_DP11_EP11
root       25005       0 10 02:06 ?        00:00:51 python3 - 940 0.02
root       25190       0  8 02:10 ?        00:00:18 python3 - 330 0.02
root       25265       0  0 02:13 pts/0    00:00:00 /bin/bash
root       25288       0  0 02:14 ?        00:00:00 bash -c date -u +%FT%TZ; ps -ef --width 250 | cut -c1-250
root       25295   25288  0 02:14 ?        00:00:00 ps -ef --width 250
root       25296   25288  0 02:14 ?        00:00:00 cut -c1-250
```

### A.3 `pdd-decode-0`（节点 0，有 `DPCoordinator`）

```
2026-10-13T02:14:43Z
UID          PID    PPID  C STIME TTY          TIME CMD
root           1       0  1 01:35 ?        00:00:39 /usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --host 0.0.0.0 --port 8002 --kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_loa
root        1407       1  0 01:35 ?        00:00:00 /usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(40)
root        1408       1  1 01:35 ?        00:00:26 VLLM::DPCoordinator
root        1609       1  3 01:35 ?        00:01:10 VLLM::EngineCore_DP0
root        1610       1  2 01:35 ?        00:01:02 VLLM::EngineCore_DP1
root        1611       1  3 01:35 ?        00:01:17 VLLM::EngineCore_DP2
root        1612       1  2 01:35 ?        00:01:05 VLLM::EngineCore_DP3
root        1613       1  2 01:35 ?        00:01:08 VLLM::EngineCore_DP4
root        1614       1  2 01:35 ?        00:01:05 VLLM::EngineCore_DP5
root        1615       1  3 01:35 ?        00:01:14 VLLM::EngineCore_DP6
root        1616       1  2 01:35 ?        00:01:08 VLLM::EngineCore_DP7
root        1617       1  0 01:35 ?        00:00:20 VLLM::APIServer_0
root        1618       1  0 01:35 ?        00:00:15 VLLM::APIServer_1
root        1619       1  1 01:35 ?        00:00:25 VLLM::APIServer_2
root        1620       1  0 01:35 ?        00:00:22 VLLM::APIServer_3
root        1621       1  0 01:35 ?        00:00:22 VLLM::APIServer_4
root        1622       1  1 01:35 ?        00:00:24 VLLM::APIServer_5
root        1623       1  0 01:35 ?        00:00:17 VLLM::APIServer_6
root        1624       1  0 01:35 ?        00:00:20 VLLM::APIServer_7
root        7043    1609 99 01:36 ?        00:41:23 VLLM::Worker_DP0_EP0
root        7048    1615 99 01:36 ?        00:40:35 VLLM::Worker_DP6_EP6
root        7049    1616 99 01:36 ?        00:40:34 VLLM::Worker_DP7_EP7
root        7205    1611 99 01:36 ?        00:41:27 VLLM::Worker_DP2_EP2
root        7206    1613 99 01:36 ?        00:40:39 VLLM::Worker_DP4_EP4
root        7207    1614 99 01:36 ?        00:40:34 VLLM::Worker_DP5_EP5
root        7208    1610 99 01:36 ?        00:41:26 VLLM::Worker_DP1_EP1
root        7209    1612 99 01:36 ?        00:40:33 VLLM::Worker_DP3_EP3
root       25249       0  9 02:06 ?        00:00:51 python3 - 940 0.02
root       25431       0  8 02:10 ?        00:00:18 python3 - 330 0.02
root       25526       0  0 02:14 ?        00:00:00 bash -c date -u +%FT%TZ; ps -ef --width 250 | cut -c1-250
root       25533   25526  0 02:14 ?        00:00:00 ps -ef --width 250
root       25534   25526  0 02:14 ?        00:00:00 cut -c1-250
```

### A.4 `pdd-decode-1`（节点 1，没有 `DPCoordinator`）

```
2026-10-13T02:14:44Z
UID          PID    PPID  C STIME TTY          TIME CMD
root           1       0  1 01:35 ?        00:00:39 /usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --host 0.0.0.0 --port 8002 --kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_loa
root        1440       1  0 01:36 ?        00:00:00 /usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(45)
root        1441       1  3 01:36 ?        00:01:15 VLLM::EngineCore_DP8
root        1442       1  3 01:36 ?        00:01:17 VLLM::EngineCore_DP9
root        1443       1  3 01:36 ?        00:01:10 VLLM::EngineCore_DP10
root        1444       1  3 01:36 ?        00:01:16 VLLM::EngineCore_DP11
root        1445       1  3 01:36 ?        00:01:15 VLLM::EngineCore_DP12
root        1446       1  3 01:36 ?        00:01:11 VLLM::EngineCore_DP13
root        1447       1  3 01:36 ?        00:01:12 VLLM::EngineCore_DP14
root        1448       1  3 01:36 ?        00:01:09 VLLM::EngineCore_DP15
root        1449       1  0 01:36 ?        00:00:18 VLLM::APIServer_0
root        1450       1  0 01:36 ?        00:00:19 VLLM::APIServer_1
root        1451       1  1 01:36 ?        00:00:29 VLLM::APIServer_2
root        1452       1  0 01:36 ?        00:00:16 VLLM::APIServer_3
root        1453       1  0 01:36 ?        00:00:16 VLLM::APIServer_4
root        1454       1  0 01:36 ?        00:00:20 VLLM::APIServer_5
root        1455       1  0 01:36 ?        00:00:22 VLLM::APIServer_6
root        1456       1  0 01:36 ?        00:00:18 VLLM::APIServer_7
root        6842    1441 99 01:36 ?        00:41:30 VLLM::Worker_DP8_EP8
root        6843    1442 99 01:36 ?        00:41:30 VLLM::Worker_DP9_EP9
root        7026    1445 99 01:36 ?        00:40:39 VLLM::Worker_DP12_EP12
root        7027    1447 99 01:36 ?        00:40:35 VLLM::Worker_DP14_EP14
root        7032    1443 99 01:36 ?        00:41:34 VLLM::Worker_DP10_EP10
root        7033    1448 99 01:36 ?        00:40:33 VLLM::Worker_DP15_EP15
root        7034    1446 99 01:36 ?        00:40:35 VLLM::Worker_DP13_EP13
root        7035    1444 99 01:36 ?        00:40:37 VLLM::Worker_DP11_EP11
root       24944       0 10 02:06 ?        00:00:56 python3 - 940 0.02
root       25135       0  7 02:10 ?        00:00:17 python3 - 330 0.02
root       25238       0  0 02:14 ?        00:00:00 bash -c date -u +%FT%TZ; ps -ef --width 250 | cut -c1-250
root       25245   25238  0 02:14 ?        00:00:00 ps -ef --width 250
root       25246   25238  0 02:14 ?        00:00:00 cut -c1-250
```

<a id="a5"></a>
### A.5 辅助脚本

2P2D 的数据用的是 1P1D 篇附录 A.4 的那套只读脚本（采集脚本：NVLink 计数器、RoCE 字节采样器、RDMA 请求计数、NIXL 指标；实验驱动 `exp_window.sh`、`exp_per_request.sh`；分析脚本 `analyze_window.py`、`analyze_per_request.py`），在多节点上这样用：

- **驱动脚本**：设置 `PODS="pdd-prefill-0 pdd-prefill-1 pdd-decode-0 pdd-decode-1" PN=2 DN=2`，例如 `LABEL=conc8 PODS="…" PN=2 DN=2 BENCH_INPUT_LEN=32768 BENCH_ARGS="--num-prompts 24 --max-concurrency 8" ./exp_window.sh`。脚本在实验开始时把 `{Pod IP: Pod 名}` 存进 `ipmap.json`。
- **分析脚本**：按目录里的文件自动识别 4 个 Pod；用 `ipmap.json` 把 Router 日志里的 `ip@节点内 rank` 换算成 global rank（P0..P15、D0..D15），输出里的网卡序号是 global rank % 8。没有 `ipmap.json` 的旧数据用 `--ipmap ip=pod,ip=pod` 指定。`analyze_window.py` 还会把 Decode 的 KV 突发按网卡分成 NIC 0～2（同时承载 Decode 集合通信）和 NIC 3～7（只有 KV）两组（`--split` 可改分界），第 7 章的对比就是这样做的。

2P2D 另外用到的脚本：

| 脚本 | 用在 | 作用 |
|---|---|---|
| `router_pairs.py` | 3.4 | 从 Router 日志统计 P→D 配对表（global rank、出现次数、两端网卡序号、是否同轨）和选择顺序；配对随 Router 重启而变，只能用同一次 Router 运行的日志 |
| `tcp_peers.sh <pod>…` | 3.3 | 统计每个 GPU Pod 到各个对端 Pod 的 ESTABLISHED TCP 连接数（池内、跨池） |
| `idle_cpu.py` | 3.5 | 空闲时每个 Worker 的 CPU 占用 |
| `sampler_ab.sh` | 4.1 | 有无 20 ms RoCE 采样器时各发几次 bench，对比 TTFT / TPOT，确认采样没有影响 |
| `steer_pairs.sh` | 6.4 | 用 `max_tokens=1` 的小请求推进 Router 的轮询，把每个 131072 token 的请求引到预定的配对上；见下面的限制 |

**注意事项**
- `steer_pairs.sh` 就是 6.4 里**引导失败**的那个脚本：`vllm bench` 会先发一个不计入日志的初始测试请求，位置差一步，12 个请求只有 1 个落在目标配对上。想重做，需要在发大请求前再多推进一步，或者不用 `vllm bench`、直接发请求。
- `tcp_peers.sh` 是按文中统计方式重写的，**没有在 2P2D 上重跑**（2P2D 已删除），在 1P1D 上测试过（P 与 D 之间 0 条连接）；文中的 398 / 0 来自当时的一次性命令。统计里包含 Pod 到自己 IP 的本地连接。
- 其他脚本的细节和注意事项（预热、计数器是累计值、NVLink 计数器不区分来源等）见 1P1D 篇 A.4。

**验证情况**：两个分析脚本在 2P2D 的原始数据上重跑过：单请求的 RoCE 字节数（如 `prefill-1` NIC 7 发送 7990 MB）、4 个 131072 token 请求的传输耗时（193.0、222.9 ms 等）和配对都与第 5、6 章一致；`idle_cpu.py`、`router_pairs.py`、`tcp_peers.sh` 在运行中的 1P1D 上做过只读调用；`sampler_ab.sh`、`steer_pairs.sh` 和驱动脚本的 2P2D 用法只做了语法检查，没有再跑（2P2D 已删除）。

[↑ 返回顶部](#top)
