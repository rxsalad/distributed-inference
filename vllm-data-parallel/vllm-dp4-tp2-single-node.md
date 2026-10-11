# vLLM 单节点数据并行：DP=4、TP=2（Qwen3-8B）

在一台 8 卡节点上部署 4 个相互独立的 Qwen3-8B 副本，每个副本用 TP=2 占 2 张 GPU，对外只有一个 8000 端口，共 4 × 2 = 8 张 GPU。

**要点**

- 稠密模型下，DP rank 之间互相独立；每个 rank 内部才有 TP（NCCL all-reduce，走 NVLink）。DP=4 等价于 4 个独立的 TP=2 副本加一个统一入口。
- 一条 `vllm serve` 命令拉起 19 个进程：4 个 APIServer、1 个 DPCoordinator、4 个 EngineCore、8 个 Worker，外加主进程和 resource_tracker。
- 4 个 APIServer 共用主进程创建的同一个监听 socket；连接由内核分配，并不均匀。请求到 DP rank 的分配则由 APIServer 按各 rank 的队列长度决定。
- DPCoordinator 不在请求路径上，只发布负载统计。
- 进程之间的小消息走 ZMQ（本机为 IPC）；EngineCore 到 Worker 的调度结果走共享内存，ZMQ 只做唤醒通知。

**环境与参考**

- 环境：server-0（StatefulSet `server` 的第 0 个 pod），8×NVIDIA B300，镜像 `vllm/vllm-openai:cu130-nightly`，vLLM 0.19.2rc1。
- 负载均衡模式：Internal（单一端点，调度在 API server 进程内完成）。
- 参考：<https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/#internal-load-balancing>
- 双节点配置见 `vllm-dp4-tp1-two-node.md`，本文不涉及。

```
              Client
                │ HTTP :8000
                ▼
     APIServer_0..3 （共用同一个监听 socket）
                │ 按各 rank 队列长度选 rank（统计来自 DPCoordinator）
                ▼ ZMQ
   EngineCore_DP0   DP1   DP2   DP3
        │            │     │     │
   Worker_TP0/1   TP0/1  TP0/1  TP0/1     每个 rank 内部 NCCL all-reduce
   GPU0,1         GPU2,3 GPU4,5 GPU6,7
```

## 1. 启动与验证

```
# 启动
vllm serve Qwen/Qwen3-8B \
    --data-parallel-size 4 \
    --tensor-parallel-size 2

# 验证
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "Who are you?"}
    ]
  }'
```

实测返回正常。Qwen3 默认输出 `<think>` 思考内容，`reasoning` 字段为 null，说明未启用 reasoning parser。

## 2. 并行方式：稠密模型下 DP 与 TP 的分工

Qwen3-8B 是稠密模型（不是 MoE），FFN 是普通 MLP，没有路由和专家。模型参数（来自 `config.json`）：36 层，hidden_size 4096，32 个 Q 头 / 8 个 KV 头，head_dim 128，intermediate_size 12288，vocab_size 151936。

- **DP 这一层**：4 个 rank 各持有完整的一份权重，处理不同的请求；rank 之间没有前向计算的通信，各自的 KV cache 也相互独立。
- **TP 这一层**：每个 rank 内的 2 张卡切分整个模型，Attention 和 FFN 都是 TP=2。

| 部分 | 切分方式 | 每张卡的大小 |
|---|---|---|
| Attention：QKV 投影 | 按头切分（列并行） | 16 个 Q 头、4 个 KV 头 |
| Attention：输出投影 | 按行切分（行并行），之后 all-reduce | |
| FFN：gate / up 投影 | 按列切分 | intermediate 6144（12288 的一半） |
| FFN：down 投影 | 按行切分（行并行），之后 all-reduce | |
| Embedding / LM head | 按词表切分 | 75968（151936 的一半） |
| KV cache | 随 KV 头切分 | 4 个 KV 头 |

通信：每层 2 次 all-reduce（Attention 后、FFN 后），36 层共 72 次，走同一个 DP rank 内的 NVLink（NCCL）；DP rank 之间不通信。该数字按标准 Megatron 式 TP 推算，非实测，未计 embedding 和 logits 的少量通信。

**与 MoE 模型（如 DeepSeek）的区别**：

| | 稠密模型（Qwen3-8B） | MoE 模型 |
|---|---|---|
| Attention | 组内 TP | 常用 DP（MLA 的 KV 小，适合 DP） |
| FFN | 组内 TP，只涉及本 rank 的 2 张卡 | 专家层用 DP×TP 组成的 TP 组，或使用 EP，所有 rank 一起参与 |
| DP rank 之间 | 完全独立 | 通过专家层耦合，每次前向都要同步 |
| dummy batch | 不需要 | 需要：只要有一个 rank 在跑，其余空闲 rank 也要做空的前向 |

## 3. 进程结构（实测）

启动后共 19 个进程（含 resource_tracker）：

```
vllm serve (主进程)
├─ resource_tracker                    Python multiprocessing 资源清理
├─ DPCoordinator                       DP 协调器
├─ EngineCore_DP0 ── Worker_TP0, Worker_TP1
├─ EngineCore_DP1 ── Worker_TP0, Worker_TP1
├─ EngineCore_DP2 ── Worker_TP0, Worker_TP1
├─ EngineCore_DP3 ── Worker_TP0, Worker_TP1
└─ APIServer_0..3                      4 个 HTTP 前端，共用 8000 端口
```

| 进程 | 数量 | 作用 |
|---|---|---|
| `vllm serve` 主进程 | 1 | 解析参数，启动并监控所有子进程；创建 8000 端口的监听 socket |
| resource_tracker | 1 | Python multiprocessing 的资源清理辅助进程 |
| APIServer | 4 | OpenAI 兼容 HTTP 接口；负责分词、请求解析、流式输出；通过 ZMQ 与 EngineCore 通信。数量由 `--api-server-count` 决定，默认等于 DP size（启动日志：`Defaulting api_server_count to data_parallel_size (4)`），与 GPU 数无关 |
| DPCoordinator | 1 | 不在请求路径上；收集各 rank 的队列长度并发布给 APIServer，维护 wave 状态（见 4.2） |
| EngineCore_DPn | 4 | 每个 DP rank 一个，负责调度、continuous batching、KV cache 块管理，本身不做 GPU 计算 |
| Worker_TP0/TP1 | 8 | 每个引擎 2 个，各独占 1 张 GPU，持有一半权重，执行前向计算；组内通过 NCCL all-reduce 通信 |

另外，容器的 PID 1 是 YAML 里的 `sleep infinity`，用来保持 pod 运行，不属于 vLLM。

**EngineCore 与 Worker 的对应关系**：一个 EngineCore 对应 TP 个 Worker（本例 2 个），二者是父子进程，Worker 的 PPID 就是所属 EngineCore 的 PID。

- EngineCore 只做调度；调度结果广播给它的 2 个 Worker，两个 Worker 同步执行前向，之间用 NCCL 做 all-reduce。
- Worker 总数 = DP × TP = 4 × 2 = 8。
- 每个 DP rank 占相邻两张卡：DP0 → GPU0/1，DP1 → GPU2/3，DP2 → GPU4/5，DP3 → GPU6/7。
- 判断 Worker 归属要看 PPID，不能看 PID 大小：PID 由进程创建时序决定，例如 DP3 的 Worker（3498、3499）PID 比 DP0 的（3515）还小。

完整进程清单（PID 为某次实测值，每次启动都会变）：

| # | PID | 进程 | 父进程 | GPU |
|---|---|---|---|---|
| 1 | 1353 | `vllm serve` 主进程 | 1334 | |
| 2 | 1742 | resource_tracker | 1353 | |
| 3 | 1743 | DPCoordinator | 1353 | |
| 4 | 1941 | EngineCore_DP0 | 1353 | |
| 5 | 1942 | EngineCore_DP1 | 1353 | |
| 6 | 1943 | EngineCore_DP2 | 1353 | |
| 7 | 1944 | EngineCore_DP3 | 1353 | |
| 8 | 1945 | APIServer_0 | 1353 | |
| 9 | 1946 | APIServer_1 | 1353 | |
| 10 | 1947 | APIServer_2 | 1353 | |
| 11 | 1948 | APIServer_3 | 1353 | |
| 12 | 3515 | Worker_TP0 | 1941 (DP0) | GPU0 |
| 13 | 3516 | Worker_TP1 | 1941 (DP0) | GPU1 |
| 14 | 3518 | Worker_TP0 | 1942 (DP1) | GPU2 |
| 15 | 3520 | Worker_TP1 | 1942 (DP1) | GPU3 |
| 16 | 3517 | Worker_TP0 | 1943 (DP2) | GPU4 |
| 17 | 3519 | Worker_TP1 | 1943 (DP2) | GPU5 |
| 18 | 3498 | Worker_TP0 | 1944 (DP3) | GPU6 |
| 19 | 3499 | Worker_TP1 | 1944 (DP3) | GPU7 |

## 4. 对外入口与负载均衡

依据 vLLM 0.19.2rc1 源码：`v1/engine/coordinator.py`、`entrypoints/openai/api_server.py`、`entrypoints/cli/serve.py`。

负载均衡分两层：

| 层 | 谁决定 | 粒度 | 依据 |
|---|---|---|---|
| 连接 → APIServer | 内核 accept | 连接 | 谁先抢到连接（4.1） |
| 请求 → DP rank | 该 APIServer 的 client | 请求 | Coordinator 发布的各 rank 队列长度（4.2） |

注意：第一层按连接分担，不是按请求。保持长连接（keep-alive）的客户端，其所有请求都会落在同一个 APIServer 上；压测时如果并发连接数太少，可能打不满 4 个 APIServer，应使用较多的并发连接。

### 4.1 4 个 APIServer 共用 8000 端口

1. 主进程（`vllm serve`）在启动引擎之前，先用 `create_server_socket()` 创建并 bind 监听 socket（`setup_server`；源码注释说明是为了先占住端口）。源码里该 socket 设置了 `SO_REUSEPORT`，但实际分流不依赖它。
2. 该 socket 对象传给 `APIServerProcessManager`，4 个 APIServer 子进程通过继承拿到同一个监听 fd，在其上运行 uvicorn。
3. 4 个 APIServer 都在这个 socket 上 `accept()`，由内核决定哪个进程拿到新连接。主进程只持有该 fd，不处理请求。
4. 分配并不均匀：在双节点实验（同样是 4 个 APIServer 共用一个监听 socket，见 `vllm-dp4-tp1-two-node.md` 5.4）中，201 个新建连接的请求被分到 4 个 APIServer 的数量为 86 / 33 / 45 / 37；本单节点配置没有单独压测。

**实测验证**（`ss -ltnp` 加 `/proc/<pid>/fd` 的 inode 核对）：`0.0.0.0:8000` 只有一个监听 socket（inode 2879442），被 5 个进程同时持有，fd 都是 37：

| 进程 | PID | 持有该监听 fd |
|---|---|---|
| `vllm serve` 主进程 | 1353 | 是 |
| APIServer_0 / 1 / 2 / 3 | 1945 / 1946 / 1947 / 1948 | 是 |
| DPCoordinator、4 个 EngineCore | | 否 |

说明这是同一个 socket 被继承，不是 4 个进程各自 bind。

**内核层面的支持**：这是经典的多进程共享监听 socket 模型（nginx、Apache prefork、gunicorn 都在用）。

- 子进程继承的 fd 指向内核里同一个 socket 对象，所以 inode 相同。已完成三次握手的连接放在它的 accept 队列里，所有进程共享这一个队列；内核把每个连接只交给其中一个进程，用户态不需要加锁。
- 分配不是严格轮询：一般取决于谁先被唤醒，繁忙的进程少拿、空闲的多拿，各进程的连接数可能不均匀。
- 惊群：多进程阻塞在同一个 `accept()` 时，Linux 2.6 起已是独占唤醒。但 uvicorn（asyncio）阻塞的是 `epoll_wait` 而不是 `accept()`，epoll 场景下可能有轻微惊群，内核 4.5 起的 `EPOLLEXCLUSIVE` 可缓解；uvicorn / asyncio 是否启用它未核对。
- 与 `SO_REUSEPORT` 的区别：`SO_REUSEPORT` 是多个进程各自 `bind` 同一端口，每个进程有独立的 socket 和 accept 队列，内核按四元组哈希分流。本例实测是“一个 socket 被继承”，不是这种方案。

### 4.2 DPCoordinator 不负责分发请求

选择 DP rank 的是 APIServer：每个 APIServer 的 client 根据 Coordinator 发布的各 rank 队列长度，选出最空闲的 EngineCore，再通过 ZMQ 直接把请求发给它，不经过 Coordinator。Coordinator 做三件事：

1. 收集各引擎的 waiting / running 队列长度，发布给所有 APIServer；
2. 跟踪 DP 的“请求波次（wave）”和引擎的运行 / 暂停状态；
3. 引擎暂停时若有新请求到达，广播 `START_DP_WAVE` 唤醒引擎。

对稠密模型，第 2、3 项意义不大（它们是为 MoE 的 dummy batch 同步服务的），主要价值是第 1 项（负载统计）。

## 5. 进程间通信：ZMQ

**ZMQ（ZeroMQ）是什么**：一个消息通信库，不是独立服务进程，没有 broker。它在 TCP、Unix 域 socket（IPC）、进程内通道之上封装“消息队列式”socket：自动重连、异步收发、按消息边界传输。本环境为 pyzmq 27.1.0（libzmq 4.3.5）。

vLLM 源码中用到的 socket 类型：

| 类型 | 语义 |
|---|---|
| ROUTER / DEALER | 异步多对多请求通道，ROUTER 可按 identity 点对点寻址 |
| PUSH / PULL | 单向流水线，一端推、一端拉 |
| XPUB / XSUB | 发布 / 订阅，可广播给多个订阅者 |
| PUB / SUB | 发布 / 订阅，EngineCore 到 Worker 的唤醒通知用它（`shm_broadcast.py`） |

**在 vLLM 里的作用**：v1 是多进程架构，ZMQ 主要是 APIServer、EngineCore、DPCoordinator 之间的控制面和数据面通道（小消息：请求、token 结果、统计、控制命令）；EngineCore 到 Worker 的调度结果广播中，ZMQ 只负责唤醒通知，数据走共享内存。

| 通道 | 方向 | socket 类型（源码） |
|---|---|---|
| 请求下发 | APIServer → EngineCore | APIServer 侧 ROUTER，EngineCore 侧 DEALER |
| 结果回传 | EngineCore → APIServer | EngineCore 侧 PUSH，APIServer 侧 PULL |
| 负载统计 | EngineCore → DPCoordinator → APIServer | Coordinator 侧 PULL 收、XPUB 发；APIServer 和 EngineCore 侧 XSUB 订阅 |
| 启动握手 | EngineCore ↔ 主进程 | ROUTER（`v1/engine/utils.py`） |
| 调度结果的唤醒通知 | EngineCore → Worker | PUB/SUB，数据本身走共享内存（`distributed/device_communicators/shm_broadcast.py`） |

对应本例：
- 请求路径：APIServer 选好 DP rank 后，经 ROUTER/DEALER 通道把请求直接发给对应 EngineCore（一个 APIServer 同时对接 4 个 EngineCore，ROUTER 按 identity 指定目标）；生成的 token 经 PUSH/PULL 通道流回 APIServer，再通过 HTTP 流式返回客户端。
- 负载统计：EngineCore 上报队列长度，DPCoordinator 用 XPUB 广播给 4 个 APIServer。
- `START_DP_WAVE` 也是由 Coordinator 经 XPUB 通道广播给引擎。

**传输方式**：源码 `v1/utils.py` 的 `get_engine_client_zmq_addr`：同机（`local_only`）使用 IPC 地址（Unix 域 socket），跨机才用 TCP 地址。本例单节点走 IPC，pod 内可见 176 条与 vLLM 相关的 Unix socket 连接，与此吻合。

**与其他通信的区别**：

| 通信 | 传什么 | 在哪里用 |
|---|---|---|
| ZMQ | 小消息：请求、token 结果、统计、控制命令 | 进程之间（APIServer / EngineCore / DPCoordinator） |
| NCCL | GPU 张量 | TP 的 all-reduce，同一 DP rank 内的 2 张 GPU，走 NVLink |
| 共享内存 + ZMQ 通知 | 调度结果广播 | EngineCore → 它的 2 个 Worker：数据放在共享内存（`MessageQueue`），读者空闲时靠 ZMQ PUB/SUB 通知唤醒，避免一直轮询（源码核对） |

备注：pod 内可见 Worker 进程在 Pod IP 上监听了大量 TCP 端口，未确认其用途（可能是 torch.distributed 等），不确定是否属于 ZMQ。

## 6. 一个请求的完整处理流程

以 `curl ... "Who are you?"` 为例（prompt 12 个 token）：

```
Client
  │ ① HTTP :8000
  ▼
APIServer_k ── ② 分词/构造请求 ── ③ 选 DP rank（用 Coordinator 的统计）
  │ ④ ZMQ (ROUTER→DEALER)
  ▼
EngineCore_DPn ── ⑤ 调度（排队、batching、KV 块分配）
  │ ⑥ 共享内存 + ZMQ 通知
  ▼
Worker_TP0 ◀──NCCL all-reduce──▶ Worker_TP1   ⑦ 前向计算 + 采样
  │ ⑧ 新 token
  ▼
EngineCore_DPn ── ⑨ ZMQ (PUSH→PULL) ──▶ APIServer_k ── ⑩ 反分词，HTTP 返回
```

| 步骤 | 进程 | 做什么 |
|---|---|---|
| ① | 内核 → APIServer_k | 客户端连 8000 端口，内核把连接交给 4 个 APIServer 之一（记为 k）；该连接上的后续请求都由它处理 |
| ② | APIServer_k | 解析 JSON，套 chat template，分词成 token id，构造内部请求对象（含采样参数）。分词在 APIServer 做，不占 EngineCore 时间，这是需要多个 APIServer 的原因之一 |
| ③ | APIServer_k 的 client | 根据 DPCoordinator 经 XPUB 发布的各 rank 队列长度，选最空闲的 rank（记为 n）；若引擎处于暂停态，同时通知 Coordinator 广播 `START_DP_WAVE`（稠密模型形式上的步骤） |
| ④ | APIServer_k → EngineCore_DPn | 经 ZMQ ROUTER 通道按 identity 直接发给目标 EngineCore（本机走 IPC）；不经过 DPCoordinator，也不经过其他 3 个 EngineCore |
| ⑤ | EngineCore_DPn | 请求进入等待队列；调度器每步决定本步跑哪些请求（continuous batching），分配 KV cache 块；新请求先 prefill 整个 prompt，之后每步 decode 1 个 token；同一 rank 上的其他请求合并到同一个 batch |
| ⑥ | EngineCore_DPn → 2 个 Worker | 把本步 batch 信息广播给 Worker_TP0 / TP1（数据放在共享内存，读者空闲时用 ZMQ 通知唤醒；依据 `shm_broadcast.py` 源码） |
| ⑦ | Worker_TP0 + TP1 | 两个 Worker 同时在各自 GPU 上算一半模型；每层 Attention 和 FFN 后通过 NCCL 在 NVLink 上 all-reduce（36 层共 72 次，推算值非实测）；最后一层出 logits 并采样得到下一个 token。其他 3 个 DP rank 的 6 张卡不参与 |
| ⑧⑨ | Worker → EngineCore → APIServer_k | token 交给 EngineCore，再经 ZMQ PUSH 通道发回最初发来该请求的 APIServer_k；流式请求每步回传一次，非流式则攒到结束 |
| ⑩ | APIServer_k | token id 反分词成文本，检查停止条件，组装成 OpenAI 格式 JSON，通过原 HTTP 连接返回 |

⑤ 到 ⑨ 循环执行，直到遇到结束符或达到 `max_tokens`（测试时设为 64，所以在 64 个 token 处以 `finish_reason: length` 结束）。

要点：
- 这个请求的路径上只有 1 个 APIServer、1 个 EngineCore、2 个 Worker 共 4 个进程参与；主进程、resource_tracker、DPCoordinator、另外 3 个 APIServer 和 3 个 EngineCore 都不在路径上，DPCoordinator 只在 ③ 提供统计。
- 同一条连接固定对应一个 APIServer，但不同请求可以被分到不同的 DP rank。
- 各 rank 的 KV cache 相互独立，同一用户的多轮对话若被分到不同 rank，前缀缓存就用不上。

## 7. 观察与结论

- 每张 GPU 显存占用约 254660 MiB，接近 B300 的全部显存。原因是 `--gpu-memory-utilization` 默认 0.9，会预分配显存给 KV cache；8B 权重（bf16 约 16 GB，推算）在 TP=2 下每卡仅约 8 GB。
- 空闲时 GPU 利用率为 0%，有请求时才会有计算。
- 对 8B 稠密模型，单卡即可放下，TP=1、DP=8 吞吐更高（省去 TP 通信，推论，未实测）。TP=2 的配置主要用于演示 DP 与 TP 的组合。

## 8. 未验证项

以下内容来自对 vLLM 和 Linux 的了解或推算，没有在本环境核对：

- 本单节点配置下 APIServer 之间的连接分配情况（未单独压测；双节点实验的结果见 `vllm-dp4-tp1-two-node.md`）。
- uvicorn / asyncio 是否使用 `EPOLLEXCLUSIVE`。
- 分词发生在 APIServer 进程内。
- 每层 2 次 all-reduce、共 72 次的通信量（按标准 TP 切分推算）。
- TP=1、DP=8 吞吐更高的判断（未实测）。
- Worker 进程在 Pod IP 上监听的大量 TCP 端口的用途。
