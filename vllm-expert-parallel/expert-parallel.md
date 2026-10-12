# vLLM Expert Parallel 部署学习笔记

来源：https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/

## 1. 核心概念
- **Expert Parallelism (EP)**：把 MoE 模型的专家分散到不同 GPU，提升局部性、效率和吞吐。
- EP 通常与 **Data Parallelism (DP)** 搭配；DP 可单独使用，但 EP 配合 DP 效果最好。
- **EP 大小**：`EP_SIZE = TP_SIZE × DP_SIZE`
- 开启 EP 后各层行为：
  - **MoE 层**：专家在所有 EP rank 间分片。
  - **Attention 层**：TP=1 时权重在各 DP rank 上复制；TP>1 时在 DP 组内按 TP rank 分片。
  - 不开 EP 时，MoE 层与稠密模型一样，在 TP×DP 组内做张量并行。

## 2. 前置依赖
1. **DeepEP**：按 vLLM 的 EP kernels 指南（`tools/ep_kernels`）配置宿主机环境。
2. **DeepGEMM**：按 DeepSeek 官方说明安装。
3. **gdrcopy**（PD 分离场景）：运行 `install_gdrcopy.sh`，需指定 OS 版本、CUDA 版本和架构。
- **NCCL（CUDA 13+）**：`deepep_v2` 后端需要 NCCL ≥ 2.30.4，PyTorch 自带版本较旧，需先升级。

## 3. All-to-All 后端（`--all2all-backend`）
| 后端 | 场景 | 说明 |
|---|---|---|
| `allgather_reducescatter` | 默认 | 通用，适用于任意 EP+DP 配置 |
| `deepep_high_throughput` | 多节点 prefill | Grouped GEMM、连续布局，适合 prefill 为主 |
| `deepep_low_latency` | 多节点 decode | 支持 CUDA graph、masked 布局，适合 decode 为主、低延迟 |
| `flashinfer_nvlink_one_sided` | MNNVL 系统 | FlashInfer 单边 A2A，高吞吐 |
| `flashinfer_nvlink_two_sided` | MNNVL 系统 | FlashInfer 双边 A2A，跨节点 NVLink |
| `passthrough` | 仅内部使用 | 不能在 CLI 指定；`--moe-backend flashinfer_moe_ep_*` 时自动设置 |

## 4. 关键参数与环境变量

**参数：**
- `--enable-expert-parallel`：开启 EP
- `--all2all-backend`：选择通信后端
- `--tensor-parallel-size` / `--data-parallel-size`：TP / DP 大小（DP 为跨节点总数）
- `--data-parallel-size-local`：本节点的 DP rank 数
- `--data-parallel-start-rank`：本节点的 rank 起始偏移
- `--data-parallel-address` / `--data-parallel-rpc-port`：主节点地址与 RPC 端口
- `--headless`：只跑 worker，不启动 API server
- `--api-server-count`：主节点 API server 数量
- `--enable-eplb` / `--eplb-config`：负载均衡
- `--enable-dbo`：Dual Batch Overlap
- `--async-scheduling`：实验性，调度与执行重叠
- `--kv-transfer-config`：PD 分离的 KV connector 配置
- `--compilation_config`：编译配置（含 CUDA graph 模式）

**环境变量：**
- `GLOO_SOCKET_IFNAME=eth0`：InfiniBand 集群上强制 torch distributed 使用以太网做发现，避免初始化卡住
- `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random`：模拟均衡路由（基准测试用）
- `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1`：随机化 DP dummy 输入（基准测试用）

## 5. 单节点部署
示例：8 卡 H200/H20 上跑 DeepSeek-V3-0324（H100 建议换小模型或用多节点）。

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel
```

## 6. 多节点部署
每个节点运行一条命令：第一个节点处理请求，其余节点 headless。示例：2 节点 × 8 GPU，`deepep_low_latency`。

```bash
# Node 1（主节点，处理请求）
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --all2all-backend deepep_low_latency \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-address 192.168.1.100 \
    --data-parallel-rpc-port 13345 \
    --api-server-count=8

# Node 2（从节点，headless）
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --all2all-backend deepep_low_latency \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-start-rank 8 \
    --data-parallel-address 192.168.1.100 \
    --data-parallel-rpc-port 13345 \
    --headless
```

要点：
- 从节点 headless，所有客户端请求走主节点。
- `--data-parallel-start-rank` = 之前各节点本地 DP 数之和。
- 主节点按需增大 `--api-server-count`（建议扩到与本地 rank 数相当）。
- 多节点下 EPLB 参数要加在每个节点的命令上。
- 文档**没有** Ray / K8s 部署说明；K8s 仅在排障中提到：InfiniBand 场景下 pod 需 `hostNetwork: true` 和 `securityContext.privileged: true`。

## 7. EPLB（专家负载均衡）
真实 token 路由往往倾斜，EPLB 在每次前向收集负载统计并周期性重排专家映射。

`--eplb-config` 参数（JSON）：
| 参数 | 说明 | 默认 |
|---|---|---|
| `window_size` | 用于重平衡决策的统计窗口（engine steps） | 1000 |
| `step_interval` | 每 N 步重平衡一次 | 3000 |
| `log_balancedness` | 记录 平均每专家 token 数 ÷ 最大每专家 token 数 | false |
| `num_redundant_experts` | 每个 EP rank 额外的全局专家数 | 0 |
| `use_async` | 非阻塞 EPLB，降低延迟开销 | true |
| `policy` | 负载均衡策略 | "default" |
| `communicator` | 专家权重传输后端：`torch_nccl` / `torch_gloo` / `pynccl` / `nixl` / `null`（自动） | null |

分布公式：
- 默认：每 rank 专家数 = `NUM_TOTAL_EXPERTS ÷ NUM_EP_RANKS`
- 含冗余：`(NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS`

显存开销：`NUM_MOE_LAYERS * BYTES_PER_EXPERT * (NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS`。冗余专家占用显存、挤占 KV cache，显存紧张的场景不太适合；DeepSeek-V3 每 rank 1 个冗余专家约 2.4 GB。大规模部署建议 `{"num_redundant_experts":32}`，保证热门专家可用。

```bash
# JSON 形式
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}'

# 分项参数形式
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-eplb \
  --eplb-config.window_size 1000 \
  --eplb-config.step_interval 3000 \
  --eplb-config.num_redundant_experts 2 \
  --eplb-config.log_balancedness true

# 单节点 + EPLB
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel \
    --enable-eplb \
    --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}'
```

## 8. 性能、排障与基准测试
**性能：**
- DeepEP 的 `high_throughput` / `low_latency` kernel 面向 PD 分离场景，混合负载下可能表现不佳。
- `--enable-dbo`：让 all-to-all 通信与计算重叠。
- `--async-scheduling`（实验性）：调度与模型执行重叠。

**排障：**
- `non-zero status: 7 cannot register cq buf`：IB/RoCE 上确认宿主机和 pod 内 `ulimit -l` 为 unlimited。
- `init failed for transport: IBGDA`：缺少 InfiniBand GDA 内核模块；在每个 GPU 节点运行 `tools/ep_kernels/configure_system_drivers.sh` 并重启（同样可解决 NVSHMEM 初始化错误）。
- NVSHMEM peer disconnect：通常是网络配置问题。

**基准测试：** 设置 `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random` 与 `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1`，使 token 均匀路由到各 EP rank。

## 9. PD 分离（Prefill/Decode Disaggregation）
适用于对 TTFT / ITL 有严格 SLA 的场景，prefill 与 decode 可独立扩缩。
- **Prefill 实例**：`deepep_high_throughput`
- **Decode 实例**：`deepep_low_latency`
- **KV cache 传输**：NIXL 或其他 KV connector

步骤：
1. 安装 gdrcopy、UCX、NIXL。gdrcopy 可选，不装时 `pip install nixl` 也能用，性能较低。
2. 两个实例都配置 KV connector：
   ```bash
   --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
   # 多 NIXL 后端
   --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["UCX","GDS"]}}'
   ```
3. 由客户端编排（文档提供基于 OpenAI SDK 的 Python 示例）：
   - 分别创建 prefill / decode 客户端；
   - 向 prefill 发请求，`max_tokens=1` 且启用 `do_remote_decode`（prompt 需超过 16 token 的 block size）；
   - 把返回的 `kv_transfer_params` 连同相同 request ID 传给 decode 实例生成输出。
   - 路由层仍在开发中。

**单独测 decode：** 用 `DecodeBenchConnector`，用随机值填充 KV cache：
```bash
--kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both"}'
# 启动时一次性填满 cache：加 "kv_connector_extra_config":{"startup_fill":true}
# 仅 decode 抓 CUDA graph：
--compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```
