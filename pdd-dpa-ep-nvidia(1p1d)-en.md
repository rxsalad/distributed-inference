<a id="top"></a>
# 1P1D: Prefill and Decode Both Using DEP (DP Attention + EP) — PD Disaggregation on NVIDIA B300x8

DeepSeek-V3 (FP8) on a B300x8 (rail-optimized) cluster with Prefill/Decode disaggregation: **1 Prefill node + 1 Decode node**, each an 8-GPU DEP group (DP8 + EP8); the KV cache is transferred from Prefill to Decode via NIXL; traffic enters through a vllm-router running on a CPU Pod.


This document has three parts: **principles** (Chapter 2), **deployment and configuration** (Chapter 3), and **observation and measurement** (Chapters 4 and 5). Chapters 6 and 7 summarize the verification results for inferred claims and what remains unverified.

- Command source: [recipes.vllm.ai DeepSeek-V3, Prefill/Decode Disaggregation, prefill_mode=dep, decode_mode=dep](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V3?hardware=b300&prefill_mode=dep&prefill_nodes=2&decode_mode=dep&decode_nodes=2).
- Annotation convention: **Page** = text from the recipes page; **Observed** = phenomena seen on this cluster; **Source-verified** = checked against the vLLM 0.31.0 source inside the image; **Inferred** = derived from mechanism, not verified. All data in Chapter 5 was collected on 2026-10-13 from the running 1P1D deployment.

<a id="toc"></a>
## Table of Contents

- [1. Summary](#s1)
- [2. Principles](#s2)
  - [2.1 Overall structure and request path](#s2-1)
  - [2.2 DEP: 8 DP ranks × 8 EP ranks](#s2-2)
  - [2.3 How one MoE layer is computed; why dummy batches exist; synchronization](#s2-3)
  - [2.4 Prefill vs. Decode under DEP; comparison with TEP](#s2-4)
  - [2.5 KV transfer mechanics (NIXL + UCX)](#s2-5)
- [3. Deployment and configuration](#s3)
  - [3.1 Files](#s3-1)
  - [3.2 Prerequisites](#s3-2)
  - [3.3 Usage and switches](#s3-3)
  - [3.4 The Prefill / Decode pools](#s3-4)
  - [3.5 Environment variables and rationale](#s3-5)
  - [3.6 Router](#s3-6)
  - [3.7 Runtime environment record](#s3-7)
- [4. Observation methods](#s4)
  - [4.1 `./pdd.sh metrics [window seconds]`](#s4-1)
  - [4.2 Limitations of logs](#s4-2)
  - [4.3 NICs, memory regions, and GPUs](#s4-3)
  - [4.4 Processes and threads](#s4-4)
- [5. Measurements](#s5)
  - [5.1 Process structure](#s5-1)
  - [5.2 Idle: GPUs at 0%, but each Worker burns a full CPU core](#s5-2)
  - [5.3 Single request: all GPUs participate, no wake-up latency](#s5-3)
  - [5.4 KV transfer: a large 131072-token KV (P → D)](#s5-4)
  - [5.5 Topology and model configuration](#s5-5)
  - [5.6 EP collective communication goes over NVLink](#s5-6)
  - [5.7 KV transfer under concurrency](#s5-7)
- [6. Verification summary for inferred claims](#s6)
- [7. Unverified items and cautions](#s7)
- [8. Appendix: raw output and statistics scripts](#s8)
  - [A.1 `pdd-prefill-0`](#a1)
  - [A.2 `pdd-decode-0`](#a2)
  - [A.3 Full command line of PID 1 (`/proc/1/cmdline`)](#a3)
  - [A.4 NVLink / RDMA traffic statistics scripts](#a4)

<a id="s1"></a>
## 1. Summary

[↑ Back to top](#top)

**What this is**: the Prefill pool and the Decode pool are each an 8-GPU DEP group (every GPU is simultaneously a DP rank and an EP rank); the only thing between the two pools is the NIXL point-to-point KV channel.

**Key findings (all measured on 1P1D; details in Chapter 5)**

| Topic | Finding |
|---|---|
| GPU behavior for a single request | Send just 1 request and all 8 GPUs of the serving pool rise to 99–100% (ranks without requests co-run dummy batches); when idle, all are 0% |
| Where EP collective communication goes | **NVLink**: idle NVLink traffic is 0; with one request, about 278 GiB flows between the 8 Prefill GPUs (32768 tokens); no collective traffic on RoCE (5.6) |
| KV transfer method | **RDMA READ, initiated by Decode**, pulling directly from Prefill's GPU memory; Prefill's CPU/GPU take no part in the transfer |
| Which NIC a transfer uses | Each transfer uses exactly **one** NIC on each side, the one closest to that rank's own GPU (rank r ↔ `mlx5_{2r}`) |
| Cross-rail or not | The Prefill/Decode pairings given by the router are fixed within one launch at 8 combinations (not 8×8 = 64); pairings change across router restarts: in the first launch all index pairs differed — all cross-rail; in the later redeployment 1 of the 8 pairs was same-rail (P6→D6). No speed difference observed between cross-rail and same-rail transfers (~48 GB/s, 5.7) |
| Transfer performance | A 9.21 GB (131072-token) KV takes ~0.19 s, about 46–48 GB/s, wire peak ~50 GB/s, close to the 400 Gb/s line rate of a single NIC |
| Share of TTFT spent on transfer | ~0.5% (TTFT ~36 s, of which prefill compute is 34–35 s) |
| KV transfer under concurrency | 24 requests of 32768 tokens, concurrency 8: NIXL-recorded transfer time rises from ~49 ms (single request) to 57–65 ms (+16–34%); wire speed unchanged (burst median 60 ms, ~39 GB/s); **there is no collective communication on RoCE in 1P1D, yet transfer time still grows, so the growth is not caused by collective communication** (5.7) |
| Concurrent throughput | The same 24 requests, 1P1D 28.6 s (0.84 req/s, mean TTFT 6.85 s, TPOT 17.4 ms; 5.7) |
| Memory regions | Each Worker registers **2** identically sized large memory regions on its own NIC, each equal to its entire KV cache; one transfer consists of a number of descriptors (one contiguous run of blocks within one layer), ranging from 61 to 249,856 |
| Idle CPU | Each Worker burns one full core at idle; this is **not** dummy batches — more likely the NIXL UCX progress thread busy-polling |
| First-request latency | The first request after ~28 minutes of idle showed no extra latency |

<a id="s2"></a>
## 2. Principles

[↑ Back to top](#top)

<a id="s2-1"></a>
### 2.1 Overall structure and request path

```
                    Client
                      │  :30000
               ┌──────▼──────┐
               │ pdd-router  │  CPU Pod, vllm-router (PD mode, round_robin)
               └──┬───────┬──┘
   request / first token │       │ request / decode
        ┌───────────┘       └────────────┐
┌───────▼──────────────┐        ┌────────▼─────────────┐
│ pdd-prefill-0        │  NIXL  │ pdd-decode-0         │
│ 1 GPU node, 8 GPUs   │ ◄───── │ 1 GPU node, 8 GPUs   │
│ kv_producer, :8001   │  READ  │ kv_consumer, :8002   │
│ DP8 + EP8            │        │ DP8 + EP8            │
└──────────────────────┘        └──────────────────────┘
```

- Uses **2 GPU nodes + 1 CPU node** in total.
- A request's path: client → Router → Prefill (compute the prompt, generate the first token) → Decode **pulls** this request's KV from Prefill via NIXL → Decode continues generating.
- Prefill and Decode are **two independent DP groups**, each with its own DP coordinator (`DPCoordinator`, inside the Pod), not connected to each other.
- Each pool is hybrid-lb: there are 8 API servers inside the Pod, load-balancing across the 8 local DP ranks. The router treats each Pod as one backend; `--intra-node-data-parallel-size 8` tells it each backend has 8 DP ranks, and it selects backends **per DP rank** (in logs, e.g. `prefill=http://<ip>:8001@N`).

<a id="s2-2"></a>
### 2.2 DEP: 8 DP ranks × 8 EP ranks

`--data-parallel-size 8 --enable-expert-parallel`, with TP left at the default 1: 8 GPUs, each taking on two roles at once.

| | Attention part (DP) | MoE part (EP) |
|---|---|---|
| Unit | 8 DP ranks, one GPU each | 8 EP ranks, one GPU each; the 8 ranks form **one EP group** |
| Weights | **a full copy of the Attention weights on every GPU** (MLA, the first 3 dense-FFN layers, gates, shared expert, etc., unsharded because TP=1; of these, the dense FFN and shared expert are Inferred) | **the 256 routed experts spread across 8 GPUs, 32 per GPU**, no duplication |
| What they process | each rank **independently** processes the requests assigned to it, with its own EngineCore (scheduler) and KV cache | every rank processes the tokens from **all DP ranks** that select its own 32 experts |
| Inter-rank communication | **none** | yes: collective communication at every MoE layer (2.3) |
| Processes | `EngineCore_DPn` (one per rank) and its `Worker_DPn_EPn` | the same `Worker_DPn_EPn` (DP and EP indices match) |

Key points:
- **"Full weights" applies only to the Attention side**: the whole model's weights are not replicated per GPU; experts are sharded by EP, 1/8 per GPU.
- **No KV cache duplication**: each rank stores only the KV of its own requests; the 8 ranks' capacities add up to the total; in a TP scheme each GPU holds a full copy.
- Per-GPU memory = full Attention weights + 32 experts + its own KV + activation reserve.
- **The EP group is entirely intra-node; collective communication goes over NVLink** (verified): the 8 GPUs are pairwise `NV18` (full NVLink interconnect); NVLink counters are 0 when idle and increase massively with a request; during a request there is no collective traffic on the RoCE NICs, only NIXL's KV READ (Sections 5.3, 5.4, 5.6). In 1P1D, RoCE is used only for KV transfer.

<a id="s2-3"></a>
### 2.3 How one MoE layer is computed; why dummy batches exist; synchronization

DeepSeek-V3 has 61 layers in total: the first 3 are dense FFN, the remaining 58 are MoE (checked against `config.json`). The flow of one MoE layer (the all2all backend is the default `allgather_reducescatter`, source-verified):

```
On each DP rank:  Attention (independent, no communication)
                   → routing gate: this rank's tokens each pick 8 experts (out of 256)
                   → all_gatherv: tokens from the 8 ranks are gathered; every rank ends up with all tokens (dispatch)
                   → each rank computes only the contribution of its own 32 experts for "all tokens"
                   → reduce_scatterv: the ranks' results are summed; each rank takes back only the outputs of its own tokens (combine)
                   → shared expert (computed locally), next layer
```

- The communication is **NCCL collectives**: one `all_gatherv` + one `reduce_scatterv` per MoE layer, done once by each of the 58 layers per forward step. Attention and dense layers have no inter-rank communication, so DEP's communication cost appears only at MoE layers.
- This backend **broadcasts all tokens to all ranks**, so traffic grows with the number of ranks — not as frugal as a true all2all that sends only to the target experts; DeepEP is unavailable on B300 (sm_103), so the default backend is the only option.

**Why ranks without requests must "co-run" (dummy batch)**

EP splits every MoE layer across all ranks: a rank's experts compute for **other ranks' tokens**, so every MoE layer is a group-wide collective, and all ranks in the group must execute the **same number of forward steps**. But DP keeps each rank's request stream independent, so a rank with no requests would otherwise not need to step at all. So as soon as one rank in the group is serving a request, the others must run a **dummy batch** (an empty forward with just 1 token, but going through all layers, including the collectives of the 58 MoE layers; the source has `execute_dummy_batch`).

- **Scope**: co-running stems from MoE's cross-rank dependency, and its scope is the entire EP group: in 1P1D, when the Prefill pool has one request, all 8 Prefill GPUs are busy; likewise for the Decode pool (verified, Section 5.3).
- **Backend-independent; dense models don't co-run**: these two points are inferred from mechanism, not verified.
- **Impact on observation**: `nvidia-smi` GPU utilization becomes misleading — a single request drives all 8 GPUs above 99%; it cannot be used to judge effective load.
- **Note the distinction**: 8-GPU co-running during a request (GPU utilization) is verified; but the Worker burning a full CPU core at idle is **not** dummy batches spinning (GPUs are 0% when idle, Section 5.2).

**Synchronization mechanisms** (source-verified, vLLM 0.31.0)

| Layer | Frequency | What it does | Channel |
|---|---|---|---|
| MoE-layer collectives | every step × 58 layers × 2 (gather + scatter) | exchange tokens; each rank computes its own experts | NCCL |
| DP metadata sync per forward step | once per step | a small int32 all-reduce: each rank's token count, CUDA graph mode, whether DP padding happens; with CUDA graphs on, all ranks' token counts are padded to the same maximum | with async scheduling this goes over **Gloo on CPU** (TCP via eth0). The log has `Disabling NCCL for DP synchronization when using async scheduling.` (Observed); the Gloo path is a source-code conclusion |
| Wave start/stop consensus | every `dp_sync_interval` steps, default 16 (checked in `config/parallel.py`) | a global all-reduce to decide whether any rank still has unfinished requests | same as above |

`DPCoordinator` (one per pool) coordinates waves and load statistics: the first request after the pool goes fully idle notifies it first; it broadcasts the start, and only then do the engines begin stepping. This is also why the Prefill and Decode pools don't affect each other: each has its own coordinator and its own waves.

**What a wave is** (per the description in `v1/engine/coordinator.py`, vLLM 0.31.0)

- All DP engines in a pool advance and retire together, alternating between the global **running (stepping)** and **paused** states. One running → paused group transition counts as completing a **wave**; the wave number is the count of such transitions (starting from 0).
- **Start**: when a new request arrives while engines are paused, the frontend also notifies `DPCoordinator`, which broadcasts `START_DP_WAVE` to all engines in the pool and the whole group enters running together; an engine that receives a request carrying a stale wave number also notifies it.
- **End**: during running, a global all-reduce happens every `dp_sync_interval` steps; when no rank has unfinished requests, the whole group returns to paused together and the wave number increments.
- **Relation to dummy batches**: dummy batches occur only in the running state; within a wave, as long as one rank has a request, the others co-run; when paused, no rank steps (idle GPUs at 0%, Section 5.2).
- **Unverified**: the above comes from source commentary; the change of wave numbers over time was not observed live; the latency impact of `dp_sync_interval` (at most a few extra empty steps) is also only inferred.

<a id="s2-4"></a>
### 2.4 Prefill vs. Decode under DEP; comparison with TEP

| | Prefill pool (`kv_producer`) | Decode pool (`kv_consumer`) |
|---|---|---|
| What each DP rank does | forward pass of the assigned request's prompt, generate the first token, leave the KV for NIXL to pull | pull the KV, generate token by token |
| Per-step character | many tokens (long prompt), compute-bound; a long prefill on one rank makes the other 7 ranks wait at every MoE layer | few tokens per step; collective communication takes a large share of every MoE-layer step, latency-sensitive; uses `FULL_DECODE_ONLY` CUDA graphs |
| Effect of load imbalance | the slowest rank sets the pace for the whole group (Inferred, not measured) | same |

**Comparison with TEP**

- **DEP** (Data-parallel attention + Expert Parallel, this setup): Attention is sharded by DP, each rank processing its own requests; MoE is sharded by EP, experts distributed across all ranks.
- **TEP** (Tensor-parallel attention + Expert Parallel): Attention is sharded by TP, one request computed jointly by all 8 GPUs; MoE likewise sharded by EP. The MoE parts are identical; the difference is Attention and request organization.

| | DEP (this setup) | TEP |
|---|---|---|
| Attention | 8 ranks each compute their own requests, no communication | 8 GPUs jointly compute one request, all-reduce every layer |
| KV cache | each rank stores only its own requests' KV, no duplication | an MLA model has a single latent head, which TP cannot shard — a full copy per GPU |
| MoE | EP8, all-gather + reduce-scatter per layer | EP, plus an additional all-reduce within the TP group |
| Idle ranks | must co-run dummy batches | a single EngineCore, no "co-running" |
| Best for | high concurrency, throughput-oriented | low concurrency, single-request latency-oriented |

Pure TP (MoE expert weights also sharded by TP) differs from TEP only in MoE; not expanded here.

**Verification status**: this comparison comes from mechanism and source inference; only DEP was deployed in this environment — **TEP/TP was never run, no measured comparison exists**; the "best for" row is conventional wisdom, with no data in this document to support it.

<a id="s2-5"></a>
### 2.5 KV transfer mechanics (NIXL + UCX)

- **Pull-based**: after Prefill finishes, Decode's Worker issues a `READ` through NIXL (source `kv_connector/v1/nixl/pull_worker.py`, `make_prepped_xfer("READ", ...)`), reading directly from Prefill's GPU memory; completion is signaled to the peer with a notification message. The underlying transport is RDMA READ; Prefill's CPU and GPU take no part in moving the data (verified, Section 5.4).
- **Rank to rank**: a request lands on some rank of Prefill and some rank of Decode; the KV stays on that Prefill GPU until the transfer; the two ranks' indices need not match.
- **Memory registration**: each Worker registers its entire KV cache (one contiguous block of GPU memory) as an RDMA memory region, and only on **its own NIC**; the log line is `Registering KV_Caches. use_mla: True, kv_buffer_device: cuda, use_host_buffer: False`. Transfers therefore naturally land on that NIC.
- **Side-channel port (handshake)**: the first time Decode pulls KV from a given Prefill engine, it must first obtain the peer's NIXL agent metadata (memory-region registrations, UCX addresses, etc.); this goes over a **side channel** separate from the KV data path. Each engine (each DP rank) starts a ZMQ ROUTER listener thread bound to `VLLM_NIXL_SIDE_CHANNEL_HOST:PORT`; Decode connects with a ZMQ REQ, sends `GET_META_MSG` (carrying the target pp rank and tp rank), the peer replies with the metadata, and Decode then calls `add_remote_agent`, after which RDMA READs can be issued. The reply also carries the peer's `perf_counter`, used to estimate the clock offset between the two ends (needed for KV block expiry). The handshake happens only on first connection; KV data goes over UCX/RoCE and never through this port again (source `kv_connector/v1/nixl/base_scheduler.py`, `base_worker.py`, vLLM 0.31.0).
  - **Port = base port + `data_parallel_index`**: one port per DP rank. Base ports: 5557 for Prefill, 5558 for Decode. Verified: in `pdd-prefill-0`, `ss -ltn` shows 8 ports 5557–5564 listening on the Pod IP; the Decode pool's range is inferred from source as 5558–5565, not individually checked.
  - **The address must be this Pod's IP, reachable from other Pods** (`POD_IP` in this deployment); `localhost` would make the peer connect back to itself.
- **descriptor**: one transfer consists of a number of descriptors; a descriptor is "one contiguous run of blocks within one layer"; the more contiguous the block allocation on both sides, the fewer descriptors. Block size is 32 tokens; each block is 36,864 B per layer (32 × 576 × 2 B).
- **KV per token**: MLA, (512 + 64) × 2 B × 61 layers = 70,272 B ≈ 70.3 KB (`kv_lora_rank=512`, `qk_rope_head_dim=64`); 32768 tokens ≈ 2.30 GB, 131072 tokens ≈ 9.21 GB. Use this to cross-check transfer bytes in `metrics`.

<a id="s3"></a>
## 3. Deployment and configuration

[↑ Back to top](#top)

<a id="s3-1"></a>
### 3.1 Files

| File | Purpose |
|---|---|
| `pdd-dep.yaml` | envsubst template: one StatefulSet + headless Service each for Prefill/Decode; the Router's ServiceAccount/Role/RoleBinding, Service, StatefulSet |
| `pdd-launch.sh` | shared launch script for Prefill/Decode, mounted at `/scripts` via the ConfigMap `pdd-scripts` |
| `pdd-router.py` | the Router's launcher, likewise mounted via ConfigMap |
| `pdd.sh` | control script: `up` / `down` / `status` / `render` / `test` / `bench` / `metrics` |
| `pdd-dap-ep-nvidia(1p1d).md` | this document |

All files except this document are in English. The scripts live in a ConfigMap: after editing a script, `up` again; already-running Pods must be restarted to pick up the new script.

> **File locations**: this document lives directly under `distributed-inference/`; all files it references — `pdd-dep.yaml`, `pdd-launch.sh`, `pdd-router.py`, `pdd.sh`, and the `traffic-tools/` scripts of A.4 — are in the `pdd-dpa-ep-nvidia/` subdirectory (alongside the Chinese original of this document).

<a id="s3-2"></a>
### 3.2 Prerequisites

- At least 2 B300×8 GPU nodes (`gpu-brand=nvidia`, with the `nvidia.com/gpu` taint), multus + `roce-net-fabricN` network attachments, `rdma/fabricN` device-plugin resources; the vLLM image 0.31.0 (the version `latest` resolved to); at least 1 non-GPU node for the Router.
- Model `deepseek-ai/DeepSeek-V3`; each node downloads it to its local hostPath cache on first launch.
- `pdd.sh up` checks that `PREFILL_NODES + DECODE_NODES` does not exceed the cluster's GPU node count.

<a id="s3-3"></a>
### 3.3 Usage and switches

```bash
PREFILL_NODES=1 DECODE_NODES=1 ./pdd.sh up      # deploy 1P1D (defaults are 2 and 2 when unset)
./pdd.sh status                                  # Pods and Services
./pdd.sh test                                    # one request through the Router, end-to-end check
./pdd.sh bench [extra vllm bench serve args]    # load test through the Router; output also written to benchmark-<timestamp>.log
./pdd.sh metrics [window seconds]               # summarize P/D /metrics (Section 4.1)
./pdd.sh render                                  # render the yaml only
./pdd.sh down                                    # delete (HF cache kept)
```

Pod names are fixed: `pdd-prefill-0`, `pdd-decode-0`, `pdd-router-0`. Optional environment variables (after changing them: `down` then `up`):

| Variable | Values | Default | Notes |
|---|---|---|---|
| `PREFILL_NODES`, `DECODE_NODES` | positive integers | 2, 2 | for 1P1D set them explicitly to 1, 1 |
| `IMAGE` | image | `vllm/vllm-openai:latest` | shared by all three workloads; `latest` may resolve to different versions at different times — use a pinned tag for consistency |
| `MODEL` | model name | `deepseek-ai/DeepSeek-V3` | |
| `PREFIX_CACHING` | `on` / `off` | `off` | `off` adds `--no-enable-prefix-caching` to P and D. vLLM 0.31.0 enables prefix caching by default; cache hits distort TTFT and KV transfer volume when testing PD disaggregation; P and D must match |
| `HIDE_HEALTH_LOG` | `on` / `off` | `off` | `on` adds `--disable-access-log-for-endpoints /health,/metrics,/ping` to P and D, keeping the `POST /v1/...` logs |
| `UCX_LOG_LEVEL` | `fatal\|error\|warn\|info\|debug\|trace` | `info` | applies to P and D only; `info` shows which transport, device, and rail UCX picked — set `warn` if it is too noisy |

`bench` defaults: 32768 input / 128 output / 1 request / request-rate 1 / seed 1, overridable with `BENCH_INPUT_LEN`, `BENCH_OUTPUT_LEN`, `BENCH_NUM_PROMPTS`, `BENCH_REQUEST_RATE`, `BENCH_SEED`, `BENCH_LOG`; arguments after `bench` are passed through to `vllm bench serve`.

**Warm-up is mandatory before testing**: the first long prefill on each DP rank incurs one-time overhead; the first run's numbers are meaningless and excluded from analysis. Before any performance test, send a batch of long requests through the Router (input no shorter than the real test) so all 8 Prefill ranks and all 8 Decode ranks have served at least once, e.g. `BENCH_NUM_PROMPTS=16 BENCH_REQUEST_RATE=inf ./pdd.sh bench --max-concurrency 8`, and confirm with `kubectl logs pdd-router-0 | grep "Selected prefill"` that every Prefill rank (`@0`..`@7`) has appeared; the Router's P→D pairings are fixed (Section 5.4), so covering all Prefill ranks covers all Decode ranks too. Timing data collected before warm-up is unusable; NVLink and NIC byte counters are unaffected. When the timing data in Chapter 5 of this document was collected, every rank involved had served requests before (verifiable in the Router logs), but no per-rank cold-vs-warm comparison was done on 1P1D.

**All tests run inside `pdd-router-0`**: `test`, `bench`, and `metrics` are all `kubectl exec pdd-router-0` — never entering the Prefill/Decode Pods — so the test client does not consume GPU-node CPU or disturb P/D. The Router Pod uses the same image as P/D, which contains `vllm bench`; the tokenizer downloads from HF on first use and is cached on the CPU node's hostPath. Reading hardware counters (NICs, `nvidia-smi`) does require entering the P/D Pods — keep it read-only and lightweight.

<a id="s3-4"></a>
### 3.4 The Prefill / Decode pools

**Launch (`pdd-launch.sh`)**: both roles share one script; the only differences come from environment variables.

| | Prefill | Decode |
|---|---|---|
| StatefulSet / headless Service | `pdd-prefill` / `pdd-prefill-headless` | `pdd-decode` / `pdd-decode-headless` |
| `KV_ROLE` | `kv_producer` | `kv_consumer` |
| API port | 8001 | 8002 |
| NIXL side-channel port | 5557 | 5558 |
| Extra args | — | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` (Page: the Decode pool additionally uses decode-only CUDA graphs) |

With 1P1D, `NODES=1`, so `DP_SIZE=8`, `START_RANK=0`, and the Pod is its own Head (ordinal `-0`). Steps:

1. Resolve the Head address `<ROLE_APP>-0.<ROLE_APP>-headless`, looping until a non-empty IP comes back (DNS has a transient race: `getent` can succeed, yet the IP fetched immediately afterwards is empty). The headless Service sets `publishNotReadyAddresses: true`, so the name resolves even before the Pod is Ready.
2. `hf download ${MODEL}`: the hostPath cache is per-node; returns immediately if already cached.
3. Set the NIXL side channel: `VLLM_NIXL_SIDE_CHANNEL_HOST=${POD_IP}` (for the handshake; must be a Pod IP reachable from other Pods), `VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_PORT}`.
4. Append args per switch: `PREFIX_CACHING=off` → `--no-enable-prefix-caching`; `HIDE_HEALTH_LOG=on` → `--disable-access-log-for-endpoints ...`; invalid values abort with an error.
5. `exec vllm serve`:

```
vllm serve ${MODEL} --trust-remote-code --host 0.0.0.0 --port ${HTTP_PORT}
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"<kv_producer|kv_consumer>","kv_load_failure_policy":"fail"}'
  --data-parallel-size 8 --data-parallel-size-local 8 --data-parallel-start-rank 0
  --data-parallel-address ${HEAD_IP} --data-parallel-rpc-port 13345 --data-parallel-hybrid-lb
  --enable-expert-parallel
  --enable-auto-tool-choice --tool-call-parser deepseek_v3 --reasoning-parser deepseek_v3
```

**Pod spec**
- Scheduled onto nodes labeled `doks.digitalocean.com/gpu-brand: nvidia`, tolerating the `nvidia.com/gpu` `NoSchedule` taint; `podManagementPolicy: Parallel`.
- Each Pod requests 8 `nvidia.com/gpu` and 8 RoCE device resources `rdma/fabric{0,2,…,14}` (10 each), and attaches `roce-net-fabric{0,2,…,14}` via multus annotations.
- `IPC_LOCK` capability; `/dev/shm` as a 128Gi memory volume; hostPath `/root/.cache/huggingface` as the model cache.
- Probes: startupProbe period 60 s, failureThreshold 60 (waits up to ~1 hour; startup itself takes 15–25 minutes); readinessProbe period 60 s, timeout 5 s, failureThreshold 3. The 60-second period (10 s is the common choice) reduces `GET /health` access logs inside vLLM; the cost is that Pod Ready can lag by up to 1 minute and a failure can take up to 3 minutes to be noticed.

<a id="s3-5"></a>
### 3.5 Environment variables and rationale

| Variable | Value | Rationale / status |
|---|---|---|
| `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME` | `eth0` | control plane / bootstrapping via the Pod NIC |
| `NCCL_IB_GID_INDEX` | `3` | **Observed** (from earlier multi-node NCCL deployments on this cluster): otherwise NCCL picks the link-local RoCE GID and multi-node hangs silently |
| `UCX_IB_GID_INDEX` | `3` | same reasoning as `NCCL_IB_GID_INDEX`, making UCX pick the right RoCE GID. **Unverified** as a variable in its own right: UCX establishes the RDMA backend and transfers work, but no A/B without it was done |
| `UCX_NET_DEVICES` | `mlx5_0:1,mlx5_2:1,…,mlx5_14:1` | **Observed**: the `fabricN` NICs sit in a VRF, so UCX's default enumeration tries `tcp/fabricN` and fails to bind (`bind ... Cannot assign requested address` → `NIXL_ERR_BACKEND`). Listing only the 8 RoCE devices (GPU i maps to `mlx5_{2i}`, on the same PCIe bridge), **leaving no `eth0` TCP fallback**, otherwise KV transfer silently falls back to TCP when RDMA is unavailable |
| `UCX_MAX_RNDV_RAILS` | `1` | at most 1 rail per rendezvous transfer (UCX default 2, max 4); each worker uses its own NIC, and 8 workers together cover all 8 NICs. **Observed**: each transfer uses 1 NIC per side (Section 5.4), consistent with this, but no A/B without it |
| `UCX_IB_TRAFFIC_CLASS` | `104` | DSCP / traffic class for RoCE traffic, carried over from an NIXL PD deployment on AMD GPUs; whether the switches classify by it — and whether it takes effect — cannot be verified in this environment |
| `UCX_LOG_LEVEL` | see 3.3 | default `info`, used to verify which path KV transfers take |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `3600` | initialization is slow; relaxes the engine-ready timeout |

**Differences from the recipes-page command**: node counts 1P+1D (DP8+EP8); added `NCCL_IB_GID_INDEX`, `VLLM_ENGINE_READY_TIMEOUT_S` and the `UCX_*` variables above; dropped `--chat-template` (the relative path in the image is unverified); the NIXL side-channel address uses the Pod IP (downward API) instead of the page's `$PREFILL_NODE_n`; the Router doesn't use the page's hardcoded URLs — `pdd-router.py` discovers Pods from K8S; added the `PREFIX_CACHING` (default `off`, opposite of vLLM 0.31.0's default behavior) and `HIDE_HEALTH_LOG` switches.

<a id="s3-6"></a>
### 3.6 Router

A CPU Pod (StatefulSet `pdd-router`, 1 replica, Pod name `pdd-router-0`), scheduled via nodeAffinity onto non-GPU nodes without the `doks.digitalocean.com/gpu-brand` label (the CPU pool name is not hardcoded, as pool names can differ across clusters and over time). Service `pdd-router:30000` is the external entry point.

- **Image** is the same as P/D (`${IMAGE}`), containing `python3` (**no `python`**), curl, and vllm, so `vllm bench` can run inside the Router Pod. Startup: `pip install --no-cache-dir vllm-router kubernetes requests` (adds packages only, changes nothing already installed) then `exec python3 -u /scripts/pdd-router.py`. The first pull of this ~9 GB image on a CPU node takes a few minutes.
- **RBAC**: ServiceAccount `pdd-router-sa` + Role (get/list/watch on `pods`; get/list on `apps/statefulsets`).
- **Readiness probe**: `GET /health` on port 30000, every 10 s. This is vllm-router's own endpoint and leaves no access logs in P/D.

**Behavior of `pdd-router.py`**

1. Reads the `replicas` of the `pdd-prefill` and `pdd-decode` StatefulSets from the K8S API.
2. Waits until both `pdd-prefill-0` and `pdd-decode-0` are Running + Ready + have IPs, and `GET http://<ip>:<port>/health` returns 200; polls every 5 s while waiting, printing the same status at most once every 30 s. So `pdd-router-0` stays not-Ready while P/D are starting — this is expected behavior.
3. Once both are ready, starts:

```
vllm-router --policy round_robin --vllm-pd-disaggregation --host 0.0.0.0 --port 30000
  --intra-node-data-parallel-size 8
  --prefill http://<prefill-pod-ip>:8001  --decode http://<decode-pod-ip>:8002
```

4. Then keeps monitoring: every 5 s checks whether the router subprocess has exited (if so, the script exits too, letting K8S restart the Pod; this check generates no requests); every `WATCH_INTERVAL` (default 60 s) checks Pod IPs and `/health`: on an IP change (e.g. the P or D Pod restarted with a new IP) it restarts the router; a temporarily not-Ready Pod only logs a message (the router has its own health checking and circuit breaking); sporadic K8S API failures don't affect the serving router.
5. `INTRA_NODE_DP=8` must equal each `vllm serve`'s `--data-parallel-size-local` (`LOCAL_DP` in `pdd-launch.sh`).
6. Without `--api-key`, `vllm-router` still starts and routes; its own health-check interval uses the default (60 s), not set explicitly by the script.

> Backends are discovered once at startup; P/D IP changes are picked up only via the step-4 monitor restarting the router; this is the script's logic, not vllm-router service discovery.

<a id="s3-7"></a>
### 3.7 Runtime environment record

The experiments in Section 5.7 (and this Section 3.7) were done on a **redeployed** 1P1D, with the same configuration, scripts, and image as the earlier chapters; Sections 5.1–5.6 were done on an earlier 1P1D deployment (at that time `pdd-prefill-0` was on `3o1v8w` and `pdd-decode-0` on `3o1v8e`; that deployment's image digest was not recorded). Collected 2026-10-13 05:08 UTC, Pods up 55 minutes.

**`kubectl get pods -o wide`**

```
NAME            READY   STATUS    RESTARTS   AGE   IP             NODE                        NOMINATED NODE   READINESS GATES
pdd-decode-0    1/1     Running   0          55m   10.154.2.102   rs-b300x8-gpu-pool-3o1v8b   <none>           <none>
pdd-prefill-0   1/1     Running   0          55m   10.154.1.73    rs-b300x8-gpu-pool-3o1v8w   <none>           <none>
pdd-router-0    1/1     Running   0          55m   10.154.3.202   rs-power-cpu-pool-3jmm1a    <none>           <none>
```

**`kubectl get nodes -o wide`** (external-IP column omitted; the cluster also has the two unused GPU nodes `3o1v8e` and `3o1v8g`)

```
NAME                        STATUS   ROLES    AGE     VERSION   INTERNAL-IP   OS-IMAGE                       KERNEL-VERSION                CONTAINER-RUNTIME
rs-b300x8-gpu-pool-3o1v8b   Ready    <none>   36h     v1.37.0   10.132.0.20   Debian GNU/Linux 13 (trixie)   6.12.96+deb13u3-amd64 (amd64)   containerd://2.3.3
rs-b300x8-gpu-pool-3o1v8e   Ready    <none>   36h     v1.37.0   10.132.0.19   Debian GNU/Linux 13 (trixie)   6.12.96+deb13u3-amd64 (amd64)   containerd://2.3.3
rs-b300x8-gpu-pool-3o1v8g   Ready    <none>   36h     v1.37.0   10.132.0.23   Debian GNU/Linux 13 (trixie)   6.12.96+deb13u3-amd64 (amd64)   containerd://2.3.3
rs-b300x8-gpu-pool-3o1v8w   Ready    <none>   36h     v1.37.0   10.132.0.22   Debian GNU/Linux 13 (trixie)   6.12.96+deb13u3-amd64 (amd64)   containerd://2.3.3
rs-power-cpu-pool-3jmm1a    Ready    <none>   7h12m   v1.37.0   10.132.0.17   Debian GNU/Linux 13 (trixie)   6.12.96+deb13u3-amd64 (amd64)   containerd://2.3.3
rs-power-cpu-pool-3jmm1e    Ready    <none>   7h12m   v1.37.0   10.132.0.13   Debian GNU/Linux 13 (trixie)   6.12.96+deb13u3-amd64 (amd64)   containerd://2.3.3
```

**Image and software versions**
- All 3 Pods use the same image: `vllm/vllm-openai:latest`, digest `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`.
- Inside the image: vLLM 0.31.0, NIXL 1.4.1 (`nixl-cu13`), NCCL 2.30.7, PyTorch 2.13.0+cu130; NVIDIA B300 SXM6, driver 580.178.04; each GPU node has the 8 RoCE NICs in use (400 Gb/s, 4X NDR, MTU 4200, per-NIC line rate ~50 GB/s).

**Startup**
- This deployment's startup: `init engine` took ~1098 s for Prefill and ~514 s for Decode (compilation ~43 s for both) — **Prefill is ~2× Decode**; the Router started after all Pods were Ready, with 1 `--prefill` and 1 `--decode` in the command. Both GPU Pods had 0 lines of `ERROR`/`Traceback`/`NIXL_ERR`, and 8 `Backend UCX was instantiated` each.
- This deployment's Router pairings (tallied during warm-up, 8 pairs): P0→D2, P1→D4, P2→D1, P3→D5, P4→D3, P5→D7, P6→D6, P7→D0, of which **P6→D6 is same-rail** (NIC index 6→6).

<a id="s4"></a>
## 4. Observation methods

[↑ Back to top](#top)

<a id="s4-1"></a>
### 4.1 `./pdd.sh metrics [window seconds]`

Reads Prefill's and Decode's `/metrics` from inside `pdd-router-0` via the headless Services' Pod DNS names — read-only and lightweight — and prints a summary: requests running/queued/succeeded/preempted; KV cache usage (max and mean across the 8 engines); prompt/generated token counts (with rates when windowed); mean TTFT, TPOT, ITL, queue, prefill, decode, and e2e; local and external (NIXL) prefix-cache hits; NIXL transfer count, total bytes, average size, average time, bandwidth (total bytes ÷ sum of per-transfer times, a rough reference only), and failure/notification-failure/expiry counters.

- Without an argument, values are cumulative since engine start; `./pdd.sh metrics 30` samples twice and shows deltas and rates over 30 s. A single read returns values already aggregated across all API servers of that Pod (verified).
- NIXL transfer metrics are populated only on the Decode (consumer) side; Prefill-side counters being 0 is expected; Prefill's TPOT being 0 is also expected (it generates only 1 token).
- Decode's external prefix-cache hit count equals its query count, meaning all of Decode's prompt KV arrives via transfer from Prefill.
- Metrics carry an `engine="0".."7"` label, one per DP rank; the `metrics` subcommand sums them. When per-rank resolution is needed (e.g. which Decode rank a transfer landed on), read the `engine` label from `/metrics` directly.

<a id="s4-2"></a>
### 4.2 Limitations of logs

- **No periodic stats**: under hybrid-lb, `api_server_count` defaults to `data-parallel-size-local` (8), and when `api_server_count > 1`, vLLM disables periodic stats logging (the log has 8 lines of `AsyncLLM created with api_server_count more than 1; disabling stats logging to avoid incomplete stats.`; checked against `v1/metrics/loggers.py`). So `Avg prompt throughput`, `Running: N reqs`, and the `KV Transfer metrics: ...` lines that come with them never appear; Prometheus `/metrics` is unaffected — hence the `metrics` subcommand of 4.1. If you must see them in the logs, add `--api-server-count 1` to `vllm serve`: the Pod then has a single API process serving 8 engines; under high concurrency the frontend may become the bottleneck (**Inferred**, untested), and vllm-router's cooperation with a single API process is also unverified; the current script does not offer this switch.
- **`GET /health` access logs** have three sources, all currently on 60-second periods: the K8S startup/readiness probes, `pdd-router.py`'s checks (`WATCH_INTERVAL`), and vllm-router's own health checks (default). `HIDE_HEALTH_LOG=on` silences them (only `/health,/metrics,/ping`; `POST` request logs are kept).
- UCX logs (`UCX_LOG_LEVEL=info`) show the transport and device in each worker's config line, e.g. `rma(rc_mlx5/mlx5_N:1)`; every worker of P and D has `Backend UCX was instantiated` and no `NIXL_ERR_BACKEND`.

<a id="s4-3"></a>
### 4.3 NICs, memory regions, and GPUs

- **Which NIC carries RDMA traffic**: `rdma statistic show link mlx5_N/1` inside the Pod (bundled with the image) gives `rx_write_requests`, `rx_read_requests`, etc. — **request counts, not bytes**; they are the target side's counters (incremented only on the side being read/written).
- **Wire bytes and rates**: `fabricN`'s `tx/rx_vport_rdma_unicast_bytes` (`fabricN`↔`mlx5_N`). The image has no `ethtool` command, but Python can read them via the `SIOCETHTOOL` ioctl (`ETHTOOL_GSSET_INFO`/`GSTRINGS`/`GSTATS`) — read-only, nothing to install; reading all 8 NICs takes ~14 ms, so sampling can reach 70 Hz; `ethtool` can also be temporarily installed (inside the running Pod only, lost on restart).
- **Registered memory regions**: `rdma resource show mr`; each line has device, `mrlen`, `pdn`, `pid` (note `pid` may be a thread id); can be grouped by process and device.
- **NIC speed**: `/sys/class/net/fabricN/speed` (400000 Mb/s on this cluster), `/sys/class/infiniband/mlx5_0/ports/1/rate` (`400 Gb/sec (4X NDR)`), MTU 4200, per-NIC line rate ~50 GB/s.
- **GPUs**: `nvidia-smi --query-gpu=utilization.gpu,memory.used`; topology `nvidia-smi topo -m`; NVLink traffic `nvidia-smi nvlink -gt d` (cumulative Tx/Rx per link, KiB; link status `nvidia-smi nvlink -s`, 53.125 GB/s per link, `NV18` meaning 18 links).
- All of these reads require entering the P/D Pods, read-only and lightweight; there is no sysfs `hw_counters` in the Pods.

<a id="s4-4"></a>
### 4.4 Processes and threads

- `ps -ef`, `/proc/<pid>/cmdline`, `/proc/<pid>/stat` (CPU time), `/proc/<pid>/task/<tid>/` (per-thread CPU, context switches, `io`'s `syscr`).
- Python stacks: `py-spy dump --nonblocking --pid <pid>` (not in the image; temporarily `pip install --target /tmp/pyspy` into the container's `/tmp`, lost on Pod restart); no `gdb`/`strace`/`perf` in the container, and `/proc/<tid>/stack` lacks the needed permissions, so native thread stacks are unavailable.

<a id="s5"></a>
## 5. Measurements

[↑ Back to top](#top)

Environment: 2026-10-13; `pdd-prefill-0`, `pdd-decode-0`, and `pdd-router-0` up for 70–90 minutes, no restarts, `PREFIX_CACHING=off`; all requests went through `pdd-router-0`.

<a id="s5-1"></a>
### 5.1 Process structure

Both Pods have the same process layout: **27 service processes** per Pod in total (raw `ps -ef` output in the appendix):

| Process | Count | Notes |
|---|---|---|
| `vllm serve` (PID 1) | 1 | main process, starts and manages the rest |
| `multiprocessing.resource_tracker` | 1 | Python multiprocessing resource tracker |
| `VLLM::DPCoordinator` | 1 | this pool's DP coordinator |
| `VLLM::APIServer_0..7` | 8 | the 8 API servers (see 4.2) |
| `VLLM::EngineCore_DP0..7` | 8 | one engine core (scheduler) per DP rank, parent is PID 1 |
| `VLLM::Worker_DPn_EPn` | 8 | one GPU worker per DP rank, **parent is the matching `EngineCore_DPn`**; DP and EP indices match, corresponding to DP8 + EP8 |

- Prefill and Decode have identical process types and counts; the only difference is PID 1's command line (`kv_role`, ports, and Decode's extra `--compilation-config`, consistent with the command in 3.4). Both Pods report `nproc` 224; thread counts (`ps -eLf`) were 6871 and 6890.
- No `vllm-router`; NIXL transfers complete inside the Worker processes (each Worker has loaded `libnixl.so`, `libplugin_UCX.so`, `libuct_ib.so` — checked).
- PID 14250 (Prefill) and 21098 (Decode) are `/bin/bash` attached to `pts/0`, started at 23:42:51 and 23:43:24 with `TERM=xterm` — interactive `kubectl exec` sessions with a TTY; who opened them cannot be determined from inside the Pod.

<a id="s5-2"></a>
### 5.2 Idle: GPUs at 0%, but each Worker burns a full CPU core

- **GPU**: continuous `nvidia-smi` sampling shows all 8 GPUs of both P and D at **0%** utilization. Idle per-GPU memory used: Prefill 242,286–247,088 MiB, Decode 229,310 MiB.
- **CPU**: the `C` column of `ps` shows ~97 (Prefill) and ~99 (Decode) for the Workers, with 57–61 minutes of cumulative CPU time, comparable to the Pods' uptime — i.e. each Worker has been burning one core continuously. Instantaneous usage over a 5-second window (`Worker_DP0_EP0`):

| Process | Prefill | Decode |
|---|---|---|
| `Worker_DP0_EP0` | 100% (31 user + 69 sys) | 101% (23 user + 78 sys) |
| `EngineCore_DP0`, `APIServer_0`, `DPCoordinator` | 0% | 0% |

- **Per-thread breakdown** (Decode's Worker 0, 153 threads in total): exactly **1 thread** burns a full core (tid 11409, 75–78% of it in the kernel); all others are below 3%; it yielded the CPU 0 times in 2 seconds, made read syscalls at ~**725,000/s**, and write 0 — continuous busy-polling.
- **Python stack** (`py-spy`): the main thread sits in `worker_busy_loop → dequeue → acquire_read → wait → zmq poll`, state idle (waiting for work); the hot thread is not in the Python thread list — it is a native thread.
- **What this thread is**:
  - Creation time: 4140 s ago, i.e. 23:39:51; the logs show NIXL initializing the agent and `Backend UCX was instantiated` at `2026-10-12 23:39:51` — matching to the second (the Worker process itself was created 4348 s ago).
  - `libplugin_UCX.so` contains `nixlUcxEngine::progressLoop`, `nixlUcxWorker::progressLoop`, `ucp_worker_progress`, and `Couldn't create progress thread control pipe`.
  - In `rdma resource show mr`, this thread (Decode tid 11409, Prefill tid 19084) owns its own RDMA memory regions, registered only on `mlx5_0` (rank 0's NIC): Decode has 2× 20 MiB + 2× 10 MiB, Prefill 2× 20 MiB.
- **Conclusion**: the Worker burning a core at idle is **not** dummy batches spinning (GPUs at 0%, main thread waiting for work); the evidence points to NIXL's UCX progress thread busy-polling. This is strong evidence, not a final verdict — the thread's own stack was not captured; a definitive answer would require redeploying with NIXL-related settings changed and comparing.

<a id="s5-3"></a>
### 5.3 Single request: all GPUs participate, no wake-up latency

Three 32768-token requests sent in sequence (different seeds): request 1 after ~28 minutes of idle (the previous request was at 00:21, this one at 00:49), request 2 immediately after request 1, request 3 thirty seconds after request 2 ended; meanwhile all 8 GPUs' utilization was sampled every 0.4 s in both P and D, and `rdma statistic` was read before and after each request.

| | Request 1 | Request 2 | Request 3 |
|---|---|---|---|
| Benchmark duration | 6.59 s | 6.55 s | 6.54 s |
| Mean TTFT | 3531.8 ms | 3518.6 ms | 3500.5 ms |
| Mean TPOT | 16.17 ms | 15.96 ms | 16.03 ms |
| Prefill rank selected by the router | `@0` | `@5` | `@7` |
| Prefill-side `rx_read_requests` delta | `mlx5_0`: 8784 | `mlx5_10`: 124928 | `mlx5_14`: 124928 |
| Other 7 NICs / all NICs' `rx_write_requests` | 0 / 0 | 0 / 0 | 0 / 0 |
| Peak utilization of the 8 Prefill GPUs during the request | 100 × 8 | 100 × 8 | 100 × 8 |
| Peak utilization of the 8 Decode GPUs during the request | 99 × 8 | 99 × 8 | 99 × 8 |

- **8-GPU co-running verified**: the router assigns the request to only one Prefill DP rank, yet all 8 GPUs of both P and D rise to 99–100% and return to 0% afterwards. Average utilization within the window is only 12–20%, because the window spans the entire `pdd.sh bench` command (~17.7 s, of which the request itself is only ~6.6 s; the rest is `vllm bench` startup overhead) — it is not the utilization during the request.
- **No wake-up latency**: the first request after 28 idle minutes had TTFT 3532 ms, no different from the one immediately after (3519 ms) or the one 30 s later (3501 ms). For warm-up requirements see Section 3.3.
- **EP communication does not go over RoCE**: `rx_write_requests` on all 8 NICs stayed 0 throughout.
- Prefill-side `rx_read_requests` differs sharply between request 1 (8784) and requests 2/3 (124928): this counter depends on the number and size of the NIXL transfer's descriptors, not on NIC behavior. RDMA requests are split at 256 KiB: for a 32768-token KV, one contiguous per-layer segment is 36 MiB, i.e. 144 requests per descriptor, so 61 descriptors (one contiguous segment per layer) yield 8784 (= 61 × 144); fully fragmented, it is 62,464 descriptors (61 layers × 1024 blocks), each 36,864 B — 2 requests each, 124,928 in total (= 62,464 × 2). The 8784 and 124928 here presumably correspond to these two cases (**Inferred**: descriptor counts for these runs were not recorded at the time; for 131072 tokens, 35,136 = 61 × 576 and 499,712 = 249,856 × 2 — see 5.4).

<a id="s5-4"></a>
### 5.4 KV transfer: a large 131072-token KV (P → D)

**Goal**: with a much larger-than-usual KV, examine performance, size, memory regions, cross-rail behavior, whether the GPU picks the nearest NIC, and whether the underlying op is READ or WRITE.

**Method**
- 6 requests sent sequentially, each 131072 tokens in, 16 tokens out (`BENCH_INPUT_LEN=131072 BENCH_OUTPUT_LEN=16 ./pdd.sh bench --num-prompts 1`, seeds 201–206); each request's KV is ~9.21 GB (131072 × 70,272 B).
- Before and after each request: read P's and D's `/metrics` (per `engine` label: NIXL bytes, transfer time, post time, descriptor count, and prefill time), and read `rdma statistic` for all 8 NICs on both sides.
- Throughout the experiment: read `tx/rx_vport_rdma_unicast_bytes` for all 8 NICs in P and D every ~15–20 ms (the ioctl method of 4.3). A transfer lasts only ~0.2 s, so each has just 10–11 samples — rates are coarse. `rdma resource show mr` was read before the experiment.

**Results**

| Request | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Prefill rank → Decode rank (router log; matches the `engine` labels) | 1 → 2 | 0 → 4 | 5 → 6 | 7 → 0 | 1 → 2 | 0 → 4 |
| NIC used on the Prefill side | `mlx5_2` | `mlx5_0` | `mlx5_10` | `mlx5_14` | `mlx5_2` | `mlx5_0` |
| NIC used on the Decode side | `mlx5_4` | `mlx5_8` | `mlx5_12` | `mlx5_0` | `mlx5_4` | `mlx5_8` |
| Bytes on the other 7 NICs | 0 | 0 | 0 | 0 | 0 | 0 |
| NIXL bytes transferred | 9,210,691,584 | same | same | same | same | same |
| NIXL descriptor count | 61 | 62,769 | 183 | 183 | 249,856 | 187,453 |
| Average bytes per descriptor | 150,994,944 B (144 MiB) | 146,739 B | 50,331,648 B (48 MiB) | 50,331,648 B | 36,864 B | 49,136 B |
| NIXL transfer time (`xfer_time`) | 191.3 ms | 192.6 ms | 191.4 ms | 192.1 ms | 198.3 ms | 196.4 ms |
| Bytes ÷ `xfer_time` | 48.1 GB/s | 47.8 GB/s | 48.1 GB/s | 48.0 GB/s | 46.4 GB/s | 46.9 GB/s |
| NIXL post time (`post_time`) | 0.1 ms | 2.7 ms | 0.2 ms | 0.2 ms | 13.4 ms | 9.6 ms |
| Wire bytes sent by the Prefill NIC | 9,386 MB | 9,392 MB | 9,386 MB | 9,386 MB | 9,410 MB | 9,404 MB |
| Peak wire rate (Prefill tx / Decode rx) | 53.2 / 50.7 GB/s | 49.8 / 50.3 | 50.1 / 50.3 | 50.0 / 50.5 | 49.7 / 50.0 | 49.9 / 49.7 |
| Duration of traffic on the NIC | ~0.18–0.20 s | 0.18 | 0.18 | 0.18–0.20 | 0.20 | 0.18–0.20 |
| Prefill-side `rx_read_requests` delta (`rx_write_requests` all 0) | 35,136 | 151,768 | 35,380 | 35,380 | 499,712 | 383,568 |
| Decode-side READ/WRITE request deltas | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Prefill's prefill time | 35.2 s | 34.6 s | 34.5 s | 34.1 s | 35.2 s | 34.6 s |
| Queue time recorded by Decode | 0.207 s | 0.209 s | 0.206 s | 0.207 s | 0.224 s | 0.215 s |
| Prefill-phase time recorded by Decode | 0.520 s | 0.552 s | 0.565 s | 0.534 s | 0.539 s | 0.571 s |
| TTFT measured at the client | 36,992 ms | 36,389 ms | 36,305 ms | 35,807 ms | 37,042 ms | 36,515 ms |

**Performance**
- A 9.21 GB transfer: NIXL-recorded time ~191–198 ms, i.e. **46–48 GB/s (~370–385 Gb/s)**; wire-sampled peaks ~50 GB/s, close to a single NIC's 400 Gb/s line rate. Each transfer uses exactly **one** NIC.
- Wire bytes exceed NIXL's payload by ~1.9–2.2% (9,386–9,410 MB vs. 9,211 MB) — protocol header overhead; the more fragmented the descriptors, the higher it is (request 5 peaks).
- The transfer is only ~0.5% of TTFT: TTFT ~36 s, of which Prefill's prefill compute is 34–35 s (~3.7k tokens/s), Decode-side queueing ~0.21 s, and Decode's recorded prefill phase ~0.52–0.57 s (includes the KV pull, larger than `xfer_time`'s 0.19 s; the gap is handshake, notifications, and scheduling, not further broken down). So at this length the KV transfer is not the bottleneck — prefill compute is.
- Descriptors going from 61 to 249,856 moves `xfer_time` only from 191 ms to 198 ms (~3.6% slower), and `post_time` from 0.1 ms to 13.4 ms. With only 6 samples, the most that can be said is "slightly slower when fragmented".

**Size and descriptors**
- The transferred byte count is fixed: 131072 tokens × 70,272 B = 9,210,691,584 B, consistent with the `config.json` arithmetic; 144 MiB per layer, 61 layers.
- A descriptor is "one contiguous run of blocks within one layer"; block size is 32 tokens, minimum 36,864 B; request 5's 249,856 = 61 layers × 4096 blocks — exactly one descriptor per block per layer; request 1 has only 61, i.e. one contiguous 144 MiB run per layer; requests 3 and 4 have 3 runs of 48 MiB per layer. **The descriptor count depends on how contiguous the block allocation is** — the same byte count can be 61 descriptors or 249,856. This count is recorded on the Decode side; the two sides' separate contributions were not teased apart.
- RDMA request counts: request 1's 35,136 is exactly 61 × 576 (each 144 MiB descriptor maps to 576 requests, i.e. 256 KiB per request); request 5's 499,712 is exactly 249,856 × 2. Smaller descriptors (36,864 B) are 2 requests; how requests 2 and 6's counts decompose was not checked individually — the counts are only recorded here. Each count is determined by the descriptor count and size, independent of which NIC is used.

**Memory regions** (`rdma resource show mr`, 8 Workers on each of P and D)
- Each Worker registered **2 identically sized large memory regions** on **its own NIC** (rank r ↔ `mlx5_{2r}`), each equal to that rank's entire KV cache: Prefill 124,301,611,008 B (115.77 GiB; the log's `GPU KV cache size: 1,768,864 tokens` — 1,768,864 × 70,272 B matches exactly), Decode 128,311,050,240 B (119.5 GiB, corresponding to 1,825,920 tokens). The two regions are in the same protection domain; why there are 2 was not determined.
- On the other 7 NICs, the Worker has no KV memory regions — only small ones (8 B, 544,768 B, 6.3 MB, 39.8 MB, etc.), whose purposes were not individually checked.
- So the KV cache is **one contiguous registered GPU memory region**, not per-layer registrations; a transfer pulls several segments out of this big region into the corresponding positions of the equally big region on the Decode side.

**Cross-rail or not**
- The 6 transfers' Prefill rank → Decode rank pairs were 1→2, 0→4, 5→6, 7→0, 1→2, 0→4 — indices all different, i.e. **all cross-rail** (`mlx5_{2r}` vs. `mlx5_{2d}`, r ≠ d), still ~48 GB/s; no speed penalty observed.
- Over the router's full logs (27 requests, including these 6 and earlier tests): the pairings are **8 fixed combinations** (of 8×8 = 64 possible): P0→D4, P1→D2, P2→D7, P3→D5, P4→D3, P5→D6, P6→D1, P7→D0 — r = d never occurred. So during **that router launch**, same-rail pairing never appeared. But pairings change across router restarts: in the later redeployment, 1 of the 8 pairs was same-rail (P6→D6); the same-rail transfer comparison is in 5.7. The pairing pattern is also partly understood: the Decode rank rotates in a fixed order and the Prefill rank in another fixed order, both with period 8, producing a fixed one-to-one mapping (**Inferred**; the `vllm-router` source was not read).

**Do GPUs pick the nearest NIC**
- Yes. Prefill rank r always uses `mlx5_{2r}`, and Decode rank d always uses `mlx5_{2d}`; `nvidia-smi topo -m` shows GPU i in the same PCIe bridge (`PXB`) as `mlx5_{2i}` and `mlx5_{2i+1}` — the even one is used, and `UCX_NET_DEVICES` lists only the even devices. Combined with "KV memory regions are registered only on the rank's own NIC", transfers naturally land there. The 6 transfers covered 4 distinct Prefill ranks (0, 1, 5, 7) and 4 distinct Decode ranks (0, 2, 4, 6), never touching any other NIC.

**The underlying op is RDMA READ, initiated by Decode**
- One Prefill NIC's `rx_read_requests` increases; `rx_write_requests` stays 0 throughout; neither counter increases on the Decode side (these counters are target-side — they increment only on the side being read/written);
- On the wire, the direction is Prefill sending, Decode receiving: Prefill's `tx_vport_rdma_unicast_bytes` and Decode's `rx_vport_rdma_unicast_bytes` each increase by ~9.4 GB, and only on that one pair of NICs;
- The source uses `make_prepped_xfer("READ", ...)`, and completion is signaled to the peer by a notification message.

Hence Decode actively pulls from Prefill's GPU memory; Prefill's CPU and GPU do not participate in the data movement.

<a id="s5-5"></a>
### 5.5 Topology and model configuration

- `nvidia-smi topo -m` (identical output from P and D): the 8 GPUs are pairwise `NV18`; GPUs 0–3 are on NUMA 0 (CPUs 0-111), GPUs 4–7 on NUMA 1 (CPUs 112-223). GPU i is `PXB` with NIC{2i} and NIC{2i+1}, `NODE` to the other NICs on the same NUMA, and `SYS` to those on the other NUMA; NIC{2i} and NIC{2i+1} are `PIX`. (`NICn` corresponds to `mlx5_n`.)
- `rdma link show`: `mlx5_0,2,…,14` are all `ACTIVE / LINK_UP`, corresponding to `fabric0,2,…,14`; `mlx5_1,3,…` have no netdev.
- The model's `config.json`: `num_hidden_layers=61`, `first_k_dense_replace=3`, `n_routed_experts=256`, `n_shared_experts=1`, `num_experts_per_tok=8`, `kv_lora_rank=512`, `qk_rope_head_dim=64`; hence 58 MoE layers, and with EP8, 32 routed experts per GPU.

<a id="s5-6"></a>
### 5.6 EP collective communication goes over NVLink

Question: do `all_gatherv` + `reduce_scatterv` stay intra-node on NVLink (since the EP group is within a single node)? Topology (`NV18` full interconnect) and the absence of RoCE traffic gave an indirect answer earlier (5.3, 5.5); here the NVLink traffic counters are read directly.

**Method**: `nvidia-smi nvlink -gt d` (cumulative Tx/Rx per NVLink, KiB), summing all links per GPU in both P and D Pods; first a 10-second idle baseline delta, then 1 request of 32768 tokens (`./pdd.sh bench --num-prompts 1`, seed 301, TTFT 3531 ms) with one reading before and after. Each read takes ~5 s, so the window is 21.9 s and includes `vllm bench` startup overhead (the request itself is ~6.6 s).

**Results**

| | 10 s idle | 1 request of 32768 tokens |
|---|---|---|
| Prefill, NVLink Tx summed over 8 GPUs | 0 | 277.9 GiB (Rx sum identical) |
| Prefill, per-GPU Tx | all 0 | 39.66 × 6, 25.64 (GPU 4), 14.29 (GPU 5) |
| Decode, NVLink Tx summed over 8 GPUs | 0 | 19.49 GiB (Rx sum identical) |
| Decode, per-GPU Tx | all 0 | 2.44 × 8 |

**Interpretation**
- No NVLink traffic while idle; with a request, massive NVLink traffic appears between the 8 GPUs of both P and D, while the same kind of request produces only NIXL's READ on the RoCE NICs (5.3). So **intra-node MoE collectives go over NVLink, not RoCE** — verified.
- Prefill's traffic far exceeds Decode's (~278 GiB vs. 19.5 GiB): Prefill processes 32768 tokens at once, and every MoE layer's `all_gatherv` broadcasts those tokens' activations to all ranks, while Decode has only a few tokens per step.
- Decode's 8 GPUs are perfectly uniform (2.44 GiB); Prefill has 6 uniform GPUs (39.66 GiB) with GPU 4 and GPU 5 lower (25.64, 14.29 GiB); the cause was not determined (possibly related to which rank the request landed on, but unverified).
- The counter only shows bytes traversing NVLink; it **cannot** distinguish which NCCL algorithm or protocol was used (e.g. P2P vs. NVLS), nor split the bytes between `all_gatherv` and `reduce_scatterv`.
- The counter covers all NVLink traffic on that GPU, including any point-to-point copies outside NCCL; NIXL's KV transfer goes over the RDMA NICs, not NVLink.

<a id="s5-7"></a>
### 5.7 KV transfer under concurrency

**Goal**: see whether KV transfers slow down under concurrency, and whether the slowdown is in the NIC or elsewhere. In 1P1D, all EP collectives are intra-node and RoCE carries only KV, so these numbers are a baseline free of collective-communication interference.

**Method**: on the redeployed 1P1D, warm up first (24 requests of 32768 tokens, concurrency 8, covering all 8 Prefill ranks and all 8 Decode ranks), then: (1) 1 single request of 32768 tokens; (2) 4 sequential requests of 32768 tokens; (3) 4 requests of 131072 tokens; (4) 24 requests of 32768/128 tokens, concurrency 8, run twice. Measurement methods per Chapter 4 and appendix A.4.

**Single and sequential requests** (1P1D, after warm-up)

| | Single request | 4 sequential requests |
|---|---|---|
| TTFT | 3.55 s | 3.51–3.55 s |
| TPOT | 16.3 ms | 16.1–16.3 ms |
| Prefill time | 3.06 s | 2.99–3.05 s |
| NIXL transfer time | 48.2 ms (62,464 descriptors, `post_time` 2.4 ms) | mean 49.3 ms (4 transfers, mean 54,671 descriptors) |
| Wire burst (Decode-side rx) | 60 ms, 39.1 GB/s, peak 49.8 | median 60 ms (40–60), 39.1 GB/s |
| NVLink traffic | Prefill 278.1 GiB, Decode 19.2 GiB | — |
| RoCE | KV only: one NIC per side, 2352 MB; other NICs ≤ 12 MB | likewise KV only |

**4 requests of 131072 tokens** (1P1D, after warm-up): P3→D5, P5→D7, P2→D1, P4→D3; NIC indices 3→5, 5→7, 2→1, 4→3 — **all cross-rail**; TTFT 36.3–36.9 s, prefill time 34.46–35.14 s; NIXL transfer time 189.8–192.6 ms (47.8–48.5 GB/s), `post_time` 0.2–4.3 ms; descriptor counts 183, 61, 61, 93,757; each transfer had exactly one Prefill NIC and one Decode NIC with >10 GB/s traffic (wire burst 0.201 s, peaks 49.7–51.9 GB/s); Prefill-side `rx_read_requests` 35,136 (the three runs with 183, 61, 61 descriptors) and 209,352 (the run with 93,757), write-request counts 0.

**24 concurrent requests, concurrency 8** (run twice)

| | Run 1 | Run 2 |
|---|---|---|
| Duration | 28.7 s | 28.6 s |
| Request throughput / token throughput | 0.84 req/s / 27.5k tok/s | 0.84 req/s / 27.6k tok/s |
| TTFT mean / median / P99 | 6.85 / 6.72 / 8.63 s | 6.86 / 6.70 / 8.59 s |
| TPOT mean / P99 | 17.4 / 17.9 ms | 17.3 / 17.7 ms |
| NIXL transfer time, mean | **65.5 ms** | **57.0 ms** |
| NIXL `post_time`, mean | 2.1 ms | 1.6 ms |
| Mean descriptors per request | 48,172 | 33,880 |
| `xfer_time` distribution | 9 ≤ 50 ms, 21 ≤ 75 ms, 24 ≤ 200 ms | 13 ≤ 50 ms, 23 ≤ 75 ms, 24 ≤ 200 ms |
| Wire burst (Decode-side rx) median | 60 ms, 39.0 GB/s (40–64 ms) | 60 ms, 39.0 GB/s (40–61 ms) |
| NVLink per request (Prefill / Decode) | 275.8 / 5.2 GiB | 275.8 / 5.4 GiB |
| Collectives on RoCE | none (each NIC 7051–7057 MB = 3 KV transfers) | none |

**Conclusions**
- **NIXL-recorded transfer times grow under concurrency**: ~49 ms single-request vs. 57.0 ms and 65.5 ms concurrent (+16–34%); the two runs alone differ by 8.5 ms.
- **Wire speed did not change**: under concurrency the KV burst median is 60 ms, ~39 GB/s — the same as single-request. RoCE carries only KV, no collectives, so **the growth is not caused by the NIC or by collectives**; more likely it is concurrency itself (the Decode engine is continuously decoding, so transfer-completion notifications wait on the engine — **Inferred**, unverified).
- **Concurrent throughput**: 24 requests, concurrency 8, 0.84 req/s, mean TTFT 6.85 s, TPOT 17.3–17.4 ms. Higher concurrency was not tested.
- **NVLink traffic per request**: Prefill roughly unchanged under concurrency (275.8 GiB vs. 278.1 GiB single-request); Decode drops from 19.2 GiB single-request to 5.2–5.4 GiB (~1/3.6), because concurrent requests share decode steps — one step's collectives are amortized across all requests in that step (**Inferred**, from the numbers).

**Same-rail vs. cross-rail** (in this deployment's pairings, P6→D6 is same-rail): in the concurrency experiments, each of the 8 Decode engines received 3 transfers; engine 6 corresponds to the same-rail P6→D6. Mean transfer times (ms) per engine over the two runs:

| Decode engine | 0 | 1 | 2 | 3 | 4 | 5 | **6 (same-rail)** | 7 |
|---|---|---|---|---|---|---|---|---|
| Run 1 | 55.4 | 57.5 | 82.4 | 49.4 | 52.7 | 87.1 | **50.4** | 89.2 |
| Run 2 | 58.3 | 51.3 | 52.6 | 50.1 | 56.7 | 53.0 | **85.1** | 48.9 |
| Both runs combined (6 transfers each) | 56.9 | 54.4 | 67.5 | 49.8 | 54.7 | 70.0 | **67.8** | 69.1 |

- Same-rail (engine 6, 6 transfers) averaged 67.8 ms; cross-rail (the other 42 transfers) averaged 60.3 ms — no stable difference: the same engine can swing from 50 ms to 85 ms between runs (engine 6 itself went 50.4 → 85.1; engines 2, 5, 7 likewise), a spread far larger than any same-rail vs. cross-rail gap. The same-rail wire bursts (NIC 6's 3 transfers) also matched cross-rail (~60 ms, ~39 GB/s).
- **Conclusion: no difference was observed between same-rail and cross-rail KV transfer speed** (6 same-rail vs. 42 cross-rail samples, with sizable run-to-run variance).

<a id="s6"></a>
## 6. Verification summary for inferred claims

[↑ Back to top](#top)

| Claim | Result |
|---|---|
| Idle Workers burning CPU is dummy batches spinning | **Refuted**: GPUs at 0% while idle, main thread waiting for work; the hotspot is a native thread, more likely NIXL's UCX progress thread (5.2) |
| A single request makes all 8 GPUs of Prefill and Decode busy | **Confirmed**, 3 times each on P and D (5.3) |
| The first request after a long idle is slower (waking the whole group) | **Not observed** (5.3) |
| NIXL transfers complete inside the worker processes | **Confirmed** (5.1) |
| EP collectives are intra-node, over NVLink, not RoCE | **Confirmed**: `NV18` full interconnect; no RDMA WRITE counts during requests (5.3, 5.5); NVLink counters 0 at idle, jumping with requests (5.6). Which NCCL algorithm/protocol specifically was not checked |
| Each worker uses the NIC nearest its own GPU (rank r ↔ `mlx5_{2r}`) | **Confirmed** (5.4) |
| KV transfer goes over RDMA, not TCP | **Confirmed**: exactly 1 RoCE NIC shows RDMA READ counts; the UCX config line is `rc_mlx5` (5.4, 4.2) |
| Transfer direction and method | **RDMA READ, initiated by Decode** (5.4) |
| DP synchronization uses Gloo on CPU under async scheduling | the log confirms NCCL is disabled; the Gloo path is a source-code conclusion |
| `dp_sync_interval` defaults to 16; stats logging disabled when `api_server_count > 1` | **Confirmed in the image's source** |
| KV per token ≈ 70.3 KB; 61 layers, 58 MoE layers, 32 experts per GPU | **Cross-checked against `config.json` and transfer bytes** |
| One Pod's `/metrics` aggregates all API servers | **Confirmed** (transfer counts match request counts) |
| The effect of `UCX_IB_GID_INDEX` and `UCX_IB_TRAFFIC_CLASS` | **Still unverified**: needs a redeploy with changed values for comparison |
| The effect of `UCX_MAX_RNDV_RAILS=1` | **Still unverified**: observed 1 NIC per transfer, consistent with it, but no comparison |
| The performance impact of `--api-server-count 1` | **Still unverified**: needs redeployment |
| The slowest rank sets the group's pace; co-running is independent of the all2all backend; dense-model DP has no co-running | **Still unverified** |
| Multi-node pools (more than 1 node per pool) | **Still unverified**: needs redeployment |

<a id="s7"></a>
## 7. Unverified items and cautions

[↑ Back to top](#top)

**Unverified**
- Same-rail transfer speed: only the post-redeployment run had 1 same-rail pair (P6→D6), which received 6 transfers, with no difference seen vs. cross-rail (5.7); small sample, large run-to-run variance.
- Bandwidth at other KV sizes (only 131072 and 32768 tokens tested); whether multiple concurrent transfers share a NIC and interfere with each other; the rate curve within a single transfer (too few samples).
- Why the 2 identically sized KV memory regions are 2; the purposes of the small memory regions; why smaller descriptors produce 2 RDMA requests; why the router's pairing order changes across restarts or redeployments (only a fixed rotation each for P and D was observed; the `vllm-router` source was not read).
- The exact identity of the idle-CPU hotspot thread (5.2).
- Which NCCL algorithm/protocol runs over NVLink, the traffic split between `all_gatherv` and `reduce_scatterv`, and why GPU 4 and GPU 5's NVLink traffic was low in Prefill (5.6).
- Multi-node pools (more than 1 node per pool, e.g. DP16+EP16); node counts that don't divide the expert count evenly (e.g. 3, 6); the non-zero-ordinal node's logic for waiting on the Head RPC port exists only in design.
- Concurrency stress tests, P/D ratios (e.g. 1P+3D), performance baselines.
- The items marked "still unverified" in Chapter 6.

**Cautions**
- `IMAGE` defaults to `:latest`; the three workloads may get different versions — use a pinned tag for consistency.
- `PREFIX_CACHING` and `HIDE_HEALTH_LOG` affect only P and D; the Router and vllm-router are not controlled by them.
- Creation, rebuild, and deletion of the deployment are up to the user; changing node counts or any switch requires `down` then `up`.

<a id="s8"></a>
## 8. Appendix: raw output and statistics scripts

[↑ Back to top](#top)

Read with `kubectl exec <pod> -- ps -ef --width 250` on 2026-10-13 00:35 UTC, Pods up ~60 minutes, 1P1D idle. No lines were removed; only two edits were made: command lines truncated to 250 columns (PID 1's full command line below), and the last 3 lines (`bash -c date ...`, `ps -ef`, `cut`) are the processes created by this collection itself, not part of the service. The `C` column is the process's lifetime-average CPU percentage, not an instantaneous value.

<a id="a1"></a>
### A.1 `pdd-prefill-0`

```
2026-10-13T00:35:27Z
UID          PID    PPID  C STIME TTY          TIME CMD
root           1       0  1 Oct12 ?        00:00:39 /usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --host 0.0.0.0 --port 8001 --kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_loa
root        1409       1  0 Oct12 ?        00:00:00 /usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(40)
root        1410       1  0 Oct12 ?        00:00:26 VLLM::DPCoordinator
root        1610       1  1 Oct12 ?        00:00:53 VLLM::EngineCore_DP0
root        1611       1  1 Oct12 ?        00:00:48 VLLM::EngineCore_DP1
root        1612       1  1 Oct12 ?        00:00:49 VLLM::EngineCore_DP2
root        1613       1  1 Oct12 ?        00:00:53 VLLM::EngineCore_DP3
root        1614       1  1 Oct12 ?        00:00:46 VLLM::EngineCore_DP4
root        1615       1  1 Oct12 ?        00:00:54 VLLM::EngineCore_DP5
root        1616       1  1 Oct12 ?        00:00:51 VLLM::EngineCore_DP6
root        1617       1  1 Oct12 ?        00:00:52 VLLM::EngineCore_DP7
root        1618       1  0 Oct12 ?        00:00:17 VLLM::APIServer_0
root        1619       1  0 Oct12 ?        00:00:15 VLLM::APIServer_1
root        1620       1  0 Oct12 ?        00:00:22 VLLM::APIServer_2
root        1621       1  0 Oct12 ?        00:00:24 VLLM::APIServer_3
root        1622       1  0 Oct12 ?        00:00:24 VLLM::APIServer_4
root        1623       1  0 Oct12 ?        00:00:20 VLLM::APIServer_5
root        1624       1  0 Oct12 ?        00:00:21 VLLM::APIServer_6
root        1625       1  0 Oct12 ?        00:00:21 VLLM::APIServer_7
root        7198    1617 97 Oct12 ?        00:57:21 VLLM::Worker_DP7_EP7
root        7199    1616 97 Oct12 ?        00:57:23 VLLM::Worker_DP6_EP6
root        7200    1610 97 Oct12 ?        00:57:27 VLLM::Worker_DP0_EP0
root        7202    1612 97 Oct12 ?        00:57:27 VLLM::Worker_DP2_EP2
root        7207    1613 97 Oct12 ?        00:57:32 VLLM::Worker_DP3_EP3
root        7208    1614 97 Oct12 ?        00:57:27 VLLM::Worker_DP4_EP4
root        7209    1615 97 Oct12 ?        00:57:20 VLLM::Worker_DP5_EP5
root        7210    1611 97 Oct12 ?        00:57:28 VLLM::Worker_DP1_EP1
root       14250       0  0 Oct12 pts/0    00:00:00 /bin/bash
root       24999       0  0 00:35 ?        00:00:00 bash -c date -u +%FT%TZ; ps -ef --width 250 | cut -c1-250
root       25006   24999  0 00:35 ?        00:00:00 ps -ef --width 250
root       25007   24999  0 00:35 ?        00:00:00 cut -c1-250
```

<a id="a2"></a>
### A.2 `pdd-decode-0`

```
2026-10-13T00:35:28Z
UID          PID    PPID  C STIME TTY          TIME CMD
root           1       0  1 Oct12 ?        00:00:39 /usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --host 0.0.0.0 --port 8002 --kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_loa
root        1406       1  0 Oct12 ?        00:00:00 /usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(40)
root        1407       1  0 Oct12 ?        00:00:26 VLLM::DPCoordinator
root        1607       1  1 Oct12 ?        00:00:55 VLLM::EngineCore_DP0
root        1608       1  1 Oct12 ?        00:00:54 VLLM::EngineCore_DP1
root        1609       1  1 Oct12 ?        01:00:00 VLLM::EngineCore_DP2
root        1610       1  1 Oct12 ?        00:00:58 VLLM::EngineCore_DP3
root        1611       1  1 Oct12 ?        00:00:57 VLLM::EngineCore_DP4
root        1612       1  1 Oct12 ?        00:00:50 VLLM::EngineCore_DP5
root        1613       1  1 Oct12 ?        00:00:44 VLLM::EngineCore_DP6
root        1614       1  1 Oct12 ?        00:00:53 VLLM::EngineCore_DP7
root        1615       1  0 Oct12 ?        00:00:19 VLLM::APIServer_0
root        1616       1  0 Oct12 ?        00:00:18 VLLM::APIServer_1
root        1617       1  0 Oct12 ?        00:00:19 VLLM::APIServer_2
root        1618       1  0 Oct12 ?        00:00:20 VLLM::APIServer_3
root        1619       1  0 Oct12 ?        00:00:16 VLLM::APIServer_4
root        1620       1  0 Oct12 ?        00:00:22 VLLM::APIServer_5
root        1621       1  0 Oct12 ?        00:00:17 VLLM::APIServer_6
root        1622       1  0 Oct12 ?        00:00:20 VLLM::APIServer_7
root        7157    1607 99 Oct12 ?        01:00:38 VLLM::Worker_DP0_EP0
root        7159    1613 99 Oct12 ?        01:00:40 VLLM::Worker_DP6_EP6
root        7189    1614 99 Oct12 ?        01:00:39 VLLM::Worker_DP7_EP7
root        7192    1610 99 Oct12 ?        01:00:49 VLLM::Worker_DP3_EP3
root        7193    1608 99 Oct12 ?        01:00:45 VLLM::Worker_DP1_EP1
root        7197    1612 99 Oct12 ?        01:00:42 VLLM::Worker_DP5_EP5
root        7198    1609 99 Oct12 ?        01:00:43 VLLM::Worker_DP2_EP2
root        7199    1611 99 Oct12 ?        01:00:37 VLLM::Worker_DP4_EP4
root       21098       0  0 Oct12 pts/0    00:00:00 /bin/bash
root       25288       0  0 00:35 ?        00:00:00 bash -c date -u +%FT%TZ; ps -ef --width 250 | cut -c1-250
root       25295   25288  0 00:35 ?        00:00:00 ps -ef --width 250
root       25296   25288  0 00:35 ?        00:00:00 cut -c1-250
```

<a id="a3"></a>
### A.3 Full command line of PID 1 (`/proc/1/cmdline`)

Prefill:

```
/usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --host 0.0.0.0 --port 8001 --kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"} --no-enable-prefix-caching --data-parallel-size 8 --data-parallel-size-local 8 --data-parallel-start-rank 0 --data-parallel-address 10.154.1.123 --data-parallel-rpc-port 13345 --data-parallel-hybrid-lb --enable-expert-parallel --enable-auto-tool-choice --tool-call-parser deepseek_v3 --reasoning-parser deepseek_v3
```

Decode:

```
/usr/bin/python3 /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --host 0.0.0.0 --port 8002 --kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"} --no-enable-prefix-caching --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"} --data-parallel-size 8 --data-parallel-size-local 8 --data-parallel-start-rank 0 --data-parallel-address 10.154.2.224 --data-parallel-rpc-port 13345 --data-parallel-hybrid-lb --enable-expert-parallel --enable-auto-tool-choice --tool-call-parser deepseek_v3 --reasoning-parser deepseek_v3
```

<a id="a4"></a>
### A.4 NVLink / RDMA traffic statistics scripts

All traffic numbers in this document (NVLink GiB, RoCE MB and wire rates, RDMA request counts, NIXL transfer times) came from the small scripts below. They live in `traffic-tools/` and are all **read-only**: they only read counters, change no configuration, and need nothing installed in the Pods.

**Collection scripts**

| Script | Runs in | Reads | How |
|---|---|---|---|
| `nvlink_counters.py` | inside P/D Pods | per-GPU cumulative NVLink Tx/Rx (KiB) | parses `nvidia-smi nvlink -gt d`, summing all links of a GPU |
| `roce_bytes_sampler.py` | inside P/D Pods | `tx/rx_vport_rdma_unicast_bytes` of the 8 NICs (cumulative bytes) | ethtool ioctl (`SIOCETHTOOL`) reading `fabric0,2,…,14`, corresponding to `mlx5_0,2,…,14`; samples every 20 ms by default |
| `rdma_request_counters.sh` | inside P/D Pods (`kubectl exec`) | RDMA WRITE / READ request counts per NIC | `rdma statistic show link mlx5_N/1` |
| `nixl_metrics.py` | inside the Router Pod | NIXL transfer bytes, transfer time, post time, descriptor count, prefill time, etc., split by engine (DP rank) | scrapes P/D `/metrics`, keeping the needed series |

**Experiment and analysis scripts**

| Script | Purpose |
|---|---|
| `exp_window.sh` | runs one bench command and measures traffic over the whole window (before/after snapshots + RoCE sampling); used for the concurrency experiments |
| `exp_per_request.sh` | measures per request, snapshotting before and after each one; used for the 131072-token large-KV runs |
| `analyze_window.py` | summarizes, within a window: NVLink GiB per GPU, RoCE MB and peak rates per NIC, READ/WRITE request counts, NIXL times (incl. `xfer_time` histogram quantiles), KV wire-burst duration and rate, and router pairings |
| `analyze_per_request.py` | one block per request: Prefill/Decode rank pair, same-rail or not, TTFT, NIXL time and bandwidth, the NICs serving the transfer, wire-burst bytes/duration/peak |
| `traffic_common.py` | functions shared by the two analysis scripts (parsing router logs, reading samples, finding bursts, etc.) |

**Other small read-only scripts**

| Script | Runs in | Purpose |
|---|---|---|
| `router_pairs.py` | any machine with `kubectl` | tallies P→D pairs from router logs: occurrences per pair, NIC indices on both ends, same-rail or not, selection order (5.4) |
| `idle_cpu.py` | inside P/D Pods | per-Worker and per-process CPU usage over 3 s when idle (5.2) |
| `rdma_mr.py` | inside P/D Pods | which memory regions each Worker registered on which NICs, and how large (5.4) |

**How the numbers are computed**

- **All counters are cumulative**: take a snapshot before and after, and subtract. NVLink: `(after − before) / 2²⁰` gives GiB (per GPU); summing the 8 GPUs gives the whole node; every byte is sent by one GPU and received by another, so the node-wide Tx total equals the Rx total.
- **Wire rate**: consecutive RoCE samples are differenced and divided by the interval; a contiguous interval with rate above 10 GB/s counts as one KV burst (idle heartbeats and any collectives on other NICs are far below this; KV transfers run at 40–50 GB/s). The time resolution is the 20 ms sampling interval.
- **READ request counts**: Decode issues RDMA READs, so the READ counter increments on the **Prefill NIC serving the data**; this counter grows with the request/descriptor count, **not with bytes** (the 8784 vs. 124928 explanation is in Section 5.4).
- **Mean transfer time and bandwidth**: `nixl_xfer_time_seconds_sum / nixl_xfer_time_seconds_count`; `nixl_bytes_transferred_sum / nixl_xfer_time_seconds_sum`.

**Caveats when using**

- Warm up before measuring so that every DP rank has served a long request (Section 3.3); otherwise the first request's one-time overhead contaminates the data.
- Requests are sent from the Router Pod (`pdd.sh bench` does exactly this); the P/D Pods only perform read-only counter reads.
- NVLink counters include all NVLink traffic regardless of source; snapshot before/after around a known load while the pool is idle; one read takes ~0.4–1 s, and the snapshot is not atomic.
- RoCE counters cover only RDMA traffic (NIXL/UCX KV transfers, plus any NCCL that goes over RoCE); TCP on eth0 is not included.
- `exp_*.sh` defaults to 1P1D; to measure multi-node pools, set the environment variables `PODS`, `PN`, `DN`; the analysis scripts infer the Pod count from the files in the directory, and IPs in router logs are mapped to Pods via the `ipmap.json` saved at experiment start.

**Verification status**: the two analysis scripts were re-run on this document's 1P1D measurement data and reproduce the results (4×131072: transfers 189.8–192.6 ms, 47.8–48.5 GB/s; 24 concurrent requests: 65.5 ms, `xfer_time` distribution 9/21/24 within 50/75/200 ms); the collection scripts and `router_pairs.py`, `idle_cpu.py`, `rdma_mr.py` were exercised read-only on a live 1P1D with normal output; the two `exp_*.sh` driver scripts only had a syntax check — the full experiment was not re-run after they were reorganized into their present form (the pre-reorganization originals were what was actually used at the time).

[↑ Back to top](#top)
