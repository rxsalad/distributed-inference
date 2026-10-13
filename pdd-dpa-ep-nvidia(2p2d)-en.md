<a id="top"></a>
# 2P2D: Prefill and Decode Each with 2 Nodes (DP16 + EP16) — PD Disaggregation Measurements on B300x8

DeepSeek-V3 (FP8) on a B300x8 (rail-optimized) cluster with Prefill/Decode disaggregation: **2 Prefill nodes + 2 Decode nodes**. Each pool is one DEP group spanning 2 nodes and 16 GPUs (DP16 + EP16); the KV cache is transferred from Prefill to Decode via NIXL; traffic enters through a vllm-router running on a CPU Pod.

**Read the 1P1D document first** (principles, deployment and configuration, observation methods, single-pool measurements). This document is its multi-node extension: the YAML and scripts used for deployment are **identical** to 1P1D's (`pdd-dep.yaml`, `pdd-launch.sh`, `pdd-router.py`, `pdd.sh` — not a single character changed); only the two pools' node counts went from 1 to 2. Topics the 1P1D document already covered (DEP and dummy batches, waves, NIXL and the side channel, deployment parameters, observation methods, warm-up rules, single-pool measurements) are not repeated here, only referenced ("1P1D doc Section x.x"); this document covers only the differences multi-node brings, and Chapter 8 summarizes the comparison with 1P1D.

**This document answers four questions**

| Question | Where |
|---|---|
| With 2 nodes each for P and D, which processes are there and how do they cooperate? | Chapter 3 |
| EP collective communication uses both NVLink and RDMA at once — how much of each, and how are the rails used? | Chapter 5 |
| How does KV cache transfer behave when cross-node and cross-rail? | Chapter 6 |
| Does the EP collectives' RoCE traffic affect NIXL/UCX KV transfer? | Chapter 7 |

**Conventions**
- **Annotations**: **Observed** = phenomena seen directly or data read on this cluster; **Inferred** = derived from mechanism, not verified; **source-verified** = checked against the vLLM 0.31.0 source inside the image.
- **Data**: all data was collected on 2026-10-13 from the running 2P2D, and all of it **after warm-up** (Section 4.3).
- **rail**: NIC index. This document assumes that same-index NICs on the various nodes (`mlx5_{2i}`, i = 0..7, "NIC i" below) connect to the same rail (NCCL's inter-node traffic is entirely "same-index NIC to same-index NIC", consistent with this, but switch configs were not read to confirm). **Cross-rail** = the two ends use different NIC indices.
- **global rank**: within a pool, DP rank 0..15 — node 0 holds 0..7, node 1 holds 8..15. P5 means Prefill-pool global rank 5; D13 means Decode-pool global rank 13.

<a id="toc"></a>
## Table of Contents

- [1. Summary](#s1)
- [2. Deployment and environment](#s2)
  - [2.1 Deployment command, and why no file needed changing](#s2-1)
  - [2.2 Runtime environment record](#s2-2)
  - [2.3 Startup timeline](#s2-3)
- [3. Processes and relationships](#s3)
  - [3.1 Processes inside each Pod](#s3-1)
  - [3.2 DP rank / EP rank distribution](#s3-2)
  - [3.3 Control plane: how the nodes connect](#s3-3)
  - [3.4 The Router and the P→D pairings](#s3-4)
  - [3.5 Idle state](#s3-5)
- [4. Measurement methods and warm-up](#s4)
  - [4.1 What to measure and how](#s4-1)
  - [4.2 How to separate collectives from KV traffic](#s4-2)
  - [4.3 Warm-up is mandatory before testing](#s4-3)
- [5. EP collective communication: NVLink and RDMA](#s5)
  - [5.1 The EP16 communication pattern](#s5-1)
  - [5.2 Collective-communication volume of one 32768-token request](#s5-2)
  - [5.3 Division of labor between NVLink and RoCE](#s5-3)
  - [5.4 Inter-node collectives are rail-aligned](#s5-4)
  - [5.5 Prefill uses 8 rails, Decode only 3](#s5-5)
  - [5.6 Traffic over time: RoCE is nowhere near saturated](#s5-6)
  - [5.7 Collective-communication volume under concurrency](#s5-7)
- [6. KV transfer: cross-node, cross-rail](#s6)
  - [6.1 Four transfers of a large KV (131072 tokens)](#s6-1)
  - [6.2 Cross-rail or not, nearest NIC or not, READ or WRITE](#s6-2)
  - [6.3 Performance](#s6-3)
  - [6.4 After the Router restart: same-rail vs. cross-rail](#s6-4)
- [7. Do collectives affect KV transfer?](#s7)
- [8. Comparison with 1P1D](#s8)
- [9. Verification and unverified items](#s9)
- [10. Appendix: raw output and helper scripts](#s10)
  - A.1–A.4: raw `ps -ef` output of the four Pods
  - [A.5 Helper scripts](#a5)

<a id="s1"></a>
## 1. Summary

[↑ Back to top](#top)

**Deployment and processes**

| Question | Finding |
|---|---|
| Do the YAML/scripts need changing | **No**: `PREFILL_NODES=2 DECODE_NODES=2 ./pdd.sh up` (also the default) suffices; `pdd.sh metrics` and the Router adapt to the node count automatically; no bugs found |
| Processes | Each Pod has the same process count as 1P1D (27); **`DPCoordinator` exists only on node 0**; node 1's engines are `EngineCore_DP8..15` and workers `Worker_DP8_EP8..DP15_EP15` |
| Intra-pool vs. inter-pool connections | The two nodes within a pool have 398 TCP connections between them (control plane, eth0); **not a single TCP connection between the P pool and the D pool** — only NIXL point-to-point |
| P→D pairings | The Router's 16 Prefill ranks → 16 Decode ranks form a fixed one-to-one mapping (not 16×16); in the Router launch that produced this document's main data **not one pair** had matching NIC indices; **after a Router restart the mapping gets rearranged, and 3 same-rail pairs appeared** (3.4, 6.4) |

**EP collective communication**

| Question | Finding |
|---|---|
| What it uses | **Both NVLink and RoCE at once**: for one 32768-token request, ~293–331 GiB over NVLink within each Prefill node and ~42–44 GB over inter-node RoCE (~12% of the NVLink bytes) |
| How the rails are used | **Rail-aligned**: node A's NIC i only talks to node B's NIC i; the Prefill pool uses all 8 rails, **the Decode pool only 3 (NIC 0, 1, 2)** |
| Bandwidth | Bursty; 2–4 GB/s per NIC, under 10% of line rate — **RoCE is nowhere near saturated** |

**KV transfer**

| Question | Finding |
|---|---|
| Method | **RDMA READ, initiated by Decode**; each transfer uses exactly one NIC per side, each rank's nearest one (rank r ↔ `mlx5_{2(r%8)}`) |
| Cross-rail or not | **All cross-rail**; cross-rail transfers still reach wire-rate peaks of ~50 GB/s |
| Performance | 9.21 GB (131072 tokens) in ~191–223 ms, about 41–48 GB/s |
| Do collectives affect it | **Wire speed is unaffected**: under 24 concurrent requests, the KV burst median is still 60 ms at ~39 GB/s, same as single-request; the NIXL-recorded transfer time rises from ~54 ms to 72 ms (~+30%) — not the NIC being saturated; the 1P1D control (no collectives on RoCE) grows 16–34% under the same concurrency, so it is **mainly not caused by collectives** (Chapter 7) |

**Comparison with 1P1D, and measurement requirements**

| Question | Finding |
|---|---|
| TTFT | 32768 tokens: 2P2D **3.55–3.71 s** vs. 1P1D 3.5 s — essentially the same; 131072-token prefill 34.7–35.4 s, same as 1P1D's 34.1–35.2 s |
| TPOT | 2P2D 21.0 ms vs. 1P1D 16.1 ms (Decode collectives go cross-node, ~5 ms slower) |
| **Warm-up is mandatory before testing** | First let all 16 Prefill ranks and 16 Decode ranks each serve a long request, then start timing (Section 4.3) |

<a id="s2"></a>
## 2. Deployment and environment

[↑ Back to top](#top)

<a id="s2-1"></a>
### 2.1 Deployment command, and why no file needed changing

```bash
cd /root/work/data/distributed-inference/pdd-dpa-ep-nvidia/
PREFILL_NODES=2 DECODE_NODES=2 ./pdd.sh up     # the defaults are 2 and 2 when the variables are unset
```

`pdd.sh up` checks that `PREFILL_NODES + DECODE_NODES = 4` does not exceed the cluster's GPU node count (4). The four Pods each occupy one B300×8 node; `pdd-router-0` sits on a CPU node. For the meaning of each file and parameter, see 1P1D doc Chapter 3.

> **File locations**: this document lives directly under `distributed-inference/`; all files it references — `pdd-dep.yaml`, `pdd-launch.sh`, `pdd-router.py`, `pdd.sh`, and the `traffic-tools/` scripts of A.5 — are in the `pdd-dpa-ep-nvidia/` subdirectory (alongside the Chinese originals of this and the 1P1D document).

**Why no file needed changing** (verified by an actual deployment):
- `pdd-launch.sh` derives `DP_SIZE = NODES × 8 = 16` and `START_RANK = ordinal × 8` from the Pod-name ordinal (node 1 → 8); non-zero-ordinal nodes first wait for the Head's (`-0`) RPC port 13345 to become connectable, then start.
- `pdd-router.py` reads both StatefulSets' `replicas`, waits for all P/D Pods to be ready, then starts `vllm-router` with 2 `--prefill` and 2 `--decode`; the logs show the Router found **32 workers** (8 DP ranks per Pod).
- `pdd.sh metrics` iterates over all nodes of each pool (`16 engines` in the output); `test` and `bench` work as usual — `./pdd.sh test` passed on the first try after startup finished (HTTP 200).
- The four files' modification times were unchanged by the deployment (all older than this deployment).

<a id="s2-2"></a>
### 2.2 Runtime environment record

Collected 2026-10-13 03:27 UTC (Pods up 112 minutes; all experiments in this document were done on this deployment).

**`kubectl get pods -o wide`**

```
NAME            READY   STATUS    RESTARTS   AGE    IP             NODE                        NOMINATED NODE   READINESS GATES
pdd-decode-0    1/1     Running   0          112m   10.154.2.116   rs-b300x8-gpu-pool-3o1v8b   <none>           <none>
pdd-decode-1    1/1     Running   0          112m   10.154.1.150   rs-b300x8-gpu-pool-3o1v8g   <none>           <none>
pdd-prefill-0   1/1     Running   0          112m   10.154.1.124   rs-b300x8-gpu-pool-3o1v8w   <none>           <none>
pdd-prefill-1   1/1     Running   0          112m   10.154.2.207   rs-b300x8-gpu-pool-3o1v8e   <none>           <none>
pdd-router-0    1/1     Running   0          112m   10.154.3.203   rs-power-cpu-pool-3jmm1a    <none>           <none>
```

The Pod IPs appearing in this text (the Router's `--prefill`/`--decode` addresses, the peers in the `ss` statistics) are the IPs above; no Pod restarted.

**`kubectl get nodes -o wide`** (external-IP column omitted)

```
NAME                        STATUS   ROLES    AGE     VERSION   INTERNAL-IP   OS-IMAGE                       KERNEL-VERSION                CONTAINER-RUNTIME
rs-b300x8-gpu-pool-3o1v8b   Ready    <none>   35h     v1.37.0   10.132.0.20   Debian GNU/Linux 13 (trixie)   6.12.96+deb13u3-amd64 (amd64)   containerd://2.3.3
rs-b300x8-gpu-pool-3o1v8e   Ready    <none>   35h     v1.37.0   10.132.0.19   Debian GNU/Linux 13 (trixie)   6.12.96+deb13u3-amd64 (amd64)   containerd://2.3.3
rs-b300x8-gpu-pool-3o1v8g   Ready    <none>   35h     v1.37.0   10.132.0.23   Debian GNU/Linux 13 (trixie)   6.12.96+deb13u3-amd64 (amd64)   containerd://2.3.3
rs-b300x8-gpu-pool-3o1v8w   Ready    <none>   35h     v1.37.0   10.132.0.22   Debian GNU/Linux 13 (trixie)   6.12.96+deb13u3-amd64 (amd64)   containerd://2.3.3
rs-power-cpu-pool-3jmm1a    Ready    <none>   5h31m   v1.37.0   10.132.0.17   Debian GNU/Linux 13 (trixie)   6.12.96+deb13u3-amd64 (amd64)   containerd://2.3.3
rs-power-cpu-pool-3jmm1e    Ready    <none>   5h31m   v1.37.0   10.132.0.13   Debian GNU/Linux 13 (trixie)   6.12.96+deb13u3-amd64 (amd64)   containerd://2.3.3
```

**Image and software versions**: the 5 Pods (4 GPU Pods + the Router) use **the same image** `vllm/vllm-openai:latest`, digest `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3` — identical to what the 1P1D doc records in 3.7 (so `latest` caused no version mismatch in this deployment). Software versions (vLLM 0.31.0, NIXL 1.4.1, NCCL 2.30.7, PyTorch 2.13.0+cu130), GPU and NIC hardware, driver, and node OS are the same as 1P1D doc 3.7 and 5.5.

<a id="s2-3"></a>
### 2.3 Startup timeline

Observed (from logs):

| Event | Time (UTC, 2026-10-13) |
|---|---|
| Four Pods created, launch scripts started | 01:35 |
| Node 1 waits for Head: `waiting for pdd-prefill-0.pdd-prefill-headless ...`, `waiting for DP master <ip>:13345 ...` | early in startup, continuing afterwards |
| All engines start initializing | 01:36:31 |
| Each rank's `Expert parallelism is enabled ... [EP Rank k/16] ... Local/global number of experts: 16/256` | 01:37:26 |
| Decode's two nodes finish `init engine` | ~531 s, 01:47:06 |
| Prefill's two nodes finish `init engine` | ~1144 s, 01:57:19 |
| Router finds all 2+2 Pods ready and starts | 01:57:56 |

- 16 experts per rank (256 / 16); compilation took ~43–45 s, identical for both pools.
- **Prefill's initialization is ~2× Decode's** (1144 s vs. 531 s); the cause was not determined. Decode uses `FULL_DECODE_ONLY` CUDA graphs while Prefill uses the default `FULL_AND_PIECEWISE` — a possibly relevant difference (**Inferred**, unverified).
- All 4 GPU Pods' logs had 0 lines of `ERROR`/`Traceback`/`NIXL_ERR`; each Pod had 8 `Backend UCX was instantiated`.

<a id="s3"></a>
## 3. Processes and relationships

[↑ Back to top](#top)

<a id="s3-1"></a>
### 3.1 Processes inside each Pod

Each GPU Pod's process layout matches 1P1D doc 5.1 (`vllm serve`, `resource_tracker`, 8 `APIServer_0..7`, 8 `EngineCore_DPn`, 8 `Worker_DPn_EPn`); the multi-node difference is that **`DPCoordinator` exists only on node 0**, and node 1's rank indices continue the sequence (raw output in the appendix):

| Pod | `DPCoordinator` | `EngineCore` | `Worker` |
|---|---|---|---|
| `pdd-prefill-0` / `pdd-decode-0` (node 0) | 1 | `DP0..DP7` | `Worker_DP0_EP0 .. DP7_EP7` |
| `pdd-prefill-1` / `pdd-decode-1` (node 1) | **none** | `DP8..DP15` | `Worker_DP8_EP8 .. DP15_EP15` |

- Node 0 has 27 service processes; node 1 has only 26.
- Within each pool it is one big **DP16 + EP16 group** with a single coordinator (node 0's `DPCoordinator`; wave coordination see 1P1D doc 2.3); the P pool and the D pool each have their own, unconnected to each other.
- Node 1's 8 EngineCores join node 0's coordinator via `--data-parallel-address` (node 0's Pod IP) and RPC port 13345.

<a id="s3-2"></a>
### 3.2 DP rank / EP rank distribution

| | Node 0 | Node 1 |
|---|---|---|
| DP rank = EP rank | 0..7 | 8..15 |
| `--data-parallel-start-rank` | 0 | 8 |
| Experts per rank | 16 (out of 256) | 16 |
| GPUs per rank | this node's GPUs 0..7 | this node's GPUs 0..7 |

Log line (each node): `rank k in world size 16 is assigned as DP rank k, PP rank 0, PCP rank 0, TP rank 0, ETP rank 0, EP rank k`, `[EP Rank k/16] ... Local/global number of experts: 16/256`.

<a id="s3-3"></a>
### 3.3 Control plane: how the nodes connect

Established TCP connections were tallied per GPU Pod with `ss -tn`, grouped by peer:

| Counted in | to the same-pool peer Pod | to Pods of the other pool |
|---|---|---|
| `pdd-prefill-0` | 398 (382 + 16) | 0 |
| `pdd-prefill-1` | 398 | 0 |
| `pdd-decode-0` | 398 | 0 |
| `pdd-decode-1` | 398 | 0 |

- The two nodes within a pool have 398 TCP connections between them (over eth0, the control plane: DP-coordination RPC/ZMQ, Gloo, TCPStore, etc.; each connection's specific purpose was not individually checked). The number is symmetric across the 4 Pods, independent of which node is Head.
- **Not a single TCP connection between the P pool and the D pool**: between P and D there is only NIXL point-to-point (after the handshake, the data goes over RDMA).
- **Router connections to the Pods are short-lived and invisible at idle**: the Router treats the 4 GPU Pods equally, regardless of Head. At idle, no Router connections can be counted either in the Router Pod or in the 4 GPU Pods; when 8 concurrent requests were sent, the peak number of simultaneously-existing connections seen in the Router Pod was `prefill-0` 4, `prefill-1` 6, `decode-0` 2, `decode-1` 7.

<a id="s3-4"></a>
### 3.4 The Router and the P→D pairings

Router startup command:

```
vllm-router --policy round_robin --vllm-pd-disaggregation --host 0.0.0.0 --port 30000 --intra-node-data-parallel-size 8
  --prefill http://10.154.1.124:8001 --prefill http://10.154.2.207:8001
  --decode http://10.154.2.116:8002 --decode http://10.154.1.150:8002
```

The Router selects backends per DP rank (logs like `prefill=http://<ip>:8001@r`, with r the in-node rank), 32 workers in total. Tallying **global-rank pairings** over all 91 requests so far in the Router logs: it is a fixed one-to-one mapping (16 Prefill ranks → 16 distinct Decode ranks, **not the 16×16 combinations**); each Prefill rank appeared 3–8 times, and every pairing falls in this table:

| P rank | D rank | P NIC index | D NIC index | Same index? |
|---|---|---|---|---|
| P0 | D5 | 0 | 5 | No |
| P1 | D15 | 1 | 7 | No |
| P2 | D0 | 2 | 0 | No |
| P3 | D4 | 3 | 4 | No |
| P4 | D7 | 4 | 7 | No |
| P5 | D11 | 5 | 3 | No |
| P6 | D3 | 6 | 3 | No |
| P7 | D10 | 7 | 2 | No |
| P8 | D14 | 0 | 6 | No |
| P9 | D12 | 1 | 4 | No |
| P10 | D6 | 2 | 6 | No |
| P11 | D1 | 3 | 1 | No |
| P12 | D8 | 4 | 0 | No |
| P13 | D9 | 5 | 1 | No |
| P14 | D2 | 6 | 2 | No |
| P15 | D13 | 7 | 5 | No |

(NIC index = global rank % 8, corresponding to `mlx5_{2×index}`.) **In this Router launch, none of the 16 pairs shares the same NIC index on both ends, so every KV transfer in this document's main data is cross-rail.** But this is not an inevitable result of Router policy: pairings change with Router restarts (the restart experiment below).

**After a Router restart, the pairings changed** (`pdd-router-0` restarted at 2026-10-13 03:54 UTC, other Pods untouched; 48 requests of 32768 tokens then sent, concurrency 8):

| P rank | P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11 | P12 | P13 | P14 | P15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D rank (after restart) | D3 | D4 | D9 | D1 | D5 | D0 | D2 | D6 | D10 | D14 | D8 | **D11** | **D12** | D15 | D13 | **D7** |
| D rank (before restart, table above) | D5 | D15 | D0 | D4 | D7 | D11 | D3 | D10 | D14 | D12 | D6 | D1 | D8 | D9 | D2 | D13 |

- After the restart it is still a fixed one-to-one mapping; each pair occurred exactly 3 times across the 48 requests; but the mapping was wholly rearranged, and **3 pairs are same-rail** (bold; NIC indices P→D are 3→3, 4→4, 7→7). So "the Router never produces same-rail pairings" held only for the pre-restart launch.
- **The pairing pattern**: the selection order in the Router logs has period 16; the Decode rank rotates strictly in the order D1, D2, D3, …, D15, D0, and the Prefill rank rotates in another fixed order (P3, P6, P0, P1, P4, P7, P15, P10, P2, P8, P11, P12, P14, P9, P13, P5). That is, P and D each run a `round_robin` rotation, advancing one step per request; both lists have length 16 with an order fixed within one launch, so they compose a fixed one-to-one mapping; after a restart the Prefill list's order changed (**Inferred**: the list order comes from the order the Router discovered workers at startup; the `vllm-router` source was not read).
- Note: each `vllm bench` run first sends an "initial single-request test", which advances the rotation one extra step but is not necessarily counted in the Router logs — using log counts to predict which pairing the next request lands on can be off by one step.

<a id="s3-5"></a>
### 3.5 Idle state

- At idle, over 8 seconds, the 8 RoCE NICs of all four Pods sent and received 0 bytes; the 8 GPUs' utilization is all 0%.
- **Each Worker still burns a full CPU core**: all 32 Workers (16 per pool) showed 100–101% CPU usage in a 3-second window; all other processes (`EngineCore`, `APIServer`, `DPCoordinator`) were below 1%. Same phenomenon as 1P1D doc 5.2 (where it was traced to a busy-polling native thread inside the Worker, most likely the NIXL UCX progress thread — not dummy batches); on 2P2D the per-thread check was not redone.

<a id="s4"></a>
## 4. Measurement methods and warm-up

[↑ Back to top](#top)

<a id="s4-1"></a>
### 4.1 What to measure and how

Measurement methods are the same as 1P1D doc Chapter 4 and appendix A.4: TTFT/TPOT via `./pdd.sh bench` (run inside `pdd-router-0`, requests through the Router); NIXL bytes, times, and descriptor counts from P/D `/metrics` (only the Decode side has values); NVLink from before/after differences of `nvidia-smi nvlink -gt d`; RoCE bytes via ioctl reads of `tx/rx_vport_rdma_unicast_bytes` (every 20 ms); RoCE request counts via `rdma statistic`. The 2P2D differences and additions:

- **global rank**: a pool has 16 ranks; the `engine` label of `/metrics` is the in-pool global rank; the r and d in Router logs (`Selected prefill=…@r … decode=…@d`) are in-node ranks that must be converted to global ranks per Pod (the pairing table in Section 3.4 was tallied that way).
- **All 4 GPU Pods must be read simultaneously**: NVLink reads are not atomic, and the window is slightly longer than the request itself; NVLink idle traffic was measured as 0 on 1P1D; it was not separately re-measured on 2P2D.
- **All test traffic is sent through `pdd-router-0`**; no requests are sent from inside Prefill/Decode Pods; reading NVLink and NIC counters does require entering the GPU Pods — read-only, lightweight.
- **Sampling does not perturb the system under test**: reading the NIC counters of the 4 GPU Pods every 20 ms did not affect TTFT (3 runs without sampling: 3.54–3.64 s; 3 runs with sampling: 3.58 s).

<a id="s4-2"></a>
### 4.2 How to separate collectives from KV traffic

Collectives and KV transfer share the same RoCE NICs, so they must be viewed separately:

- **Different RDMA op types**: collectives (NCCL) use RDMA WRITE; KV transfer (NIXL) uses RDMA READ. So `rx_write_requests` mainly reflects collectives, and `rx_read_requests` reflects KV.
- **KV bytes have a fixed location**: one KV transfer adds extra bytes on one NIC of the source node and one NIC of the destination node (equal to the KV size, ~2.35 GB / 32768 tokens); these two NICs are determined by the Router-log pairing and the NIC-GPU mapping. Both places are singled out in this document's tables and subtracted when computing collective-communication volume.
- **Separated in time as well**: collectives happen during prefill and decode; the KV transfer happens after prefill ends, lasts only ~0.2 s, and is a burst of ~50 GB/s on a single NIC.

<a id="s4-3"></a>
### 4.3 Warm-up is mandatory before testing

The rule is the same as 1P1D doc 3.3: **before any performance test, first let every DP rank serve a long request once**. The first long prefill on each rank incurs one-time overhead (~8 s observed on this 2P2D); first-run data is excluded from this document. The procedure on 2P2D:

- Send a batch of long requests through the Router, covering all 16 Prefill ranks and all 16 Decode ranks. The P→D pairing is fixed (Section 3.4), so covering the 16 Prefill ranks covers the 16 Decode ranks too.
- Recommended command (32768 tokens, concurrency 8): `BENCH_NUM_PROMPTS=24 BENCH_REQUEST_RATE=inf ./pdd.sh bench --max-concurrency 8`. In practice, one run of 24 requests covered all 16 pairs (1–2 times each).
- **Confirm coverage from the Router logs**: `kubectl logs pdd-router-0 | grep "Selected prefill"` should show `@0`..`@7` for both Prefill nodes. The rank-selection pattern does not guarantee a batch covers everything — send another batch if it doesn't.
- Warm-up request inputs must be no shorter than the real test (≥ 32768 tokens); a few very short requests (e.g. the 10 tokens of `./pdd.sh test`) do not count as warm-up.
- After warm-up, the same request sent a few times in a row should give stable TTFT (~3.6 s for a single 32768-token request).

Timing data collected before warm-up (TTFT, prefill time, transfer time) is unusable; NVLink and RoCE byte counters are unaffected.

<a id="s5"></a>
## 5. EP collective communication: NVLink and RDMA

[↑ Back to top](#top)

This chapter covers only **EP (expert-parallel) collective communication**: what it uses intra-node, what inter-node, how much of each, and how it uses each rail. KV transfer is Chapter 6; the control-plane connections between the P and D pools are Section 3.3; the interplay between KV traffic and collectives is Chapter 7. Measurement methods in Chapter 4.

<a id="s5-1"></a>
### 5.1 The EP16 communication pattern

Each pool is DP16 + EP16; each MoE layer (58 in total) does one `all_gatherv` (each rank sends its tokens to all 16 ranks of the pool) and one `reduce_scatterv` (summing the ranks' expert outputs for "my tokens"). These two function names are the vLLM source's own (the `v` means variable-length; the ranks' token counts may differ). For each rank, of the other 15 ranks, 7 are on the same node and 8 on the other node:

| Peer | Count | What it uses |
|---|---|---|
| Same-node ranks | 7 | **NVLink** (`NV18` full interconnect) |
| Other-node ranks | 8 | **RoCE, RDMA WRITE**, via the NIC nearest this rank (GPU i ↔ `mlx5_{2i}`) |

Ranks without requests still co-run dummy batches (reason in 1P1D doc 2.3), so as soon as one request exists in the pool, all 32 GPUs have non-zero collective traffic (5.2).

<a id="s5-2"></a>
### 5.2 Collective-communication volume of one 32768-token request

One 32768/128-token request was sent on a warmed-up, idle 2P2D (`./pdd.sh bench`, via `pdd-router-0`). The request landed on `pdd-prefill-1` rank 7 (P15) and `pdd-decode-1` rank 5 (D13); TTFT 3.58 s, TPOT 21.0 ms, prefill time 3.1 s.

**NVLink** (Tx summed over the 8 GPUs in each Pod, GiB)

| Pod | Total | Per GPU |
|---|---|---|
| `pdd-prefill-0` | 330.6 | 39.9, 41.4, 41.4, 41.5, 41.6, 41.6, 41.6, 41.6 |
| `pdd-prefill-1` | 293.1 | 29.4, 41.4, 41.4, 41.5, 41.6, 41.6, 41.6, 14.6 |
| `pdd-decode-0` | 36.5 | 5.2, 3.4, 3.3, 3.8, 5.2, 5.2, 5.2, 5.2 |
| `pdd-decode-1` | 36.5 | 5.2, 3.4, 3.3, 3.8, 5.2, 5.2, 5.2, 5.2 |

- **The request landed on a single rank, yet all 4 Pods and all 32 GPUs had NVLink traffic**: the other ranks co-run — EP16's collectives engage the whole pool.
- Prefill's traffic is ~8–9× Decode's: a prefill step processes tens of thousands of tokens; a decode step only a few.
- `pdd-prefill-1`'s GPU 0 and GPU 7 were low (29.4, 14.6 GiB); cause not determined.

**RoCE** (bytes sent/received per NIC during the request, MB; NIC indices 0..7 ↔ `mlx5_0,2,…,14`)

| Pod | tx (send) | rx (receive) |
|---|---|---|
| `pdd-prefill-0` | 5808, 5879, 5816, 5649, 5654, 5649, 5644, 3725 (total 43.8 GB) | 5803, 5867, 5824, 5649, 5668, 5660, 1928, 5637 |
| `pdd-prefill-1` | 5803, 5867, 5826, 5649, 5676, 5667, 1928, **7990** (total 44.4 GB) | 5808, 5877, 5816, 5648, 5653, 5649, 5644, 3737 |
| `pdd-decode-0` | 2289, 2243, 1707, 0, 0, 0, 0, 0 (total 6.2 GB) | 2290, 2244, 1707, 0, 0, 0, 0, 0 |
| `pdd-decode-1` | 2290, 2244, 1707, 0, 0, 0, 0, 0 | 2289, 2243, 1707, 0, 0, **2352**, 0, 0 |

- The two bolded entries are this request's KV transfer (Section 4.2): `pdd-prefill-1`'s NIC 7 sent 7990 MB — about 2.35 GB more than that Pod's other NICs (~5650); `pdd-decode-1`'s NIC 5 received 2352 MB, while that Pod's NICs 0–2 carried only collective traffic (1707–2289 MB) and the rest 0. **After subtracting the KV**, each Prefill node's collective send is ~42–44 GB, each Decode node's ~6.2 GB.
- `rx_write_requests` deltas: the two Prefill nodes are identical — NIC 0 ~136k, NICs 1 and 2 ~75k, NICs 3, 4, 5 ~26k, NIC 6 ~20k, NIC 7 ~23k; the two Decode nodes have only NICs 0, 1, 2 (1M, 500k, 500k).

<a id="s5-3"></a>
### 5.3 Division of labor between NVLink and RoCE

| | NVLink (intra-node) | RoCE (inter-node, KV subtracted) | RoCE as % of NVLink bytes |
|---|---|---|---|
| Per Prefill node | 293–331 GiB (~315–355 GB) | ~42–44 GB | ~12% |
| Per Decode node | 36.5 GiB (~39 GB) | ~6.2 GB | ~16% |

- **The same MoE layer's communication uses both NVLink and RoCE simultaneously**; NVLink carries most of the bytes.
- RoCE's byte count is far below the naive estimate of "each rank sending its data to 8 remote ranks", consistent with a hierarchical algorithm of "each piece of inter-node data crosses the network once, then is forwarded within the node over NVLink" (**Inferred**; NCCL's algorithm choice was not read).

<a id="s5-4"></a>
### 5.4 Inter-node collectives are rail-aligned

Comparing the two Prefill nodes' same-index NIC send/receive bytes (the 5.2 table):

| NIC index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `prefill-0` send | 5808 | 5879 | 5816 | 5649 | 5654 | 5649 | 5644 | 3725 |
| `prefill-1` receive | 5808 | 5877 | 5816 | 5648 | 5653 | 5649 | 5644 | 3737 |
| `prefill-1` send (after subtracting the ~2.35 GB KV on NIC 7) | 5803 | 5867 | 5826 | 5649 | 5676 | 5667 | 1928 | ~5640 |
| `prefill-0` receive | 5803 | 5867 | 5824 | 5649 | 5668 | 5660 | 1928 | 5637 |

- The number of bytes node 0's NIC i sends **is almost exactly the number node 1's NIC i receives** (differences within a few dozen MB), and vice versa; likewise between the two Decode nodes (`decode-0`'s tx matches `decode-1`'s rx NIC by NIC). So **inter-node collectives are rail-aligned: NIC i ↔ NIC i, no cross-rail**.
- Each GPU talks to the peer node's same-index GPU via its nearest NIC (GPU i ↔ `mlx5_{2i}`, same PCIe bridge); collective traffic therefore does not need the spine layer's cross-rail paths (**Inferred**; switch topology not read).
- Asymmetry: in this request, `prefill-1`'s NIC 6 sent only 1.9 GB and `prefill-0`'s NIC 7 only 3.7 GB (others ~5.6 GB); per-direction byte counts are not all equal; cause not determined (channel counts and ring/tree structure were not read from NCCL logs; not checked).

<a id="s5-5"></a>
### 5.5 Prefill uses 8 rails, Decode only 3

- Prefill pool: all 8 NICs carry collective traffic, ~5.6–5.9 GB each (for one 32768-token request).
- Decode pool: **only NICs 0, 1, 2 (`mlx5_0,2,4`) carry collective traffic** (sending 2.3, 2.2, 1.7 GB); NICs 3–7 carry none (0 bytes). Same in the 24-request concurrency experiment (5.7).
- Cause not determined. **Inferred**: Decode has very few tokens per step, so NCCL uses only a few channels for small messages, occupying only the first few rails; Prefill's messages are large and use all 8. NCCL's channel counts were not read.
- Implication for KV transfer: when the Decode rank index maps to NICs 0–2 (D0, D1, D2, D8, D9, D10), the KV lands on a NIC that also carries Decode collectives; whether this affects KV transfer speed is Chapter 7 (no effect was seen).

<a id="s5-6"></a>
### 5.6 Traffic over time: RoCE is nowhere near saturated

Aggregating the 20 ms samples into 1-second bins, this single request's send rates (GB/s; the value in parentheses is the single-NIC maximum):

| Phase | Per Prefill node (8 NICs summed) | Per Decode node (8 NICs summed) |
|---|---|---|
| prefill (~3 s) | peak 18.2 (single NIC ≤ 4.0); other seconds 10–11 (≤ 2.8) | 0 |
| decode (128 tokens, ~2.7 s) | 0 | ~2.0 (single NIC ≤ 0.8) |

- Collectives are **bursty**, and **a single NIC's rate is only 2–4 GB/s, under 10% of the 50 GB/s line rate**; during decode a single NIC peaks at 0.8 GB/s.
- So the collectives are not RoCE-bandwidth-bound (**Inferred**: they are bound by the number and latency of synchronizations among the 16 ranks; not decomposed and verified).
- For comparison: at the moment of a KV transfer, a single NIC reaches ~50 GB/s. Collectives and KV bursts differ by an order of magnitude in bandwidth — this is the basis of Chapter 7's conclusion.

<a id="s5-7"></a>
### 5.7 Collective-communication volume under concurrency

24 requests of 32768 tokens, concurrency 8 (the Chapter 7 experiment):

| | Per-node total | Per request | Single-request (5.2) comparison |
|---|---|---|---|
| Prefill node NVLink | 7094.9 / 7168.4 GiB | 295.6 / 298.7 GiB | 293–331 GiB |
| Prefill node RoCE send (incl. KV) | 1035.4 / 978.3 GB | after subtracting the 24 KV transfers (~56 GB) from the two nodes' totals, ~40.8 GB per node | 42–44 GB |
| Decode node NVLink | 275.8 GiB (identical on both nodes) | 11.5 GiB | 36.5 GiB |
| Decode node RoCE send (NICs 0, 1, 2) | 17.3 + 17.0 + 12.9 = 47.2 GB | 1.97 GB | 6.2 GB |

- **Per-request collective volume in the prefill phase is essentially independent of concurrency** (NVLink ~296–299 GiB, RoCE ~41 GB per node), because a prefill's token count is fixed and every token goes through these communications.
- **Per-request collective volume in the decode phase drops sharply with concurrency** (NVLink 11.5 vs. 36.5 GiB, RoCE 1.97 vs. 6.2 GB — about 1/3): concurrent requests share decode steps, and one step's collectives are amortized across all requests in that step (**Inferred**, from the numbers; not verified step by step).
- Under concurrency, Decode collectives still use only NICs 0, 1, 2; NICs 3–7 sent 0.
- Under concurrency, the two Prefill nodes' 8-NIC summed send rate peaked at ~113–116 GB/s (including KV bursts); the collectives alone remained a few GB/s.

<a id="s6"></a>
## 6. KV transfer: cross-node, cross-rail

[↑ Back to top](#top)

<a id="s6-1"></a>
### 6.1 Four transfers of a large KV (131072 tokens)

After warm-up, 4 requests of 131072/16 tokens were sent sequentially (each KV ~9.21 GB). `/metrics` and `rdma statistic` were read before and after each request, and all 8 NICs' send/receive bytes were sampled every 20 ms throughout (Section 4.1).

| Request | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Prefill rank → Decode rank (global) | P4 → D7 | P13 → D9 | P5 → D11 | P15 → D13 |
| In-node location | `prefill-0` r4 → `decode-0` r7 | `prefill-1` r5 → `decode-1` r1 | `prefill-0` r5 → `decode-1` r3 | `prefill-1` r7 → `decode-1` r5 |
| Prefill-side sending NIC | `mlx5_8` | `mlx5_10` | `mlx5_10` | `mlx5_14` |
| Decode-side receiving NIC | `mlx5_14` | `mlx5_2` | `mlx5_6` | `mlx5_10` |
| NIC indices (P → D) | 4 → 7 | 5 → 1 | 5 → 3 | 7 → 5 |
| Other NICs with >10 GB/s traffic | none | none | none | none |
| NIXL bytes transferred | 9,210,691,584 | same | same | same |
| NIXL descriptor count | 62,769 | 62,769 | 62,769 | 732 |
| NIXL transfer time (`xfer_time`) | 193.0 ms | 222.9 ms | 194.2 ms | 191.4 ms |
| Bytes ÷ `xfer_time` | 47.7 GB/s | 41.3 GB/s | 47.4 GB/s | 48.1 GB/s |
| Post time (`post_time`) | 2.3 ms | 2.8 ms | 2.7 ms | 1.7 ms |
| Prefill-side NIC send burst >10 GB/s | 9,353 MB / 0.201 s / peak 51.0 GB/s | 9,322 MB / 0.221 s / 49.3 | 9,376 MB / 0.201 s / 49.8 | 9,386 MB / 0.201 s / 53.1 |
| Decode-side NIC receive burst >10 GB/s | 9,392 MB / 0.201 s / 50.8 | 9,405 MB / 0.223 s / 52.4 | 9,026 MB / 0.181 s / 51.1 | 9,386 MB / 0.201 s / 50.8 |
| Prefill-side `rx_read_requests` delta (write requests unchanged) | 151,768 (`mlx5_8`) | 151,768 (`mlx5_10`) | 151,768 (`mlx5_10`) | 36,234 (`mlx5_14`) |
| Prefill's prefill time | 35.37 s | 34.70 s | 34.81 s | 35.14 s |
| Queue time recorded by Decode | 0.219 s | 0.245 s | 0.218 s | 0.223 s |
| TTFT | 37.1 s | 36.4 s | 36.4 s | 36.9 s |

Notes:
- Burst durations are determined by the 20 ms sampling, ±20 ms; peaks above 50 GB/s (up to 53.1) are jitter from sampling boundaries, not above line rate.
- During each transfer, **only one Prefill NIC and one Decode NIC** show >10 GB/s traffic; none of the others.
- The descriptor count depends on how contiguous the KV block allocation is: requests 1–3 had 62,769 (block size 32 tokens; 131072 / 32 × 61 = 249,856 is the fully fragmented upper bound, and 62,769 is about a quarter of it, meaning roughly every 4 blocks merge into one descriptor); request 4 only 732 (nearly contiguous). Regardless of fragmentation, transfer time stays within 191–223 ms, and post time is only 1.7–2.8 ms.
- The `rx_read_requests` count depends on the descriptors' count and size: 732 descriptors correspond to 36,234, and 62,769 to 151,768 (for the same KV size).

<a id="s6-2"></a>
### 6.2 Cross-rail or not, nearest NIC or not, READ or WRITE

- **Cross-rail**: the four transfers' NIC indices were 4→7, 5→1, 5→3, 7→5 — **all different**; in Section 3.4's complete pairing table, none of the 16 pairs match either. KV transfers necessarily traverse cross-rail paths, still reaching 49–53 GB/s peaks — so cross-rail itself has no visible speed penalty; the same-rail samples are in 6.4 (only 1 caught after the Router restart).
- **Nearest NIC**: Prefill rank r always uses `mlx5_{2(r%8)}` and Decode rank d always uses `mlx5_{2(d%8)}`, on the same PCIe bridge as the GPU (`PXB`). The 4 large-KV transfers plus the single-request one make 5 transfers, covering 5 distinct Prefill ranks and 5 distinct Decode ranks — all consistent; no other NIC was ever used. The KV cache is registered on the rank's own NIC (checked in 1P1D doc 5.4 via `rdma resource show mr`; not re-read on 2P2D).
- **READ**: the Prefill side's `rx_read_requests` increases and `rx_write_requests` does not grow from KV (the collectives' WRITE counted separately — see Chapter 5); the wire direction is Prefill sending, Decode receiving. As in 1P1D, Decode pulls from Prefill's GPU memory via RDMA READ (1P1D doc 2.5, 5.4).
- **Cross-node**: the P pool and the D pool are always on different nodes (different Pods), so every KV transfer is cross-node.

<a id="s6-3"></a>
### 6.3 Performance

- A 9.21 GB transfer: `xfer_time` ~191–223 ms (41–48 GB/s); the wire send burst lasts ~0.18–0.22 s with peaks of 49–53 GB/s, close to a single NIC's 400 Gb/s (50 GB/s) line rate.
- **`xfer_time` ≈ `post_time` + wire time**: `post_time` is only 1.7–2.8 ms (~0.04 μs per descriptor, same as 1P1D); the wire transfer is ~190–220 ms.
- The transfer is only ~0.5% of TTFT: for 131072 tokens, TTFT 36–37 s, of which Prefill compute 34.7–35.4 s, KV transfer ~0.2 s, Decode queueing ~0.22–0.25 s. For 32768 tokens (2.3 GB), `xfer_time` is ~49–64 ms (5 samples: the two with 61 descriptors were 54.3 and 63.7 ms; the three with 62,464 were 48.9, 52.2, 52.5 ms).
- Request 2's `xfer_time` (222.9 ms) is ~30 ms higher than the other three; cause not determined.

<a id="s6-4"></a>
### 6.4 After the Router restart: same-rail vs. cross-rail

After the Router restart there were 3 same-rail pairs (P11→D11, P12→D12, P15→D7, Section 3.4), so an attempt was made to steer each 131072/16-token request onto chosen pairings with small requests (`max_tokens=1`), to compare same-rail vs. cross-rail KV transfer. **The steering mostly failed**: the initial test request `vllm bench` sends first advanced the rotation one extra step, which my count from the logs had not accounted for, so only 1 of the 12 large requests landed on a same-rail pairing (P12→D12, by luck). By the time the experiment finished, the 2P2D had been deleted; there was no chance to redo it. Below are the measurements of these 12 transfers (after warm-up, in the post-Router-restart state):

| Request | P→D | NIC indices P→D | Same-rail? | Descriptor count | `xfer_time` | Bytes ÷ `xfer_time` | `post_time` | Decode-side receive burst duration | Prefill's prefill time |
|---|---|---|---|---|---|---|---|---|---|
| 1 | P10→D8 | 2→0 | No | 61 | 230.6 ms | 39.9 GB/s | 0.1 ms | 0.241 s | 35.55 s |
| 2 | P14→D13 | 6→5 | No | 249,551 | 201.2 ms | 45.8 GB/s | 13.4 ms | 0.201 s | 34.37 s |
| 3 | P13→D15 | 5→7 | No | 125,477 | 190.7 ms | 48.3 GB/s | 5.5 ms | 0.201 s | 34.70 s |
| 4 | P12→D12 | 4→4 | **Yes** | 249,856 | 205.3 ms | 44.9 GB/s | 11.6 ms | 0.201 s | 35.58 s |
| 5 | P13→D15 | 5→7 | No | 249,856 | 203.2 ms | 45.3 GB/s | 10.7 ms | 0.201 s | 34.72 s |
| 6 | P0→D3 | 0→3 | No | 62,525 | 190.3 ms | 48.4 GB/s | 3.0 ms | 0.201 s | 34.90 s |
| 7 | P9→D14 | 1→6 | No | 249,856 | 198.4 ms | 46.4 GB/s | 11.9 ms | 0.201 s | 35.40 s |
| 8 | P0→D3 | 0→3 | No | 249,856 | 197.7 ms | 46.6 GB/s | 11.7 ms | 0.201 s | 34.90 s |
| 9 | P4→D5 | 4→5 | No | 187,453 | 196.6 ms | 46.9 GB/s | 10.5 ms | 0.188 s | 35.37 s |
| 10 | P6→D2 | 6→2 | No | 187,270 | 206.7 ms | 44.6 GB/s | 10.8 ms | 0.201 s | 34.31 s |
| 11 | P4→D5 | 4→5 | No | 187,270 | 192.0 ms | 48.0 GB/s | 6.7 ms | 0.201 s | 35.40 s |
| 12 | P2→D9 | 2→1 | No | 249,856 | 229.1 ms | 40.2 GB/s | 13.5 ms | 0.241 s | 35.31 s |

Observations
- **The 1 same-rail transfer (request 4) is indistinguishable from cross-rail**: `xfer_time` 205.3 ms, wire burst 0.201 s, peaks 49.7 (Prefill send) / 52.5 GB/s (Decode receive) — inside the cross-rail ranges (190.3–206.7 ms, 0.188–0.201 s). With only 1 sample, one can only say no difference was seen, not that same-rail is as fast as cross-rail.
- The other 11 were cross-rail: `xfer_time` 190.3–230.6 ms, Decode-side receive bursts 0.188–0.241 s. Each transfer's post time is proportional to its descriptor count (~0.04–0.05 μs each).
- **Transfers landing on `pdd-decode-1`'s NICs 0 and 1 (D8, D9) were slower**: request 1 (D8) 230.6 ms and request 12 (D9) 229.1 ms, plus request 2 in 6.1 (D9) 222.9 ms — Decode-side receive bursts of 0.223–0.241 s, about 15% longer than the other transfers (190.3–206.7 ms, 0.188–0.201 s; ~40 GB/s vs. ~47 GB/s). These three differ in both Prefill-side NIC and descriptor count; only the Decode-side NIC index (0 or 1) is shared. Cause not determined; NICs 0, 1 happen to be the rails carrying most of the Decode collectives, but Decode was idle when these transfers happened, so it is not collective traffic. Only 3 samples, and `pdd-decode-0`'s NICs 0, 1 (D0, D1) were not tested — no conclusion can be drawn.
- **The 1P1D same-rail samples**: the redeployed 1P1D's pairings included 1 same-rail pair (P6→D6), which received 6 transfers in the concurrency experiments (3 in each of 2 runs), averaging 67.8 ms; the other 42 cross-rail transfers averaged 60.3 ms — no stable difference (the same engine can swing from 50 ms to 85 ms between runs); the wire bursts were also the same (~60 ms). Taking 2P2D's 1 sample together with 1P1D's 6, **no difference was observed between same-rail and cross-rail KV transfer speed**.

<a id="s7"></a>
## 7. Do collectives affect KV transfer?

[↑ Back to top](#top)

**Experiment**: after warm-up, 24 requests of 32768/128 tokens were sent with concurrency 8 (`BENCH_NUM_PROMPTS=24 BENCH_REQUEST_RATE=inf ./pdd.sh bench --max-concurrency 8`). At this point the Prefill pool's two nodes were prefilling multiple requests simultaneously (heavy collective traffic on RoCE), the Decode pool was decoding at the same time, and the KV transfers overlapped with these collectives. The control is the earlier single-request KV transfers (when idle, only KV is on RoCE).

**Overall results**

| | Value |
|---|---|
| Requests | all 24 succeeded, 786,432 input tokens in total |
| Duration / throughput | 34.6 s, 0.69 req/s, 22.8k tokens/s (input + output) |
| TTFT | mean 8.1 s, median 8.5 s, P99 9.1 s |
| TPOT | mean 22.3 ms, P99 23.1 ms |
| Prefill-node RoCE | each NIC sent 118–132 GB and received 115–129 GB (collectives + KV); the two nodes' 8-NIC summed send rate peaked at 116.3 / 112.8 GB/s |
| Decode-node RoCE | NICs 0, 1, 2 sent 17.3 / 17.0 / 12.9 GB; NICs 3–7 sent 0 and only received KV (2.3–4.7 GB each) |
| Peak one-way rate per NIC | Prefill 47.3–50.3 GB/s, Decode 43.7–51.2 GB/s (these are the KV bursts) |

**NIXL-recorded transfers**

| | 24 transfers, concurrent | Single-request (32768 tokens) control |
|---|---|---|
| Mean `xfer_time` | **71.9 ms** (`decode-0` 71.6, `decode-1` 72.4) | 48.9–63.7 ms (5 samples, mean ~54 ms) |
| Mean `post_time` | 1.5 ms | 0.1–3.0 ms |
| Mean descriptor count per transfer | 33,343 / 41,710 | 61 or 62,464 |
| Bytes ÷ cumulative `xfer_time` | 32.0 GB/s | ~36–47 GB/s |
| `xfer_time` distribution | 18 ≤ 75 ms, 21 ≤ 100 ms, 24 ≤ 200 ms | all 5 ≤ 75 ms |

**Wire speed** (measured by the Decode-side NICs' receive bursts, where the received traffic is almost exclusively KV; each burst ≥ 0.5 GB)

| | Concurrent (n = 24) | Single request (n = 5: 1 single-request run + 4 sequential-request runs) |
|---|---|---|
| NICs 0–2 (also carrying Decode collectives) | n = 9, median burst duration 60 ms (60–80), median average wire rate 39.1 GB/s (29.8–39.4) | n = 1: 80 ms, 29.8 GB/s |
| NICs 3–7 (KV only) | n = 15, median burst duration 60 ms (60–60), median average wire rate 39.0 GB/s (35.8–39.1) | n = 4: 60 ms, 38.4–39.1 GB/s |
| Prefill-side NIC send bursts (n = 24) | median 60 ms (60–80), average wire rate 39.1 GB/s (31.7–42.0) | same, ~60 ms |

Conclusions
- **Wire speed is essentially unaffected by the collectives**: under concurrency, each KV burst's duration (median 60 ms) and average wire rate (~39 GB/s) match the single-request case; even when the KV lands on NICs 0–2, which simultaneously carry Decode collectives (n = 9), there is no difference from the clean NICs 3–7 (n = 15). Reason: a collective's per-NIC rate is only 2–4 GB/s (5.6), while a KV burst occupies 40–50 GB/s for under 0.1 s — the two barely conflict in time or bandwidth.
- **The NIXL-recorded transfer time grew ~30%** (72 ms vs. ~54 ms), while the wire time did not change (~60 ms) and `post_time` was only 1.5 ms — so the extra ~12–18 ms is neither in the posting phase nor on the NIC. **The 1P1D control (below) shows this is not caused by the collectives**: 1P1D has no collectives on RoCE, yet its transfers grow similarly under concurrency. The likely cause is that under concurrency the Decode engine is continuously decoding, so completion notifications wait for the engine's next step boundary (**Inferred**, unverified).
- Descriptors grew from 61 into the tens of thousands, yet `post_time` stayed at 1.5 ms — descriptor fragmentation itself is not what slows the transfer.
- **Direction not tested**: whether KV READ traffic conversely slows the collectives on the same NIC (e.g. lengthening prefill steps) was not measured.

**Control experiment: 1P1D (no collectives on RoCE)**

The same experiment was run on the redeployed 1P1D (detailed data in 1P1D doc 5.7; after warm-up, 24 requests of 32768/128 tokens, concurrency 8, run twice; single request as baseline):

| | Single request (baseline) | 24 concurrent (run 1) | 24 concurrent (run 2) |
|---|---|---|---|
| 1P1D: mean NIXL transfer time | 48.2–49.3 ms | **65.5 ms** | **57.0 ms** |
| 1P1D: wire burst (Decode side) median | 60 ms, 39.1 GB/s | 60 ms, 39.0 GB/s | 60 ms, 39.0 GB/s |
| 1P1D: collectives on RoCE | none | none (each NIC carried only 3 KV transfers, 7051–7057 MB) | none |
| 2P2D: mean NIXL transfer time | ~54 ms | **71.9 ms** | — |

- **1P1D's NIXL transfer time also grows under concurrency** (49 ms → 57–65 ms, +16–34%), with unchanged wire speed. So the growth under concurrency is **mainly not caused by RoCE collectives** — it is concurrency itself.
- 2P2D (71.9 ms) is 6–15 ms higher than 1P1D (57.0–65.5 ms), but 1P1D's two runs alone differ by 8.5 ms, so this part cannot be attributed to the collectives, nor ruled out.
- The wire speed is identical on 1P1D and 2P2D, so the RoCE collectives have no measurable effect on the KV wire transfer.

<a id="s8"></a>
## 8. Comparison with 1P1D

[↑ Back to top](#top)

**What 1P1D is**: Prefill and Decode each with 1 node, each pool an 8-GPU DEP group (DP8 + EP8), EP collectives entirely intra-node (NVLink), and only P→D KV transfer on RoCE; deployment, principles, observation methods, and single-pool measurements are in the 1P1D doc. 2P2D expands each pool to 2 nodes (DP16 + EP16); what is new is inter-node EP collectives, so the differences between the two mainly are: whether collectives use RoCE, per-step decode latency, and concurrent throughput.

| | 1P1D | 2P2D |
|---|---|---|
| Pool size | 8 GPUs (DP8 + EP8) | 16 GPUs (DP16 + EP16, across 2 nodes) |
| Experts per rank | 32 | 16 |
| `init engine` time | Prefill ~1098 s, Decode ~514 s | Prefill ~1144 s, Decode ~531 s |
| 32768-token TTFT | 3.51–3.55 s | 3.55–3.71 s |
| 32768-token TPOT | 16.1–16.3 ms (17.3–17.4 ms at concurrency 8) | 20.8–21.3 ms (22.3 ms at concurrency 8) |
| 131072-token prefill time | 34.1–35.2 s (two deployments) | 34.70–35.37 s |
| NVLink traffic per Prefill node (32768-token request) | ~278 GiB | 293–331 GiB |
| NVLink traffic per Decode node (same) | 19.2–19.5 GiB | 36.5 GiB |
| Collective traffic on RoCE | none | Prefill ~42–44 GB per node, Decode ~6.2 GB per node |
| 9.21 GB KV transfer | 190–198 ms (46–48 GB/s) | 191–223 ms (41–48 GB/s) |
| KV transfer post time | 0.1–13.4 ms (0.04 μs / descriptor) | 1.7–2.8 ms (0.04 μs / descriptor) |
| P–D rank pairing | 8 fixed per launch; all cross-rail in the first deployment, 1 same-rail pair after the redeployment | 16 fixed per launch; all cross-rail in this document's main-data launch, 3 same-rail pairs after the Router restart |
| 24 × 32768-token requests, concurrency 8: duration / throughput | 28.6–28.7 s, 0.84 req/s, 27.5–27.6k tok/s | 34.6 s, 0.69 req/s, 22.8k tok/s |
| same: mean TTFT | 6.85 s | 8.08 s |
| same: mean NIXL transfer time | 57.0–65.5 ms | 71.9 ms |
| same: NVLink per request (Prefill / Decode) | 275.8 / 5.2–5.4 GiB | 296–299 / 11.5 GiB (per node) |
| Idle Worker CPU | 100% | 100% |

- **2P2D's prefill speed is essentially the same as 1P1D's**: although EP16's collectives span 2 nodes and RoCE's byte count is not small (~43 GB per Prefill node), single-request TTFT is only 3.55–3.71 s vs. 3.5 s. 2P2D's cost shows up in decode: TPOT ~21 ms is ~5 ms more than 1P1D's 16 ms, and Decode's NVLink traffic is nearly 2× (each step's collectives go cross-node).
- **Why TPOT is ~5 ms higher (what the data supports, and what it doesn't)**:
  - **Observed**: single-request TPOT 16.1–16.3 ms vs. 20.8–21.3 ms; at concurrency 8, 17.3–17.4 ms vs. 22.3 ms — a ~5 ms gap under both loads; the Decode pool's collectives in 2P2D go over RoCE (NICs 0–2, single-NIC peak only 0.8 GB/s, far from saturated), while in 1P1D they are entirely within NVLink; yet TTFT for the same prefill (large messages) shows no difference.
  - **Consistent with "cross-node communication latency lengthens each step"**: the gap does not change with concurrency, so it looks like fixed extra time per decode step rather than compute growing with token count; RoCE bandwidth is far from used up, so it is not a bandwidth issue; prefill's messages are large and its compute long, so communication latency is hidden — hence TTFT is unaffected. A rough conversion: 58 layers × 2 collectives = 116 per step, so 5 ms spread over each is ~40 μs (**Inferred** — this is division, not measurement).
  - **But this cannot be asserted**: per-layer collective times were not measured, and "cross-node" was not separated from the other changes EP16 brings — which equally slow each step: the collective's participating ranks grow from 8 to 16, experts per rank drop from 32 to 16, and the per-step DP metadata sync (Gloo over TCP via eth0, 1P1D doc 2.3) also goes cross-node. "Compute is not the bottleneck" is likewise only inferred from Decode's few tokens per step; GPU compute time was not measured. Verifying this needs per-layer timing (e.g. timing each `all_gatherv`/`reduce_scatterv` in the NCCL layer or a profiler) or a same-node DP16 control — not done in this environment.
- **At concurrency 8, 1P1D is faster than 2P2D**: for the same 24 requests, 1P1D 28.6–28.7 s (0.84 req/s) vs. 2P2D 34.6 s (0.69 req/s); TTFT (6.85 vs. 8.08 s) and TPOT (17.4 vs. 22.3 ms) also favor 1P1D. 2P2D's pools have 16 ranks each and more capacity, but this shows no advantage at this concurrency; whether 2P2D overtakes 1P1D at higher concurrency (e.g. 16, 32) was not tested.
- **KV transfer**: identical wire speed (burst medians of 60 ms at ~39 GB/s under concurrency on both); under concurrency, the NIXL-recorded transfer time grows on both (1P1D +16–34%, 2P2D ~+33%), so the growth is mainly not caused by the collectives (Chapter 7).

<a id="s9"></a>
## 9. Verification and unverified items

[↑ Back to top](#top)

**Verified this time**

| Claim | Result |
|---|---|
| No YAML/script changes needed | **Confirmed** (deployed with default node counts, files unmodified, the Router auto-discovered 32 workers, `metrics` aggregated 16 engines) |
| `DPCoordinator` only on node 0 | **Confirmed** (compared `ps`) |
| Control-plane connections within a pool, no TCP between P and D | **Confirmed** (398 vs. 0) |
| The Router only connects to the Head | **Refuted**: the Router connects to all 4 GPU Pods; invisible at idle because the connections are short-lived (3.3) |
| P→D pairings are 16×16 | **Refuted**: only 16 fixed pairs; the pairing rearranges per Router launch, and same-rail pairs appeared after the restart (3.4) |
| EP collectives use both NVLink and RoCE | **Confirmed** (NVLink counters, RoCE bytes, and `rx_write_requests`) |
| Inter-node collectives are rail-aligned (NIC i ↔ NIC i) | **Confirmed** (per-NIC byte counts match between the two Prefill nodes and the two Decode nodes) |
| The Decode pool uses only 3 rails for collectives | **Confirmed** (phenomenon); the cause is inferred |
| KV transfer is RDMA READ, one NIC per side, nearest NIC, all cross-rail | **Confirmed** (4 large-KV runs + 1 single request + the 16 pairings) |
| The collectives' RoCE traffic slows the KV wire transfer | **Not observed** (24 concurrent runs; wire speed same as single-request) |
| The collectives lengthen the NIXL-recorded transfer time | **Mainly not**: ~30% growth under concurrency (72 ms vs. ~54 ms); 1P1D (no collectives on RoCE) grows 16–34% under the same concurrency; wire time unchanged on both; whether there is a ≤10 ms contribution cannot be determined (Chapter 7) |

**Unverified**
- Same-rail transfer speed: only 1 sample caught on 2P2D and 6 on 1P1D, none showing a difference vs. cross-rail (6.4); the samples are still few, and 1P1D has sizable run-to-run variance.
- The reverse effect of KV READ traffic on the collectives (prefill step time).
- NCCL's algorithm and channel counts, and why Decode uses only 3 rails (NCCL logs not read; setting `NCCL_DEBUG` requires changing the YAML, which was not done this time).
- The source of the extra ~12–18 ms in NIXL-recorded transfer time under concurrency (the 1P1D control shows it is mainly not the collectives, Chapter 7).
- Whether same-index NICs are actually on the same rail on the switches (assumed).
- Why the Router's pairing order changes after a restart (only two fixed-rotation lists for P and D were observed; the `vllm-router` source was not read, 3.4).
- Why KV transfers landing on `pdd-decode-1`'s NICs 0, 1 were ~15% slower, and whether `pdd-decode-0`'s NICs 0, 1 behave the same (6.4).
- Why Prefill initializes ~2× slower than Decode; why individual GPUs' NVLink traffic was low in `pdd-prefill-1`; why individual NICs' send bytes were asymmetric in 5.4; why request 2's `xfer_time` in 6.1 was high.
- The identity of the thread burning a CPU core in idle Workers on 2P2D (only the CPU usage was confirmed; the per-thread check was not redone).
- A systematic sweep of concurrency and input length (only concurrency 8 was tested; whether 2P2D overtakes 1P1D at higher concurrency), and P/D ratios (e.g. 1P+3D).
- Creation, rebuild, and deletion of the deployment are up to the user; changing node counts or any switch requires `down` then `up`.

<a id="s10"></a>
## 10. Appendix: raw output and helper scripts

[↑ Back to top](#top)

Read with `kubectl exec <pod> -- ps -ef --width 250` on 2026-10-13 02:14 UTC, Pods up ~39 minutes, 2P2D idle. No lines were removed; command lines were truncated to 250 columns; the trailing `bash -c date ...`, `ps -ef`, `cut` are processes created by the collection command itself; the `/bin/bash` attached to `pts/0` inside `pdd-prefill-0` and `pdd-prefill-1` is an interactive `kubectl exec` session with a TTY, not created by this collection.

### A.1 `pdd-prefill-0` (node 0, has `DPCoordinator`)

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

### A.2 `pdd-prefill-1` (node 1, no `DPCoordinator`)

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

### A.3 `pdd-decode-0` (node 0, has `DPCoordinator`)

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

### A.4 `pdd-decode-1` (node 1, no `DPCoordinator`)

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
### A.5 Helper scripts

The 2P2D data used the same read-only script set as 1P1D doc appendix A.4 (collection scripts: NVLink counters, RoCE byte sampler, RDMA request counters, NIXL metrics; experiment drivers `exp_window.sh`, `exp_per_request.sh`; analysis scripts `analyze_window.py`, `analyze_per_request.py`), used on multiple nodes like this:

- **Driver scripts**: set `PODS="pdd-prefill-0 pdd-prefill-1 pdd-decode-0 pdd-decode-1" PN=2 DN=2`, e.g. `LABEL=conc8 PODS="…" PN=2 DN=2 BENCH_INPUT_LEN=32768 BENCH_ARGS="--num-prompts 24 --max-concurrency 8" ./exp_window.sh`. The script saves `{Pod IP: Pod name}` into `ipmap.json` at experiment start.
- **Analysis scripts**: auto-detect the 4 Pods from the files in the directory; use `ipmap.json` to convert the Router logs' `ip@in-node rank` into global ranks (P0..P15, D0..D15); the NIC indices in the output are global rank % 8. For old data without `ipmap.json`, specify with `--ipmap ip=pod,ip=pod`. `analyze_window.py` also splits the Decode KV bursts by NIC into NICs 0–2 (also carrying Decode collectives) and NICs 3–7 (KV only) (`--split` changes the boundary) — the Chapter 7 comparison was done this way.

Scripts additionally used for 2P2D:

| Script | Used in | Purpose |
|---|---|---|
| `router_pairs.py` | 3.4 | tallies the P→D pairing table from Router logs (global ranks, occurrence counts, NIC indices on both ends, same-rail or not) and the selection order; pairings change per Router restart, so only logs from a single Router run can be used |
| `tcp_peers.sh <pod>…` | 3.3 | counts ESTABLISHED TCP connections from each GPU Pod to each peer Pod (intra-pool, cross-pool) |
| `idle_cpu.py` | 3.5 | per-Worker CPU usage at idle |
| `sampler_ab.sh` | 4.1 | runs a few bench runs with and without the 20 ms RoCE sampler, comparing TTFT/TPOT, confirming the sampling has no effect |
| `steer_pairs.sh` | 6.4 | advances the Router's rotation with `max_tokens=1` small requests, steering each 131072-token request onto a chosen pairing; see the caveat below |

**Caveats**
- `steer_pairs.sh` is exactly the script whose **steering failed** in 6.4: `vllm bench` first sends an initial test request that is not counted in the logs, putting the position off by one step, so only 1 of 12 requests landed on the target pairing. To redo it, advance one extra step before sending the large requests, or send requests directly without `vllm bench`.
- `tcp_peers.sh` was rewritten to match how the numbers in this text were tallied and **was not re-run on 2P2D** (the 2P2D was already deleted); it was tested on 1P1D (0 connections between P and D); the 398 / 0 in this text came from a one-off command at the time. The tally includes the Pod's local connections to its own IP.
- Details and caveats of the other scripts (warm-up, counters being cumulative, NVLink counters not distinguishing sources, etc.) are in 1P1D doc A.4.

**Verification status**: the two analysis scripts were re-run on the 2P2D raw data: the single-request RoCE bytes (e.g. `prefill-1` NIC 7 sending 7990 MB), the 4×131072-token transfer times (193.0, 222.9 ms, etc.), and the pairings all match Chapters 5 and 6; `idle_cpu.py`, `router_pairs.py`, `tcp_peers.sh` were exercised read-only on a running 1P1D; `sampler_ab.sh`, `steer_pairs.sh`, and the driver scripts' 2P2D usage had only a syntax check and were not re-run (the 2P2D was already deleted).

[↑ Back to top](#top)
