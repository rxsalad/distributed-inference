# distributed-inference: LLM Distributed Inference Knowledge Base

Concepts, deployment notes, and measurement write-ups accumulated around one question: **how do MLA + MoE large models (DeepSeek-V3 as the reference) run inference on multi-GPU clusters?** Deployments run on DigitalOcean Kubernetes (DOKS) across two kinds of GPU clusters — **NVIDIA B300×8** (the current mainline) and **AMD MI350x8 / MI355x8** (earlier). The inference engine is vLLM; topics include parallelism combinations (DP / TP / PP / EP), Prefill/Decode disaggregation (PDD), and KV cache transfer (Mooncake / NIXL).

Annotation convention (shared across documents; each document's opening notes take precedence): **Observed** = command output or logs from this cluster; **source-verified** = checked against the vLLM source inside the image; **Inferred** = derived from mechanism, not verified. Scheme naming (TP / DEP / TEP / DTP, etc.) is defined in `concepts/parallelism-basics.md` Section 6.

**Languages — read this before clicking**: most write-ups in this knowledge base are in **Chinese**. English versions exist only for **this README** and the **two PDD measurement documents** (1P1D / 2P2D, `-en.md` at the top level); the three top-level AMD DOKS guides (`.MD`) are in English. Everything else — `concepts/`, `vllm-expert-parallel/`, `vllm-data-parallel/`, `combinations/`, and the PDD originals in `pdd-dpa-ep-nvidia/` — is Chinese only. Code, yamls, and scripts are in English throughout. Each entry below is marked with its language.

## Directory Overview

### Concepts: `concepts/` (Chinese)

A series of core-concept articles on distributed LLM inference, organized as a reading path (has its own `README.md` with the reading-order diagram and a scheme-overview table):

```
decoder-only ─► inference ─► parallelism-basics ─► moe ─► dp_attention ─► dpa_ep ─► ep_values
 model & training   inference      parallelism &         MoE     DP Attention    DPA+EP     Wide EP
                                   collectives                                  combo
```

### Parallelism experiments (B300): `vllm-expert-parallel/`, `vllm-data-parallel/`, `combinations/` (all Chinese)

| Directory | Contents |
|---|---|
| [`vllm-expert-parallel/`](vllm-expert-parallel/README.md) | DeepSeek-V3 on 2 B300 nodes, **three strategies measured**: TEP16 (strategy 1), TP8×PP2 (strategy 2), DEP16 hybrid-lb (strategy 3); `README.md` is that directory's index (environment, documents, deployment files, common commands). Special topics: `dummy-batch-why.md` (the co-run/dummy-batch mechanism under DP+EP); earlier experiments: single-/two-node DP+EP (DeepSeek-V3-0324) |
| [`vllm-data-parallel/`](vllm-data-parallel/) | DP basics (Qwen3-8B): the official DP docs condensed with local measurements, single-node DP4×TP2, two-node DP4×TP1 |
| [`combinations/`](combinations/single-node.md) | **Four single-node parallelism combinations** deployed simultaneously and tested in one batch: TP8 / TEP8 / DTP8 / DEP8 (covering two forms the recipes page never gave) |

### Prefill/Decode Disaggregation (PDD)

**B300 + NIXL** (current mainline; **English versions available**, Chinese originals alongside):

| Document | Contents |
|---|---|
| [`pdd-dpa-ep-nvidia/pdd-dpa-ep-nvidia(1p1d).md`](<pdd-dpa-ep-nvidia/pdd-dpa-ep-nvidia(1p1d).md>) (Chinese), [`pdd-dpa-ep-nvidia(1p1d)-en.md`](<pdd-dpa-ep-nvidia(1p1d)-en.md>) (English) | **1P1D**: 1 Prefill node + 1 Decode node, each an 8-GPU DEP group (DP8+EP8), KV moved by NIXL. Principles, deployment and configuration, observation methods, single-pool measurements (EP collectives over NVLink; KV transfer via RDMA READ at ~48 GB/s; etc.) |
| [`pdd-dpa-ep-nvidia/pdd-dpa-ep-nvidia(2p2d).md`](<pdd-dpa-ep-nvidia/pdd-dpa-ep-nvidia(2p2d).md>) (Chinese), [`pdd-dpa-ep-nvidia(2p2d)-en.md`](<pdd-dpa-ep-nvidia(2p2d)-en.md>) (English) | **2P2D**: 2+2 nodes (DP16+EP16). Multi-node differences (EP collectives cross-node over RoCE, rail-aligned), comparison with 1P1D |

Deployment files (`pdd-dep.yaml`, `pdd-launch.sh`, `pdd-router.py`, `pdd.sh`) and the traffic-statistics scripts (`traffic-tools/`) live in the [`pdd-dpa-ep-nvidia/`](pdd-dpa-ep-nvidia/) subdirectory; the two pools' node counts are just environment variables — no file changes needed.

**AMD DOKS** (English guides, uppercase `.MD` suffix):

| Document / directory | Contents |
|---|---|
| [`mc-pdd-doks-amd.MD`](mc-pdd-doks-amd.MD) | PDD with vLLM + MooncakeConnector on DOKS: step-by-step deployment plus KV-transfer performance across TP1/4/8, intra-/inter-node, and NUMA-aligned/misaligned scenarios |
| [`mc-kv-cache-sharing.MD`](mc-kv-cache-sharing.MD) | KV cache sharing with MooncakeStoreConnector (cross-worker prefix-cache sharing, lower TTFT) |
| [`nixl-pdd-doks-amd.MD`](nixl-pdd-doks-amd.MD) | Deployment notes for PDD with NixlConnector (concepts, architecture, components) |
| `pdd/`, `nixl/`, `sharing/`, `pdd-vllm-mooncake-mi350x8/` | Deployment yamls, router and plotting scripts for those AMD experiments (Mooncake PDD, NIXL PDD, Mooncake store, MI350x8/MI355x8 deployments) |

## Cluster Environments

| | B300 cluster (current mainline) | AMD DOKS (earlier) |
|---|---|---|
| GPU nodes | 4 × B300×8 (sm_103, 267.69 GiB per GPU; NV18 full interconnect within a node; 8 × 400G RoCE NICs, GPU i ↔ `mlx5_{2i}` on the same PCIe bridge) + 2 CPU nodes | 2 × workers, 16 GPUs total (MI350x8 / MI355x8), RDMA/RoCE |
| Network | Rail-optimized RoCE (cross-rail reachable); requires Multus + `roce-net-fabricN` attachments and the `rdma/fabricN` device plugin | Multus + ipvlan + Whereabouts IPAM |
| Software | vLLM 0.31.0, NCCL 2.30.7, NIXL 1.4.1, PyTorch 2.13.0+cu130 | vLLM + Mooncake / NIXL |
| Model | `deepseek-ai/DeepSeek-V3` (FP8; earlier experiments used DeepSeek-V3-0324, Qwen3-8B) | DeepSeek family |

## Key Operational Lessons (measured on these clusters; see the documents for details)

- Multi-node NCCL **must** set `NCCL_IB_GID_INDEX=3`; otherwise it picks the link-local RoCE GID and **hangs silently**. Same for UCX (`UCX_IB_GID_INDEX=3`).
- Multi-node TP needs `--disable-custom-all-reduce`; DeepEP is unavailable on B300 (sm_103) (the image has no matching cubin), so EP falls back to the default `allgather_reducescatter` backend.
- Initialization is slow: `VLLM_ENGINE_READY_TIMEOUT_S=3600`; resolve the head Pod IP in yaml with a "loop until non-empty" pattern to avoid a DNS race; never kill processes with `pkill -f` (it kills itself).
- Warm up before any performance test (every rank must serve one long request first), or the first request's one-time overhead pollutes the data (see PDD docs 3.3/4.3).

## Suggested Reading Path

1. `concepts/` — the conceptual foundation (start here without prior background) — **Chinese**
2. `vllm-data-parallel/` — how DP actually runs — **Chinese**
3. `vllm-expert-parallel/` — the three strategies and EP mechanics (read its `README.md` first) — **Chinese**
4. `combinations/single-node.md` — the single-node four-combination comparison — **Chinese**
5. PDD: 1P1D → 2P2D (B300 + NIXL, current mainline) — **English versions available** ([1P1D](<pdd-dpa-ep-nvidia(1p1d)-en.md>), [2P2D](<pdd-dpa-ep-nvidia(2p2d)-en.md>))
6. The three AMD DOKS guides (`.MD`) — earlier Mooncake/NIXL PDD and KV cache sharing — **English**
