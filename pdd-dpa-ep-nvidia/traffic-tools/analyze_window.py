#!/usr/bin/env python3
"""Analyse an exp_window.sh run (/tmp/exp4-<LABEL>): traffic per node and per NIC.

usage: ./analyze_window.py <LABEL> [--ipmap ip=pod,ip=pod] [--split K]
Works for any number of Prefill/Decode pods (taken from the files in the directory).

Prints, for the measured window [t-start, t-end]:
  - bench summary: requests, duration, throughput, TTFT, TPOT         bench*.log
  - Router pairing as global ranks, with count per pair               router.txt (+ ipmap.json)
  - NVLink: GiB sent per GPU and per node                              nvl-*-before/after.json
        (delta of the nvidia-smi counters; KiB / 2**20 = GiB)
  - RoCE: GB sent / received per NIC, peak single-NIC rate            bytes-<pod>.jsonl
  - RDMA WRITE/READ request count delta per NIC                        rdma-*-before/after.txt
  - NIXL: per Decode engine and in total: transfers, size, mean xfer/post time, bandwidth,
        descriptors, xfer-time histogram quantiles                     metrics-*.json
  - Prefill time per Prefill engine
  - KV bursts on the wire (see traffic_common.bursts): Prefill tx and Decode rx, duration and rate.
        Decode NICs with index < K and >= K are listed separately (--split K, default 3): in
        2P2D the Decode EP collectives only use NIC 0-2, so this separates "KV on a NIC that also
        carries collectives" from "KV on a NIC that only carries KV".
"""
import argparse, glob, json, os, re, statistics as st, collections
import traffic_common as C

ap = argparse.ArgumentParser()
ap.add_argument('label'); ap.add_argument('--ipmap'); ap.add_argument('--split', type=int, default=3)
args = ap.parse_args()
O = f'/tmp/exp4-{args.label}'
PODS = C.pods_in(O)
t0 = float(open(f'{O}/t-start').read()); t1 = float(open(f'{O}/t-end').read())
print(f'== {args.label}: window {t1 - t0:.1f}s, pods {PODS}')

# --- bench summaries (parsed from the vllm bench output) ---
for f in sorted(glob.glob(f'{O}/bench*.log')):
    b = open(f).read()
    g = lambda pat: (re.search(pat, b) or [None, None])[1]
    print(' ', os.path.basename(f), 'success', g(r'Successful requests:\s+(\d+)'), 'input tokens', g(r'Total input tokens:\s+(\d+)'),
          'dur', g(r'Benchmark duration \(s\):\s+([\d.]+)'), 'req/s', g(r'Request throughput \(req/s\):\s+([\d.]+)'),
          'tok/s', g(r'Total token throughput \(tok/s\):\s+([\d.]+)'), 'TTFT mean/med/p99',
          g(r'Mean TTFT \(ms\):\s+([\d.]+)'), g(r'Median TTFT \(ms\):\s+([\d.]+)'), g(r'P99 TTFT \(ms\):\s+([\d.]+)'),
          'TPOT mean/p99', g(r'Mean TPOT \(ms\):\s+([\d.]+)'), g(r'P99 TPOT \(ms\):\s+([\d.]+)'))

# --- Router pairing as global ranks ---
ipmap = C.load_ipmap(O, args.ipmap)
pairs = [C.parse_router_line(l, ipmap, PODS) for l in open(f'{O}/router.txt')]
pairs = [f'P{p[1]}->D{p[3]}' for p in pairs if p]
print(f'  router pairs (global ranks), count per pair over the last {len(pairs)} log lines:', dict(collections.Counter(pairs)))

# --- NVLink: delta of the cumulative per-GPU Tx counters ---
print('  NVLink (GiB, tx per GPU):')
for p in PODS:
    fa, fb = f'{O}/nvl-{p}-before.json', f'{O}/nvl-{p}-after.json'
    if not os.path.exists(fa):
        continue
    a, b = json.load(open(fa)), json.load(open(fb))
    per = [round((b['tx'][g] - a['tx'][g]) / 2**20, 1) for g in sorted(a['tx'], key=int)]
    print(f'    {p}: total tx {sum(per):.1f} GiB per-GPU {per}')

# --- RoCE bytes: first vs last sampler value in the window; peak single-NIC rate ---
for p in PODS:
    r = C.load_samples(O, p, t0, t1)
    tx = [round((r[-1]['v'][i][0] - r[0]['v'][i][0]) / 1e6) for i in range(8)]
    rx = [round((r[-1]['v'][i][1] - r[0]['v'][i][1]) / 1e6) for i in range(8)]
    pk = [0.0] * 8; agg = 0.0
    for a, c in zip(r, r[1:]):
        dt = c['t'] - a['t']
        if dt <= 0:
            continue
        agg = max(agg, sum(c['v'][i][0] - a['v'][i][0] for i in range(8)) / dt / 1e9)
        for i in range(8):
            pk[i] = max(pk[i], max(c['v'][i][0] - a['v'][i][0], c['v'][i][1] - a['v'][i][1]) / dt / 1e9)
    print(f'  RoCE {p}: tx MB per NIC {tx} | rx MB per NIC {rx} | peak single-NIC GB/s {[round(x, 1) for x in pk]} | peak node tx {agg:.1f} GB/s (samples {len(r)})')

# --- RDMA request counters ---
for p in PODS:
    a, b = C.read_rdma(O, p, 'before'), C.read_rdma(O, p, 'after')
    d = {n: (b[n][0] - a[n][0], b[n][1] - a[n][1]) for n in a if (b[n][0] - a[n][0]) + (b[n][1] - a[n][1]) > 0}
    print(f'  rdma req delta {p} (write,read):', d)

# --- NIXL metrics ---
ma, mb = json.load(open(f'{O}/metrics-after.json')), json.load(open(f'{O}/metrics-before.json'))
dl = lambda pod, name: C.delta(ma, mb, pod, name)
tot = dict(n=0, by=0, xf=0, ps=0)
buckets = {}
for pod in [p for p in PODS if 'decode' in p]:
    n = sum(dl(pod, 'vllm:nixl_bytes_transferred_count').values()); by = sum(dl(pod, 'vllm:nixl_bytes_transferred_sum').values())
    xf = sum(dl(pod, 'vllm:nixl_xfer_time_seconds_sum').values()); ps = sum(dl(pod, 'vllm:nixl_post_time_seconds_sum').values())
    ds = sum(dl(pod, 'vllm:nixl_num_descriptors_sum').values())
    print(f'  NIXL {pod}: transfers {int(n)}, {by / 1e9:.1f} GB, xfer mean {xf / max(n, 1) * 1000:.1f} ms, post mean {ps / max(n, 1) * 1000:.1f} ms, '
          f'bandwidth(bytes/sum xfer) {by / xf / 1e9 if xf else 0:.1f} GB/s, descs/xfer {ds / max(n, 1):.0f}')
    per_engine = {e: dict(n=int(dl(pod, 'vllm:nixl_bytes_transferred_count').get(e, 0)),
                          xfer_ms=round(dl(pod, 'vllm:nixl_xfer_time_seconds_sum').get(e, 0) * 1000, 1)) for e in dl(pod, 'vllm:nixl_bytes_transferred_sum')}
    print(f'     per engine (DP rank in pool): {per_engine}')
    tot['n'] += n; tot['by'] += by; tot['xf'] += xf; tot['ps'] += ps
    x, y = ma[pod].get('xfer_bucket', {}), mb[pod].get('xfer_bucket', {})
    for le, v in x.items():
        buckets[le] = buckets.get(le, 0) + v - y.get(le, 0)
if tot['n']:
    print(f"  NIXL total: transfers {int(tot['n'])}, mean size {tot['by'] / tot['n'] / 1e9:.2f} GB, xfer mean {tot['xf'] / tot['n'] * 1000:.1f} ms, "
          f"post mean {tot['ps'] / tot['n'] * 1000:.1f} ms, bandwidth {tot['by'] / tot['xf'] / 1e9:.1f} GB/s")
    # xfer_time is a cumulative histogram: the first bucket whose count reaches p*total is an upper bound of the p-quantile
    les = sorted((float(le) if le != '+Inf' else 1e9, v) for le, v in buckets.items())
    total = les[-1][1]
    q = lambda p: next(le for le, v in les if v >= p * total)
    print(f'  xfer_time quantile upper bounds (s): p50 <= {q(.5)}, p90 <= {q(.9)}, p99 <= {q(.99)}; cumulative counts {[(l, int(v)) for l, v in les if l < 1e8][:14]}')
for pod in [p for p in PODS if 'prefill' in p]:
    print(f'  prefill time {pod}:', {e: round(v, 2) for e, v in dl(pod, 'vllm:request_prefill_time_seconds_sum').items()})

# --- KV bursts on the wire (>= 0.5 GB per burst = a KV transfer; collectives are far smaller/slower) ---
def stat(name, xs):
    if not xs:
        print(f'  {name}: none'); return
    d = [(c[1] - c[0]) * 1000 for c in xs]; r = [c[2] / (c[1] - c[0]) / 1e9 for c in xs]
    print(f'  {name}: n={len(xs)} duration ms median {st.median(d):.0f} (min {min(d):.0f}, max {max(d):.0f}); '
          f'mean rate GB/s median {st.median(r):.1f} (min {min(r):.1f}, max {max(r):.1f}); peak sample median {st.median([c[3] for c in xs]):.1f}')

for role, col, label in (('prefill', 0, 'Prefill tx'), ('decode', 1, 'Decode rx')):
    allb = []
    for p in [p for p in PODS if role in p]:
        rows = C.load_samples(O, p, t0, t1)
        for nic in range(8):
            for c in C.bursts(rows, nic, col):
                if c[2] > 0.5e9:
                    allb.append((nic, c))
    stat(f'{label} bursts, all NICs', [c for _, c in allb])
    if role == 'decode':
        stat(f'{label} bursts, NIC < {args.split} (may also carry collectives)', [c for n, c in allb if n < args.split])
        stat(f'{label} bursts, NIC >= {args.split} (KV only)', [c for n, c in allb if n >= args.split])
