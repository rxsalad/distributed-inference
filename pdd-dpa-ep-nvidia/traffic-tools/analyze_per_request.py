#!/usr/bin/env python3
"""Analyse an exp_per_request.sh run (/tmp/exp5-<LABEL>), one block per request.

usage: ./analyze_per_request.py <LABEL> [--ipmap ip=pod,ip=pod]
Works for 1P1D and for multi-node pools (2P2D): pods come from the files in the directory,
the Router's IPs are mapped to pods with ipmap.json (written by exp_per_request.sh; use
--ipmap for data recorded without it).

For every request k it prints
  the Prefill -> Decode pair as pod + node rank and as global ranks (P<g> -> D<g'>),
  the NIC index of both sides (global rank % 8; the NIC of a rank is mlx5_{2*(rank%8)}) and
  whether they are the same rail,
  TTFT, TPOT, prefill time, and from the Decode engine's metrics deltas: bytes, NIXL transfer
  time, bandwidth (bytes / xfer time), post time, descriptors, queue time,
  the Prefill NIC(s) that served RDMA READ requests (rdma-*-before/after<k>.txt),
  the on-wire KV burst seen by the 20 ms RoCE sampler on the serving Prefill NIC (tx) and on the
  receiving Decode NIC (rx): MB, duration, peak GB/s,
  and every other NIC (any pod) that shows a >10 GB/s burst in the request's window.

A "burst" = consecutive sampler intervals above 10 GB/s (traffic_common.bursts). Idle traffic and
EP collectives are far below that, KV transfers are 40-50 GB/s. Duration resolution = sampling
interval (20 ms), so peaks slightly above the 50 GB/s line rate are sampling-boundary jitter.
A summary of all requests is written to <dir>/summary.json.
"""
import argparse, glob, json, re
import traffic_common as C

ap = argparse.ArgumentParser()
ap.add_argument('label'); ap.add_argument('--ipmap')
args = ap.parse_args()
O = f'/tmp/exp5-{args.label}'
PODS = C.pods_in(O)
ipmap = C.load_ipmap(O, args.ipmap)
B = {p: C.load_samples(O, p) for p in PODS}                 # all sampler rows of every pod
N = len(glob.glob(f'{O}/t-end*'))                           # number of requests
res = []

for k in range(1, N + 1):
    t0 = float(open(f'{O}/t-start{k}').read()); t1 = float(open(f'{O}/t-end{k}').read())
    mb = json.load(open(f'{O}/metrics-before{k}.json')); ma = json.load(open(f'{O}/metrics-after{k}.json'))
    bench = open(f'{O}/bench{k}.log').read()
    g = lambda pat: (re.search(pat, bench) or [None, None])[1]
    pp, pg, dp, dg = C.parse_router_line(open(f'{O}/router{k}.txt').read(), ipmap, PODS)   # pods + global ranks
    dl = lambda pod, name: C.delta(ma, mb, pod, name)

    out = {'k': k, 'pair': f'{pp} r{pg % 8} -> {dp} r{dg % 8}', 'global ranks': f'P{pg} -> D{dg}',
           'NIC P->D': f'{pg % 8} -> {dg % 8}', 'same_rail': pg % 8 == dg % 8,
           'tokens': g(r'Total input tokens:\s+(\d+)'), 'TTFT_ms': g(r'Mean TTFT \(ms\):\s+([\d.]+)'), 'TPOT_ms': g(r'Mean TPOT \(ms\):\s+([\d.]+)')}
    # The Decode engine that pulled the KV is the one whose NIXL byte counter moved.
    by = dl(dp, 'vllm:nixl_bytes_transferred_sum'); xf = dl(dp, 'vllm:nixl_xfer_time_seconds_sum')
    ps = dl(dp, 'vllm:nixl_post_time_seconds_sum'); ds = dl(dp, 'vllm:nixl_num_descriptors_sum')
    q = dl(dp, 'vllm:request_queue_time_seconds_sum')
    e = list(by)[0]
    out.update(decode_engine=e, bytes=int(by[e]), xfer_ms=round(xf[e] * 1000, 1), **{'GB/s': round(by[e] / xf[e] / 1e9, 1)},
               post_ms=round(ps[e] * 1000, 1), descs=int(ds[e]), d_queue_s=round(q.get(e, 0), 3))
    pf = dl(pp, 'vllm:request_prefill_time_seconds_sum')
    out.update(prefill_engine=list(pf)[0], prefill_s=round(list(pf.values())[0], 2))

    # RDMA READ requests are counted on the Prefill NIC that served the data (the Decode side issues READs)
    reads = {}
    for p in [x for x in PODS if 'prefill' in x]:
        a, b = C.read_rdma(O, p, f'before{k}'), C.read_rdma(O, p, f'after{k}')
        for n in a:
            if b[n][1] - a[n][1] > 0:
                reads[(p, n)] = b[n][1] - a[n][1]
    out['P NIC with RDMA READ'] = {f'{p}/{n}': v for (p, n), v in reads.items()}
    (rp, rn) = list(reads)[0]
    pnic = int(rn.split('_')[1]) // 2                      # mlx5_{2i} -> NIC index i

    def window(pod):
        return [r for r in B[pod] if t0 <= r['t'] <= t1 + 3]
    def summarize(bs):
        return dict(MB=round(sum(v[2] for v in bs) / 1e6), dur_s=round(bs[-1][1] - bs[0][0], 3), peak_GBs=round(max(v[3] for v in bs), 1)) if bs else None
    out['KV burst P tx'] = summarize(C.bursts(window(rp), pnic, 0))
    out['KV burst D rx'] = summarize(C.bursts(window(dp), dg % 8, 1))
    # any other NIC with a >10 GB/s burst in this window (would indicate extra KV-like traffic)
    others = []
    for p in PODS:
        rows = window(p)
        for i in range(8):
            if C.bursts(rows, i, 0) or C.bursts(rows, i, 1):
                mx = max([c[3] for c in C.bursts(rows, i, 0) + C.bursts(rows, i, 1)])
                others.append((p, f'mlx5_{2 * i}', round(mx, 1)))
    out['NICs with >10GB/s (tx or rx)'] = others
    res.append(out)

for o in res:
    print()
    for kk, v in o.items():
        print(f'  {kk}: {v}')
json.dump(res, open(f'{O}/summary.json', 'w'), indent=1)
