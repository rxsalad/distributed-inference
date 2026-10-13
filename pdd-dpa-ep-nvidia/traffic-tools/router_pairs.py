#!/usr/bin/env python3
"""Show how the Router pairs Prefill and Decode ranks (read-only, run on the machine with kubectl).

usage: ./router_pairs.py [router pod, default pdd-router-0]

The Router (round_robin, PD disaggregation) logs one line per request:
    Selected prefill=http://<ip>:8001@<rank in node> ... decode=http://<ip>:8002@<rank in node>
This script maps the IPs to pods with `kubectl get pods`, turns the ranks into global ranks
(node index * 8 + rank in node), and prints
  - the pair table: P rank -> D rank, how often it occurred, the NIC index of both sides
    (global rank % 8; mlx5_{2*index}) and whether it is the same rail;
  - the number of distinct pairs (a fixed one-to-one mapping gives as many pairs as Prefill
    ranks, not P x D);
  - the selection order of P and D over the last requests (the round_robin cycle).
The pairing changes when the Router is restarted, so only use log lines of one Router run.
Note: `vllm bench` first sends one test request that advances the round robin by one step.
"""
import collections, json, subprocess, sys
import traffic_common as C

router = sys.argv[1] if len(sys.argv) > 1 else 'pdd-router-0'
pods = json.loads(subprocess.run(['kubectl', 'get', 'pods', '-o', 'json'], capture_output=True, text=True, check=True).stdout)
ipmap = {p['status']['podIP']: p['metadata']['name'] for p in pods['items'] if p['status'].get('podIP')}
log = subprocess.run(['kubectl', 'logs', router], capture_output=True, text=True, check=True).stdout
log = __import__('re').sub(r'\x1b\[[0-9;]*m', '', log)        # strip colour codes
seq = [C.parse_router_line(l, ipmap) for l in log.splitlines() if 'Selected prefill' in l]
seq = [(s[1], s[3]) for s in seq if s]
cnt = collections.Counter(seq)
print(f'{len(seq)} requests, {len(cnt)} distinct (P, D) pairs')
print('P rank -> D rank | count | NIC P->D | same rail')
for (p, d), n in sorted(cnt.items()):
    print(f'  P{p:<3} -> D{d:<3} | {n:4d} | {p % 8} -> {d % 8} | {"yes" if p % 8 == d % 8 else "no"}')
print('selection order, last 32 requests: P', [p for p, _ in seq[-32:]])
print('                                   D', [d for _, d in seq[-32:]])
