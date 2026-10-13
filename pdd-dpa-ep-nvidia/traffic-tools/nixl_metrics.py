#!/usr/bin/env python3
"""Dump the NIXL / latency metrics of every Prefill and Decode engine (per DP rank).

Run it in the ROUTER Pod (it can reach the headless services of the pools; tests and
metric reads are kept out of the Prefill/Decode Pods):

    kubectl exec -i pdd-router-0 -- python3 - <prefill_nodes> <decode_nodes> < nixl_metrics.py
    e.g. 1P1D:  ... python3 - 1 1 < nixl_metrics.py        2P2D: ... python3 - 2 2 < ...

It scrapes http://pdd-<role>-<i>.pdd-<role>-headless:<port>/metrics (Prefill 8001,
Decode 8002) and keeps selected Prometheus series, grouped by Pod and by the `engine`
label (= DP rank within the pool). Counters are cumulative: take a snapshot before and
after a run and subtract.

Metrics of interest (from the Decode engine that pulled the KV)
  vllm:nixl_bytes_transferred_{sum,count}  bytes moved by NIXL / number of transfers
  vllm:nixl_xfer_time_seconds_sum          total transfer time (post -> completion)
  vllm:nixl_post_time_seconds_sum          time spent posting the transfer request
  vllm:nixl_num_descriptors_sum            number of descriptors per transfer
  vllm:request_prefill_time_seconds_sum    prefill-phase time of the request
  xfer_bucket                              histogram of transfer times summed over engines
                                           (cumulative buckets, key = "le" bound in seconds)
  mean transfer time = xfer_time_sum / count;  bandwidth = bytes_sum / xfer_time_sum

Output: one JSON object {pod: {metric: {engine: value}}}
"""
import urllib.request, re, json, sys

PN, DN = int(sys.argv[1]), int(sys.argv[2])     # number of Prefill / Decode Pods
KEEP = ('vllm:nixl_bytes_transferred_sum', 'vllm:nixl_bytes_transferred_count', 'vllm:nixl_xfer_time_seconds_sum',
        'vllm:nixl_xfer_time_seconds_count', 'vllm:nixl_post_time_seconds_sum', 'vllm:nixl_post_time_seconds_count',
        'vllm:nixl_num_descriptors_sum', 'vllm:nixl_num_descriptors_count', 'vllm:request_prefill_time_seconds_sum',
        'vllm:request_prefill_time_seconds_count', 'vllm:request_queue_time_seconds_sum', 'vllm:request_decode_time_seconds_sum',
        'vllm:time_to_first_token_seconds_sum', 'vllm:time_to_first_token_seconds_count', 'vllm:prompt_tokens_total',
        'vllm:generation_tokens_total', 'vllm:external_prefix_cache_hits_total', 'vllm:num_requests_running',
        'vllm:nixl_num_failed_transfers_total', 'vllm:nixl_num_failed_notifications_total',
        'vllm:request_time_per_output_token_seconds_sum', 'vllm:request_time_per_output_token_seconds_count')
out = {}
for role, port, n in (('prefill', 8001, PN), ('decode', 8002, DN)):
    for i in range(n):
        pod = f'pdd-{role}-{i}'
        t = urllib.request.urlopen(f'http://{pod}.pdd-{role}-headless:{port}/metrics', timeout=15).read().decode()
        d = {}
        for line in t.splitlines():
            m = re.match(r'^([A-Za-z_:][A-Za-z0-9_:]*)\{([^}]*)\} (\S+)$', line)   # name{labels} value
            if not m:
                continue
            name, lab = m.group(1), m.group(2)
            e = re.search(r'engine="(\d+)"', lab)       # DP rank of the engine within its pool
            e = e.group(1) if e else '-'
            if name in KEEP:
                d.setdefault(name, {})[e] = float(m.group(3))
            elif name == 'vllm:nixl_xfer_time_seconds_bucket':
                le = re.search(r'le="([^"]+)"', lab).group(1)
                d.setdefault('xfer_bucket', {})[le] = d.get('xfer_bucket', {}).get(le, 0) + float(m.group(3))
        out[pod] = d
print(json.dumps(out))
