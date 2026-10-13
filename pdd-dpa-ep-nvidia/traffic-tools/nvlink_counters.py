#!/usr/bin/env python3
"""Read the cumulative NVLink data counters of all GPUs in this Pod (read-only).

Run it INSIDE a Prefill/Decode Pod (the GPUs are only visible there):

    kubectl exec -i pdd-prefill-0 -- python3 - < nvlink_counters.py

How it works
  `nvidia-smi nvlink -gt d` prints, for every GPU and every NVLink link, lines like
      GPU 0: NVIDIA B300 SXM6 ...
           Link 0: Data Tx: 123456 KiB
           Link 0: Data Rx: 654321 KiB
  We add up all links of one GPU, so the result is "bytes this GPU has sent / received
  over NVLink since the driver was loaded" (unit: KiB).

How to use the output
  The counters are cumulative. Take one snapshot before and one after a request and
  subtract: (after - before) / 2**20 gives GiB per GPU. Sum over the 8 GPUs for the
  traffic of the whole node. Tx is what we report: every byte sent by one GPU is
  received by another, so Rx of the whole node equals Tx of the whole node.

Caveats
  - The counters also include traffic that is not a request (dummy batches, warm-up,
    the DP all-reduce of CUDA-graph metadata is NOT included: it runs over Gloo/TCP).
    Always measure the delta around a known workload, with the pool otherwise idle.
  - The call takes about a second ('dt' in the output); the snapshot is not atomic.

Output: one JSON line {"t": unix time, "dt": run time, "tx": {gpu: KiB}, "rx": {gpu: KiB}}
"""
import subprocess, re, json, time

t = time.time()
out = subprocess.run(['nvidia-smi', 'nvlink', '-gt', 'd'], capture_output=True, text=True).stdout
g = -1          # GPU index of the section we are currently parsing
tx = {}         # GPU index -> total NVLink Tx KiB (all links)
rx = {}         # GPU index -> total NVLink Rx KiB (all links)
for line in out.splitlines():
    m = re.match(r'GPU (\d+):', line)
    if m:                                   # a new GPU section starts
        g = int(m.group(1)); tx[g] = 0; rx[g] = 0
        continue
    m = re.search(r'Data (Tx|Rx): (\d+) KiB', line)   # one link of the current GPU
    if m:
        (tx if m.group(1) == 'Tx' else rx)[g] += int(m.group(2))
print(json.dumps({'t': time.time(), 'dt': time.time() - t, 'tx': tx, 'rx': rx}))
