#!/usr/bin/env python3
"""Which RDMA memory regions (MRs) does each Worker have, on which NICs, and how big (read-only).

Run it INSIDE a Prefill/Decode Pod:
    kubectl exec -i pdd-prefill-0 -- python3 - < rdma_mr.py

`rdma resource show mr` lists every MR as
    dev <mlx5_N> mrn <n> mrlen <bytes> pdn <n> pid <pid>
The script maps pid -> Worker (from the process title VLLM::Worker_DPn_EPn) and prints per
Worker: the NICs it registered memory on, the number of MRs, their total size and the largest
sizes. This is how "each Worker registers its whole KV cache on its own NIC only" was checked:
the large MRs (tens of GiB) appear on exactly one device per Worker.
"""
import subprocess, os, collections, re

out = subprocess.run(['rdma', 'resource', 'show', 'mr'], capture_output=True, text=True).stdout
names = {}                                     # pid -> DP rank of the Worker
for d in os.listdir('/proc'):
    if d.isdigit():
        try:
            c = open(f'/proc/{d}/cmdline', 'rb').read().replace(b'\0', b' ').decode()
        except Exception:
            continue
        m = re.match(r'VLLM::Worker_DP(\d+)_EP\d+', c)
        if m:
            names[int(d)] = int(m.group(1))
mr = collections.defaultdict(list)             # (pid, device) -> list of MR lengths
for line in out.splitlines():
    m = re.match(r'dev (\S+) mrn (\d+) mrlen (\d+) pdn (\d+) pid (\d+)', line)
    if m:
        mr[(int(m.group(5)), m.group(1))].append(int(m.group(3)))
print('total MR lines', sum(len(v) for v in mr.values()))
bypid = collections.defaultdict(dict)
for (pid, dev), v in mr.items():
    bypid[pid][dev] = v
dev_order = lambda d: int(d.split('_')[1])
for pid in sorted(bypid, key=lambda p: names.get(p, 99)):
    devs = bypid[pid]
    print(f'rank {names.get(pid, "?")} pid {pid}: devices {sorted(devs, key=dev_order)}')
    for dev in sorted(devs, key=dev_order)[:1]:
        v = devs[dev]; c = collections.Counter(v)
        big = sorted(c.items(), key=lambda kv: -kv[0])[:6]
        print(f'   on {dev}: {len(v)} MRs, total {sum(v) / 2**30:.2f} GiB, largest sizes (bytes x count): {big}')
    print('   MRs per device:', {d.split('_')[1]: len(v) for d, v in sorted(devs.items(), key=lambda kv: dev_order(kv[0]))})
