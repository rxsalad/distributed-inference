#!/usr/bin/env python3
"""CPU usage (%) of the vLLM processes of one Pod over 3 seconds (read-only).

Run it INSIDE a Prefill/Decode Pod while the pool is idle:
    kubectl exec -i pdd-prefill-0 -- python3 - < idle_cpu.py

It finds the processes by their title (VLLM::Worker_DPn_EPn, EngineCore_DPn, APIServer_n,
DPCoordinator), reads utime+stime (fields 14 and 15 of /proc/<pid>/stat) twice, 3 s apart, and
converts the difference to percent of one core. On an idle pool every Worker shows 100%
(a busy-polling native thread, probably the NIXL/UCX progress thread) while all other processes
are below 1%.
"""
import os, time, re

tck = os.sysconf('SC_CLK_TCK')                     # clock ticks per second


def procs():
    d = {}
    for p in os.listdir('/proc'):
        if p.isdigit():
            try:
                c = open(f'/proc/{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode()
            except Exception:
                continue                           # process gone
            m = re.match(r'VLLM::(Worker_DP\d+_EP\d+|EngineCore_DP\d+|APIServer_\d+|DPCoordinator)', c)
            if m:
                # strip "pid (comm)" first: comm may contain spaces; fields after it start at index 3 (state)
                f = open(f'/proc/{p}/stat').read().rsplit(')', 1)[1].split()
                d[m.group(1)] = int(f[11]) + int(f[12])      # utime + stime in ticks
    return d


a = procs(); time.sleep(3); b = procs()
cpu = {k: (b[k] - a[k]) * 100 / (3 * tck) for k in b if k in a}
w = {k: round(v) for k, v in cpu.items() if k.startswith('Worker')}
print('Worker CPU% (3 s):', dict(sorted(w.items(), key=lambda kv: int(re.search(r'DP(\d+)', kv[0]).group(1)))))
others = [v for k, v in cpu.items() if not k.startswith('Worker')]
print('non-Worker processes: n=%d max CPU%%=%.1f' % (len(others), max(others) if others else 0))
