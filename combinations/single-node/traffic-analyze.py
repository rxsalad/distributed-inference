#!/usr/bin/env python3
import json, re, sys, os
from statistics import mean

BASE = sys.argv[1] if len(sys.argv) > 1 else "traffic-results"
PODS = ["tp8", "tep8", "dtp8", "dep8"]

def parse(path):
    snaps, benches, engs, idlechk = {}, {}, {}, None
    for line in open(path):
        p = line.split(None, 2)
        if len(p) < 2: continue
        if p[0] == "SNAP": snaps[p[1]] = json.loads(p[2])
        elif p[0] == "BENCH":
            a, b = p[2].split(); benches[p[1]] = (float(a), float(b))
        elif p[0] == "ENG": engs[p[1]] = dict(re.findall(r"(e\d+)=(\d+)", p[2]))
        elif p[0] == "IDLECHK": idlechk = float(p[1])
    return snaps, benches, engs, idlechk

def giB(k): return k / 2**20

# 8 行结果表，key 用索引避免中文编码问题
R = [[] for _ in range(8)]
LBL = ["r128_avg","r1_avg","decode","node","ttft","dur","bw","dist"]

for pod in PODS:
    path = f"{BASE}/{pod}.txt"
    if not os.path.exists(path):
        for r in R: r.append("-")
        continue
    snaps, benches, engs, idle = parse(path)
    r128_t, r1_t = [], []
    r128_d, r1_d, r128_ttft = [], [], []
    r128_pg, r1_pg = [], []
    for series in ["r128","r1"]:
        for tag in sorted(t for t in benches if t.startswith(series+"-")):
            b, a = snaps[f"{tag}-b"], snaps[f"{tag}-a"]
            d = [y-x for x,y in zip(b["tx"],a["tx"])]
            ttft, dur = benches[tag]
            if series=="r128":
                r128_t.append(sum(d)); r128_d.append(dur); r128_ttft.append(ttft); r128_pg.extend(d)
            else:
                r1_t.append(sum(d)); r1_d.append(dur); r1_pg.extend(d)
    R[0].append(f"{giB(mean(r128_t))/8:.1f}")
    R[1].append(f"{giB(mean(r1_t))/8:.1f}")
    R[2].append(f"+{(giB(mean(r128_t))-giB(mean(r1_t)))/8:.2f}")
    R[3].append(f"{giB(mean(r128_t)):.1f}")
    R[4].append(f"{mean(r128_ttft):.0f}")
    R[5].append(f"{mean(r1_d):.2f}")
    R[6].append(f"{giB(mean(r128_t))/8/mean(r128_d):.0f}")
    if r128_pg:
        lo, hi = giB(min(r128_pg)), giB(max(r128_pg))
        R[7].append("uniform" if hi/lo<1.05 else f"{lo:.1f}-{hi:.1f}")

print("| | tp8 | tep8 | dtp8 | dep8 |")
print("|---|---|---|---|---|")
for i, lbl in enumerate(LBL):
    print(f"| {lbl} | " + " | ".join(R[i]) + " |")
