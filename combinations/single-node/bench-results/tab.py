#!/usr/bin/env python3
"""v2 benchmark JSON 汇总表。用法: python3 tab.py [pattern ...]（如 p2 p3 p4）"""
import json, glob, os, re, sys
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)))
PODS = ["tp8", "tep8", "dtp8", "dep8"]
COLS = [("output_throughput", "out"), ("mean_ttft_ms", "ttft"),
        ("p99_ttft_ms", "ttftp99"), ("mean_tpot_ms", "tpot"),
        ("p99_itl_ms", "p99itl")]


def load(pat):
    data = {}
    for pod in PODS:
        rows = defaultdict(list)
        for f in glob.glob(f"{BASE}/{pod}/{pat}.json"):
            tag = re.sub(r"-r\d+$", "", os.path.basename(f)[:-5])
            rows[tag].append(json.load(open(f)))
        data[pod] = rows
    return data


def val(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def cnum(t):
    m = re.search(r"c(\d+)", t)
    return int(m.group(1)) if m else 0


def table(pat):
    data = load(pat)
    all_tags = sorted({t for rows in data.values() for t in rows}, key=cnum)
    if not all_tags:
        return
    print(f"### {pat}")
    print("| tag | " + " | ".join(f"{p} " + " ".join(c[1] for c in COLS) + " |" for p in PODS).replace("||", "|"))
    head = "|---|"
    for p in PODS:
        head += "---|"
    print(head)
    for t in all_tags:
        cells = []
        for p in PODS:
            rows = data[p].get(t)
            if rows:
                cells.append(" / ".join(f"{val(rows, k):.1f}" for k, _ in COLS))
            else:
                cells.append("—")
        print(f"| {t} | " + " | ".join(cells) + " |")
    print()


if __name__ == "__main__":
    pats = sys.argv[1:] or ["p1-*", "p2-*"]
    for pat in pats:
        table(pat)
