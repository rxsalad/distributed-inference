#!/usr/bin/env python3
"""v2 benchmark 能效分析：*-r1.gpu.log（功耗采样）+ 对应 JSON → J/token。
用法: python3 energy.py [pattern ...]（默认 p1 p2 p3 p4 p5）"""
import json, glob, os, re, sys

BASE = os.path.dirname(os.path.abspath(__file__))
PODS = ["tp8", "tep8", "dtp8", "dep8"]


def analyze_gpu_log(path):
    """返回 (8 卡平均合计功率 W, 8 卡平均利用率 %)"""
    pw, ut = [], []
    for line in open(path):
        parts = line.strip().split(" ", 1)
        if len(parts) != 2:
            continue
        p = u = n = 0
        for g in parts[1].split(";"):
            if "," not in g:
                continue
            a, b = g.split(",")
            u += float(a)
            p += float(b)
            n += 1
        if n:
            pw.append(p)          # 8 卡功率合计（W）
            ut.append(u / n)      # 平均每卡利用率（%）
    if not pw:
        return 0.0, 0.0
    return sum(pw) / len(pw), sum(ut) / len(ut)


def main():
    pats = sys.argv[1:] or ["p1-*", "p2-*", "p3-*", "p4-*", "p5-*"]
    for pat in pats:
        all_tags = set()
        for pod in PODS:
            for f in glob.glob(f"{BASE}/{pod}/{pat}-r1.gpu.log"):
                all_tags.add(os.path.basename(f)[:-8])  # 去掉 ".gpu.log"，保留 "-r1"

        def key(t):
            m = re.search(r"c(\d+)", t)
            return (0, int(m.group(1))) if m else (1, t)

        if not all_tags:
            continue
        print(f"### {pat}（能效，rep1）")
        print("| tag | 指标 | " + " | ".join(PODS) + " |")
        print("|---|---|" + "---|" * len(PODS))
        for tag in sorted(all_tags, key=key):
            rows = []
            for pod in PODS:
                log, js = f"{BASE}/{pod}/{tag}.gpu.log", f"{BASE}/{pod}/{tag}.json"
                if os.path.exists(log) and os.path.exists(js):
                    d = json.load(open(js))
                    watts, util = analyze_gpu_log(log)
                    dur = d["duration"]
                    energy = watts * dur
                    out_tok = d.get("total_output_tokens") or d.get("total_generated_tokens") or 0
                    in_tok = d.get("total_input_tokens") or 0
                    rows.append((watts, util,
                                 energy / out_tok if out_tok else None,
                                 energy / (in_tok + out_tok) if in_tok + out_tok else None))
                else:
                    rows.append(None)

            def fmt(i, nd, scale=1):
                return " | ".join("-" if r is None or r[i] is None
                                  else f"{r[i] / scale:.{nd}f}" for r in rows)
            print(f"| {tag} | 8卡合计功率 (W) | {fmt(0, 0)} |")
            print(f"| {tag} | 平均利用率 (%) | {fmt(1, 0)} |")
            print(f"| {tag} | J/输出token (mJ) | {fmt(2, 2, 0.001)} |")
        print()


if __name__ == "__main__":
    main()
