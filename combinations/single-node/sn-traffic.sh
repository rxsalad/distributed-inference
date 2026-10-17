#!/bin/bash
# 32768/128 流量实测（Pod 内执行）：
#   kubectl exec -i <pod>-0 -- bash -s < sn-traffic.sh
# Phase 0: 空闲 10s NVLink 基线（应为 0）
# Phase 1: 预热 32768/8 直到全 engine 覆盖（32768 prefill 是新形状；out=8 已够，decode 形状已热）
# Phase 2: 5 轮 32768/128 单请求，每轮前后采 NVLink（per-GPU Tx KiB）+ engine 归属
# Phase 3: 3 轮 32768/1 纯 prefill，同样前后采样
# 输出行格式（供 dev 侧 python 解析）：
#   SNAP <tag> <t> <tx0,tx1,...,tx7>          # NVLink 快照（KiB，累计值）
#   BENCH <tag> <ttft_ms> <dur_s>            # vllm bench 结果
#   ENG <tag> <e0=eN e1=eN ...>              # 每 engine 累计完成数
#   IDLECHK <delta_GiB>                      # 空闲基线差值
set -u
M=deepseek-ai/DeepSeek-V3

nvln() {  # per-GPU NVLink Tx (KiB, 累计)
  python3 - <<'PY'
import subprocess, re, json, time
t = time.time()
out = subprocess.run(['nvidia-smi', 'nvlink', '-gt', 'd'],
                     capture_output=True, text=True).stdout
g, tx = -1, {}
for line in out.splitlines():
    m = re.match(r'GPU (\d+):', line)
    if m:
        g = int(m.group(1)); tx[g] = 0; continue
    m = re.search(r'Data (Tx|Rx): (\d+) KiB', line)
    if m and m.group(1) == 'Tx':
        tx[g] += int(m.group(2))
print(json.dumps({'t': round(t, 1), 'tx': [tx[i] for i in range(len(tx))]}))
PY
}

eng() {  # 每 engine 累计完成请求数（排除 abort/error），空格分隔
  curl -s localhost:8000/metrics | awk '
    /^vllm:request_success_total\{/ && !/abort|error/ {
      if (match($0, /engine="[^"]*"/)) {
        e = substr($0, RSTART + 8, RLENGTH - 9)
        c[e] += $NF
      }
    }
    END { for (k in c) printf "e%s=%d ", k, c[k]; print "" }'
}

bench() {  # $1=in $2=out $3=seed —— 输出: TTFT ms / 时长 s
  vllm bench serve --backend vllm --model "$M" --dataset-name random \
    --random-input-len "$1" --random-output-len "$2" --num-prompts 1 \
    --request-rate 10000 --ignore-eos --seed "$3" 2>&1 \
    | awk '/Mean TTFT/ {ttft=$NF} /Benchmark duration/ {dur=$NF}
           END {printf "%.1f %.2f", ttft, dur}'
}

round() {  # $1=tag $2=in $3=out $4=seed —— 一轮：快照+bench+快照+归属
  local tag=$1 in=$2 out=$3 seed=$4 a b ttft dur
  eng > /tmp/eng_b_$$; a=$(nvln)
  read -r ttft dur <<< "$(bench "$in" "$out" "$seed")"
  b=$(nvln); eng > /tmp/eng_a_$$
  echo "SNAP ${tag}-b ${a}"
  echo "SNAP ${tag}-a ${b}"
  echo "BENCH ${tag} ${ttft} ${dur}"
  echo "ENG ${tag}-b $(cat /tmp/eng_b_$$)"
  echo "ENG ${tag}-a $(cat /tmp/eng_a_$$)"
}

echo "### traffic-test on $(hostname) $(date +%H:%M:%S)"

# Phase 0: 空闲基线
A=$(nvln); sleep 10; B=$(nvln)
python3 - "$A" "$B" <<'PY'
import json, sys
a, b = json.loads(sys.argv[1]), json.loads(sys.argv[2])
print("IDLECHK %.3f" % (sum(x - y for x, y in zip(b["tx"], a["tx"])) / 2**20))
PY

# Phase 1: 预热（32768 prefill 形状）直到全 engine 覆盖，最多 4 轮
tries=0
while :; do
  bench 32768 8 $((500 + tries)) >/dev/null
  n=$(eng | grep -oc 'e[0-9]*=[1-9]')
  total=$(eng | grep -oc 'e[0-9]*=')
  echo "WARMUP round$((tries+1)) covered ${n}/${total}"
  [ "$n" = "$total" ] && break
  tries=$((tries + 1))
  [ "$tries" -ge 4 ] && { echo "WARMUP WARN 未全覆盖，继续"; break; }
done

# Phase 2: 5 轮 32768/128
for r in 1 2 3 4 5; do
  round "r128-$r" 32768 128 $((700 + r))
done

# Phase 3: 3 轮 32768/1（纯 prefill）
for r in 1 2 3; do
  round "r1-$r" 32768 1 $((800 + r))
done

echo "### traffic-test done on $(hostname) $(date +%H:%M:%S)"
