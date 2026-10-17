#!/bin/bash
# v2 benchmark 预热脚本（Pod 内执行）：
#   kubectl exec -i <pod>-0 -- bash -s < sn-warmup2.sh
# 与 sn-warmup.sh 相同流程，prefill 形状扩为 in=1000 / 8000 / 32000（覆盖 v2 新增的
# 32k 分块 prefill 与 KV 压力用例的形状）；Phase B C=512；Phase C 金丝雀。
set -u
M=deepseek-ai/DeepSeek-V3
TMP=/tmp/warmup
mkdir -p "$TMP"

echo "### warmup on $(hostname)"

snap() { # $1=输出文件: 各 engine 累计完成请求数（排除 abort/error）
  curl -s localhost:8000/metrics | awk '
    /^vllm:request_success_total\{/ {
      line = $0
      if (line ~ /finished_reason="abort"/ || line ~ /finished_reason="error"/) next
      if (match(line, /engine="[^"]*"/)) {
        e = substr(line, RSTART + 8, RLENGTH - 9)
        c[e] += $NF
      }
    }
    END { for (k in c) print k, c[k] + 0 }' | sort -n > "$1"
}

bench() { # $1=in_len $2=out_len $3=num_prompts
  vllm bench serve --backend vllm --model "$M" \
    --dataset-name random --random-input-len "$1" --random-output-len "$2" \
    --num-prompts "$3" --request-rate 10000 --ignore-eos \
    --save-result --result-dir "$TMP" >/dev/null 2>&1 \
    || echo "  WARN: vllm bench (in=$1 out=$2 n=$3) 非零退出码"
}

covered() { # $1=before $2=after -> 输出 "覆盖数 总数"
  awk 'NR == FNR { a[$1] = $2; next }
       { if ($2 - a[$1] > 0) d++ }
       END { printf "%d %d", d + 0, FNR }' "$1" "$2"
}

warm_shape() { # $1=in_len
  local in=$1 tries=0 n total
  echo "== Phase A: in=$in out=8 n=16 =="
  snap "$TMP/b"
  bench "$in" 8 16
  while :; do
    snap "$TMP/a"
    set -- $(covered "$TMP/b" "$TMP/a")
    n=$1; total=$2
    echo "  engine coverage: ${n}/${total}"
    [ "$n" -eq "$total" ] && break
    tries=$((tries + 1))
    if [ "$tries" -ge 3 ]; then
      echo "  WARN: 3 轮后仍未全覆盖（${n}/${total}），继续但记录在案"
      break
    fi
    echo "  -> 补发一轮 16 prompts"
    bench "$in" 8 16
  done
}

warm_shape 1000
warm_shape 8000
warm_shape 32000

echo "== Phase B: C=512 in=1000 out=64 =="
bench 1000 64 512

echo "== Phase C: canary（2 次单请求 600 token）=="
REQ="{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from one to fifty.\"}],\"max_tokens\":600,\"ignore_eos\":true}"
t1=0; t2=0
for run in 1 2; do
  T0=$(date +%s.%N)
  curl -s -o /dev/null localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" -d "$REQ"
  T1=$(date +%s.%N)
  d=$(echo "$T1 $T0" | awk '{printf "%.3f", $1 - $2}')
  echo "  run$run: ${d} s"
  [ "$run" = 1 ] && t1=$d || t2=$d
done
awk -v a="$t1" -v b="$t2" 'BEGIN {
  if (b < a * 0.95) print "  canary: RECHECK（run2 比 run1 快 >5%，仍有冷路径）"
  else print "  canary: PASS（两次耗时接近）"
}'

echo "### warmup done on $(hostname)"
