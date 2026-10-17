#!/bin/bash
# 单节点组合 benchmark 执行脚本（Pod 内执行）：
#   kubectl exec -i <pod>-0 -- bash -s < sn-bench2.sh <phase>    # phase ∈ 1..7
#
# 阶段 1: 单流延迟    100/100 C=1、1000/2000 C=1（各 2 次）
# 阶段 2: 并发扫描    1000/1000，C ∈ {8,32,64,128,160,192,224,256,320,384,512}（各 2 次）
# 阶段 3: 负载形态    纯prefill 8000/1 C=64、prefill为主 8000/1000 C=64、decode为主 1000/8000 C=64、
#                    长上下文 20000/1000 C=16（各 2 次）+ KV 压力 3 梯度（各 2 次）：
#                    32k/4096，C=64/80/104 → KV 需求 2.36M/2.95M/3.83M ≈ TP 组实例容量
#                    （2.41–2.43M）的 97%/121%/158%；DP 组合计容量（14.2–14.9M）远未触顶
# 阶段 4: 饱和吞吐    500/512，num-prompts=3072 + --max-concurrency 1024（3 波次，3 次）
# 阶段 5: 开环 SLO    1000/1000，Poisson 到达 2/5/8 req/s × 240 prompts（各 1 次）
# 阶段 6: ShareGPT    真实数据集（/tmp/ShareGPT_subset.json），C=64 与 C=256 闭环（各 2 次，
#                    固定 seed：同 rep 同 C 跨配置用同 seed，采样可比）
# 阶段 7: openai-chat 1000/1000 C=256、500/512 C=1024（含服务端分词，各 1 次）
#
# 统一：--ignore-eos；JSON 存 /tmp/bench/<tag>.json；每用例第 1 次附带 GPU 采样与
#       per-engine 快照（含 preemption 计数）；每阶段起止各记一次 RDMA 计数（预期 0）。
set -u
M=deepseek-ai/DeepSeek-V3
OUT=/tmp/bench
mkdir -p "$OUT"
PHASE=${1:?usage: sn-bench2.sh <phase 1-7>}
SB=$((PHASE * 100))
BK=vllm        # backend: vllm / openai-chat
DS=random      # dataset: random / sharegpt
RR=10000       # request-rate
EXTRA=""       # 追加参数（如 --max-concurrency）

mon_on() { # $1=tag
  ( while :; do
      echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits | paste -sd';' -)"
      sleep 2
    done > "$OUT/$1.gpu.log" ) &
  echo $! > /tmp/mon.pid
}
mon_off() { kill "$(cat /tmp/mon.pid)" 2>/dev/null || true; }

eng_snap() { # $1=文件：各 engine 的完成请求数 + preemption 计数
  curl -s localhost:8000/metrics | grep -E '^vllm:(request_success_total|num_preemptions_total)\{' \
    | grep -vE 'finished_reason="(abort|error)"' > "$1"
}

bench() { # $1=tag $2=in $3=out $4=n $5=seed $6=rep
  echo "=== [$(hostname)] $1 (in=$2 out=$3 n=$4 seed=$5 rep=$6) ==="
  if [ "$6" = 1 ]; then mon_on "$1"; eng_snap "$OUT/$1.engines.b"; fi
  if [ "$DS" = sharegpt ]; then
    vllm bench serve --backend "$BK" --model "$M" --dataset-name sharegpt \
      --num-prompts "$4" --request-rate "$RR" --ignore-eos --seed "$5" $EXTRA \
      --save-result --result-dir "$OUT" --result-filename "$1.json" 2>&1 | tail -32
  else
    vllm bench serve --backend "$BK" --model "$M" --dataset-name random \
      --random-input-len "$2" --random-output-len "$3" \
      --num-prompts "$4" --request-rate "$RR" --ignore-eos --seed "$5" $EXTRA \
      --save-result --result-dir "$OUT" --result-filename "$1.json" 2>&1 | tail -32
  fi
  local rc=$?
  if [ "$6" = 1 ]; then mon_off; eng_snap "$OUT/$1.engines.a"; fi
  echo "--- exit=$rc"
}

rdma_cnt() { rdma statistic show link 2>/dev/null | awk '/rx_write_requests/{s+=$2} END{print s+0}'; }

echo "### phase $PHASE start on $(hostname) $(date +%H:%M:%S)"
echo "rdma.p$PHASE.start = $(rdma_cnt)"

case $PHASE in
1)
  bench p1-short-100-100-r1    100  100  1  $((SB+11)) 1
  bench p1-short-100-100-r2    100  100  1  $((SB+12)) 2
  bench p1-long-1000-2000-r1   1000 2000 1  $((SB+13)) 1
  bench p1-long-1000-2000-r2   1000 2000 1  $((SB+14)) 2
  ;;
2)
  for C in 8 32 64 128 160 192 224 256 320 384 512; do
    bench p2-1000-1000-c$C-r1 1000 1000 $C $((SB+C))     1
    bench p2-1000-1000-c$C-r2 1000 1000 $C $((SB+C+1000)) 2
  done
  ;;
3)
  bench p3-pureprefill-8000-1-r1   8000  1    64 $((SB+21)) 1
  bench p3-pureprefill-8000-1-r2   8000  1    64 $((SB+22)) 2
  bench p3-prefill-8000-1000-r1    8000  1000 64 $((SB+23)) 1
  bench p3-prefill-8000-1000-r2    8000  1000 64 $((SB+24)) 2
  bench p3-decode-1000-8000-r1     1000  8000 64 $((SB+25)) 1
  bench p3-decode-1000-8000-r2     1000  8000 64 $((SB+26)) 2
  bench p3-long-20000-1000-r1      20000 1000 16  $((SB+27)) 1
  bench p3-long-20000-1000-r2      20000 1000 16  $((SB+28)) 2
  bench p3-kv97-32000-4096-c64-r1  32000 4096 64  $((SB+29)) 1
  bench p3-kv97-32000-4096-c64-r2  32000 4096 64  $((SB+29)) 2
  bench p3-kv121-32000-4096-c80-r1 32000 4096 80  $((SB+30)) 1
  bench p3-kv121-32000-4096-c80-r2 32000 4096 80  $((SB+30)) 2
  bench p3-kv158-32000-4096-c104-r1 32000 4096 104 $((SB+39)) 1
  bench p3-kv158-32000-4096-c104-r2 32000 4096 104 $((SB+39)) 2
  ;;
4)
  EXTRA="--max-concurrency 1024"
  bench p4-sat-500-512-w3072-r1 500 512 3072 $((SB+31)) 1
  bench p4-sat-500-512-w3072-r2 500 512 3072 $((SB+32)) 2
  bench p4-sat-500-512-w3072-r3 500 512 3072 $((SB+33)) 3
  EXTRA=""
  ;;
5)
  for R in 2 5 8; do
    RR=$R
    bench p5-openloop-r${R}rps-r1 1000 1000 240 $((SB+40+R)) 1
    RR=10000
  done
  ;;
6)
  DS=sharegpt
  EXTRA="--dataset-path /tmp/ShareGPT_subset.json"
  bench p6-sharegpt-c64-r1  0 0 64  $((SB+51)) 1
  bench p6-sharegpt-c64-r2  0 0 64  $((SB+151)) 2
  bench p6-sharegpt-c256-r1 0 0 256 $((SB+52)) 1
  bench p6-sharegpt-c256-r2 0 0 256 $((SB+152)) 2
  EXTRA=""
  DS=random
  ;;
7)
  BK=openai-chat
  EXTRA="--endpoint /v1/chat/completions"
  bench p7-chat-1000-1000-c256-r1 1000 1000 256 $((SB+61)) 1
  bench p7-chat-500-512-c1024-r1 500  512  1024 $((SB+62)) 1
  EXTRA=""
  BK=vllm
  ;;
*) echo "unknown phase $PHASE"; exit 1;;
esac

echo "rdma.p$PHASE.end = $(rdma_cnt)"
echo "### phase $PHASE done on $(hostname) $(date +%H:%M:%S)"
ls -la "$OUT" | tail -3
