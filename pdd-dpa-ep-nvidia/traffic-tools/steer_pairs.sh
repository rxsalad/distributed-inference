#!/bin/bash
# Same-rail vs cross-rail KV transfer: try to steer each 131072-token request to a chosen
# position of the Router's selection cycle (the P->D pairing is a fixed cycle, see router_pairs.py).
#
# usage: TARGETS="6 10 11 7 9 12" ./steer_pairs.sh        (cycle length CYCLE, default 16 = 2P2D)
# For every target position t it
#   1. counts the "Selected prefill" lines in the Router log (N),
#   2. sends (t - N mod CYCLE) tiny filler requests (max_tokens=1) to advance the round robin,
#   3. snapshots counters, sends ONE 131072-token request (pdd.sh bench), snapshots again.
# Output in /tmp/exp6 has the same layout as exp_per_request.sh (analyze with a copy renamed to
# /tmp/exp5-<label> or by pointing analyze_per_request.py at it), plus plan.txt.
#
# KNOWN LIMITATION (why it only half worked): `vllm bench` first sends one initial test request
# that advances the cycle by one extra step but is not counted in the log, so the position is
# off by one; 11 of the 12 requests missed their target pair. Fix: add one more filler or send
# the big request directly with curl/python instead of `vllm bench`.
TOOLS=$(cd "$(dirname "$0")" && pwd)
cd /root/work/data/distributed-inference/pdd-dpa-ep-nvidia/
O=/tmp/exp6; rm -rf $O; mkdir -p $O
PODS=${PODS:-"pdd-prefill-0 pdd-prefill-1 pdd-decode-0 pdd-decode-1"}; PN=${PN:-2}; DN=${DN:-2}; CYCLE=${CYCLE:-16}
TARGETS=${TARGETS:-"6 10 11 7 9 12 6 10 11 7 9 12"}
count() { kubectl logs pdd-router-0 | sed 's/\x1b\[[0-9;]*m//g' | grep -c "Selected prefill"; }
snap() {
  local pids=""
  for p in $PODS; do $TOOLS/rdma_request_counters.sh $p > $O/rdma-$p-$1.txt & pids="$pids $!"; done
  kubectl exec -i pdd-router-0 -- python3 - $PN $DN < $TOOLS/nixl_metrics.py > $O/metrics-$1.json &
  pids="$pids $!"
  wait $pids
}
filler() {   # one 1-token completion through the Router: advances the round robin, costs nothing
  kubectl exec pdd-router-0 -- python3 -c "
import json,urllib.request
b={'model':'deepseek-ai/DeepSeek-V3','prompt':'hi','max_tokens':1}
r=urllib.request.Request('http://localhost:30000/v1/completions',json.dumps(b).encode(),{'Content-Type':'application/json'})
urllib.request.urlopen(r,timeout=60).read()" > /dev/null 2>&1
}
kubectl get pods -o json | python3 -c "import json,sys; print(json.dumps({p['status']['podIP']:p['metadata']['name'] for p in json.load(sys.stdin)['items'] if p['status'].get('podIP')}))" > $O/ipmap.json
SP=""
for p in $PODS; do kubectl exec -i $p -- python3 - 1500 0.02 < $TOOLS/roce_bytes_sampler.py > $O/bytes-$p.jsonl 2>$O/bytes-$p.err & SP="$SP $!"; done
sleep 3
k=0
for t in $TARGETS; do
  k=$((k+1))
  N=$(count); need=$(( (t - N % CYCLE + CYCLE) % CYCLE ))
  for i in $(seq 1 $need); do filler; done
  snap before$k
  date +%s.%N > $O/t-start$k
  BENCH_SEED=$((800+k)) BENCH_LOG=$O/bench$k.log BENCH_INPUT_LEN=131072 BENCH_OUTPUT_LEN=16 ./pdd.sh bench --num-prompts 1 > /dev/null 2>&1
  date +%s.%N > $O/t-end$k
  snap after$k
  kubectl logs pdd-router-0 | sed 's/\x1b\[[0-9;]*m//g' | grep "Selected prefill" | tail -1 > $O/router$k.txt
  echo "$k target=$t fillers=$need count_before=$N" >> $O/plan.txt
done
touch $O/DONE
wait $SP
