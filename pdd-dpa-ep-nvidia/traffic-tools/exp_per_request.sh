#!/bin/bash
# Traffic experiment, "per request" mode: N sequential single requests, with before/after
# counter snapshots around EACH request, so every request can be analysed on its own
# (which rank pair, which NIC, how many bytes, how long on the wire).
#
# usage:  LABEL=big N=4 LEN=131072 OUTLEN=16 ./exp_per_request.sh
# env:    LABEL (-> /tmp/exp5-$LABEL), N requests, LEN input tokens, OUTLEN output tokens,
#         PODS / PN / DN as in exp_window.sh (default 1P1D; 2P2D: PODS="pdd-prefill-0 pdd-prefill-1 pdd-decode-0 pdd-decode-1" PN=2 DN=2), DUR sampler seconds
# Preconditions: pools warm (see the document). Requests are sent from pdd-router-0.
TOOLS=$(cd "$(dirname "$0")" && pwd)
cd /root/work/data/distributed-inference/pdd-dpa-ep-nvidia/
LABEL=${LABEL:-big}; N=${N:-4}; LEN=${LEN:-131072}; OUTLEN=${OUTLEN:-16}
PODS=${PODS:-"pdd-prefill-0 pdd-decode-0"}; PN=${PN:-1}; DN=${DN:-1}
O=/tmp/exp5-$LABEL; rm -rf $O; mkdir -p $O
# Save {pod IP: pod name}: the Router log only has IPs, the analysis scripts need the pod names.
kubectl get pods -o json | python3 -c "import json,sys; print(json.dumps({p['status']['podIP']:p['metadata']['name'] for p in json.load(sys.stdin)['items'] if p['status'].get('podIP')}))" > $O/ipmap.json

# Lighter snapshot than exp_window.sh: only RDMA request counters + NIXL metrics (the
# NVLink counters take ~1 s per call and are not needed per request here).
snap() {
  local pids=""
  for p in $PODS; do
    $TOOLS/rdma_request_counters.sh $p > $O/rdma-$p-$1.txt &
    pids="$pids $!"
  done
  kubectl exec -i pdd-router-0 -- python3 - $PN $DN < $TOOLS/nixl_metrics.py > $O/metrics-$1.json &
  pids="$pids $!"
  wait $pids
}

DUR=${DUR:-$((N*220+60))}                         # ~220 s per 131072-token request, plus margin
SP=""
for p in $PODS; do
  kubectl exec -i $p -- python3 - $DUR 0.02 < $TOOLS/roce_bytes_sampler.py > $O/bytes-$p.jsonl 2>$O/bytes-$p.err &
  SP="$SP $!"
done
sleep 3
for k in $(seq 1 $N); do
  snap before$k
  date +%s.%N > $O/t-start$k
  BENCH_SEED=$((500+k)) BENCH_LOG=$O/bench$k.log BENCH_INPUT_LEN=$LEN BENCH_OUTPUT_LEN=$OUTLEN ./pdd.sh bench --num-prompts 1 > /dev/null 2>&1
  date +%s.%N > $O/t-end$k
  snap after$k
  # the last "Selected prefill" line belongs to this request (requests are sequential)
  kubectl logs pdd-router-0 --since=10m | sed 's/\x1b\[[0-9;]*m//g' | grep "Selected prefill" | tail -1 > $O/router$k.txt
  sleep 2
done
touch $O/DONE
wait $SP
echo "done: $O   (analyze with: ./analyze_per_request.py $LABEL)"
