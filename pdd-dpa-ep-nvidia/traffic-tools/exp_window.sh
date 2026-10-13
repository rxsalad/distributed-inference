#!/bin/bash
# Traffic experiment, "window" mode: run ONE bench command and measure the NVLink / RoCE /
# RDMA-request / NIXL traffic of the whole run (before/after counters + a 20 ms RoCE sampler).
#
# usage:
#   LABEL=conc8 BENCH_INPUT_LEN=32768 BENCH_OUTPUT_LEN=16 REPEAT=1 \
#   BENCH_ARGS="--num-prompts 24 --max-concurrency 8" ./exp_window.sh
# env:
#   LABEL     output directory suffix            -> /tmp/exp4-$LABEL
#   PODS      Pods to measure (default 1P1D)     "pdd-prefill-0 pdd-decode-0"
#             2P2D: PODS="pdd-prefill-0 pdd-prefill-1 pdd-decode-0 pdd-decode-1" PN=2 DN=2
#   PN, DN    number of Prefill / Decode Pods    (default 1 1; used by nixl_metrics.py)
#   REPEAT    how many times to run the bench    (default 1)
#   DUR       sampler duration in seconds        (default 400; must cover the whole run)
#   BENCH_*   passed through to `./pdd.sh bench` (BENCH_INPUT_LEN, BENCH_OUTPUT_LEN, ...)
# Preconditions: the pools are WARM (every DP rank has processed one long request), see the
# document; otherwise the first-request overhead pollutes the numbers.
# All tests are sent from pdd-router-0 (pdd.sh does that); the Prefill/Decode Pods only get
# read-only counter reads.
TOOLS=$(cd "$(dirname "$0")" && pwd)             # where the helper scripts live
cd /root/work/data/distributed-inference/pdd-dpa-ep-nvidia/   # where pdd.sh lives
LABEL=${LABEL:-run}; REPEAT=${REPEAT:-1}
PODS=${PODS:-"pdd-prefill-0 pdd-decode-0"}; PN=${PN:-1}; DN=${DN:-1}
O=/tmp/exp4-$LABEL; rm -rf $O; mkdir -p $O
# Save {pod IP: pod name}: the Router log only has IPs, the analysis scripts need the pod names.
kubectl get pods -o json | python3 -c "import json,sys; print(json.dumps({p['status']['podIP']:p['metadata']['name'] for p in json.load(sys.stdin)['items'] if p['status'].get('podIP')}))" > $O/ipmap.json

# snap <tag>: take one snapshot of every cumulative counter, in parallel.
#   nvl-<pod>-<tag>.json    NVLink KiB per GPU           (nvlink_counters.py, inside the Pod)
#   rdma-<pod>-<tag>.txt    RDMA WRITE/READ request counts (rdma_request_counters.sh)
#   metrics-<tag>.json      NIXL + latency metrics        (nixl_metrics.py, inside the router Pod)
snap() {
  local pids=""
  for p in $PODS; do
    ( kubectl exec -i $p -- python3 - < $TOOLS/nvlink_counters.py > $O/nvl-$p-$1.json
      $TOOLS/rdma_request_counters.sh $p > $O/rdma-$p-$1.txt ) &
    pids="$pids $!"
  done
  kubectl exec -i pdd-router-0 -- python3 - $PN $DN < $TOOLS/nixl_metrics.py > $O/metrics-$1.json &
  pids="$pids $!"
  wait $pids
}

# Start one RoCE byte sampler per Pod (50 samples/s) before the run, so that the whole
# run, including its start and end, is covered.
DUR=${DUR:-400}
SP=""
for p in $PODS; do
  kubectl exec -i $p -- python3 - $DUR 0.02 < $TOOLS/roce_bytes_sampler.py > $O/bytes-$p.jsonl 2>$O/bytes-$p.err &
  SP="$SP $!"
done
sleep 3
snap before
date +%s.%N > $O/t-start                          # wall-clock window, used to cut the sampler data
for r in $(seq 1 $REPEAT); do
  BENCH_SEED=$((400+r)) BENCH_LOG=$O/bench$r.log ./pdd.sh bench $BENCH_ARGS > /dev/null 2>&1
done
date +%s.%N > $O/t-end
snap after
# The router logs which Prefill / Decode rank each request was paired with; needed to know
# which NIC pair carried which request.
kubectl logs pdd-router-0 --since=15m | sed 's/\x1b\[[0-9;]*m//g' | grep "Selected prefill" > $O/router.txt
touch $O/DONE
wait $SP
echo "done: $O   (analyze with: ./analyze_window.py $LABEL)"
