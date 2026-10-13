#!/bin/bash
# A/B check that the 20 ms RoCE sampler does not disturb the measured system.
#
# usage: ./sampler_ab.sh          (PODS: default 2P2D; set PODS for other layouts)
# Runs the default bench (32768 tokens in / 128 out) several times WITHOUT the sampler, several
# times WITH roce_bytes_sampler.py running in every GPU Pod, and again without, and prints the
# TTFT / TPOT of each run to /tmp/ab/result.txt. If the numbers of the two groups overlap, the
# sampling has no visible effect (result in the document: 3.54-3.64 s without, 3.58 s with).
# Pools must be warm (see the document); requests go through pdd-router-0 (pdd.sh bench).
TOOLS=$(cd "$(dirname "$0")" && pwd)
cd /root/work/data/distributed-inference/pdd-dpa-ep-nvidia/
PODS=${PODS:-"pdd-prefill-0 pdd-prefill-1 pdd-decode-0 pdd-decode-1"}
O=/tmp/ab; rm -rf $O; mkdir -p $O
run() { # <label> <seed>
  BENCH_SEED=$2 BENCH_LOG=$O/$1-$2.log ./pdd.sh bench > /dev/null 2>&1
  echo "$1 seed=$2 TTFT=$(grep 'Mean TTFT' $O/$1-$2.log | awk '{print $4}') TPOT=$(grep 'Mean TPOT' $O/$1-$2.log | awk '{print $4}')" | tee -a $O/result.txt
}
for s in 1 601 602; do run nosampler $s; done
for s in 1 603 604; do
  SP=""
  for p in $PODS; do kubectl exec -i $p -- python3 - 45 0.02 < $TOOLS/roce_bytes_sampler.py > /dev/null 2>&1 & SP="$SP $!"; done
  sleep 3                                          # let the samplers start before the request
  run sampler20ms $s
  wait $SP
done
for s in 1 605; do run nosampler $s; done
touch $O/DONE
