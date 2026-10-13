#!/bin/bash
# Control script for the PD disaggregation deployment.
#   PREFILL_NODES=2 DECODE_NODES=2 ./pdd.sh up       # deploy (re-runnable, e.g. to change node counts)
#   ./pdd.sh render                                  # render the yaml only, do not deploy
#   ./pdd.sh status
#   ./pdd.sh test                                    # send one request through the Router to verify end to end
#   ./pdd.sh bench [extra vllm bench serve args]     # benchmark through the Router; output is also written to benchmark-<time>.log
#       Defaults: 32768 input / 128 output / 1 request / request-rate 1 / seed 1; override with env vars:
#       BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_NUM_PROMPTS BENCH_REQUEST_RATE BENCH_SEED BENCH_LOG
#   ./pdd.sh metrics [window_seconds]                # summarize Prefill/Decode /metrics (KV transfer, latency, queue, KV cache); with a window, show deltas/rates
#   ./pdd.sh down                                    # delete (the HF cache is kept)
#
# Optional env vars: IMAGE (default vllm/vllm-openai:latest), MODEL (default deepseek-ai/DeepSeek-V3),
#   PREFIX_CACHING (on|off, default off: prefix caching is disabled when testing PD disaggregation, applied to both P and D),
#   HIDE_HEALTH_LOG (on|off, default off = not hidden; on hides the /health etc. access logs of P and D),
#   UCX_LOG_LEVEL (fatal|error|warn|info|debug|trace, default info: shows UCX transport/device selection; use warn to cut log volume)
set -euo pipefail
cd "$(dirname "$0")"

export PREFILL_NODES=${PREFILL_NODES:-2}
export DECODE_NODES=${DECODE_NODES:-2}
export IMAGE=${IMAGE:-vllm/vllm-openai:latest}
export MODEL=${MODEL:-deepseek-ai/DeepSeek-V3}
export PREFIX_CACHING=${PREFIX_CACHING:-off}
export HIDE_HEALTH_LOG=${HIDE_HEALTH_LOG:-off}
export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-info}
VARS='${PREFILL_NODES} ${DECODE_NODES} ${IMAGE} ${MODEL} ${PREFIX_CACHING} ${HIDE_HEALTH_LOG} ${UCX_LOG_LEVEL}'

render() { envsubst "${VARS}" < pdd-dep.yaml; }

apply_configmap() {
  # The scripts live in a ConfigMap: re-run up after editing them; running Pods must be restarted to pick up new scripts
  kubectl create configmap pdd-scripts \
    --from-file=pdd-launch.sh --from-file=pdd-router.py \
    --dry-run=client -o yaml | kubectl apply -f -
}

case "${1:-}" in
  render)
    render
    ;;
  up)
    for n in "${PREFILL_NODES}" "${DECODE_NODES}"; do
      [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "node count must be a positive integer: $n" >&2; exit 1; }
    done
    [[ "${PREFIX_CACHING}" =~ ^(on|off)$ ]] || { echo "PREFIX_CACHING must be on or off: ${PREFIX_CACHING}" >&2; exit 1; }
    echo "prefix caching: ${PREFIX_CACHING}"
    [[ "${HIDE_HEALTH_LOG}" =~ ^(on|off)$ ]] || { echo "HIDE_HEALTH_LOG must be on or off: ${HIDE_HEALTH_LOG}" >&2; exit 1; }
    echo "hide health log: ${HIDE_HEALTH_LOG}"
    [[ "${UCX_LOG_LEVEL}" =~ ^(fatal|error|warn|info|debug|trace)$ ]] || { echo "UCX_LOG_LEVEL must be one of fatal|error|warn|info|debug|trace: ${UCX_LOG_LEVEL}" >&2; exit 1; }
    echo "UCX log level: ${UCX_LOG_LEVEL}"
    gpu_nodes=$(kubectl get nodes -l doks.digitalocean.com/gpu-brand=nvidia --no-headers | wc -l)
    need=$((PREFILL_NODES + DECODE_NODES))
    echo "Prefill ${PREFILL_NODES} node(s) (DP$((PREFILL_NODES * 8))+EP$((PREFILL_NODES * 8))) + Decode ${DECODE_NODES} node(s) (DP$((DECODE_NODES * 8))+EP$((DECODE_NODES * 8))) = ${need} node(s); the cluster has ${gpu_nodes} GPU node(s)"
    if [ "${need}" -gt "${gpu_nodes}" ]; then
      echo "Error: the required node count exceeds the number of GPU nodes in the cluster" >&2; exit 1
    fi
    apply_configmap
    render | kubectl apply -f -
    echo "Logs: kubectl logs -f pdd-prefill-0 / pdd-decode-0 / pdd-router-0"
    echo "Verify: ./pdd.sh test      (sends the request with python inside the Pod)"
    ;;
  test)
    kubectl exec -i pdd-router-0 -- python3 - <<'PYEOF'
import json, os, time, urllib.request
body = {"model": os.environ.get("MODEL", "deepseek-ai/DeepSeek-V3"),
        "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}], "max_tokens": 48}
req = urllib.request.Request("http://localhost:30000/v1/chat/completions",
                             json.dumps(body).encode(), {"Content-Type": "application/json"})
t = time.time()
r = urllib.request.urlopen(req, timeout=120)
d = json.loads(r.read())
print("HTTP", r.status, "%.2fs" % (time.time() - t))
print(d["choices"][0]["message"]["content"], d.get("usage"))
PYEOF
    ;;
  metrics)
    # Read /metrics of every Prefill / Decode node from pdd-router-0 (light, read-only) and print a summary.
    # Optional argument: window in seconds. Without it the values are cumulative since the engines started;
    # with it the metrics are sampled twice and the deltas / rates over the window are shown.
    window=${2:-0}
    [[ "${window}" =~ ^[0-9]+$ ]] || { echo "usage: ./pdd.sh metrics [window_seconds]" >&2; exit 1; }
    pn=$(kubectl get statefulset pdd-prefill -o jsonpath='{.spec.replicas}')
    dn=$(kubectl get statefulset pdd-decode -o jsonpath='{.spec.replicas}')
    kubectl exec -i pdd-router-0 -- python3 - "${pn}" "${dn}" "${window}" <<'PYEOF'
import re, sys, time, urllib.request

pn, dn, window = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
ROLES = [("prefill", pn, 8001), ("decode", dn, 8002)]
LINE = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})? (\S+)$')


def snapshot(role, nodes, port):
    """Sum every sample by metric name across all nodes and engines; keep per-engine KV cache usage."""
    sums, kv = {}, []
    for i in range(nodes):
        url = f"http://pdd-{role}-{i}.pdd-{role}-headless:{port}/metrics"
        text = urllib.request.urlopen(url, timeout=15).read().decode()
        for line in text.splitlines():
            m = LINE.match(line)
            if not m or m.group(1).endswith("_bucket"):
                continue
            name, value = m.group(1), float(m.group(3))
            sums[name] = sums.get(name, 0.0) + value
            if name == "vllm:kv_cache_usage_perc":
                kv.append(value)
    return sums, kv


def take():
    return {r: snapshot(r, n, p) for r, n, p in ROLES}


s0 = take()
if window:
    time.sleep(window)
    s1 = take()
else:
    s1 = s0


def d(role, name):
    v1 = s1[role][0].get(name, 0.0)
    return v1 - (s0[role][0].get(name, 0.0) if window else 0.0)


def mean_ms(role, base):
    c = d(role, base + "_count")
    return f"{d(role, base + '_sum') / c * 1000:.1f} ms (n={c:.0f})" if c else "-"


def ratio(role, hits, queries):
    q = d(role, queries)
    return f"{d(role, hits) / q * 100:.1f}% ({d(role, hits):.0f}/{q:.0f})" if q else "-"


print(f"{'delta over %ds' % window if window else 'cumulative since engine start'}; prefill nodes={pn}, decode nodes={dn}")
for role, _, _ in ROLES:
    g = s1[role][0]
    kv = s1[role][1]
    print(f"\n== {role}")
    print(f"requests          running={g.get('vllm:num_requests_running', 0):.0f} waiting={g.get('vllm:num_requests_waiting', 0):.0f}"
          f"  succeeded={d(role, 'vllm:request_success_total'):.0f}  preemptions={d(role, 'vllm:num_preemptions_total'):.0f}")
    print(f"KV cache usage    " + (f"max={max(kv) * 100:.1f}% mean={sum(kv) / len(kv) * 100:.1f}% ({len(kv)} engines)" if kv else "-"))
    pt, gt = d(role, "vllm:prompt_tokens_total"), d(role, "vllm:generation_tokens_total")
    rate = f"  ({pt / window:.1f} / {gt / window:.1f} tok/s)" if window else ""
    print(f"tokens            prompt={pt:.0f} generation={gt:.0f}{rate}")
    for label, base in [("TTFT", "vllm:time_to_first_token_seconds"), ("TPOT", "vllm:request_time_per_output_token_seconds"),
                        ("ITL", "vllm:inter_token_latency_seconds"), ("queue time", "vllm:request_queue_time_seconds"),
                        ("prefill time", "vllm:request_prefill_time_seconds"), ("decode time", "vllm:request_decode_time_seconds"),
                        ("e2e latency", "vllm:e2e_request_latency_seconds")]:
        print(f"{label:<17} mean {mean_ms(role, base)}")
    print(f"prefix cache      local={ratio(role, 'vllm:prefix_cache_hits_total', 'vllm:prefix_cache_queries_total')}"
          f"  external(NIXL)={ratio(role, 'vllm:external_prefix_cache_hits_total', 'vllm:external_prefix_cache_queries_total')}")
    n = d(role, "vllm:nixl_xfer_time_seconds_count")
    b, t = d(role, "vllm:nixl_bytes_transferred_sum"), d(role, "vllm:nixl_xfer_time_seconds_sum")
    if n:
        print(f"NIXL transfers    count={n:.0f} total={b / 1e9:.2f} GB avg={b / n / 1e6:.1f} MB avg_time={t / n * 1000:.1f} ms"
              f" bandwidth={b / t / 1e9:.1f} GB/s (bytes / summed xfer time)")
    else:
        print("NIXL transfers    count=0 (only the consumer/Decode side records transfers)")
    print(f"NIXL failures     transfers={d(role, 'vllm:nixl_num_failed_transfers_total'):.0f}"
          f" notifications={d(role, 'vllm:nixl_num_failed_notifications_total'):.0f}"
          f" expired_reqs={d(role, 'vllm:nixl_num_kv_expired_reqs_total'):.0f}")
PYEOF
    ;;
  bench)
    shift
    # Runs in pdd-router-0: the Router uses the same image as P/D (has vllm) and sits on a CPU node, so it does not use CPU of the GPU Pods;
    # it talks directly to this Pod's vllm-router (localhost:30000). The tokenizer is downloaded from HF on first use and cached on the hostPath.
    kubectl wait --for=condition=Ready pod/pdd-router-0 --timeout=5s >/dev/null \
      || { echo "pdd-router-0 is not Ready yet, run ./pdd.sh status first" >&2; exit 1; }
    log=${BENCH_LOG:-benchmark-$(date +%Y%m%d-%H%M%S).log}
    echo "Log written to ${log}; see PREFIX_CACHING for the prefix caching switch (with a fixed seed, repeated runs hit the cache when it is on)" >&2
    kubectl exec pdd-router-0 -- vllm bench serve \
      --host localhost --port 30000 --seed "${BENCH_SEED:-1}" \
      --backend vllm --model "${MODEL}" \
      --dataset-name random \
      --random-input-len "${BENCH_INPUT_LEN:-32768}" --random-output-len "${BENCH_OUTPUT_LEN:-128}" \
      --num-prompts "${BENCH_NUM_PROMPTS:-1}" --burstiness 1 \
      --request-rate "${BENCH_REQUEST_RATE:-1}" "$@" 2>&1 | tee "${log}"
    ;;
  status)
    kubectl get pods,svc -l 'app in (pdd-prefill,pdd-decode,pdd-router)' -o wide
    ;;
  down)
    render | kubectl delete -f - --ignore-not-found
    kubectl delete configmap pdd-scripts --ignore-not-found
    ;;
  *)
    sed -n '2,/^set -/p' "$0" | sed '$d'; exit 1
    ;;
esac
