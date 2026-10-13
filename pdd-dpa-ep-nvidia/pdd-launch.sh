#!/bin/bash
# Startup script shared by Prefill / Decode (mounted at /scripts/pdd-launch.sh via ConfigMap pdd-scripts).
# Each Pod = one node = 8 local DP ranks; the node count comes from the NODES env var, total DP = NODES * 8.
#
# Environment variables injected by the StatefulSet:
#   ROLE_APP   pdd-prefill | pdd-decode (also the StatefulSet name and the headless Service name prefix)
#   KV_ROLE    kv_producer | kv_consumer
#   HTTP_PORT  API port of this role (Prefill 8001, Decode 8002)
#   SIDE_PORT  NIXL side-channel port (Prefill 5557, Decode 5558)
#   NODES      node count of this role (= StatefulSet replicas)
#   MODEL      model name
#   POD_IP     this Pod's IP (downward API), used as the NIXL side-channel address
#   PREFIX_CACHING  on | off (default off). vLLM 0.31.0 enables prefix caching by default (measured),
#              turn it off when testing PD disaggregation, otherwise cache hits distort TTFT and KV transfer volume; P and D must match
#   HIDE_HEALTH_LOG  on | off (default off, i.e. not hidden). When on, hides the uvicorn access log of /health, /metrics and /ping
#              (probes and Router health checks otherwise log a line every few seconds), while keeping POST /v1/... request logs
set -ex

LOCAL_DP=8
RPC_PORT=13345

NODE_IDX=${HOSTNAME##*-}
DP_SIZE=$((NODES * LOCAL_DP))
START_RANK=$((NODE_IDX * LOCAL_DP))
HEAD_HOST=${ROLE_APP}-0.${ROLE_APP}-headless

# Loop until a non-empty IP is resolved (DNS has a transient race: getent may succeed and the IP be empty right after)
HEAD_IP=""
until [ -n "${HEAD_IP}" ]; do
  HEAD_IP=$(getent ahostsv4 ${HEAD_HOST} 2>/dev/null | awk 'NR==1{print $1}')
  [ -n "${HEAD_IP}" ] || { echo "waiting for ${HEAD_HOST} ..."; sleep 2; }
done
echo "PREFIX_CACHING=${PREFIX_CACHING:-off} HIDE_HEALTH_LOG=${HIDE_HEALTH_LOG:-off}"
echo "ROLE=${ROLE_APP} NODE_IDX=${NODE_IDX}/${NODES} DP_SIZE=${DP_SIZE} START_RANK=${START_RANK} HEAD_IP=${HEAD_IP} POD_IP=${POD_IP}"

# The hostPath cache is not shared between nodes, so each node downloads on its own (returns immediately if already cached)
hf download ${MODEL}

# Non-zero nodes: wait for the DP coordinator's (-0) RPC port before starting
if [ "${NODE_IDX}" != "0" ]; then
  until timeout 2 bash -c "</dev/tcp/${HEAD_IP}/${RPC_PORT}" 2>/dev/null; do
    echo "waiting for DP master ${HEAD_IP}:${RPC_PORT} ..."; sleep 5
  done
fi

# NIXL side channel: used for the handshake; the address must be this Pod's IP, reachable from other Pods
export VLLM_NIXL_SIDE_CHANNEL_HOST=${POD_IP}
export VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_PORT}

EXTRA=()
case "${HIDE_HEALTH_LOG:-off}" in
  on)  EXTRA+=(--disable-access-log-for-endpoints /health,/metrics,/ping) ;;
  off) ;;   # do not hide (vLLM default)
  *)   echo "HIDE_HEALTH_LOG must be on or off, got: ${HIDE_HEALTH_LOG}" >&2; exit 1 ;;
esac
case "${PREFIX_CACHING:-off}" in
  off) EXTRA+=(--no-enable-prefix-caching) ;;
  on)  ;;   # no flag, use vLLM's default (enabled)
  *)   echo "PREFIX_CACHING must be on or off, got: ${PREFIX_CACHING}" >&2; exit 1 ;;
esac
if [ "${KV_ROLE}" = "kv_consumer" ]; then
  # Per the page: the Decode pool additionally uses decode-only CUDA graphs
  EXTRA+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
fi

exec vllm serve ${MODEL} \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port ${HTTP_PORT} \
  --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"${KV_ROLE}\",\"kv_load_failure_policy\":\"fail\"}" \
  "${EXTRA[@]}" \
  --data-parallel-size ${DP_SIZE} \
  --data-parallel-size-local ${LOCAL_DP} \
  --data-parallel-start-rank ${START_RANK} \
  --data-parallel-address ${HEAD_IP} \
  --data-parallel-rpc-port ${RPC_PORT} \
  --data-parallel-hybrid-lb \
  --enable-expert-parallel \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v3 \
  --reasoning-parser deepseek_v3
