#!/bin/bash
# Count the ESTABLISHED TCP connections of a GPU Pod by peer pod (read-only).
#
# usage: ./tcp_peers.sh <pod> ...         e.g. ./tcp_peers.sh pdd-prefill-0 pdd-prefill-1 pdd-decode-0 pdd-decode-1
#
# For every Pod given it prints how many connections go to each other pod (peer IP mapped to the
# pod name with `kubectl get pods`) and to "other" (e.g. the Kubernetes API, Router, 127.0.0.1).
# Connections of a pod to its own IP are the local ones between its own processes. Used to show that the two nodes of a pool are
# connected by hundreds of control-plane connections (eth0), while a Prefill pod and a Decode pod
# have none: P and D only talk through NIXL point-to-point (RDMA), not TCP.
# NOTE: this script is a rewrite of the one-off command used for the numbers in the document
# (398 / 0); it has been tested on a 1P1D deployment (P<->D: 0) but not re-run on 2P2D.
# Router connections are short-lived, so an idle Pod usually shows none.
declare -A NAME
while read -r ip name; do NAME[$ip]=$name; done < <(kubectl get pods -o jsonpath='{range .items[*]}{.status.podIP}{" "}{.metadata.name}{"\n"}{end}')
for pod in "$@"; do
  echo "== $pod"
  # column 4 of `ss -tn` is the peer address ip:port
  # peer = column 4 (ip:port); strip the port, IPv6 brackets and the ::ffff: prefix of IPv4-mapped addresses
  kubectl exec "$pod" -- ss -tn state established | awk 'NR>1{ip=$4; sub(/:[0-9]+$/,"",ip); gsub(/[\[\]]/,"",ip); sub(/^::ffff:/,"",ip); print ip}' | sort | uniq -c | while read -r n ip; do
    echo "  ${NAME[$ip]:-other($ip)}: $n"
  done
done
