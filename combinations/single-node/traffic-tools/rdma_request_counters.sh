#!/bin/bash
# Print the RDMA request counters of the 8 data NICs of one Pod (read-only).
#
# usage: ./rdma_request_counters.sh <pod>
# output: one line per NIC:  <device> <rx_write_requests> <rx_read_requests>
#
# These are the number of RDMA WRITE / READ requests that arrived at this NIC (as the
# responder). They are cumulative, so take a snapshot before and after and subtract.
#   - In this deployment the Decode side issues RDMA READs (NIXL pull mode), so the
#     READ requests are counted on the NIC of the PREFILL Pod that serves the data.
#     A non-zero read delta therefore tells which Prefill NIC served a request.
#   - The count is per request/descriptor, not per byte: e.g. 61 descriptors x 144
#     requests each = 8784 requests for a 32768-token KV, 62,464 descriptors give
#     124,928 requests for a 131072-token KV.
#
# `rdma statistic show link mlx5_N/1` prints one long line of "name value" pairs;
# sed keeps only the device name and the two counters we need.
POD=${1:?usage: $0 <pod>}
kubectl exec "$POD" -- bash -c '
for n in 0 2 4 6 8 10 12 14; do
  rdma statistic show link mlx5_$n/1 \
    | sed -E "s/^link (mlx5_[0-9]+)\/1 rx_write_requests ([0-9]+) rx_read_requests ([0-9]+).*/\1 \2 \3/"
done'
