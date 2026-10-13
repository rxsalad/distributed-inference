#!/usr/bin/env python3
"""High-rate, read-only sampler of RDMA (RoCE) byte counters for the 8 NICs of a Pod.

Run it INSIDE a Prefill/Decode Pod, for DUR seconds, sampling every INTERVAL seconds:

    kubectl exec -i pdd-prefill-0 -- python3 - 400 0.02 < roce_bytes_sampler.py > bytes-pdd-prefill-0.jsonl

What it reads
  The mlx5 driver exposes the counters tx_vport_rdma_unicast_bytes and
  rx_vport_rdma_unicast_bytes through ethtool statistics of the netdev attached to each
  RDMA device. In this cluster the netdevs are fabric0, fabric2, ..., fabric14, which
  belong to mlx5_0, mlx5_2, ..., mlx5_14 (the 8 data NICs; the odd ones are not used).
  "vport rdma unicast" counts only RDMA traffic, i.e. NIXL/UCX KV transfers and NCCL
  over IB verbs/RoCE; ordinary TCP traffic on eth0 is not included.

Why raw ioctl instead of `ethtool -S`
  The Pod has no ethtool binary and we do not want to install anything. SIOCETHTOOL is
  the same ioctl ethtool uses, so only the Python standard library is needed. The
  three requests used are
      ETHTOOL_GSSET_INFO (0x37)  -> how many statistics exist (n)
      ETHTOOL_GSTRINGS   (0x1b)  -> the names of the n statistics (32 bytes each)
      ETHTOOL_GSTATS     (0x1d)  -> the n 64-bit values
  Names are read once; later samples only issue GSTATS (cheap, ~sub-millisecond).

Output: one JSON line per sample
    {"t": unix time, "v": [[tx_bytes, rx_bytes], ...]}   # 8 entries, index i = mlx5_{2*i}
  Counters are cumulative: rate = (v[k+1] - v[k]) / (t[k+1] - t[k]). With a 20 ms
  interval a 131072-token KV transfer (9.2 GB, ~200 ms) shows up as ~10 samples at
  40-50 GB/s, which is how the on-wire time and peak rate were measured.
"""
import fcntl, socket, struct, array, sys, time, json

SIOCETHTOOL = 0x8946
S = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # any socket works as the ioctl handle


def _ioctl(dev, buf):
    # struct ifreq: 16-byte interface name + pointer to the ethtool command buffer
    fcntl.ioctl(S.fileno(), SIOCETHTOOL, struct.pack('16sP', dev.encode(), buf.buffer_info()[0]))


def _names(dev):
    """Return (number of statistics, list of their names) for one netdev."""
    # ETHTOOL_GSSET_INFO, sset_mask = 1 << ETH_SS_STATS
    b = array.array('B', struct.pack('IIQI', 0x37, 0, 1 << 1, 0) + b'\0' * 4); _ioctl(dev, b)
    n = struct.unpack_from('I', b.tobytes(), 16)[0]
    # ETHTOOL_GSTRINGS for string set ETH_SS_STATS (=1), ETH_GSTRING_LEN = 32
    b = array.array('B', struct.pack('III', 0x1b, 1, n) + b'\0' * (n * 32)); _ioctl(dev, b)
    raw = b.tobytes()[12:12 + n * 32]
    return n, [raw[i * 32:(i + 1) * 32].split(b'\0')[0].decode() for i in range(n)]


DEVS = [f'fabric{i}' for i in range(0, 16, 2)]          # netdev of mlx5_0, 2, ..., 14
N, NAMES = _names(DEVS[0])                                # all NICs have the same statistics layout
IDX = (NAMES.index('tx_vport_rdma_unicast_bytes'), NAMES.index('rx_vport_rdma_unicast_bytes'))
# One pre-built GSTATS buffer per device, reused for every sample
BUF = {d: array.array('B', struct.pack('II', 0x1d, N) + b'\0' * (N * 8)) for d in DEVS}


def sample():
    """One snapshot: [(tx_bytes, rx_bytes), ...] for the 8 NICs."""
    out = []
    for d in DEVS:
        _ioctl(d, BUF[d])
        v = struct.unpack_from('Q' * N, BUF[d].tobytes(), 8)   # N unsigned 64-bit values after the 8-byte header
        out.append((v[IDX[0]], v[IDX[1]]))
    return out


if __name__ == '__main__':
    dur, interval = float(sys.argv[1]), float(sys.argv[2])
    t_end = time.time() + dur
    while time.time() < t_end:
        t = time.time()
        print(json.dumps({'t': t, 'v': sample()}), flush=True)
        dt = interval - (time.time() - t)                       # keep a steady sampling period
        if dt > 0:
            time.sleep(dt)
