"""Shared helpers of the analysis scripts (analyze_window.py, analyze_per_request.py).

Data layout written by exp_window.sh / exp_per_request.sh in the output directory O:
  bytes-<pod>.jsonl      RoCE byte-counter samples   (roce_bytes_sampler.py)
  nvl-<pod>-<tag>.json   NVLink counters             (nvlink_counters.py)
  rdma-<pod>-<tag>.txt   RDMA request counters       (rdma_request_counters.sh)
  metrics-<tag>.json     NIXL / latency metrics      (nixl_metrics.py)
  router*.txt            the Router's "Selected prefill=... decode=..." lines
  ipmap.json             {pod IP: pod name}, saved at the start of the run
"""
import glob, json, os, re

# One Router log line:  ... prefill=http://<ip>:<port>@<rank in node> ... decode=http://<ip>:<port>@<rank in node>
ROUTER_RE = re.compile(r'prefill=http://([\d.]+):\d+@(\d+).*decode=http://([\d.]+):\d+@(\d+)')


def pods_in(O):
    """Pods that have sampler data: Prefill pods first, then Decode pods, each sorted by index."""
    pods = [os.path.basename(f)[len('bytes-'):-len('.jsonl')] for f in glob.glob(f'{O}/bytes-*.jsonl')]
    return sorted(pods, key=lambda p: (0 if 'prefill' in p else 1, int(p.rsplit('-', 1)[1])))


def load_ipmap(O, override=None):
    """{pod IP: pod name}. Uses O/ipmap.json; `override` ("ip=pod,ip=pod") is for data
    recorded before ipmap.json existed."""
    if override:
        return dict(kv.split('=') for kv in override.split(','))
    f = f'{O}/ipmap.json'
    return json.load(open(f)) if os.path.exists(f) else {}


def global_rank(pod, r):
    """Global DP rank inside the pool: node index * 8 + rank in the node (8 DP ranks per node)."""
    return int(pod.rsplit('-', 1)[1]) * 8 + r


def parse_router_line(line, ipmap, pods=()):
    """-> (prefill pod, prefill global rank, decode pod, decode global rank) or None.
    An IP missing from ipmap is mapped to the only Prefill (Decode) pod of `pods` if there is
    exactly one (1P1D data recorded without ipmap.json); otherwise it stays as the 'pod name'."""
    m = ROUTER_RE.search(line)
    if not m:
        return None
    only = lambda role: [p for p in pods if role in p]
    pp = ipmap.get(m.group(1)) or (only('prefill')[0] if len(only('prefill')) == 1 else m.group(1))
    dp = ipmap.get(m.group(3)) or (only('decode')[0] if len(only('decode')) == 1 else m.group(3))
    pidx = lambda pod: int(pod.rsplit('-', 1)[1]) if pod.rsplit('-', 1)[-1].isdigit() else 0
    return pp, pidx(pp) * 8 + int(m.group(2)), dp, pidx(dp) * 8 + int(m.group(4))


def read_rdma(O, pod, tag):
    """{device: (write requests, read requests)} from rdma-<pod>-<tag>.txt."""
    d = {}
    for line in open(f'{O}/rdma-{pod}-{tag}.txt'):
        a = line.split()
        if len(a) == 3:
            d[a[0]] = (int(a[1]), int(a[2]))
    return d


def load_samples(O, pod, t0=None, t1=None):
    """Sampler rows [{'t':..., 'v': [[tx, rx] x 8]}], optionally cut to [t0-1, t1+3]."""
    rows = []
    for line in open(f'{O}/bytes-{pod}.jsonl'):
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass                                   # truncated last line when the sampler stops
    if t0 is not None:
        rows = [r for r in rows if t0 - 1 <= r['t'] <= t1 + 3]
    return rows


def bursts(rows, nic, col, thr=10.0):
    """KV bursts on one NIC: runs of consecutive sampler intervals whose rate exceeds `thr` GB/s.
    col 0 = tx bytes, col 1 = rx bytes. Returns [[t_start, t_end, bytes, peak GB/s], ...]."""
    out, cur = [], None
    for a, b in zip(rows, rows[1:]):
        dt = b['t'] - a['t']
        if dt <= 0:
            continue
        nbytes = b['v'][nic][col] - a['v'][nic][col]
        rate = nbytes / dt / 1e9
        if rate > thr:
            if cur is None:
                cur = [a['t'], b['t'], nbytes, rate]
            else:
                cur[1] = b['t']; cur[2] += nbytes; cur[3] = max(cur[3], rate)
        elif cur:
            out.append(cur); cur = None
    if cur:
        out.append(cur)
    return out


def delta(ma, mb, pod, name):
    """Per-engine increase of one cumulative metric between two nixl_metrics.py snapshots."""
    x, y = ma[pod].get(name, {}), mb[pod].get(name, {})
    return {e: x[e] - y.get(e, 0) for e in x if x[e] - y.get(e, 0) != 0}
