#!/usr/bin/env python3
"""
Router launcher for PD disaggregation (runs in a CPU Pod, mounted via ConfigMap pdd-scripts).

It does three things:
1. Read the replicas of the Prefill / Decode StatefulSets from the K8S API as the expected node counts;
2. Wait until all Prefill and Decode Pods are Running + Ready + have an IP and /health returns 200;
3. Start vllm-router (PD disaggregation mode) once everything is ready, then keep monitoring: when the set of Pod IPs changes (e.g. a Pod restart
   changes its IP) restart the router; when a Pod is temporarily not Ready only log it, since the router has its own health checks and circuit breaking.
"""
import os
import signal
import subprocess
import sys
import time

import requests
from kubernetes import client, config

NAMESPACE = os.getenv("POD_NAMESPACE", "default")

PREFILL_APP = os.getenv("PREFILL_APP", "pdd-prefill")
PREFILL_PORT = int(os.getenv("PREFILL_PORT", "8001"))
DECODE_APP = os.getenv("DECODE_APP", "pdd-decode")
DECODE_PORT = int(os.getenv("DECODE_PORT", "8002"))

ROUTER_HOST = os.getenv("ROUTER_HOST", "0.0.0.0")
ROUTER_PORT = int(os.getenv("ROUTER_PORT", "30000"))
ROUTER_POLICY = os.getenv("ROUTER_POLICY", "round_robin")
INTRA_NODE_DP = os.getenv("INTRA_NODE_DP", "8")  # must equal --data-parallel-size-local of every vllm serve

POLL_INTERVAL = 5       # polling interval while waiting for readiness (seconds)
# interval (seconds) for checking Pod IPs and /health while the router runs; larger values reduce /health access logs in vLLM
WATCH_INTERVAL = int(os.getenv("WATCH_INTERVAL", "60"))
EXIT_POLL_INTERVAL = 5  # interval (seconds) for checking whether the router subprocess exited; sends no requests
LOG_EVERY = 30          # while waiting, print the same status at most every 30 seconds


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def expected_replicas(apps, name):
    return apps.read_namespaced_stateful_set(name, NAMESPACE).spec.replicas


def pod_ready(pod):
    return any(c.type == "Ready" and c.status == "True" for c in (pod.status.conditions or []))


def health_ok(ip, port):
    try:
        return requests.get(f"http://{ip}:{port}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def snapshot(v1, apps, app, port):
    """Return (ips, status text). When everything is ready ips is the list of IPs sorted by Pod name, otherwise None."""
    want = expected_replicas(apps, app)
    pods = v1.list_namespaced_pod(NAMESPACE, label_selector=f"app={app}").items
    pods = sorted(pods, key=lambda p: p.metadata.name)
    if len(pods) < want:
        return None, f"{app}: {len(pods)}/{want} pods exist"
    ips, bad = [], []
    for p in pods:
        name = p.metadata.name
        if p.status.phase != "Running" or not pod_ready(p) or not p.status.pod_ip:
            bad.append(f"{name}({p.status.phase}{'' if pod_ready(p) else ',not Ready'})")
        elif not health_ok(p.status.pod_ip, port):
            bad.append(f"{name}(/health failed)")
        else:
            ips.append(p.status.pod_ip)
    if bad:
        return None, f"{app}: {len(ips)}/{want} ready, waiting for {', '.join(bad)}"
    return ips, f"{app}: {want}/{want} ready"


def snapshot_all(v1, apps):
    p_ips, p_msg = snapshot(v1, apps, PREFILL_APP, PREFILL_PORT)
    d_ips, d_msg = snapshot(v1, apps, DECODE_APP, DECODE_PORT)
    msg = f"{p_msg} | {d_msg}"
    if p_ips is None or d_ips is None:
        return None, msg
    return (tuple(p_ips), tuple(d_ips)), msg


def wait_all_ready(v1, apps):
    last_msg, last_t = None, 0.0
    while True:
        snap, msg = snapshot_all(v1, apps)
        if snap:
            log(msg)
            return snap
        if msg != last_msg or time.time() - last_t > LOG_EVERY:
            log(f"waiting: {msg}")
            last_msg, last_t = msg, time.time()
        time.sleep(POLL_INTERVAL)


def build_cmd(snap):
    p_ips, d_ips = snap
    cmd = [
        "vllm-router",
        "--policy", ROUTER_POLICY,
        "--vllm-pd-disaggregation",
        "--host", ROUTER_HOST,
        "--port", str(ROUTER_PORT),
        "--intra-node-data-parallel-size", INTRA_NODE_DP,
    ]
    for ip in p_ips:
        cmd += ["--prefill", f"http://{ip}:{PREFILL_PORT}"]
    for ip in d_ips:
        cmd += ["--decode", f"http://{ip}:{DECODE_PORT}"]
    return cmd


def stop(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    config.load_incluster_config()
    v1, apps = client.CoreV1Api(), client.AppsV1Api()

    proc = None

    def on_term(signum, _frame):
        stop(proc)
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    while True:
        log("waiting for all Prefill and Decode pods ...")
        snap = wait_all_ready(v1, apps)
        cmd = build_cmd(snap)
        log("starting router: " + " ".join(cmd))
        proc = subprocess.Popen(cmd)

        degraded = False
        last_check = time.time()
        while True:
            time.sleep(EXIT_POLL_INTERVAL)
            if proc.poll() is not None:
                log(f"router exited with code {proc.returncode}")
                sys.exit(proc.returncode or 1)   # exit so K8S restarts the Pod
            if time.time() - last_check < WATCH_INTERVAL:
                continue                          # not yet time to check Pod IPs / /health
            last_check = time.time()
            try:
                cur, msg = snapshot_all(v1, apps)
            except Exception as e:               # an occasional API failure must not affect the serving router
                log(f"snapshot failed: {e}")
                continue
            if cur is None:
                if not degraded:
                    log(f"degraded (router keeps running): {msg}")
                    degraded = True
            elif cur != snap:
                log(f"backend set changed: {snap} -> {cur}, restarting router")
                stop(proc)
                break
            elif degraded:
                log("all pods ready again")
                degraded = False


if __name__ == "__main__":
    main()
