#!/usr/bin/env python3

import os
import time
import requests
from kubernetes import client, config

PREFILL_APP = os.getenv("PREFILL_APP", "mc-pdd-prefill")
PREFILL_PORT = int(os.getenv("PREFILL_PORT", "8010"))
DECODE_APP = os.getenv("DECODE_APP", "mc-pdd-decode")
DECODE_PORT = int(os.getenv("DECODE_PORT", "8020"))
ROUTER_PORT = int(os.getenv("ROUTER_PORT", "8000"))
NAMESPACE = os.getenv("POD_NAMESPACE", "default")

POLL_INTERVAL = 2

def load_kube_client():
    """
    Load in-cluster Kubernetes config.
    """
    config.load_incluster_config()
    return client.CoreV1Api()

def is_pod_ready(pod):
    if not pod.status.conditions:
        return False
    for cond in pod.status.conditions:
        if (
            cond.type == "Ready"
            and cond.status == "True"
        ):
            return True
    return False

def get_all_pods(v1, app):
    pods = v1.list_namespaced_pod(
        namespace=NAMESPACE,
        label_selector=f"app={app}",
    )
    return pods.items

def get_ready_pod_ips(v1, app):
    pods = get_all_pods(v1, app)
    if not pods:
        print(f"{app}: no pods found",flush=True,)
        return None

    ready_ips = []
    for pod in pods:
        name = pod.metadata.name
        if pod.status.phase != "Running":
            print(f"{app}: {name} state={pod.status.phase}",flush=True,)
            return None
        if not is_pod_ready(pod):
            print(f"{app}: {name} not Ready",flush=True,)
            return None
        if not pod.status.pod_ip:
            print(f"{app}: {name} missing pod IP",flush=True,)
            return None
        ready_ips.append(pod.status.pod_ip)

    return ready_ips

def check_health(ip, port):
    url = f"http://{ip}:{port}/health"

    try:
        r = requests.get(url,timeout=3,)
        if r.status_code == 200:
            return True
        print(f"Health failed {url}: HTTP {r.status_code}",flush=True,)

    except Exception as e:
        #print(f"Health failed {url}: {e}",flush=True,)
        pass

    return False


def wait_for_pods_and_health(v1, app, port):

    while True:
        print(f"Checking {app}...",flush=True,)
        pod_ips = get_ready_pod_ips(v1,app,)
        if pod_ips is None:
            print(f"{app}: waiting for ALL pods Ready",flush=True,)
            time.sleep(POLL_INTERVAL)
            continue

        all_healthy = True
        for ip in pod_ips:
            if not check_health(ip,port,):
                all_healthy = False
                break

        if all_healthy:
            print(f"{app}: {len(pod_ips)} pods ready and healthy",flush=True,)
            return pod_ips

        print(f"{app}: waiting for all health checks",flush=True,)
        time.sleep(POLL_INTERVAL)

def build_router_args(prefill_ips, decode_ips):

    args = []
    for ip in prefill_ips:
        args.extend(["--prefill",f"http://{ip}:{PREFILL_PORT}",])

    for ip in decode_ips:
        args.extend(["--decode",f"http://{ip}:{DECODE_PORT}",])

    return args



def main():

    v1 = load_kube_client()
    print("Waiting for prefill workers...",flush=True,)
    prefill = wait_for_pods_and_health(v1,PREFILL_APP,PREFILL_PORT,)

    print("Waiting for decode workers...",flush=True,)
    decode = wait_for_pods_and_health(v1,DECODE_APP,DECODE_PORT,)

    args = build_router_args(prefill,decode,)

    print("Prefill backends:")
    for ip in prefill:
        print(f"  http://{ip}:{PREFILL_PORT}")

    print("Decode backends:")
    for ip in decode:
        print(f"  http://{ip}:{DECODE_PORT}")

    cmd = [
        "python",
        "/app/vllm/examples/disaggregated/mooncake_connector/"
        "mooncake_connector_proxy.py",
    ]
    cmd.extend(args)

    cmd.extend(["--host","0.0.0.0","--port",str(ROUTER_PORT),])
    print("Starting router:"," ".join(cmd),flush=True,)

    os.execvp(cmd[0],cmd,)


if __name__ == "__main__":
    main()