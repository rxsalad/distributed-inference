import json
import os
import time
import socket
from kubernetes import client, config

MASTER_APP  = os.getenv("MASTER_APP", "mc-master")
MASTER_PORT = int(os.getenv("MASTER_PORT", "50051"))
NAMESPACE   = os.getenv("POD_NAMESPACE", "default")

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "2"))
TIMEOUT = int(os.getenv("MASTER_TIMEOUT", "300"))

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
        if cond.type == "Ready" and cond.status == "True":
            return True
    return False

def get_master_pod_ip(v1):
    pods = v1.list_namespaced_pod(
        namespace=NAMESPACE,
        label_selector=f"app={MASTER_APP}",
    ).items

    if not pods:
        print(f"{MASTER_APP}: no pod found", flush=True)
        return None

    pod = pods[0]

    if pod.status.phase != "Running":
        print(
            f"{MASTER_APP}: pod={pod.metadata.name} "
            f"state={pod.status.phase}",
            flush=True,
        )
        return None

    if not is_pod_ready(pod):
        print(
            f"{MASTER_APP}: pod={pod.metadata.name} not Ready",
            flush=True,
        )
        return None

    if not pod.status.pod_ip:
        print(
            f"{MASTER_APP}: pod={pod.metadata.name} missing IP",
            flush=True,
        )
        return None

    return pod.status.pod_ip

def check_tcp_port(ip, port):
    """
    Check whether TCP port is accepting connections.
    """
    try:
        with socket.create_connection(
            (ip, port),
            timeout=3,
        ):
            return True
    except OSError:
        return False

def wait_for_master(v1):
    start = time.time()

    while True:
        if time.time() - start > TIMEOUT:
            raise TimeoutError(
                f"Timeout waiting for {MASTER_APP}:{MASTER_PORT}"
            )

        ip = get_master_pod_ip(v1)

        if ip:
            if check_tcp_port(ip, MASTER_PORT):
                print(
                    f"{MASTER_APP} is ready: {ip}:{MASTER_PORT}",
                    flush=True,
                )
                return ip, MASTER_PORT

            print(
                f"{MASTER_APP}: {ip}:{MASTER_PORT} not accepting connections",
                flush=True,
            )

        time.sleep(POLL_INTERVAL)



v1 = load_kube_client()
master_ip, master_port = wait_for_master(v1)
print(
        f"Mooncake master available at {master_ip}:{master_port}",
        flush=True,
)

config = {
    "mode": "embedded",
    "metadata_server": f"http://{master_ip}:8080/metadata",
    "master_server_address": f"{master_ip}:50051",
    "global_segment_size": "80GB" ,
    "local_buffer_size": "4GB",
    "protocol": "rdma",
    "device_name": "",
    "enable_offload": False
}

file_path = "mooncake_config.json"

with open(file_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Created {file_path}")

