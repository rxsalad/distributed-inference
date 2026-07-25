# Prefill–Decode Disaggregation with vLLM and NixlConnector on DOKS (Notes)

Follow [amd-mn-k8s-environment](https://github.com/rxsalad/multi-node-solutions/blob/main/amd-mn-k8s-environment.MD) and [amd-mn-k8s-rdma-sharing.MD](https://github.com/rxsalad/multi-node-solutions/blob/main/amd-mn-k8s-rdma-sharing.MD) to prepare the environment before proceeding. The setup requires two DOKS workers with 16 GPUs and RDMA/RoCE support. Ensure that Multus CNI, ipvlan CNI, and Whereabouts IPAM CNI are installed and configured.

For a general introduction to Prefill–Decode (PD) Disaggregation and the vLLM MooncakeConnector solution, refer to [mc-pdd-doks-amd.MD](mc-pdd-doks-amd.MD).

This document describes the step-by-step process for deploying PD Disaggregation using vLLM and NixlConnector on DOKS, and provides an introduction to the underlying concepts, architecture, and components.


## Concepts

### UCX

UCX (Unified Communication X) is an open-source, high-performance communication framework designed for HPC and distributed AI/ML workloads. It provides a unified communication layer for efficient data movement across CPUs, GPUs, and compute nodes while supporting a broad range of networking technologies and memory types, including InfiniBand, RoCE, TCP, shared memory, and GPU memory.

UCX abstracts underlying communication transports and automatically selects optimal communication paths based on hardware capabilities, memory types, and system topology. It can leverage high-speed intra-node GPU interconnects such as NVLink and XGMI, inter-node technologies such as RDMA and GPUDirect RDMA, and shared memory for local communication, enabling low-latency, high-bandwidth transfers without requiring hardware-specific application logic.

Designed for heterogeneous, multi-vendor environments, UCX supports NVIDIA GPUs through CUDA and GPUDirect RDMA, AMD GPUs through ROCm/HIP, and various network technologies through InfiniBand, RoCE, and TCP. It provides a consistent interface while transparently managing transport selection, endpoint management, memory registration, and hardware-optimized communication. Libraries such as NIXL and frameworks such as Open MPI leverage UCX to accelerate data movement across modern HPC and AI/ML systems.

#### Documentation and Resources 

- [UCX Documentation](https://openucx.org/documentation/)

- [UCX GitHub Repo](https://github.com/openucx/ucx)

- [UCX Network Capabilities](https://github.com/openucx/ucx/blob/master/docs/source/faq.md#network-capabilities)

- [UCX Programming Guide](https://docs.nvidia.com/doca/archive/doca-v2.2.1/ucx-programming-guide/index.html)

- [UCX Configuration](https://ucx-py.readthedocs.io/en/latest/configuration.html)

- [OpenMPI Network Support](https://docs.open-mpi.org/en/v5.0.x/release-notes/networks.html#network-support)

### NIXL

NIXL (NVIDIA Inference Xfer Library) is a high-performance data-transfer framework designed for distributed and disaggregated AI inference workloads. It provides a unified abstraction for moving inference data, such as KV cache, across GPU memory, CPU memory, storage, and remote nodes.

NIXL separates data-transfer semantics from underlying communication mechanisms through a pluggable backend architecture. It supports various transport technologies, including RDMA, shared memory, and GPU-aware communication frameworks. In vLLM, NIXL enables efficient KV cache transfer between disaggregated prefill and decode instances by managing buffer registration, metadata exchange, and transfer orchestration, while communication layers such as UCX provide high-performance data movement over transports such as RoCE and InfiniBand.

When UCX is used as the underlying communication layer, its behavior and transport selection can be controlled through UCX environment variables. For example, variables such as UCX_TLS, UCX_NET_DEVICES, and other UCX configuration options can be used to select transports, specify network interfaces, and tune GPU-aware RDMA behavior. 

RIXL is the AMD ROCm-based implementation of NIXL. It preserves the NIXL API and transfer framework while adding ROCm/HIP GPU memory support and integration with ROCm-enabled UCX transports, for efficient data transfer between AMD GPU memory and RDMA networks. AMD GPU support developed through RIXL has been contributed upstream to the main NIXL project.

#### Documentation and Resources 

- [NIXL GitHub Repo](https://github.com/ai-dynamo/nixl)

- [RIXL GitHub Rep (DEPRECATED))](https://github.com/rocm/rixl)

- [Anyscale PDD using RIXL ](https://www.anyscale.com/blog/ray-vllm-prefill-decode-disaggregation-amd-mi325x-67-percent-savings)

### NIXL, NCCL and MTE

NCCL (NVIDIA Collective Communications Library) and RCCL (ROCm Communication Collectives Library) are GPU-optimized communication libraries for high-performance GPU-to-GPU communication. They provide collective operations such as AllReduce, AllGather, ReduceScatter, and Broadcast, as well as point-to-point communication. By leveraging hardware topology, including NVLink/NVSwitch, PCIe, high-speed networking, and AMD GPU interconnects, NCCL and RCCL maximize bandwidth and minimize latency for distributed AI training and inference workloads.

NIXL is a high-performance data-transfer framework designed for large-scale, disaggregated AI inference workloads. Unlike NCCL/RCCL, which focus on GPU communication primitives, NIXL provides a unified abstraction for moving inference data such as KV cache and tensors across GPU memory, CPU memory, storage, and remote nodes. Its pluggable backend architecture allows different transport technologies to be used across diverse hardware environments.

NCCL/RCCL and NIXL serve complementary roles in distributed AI systems. NCCL/RCCL handle GPU communication within compute workloads, such as tensor parallel synchronization and collective operations, while NIXL enables data movement across distributed inference components, such as Prefill–Decode disaggregation and KV cache transfer.

Mooncake MTE (Mooncake Transfer Engine) is the data-transfer engine used in the Mooncake distributed serving architecture. It is optimized for large-scale LLM inference data movement, especially KV cache transfer across heterogeneous memory and storage resources. MTE provides a unified transfer interface and supports transports such as TCP, RDMA, CXL/shared memory, and NVMe-oF to enable high-throughput, low-overhead, and zero-copy data movement.

NIXL and Mooncake MTE address similar data-movement challenges but come from different inference ecosystems. NIXL provides a general-purpose inference data-transfer abstraction, while MTE is the optimized transfer engine within the Mooncake serving architecture. They can also be integrated through the NIXL Mooncake backend, where NIXL provides the common transfer interface and semantics, while MTE performs the underlying high-performance data movement. The Mooncake backend is currently in Preview.


#### Documentation and Resources 

- [NIXL - Mooncake Backend Plugin](https://github.com/ai-dynamo/nixl/blob/main/src/plugins/mooncake/README.md)

- [NIXL - Python Examples](https://github.com/ai-dynamo/nixl/tree/main/examples/python#python-examples)

- [Mooncake Transfer Engine](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html)

- [Everything You Want to Know about KV Cache Transfer Engine](https://uccl-project.github.io/posts/kv-transfer-engine/)


## NIXL Examples

### Images

The recent ROCm/vLLM images already includes the RIXL Python package, and the required UCX runtime components for AMD ROCm GPUs. Note that the image includes the UCX runtime libraries and ROCm transport plugins, but not the UCX command-line tools or performance utilities.

``` bash
# vllm/vllm-openai-rocm:v0.21.0

# pip list | grep -i rixl
# pip show rixl
Name: rixl
Version: 1.1.0
Summary: RIXL Python API
Home-page: 
Author: 
Author-email: NIXL Developers <nixl-developers@nvidia.com>
License-Expression: MIT AND Apache-2.0
Location: /usr/local/lib/python3.12/dist-packages
Requires: numpy, torch

# The Python-facing RIXL API, along with the compiled native extension modules used by RIXL.
# tree /usr/local/lib/python3.12/dist-packages/rixl
/usr/local/lib/python3.12/dist-packages/rixl
|-- __init__.py
|-- __pycache__
|   |-- __init__.cpython-312.pyc
|   |-- _api.cpython-312.pyc
|   `-- logging.cpython-312.pyc
|-- _api.py
|-- _bindings.cpython-312-x86_64-linux-gnu.so
|-- _utils.cpython-312-x86_64-linux-gnu.so
|-- logging.py
`-- py.typed

# RIXL packaged native dependencies and transport components
# tree /usr/local/lib/python3.12/dist-packages/rixl.libs
/usr/local/lib/python3.12/dist-packages/rixl.libs
|-- libaio-9230aa3b.so.1.0.1
|-- libucm-d3d5eeef.so.0.0.0
|-- libucp-68771e91.so.0.0.0
|-- libucs-73927279.so.0.0.0
|-- libuct-42edfc9d.so.0.0.0
|-- nixl
|   |-- libplugin_POSIX.so
|   |-- libplugin_UCX.so
|   `-- libtelemetry_exporter_prometheus.so
`-- ucx
    |-- libucm_rocm.la
    |-- libucm_rocm.so
    |-- libucm_rocm.so.0
    |-- libucm_rocm.so.0.0.0
    |-- libuct_cma.la
    |-- libuct_cma.so
    |-- libuct_cma.so.0
    |-- libuct_cma.so.0.0.0
    |-- libuct_ib.la
    |-- libuct_ib.so
    |-- libuct_ib.so.0
    |-- libuct_ib.so.0.0.0
    |-- libuct_ib_efa.la
    |-- libuct_ib_efa.so
    |-- libuct_ib_efa.so.0
    |-- libuct_ib_efa.so.0.0.0
    |-- libuct_ib_mlx5.la
    |-- libuct_ib_mlx5.so
    |-- libuct_ib_mlx5.so.0
    |-- libuct_ib_mlx5.so.0.0.0
    |-- libuct_rdmacm.la
    |-- libuct_rdmacm.so
    |-- libuct_rdmacm.so.0
    |-- libuct_rdmacm.so.0.0.0
    |-- libuct_rocm.la
    |-- libuct_rocm.so
    |-- libuct_rocm.so.0
    |-- libuct_rocm.so.0.0.0
    |-- libucx_perftest_rocm.la
    |-- libucx_perftest_rocm.so
    |-- libucx_perftest_rocm.so.0
    `-- libucx_perftest_rocm.so.0.0.0
```

The software stack in the ROCm/vLLM images can be viewed as:
```
                    Python Application
                           │
                           ▼
                    RIXL Python API
                           │
                           ▼
                 RIXL Native Bindings
                           │
                           ▼
                         NIXL
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       POSIX Plugin              UCX Plugin
                                        │
                                        ▼
                              UCX Runtime Libraries
                                        │
                     ┌──────────────────┼──────────────────┐
                     │                  │                  │
                     ▼                  ▼                  ▼
                ROCm GPU          InfiniBand/RDMA       Other
                Transport             Transport        Transports
                     │                  │
                     ▼                  ▼
                AMD ROCm GPUs       NIC / RDMA Fabric
```

#### Documentation and Resources 

- [ROCm vLLM Dockerfile](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm)

- [ ROCm vLLM Image](https://hub.docker.com/r/vllm/vllm-openai-rocm/tags)

### Tests

Run [nixl/basic_two_peers.py](nixl/basic_two_peers.py), which demonstrates a two-party tensor transfer workflow, including registration, metadata exchange, READ operation, notification, verification, and teardown.

``` bash
# Initiator Pod: 10.119.1.1
# python basic_two_peers.py --ip 10.119.0.241 --mode initiator
2026-07-24 17:40:22 NIXL INFO    _api.py:369 Backend UCX was instantiated
2026-07-24 17:40:22 NIXL INFO    _api.py:247 Initialized NIXL agent: initiator
2026-07-24 17:40:22 NIXL INFO    basic_two_peers.py:55 Running test with tensor shape (10, 16) in mode initiator
2026-07-24 17:40:22 NIXL INFO    basic_two_peers.py:92 Initiator sending to 10.119.0.241
2026-07-24 17:40:22 NIXL INFO    basic_two_peers.py:113 Ready for transfer
2026-07-24 17:40:22 NIXL INFO    basic_two_peers.py:136 initiator Data verification passed
2026-07-24 17:40:22 NIXL INFO    basic_two_peers.py:146 Test Complete.

# Target Pod: 10.119.0.241 
# python basic_two_peers.py --mode target
2026-07-24 17:40:19 NIXL INFO    _api.py:369 Backend UCX was instantiated
2026-07-24 17:40:19 NIXL INFO    _api.py:247 Initialized NIXL agent: target
2026-07-24 17:40:19 NIXL INFO    basic_two_peers.py:55 Running test with tensor shape (10, 16) in mode target
2026-07-24 17:40:22 NIXL INFO    basic_two_peers.py:82 Waiting for transfer
2026-07-24 17:40:22 NIXL INFO    basic_two_peers.py:146 Test Complete.
```

## vLLM NixlConnector

NixlConnector is a high-performance KV-cache transfer connector for vLLM's disaggregated prefilling architecture. It provides fully asynchronous send/receive operations and uses the NIXL as the underlying data-transfer framework to efficiently move KV cache across processes and nodes.

NIXL provides a pluggable transport-backend architecture and can use different communication mechanisms depending on the deployment environment. UCX is the primary transport backend used by NIXL for high-performance communication, including GPU memory and RDMA-based data transfers.

When NixlConnector uses the NIXL UCX backend, the underlying UCX communication behavior can be configured through UCX environment variables. These variables control aspects such as transport selection, network device selection, and logging.

### Example Walkthrough 

``` bash
# kubectl get pods -o wide
NAME            READY   STATUS    RESTARTS   AGE    IP             NODE                          NOMINATED NODE   READINESS GATES
pdd-decode-0    1/1     Running   0          8m3s   10.119.1.81    rs-mn-mi325x8-pool-3cmiju     <none>           <none>
pdd-prefill-0   1/1     Running   0          8m3s   10.119.0.240   rs-mn-mi325x8-pool-3c35jt     <none>           <none>
pdd-router-0    1/1     Running   0          8m3s   10.119.0.36    rs-mn-cpu-power-pool-37f7cs   <none>           <none>

# Related UCX environment variables
UCX_TLS=all
UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
UCX_NET_DEVICES=all 
UCX_MAX_RNDV_RAILS=4      # 2 by default, and up to 4
UCX_IB_TRAFFIC_CLASS=104  # DSCP prioirity
UCX_LOG_LEVEL=info        

# Related vLLM NIXL environment variables
VLLM_NIXL_SIDE_CHANNEL_PORT=5600         # NIXL handshaking between remote agents
VLLM_NIXL_SIDE_CHANNEL_HOST=10.119.0.240 # NIXL handshaking between remote agents

# Prefill GPU Pod (8 GPUs with 8 rails)

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
UCX_NET_DEVICES=all \
UCX_LOG_LEVEL=info \
UCX_MAX_RNDV_RAILS=4 \
UCX_IB_TRAFFIC_CLASS=104 \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
VLLM_NIXL_SIDE_CHANNEL_HOST=10.119.0.240 \
vllm serve Qwen/Qwen3-8B \
  --no-enable-prefix-caching \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 \
  --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'


# Decode Pod (8 GPUs with 8 rails)

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
UCX_NET_DEVICES=all \
UCX_LOG_LEVEL=info \
UCX_MAX_RNDV_RAILS=4 \
UCX_IB_TRAFFIC_CLASS=104 \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
VLLM_NIXL_SIDE_CHANNEL_HOST=10.119.1.81 \
vllm serve Qwen/Qwen3-8B \
  --no-enable-prefix-caching \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 \
  --port 8200 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'


# Proxy CPU Pod 
git clone https://github.com/vllm-project/vllm.git /myvllm

python3 /myvllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --port 8000 \
  --prefiller-hosts 10.119.0.240 \
  --prefiller-ports 8100 \
  --decoder-hosts 10.119.1.81 \
  --decoder-ports 8200

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "How to learn AI/ML, 100 words."}
    ]
  }'

vllm bench serve --port 8000 --seed "1" \
  --backend vllm --model "Qwen/Qwen3-8B"  \
  --dataset-name random --random-input-len 32784 --random-output-len 128 \
  --num-prompts 1 --burstiness 1 --request-rate 1 | tee benchmark.log
```

#### Test Results

```
# Client running on Proxy Pod 
# vllm bench serve --port 8000 --seed "1" \
  --backend vllm --model "Qwen/Qwen3-8B"  \
  --dataset-name random --random-input-len 32784 --random-output-len 128 \
  --num-prompts 1 --burstiness 1 --request-rate 1 | tee benchmark.log

============ Serving Benchmark Result ============
Successful requests:                     1         
Failed requests:                         0         
Request rate configured (RPS):           1.00      
Benchmark duration (s):                  2.50      
Total input tokens:                      32784     
Total generated tokens:                  128       
Request throughput (req/s):              0.40      
Output token throughput (tok/s):         51.10     
Peak output token throughput (tok/s):    124.00    
Peak concurrent requests:                1.00      
Total token throughput (tok/s):          13138.72  
---------------Time to First Token----------------
Mean TTFT (ms):                          1125.81   
Median TTFT (ms):                        1125.81   
P99 TTFT (ms):                           1125.81   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2.97      
Median TPOT (ms):                        2.97      
P99 TPOT (ms):                           2.97      
---------------Inter-token Latency----------------
Mean ITL (ms):                           3.06      
Median ITL (ms):                         3.10      
P99 ITL (ms):                            3.61      
==================================================

# Proxy Pod

INFO:     127.0.0.1:36236 - "POST /v1/completions HTTP/1.1" 200 OK

# Prefill Pod 

(APIServer pid=652) INFO:     10.119.0.36:51648 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=652) INFO 07-24 22:21:03 [loggers.py:271] Engine 000: Avg prompt throughput: 3278.3 tokens/s, Avg generation throughput: 0.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%, External prefix cache hit rate: 0.0%

# Decode Pod 

(APIServer pid=652) INFO:     10.119.0.36:46668 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=652) INFO 07-24 22:21:18 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%, External prefix cache hit rate: 100.0%

(APIServer pid=652) INFO 07-24 22:21:08 [metrics.py:103] KV Transfer metrics: Num successful transfers=8, Avg xfer time (ms)=49.604, P90 xfer time (ms)=88.413, Avg post time (ms)=0.252, P90 post time (ms)=0.314, Avg MB per transfer=576.281, Throughput (MB/s)=11617.52, Avg number of descriptors=72.0

# The KV cache is sharded across the 8 TP GPUs, with each GPU transferring a 576 MiB KV shard to its peer Decode GPU. All eight GPU-to-GPU transfers execute concurrently.
```

### HND and NHD

HND (Head-Number-Dimension) and NHD (Number-Head-Dimension) are two common memory layouts used to store the Key (K) and Value (V) tensors in a transformer's KV cache. Both layouts contain exactly the same KV data—the only difference is the order in which the dimensions are arranged in memory. This difference does not affect the model's computation or output, but it does influence GPU memory access patterns, attention kernel efficiency, and KV cache transfer performance.

In the HND layout, the KV cache is organized as [NumHeads, SeqLen, HeadDim]. All tokens belonging to the same attention (or KV) head are stored contiguously, making the complete history for each head sequential in memory. This organization is well suited for attention kernels that process one head at a time, allowing efficient streaming through the KV history of a single head.

```
HND: [NumHeads, SeqLen, HeadDim]

Head 0:
    Token 0
    Token 1
    ...
    Token N

Head 1:
    Token 0
    Token 1
    ...
```

In the NHD layout, the KV cache is organized as [SeqLen, NumHeads, HeadDim]. All attention heads corresponding to the same token are stored together. As new tokens are generated during autoregressive decoding, the KV data for all heads of the new token can be appended as a single contiguous block, which is advantageous for token-oriented memory operations.

```
NHD: [SeqLen, NumHeads, HeadDim]

Token 0:
    Head 0
    Head 1
    ...
    Head H

Token 1:
    Head 0
    Head 1
    ...
```

Neither layout is inherently better than the other. The preferred layout depends on the implementation of the attention kernel and the target hardware. Some optimized kernels achieve higher performance with HND because they operate on one head at a time, while others prefer NHD because they process data on a per-token basis. Consequently, inference frameworks such as vLLM support both layouts and can perform a layout permutation when the producer and consumer of the KV cache use different formats. This flexibility allows each stage of the inference pipeline to use the memory layout that delivers the best performance without changing the underlying KV data.

For Qwen3-8B with TP=8, HND and NHD should have very similar performance because each TP rank owns only one KV head, making the layout difference largely negligible.

If the prefill and decode workers use different KV-cache layouts, enable [enable_permute_local_kv](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/#heterogeneous-kv-layout-support) to perform the required local KV-cache layout conversion.

```
HND: [1, SeqLen, HeadDim]

or

NHD: [SeqLen, 1, HeadDim]
```

### Num of Descriptors

More descriptors generally degrade KV transfer performance because each descriptor represents an additional scatter-gather memory segment that NIXL must manage during data movement.

Descriptor count is mainly determined by the number of physically separate memory regions in the KV cache after allocation and by NIXL's ability to merge adjacent regions. The ideal count is driven by the logical KV tensor layout (72 descriptors in this Qwen3-8B TP=8 case: 36 layers × 2 tensors for K and V). A low descriptor count indicates that the KV cache is stored in large, contiguous memory regions, allowing efficient registration and transfer.

When descriptor count becomes very high, several performance impacts can occur:

- Higher descriptor management overhead: NIXL needs to process, register, and track more scatter-gather entries.
- More communication setup overhead: Each segment adds metadata that must be handled before data transfer begins.
- Reduced transfer efficiency: Many small memory regions can prevent the system from reaching peak bandwidth.
- Higher CPU/runtime overhead: Building and handling large descriptor lists consumes additional resources.
- Potential transfer failures: Extremely large descriptor counts may exceed backend limits or registration resources.

For example:

```
# Optimal case

Descriptor count = 72
KV size = 576 MB
Average region ≈ 8 MB per descriptor
This represents large contiguous KV regions and efficient transfer.

# Fragmented case
Descriptor count = 147,528
KV size = 576 MB
Average region ≈ 4 KB per descriptor
This indicates severe fragmentation, where the same amount of KV data is represented by thousands of small memory segments.
```

Therefore, descriptor count is an important indicator of KV cache memory health: lower descriptor counts generally mean better memory locality, lower overhead, and higher KV transfer performance. Extremely high descriptor counts usually indicate fragmentation or inefficient KV layout.

On attention backends that support cross-layer blocks, each logical KV-cache block is stored contiguously in physical memory. Enabling [enable_cross_layers_blocks](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/#cross-layers-blocks) groups blocks across layers into larger contiguous regions, reducing the number of buffers that need to be transferred and, consequently, the associated transfer fragmentation and per-buffer overhead.

### UCX_MAX_RNDV_RAILS

NIXL uses UCX as one of its communication backends. When transferring large objects such as a 576 MiB KV cache, UCX typically selects the rendezvous protocol, in which NIXL registers the KV cache memory and exchanges the necessary metadata, while UCX performs the rendezvous handshake and transfers the data directly using RDMA. For large transfers, UCX stripes the data into multiple chunks and sends them concurrently across multiple RDMA rails to maximize aggregate bandwidth.

[UCX_MAX_RNDV_RAILS](https://openucx.readthedocs.io/en/master/faq.html#is-it-possible-to-use-more-than-2-rails) is a per-process setting that specifies the maximum number of RDMA rails UCX can use for a single rendezvous transfer. The default is 2 rails, and the current maximum supported value is 4.

For TP=8, there are typically eight vLLM worker processes—one per GPU (one TP rank per process). UCX_MAX_RNDV_RAILS is applied independently to each worker, not globally across the entire vLLM deployment. As a result, while an individual worker is limited to the configured number of RDMA rails (currently up to four), the eight workers collectively can utilize all available NICs in the system. The actual rail utilization depends on factors such as GPU-to-NIC affinity, PCIe topology, NUMA locality, and UCX device selection.

### Rail-Only Fabrics

UCX establishes RDMA communication by discovering available RDMA devices, exchanging endpoint information between nodes, and selecting only physically reachable communication paths. During initialization, UCX identifies available HCAs (for example, mlx5_0 through mlx5_7) and gathers device information such as GIDs, ports, and transport capabilities.

In a rail-only RDMA topology, the physical network wiring determines the valid NIC-to-NIC communication paths. UCX uses device information such as NIC identity, GID, network address, and transport attributes to identify reachable connections. For example, with eight independent RDMA fabrics, mlx5_0 on Node A can communicate only with mlx5_0 on Node B, mlx5_1 with mlx5_1, and so on. Although there are 64 possible NIC combinations (8 × 8), UCX discovers only the eight reachable rail pairs and creates communication resources only for those valid paths.

For each selected rail, UCX creates endpoints and RDMA Queue Pairs (QPs) bound to the corresponding NICs. For large transfers such as vLLM KV cache movement, UCX can split data across multiple QPs and rails to aggregate bandwidth.

NIXL uses these UCX communication paths rather than creating QPs directly. NIXL manages memory registration, KV cache descriptors, and transfer scheduling, while UCX handles RDMA device discovery, rail selection, QP creation, and data movement. This allows PD disaggregation workloads to efficiently use only the connected RDMA rails in the underlying fabric.

### TP Mismatch

A general issue in disaggregated inference is the TP mismatch between prefill and decode instances. When they use different TP sizes, the KV cache is partitioned differently across GPUs, requiring additional redistribution and remapping during transfer. This can increase transfer overhead, network traffic, and complexity, potentially impacting overall performance.

### Documentation and Resources 

- [vLLM Disaggregated Prefilling](https://docs.vllm.ai/en/stable/features/disagg_prefill/#why-disaggregated-prefilling)

- [vLLM NixlConnector Usage Guide](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/)

- [vLLM Environment Varaibles](https://docs.vllm.ai/en/stable/configuration/env_vars/)

- [UCX Environment Varaibles](https://github.com/openucx/ucx/blob/master/docs/source/faq.md)

- [UCX Configuration](https://ucx-py.readthedocs.io/en/latest/configuration.html)

- [vLLM KV Transfer Metrics](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/#metrics-reference)



## Integration

``` bash
kubectl create configmap rs-router-script --from-file=nixl/nixl-pdd-router.py
kubectl describe configmap rs-router-script
kubectl delete configmap rs-router-script

kubectl apply -f nixl/nixl-pdd-1P3D2R-tp4-8rails.yaml
kubectl delete -f nixl/nixl-pdd-1P3D2R-tp4-8rails.yaml

# kubectl get pods -o wide
NAME            READY   STATUS    RESTARTS   AGE   IP             NODE                          NOMINATED NODE   READINESS GATES
pdd-decode-0    1/1     Running   0          85m   10.119.0.241   rs-mn-mi325x8-pool-3c35jt     <none>           <none>
pdd-decode-1    1/1     Running   0          85m   10.119.1.52    rs-mn-mi325x8-pool-3cmiju     <none>           <none>
pdd-decode-2    1/1     Running   0          85m   10.119.0.243   rs-mn-mi325x8-pool-3c35jt     <none>           <none>
pdd-prefill-0   1/1     Running   0          85m   10.119.1.121   rs-mn-mi325x8-pool-3cmiju     <none>           <none>
pdd-router-0    1/1     Running   0          85m   10.119.0.59    rs-mn-cpu-power-pool-37f7cs   <none>           <none>
pdd-router-1    1/1     Running   0          85m   10.119.2.100   rs-mn-cpu-power-pool-3crqkh   <none>           <none>
```


## Benchmark - KV Cache Transnfer Performance


### Test Environment and Scripts

```
DOKS version: 1.34.1-do.2, RDMA Sharing enabled
Hardware: 2 MI325x8 nodes, with 8 × 400 Gbps Rails

Model: Qwen3-8B
vLLM version: v0.21.0
UCX: 1.20.1
RIXL: 1.1.0

NIXL settings:
  UCX_NET_DEVICES=all 
  UCX_MAX_RNDV_RAILS=4 
  UCX_IB_TRAFFIC_CLASS=104 

K8S manifest files:
   nixl-pdd-1P1D1R-tp8-8rails.yaml (TP8 -> TP8)
   nixl-pdd-1P3D2R-tp4-8rails.yaml (TP4 -> TP4)
   nixl/nixl-pdd-1P1D1R-tp8-tp4-8rails.yaml (TP Mismatch: TP8 -> TP4)
```

### Test Results

| TP | Prefill Instance | Interconnect                             | Decode Instance  | TTFT (ms) | KV Cache</BR>Transfer Latency (ms) | Overhead: </BR>Transfer Latency / FFTT | Throughput (GB/s) | Link Capacity (GB/s) | Utilization |
| -: | ---------------- | ---------------------------------------- | ---------------- | --------: | -----------------------------: | ----------------------: | ----------------: | -------------------: | ----------: |
|  8 | 8 GPUs           | 8 Rails                                  | 8 GPUs           |    909.73 |                         25.965 |               **2.85%** |           177.552 |                  400 |  **44.39%** |
|  4 | 4 GPUs (0/1/2/3) | 8 Rails                                  | 4 GPUs (0/1/2/3) |   1314.43 |                         41.066 |               **3.12%** |           112.264 |                  400 |  **28.07%** |
|  4 | 4 GPUs (0/1/2/3) | 4 Rails (0/1/2/3)</br>**NUMA Alignment**    | 4 GPUs (0/1/2/3) |   1306.14 |                         44.173 |               **3.38%** |           104.368 |                  200 |  **53.05%** |
|  4 | 4 GPUs (0/1/2/3) | 4 Rails (4/5/6/7)</br>**NUMA Misalignment** | 4 GPUs (0/1/2/3) |   1380.97 |                         76.811 |               **5.56%** |            60.020 |                  200 |  **30.01%** |
|  4 | 4 GPUs (0/1/2/3) | XGMI (Intra-node)                        | 4 GPUs (4/5/6/7) |   1372.22 |                         115.64 |               **8.43%** |            39.867 |                  512 |  **7.79%** |

NixlConnector demonstrates solid functionality and benefits significantly from NUMA alignment. However, across all tested topologies, [MooncakeConnector](mc-pdd-doks-amd.MD#test-2-tp1tp4tp8-kv-cache-transfer-performance-across-different-rail-configurations) consistently achieves higher KV-transfer throughput and lower transfer latency, resulting in lower overall TTFT.