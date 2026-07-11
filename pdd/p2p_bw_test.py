import os
import torch
import torch.distributed as dist

def main():

    # Initialize distributed process group to set up RCCL communication between all participating processes.
    # init_method="env://":
    #   PyTorch reads all configuration from environment variables:
    #   - MASTER_ADDR  : IP of rank 0 node (coordination server)
    #   - MASTER_PORT  : port used for bootstrap communication
    #   - RANK         : unique ID of this process (0 ... world_size-1)
    #   - WORLD_SIZE   : total number of processes across all nodes
    # backend="nccl":
    #   Uses AMD RCCL or NVIDIA NCCL for GPU-to-GPU communication.
    #   Enables high-performance RDMA, XGM or transfers.
    dist.init_process_group(backend="nccl", init_method="env://")

    # Read runtime configuration
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpu_id = int(os.environ["GPU_ID"])

    # Bind process to a specific GPU to ensures all CUDA operations from this process use the correct device
    torch.cuda.set_device(gpu_id)
    # Create a torch.device object for tensor allocations and operations
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Rank {rank} / {world_size}: using GPU {gpu_id}")

    # size_bytes: total data transferred per iteration
    # 16 GiB chosen to stress RDMA / RCCL pipeline sufficiently
    # num_elements: number of float32 elements (4 bytes each)
    size_bytes = 16 * 1024**3
    num_elements = size_bytes // 4
    print(f"Rank {rank} / {world_size}: allocating tensor")

    # Allocate GPU memory
    # tensor: source buffer (rank 0 sends this)
    # recv_tensor: destination buffer (rank 1 receives into)
    tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
    recv_tensor = torch.empty_like(tensor)
    torch.cuda.synchronize() # blocks the CPU until all previously issued GPU operations (compute, memory, NCCL communication) are fully completed

    # Warmup to trigger NCCL/RCCL communicator initialization, and warm up RDMA connections and GPU kernels
    N_WARMUP = 2
    for i in range(N_WARMUP):
        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            dist.recv(recv_tensor, src=0)
    torch.cuda.synchronize()

    # Measurement using N_ITERS runs to compute average bandwidth
    N_ITERS = 10
    times = []
    for i in range(N_ITERS):

        torch.cuda.synchronize() # Ensure previous GPU work is finished before timing
        # CUDA events provide GPU-side accurate timing (not CPU wall clock)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # This is a blocking point-to-point operation using NCCL/RDMA
        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            dist.recv(recv_tensor, src=0)
        end.record()
        torch.cuda.synchronize() # Ensure GPU operations complete before reading timing

        # Convert CUDA event timing to seconds
        elapsed_ms = start.elapsed_time(end)
        elapsed_sec = elapsed_ms / 1000.0
        times.append(elapsed_sec)

        if rank == 0:
            print(f"iter {i}: {elapsed_sec:.4f} sec")

    # Final statistics - Only rank 0 aggregates and prints results
    if rank == 0:
        avg = sum(times) / len(times)
        bandwidth_gbps = (size_bytes * 8) / avg / 1e9
        bandwidth_giBs = size_bytes / avg / (1024**3)
        print("\n==== RESULT ====")
        print(f"Avg time: {avg:.4f} sec")
        print(f"Bandwidth: {bandwidth_gbps:.2f} Gbps")
        print(f"Bandwidth: {bandwidth_giBs:.2f} GiB/s")

    # Cleanup distributed resources: properly releases NCCL/RCCL communicators and network resources
    dist.destroy_process_group()


if __name__ == "__main__":
    main()