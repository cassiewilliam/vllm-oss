import cupy as cp
import os
import torch

from vllm.utils import get_total_num_gpus, MAX_SLOT_IDS

try:
    import mscclpp.comm as mscclpp_comm
    from mscclpp import Transport
    from mscclpp.utils import KernelBuilder, pack
except ImportError:
    raise ImportError(
        "MSCCL++ is not installed. Please install MSCCL++ to use this feature."
    )

# Flush MSCCL++ fifo every 128 operations
FLUSH_COUNT = 128

HEAD_TYPES = [0, 1]  # 0 for keys, 1 for values

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../csrc"


class SendKVKernel:
    """ SendKVKernel is a wrapper around a CUDA kernel that uses
    MSCCL++ proxy channels to asynchronously send key-value cache
    """

    def __init__(self):
        self._kernel = KernelBuilder(
            file="kv_comm_kernels.cu",
            kernel_name="nw_cache_out_kernel",
            file_dir=KERNEL_DIR).get_compiled_kernel()
        self.nblocks = 1
        self.nthreads = 1
        self.stream = torch.cuda.current_stream()

    # nw_cache_out_kernel takes device handles, memory offset, memory size,
    # and flush flag as parameters
    def __call__(self, params):
        # cur_stream_cp = cp.cuda.get_current_stream()
        # sync_point = cp.cuda.Event()
        # sync_point.record(stream=cur_stream)
        # self.stream.wait_event(sync_point)
        # cur_stream = torch.cuda.current_stream()
        # print("Expected compute stream = ", cur_stream.cuda_stream, " cp_compute_stream = ", cur_stream_cp.ptr, " comm_stream =", self.stream.cuda_stream, flush=True)
        # with torch.cuda.stream(self.stream):
        #     self.stream.wait_stream(cur_stream)
        return self._kernel.launch_kernel(params,
                                          self.nblocks,
                                          self.nthreads,
                                          shared=0,
                                          stream=self.stream)


class SignalKVKernel:
    """ SignalKVKernel is a wrapper around a CUDA kernel that signals
    the semaphore associated with the MSCCL++ proxy channel
    """

    def __init__(self):
        self._kernel = KernelBuilder(
            file="kv_comm_kernels.cu",
            kernel_name="nw_cache_out_signal_kernel",
            file_dir=KERNEL_DIR).get_compiled_kernel()
        self.nblocks = 1
        self.nthreads = 1
        self.stream = torch.cuda.current_stream()

    # nw_cache_out_signal_kernel takes device handles of proxy channels
    # as parameters
    def __call__(self, params):
        return self._kernel.launch_kernel(params,
                                          self.nblocks,
                                          self.nthreads,
                                          shared=0,
                                          stream=self.stream)


class WaitKVKernel:
    """ WaitKVKernel is a wrapper around a CUDA kernel that waits on
    the semaphore associated with the MSCCL++ proxy channel
    """

    def __init__(self):
        self._kernel = KernelBuilder(
            file="kv_comm_kernels.cu",
            kernel_name="nw_cache_in_kernel",
            file_dir=KERNEL_DIR).get_compiled_kernel()
        self.nblocks = 1
        self.nthreads = 1
        self.stream = torch.cuda.current_stream()

    # nw_cache_in_kernel takes device handles of proxy channels as parameters
    def __call__(self, params):
        return self._kernel.launch_kernel(params,
                                          self.nblocks,
                                          self.nthreads,
                                          shared=0,
                                          stream=self.stream)


class KVCacheCommunicator:
    """ KVCacheCommunicator provides an interface to communicate the KV cache
    between prompt and token workers using MSCCL++ proxy channels.

    block_size: int - size of a single KV cache block
    device_handles: dict - device handles of MSCCL++ proxy channels
    flush_counter: int - counter to keep track of number of operations
    memory_ids: dict - memory ids of KV cache on prompt and token workers
    my_rank: int - rank of the prompt worker
    remote_rank: int - rank of the token worker

    SendKVKernel and SignalKVKernel put KV cache data and signal semaphores on the prompt side
    WaitKVKernel waits on semaphores on the token side.
    """

    def __init__(self, block_size, device_handles, memory_ids, my_rank,
                 remote_rank):
        self.block_size = block_size
        self.device_handles = device_handles
        self.memory_ids = memory_ids
        self.my_rank = my_rank
        self.remote_rank = remote_rank
        self.flush_counter = 0
        self.send_kernel = SendKVKernel()
        self.signal_kernel = SignalKVKernel()
        self.wait_kernel = WaitKVKernel()
        # self.comm_stream = torch.cuda.Stream()

    def get_device_handles(self, sem_ids):
        device_handles = [self.device_handles[sem_id] for sem_id in sem_ids]
        return cp.asarray(memoryview(b"".join(device_handles)), dtype=cp.uint8)

    def wait(self, sem_id):
        dh = self.get_device_handles([sem_id])
        params = pack(dh)
        self.wait_kernel(params)
        # self.wait_kernel(params, self.comm_stream)

    def signal_and_flush(self, sem_id):
        dh = self.get_device_handles([sem_id])
        params = pack(dh, self.my_rank)
        self.signal_kernel(params)
        # self.signal_kernel(params, self.comm_stream)
        self.flush_counter = 0

    def put(self, sem_id, layer_id, block_start, num_blocks):
        block_size = self.block_size
        remote_rank = self.remote_rank
        my_rank = self.my_rank
        for head_type in HEAD_TYPES:
            block_offset = block_start * block_size
            dh = self.get_device_handles([sem_id])
            self.flush_counter += 1
            flush = self.flush_counter >= (FLUSH_COUNT - 1)
            if flush:
                self.flush_counter = 0
            params = pack(dh,
                          self.memory_ids[layer_id][head_type][remote_rank],
                          self.memory_ids[layer_id][head_type][my_rank],
                          block_offset, block_size * num_blocks, my_rank, flush)
            self.send_kernel(params)
            # self.send_kernel(params, self.comm_stream)


class KVCacheCommManager:

    def __init__(self, rank, world_size, num_prompt_workers,
                 mscclpp_init_method) -> None:
        self.kvcache_comm = None
        self.proxy_service = None
        self.proxy_channel = None

        # Initialize the MSCCL++ group.
        self.mscclpp_group = mscclpp_comm.CommGroup(
            rank=rank,
            size=world_size,
            interfaceIpPortTrio=mscclpp_init_method,
        )

        # Setup up connections.
        self.corr_worker_rank = (rank + num_prompt_workers) % world_size
        # transport = self.mscclpp_group.my_ib_device(rank %
        #                                             get_total_num_gpus())
        transport = Transport.CudaIpc
        self.mscclpp_conns = self.mscclpp_group.make_connection(
            [self.corr_worker_rank], transport)

    def setup_comm(self, num_layers, kv_cache) -> None:
        # Set up proxy service and proxy channels for KV cache communication.
        self.proxy_service = mscclpp_comm.ProxyService()
        self.proxy_service.start_proxy()

        # register KV cache memory with MSCCL++ proxy channel
        memory_ids = [[None, None] for _ in range(num_layers)]
        for layer_id in range(num_layers):
            for head_type in HEAD_TYPES:
                memory_ids[layer_id][
                    head_type] = self.mscclpp_group.register_memory_with_proxy(
                        self.proxy_service,
                        kv_cache[layer_id][head_type],
                        self.mscclpp_conns,
                    )

        # register semaphores with MSCCL++ proxy channel
        # one for each sequence
        proxy_channels = [None for _ in range(MAX_SLOT_IDS)]
        device_handles = [None for _ in range(MAX_SLOT_IDS)]
        for sem_id in range(MAX_SLOT_IDS):
            proxy_channels[
                sem_id] = self.mscclpp_group.register_semaphore_with_proxy(
                    self.proxy_service,
                    self.mscclpp_conns,
                )[self.corr_worker_rank]
            device_handles[sem_id] = proxy_channels[sem_id].device_handle().raw
        self.proxy_channel = proxy_channels
        all_blocks_size = (kv_cache[0][0].numel() *
                           kv_cache[0][0].element_size())
        block_size = all_blocks_size // kv_cache[0][0].size(0)

        # Set up KV cache communicator.
        self.kvcache_comm = KVCacheCommunicator(block_size, device_handles,
                                                memory_ids,
                                                self.mscclpp_group.my_rank,
                                                self.corr_worker_rank)

    def destroy_comm(self) -> None:
        self.proxy_service.stop_proxy()
        del self.proxy_service
        del self.proxy_channel
        del self.kvcache_comm
        del self.mscclpp_group

    def wait(self, sem_id):
        self.kvcache_comm.wait(sem_id)

    def signal_and_flush(self, sem_id):
        self.kvcache_comm.signal_and_flush(sem_id)

    def put(self, sem_id, layer_id, block_start, num_blocks):
        self.kvcache_comm.put(sem_id, layer_id, block_start, num_blocks)
