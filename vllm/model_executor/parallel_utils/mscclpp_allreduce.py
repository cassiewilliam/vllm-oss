import torch
import cupy as cp

from mscclpp import ProxyService
from mscclpp_benchmark import MscclppAllReduce1, MscclppAllReduce2, MscclppAllReduce3
from vllm.model_executor.parallel_utils.communication_op import broadcast
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_group, get_msccl_tensor_model_parallel_group, get_msccl_tensor_model_parallel_rank)

def type_to_str(dtype):
    if dtype == torch.float16:
        return "__half"
    elif dtype == torch.float32:
        return "float"
    elif dtype == torch.int32:
        return "int"
    else:
        raise RuntimeError("Unknown data type")

def torch_to_cupy_type(dtype):
    if dtype == torch.float16:
        return cp.float16
    elif dtype == torch.float32:
        return cp.float32
    elif dtype == torch.int32:
        return cp.int32
    else:
        raise RuntimeError("Unknown data type")

def bench_time(niter: int, func):
    # capture cuda graph for nites of the kernel launch
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for i in range(niter):
            func(stream)
        graph = stream.end_capture()

    # now run a warm up round
    graph.launch(stream)

    # now run the benchmark and measure time
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record(stream)
    graph.launch(stream)
    end.record(stream)
    end.synchronize()

    return cp.cuda.get_elapsed_time(start, end) / niter * 1000.0

def find_best_config(mscclpp_call, niter, *args):
    best_time = 10000000.0
    for config in mscclpp_call.auto_tune():
        cur_time = bench_time(niter, mscclpp_call, *args)
        if cur_time < best_time:
            best_time = cur_time
            best_config = config
    best_config_tensor = torch.tensor(best_config, dtype=torch.int32, device=torch.cuda.current_device())
    best_config_tensor = broadcast(best_config_tensor, src=0, group=get_tensor_model_parallel_group())
    best_config = best_config_tensor.cpu().tolist()
    if get_msccl_tensor_model_parallel_rank() == 0:
        print(best_config, end="", flush=True)
    return best_config

def find_best_algo(mscclpp_algos, niter):
    assert len(mscclpp_algos) > 0
    best_time = 10000000.0
    best_algo = None
    for algo in mscclpp_algos:
        config, cur_time = find_best_config(algo, niter)
        if cur_time < best_time:
            best_time = cur_time
            best_algo = algo
            algo.set_params(*config)
    if get_msccl_tensor_model_parallel_rank() == 0:
        print(best_algo, end="", flush=True)
    return best_algo

class MscclppAllReduce:
    all_reduce_buff = {}
    ar_kernel = {}

    @classmethod
    def get_best_ar_kernel(cls, nelem, data_type):
        data_type_str = type_to_str(data_type)
        if (nelem, data_type_str) in cls.ar_kernel:
            return cls.ar_kernel[(nelem, data_type_str)]

        mscclpp_group = get_msccl_tensor_model_parallel_group()
        memory = cp.zeros(nelem, dtype=torch_to_cupy_type(data_type))
        memory_out = cp.zeros(nelem, dtype=torch_to_cupy_type(data_type))
        cp.cuda.runtime.deviceSynchronize()

        proxy_service = ProxyService()
        if memory.nbytes < 2**20:
            mscclpp_algos = [MscclppAllReduce2(mscclpp_group, memory, memory_out)]
        else:
            mscclpp_algos = [
                MscclppAllReduce1(mscclpp_group, memory),
                MscclppAllReduce3(mscclpp_group, memory, proxy_service),
            ]

        proxy_service.start_proxy()
        mscclpp_group.barrier()
        mscclpp_call = find_best_algo(mscclpp_algos, 20)

        cls.ar_kernel[(nelem, data_type_str)] = mscclpp_call
        if data_type_str not in cls.all_reduce_buff or cls.all_reduce_buff[data_type_str].size(0) < nelem:
            cls.all_reduce_buff[data_type_str] = torch.zeros(nelem, device=torch.cuda.current_device, dtype=data_type)
        
        return mscclpp_call

    @classmethod
    def mscclpp_allreduce(cls, input_: torch.Tensor):
        nelem = input_.size(0) * input_.size(1)
        data_type_str = type_to_str(input_.dtype)
        mscclpp_ar_call = cls.get_best_ar_kernel(nelem, input_.dtype)
        if hasattr(mscclpp_ar_call, "memory_out"):
            dummy_output = torch.zeros(nelem, device=torch.cuda.current_device(), dtype=input_.dtype)
            mscclpp_ar_call.memory_out = dummy_output
        cls.all_reduce_buff[data_type_str][:nelem] = input_.reshape(-1)
        return mscclpp_ar_call(torch.cuda.current_stream().cuda_stream, nelem)[:nelem].clone().reshape(b1,b2,b3)
