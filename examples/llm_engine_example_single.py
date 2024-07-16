import argparse
import math
import time
import random
import numpy as np

from typing import List, Tuple
from collections import defaultdict
from typing import Dict

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.transformers_utils.tokenizer import get_tokenizer

def create_test_prompts(tokenizer, batch_size = 8, input_size = 128) -> List[str]:
    """Create a list of test prompts with their sampling parameters."""

    with open('/home/workcode/examples/seattle.txt', 'r') as f:
        text = f.read()
        encoded_text = tokenizer(text).input_ids
        prompt = tokenizer.decode(encoded_text[:input_size])
    prompts = [prompt]
    
    if batch_size > len(prompts):
        prompts *= math.ceil(batch_size / len(prompts))

    prompts = prompts[:batch_size]

    return prompts

# def create_test_prompts(batch_size = 8, ) -> List[Tuple[str, SamplingParams]]:
#     """Create a list of test prompts with their sampling parameters."""
#     # prompt_str = "A robot may not injure a human being. " * 20 + "Repeat this 1 more time."
#     # return [
#     #     (prompt_str,
#     #      SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1)),
#     #      ]
#     return [
#         ("A robot may not injure a human being",
#          SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1)),
#         ("To be or not to be,",
#          SamplingParams(temperature=0.0)),
#         ("What is the meaning of life?",
#          SamplingParams(temperature=0.0)),
#         ("It is only with the heart that one can see rightly",
#          SamplingParams(temperature=0.0)),
#     ]


def process_requests(engine: LLMEngine, test_prompts: List[str], max_tokens = 256):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    step_id = 0

    num_requests = len(test_prompts)

    ttft_start_time: Dict[int, float] = {}
    ttft_latency: Dict[int, float] = {}
    e2e_latency: Dict[int, float] = {}
    tbt_latency: Dict[int, List[float]] = defaultdict(list)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    while True:
        # Add a new request if the mapped time is less than the current time
        if not engine.has_unfinished_requests():
            prompt = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            ttft_start_time[request_id] = time.perf_counter()
            request_id += 1

        while engine.has_unfinished_requests():
            #  print("\nSTEP", step_id)
             start = time.perf_counter()
             request_outputs = engine.step()
             end = time.perf_counter()
             step_time = (end - start) * 1000

             for request_output in request_outputs:
                output_request_id = request_output.request_id
                if (output_request_id not in ttft_latency.keys()):
                    ttft_latency[output_request_id] = (end - ttft_start_time[int(output_request_id)]) * 1000
                # else:
                #     tbt_latency[output_request_id].extend([step_time])
                elif num_requests != len(ttft_latency.keys()):
                    # NOTE: 只统计prefill和decode交叉运行在一起的部分
                    #       如果prefill运行完成了，此时prefill gpu空闲，统计数据意义不大
                    tbt_latency[output_request_id].extend([step_time])
                
                if request_output.finished:
                    e2e_latency[output_request_id] = (end - ttft_start_time[int(output_request_id)]) * 1000
             if np.random.poisson(1) > 0 and test_prompts:
                engine.add_request(str(request_id), test_prompts.pop(0), sampling_params)
                ttft_start_time[request_id] = time.perf_counter()
                request_id += 1
            
             step_id += 1
    
        if not (engine.has_unfinished_requests() or test_prompts):
            break

    # 提取所有数值到一个单一的列表中
    all_latencies = []
    for latencies in tbt_latency.values():
        all_latencies.extend(latencies)

    # 计算均值
    print(f"e2e_latency: {e2e_latency}")
    print(f"ttft_latency: {ttft_latency}")
    # print(f"tbt_latency: {tbt_latency}")
    mean_e2e_latency = sum(e2e_latency.values()) / len(e2e_latency.values())
    mean_ttft_latency = sum(ttft_latency.values()) / len(ttft_latency.values())
    mean_tbt_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    print(f"prefill_perf, request_id: {request_id}, 'phase': 'prefill', length: {len(ttft_latency.values())}, 'latency': {mean_ttft_latency}")
    print(f"decode_perf, request_id: {request_id}, 'phase': 'decode', length: {len(all_latencies)}, 'latency': {mean_tbt_latency}")
    print(f"e2e_perf, request_id: {request_id}, 'phase': 'e2e', length: {len(e2e_latency.values())}, 'latency': {mean_e2e_latency}")

def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)

def main(args: argparse.Namespace):
    # Parse the CLI argument and initialize the engine.
    engine = initialize_engine(args)

    # Test the following prompts.
    test_prompts = [
        ("How is Seattle?",
         SamplingParams(temperature=0.8,
                        top_p=0.95,
                        frequency_penalty=0.1)),
        ("What is the meaning of life?",
         SamplingParams(temperature=0.8,
                        top_p=0.95,
                        frequency_penalty=0.1)),
        ("Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, allowing the body's immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or sometimes administered intravenously. They are not effective against viral infections, and using them inappropriately can lead to antibiotic resistance. Explain the above in one sentence:",
        SamplingParams(temperature=0.8,
                       top_p=0.95,
                       frequency_penalty=0.1)),
        ("Author-contribution statements and acknowledgements in research papers should state clearly and specifically whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their manuscript and analysis. They should also indicate which LLMs were used. This will alert editors and reviewers to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting submitted manuscripts. Mention the large language model based product mentioned in the paragraph above:",
        SamplingParams(temperature=0.0)),
        ("It is only with the heart that one can see rightly",
         SamplingParams(temperature=0.0)),
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0)),
        ("How is San Diego? Generate a list of ten titles for my autobiography. The book is about my journey as an adventurer who has lived an unconventional life, meeting many different personalities and finally finding peace in gardening. Generate a list of ten titles for my autobiography. The book is about my journey as an adventurer who has lived an unconventional life, meeting many different personalities and finally finding peace in        gardening. Generate a list of ten         titles      for my autobiography. The book is about my journey as an adventurer who has lived an unconventional life, meeting many different personalities and finally finding peace in        gardening. Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, allowing the body’s immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid         solutions, or sometimes administered intravenously. They are not effective against viral infections, and using them inappropriately can lead to antibiotic resistance. Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, allowing the body’s immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or         sometimes administered intravenously. They are not effective against viral infections, and using them inappropriately can lead to antibiotic resistance. Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, allowing the body’s immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or sometimes administered     intravenously. They are not effective against viral infections, and using them inappropriately can lead to antibiotic resistance.",
         SamplingParams(temperature=0.9,
                        top_p=0.95,
                        frequency_penalty=0.1)),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
    ]
    
    tokenizer = get_tokenizer(args.model)
    prod_prompts = create_test_prompts(tokenizer, 
                                       batch_size=args.batch_size,
                                       input_size=args.input_len)
    
    print("Warmup begin")
    request_id = 1
    # Warmup
    for prompt, sampling_params in test_prompts:
        engine.add_request(str(request_id), prompt, sampling_params)
        request_id += 1

    while engine.has_unfinished_requests():
        _ = engine.step()
    print("Warmup done")

    print("Profile begin")
    process_requests(engine, prod_prompts, args.max_output_len)
    print("Profile End")
    engine.destroy_kvcache_comm()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    
        # 添加额外的参数
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--input_len', type=int, default=256, help='input_len')
    parser.add_argument('--max_output_len', type=int, default=256, help='max_output_len')

    args = parser.parse_args()
    main(args)
