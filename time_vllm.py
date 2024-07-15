from argparse import ArgumentParser
from vllm import LLM, SamplingParams
import time
import math
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = ArgumentParser()
parser.add_argument("--model", default='bigscience/bloom', type=str, help="model_name")
parser.add_argument("--input_size", type=int, default=128, help="input prompt token size")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--max_new_tokens", default=50, type=int, help="maximum new tokens to generate")
parser.add_argument("--tensor_para_size", default=8, type=int, help="tensor parallelism")
parser.add_argument("--iters", default=5, type=int, help="number of iterations")
parser.add_argument("--greedy", action='store_true', help="greedy generation mode - temperature=0")
parser.add_argument("--print_output", action='store_true', help="print generated output text")
parser.add_argument("--test_perf", action='store_true', help="test performance to include warmup runs")
parser.add_argument("--separate_pt", action='store_true', help="separate prompt token")
args = parser.parse_args()

max_model_len = (args.max_new_tokens + args.input_size) * args.batch_size
llm = LLM(args.model,
        tensor_parallel_size=args.tensor_para_size,
        dtype=torch.float16,
        max_model_len=max_model_len,
        disable_log_stats=False,
        sep_prompt_token=args.separate_pt,
        )

tokenizer = llm.get_tokenizer()

with open('/home/workcode/examples/seattle.txt', 'r') as f:
    text = f.read()
    encoded_text = tokenizer(text).input_ids
    prompt = tokenizer.decode(encoded_text[:args.input_size])

prompts = [prompt]
if args.batch_size > len(prompts):
    prompts *= math.ceil(args.batch_size / len(prompts))

prompts = prompts[:args.batch_size]
if args.greedy:
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
else:
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens)

# warmup
if args.test_perf:
    outputs = llm.generate(prompts, sampling_params)

for i in range(args.iters):
    # counter_process = multiprocessing.Process(target=monitor, args=(1, True))
    # counter_process.start()
    # start = time.time()
    print(f"iteration {i+1}/{args.iters}")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    # print("elapsed time:", time.time() - start)
    # counter_process.terminate()

    if args.print_output:
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            #print(f"output: {output}")
            print(f"Generated text: {generated_text!r}")
