import argparse
import dataclasses
import time
import torch

from vllm import EngineArgs, LLM, LLMEngine, SamplingParams


def run_nvprof(llm, sampling_params, prompt_token_ids, prompt_len):
    outputs = []
    num_iters = 5
    start_iter = 2
    nvtx_enabled = False
    for i in range(num_iters):
        if i == start_iter:
            torch.cuda.cudart().cudaProfilerStart()
            nvtx_enabled = True
        if nvtx_enabled:
            print("ITERATION", i, flush=True)
            torch.cuda.nvtx.range_push("iteration{}".format(i))
        llm.llm_engine.add_request(str(i), prompt=None, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
        step_num = 0
        while llm.llm_engine.has_unfinished_requests():
            if nvtx_enabled:
                torch.cuda.nvtx.range_push("step{}-{}".format(i,step_num))
            step_num += 1
            step_outputs = llm.llm_engine.step()
            if nvtx_enabled:
                torch.cuda.nvtx.range_pop()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        if nvtx_enabled:
            torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

def bench_e2e_time(llm, sampling_params, prompt_token_ids, prompt_len):
    outputs = []
    num_iters = 50
    start_iter = 20
    for i in range(num_iters):
        if i == start_iter:
            torch.cuda.synchronize()
            start_time = time.time()
        llm.llm_engine.add_request(str(i), prompt=None, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
        step_num = 0
        while llm.llm_engine.has_unfinished_requests():
            step_num += 1
            step_outputs = llm.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
    torch.cuda.synchronize()
    total_time = time.time() - start_time

    print(f"[BENCH] time (ms): {prompt_len} {total_time*1000/(num_iters-start_iter)}")

def main(args: argparse.Namespace):
    # Parse the CLI argument and initialize the engine.
    prompt_len = args.prompt_len
    args.max_model_len = prompt_len + args.max_tokens
    args.disable_log_stats = True
    attrs = [attr.name for attr in dataclasses.fields(EngineArgs)]
    llm = LLM(**{attr: getattr(args, attr) for attr in attrs})
    print(f"Using MSCCLPP: {llm.llm_engine.parallel_config.do_mscclpp_tp}", flush=True)

    with open('/home/azureuser/splitwise/vllm/examples/seattle.txt', 'r') as file:
        full_prompt = file.read()
    all_prompt_token_ids = llm.get_tokenizer().encode(full_prompt)
    assert len(all_prompt_token_ids) >= prompt_len
    print(f"[BENCH] prompt_len: {prompt_len} / {len(all_prompt_token_ids)}", flush=True)

    # Test the following prompts.
    prompt_token_ids = all_prompt_token_ids[:prompt_len]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    if args.bench_e2e_time:
        bench_e2e_time(llm, sampling_params, prompt_token_ids, prompt_len)
    elif args.bench_nvprof:
        run_nvprof(llm, sampling_params, prompt_token_ids, prompt_len)
    else:
        print("No benchmark selected", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument(
        '--prompt-len',
        type=int,
        default=512,
        help='Number of tokens in the prompt')
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=16,
        help='Number of tokens to sample')
    parser.add_argument(
        '--bench-e2e-time',
        action='store_true',
        help='Benchmark end-to-end time')
    parser.add_argument(
        '--bench-nvprof',
        action='store_true',
        help='Benchmark with nvprof')
    args = parser.parse_args()
    main(args)
