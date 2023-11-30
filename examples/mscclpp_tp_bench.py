import argparse
import dataclasses
import time
import torch

from vllm import EngineArgs, LLM, LLMEngine, SamplingParams


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
    warmup_sampling_params = SamplingParams(temperature=0.0)

    # warmup
    outputs = llm.generate(prompt_token_ids=[prompt_token_ids], sampling_params=warmup_sampling_params, use_tqdm=False)
    print("warmup done", len(outputs), flush=True)
    time.sleep(5)
    if args.max_tokens == 1:
        print("prompt_len,prompt_time(s),")
    else:
        print("prompt_len,token_time(s),num_tokens,")
    torch.cuda.synchronize()

    num_iters = 100
    start_time = time.time()
    for i in range(num_iters):
        llm.llm_engine.add_request(str(i), prompt=None, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
        while llm.llm_engine.has_unfinished_requests():
            step_outputs = llm.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
    torch.cuda.synchronize()
    total_time = time.time() - start_time

    if args.max_tokens == 1:
        print(f"[BENCH] time: {prompt_len} {total_time*1000/num_iters}")
    else:
        print(f"[BENCH] time: {prompt_len} {total_time*1000/num_iters}")

    # outputs = sorted(outputs, key=lambda x: int(x.request_id))
    # for output in outputs:
    #     print(output)


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
    args = parser.parse_args()
    main(args)
