import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import argparse
import time
import math
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
#from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm import EngineArgs, LLMEngine, SamplingParams

# To see the effect of batching beyond allowed sizes, try python examples/esha_llm_engine.py --max-num-batched-tokens 450 --max-model-len 450
# or python examples/esha_llm_engine.py --max-num-batched-tokens 430 --max-model-len 430 --model 'bigscience/bloom' --tensor-parallel-size 8

def main(args: argparse.Namespace):
    # Parse the CLI argument and initialize the engine.
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)

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

    tokenizer = engine.tokenizer
    # Load the prompt from a file.
    filename = f"./input_sample.txt"
    with open(filename, 'r') as f:
        text = f.read()
        encoded_text = tokenizer(text).input_ids

    # Processing Prod Request Trace
    # prod_trace = pd.read_csv('./PT_trace_1.csv')
    # prod_trace['PreciseTimeStamp'] = pd.to_datetime(prod_trace['PreciseTimeStamp'])
    # prod_trace_single = prod_trace.iloc[::30].reset_index(drop=True)
    # prod_trace_single = prod_trace_single[(prod_trace_single['ContextTokens'] != 0) & (prod_trace_single['GeneratedTokens'] != 0) & (prod_trace_single['ContextTokens'] < 8000) & (prod_trace_single['GeneratedTokens'] < 8000)]
    # Mark the start of the trace
    # earliest_timestamp = prod_trace_single['PreciseTimeStamp'].min()

    # Calculate the time differences from the earliest timestamp to each timestamp
    # prod_trace_single['time_diff'] = (prod_trace_single['PreciseTimeStamp'] - earliest_timestamp).dt.total_seconds() * 1000
    # # Map the time differences to the current time
    # prod_trace_single['mapped_time'] = datetime.now() + pd.to_timedelta(prod_trace_single['time_diff'], unit='ms')


    # Processing Synthetic Request Trace
    trace_name = 'rr_chat_2'
    syn_trace = pd.read_csv(f'./{trace_name}.csv')
    syn_trace = syn_trace[syn_trace['arrival_timestamp'] < 120]
    syn_trace = syn_trace[(syn_trace['ContextTokens'] != 0) & (syn_trace['GeneratedTokens'] != 0) & (syn_trace['ContextTokens'] < 8000) & (syn_trace['GeneratedTokens'] < 8000)]

    request_id = 1
    # Warmup
    for prompt, sampling_params in test_prompts:
        engine.add_request(str(request_id), prompt, sampling_params)
        request_id += 1

    while engine.has_unfinished_requests():
        _ = engine.step()

    print("Warmup done")
    # Run the engine by calling `engine.step()` manually.
    request_id = 1
    step_id = 0
    tot_time = 0

    # Create a list of requests to be added to the engine
    prod_prompts = [(ContextTokens,
                     tokenizer.decode(encoded_text[::ContextTokens]),
                     SamplingParams(max_tokens=GeneratedTokens)) for ContextTokens, GeneratedTokens in zip(
                           syn_trace['ContextTokens'], 
                           syn_trace['GeneratedTokens'])]
    df = pd.DataFrame(columns=['request_id', 'phase', 'latency'])

    request_prompt_size = {}
    while True:
        # Add a new request if the mapped time is less than the current time
        if not engine.has_unfinished_requests():
            contextTokens, prompt, sampling_params = prod_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_prompt_size[str(request_id)] = contextTokens
            # print("Added request", request_id, len(prompt.split(' ')), sampling_params.max_tokens)
            request_id += 1

        tot_time = 0
        latency = []
        while engine.has_unfinished_requests():
            print("\nSTEP", step_id)
            # torch.cuda.synchronize()
            start = time.perf_counter()
            request_outputs = engine.step()
            # torch.cuda.synchronize()
            end = time.perf_counter()
            step_time = end - start
            tot_time += step_time
            latency.append(step_time)
            step_id += 1
            # for request_output in request_outputs:
            #     request_output_len = len(request_output.outputs[0].token_ids)
            #     #   Based on the length of the output, we determine how many tokens were processed
            #     if request_output_len == 1:
            #         batched_tokens += request_prompt_size[request_output.request_id]
            #     else:
            #         batched_tokens += 1
            # print("LLMEngine: Tokens Processed:", batched_tokens, ", Time(ms):", step_time*1000)
        output_request_id = request_outputs[0].request_id
        print(request_id)
        prompt_perf = pd.DataFrame([{'request_id': output_request_id, 'phase': 'prompt', 'latency': latency.pop(0)}])
        e2e_perf = pd.DataFrame([{'request_id': output_request_id, 'phase': 'e2e', 'latency': tot_time}])
        df = pd.concat([df, prompt_perf,  e2e_perf], ignore_index=True)
        if latency:
            token_perf = pd.DataFrame({'request_id': [output_request_id] * len(latency), 'phase': ['token'] * len(latency), 'latency': latency})
            df = pd.concat([df, token_perf], ignore_index=True)
        if not (engine.has_unfinished_requests() or prod_prompts):
            break
    # fig = px.timeline(df, x_start="start_time", x_end="end_time", y="request_id", color="phase")
    # fig.update_yaxes(autorange="reversed", title_text="Request ID") # otherwise tasks are listed from the bottom up
    # fig.write_image(file='./512_mixed.png', format='png')
    df.to_csv(f'./{trace_name}_bloom_perf.csv', index=False)

    # print('request_id, prompt_latency, avg_token_latency, max_token_latency, p99_token_latency')
    # for req_id in range(1, request_id, 1+extra_num_prompts):
    #     req_id = str(req_id)
    #     if req_id not in token_time:
    #         token_time[req_id] = [0]
    #     average_token_time = sum(token_time[req_id]) / len(token_time[req_id])
    #     print(req_id, prompt_time[req_id], average_token_time, max(token_time[req_id]), np.percentile(token_time[req_id], 99))
    # print("Total time: ", tot_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
