#!/bin/bash

# 检查是否提供了参数
if [ -z "$1" ]; then
  echo "Usage: $0 {"
  echo "  1: Validating communication of KV cache"
  echo "  2: Running without MSCCL++ environment, no stage parallelism"
  echo "  3: Running with stage parallelism"
  echo "}"
  exit 1
fi

export PYTHONPATH=$(pwd):$(pwd)/mscclpp/python
# aryl149/llama-2-7b-hf

# 根据参数选择要运行的命令
case "$1" in
  1)
    echo "Validating communication of KV cache"
    python tests/distributed/test_kvcache_comm.py --tensor-parallel-size $2 --model daryl149/llama-2-13b-chat-hf
    ;;
  2)
    echo "Running without MSCCL++ environment, no stage parallelism"
    python examples/llm_engine_example_single.py --tensor-parallel-size $2 --model daryl149/llama-2-13b-chat-hf --batch_size $3 --input_len $4 --max_output_len $5
    ;;
  3)
    echo "Running with stage parallelism"
    python examples/llm_engine_example_single.py --tensor-parallel-size $2 --model daryl149/llama-2-13b-chat-hf --sep-prompt-token --batch_size $3 --input_len $4 --max_output_len $5
    ;;
  *)
    echo "Invalid option: $1"
    echo "Usage: $0 {1|2|3}"
    exit 1
    ;;
esac
