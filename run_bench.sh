#!/bin/bash

# 检查是否提供了参数
if [ -z "$1" ]; then
  echo "Usage: $0 {"
  echo "  1: Running with stage parallelism"
  echo "  2: Running without MSCCL++ environment, no stage parallelism"
  echo "}"
  exit 1
fi

export PYTHONPATH=$(pwd):$(pwd)/mscclpp/python

# 根据参数选择要运行的命令
case "$1" in
  1)
    echo "Running with MSCCL++ environment, no stage parallelism"
    for ((i = 128; i <= 2048; i *= 2)); do
        echo "Running plen=$i";
        j=$(printf "%04d" "$i")
        python time_vllm.py --input_size $i --tensor_para_size 4 --model daryl149/llama-2-7b-hf --separate_pt --greedy 2>&1 > disagg_llama2_7b_plen_$j.out
    done
    ;;
  2)
    echo "Running without MSCCL++ environment, no stage parallelism"
    for ((i = 128; i <= 2048; i *= 2)); do
        echo "Running plen=$i";
        j=$(printf "%04d" "$i")
        python time_vllm.py --input_size $i --tensor_para_size 8 --model daryl149/llama-2-7b-hf --greedy 2>&1 | tee raw_llama2_7b_plen_$j.out
    done
    ;;
  *)
    echo "Invalid option: $1"
    echo "Usage: $0 {1|2}"
    exit 1
    ;;
esac