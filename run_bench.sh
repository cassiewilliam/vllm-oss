for ((i = 128; i <= 2048; i += 128)); do
    echo "Running plen=$i";
    j=$(printf "%04d" "$i")
    # python time_vllm.py --input_size $i --tensor_para_size 2 --model meta-llama/Llama-2-70b-hf --separate_pt --greedy 2>&1 > llama2_plen_$j.out
    python time_vllm.py --input_size $i --tensor_para_size 8 --model daryl149/llama-2-7b-hf --greedy 2>&1 | tee og_llama_plen_$j.out
done