CUDA_VISIBLE_DEVICES=3 python api_server_no_stream.py \
    --port 3091 \
    --model ../../../LLAMA2/model/llama-2-7b-chat-hf \
    --use-np-weights \
    --max-num-batched-tokens 8000 \
    --dtype half \
    --tensor-parallel-size 1

