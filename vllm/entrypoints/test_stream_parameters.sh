CUDA_VISIBLE_DEVICES=5 python api_server.py \
    --port 3094 \
    --model ../../../LLAMA2/model/llama-2-7b-chat-hf \
    --use-np-weights \
    --max-num-batched-tokens 8000 \
    --dtype half \
    --tensor-parallel-size 1

