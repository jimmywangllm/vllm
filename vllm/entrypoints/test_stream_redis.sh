CUDA_VISIBLE_DEVICES=1 python api_server_redis.py \
    --port 3099 \
    --model ../../../LLAMA2/model/llama-2-7b-chat-hf \
    --use-np-weights \
    --max-num-batched-tokens 8000 \
    --dtype half \
    --tensor-parallel-size 1

