import argparse
import json
from typing import AsyncGenerator

import time
from flasgger import Swagger
from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

import redis


redis_pool = redis.ConnectionPool(
    host='192.168.0.48',
    port=3379,
    password='vitonguE@1@1',
    db=0,
    decode_responses=True)

redis_conn = redis.Redis(connection_pool=redis_pool)


TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None

class Request_1(BaseModel):
    chat_input: str
    id: str
    msg_id: str
    
    class Config:
        schema_extra = {
            "example":{
                "chat_input": "Who are you?",
                "id":"1c7ecc18-9f04-4261-a566-c47dfeda25f5",
                "msg_id":"a91b38487b174b9d8c3fc34e39f767a0",
            }
        }
class no_stream_output(BaseModel):
    text: str 
    class Config:
        schema_extra = {
            "example":{
                "response": ["<<SYS>> You are an assisante. Please answer the questions in less than 512 words <</SYS>> [INST] Who are you? [/INST] Hello! I'm just an AI assistant, here to help"],
                "status": "success",
                "running_time": 0.154534
            }
        }


@app.post("/generate", summary="Streaming accelerate vllm LLAMA 7B", response_description="Streaming Output is a series of the template result.", response_model=no_stream_output)
async def generate(item: Request_1, request: Request) -> Response:
# =Body(example={
#                 "prompt": "<<SYS>> You are an assisante. Please answer the questions in less than 512 words <</SYS>> [INST] Who are you? [/INST]",
#                 "stream": True
#             })
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    -- eg. "max_tokens", "top_p", "top_k", etc

    """
    start = time.time()
    request_dict = await request.json()
    prompt = request_dict.pop("chat_input")
    stream = request_dict.pop("stream", True)
    msgId = request_dict.pop("msg_id")
    chatId = request_dict.pop("id")

    # Add parameters
    if 'max_tokens' not in request_dict:
        request_dict['max_tokens'] = 512
    if 'top_p' not in request_dict:
        request_dict['top_p'] = 0.9
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            # text_outputs = [
            #     prompt + output.text for output in request_output.outputs
            # ]
            # text_outputs = [
            #     output.text for output in request_output.outputs
            # ]
            if len(request_output.outputs[-1].token_ids) == 1:
                text_outputs = request_output.outputs[-1].text
                last_text = request_output.outputs[-1].text
            else:
                tmp_text_outputs = request_output.outputs[-1].text
                text_outputs = tmp_text_outputs.replace(last_text, '')
                last_text = tmp_text_outputs
            print(f"text_output: {text_outputs}")
            ret = {"text": text_outputs}
            message1 = {"chatId": chatId,"msgId": msgId, "response": text_outputs}
            stream_name = f"momrah:sse:chat:{msgId}"
            message1_id = redis_conn.xadd(stream_name, message1)
        
        # Add final signal
        final_message = {"chatId": chatId,"msgId": msgId, "response": "[\FINAL\]"}
        stream_name = f"momrah:sse:chat:{msgId}"
        message1_id = redis_conn.xadd(stream_name, final_message)

    if stream:
        await stream_results()
        return Response(status_code=200)
    
    

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [output.text for output in final_output.outputs]
    # text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"response": text_outputs, "status": "success", "running_time": float(time.time() - start)}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
