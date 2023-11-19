import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TRANSFORMERS_CACHE"] = "/data/hugging_face_model/"
os.environ["TORCH_HOME"] = "/data/hugging_face_model/"
os.environ["HUGGINGFACE_TOKEN"] = "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"
os.environ["HF_HOME"] = "/data/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/.cache/huggingface/hub"

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



TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None

class Request_1(BaseModel):
    prompt: str
    stream: bool
    
    class Config:
        schema_extra = {
            "example":{
                "prompt": "<<SYS>> You are an assisante. Please answer the questions in less than 512 words <</SYS>> [INST] Who are you? [/INST]",
                "stream": False,
                "max_tokens": 512,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "use_beam_search": False
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
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", True)
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
            text_outputs = [
                output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

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
