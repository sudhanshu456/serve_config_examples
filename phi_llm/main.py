from fastapi import FastAPI
import logging

from starlette.requests import Request
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from starlette.responses import StreamingResponse, JSONResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat

from ray import serve

logger = logging.getLogger("ray.serve")

fastapi_app = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(fastapi_app)
class FastAPIIngress:
    def __init__(self, model_handle) -> None:
        self.handle = model_handle

    @fastapi_app.post("/v1/chat/completions")
    async def chat(self, request: ChatCompletionRequest, raw_request: Request):
        logger.info(f"Request: {request}")
        # now send the request to the model handle which will handle it
        return await self.handle.create_chat_completion.remote(request, raw_request)

def prepare_engine_args():
    engine_args = AsyncEngineArgs(
        gpu_memory_utilization=0.95,
        model="microsoft/Phi-3-mini-4k-instruct",
        tensor_parallel_size=1,
        worker_use_ray=True,
    )
    return engine_args


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 1},
)
class VLLMDeployment:
    def __init__(self, engine_args: AsyncEngineArgs):
        self.served_model_names = [engine_args.model]
        self.response_role = "assistant"
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.openai_serving_chat = self._async_serving_chat_init()

    async def _async_serving_chat_init(self) -> OpenAIServingChat:
        model_config = await self.engine.get_model_config()
        openai_serving_chat = OpenAIServingChat(self.engine,
                                                model_config,
                                                self.served_model_names,
                                                response_role="assistant",
                                                chat_template=None)
        return openai_serving_chat

    async def create_chat_completion(self, request: ChatCompletionRequest,
                                     raw_request: Request):
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator,
                                     media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


app = FastAPIIngress.bind(VLLMDeployment.bind(prepare_engine_args()))
