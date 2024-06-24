from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve
from ray.serve.handle import DeploymentHandle

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, handle: DeploymentHandle) -> None:
        self.handle = handle

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        logger.info(f"Request: {request}")
        response = await self.handle.generate.remote(
            request, raw_request
        )
        return response
        

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 0, "max_replicas": 1},
)
class VLLMDeployment:
    def __init__(self):
        engine_args = AsyncEngineArgs(
        # gpu_memory_utilization=0.8,
        model="microsoft/Phi-3-mini-128k-instruct",
        tensor_parallel_size=1,
        )

        engine_args.worker_use_ray = True
        response_role="system"
        lora_modules: Optional[List[LoRAModulePath]] = None,
        chat_template: Optional[str] = None

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        served_model_names = [engine_args.model]
        self.openai_serving_chat = OpenAIServingChat(
            self.engine, served_model_names, response_role, lora_modules, chat_template
        )
       
    def generate(self,  request: ChatCompletionRequest, raw_request: Request):
        generator = self.openai_serving_chat.create_chat_completion(request, raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

entrypoint = APIIngress.bind(VLLMDeployment.bind())
