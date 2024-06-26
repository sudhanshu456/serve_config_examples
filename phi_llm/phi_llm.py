from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

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


@serve.deployment()
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Determine the name of the served model for the OpenAI client.
        if engine_args.served_model_name is not None:
            served_model_names = engine_args.served_model_name
        else:
            served_model_names = [engine_args.model]
        self.openai_serving_chat = OpenAIServingChat(
            self.engine, served_model_names, response_role, lora_modules, chat_template
        )

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app with predefined arguments."""
    engine_args = AsyncEngineArgs(
        gpu_memory_utilization=float(cli_args.get("gpu_memory_utilization", 0.8)),
        model=cli_args.get("model", "microsoft/Phi-3-mini-4k-instruct"),
        tensor_parallel_size=int(cli_args.get("tensor_parallel_size", 1)),
        served_model_name=cli_args.get("served_model_name"),  # Set this if you have a specific model name
        # Add other required parameters here
    )
    engine_args.worker_use_ray = True

    tp = engine_args.tensor_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    for i in range(tp):
        pg_resources.append({"CPU": 1, "GPU": 1})  # for the vLLM actors

    # We use the "STRICT_PACK" strategy below to ensure all vLLM actors are placed on
    # the same Ray node.
    return VLLMDeployment.options(
        placement_group_bundles=pg_resources, placement_group_strategy="PACK"
    ).bind(
        engine_args,
        cli_args.get("response_role", "system"),  # Set the response role here
        None,  # Set the LoRA modules here if any
        None,  # Set the chat template here if any
    )

# Example usage
cli_args = {
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "tensor_parallel_size": 1,
    "response_role": "system",
    "gpu_memory_utilization": 0.8,  # Adjust this value as needed 
}

app = build_app(cli_args)