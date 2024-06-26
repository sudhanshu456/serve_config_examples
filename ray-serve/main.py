from typing import Dict, List, AsyncGenerator
import logging
import uuid
from http import HTTPStatus
from ray import serve
from ray.serve import Application
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import StreamingResponse, Response, JSONResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from pydantic import BaseModel
from typing import TYPE_CHECKING, Literal, Optional, TypeVar, Union
from typing import Set, Tuple, Type
from pydantic import BaseModel, root_validator, validator
import yaml
import os

logger = logging.getLogger("ray.serve")

app = FastAPI()


class GenerateRequest(BaseModel):
    """Generate completion request.

        prompt: Prompt to use for the generation
        max_tokens: Maximum number of tokens to generate per output sequence.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        ignore_eos: Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
    
        Note that vLLM supports many more sampling parameters that are ignored here.
        See: vllm/sampling_params.py in the vLLM repository.
        """
    prompt: Optional[str]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    ignore_eos: Optional[bool] = False


class GenerateResponse(BaseModel):
    """Generate completion response.

        output: Model output
        finish_reason: Reason the genertion has finished
    """
    output: str
    finish_reason: Optional[str]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return create_error_response(HTTPStatus.BAD_REQUEST, 'Error parsing JSON payload')


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code.value, content={"detail": message})


@serve.deployment(name='VLLMInference',
                  num_replicas=1,
                  max_concurrent_queries=10,
                  ray_actor_options={"num_gpus": 1.0})
@serve.ingress(app)
class VLLMGenerateDeployment:
    def __init__(self, **kwargs):
        """
        Construct a VLLM deployment.

        Args:
            model: name or path of the huggingface model to use
            download_dir: directory to download and load the weights,
                default to the default cache dir of huggingface.
            load_format: The format of the model weights to load.
                "auto" will try to load the weights in the safetensors format and fall 
                back to the pytorch bin format if safetensors format is not available.
                "pt" will load the weights in the pytorch bin format.
                "safetensors" will load the weights in the safetensors format.
                "npcache" will load the weights in pytorch format and store a numpy 
                cache to speed up the loading.
                "dummy" will initialize the weights with random values, which is mainly 
                for profiling.
            dtype: data type for model weights and activations.
                The "auto" option will use FP16 precision
                for FP32 and FP16 models, and BF16 precision.
                for BF16 models.
            max_model_len: model context length. If unspecified, will be automatically 
                derived from the model.
            worker_use_ray: use Ray for distributed serving, will be
                automatically set when using more than 1 GPU
            pipeline_parallel_size: number of pipeline stages.
            tensor_parallel_size: number of tensor parallel replicas.
            block_size: token block size.
            swap_space: CPU swap space size (GiB) per GPU.
            gpu_memory_utilization: the percentage of GPU memory to be used for
                the model executor
            max_num_batched_tokens: maximum number of batched tokens per iteration
            max_num_seqs: maximum number of sequences per iteration.
            disable_log_stats: disable logging statistics.
            quantization: method used to quantize the weights
            engine_use_ray: use Ray to start the LLM engine in a separate
                process as the server process.
            disable_log_requests: disable logging requests.
        """
        args = AsyncEngineArgs(**kwargs)
        logger.info(args)
        self.engine = AsyncLLMEngine.from_engine_args(args)
        engine_model_config = self.engine.engine.get_model_config()
        self.max_model_len = kwargs.get('max_model_len', engine_model_config.max_model_len)

    def _next_request_id(self):
        return str(uuid.uuid1().hex)

    async def _abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    @app.get("/health")
    async def health(self) -> Response:
        """Health check."""
        return Response(status_code=200)

    @app.post("/generate")
    async def generate(self, request: GenerateRequest, raw_request: Request) -> Response:
        """Generate completion for the request.

        Args:
            request (GenerateRequest): Request object
            raw_request (Request): FastAPI request object

        Returns:
            Response: Response object
        """
        logger.info(f"Request: {request}")

        try:
            if not request.prompt and not request.messages:
                return create_error_response(HTTPStatus.BAD_REQUEST, "Missing parameter 'prompt' or 'messages'")

            if request.prompt:
                prompt = request.prompt
            else:
                raise Exception("no prompt found in request")

            request_dict = request.dict(exclude=set(['prompt', 'messages', 'stream']))

            sampling_params = SamplingParams(**request_dict)
            
            request_id = self._next_request_id()

            output_generator = self.engine.generate(prompt,
                                                    sampling_params,
                                                    request_id)
            
            final_output = None
            async for request_output in output_generator:
                if await raw_request.is_disconnected():
                    await self.engine.abort(request_id)
                    return Response(status_code=200)
                final_output = request_output

            text_outputs = final_output.outputs[0].text
            finish_reason = final_output.outputs[0].finish_reason
            return GenerateResponse(output=text_outputs, finish_reason=finish_reason)

        except ValueError as e:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:
            logger.error('Error in generate()', exc_info=1)
            raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, 'Server error')


def deployment(args: Dict[str, str]) -> Application:
    return VLLMGenerateDeployment.bind(**args)
