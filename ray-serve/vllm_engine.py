from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from starlette.responses import Response
from starlette.requests import Request
from http import HTTPStatus

from fastapi import FastAPI, HTTPException
import logging
import uuid

import nest_asyncio
from ray import serve
from typing import Optional
from pydantic import BaseModel


logger = logging.getLogger("ray.serve")

app = FastAPI()


class GenerateRequest(BaseModel):
    """Generate completion request.

        prompt: Prompt to use for the generation
        max_tokens: Maximum number of tokens to generate per output sequence.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        """
    prompt: Optional[str]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7


class GenerateResponse(BaseModel):
    """Generate completion response.
        output: Model output
        finish_reason: Reason the genertion has finished

    """
    output: Optional[str]
    finish_reason: Optional[str]
    prompt: Optional[str]


def _prepare_engine_args():
    engine_args = AsyncEngineArgs(
        # gpu_memory_utilization=0.98,
        model="Sreenington/Phi-3-mini-4k-instruct-AWQ",
        quantization="AWQ",
        # tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=4096,
        # enforce_eager=True
    )
    return engine_args


@serve.deployment(name='VLLMInference',
                  num_replicas=1,
                  max_concurrent_queries=10,
                  ray_actor_options={"num_gpus": 0.0}
                  )
@serve.ingress(app)
class VLLMInference:
    def __init__(self):
        super().__init__(app)
        self.engine = AsyncLLMEngine.from_engine_args(_prepare_engine_args())
        self.tokenizer = self._prepare_tokenizer()

    @staticmethod
    def _prepare_tokenizer():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(_prepare_engine_args().model)
        tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + '<|end|>' }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + '<|end|>' }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + '<|end|>' }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        return tokenizer

    @app.post("/generate", response_model=GenerateResponse)
    async def generate_text(self, request: GenerateRequest, raw_request: Request) -> GenerateResponse:
        logging.info(f"Received request: {request}")
        try:
            generation_args = request.dict(exclude={'prompt', 'messages'})
            if generation_args is None:
                generation_args = {
                    "max_tokens": 500,
                    "temperature": 0.1,
                }
            if request.prompt is None:
                raise ValueError("Prompt is required")
            request_prompt = request.prompt

            sampling_params = SamplingParams(**generation_args)
            request_id = self._next_request_id()

            # Assuming the prompt is a chat template
            prompt = self.tokenizer.apply_chat_template(
                request_prompt,
                tokenize=False,
                add_generation_prompt=True
            )

            """Example of Prompt
            messages = [
                {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
                {"role": "assistant",
                 "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
                {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
            ]
            """

            results_generator = self.engine.generate(prompt, sampling_params, request_id)

            final_result = None
            async for result in results_generator:
                if await raw_request.is_disconnected():
                    await self.engine.abort(request_id)
                    return GenerateResponse()
                final_result = result  # Store the last result
            if final_result:
                return GenerateResponse(output=final_result.outputs[0].text,
                                        finish_reason=final_result.outputs[0].finish_reason,
                                        prompt=final_result.prompt)
                # print(json.dumps({"text": [final_result.prompt + output.text for output in final_result.outputs]}))
                # print(json.dumps(
                #     {"prompt": final_result.prompt,
                #      "text": [output.text for output in final_result.outputs],
                #      "finish_reason": final_result.outputs[0].finish_reason, },
                #     indent=4))
            else:
                raise ValueError("No results found")
        except ValueError as e:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:
            logger.error('Error in generate()', exc_info=1)
            raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, 'Server error')

    @staticmethod
    def _next_request_id():
        return str(uuid.uuid1().hex)

    async def _abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    @app.get("/health")
    async def health(self) -> Response:
        """Health check."""
        return Response(status_code=200)


deployment_llm = VLLMInference.bind()