import requests
from starlette.requests import Request
from typing import Dict

from ray import serve


@serve.deployment(ray_actor_options={"num_gpus": 1})
class PhiLLM:
    def __init__(
        self,
    ):
        super().__init__()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        torch.random.manual_seed(0)
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct"
        )

        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate(self, text):
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        return pipe(text + "\n", **generation_args)[0]["generated_text"]

    async def __call__(self, request: Request) -> Dict:
        data = await request.json()
        text = data["text"]
        generated_text = self.generate(text)
        return {"response": generated_text}


app = PhiLLM.bind()
