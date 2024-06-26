import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from ray import serve

fastapi_app = FastAPI()


@serve.deployment
@serve.ingress(fastapi_app)
class FastAPIIngress:
    @fastapi_app.get("/{name}")
    async def say_hi(self, name: str) -> str:
        return PlainTextResponse(f"Hello {name}!")

@serve.deployment
class VLLMDeployment:

app = FastAPIIngress.bind()
