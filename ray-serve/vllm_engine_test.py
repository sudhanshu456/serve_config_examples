import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

url = "http://127.0.0.1:8000/generate"
payload = json.dumps({
  "prompt": "<|user|>\n<|user|>\n What are Large Language Models?<|end|>\n<|assistant|>",
  "messages": [],
  "max_tokens": 500,
  "temperature": 0.1
})
headers = {
  'Content-Type': 'application/json'
}

def make_request():
    response = requests.post(url, headers=headers, data=payload)
    return response.text

max_workers = 100
# Run 10 requests concurrently
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(make_request) for _ in range(max_workers)]
    for future in as_completed(futures):
        print(future.result())
