import requests

prompt = "explain how to write a function in python"

print(requests.post(url="http://localhost:8000/",json={ "text" : prompt}).text)