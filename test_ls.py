import requests

response = requests.post(
    "http://localhost:8000/joke/invoke",
    json={'input': {'topic': 'cats'}}
)
#print(response.json())
print(response.json()['output']['content'])
# Expected output: {"output": "Why are cats bad storytellers? They only have one tale."}