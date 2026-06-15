import requests

url = "http://localhost:11434/api/generate"

# Ensure 'model' matches your exact 'ollama list' output
payload = {
    "model": "llama3",  
    "prompt": "Why is the sky blue?",
    "stream": False
}

# Must use requests.post, NOT requests.get
response = requests.post(url, json=payload)

# This will no longer throw an exception if the model exists
response.raise_for_status() 

print(response.json()['response'])
