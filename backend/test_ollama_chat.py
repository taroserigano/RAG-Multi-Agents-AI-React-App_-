import requests
import json

# Test Ollama chat endpoint
url = "http://localhost:11434/api/chat"
payload = {
    "model": "llama3.1",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": False
}

print("Testing Ollama chat endpoint...")
print(f"URL: {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")
print()

try:
    response = requests.post(url, json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
