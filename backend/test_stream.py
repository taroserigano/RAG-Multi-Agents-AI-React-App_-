"""Test streaming endpoint directly"""
import requests
import sys

def test_stream():
    print("Testing streaming endpoint...")
    try:
        resp = requests.post(
            'http://localhost:8000/api/chat/stream',
            json={'user_id':'test','question':'What is remote work?','provider':'ollama'},
            stream=True,
            timeout=60
        )
        print(f'Status: {resp.status_code}')
        print(f'Headers: {dict(resp.headers)}')
        print('---')
        
        count = 0
        for line in resp.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                print(f'[{count}] {decoded}')
                count += 1
                if count > 15:
                    print('... (truncated after 15 lines)')
                    break
                    
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)

if __name__ == '__main__':
    test_stream()
