import json
import requests
import time

def send_single_request():
    port = 8080
    api_url = f"http://localhost:{port}/v1/completions"
    model_name = "/dev/shm/Qwen3-30B-A3B"
    
    # Prompt đơn giản để test
    prompt = "Solve the equation: 2x + 3 = 7. What is x?"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0.0,
        "request_id": "test_request_1"
    }
    
    print(f"Gửi request: {prompt}")
    print(f"URL: {api_url}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        
        latency = time.time() - start_time
        result = response.json()
        
        print(f"✓ Thành công!")
        print(f"Latency: {latency:.2f}s")
        print(f"\nKết quả:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except requests.exceptions.RequestException as e:
        print(f"✗ LỖI: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Chi tiết: {e.response.text}")

if __name__ == "__main__":
    send_single_request()
