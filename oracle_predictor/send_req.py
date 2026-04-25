import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def send_single_request(idx, total, item, api_url, model_name):
    true_req_id = item["true_req_id"]
    prompt = item.get("text", "")

    print(f"[{idx+1}/{total}] Bắt đầu gửi request ID: {true_req_id}")

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0.0,
        "request_id": str(true_req_id) 
    }

    try:
        start_time = time.time()
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        latency = time.time() - start_time
        
        generated_text = result["choices"][0]["text"].strip()
        server_req_id = result.get("id")
        
        print(f"[{true_req_id}] Hoàn thành! Server ID: {server_req_id} | Latency: {latency:.2f}s")
        return True
        
    except requests.exceptions.RequestException as e:
        error_msg = f"[{true_req_id}] LỖI: {e}"
        if e.response is not None:
            error_msg += f" | Chi tiết: {e.response.text}"
        print(error_msg)
        return False

def main():
    port = 8004
    api_url = f"http://localhost:{port}/v1/completions"
    model_name = "/dev/shm/Qwen3-30B-A3B"
    dataset_path = "dataset_generate/sharegpt_128_test.json"
    
    # Số lượng luồng (threads) chạy song song
    # Bạn có thể tăng số này lên nếu muốn gửi nhiều request cùng lúc hơn
    max_workers = 32

    print(f"Đang đọc dữ liệu từ {dataset_path}...")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return
    
    dataset = dataset[:64]

    total_reqs = len(dataset)
    print(f"Tổng số requests cần gửi: {total_reqs}")
    print(f"Số luồng chạy song song (max_workers): {max_workers}")
    print("-" * 50)

    start_time_all = time.time()

    # Sử dụng ThreadPoolExecutor để gửi request đồng thời
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tất cả các task vào executor
        futures = [
            executor.submit(send_single_request, idx, total_reqs, item, api_url, model_name)
            for idx, item in enumerate(dataset)
        ]
        
        # Chờ và theo dõi tiến độ
        success_count = 0
        for future in as_completed(futures):
            if future.result():
                success_count += 1

    total_time = time.time() - start_time_all
    print("-" * 50)
    print(f"Đã gửi xong {total_reqs} requests trong {total_time:.2f}s")
    print(f"Thành công: {success_count}/{total_reqs}")

if __name__ == "__main__":
    main()
