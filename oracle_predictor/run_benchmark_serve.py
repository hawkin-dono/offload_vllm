import os
import json
import subprocess
import argparse

def run_benchmark(
    model_path: str,
    dataset_path: str,
    num_requests: int,
    port: int,
    max_tokens: int,
):
    """
    Chạy lệnh vllm bench serve và lưu kết quả vào file json.
    """
    vllm_exec = "/home/hieuvt/vllm-hpclab/.venv/bin/vllm"
    out_json = "benchmark_results.json"

    # Đảm bảo thư mục chứa file kết quả tồn tại
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)

    client_cmd = [
        vllm_exec, "bench", "serve",
        "--backend", "vllm",
        "--model", model_path,
        "--dataset-name", "sharegpt", 
        "--dataset-path", dataset_path, 
        "--num-prompts", str(num_requests),
        "--endpoint", f"/v1/completions",
        "--port", str(port),
        "--sharegpt-output-len", str(max_tokens),
        "--request-rate", "inf",
        "--save-result",
        "--result-filename", out_json
    ]

    print(f"Running command: {' '.join(client_cmd)}")
    
    try:
        # Chạy lệnh và chờ kết thúc
        subprocess.run(client_cmd, check=True)
        print(f"Benchmark completed successfully. Raw results saved to {out_json}")
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        return

    # Đọc file kết quả thô do vllm tạo ra
    if not os.path.exists(out_json):
        print(f"Error: Could not find output file {out_json}")
        return

    try:
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading result file: {e}")
        return

    # Trích xuất các metrics cần thiết
    res_dict = {}
    res_dict["successful_requests"] = data.get("completed")
    res_dict["benchmark_duration_s"] = data.get("duration")
    res_dict["total_input_tokens"] = data.get("total_input_tokens")
    res_dict["total_generated_tokens"] = data.get("total_output_tokens")
    res_dict["request_throughput"] = data.get("request_throughput")
    res_dict["output_token_throughput"] = data.get("output_throughput")
    res_dict["total_token_throughput"] = data.get("total_token_throughput")
    res_dict["mean_ttft_ms"] = data.get("mean_ttft_ms")
    res_dict["median_ttft_ms"] = data.get("median_ttft_ms")
    res_dict["p99_ttft_ms"] = data.get("p99_ttft_ms")
    res_dict["mean_tpot_ms"] = data.get("mean_tpot_ms")
    res_dict["median_tpot_ms"] = data.get("median_tpot_ms")
    res_dict["p99_tpot_ms"] = data.get("p99_tpot_ms")
    res_dict["mean_itl_ms"] = data.get("mean_itl_ms")
    res_dict["median_itl_ms"] = data.get("median_itl_ms")
    res_dict["p99_itl_ms"] = data.get("p99_itl_ms")

    # Lưu lại file kết quả đã được lọc
    filtered_out_json = out_json.replace(".json", "_filtered.json")
    try:
        # Nếu file đã tồn tại, đọc nội dung cũ
        existing_data = []
        if os.path.exists(filtered_out_json):
            with open(filtered_out_json, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except json.JSONDecodeError:
                    existing_data = []
        
        # Thêm kết quả mới vào danh sách
        existing_data.append(res_dict)
        
        with open(filtered_out_json, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4)
        print(f"Filtered metrics appended to {filtered_out_json}")
    except Exception as e:
        print(f"Error saving filtered results: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM serve benchmark")
    parser.add_argument("--model", type=str, default="/dev/shm/Qwen3-30B-A3B", help="Path to the model")
    parser.add_argument("--dataset", type=str, default="dataset_generate/sharegpt_128_test.json", help="Path to the dataset")
    parser.add_argument("--num-requests", type=int, default=4, help="Number of requests to send")
    parser.add_argument("--port", type=int, default=8080, help="Port of the vLLM server")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")

    args = parser.parse_args()

    run_benchmark(
        model_path=args.model,
        dataset_path=args.dataset,
        num_requests=args.num_requests,
        port=args.port,
        max_tokens=args.max_tokens
    )
