import argparse
import json
import time
import pandas as pd
import os
import subprocess
import itertools
import sys
import requests

def wait_for_server(url, timeout=360):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200 or response.status_code == 404: # /v1/models is better but basic HTTP works
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    return False

def wait_for_health(port, timeout=360):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            res = requests.get(f"http://localhost:{port}/health")
            if res.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    return False

def run_serving_benchmark(model_path, max_seqs, max_tokens, port, accuracy):
    print(f"\n--- Server Starting: max_num_seqs={max_seqs}, max_tokens={max_tokens}, accuracy={accuracy} ---")

    # Limit GPU memory utilization based on test
    gpu_util = "0.9"
    python_exec = "/home/hieuvt/vllm-hpclab/.venv/bin/python"
    vllm_exec = "/home/hieuvt/vllm-hpclab/.venv/bin/vllm"
    
    server_env = os.environ.copy()
    server_env["ORACLE_PREDICTOR_ACCURACY"] = str(accuracy)

    server_cmd = [
        python_exec, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--max-num-seqs", str(max_seqs),
        "--gpu-memory-utilization", gpu_util,
        "--port", str(port),
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--enforce-eager"
    ]
    
    server_process = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=server_env)
    
    if not wait_for_health(port, timeout=300):
        server_process.kill()
        print("Server failed to start or timed out.")
        return {
            "max_num_seqs": max_seqs,
            "max_tokens": max_tokens,
            "accuracy": accuracy,
            "error": "Server failed to start"
        }
        
    print("Server is ready. Starting benchmark client...")
    
    out_json = f"bench_res_{max_seqs}_{max_tokens}_{accuracy}.json"
    num_requests = max_seqs * 4 
    
    client_cmd = [
        vllm_exec, "bench", "serve",
        "--backend", "vllm",
        "--model", model_path,
        "--dataset-name", "sharegpt", 
        "--dataset-path", "dataset_generate/sharegpt_128_test.json", 
        "--num-prompts", str(num_requests),
        "--endpoint", f"/v1/completions",
        "--port", str(port),
        "--sharegpt-output-len", str(max_tokens),
        "--request-rate", "inf",
        "--save-result",
        "--result-filename", out_json
    ]
    
    try:
        # Run the serving benchmark and wait for it to complete
        subprocess.run(client_cmd, check=True, env=os.environ.copy())
    except subprocess.CalledProcessError as e:
        server_process.kill()
        print(f"Client benchmarking failed: {e}")
        return {
            "max_num_seqs": max_seqs, "max_tokens": max_tokens, "accuracy": accuracy, "error": "Client failed"
        }

    # Shut down the server eagerly
    server_process.terminate()
    try:
        server_process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server_process.kill()

    # Parse JSON output from vLLM benchmark_serving
    res_dict = {
        "max_num_seqs": max_seqs,
        "max_tokens": max_tokens,
        "accuracy": accuracy,
        "successful_requests": None,
        "benchmark_duration_s": None,
        "total_input_tokens": None,
        "total_generated_tokens": None,
        "request_throughput": None,
        "output_token_throughput": None,
        "mean_ttft_ms": None,
        "median_ttft_ms": None,
        "p99_ttft_ms": None,
        "mean_tpot_ms": None,
        "median_tpot_ms": None,
        "p99_tpot_ms": None,
        "mean_itl_ms": None,
        "median_itl_ms": None,
        "p99_itl_ms": None,
        "error": None
    }
    
    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            data = json.load(f)
            
            res_dict["successful_requests"] = data.get("completed")
            res_dict["benchmark_duration_s"] = data.get("duration")
            res_dict["total_input_tokens"] = data.get("total_input_tokens")
            res_dict["total_output_tokens"] = data.get("total_output_tokens")
            res_dict["request_throughput"] = data.get("request_throughput")
            res_dict["output_token_throughput"] = data.get("output_throughput")
            res_dict["mean_ttft_ms"] = data.get("mean_ttft_ms")
            res_dict["median_ttft_ms"] = data.get("median_ttft_ms")
            res_dict["p99_ttft_ms"] = data.get("p99_ttft_ms")
            res_dict["mean_tpot_ms"] = data.get("mean_tpot_ms")
            res_dict["median_tpot_ms"] = data.get("median_tpot_ms")
            res_dict["p99_tpot_ms"] = data.get("p99_tpot_ms")
            res_dict["mean_itl_ms"] = data.get("mean_itl_ms")
            res_dict["median_itl_ms"] = data.get("median_itl_ms")
            res_dict["p99_itl_ms"] = data.get("p99_itl_ms")

        os.remove(out_json) # Clean up
        
    return res_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/dev/shm/Qwen3-30B-A3B", help="Model path")
    parser.add_argument("--port", type=int, default=8808, help="vLLM server port")
    args = parser.parse_args()

    # Define ranges for grid search
    seq_nums = [1, 2, 4, 8] #1,4,8
    token_lengths = [64]
    accuracies = [1.0]
    
    configs = [
        {"max_num_seqs": s, "max_tokens": t, "accuracy": a}
        for s, t, a in itertools.product(seq_nums, token_lengths, accuracies)
    ]
    
    results = []
    
    for config in configs:
        res = run_serving_benchmark(
            model_path=args.model,
            max_seqs=config["max_num_seqs"],
            max_tokens=config["max_tokens"],
            port=args.port,
            accuracy=config["accuracy"]
        )
        
        results.append(res)
        pd.DataFrame(results).to_csv("benchmark_results.csv", index=False)
        print(f"Finished config: {config}. Results appended.")
        
    print("\n=== All Serving Benchmarks Completed ===")
    df = pd.DataFrame(results)
    print(df.to_string())
    print("Full results saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
