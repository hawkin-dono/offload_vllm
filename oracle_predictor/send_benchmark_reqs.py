import argparse
import json
import time
import pandas as pd
import os
import subprocess
import itertools
import sys
import requests


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

    vllm_exec = "/home/hieuvt/vllm-hpclab/.venv/bin/vllm"
    
    server_env = os.environ.copy()
    server_env["ORACLE_PREDICTOR_ACCURACY"] = str(accuracy)
    
    out_json = f"bench_res_{max_seqs}_{max_tokens}_{accuracy}.json"
    num_requests = 32
    
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
        print(f"Client benchmarking failed: {e}")
        return {
            "max_num_seqs": max_seqs, "max_tokens": max_tokens, "accuracy": accuracy, "error": "Client failed"
        }
        
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/dev/shm/deepseek-moe-16b-base", help="Model path")
    parser.add_argument("--port", type=int, default=8088, help="vLLM server port")
    args = parser.parse_args()

    # Define ranges for grid search
    seq_nums = [1] #1,4,8
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
        
    #     results.append(res)
    #     pd.DataFrame(results).to_csv("benchmark_results.csv", index=False)
    #     print(f"Finished config: {config}. Results appended.")
        
    # print("\n=== All Serving Benchmarks Completed ===")
    # df = pd.DataFrame(results)
    # print(df.to_string())
    # print("Full results saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
