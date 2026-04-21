import os
import json
from vllm import LLM, SamplingParams
import json 

# Import global_tracker từ file qwen3_moe.py mà chúng ta vừa sửa
from vllm.model_executor.models.qwen3_moe import global_tracker

def main():
    # CẤU HÌNH PATH & MODEL
    # Hãy thay đổi model_name thành đường dẫn chứa model Qwen MoE của bạn
    model_name = "/dev/shm/Qwen3-30B-A3B" 
    output_h5_path = "dataset_hidden_states.h5"
    
    # Đặt đường dẫn file HDF5 cho tracker
    global_tracker.filepath = output_h5_path

    print(f"Loading model {model_name}...")
    
    # KHỞI TẠO LLM
    # Lưu ý quan trọng: 
    # 1. max_num_seqs=1 để ép vLLM chạy từng sequence một
    # 2. enforce_eager=True để quá trình capture tensor chính xác (tránh lỗi do CUDA Graph)
    llm = LLM(
        model=model_name,
        max_num_seqs=1,
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=64 # Adjust decode length
    )
    
    ########### load dataset  
    dataset_path = "sharegpt_128_test.json"
    dataset = json.load(open(dataset_path, "r"))

    print(f"Bắt đầu generate cho {len(dataset)} sequences...")
    
    for idx, item in enumerate(dataset):
        
        # Tracker bên trong qwen3_moe.py tự đếm bắt đầu từ 1 và format :03d
        true_req_id = f"seq_{idx + 1:03d}"
        item["true_req_id"] = true_req_id
        prompt = item["text"]
        
        print(f"\n[+] Processing sequence: {true_req_id}")
        
        # 2. Sinh văn bản
        # Hàm generate này sẽ tự chạy forward_pass nhiều lần qua model (1 lần prefill + N lần decode)
        # Các hook ở qwen3_moe.py (chạy trên Worker Process) sẽ tự động thu thập tensor,
        # tự phát hiện Seq_ID mới, và tự động lưu/xoá RAM vào file HDF5 sau mỗi step.
        outputs = llm.generate([prompt], sampling_params)
        
        print(f"    - Generated text: {outputs[0].outputs[0].text.strip()}")
        print(f"    - Model Worker auto-saved hidden states for this sequence.")

    # Ghi lại file json với các ID map với HDF5 để dễ dàng trace dữ liệu
    output_json_path = dataset_path.replace(".json", "_with_ids.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print("\nHoàn tất! File dữ liệu HDF5 được lưu tại:", os.path.abspath(output_h5_path))
    print("Thông tin request mapping được lưu tại:", os.path.abspath(output_json_path))

if __name__ == "__main__":
    main()
