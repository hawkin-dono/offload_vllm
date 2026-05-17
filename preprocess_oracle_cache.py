import h5py
import torch
import pickle
import time
import os

def preprocess(h5_path, output_pkl, top_k=8):
    cache = {}
    print(f"Reading from {h5_path}...")
    start_time = time.time()
    
    if not os.path.exists(h5_path):
        print(f"Error: File {h5_path} not found.")
        return

    with h5py.File(h5_path, "r") as f:
        for seq_id in f.keys():
            for step_key in f[seq_id].keys():
                if not step_key.startswith("step_"): continue
                step_idx = int(step_key.split("_")[1])
                
                for layer_key in f[seq_id][step_key].keys():
                    if not layer_key.startswith("layer_"): continue
                    try:
                        layer_idx = int(layer_key.split("_")[1])
                    except Exception as e:
                        continue
                    
                    path = f"{seq_id}/{step_key}/{layer_key}/router_logits"
                    if path in f:
                        data = f[path][:]
                        data_tensor = torch.tensor(data)
                        topk_ids = torch.topk(data_tensor, k=top_k, dim=-1).indices
                        # Save as python list for smaller size and fast pickling
                        cache[(seq_id, step_idx, layer_idx)] = topk_ids.view(-1).tolist()
    
    print(f"Computed {len(cache)} entries. Saving to {output_pkl}...")
    with open(output_pkl, "wb") as pf:
        pickle.dump(cache, pf)
    
    print(f"Done in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Thay đổi đường dẫn này theo đúng file H5 lớn nhất của bạn
    # Ví dụ: file data của bạn là vllm_hidden_states.h5
    preprocess(
        h5_path="deepseek_moe_vllm_hidden_states.h5", 
        output_pkl="/home/hieuvt/vllm-hpclab/deepseek_moe_oracle_cache.pkl", 
        top_k=6
    )