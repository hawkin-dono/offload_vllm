from typing import Any


import torch 
import numpy as np 
import h5py 
import os 
import time
class OraclePredictor: 
    cache: dict = {}
    def __init__(self, data_path: str, top_k: int, device: str = "cpu", acc= 1.0): 
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data_path = data_path
        self.device = device
        self.top_k = top_k
        self.acc = acc  # accuracy for simulating prediction errors
        self._preload_data()
        
    def _preload_data(self):
        if OraclePredictor.cache: return
        start_time = time.time()
        print(f"hieuvt:Loading and precomputing Oracle Predictor data from {self.data_path} to {self.device}...")
        with h5py.File(self.data_path, "r") as f:
            for seq_id in f.keys():
                # if int(seq_id[-1]) >= 5: break
                # seq_id dạng: 'seq_001'
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
                            data_tensor = torch.tensor(data, device=self.device)
                            topk_ids = torch.topk(data_tensor, k=self.top_k, dim=-1).indices
                            OraclePredictor.cache[(seq_id, step_idx, layer_idx)] = topk_ids.view(-1)
                        else:
                            if layer_key != "layer_embed":
                                print(f"hieuvt:Router logits not found for {seq_id}, {step_key}, {layer_key}")
        end_time = time.time()
        print(f"hieuvt:Oracle Predictor loaded {len(OraclePredictor.cache)} entries successfully in {end_time - start_time} seconds!")
        
    def predict_experts_batch(self, seq_ids: list, steps:list, layer_ids):
        predictions = []
        # steps = steps.tolist()
        # print(f"get seq_ids: {seq_ids}")
        # print(f"get steps: {steps}")
        for seq_id, step in zip(seq_ids, steps):
            # cmpl-seq_001-0 -> seq_001
            # Handle both cmpl-seq_001-0 and seq_001-0 formats
            if seq_id.startswith("cmpl-") or seq_id.startswith("chatcmpl-"):
                true_seq_id = seq_id.split("-")[1]
            else:
                true_seq_id = seq_id.split("-")[0]
            
            # Convert step to int in case it's a PyTorch tensor, 
            # since tensors have different hash/equality in dict keys than plain ints.
            step_val = int(step)
                
            data = OraclePredictor.cache.get((true_seq_id, step_val, layer_ids))
            if data is not None:
                predictions.append(data)
        if not predictions:
            return torch.tensor([], device=self.device)
        res = torch.cat(predictions)
        res = res[:int(len(res)*self.acc)]  # Simulate prediction errors by keeping only a fraction of the predictions
        return res
    
def main():
    predictor = OraclePredictor(data_path="/home/hieuvt/vllm-hpclab/vllm_hidden_states.h5", top_k=8, device="cpu")
    seq = ['cmpl-seq_004-0']
    step = [313]
    layer_id = 9
    predictions = predictor.predict_experts_batch(seq, step, layer_id)
    print(predictions)

if __name__ == "__main__":
    main()