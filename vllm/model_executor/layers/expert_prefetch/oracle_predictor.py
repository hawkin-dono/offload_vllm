from typing import Any


import torch 
import numpy as np 
import h5py 
import os 
class OraclePredictor: 
    def __init__(self, data_path: str): 
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data_path = data_path
        
    def predict_experts(self, seq_id, step, layer_id, top_k=8):
        
        with h5py.File(self.data_path, "r") as f:
            base_data = f[f"{seq_id}/step_0/layer_embed/embedding"][:]
            prefill_step = base_data.shape[0]
            print(f"get prefill_step seq_id: {seq_id}, prefill_step: {prefill_step}")
            path = f"{seq_id}/step_{step - prefill_step}/layer_{layer_id}/router_logits"
            print(f"finding path: {path}")
            if path in f:
                print(f"found path: {path}")
                data = f[path][:]
                data_tensor = torch.tensor(data)
                topk_ids = torch.topk(data_tensor, k=top_k, dim=-1).indices
                return topk_ids.numpy()
            else:
                return None  
    def predict_experts_batch(self, seq_ids, steps, layer_ids, top_k=8):
        predictions = []
        steps = steps.tolist()
        print(f"get seq_ids: {seq_ids}")
        print(f"get steps: {steps}")
        for seq_id, step in zip(seq_ids, steps):
            true_seq_id = seq_id.split("-")[1]

            data = self.predict_experts(true_seq_id, step, layer_ids, top_k)
            if data is not None:
                predictions.append(data)
        
        if not predictions:
            return torch.tensor([], device="cpu")
        return torch.tensor(np.array(predictions), device="cpu")
    
def main():
    predictor = OraclePredictor(data_path="/home/hieuvt/vllm-hpclab/dataset_generate/dataset_hidden_states.h5")
    seq = ['cmpl-seq_001-0', 'cmpl-seq_002-0', 'cmpl-seq_004-0', 'cmpl-seq_005-0']
    step = torch.tensor([ 51,  15, 305,  54], device='cuda:0', dtype=torch.int32)
    layer_id = 13
    top_k = 8
    predictions = predictor.predict_experts_batch(seq, step, layer_id, top_k)
    print(predictions)

if __name__ == "__main__":
    main()