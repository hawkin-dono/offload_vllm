from typing import Any


import torch 
import numpy as np 
import h5py 
import os 
import time
import pickle

class OraclePredictor: 
    cache: dict = {}
    def __init__(self, data_path: str, top_k: int, device: str = "cpu", acc=None): 
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data_path = data_path
        self.device = device
        self.top_k = top_k
        if acc is None:
            self.acc = float(os.environ.get("ORACLE_PREDICTOR_ACCURACY", "1.0"))
        else:
            self.acc = acc  # accuracy for simulating prediction errors
        self._preload_data()
        
    def _preload_data(self):
        if OraclePredictor.cache: return
        start_time = time.time()
        print(f"hieuvt:Loading Oracle Predictor data from {self.data_path}...")
        
        # Load from PKL directly
        with open(self.data_path, "rb") as f:
            OraclePredictor.cache = pickle.load(f)
            
        end_time = time.time()
        print(f"hieuvt:Oracle Predictor loaded {len(OraclePredictor.cache)} entries successfully in {end_time - start_time:.2f} seconds!")
        
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
                # convert cached list array back to tensor on correct device
                predictions.append(torch.tensor(data, device=self.device, dtype=torch.int32))
        if not predictions:
            return torch.tensor([], device=self.device, dtype=torch.int32)
        res = torch.cat(predictions)
        res = res[:int(len(res)*self.acc)]  # Simulate prediction errors by keeping only a fraction of the predictions
        return res
    
def main():
    # Update main function to point to the new pickle file instead
    predictor = OraclePredictor(data_path="/home/hieuvt/vllm-hpclab/oracle_cache.pkl", top_k=8, device="cpu")
    seq = ['cmpl-seq_004-0']
    step = [313]
    layer_id = 9
    predictions = predictor.predict_experts_batch(seq, step, layer_id)
    print(predictions)

if __name__ == "__main__":
    main()