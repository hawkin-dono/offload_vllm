import torch 
import numpy as np 
import h5py 
import os 
class OraclePredictor: 
    def __init__(self, data_path: str): 
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data_path = data_path
        
    def predict_experts(self, seq_id, step, layer_id):
        
        with h5py.File(self.data_path, "r") as f:
            base_data = f[f"{seq_id}/step_0/layer_embed/embedding"][:]
            prefile_step = base_data.shape[0]
            path = f"{seq_id}/step_{step - prefile_step}/layer_{layer_id}/router_logits"
            if path in f:
                data = f[path][:]
                return data
            else:
                return None  
    def predict_experts_batch(self, seq_ids, steps, layer_ids):
        predictions = []
        for seq_id, step in zip(seq_ids, steps):
            true_seq_id = seq_id.split("-")[1]
            data = self.predict_experts(true_seq_id, step, layer_ids)
            if data is not None:
                predictions.append(data)
        return torch.tensor(predictions, device="cpu")