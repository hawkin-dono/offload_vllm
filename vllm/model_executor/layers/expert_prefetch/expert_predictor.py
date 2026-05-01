import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import pytorch_lightning as pl


import torch
import torch.nn as nn


class Architecture8(nn.Module):
    """
    Architecture 8: Linear -> SiLU -> Linear -> SiLU -> Linear -> SiLU -> Linear
    
    Args:
        input_dim (int): Input dimension
        num_experts (int): Number of expert outputs
        hidden_dim (int): Hidden dimension (default: 2048)
    """
    
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 2048, embedding_length: int = 100):
        super(Architecture8, self).__init__()
        
        self.embedding = nn.Embedding(embedding_length, hidden_dim)  # Embedding layer cho layer_ids
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.silu2 = nn.SiLU()
        self.linear3 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass  
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            layer_ids (torch.Tensor): Tensor of layer IDs for embedding lookup, shape (batch_size,)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_experts)
        """
        embed = self.embedding(layer_ids)  # Shape: (batch_size, hidden_dim)
        x = self.linear1(x)
        x = self.activation(x)
        
        res = x + embed
        
        x = self.linear2(res)
        x = self.silu2(x)
        
        x = x + res
        x = self.linear3(x)

        return x


class ExpertPredictionModel(pl.LightningModule):
    def __init__(self, input_dim, num_experts, loss_type='cross_entropy', top_k=4, num_predict_layers: int = 1, num_layers=1):
        super().__init__()
        self.save_hyperparameters()
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.loss_type = loss_type
        self.top_k = top_k
        self.num_predict_layers = num_predict_layers

        
        self.model = Architecture8(input_dim, num_experts * num_predict_layers, embedding_length= self.num_layers)

    def forward(self, x, layer_ids=None):
        return self.model(x, layer_ids)

    def inference(self, x, layer_ids, device='cpu'):
        """
        Run inference and return predictions on a random batch
        """
        self.eval()
        
        with torch.no_grad():
            
            # Forward pass
            logits = self(x, layer_ids)  
            
            # Extract top-k indices
            if logits.dim() == 3:
                # If shape is [batch, layers, experts]
                _, pred_indices = torch.topk(logits, self.top_k, dim=2)
            else:
                # If shape is [batch, experts]
                _, pred_indices = torch.topk(logits, self.top_k, dim=1)
                
        return pred_indices.cpu()
    def predict_experts_batch(self, x, layer_ids, device='cpu'):
        """
        Run prediction and return top-k predictions on a batch
        """
        experts = self.inference(x, layer_ids, device=device)
        return torch.unique(experts.reshape(-1))
    
    def measure_inference_time(self, x, layer_ids, device='cpu'):
        """
        Measure inference time for a batch
        """
        self.eval()
        
        with torch.no_grad():
            x = x.to(device)
            layer_ids = layer_ids.to(device) if layer_ids is not None else None
            
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            _ = self(x, layer_ids)  # Forward pass
            end_time.record()
            
            # Wait for the events to be recorded
            torch.cuda.synchronize()
            
            inference_time_ms = start_time.elapsed_time(end_time)
        
        return inference_time_ms

def load_checkpoint(checkpoint_path):
    """Load model from PyTorch Lightning checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    model = ExpertPredictionModel.load_from_checkpoint(checkpoint_path)
    model.eval()  # Set to evaluation mode
    return model

def main():
    parser = argparse.ArgumentParser(description='Run Random Inference')
    parser.add_argument('--checkpoint', type=str, default="vllm/model_executor/layers/expert_prefetch/checkpoints/epoch=06-val_acc=0.8151.ckpt",help='Path to checkpoint file (.ckpt)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference')
    parser.add_argument('--top_k', type=int, default=4, help='Top-K predictions to return')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # --- 1. Load Checkpoint ---
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        return
    
    model = load_checkpoint(args.checkpoint)
    model = model.to(args.device)
    if hasattr(model, 'model'):
        model.model = model.model.to(args.device)
        
    print(f"✓ Model loaded successfully")
    input_dim = model.hparams.input_dim if hasattr(model.hparams, 'input_dim') else 4096
    num_layers = model.hparams.num_layers if hasattr(model.hparams, 'num_layers') else 1

    # --- 2. Generate Random Batch ---
    print(f"\nGenerating random batch for inference...")
    x_random = torch.randn(args.batch_size, input_dim)
    
    # Simulate layer IDs (0 to num_layers - 1)
    if num_layers > 1:
        # Assuming you predict for multiple layers or need specific layers
        layer_ids_random = torch.randint(0, num_layers, (args.batch_size,)) 
    else:
        # For single layer or models that don't need layer_ids
        layer_ids_random = torch.zeros(args.batch_size, dtype=torch.long)
    
    print(f"✓ Random input shape: {x_random.shape}")
    print(f"✓ Random layer_ids shape: {layer_ids_random.shape}")
    
    # --- 3. Run Inference ---
    print(f"\n{'='*80}")
    print("RUNNING INFERENCE ON RANDOM BATCH")
    print(f"{'='*80}")
    
    predictions = model.inference(x_random, layer_ids_random, device=args.device)
    
    # --- 4. Print & Save Predictions ---
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"First 5 predictions:\n{predictions[:5]}")
    
    #--- Test predict batch method ---
    unique_experts = model.predict_experts_batch(x_random, layer_ids_random, device=args.device)
    print(f"\nUnique experts predicted in batch: {unique_experts}")
    
    #--- 5. Measure Inference Time ---
    inference_time_ms = model.measure_inference_time(x_random, layer_ids_random, device=args.device)
    print(f"\nInference time for batch: {inference_time_ms:.2f} ms")

if __name__ == "__main__":
    main()
