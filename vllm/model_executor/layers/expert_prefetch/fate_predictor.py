import torch
import torch.nn.functional as F
class FatePredictor:
    def __init__(self, num_layers: int, top_k: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.top_k = top_k
        self.num_layers = num_layers
        self.gate_weights: list[torch.Tensor] = [None] * num_layers

    def load_gate_weights(self, moe_layers):
        """Hàm này được gọi SAU KHI vLLM load_weights xong toàn bộ model."""
        for i, layer in enumerate(moe_layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                # Lấy tensor weight, detach khỏi đồ thị tính toán và đẩy lên device
                weight = layer.mlp.gate.weight.detach().clone().to(self.device)
                self.gate_weights[i] = weight

    def predict_experts_batch(self, layer_id: int, hidden_states: torch.Tensor):
        if layer_id >= self.num_layers or self.gate_weights[layer_id] is None:
            return torch.tensor([], device=self.device, dtype=torch.int32)

        hs_device = hidden_states.detach().to(self.device, non_blocking=True)
        if hs_device.dim() > 2:
            num_tokens, hidden_dim = hs_device.shape[-2], hs_device.shape[-1]
            hs_device = hs_device.view(-1, hidden_dim)

        weight = self.gate_weights[layer_id]
        logits = F.linear(hs_device, weight)

        topk_ids = torch.topk(logits, k=self.top_k, dim=-1).indices
        
        predicted_ids = torch.unique(topk_ids.reshape(-1)).to(torch.int32)
        return predicted_ids