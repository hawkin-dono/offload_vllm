from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


class FatePredictor:
    def __init__(self, num_layers: int, top_k: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.top_k = top_k
        self.num_layers = num_layers
        self.gate_weights: list[torch.Tensor] = [None] * num_layers

    def load_gate_weights(self, moe_layers):
        for i, layer in enumerate(moe_layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                weight = layer.mlp.gate.weight.detach().clone().to(self.device)
                self.gate_weights[i] = weight
                
        # self.save_gate_weights()

    def save_gate_weights(self, path: str | Path = "vllm/model_executor/layers/expert_prefetch/checkpoints/fate_predictor.pt") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors: list[torch.Tensor | None] = [
            w.detach().cpu().clone() if w is not None else None
            for w in self.gate_weights
        ]
        torch.save(
            {
                "num_layers": self.num_layers,
                "top_k": self.top_k,
                "gate_weights": tensors,
            },
            path,
        )

    def load_gate_weights_from_file(
        self,
        path: str | Path,
        *,
        strict_meta: bool = True,
    ) -> None:
        """Nạp gate weights từ file do `save_gate_weights` tạo (không cần model vLLM)."""
        path = Path(path)
        payload = torch.load(path, map_location=self.device, weights_only=False)
        num_layers = int(payload["num_layers"])
        top_k = int(payload["top_k"])
        if strict_meta and (num_layers != self.num_layers or top_k != self.top_k):
            raise ValueError(
                f"Checkpoint meta mismatch: file has num_layers={num_layers}, "
                f"top_k={top_k}, but this FatePredictor has num_layers={self.num_layers}, "
                f"`FatePredictor.from_saved_checkpoint(...)`."
            )
        weights: list = payload["gate_weights"]
        if len(weights) != self.num_layers:
            raise ValueError(
                f"Checkpoint has {len(weights)} gate tensors, "
                f"expected num_layers={self.num_layers}."
            )
        for i, w in enumerate(weights):
            self.gate_weights[i] = (
                None if w is None else w.to(self.device).detach().clone()
            )

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