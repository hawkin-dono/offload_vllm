import argparse
import os
import re
import time
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn

class Architecture12(nn.Module):
    """
    Architecture 12: Linear -> SiLU -> Dropout -> Linear -> SiLU -> Linear
    Designed for moe input
    Args:
        input_dim (int): Input dimension
        num_experts (int): Number of expert outputs
        hidden_dim (int): Hidden dimension (default: 256)
        dropout_rate (float): Dropout rate (default: 0.05)
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 256,
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.linear3 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: MoE_Input
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.silu(x)
        x = self.linear3(x)
        return x


class Architecture2(nn.Module):
    """
    Architecture 2: Linear -> SiLU -> Linear

    Args:
        input_dim (int): Input dimension
        num_experts (int): Number of expert outputs
        hidden_dim (int): Hidden dimension (default: 2048)
    """

    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 2048):
        super(Architecture2, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_experts)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


_CKPT_NAME_RE = re.compile(r"^(?P<input_type>.+)_layer_(?P<start>\d+)_(?P<end>\d+)$")


def parse_final_checkpoint_stem(stem: str) -> tuple[str, int, int]:
    """
    Parse stem like ``{input_type}_layer_{layer_start}_{layer_end}`` (no ``.ckpt``).
    ``input_type`` may contain underscores; suffix is anchored at ``_layer_<int>_<int>``.
    """
    m = _CKPT_NAME_RE.match(stem)
    if not m:
        raise ValueError(
            f"Checkpoint stem must match '{{input_type}}_layer_{{start}}_{{end}}', got: {stem!r}"
        )
    return m.group("input_type"), int(m.group("start")), int(m.group("end"))


def _sanitize_module_name(stem: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", stem)


class ExpertPredictionModel(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        num_experts,
        top_k=4,
        num_predict_layers: int = 1,
        num_layers=1,
        model_name="architecture12",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.top_k = top_k
        self.num_predict_layers = num_predict_layers

        if model_name == "mlp":
            self.model = Architecture2(input_dim, num_experts * num_predict_layers)
        elif model_name == "mlpv12":
            self.model = Architecture12(input_dim, num_experts * num_predict_layers)
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class MultiCheckpointExpertPredictor(nn.Module):

    def __init__(
        self,
        device: Union[torch.device, str, None] = None,
        checkpoint_dir: str | None = None,
        top_k: int = 4,
    ):
        super().__init__()
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                os.path.dirname(__file__),
                "checkpoints",
                "final",
            )
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.layer_to_input: dict[int, str] = {}
        self.layer_to_model: dict[int, ExpertPredictionModel] = {}
        self.top_k = top_k

        self._load_all_checkpoints()
        self.to(self.device)

    def _load_all_checkpoints(self) -> None:
        if not os.path.isdir(self.checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        used_module_names: dict[str, int] = {}

        for name in sorted(os.listdir(self.checkpoint_dir)):
            if not name.endswith(".ckpt"):
                continue
            path = os.path.join(self.checkpoint_dir, name)
            stem = name[: -len(".ckpt")]
            try:
                input_type, layer_start, layer_end = parse_final_checkpoint_stem(stem)
            except ValueError:
                continue

            print(f"Loading expert predictor checkpoint: {path}")
            pl_model = ExpertPredictionModel.load_from_checkpoint(path)
            pl_model.to(self.device)
            pl_model.eval()

            base = _sanitize_module_name(stem)
            suffix = used_module_names.get(base, 0)
            used_module_names[base] = suffix + 1
            mod_name = base if suffix == 0 else f"{base}_{suffix}"
            self.add_module(mod_name, pl_model)

            for lid in range(layer_start, layer_end + 1):
                if lid in self.layer_to_model:
                    print(
                        f"Warning: layer {lid} already mapped to a checkpoint; "
                        f"overwriting with {path}"
                    )
                self.layer_to_input[lid] = input_type
                self.layer_to_model[lid] = pl_model

    def forward(
        self,
        hidden_state: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        if layer_id not in self.layer_to_model:
            raise KeyError(
                f"No predictor checkpoint registered for layer_id={layer_id}. "
                f"Known layers: {sorted(self.layer_to_model.keys())}"
            )
        model = self.layer_to_model[layer_id]
        return model(hidden_state)

    def inference(
        self,
        hidden_state: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(hidden_state, layer_id=layer_id)
            if logits.dim() == 3:
                _, pred_indices = torch.topk(logits, self.top_k, dim=2)
            else:
                _, pred_indices = torch.topk(logits, self.top_k, dim=1)
        return pred_indices.cpu()

    def predict_experts_batch(
        self,
        hidden_state: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        logits = self.forward(hidden_state, layer_id=layer_id)
        if logits.dim() == 3:
            _, pred_indices = torch.topk(logits, int(self.top_k * 0.7), dim=2)
        else:
            _, pred_indices = torch.topk(logits, int(self.top_k * 0.7), dim=1)
        return torch.unique(pred_indices.cpu().reshape(-1))


def main():
    default_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "final")
    parser = argparse.ArgumentParser(
        description="Smoke-test MultiCheckpointExpertPredictor (checkpoints/final/*.ckpt)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=default_dir,
        help="Directory of .ckpt files named {input_type}_layer_{start}_{end}.ckpt",
    )
    parser.add_argument(
        "--layer-id",
        type=int,
        default=None,
        help="Layer index to route (default: smallest loaded layer id)",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference")
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Fallback top-k if submodel has no top_k (loaded checkpoints usually define their own)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {args.checkpoint_dir}")
        return

    model = MultiCheckpointExpertPredictor(
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        top_k=args.top_k,
    )
    if not model.layer_to_model:
        print(
            f"❌ No checkpoints loaded from {args.checkpoint_dir}. "
            "Add .ckpt files matching {{input_type}}_layer_{{start}}_{{end}}.ckpt"
        )
        return

    test_layer = args.layer_id if args.layer_id is not None else min(model.layer_to_model.keys())
    if test_layer not in model.layer_to_model:
        print(
            f"❌ layer_id={test_layer} not in loaded layers: {sorted(model.layer_to_model.keys())}"
        )
        return

    sub = model.layer_to_model[test_layer]
    input_dim = int(sub.hparams.input_dim) if hasattr(sub.hparams, "input_dim") else 4096

    print("✓ MultiCheckpointExpertPredictor loaded")
    print(f"  checkpoint_dir={model.checkpoint_dir}")
    print(f"  layers={sorted(model.layer_to_model.keys())}")
    print(f"  test_layer={test_layer} input_type={model.layer_to_input.get(test_layer)!r}")

    print("\nGenerating random batch for inference...")
    x_random = torch.randn(args.batch_size, input_dim, device=args.device)

    print(f"✓ Random input shape: {x_random.shape}")

    print(f"\n{'=' * 80}")
    print("RUNNING INFERENCE (MultiCheckpointExpertPredictor)")
    print(f"{'=' * 80}")

    predictions = model.inference(x_random, layer_id=test_layer)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"First rows:\n{predictions[: min(5, predictions.shape[0])]}")

    unique_experts = model.predict_experts_batch(x_random, layer_id=test_layer)
    print(f"\nUnique experts predicted in batch: {unique_experts}")

    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.forward(x_random, layer_id=test_layer)
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    inference_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"\nForward time (one batch, routed layer {test_layer}): {inference_time_ms:.3f} ms")


if __name__ == "__main__":
    main()
