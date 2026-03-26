"""
Manual LoRA (Low-Rank Adaptation) for Qwen2.5-VL-7B LLM layers.

Applies low-rank adapters to LLM Q/V projections without external dependencies.
No model wrapping — hooks and device_map work unchanged.

Usage:
    from gazeqwen.lora import apply_lora, get_lora_state_dict, load_lora_state_dict

    # After loading and freezing model:
    lora_layers = apply_lora(model, rank=8, alpha=16.0)

    # Training: get LoRA params for optimizer
    lora_params = get_lora_params(model)

    # Save: extract LoRA weights
    state = get_lora_state_dict(model)

    # Load: restore LoRA weights (model must already have LoRA applied)
    load_lora_state_dict(model, state)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    LoRA adapter wrapping a frozen nn.Linear layer.

    Output = original(x) + (x @ A^T @ B^T) * (alpha / rank)

    Initialization: A is Kaiming uniform, B is zero.
    At init, output = original(x) + 0 = original(x) — no behavior change.
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original  # frozen, requires_grad=False
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # LoRA A: down-projection (rank, in_features) — Kaiming init
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # LoRA B: up-projection (out_features, rank) — zero init
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original (frozen) forward pass
        result = self.original(x)

        # LoRA forward: x @ A^T @ B^T * scaling
        # Cast LoRA params to input dtype (model may be bfloat16)
        lora_out = F.linear(
            F.linear(x, self.lora_A.to(dtype=x.dtype)),
            self.lora_B.to(dtype=x.dtype),
        ) * self.scaling

        return result + lora_out


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: list = None,
) -> list:
    """
    Apply LoRA to target Linear layers in the model (in-place).

    Replaces each target nn.Linear with a LoRALayer wrapper.
    The original linear stays frozen inside the wrapper.

    For Qwen2.5-VL, target_modules=["q_proj", "v_proj"] hits only LLM layers
    (ViT uses "qkv" for its combined projection).

    Args:
        model:          Model to modify (in-place)
        rank:           LoRA rank (default: 8)
        alpha:          LoRA scaling factor (default: 16.0)
        target_modules: List of module name suffixes to target.
                        Default: ["q_proj", "v_proj"]

    Returns:
        List of LoRALayer modules created
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    # Collect modules to replace (can't modify dict during iteration)
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.endswith(t) for t in target_modules):
                to_replace.append((name, module))

    lora_layers = []
    for name, original in to_replace:
        # Create LoRA wrapper
        lora = LoRALayer(original, rank=rank, alpha=alpha)

        # Move LoRA params to same device as original weights
        device = next(original.parameters()).device
        lora = lora.to(device)

        # Replace module in parent
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora)

        lora_layers.append(lora)

    return lora_layers


def get_lora_state_dict(model: nn.Module) -> dict:
    """
    Extract all LoRA parameters from the model as a state dict.

    Returns dict mapping parameter names to tensors (only lora_A/lora_B params).
    """
    state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            state[name] = param.data.clone()
    return state


def load_lora_state_dict(model: nn.Module, state_dict: dict) -> None:
    """
    Load LoRA parameters from a state dict.

    The model must already have LoRA layers applied (via apply_lora).
    """
    model_params = dict(model.named_parameters())
    for name, tensor in state_dict.items():
        if name in model_params:
            model_params[name].data.copy_(tensor)


def get_lora_params(model: nn.Module) -> list:
    """
    Return all LoRA parameters (requires_grad=True) from the model.

    Useful for adding to an optimizer.
    """
    params = []
    for name, param in model.named_parameters():
        if ("lora_A" in name or "lora_B" in name) and param.requires_grad:
            params.append(param)
    return params


def count_lora_params(model: nn.Module) -> int:
    """Count total number of trainable LoRA parameters."""
    return sum(p.numel() for p in get_lora_params(model))
