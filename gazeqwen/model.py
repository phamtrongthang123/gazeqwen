"""
GazeLens f_theta module.

V-JEPA 2.1-backed Voila Perceiver for LLM hidden-state modulation (~1-5M trainable params).
Uses G3 Coord-PE gaze input with S2 per-layer sharing (4 independent modules).

Signature:
    forward(vjepa_features, scanpath, current_frame_time, layer_index, llm_H, llm_W)
        → (H*W, D_llm) residual bias
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gazeqwen.config import (
    QWEN_LLM_HIDDEN_DIM,
    QWEN_LLM_DEPTH,
    VJEPA_HIDDEN_DIM,
    LLM_ACTIVE_LAYERS,
)

_CONFIG_KEYS = ("d_vjepa", "d_llm", "n_latents", "d_latent", "n_blocks", "n_layers")


def _migrate_legacy_state_dict(state_dict):
    """Migrate old flat-naming state dict keys to new nested-naming.

    Old: proj_in.weight, proj_gaze.0.weight, q_projs.0.weight, proj_out.weight, latents
    New: proj_ins.0.weight, proj_gazes.0.0.weight, q_projs.0.0.weight, proj_outs.0.weight, latents.0
    """
    if not any(k.startswith("proj_in.") for k in state_dict):
        return state_dict  # already new format

    new_sd = {}
    block_keys = ("q_projs", "k_projs", "g_projs", "v_projs", "ff_blocks", "block_norms")

    for key, val in state_dict.items():
        if key == "latents":
            new_sd["latents.0"] = val
        elif key.startswith("proj_in."):
            new_sd[key.replace("proj_in.", "proj_ins.0.", 1)] = val
        elif key.startswith("proj_gaze."):
            rest = key[len("proj_gaze."):]
            new_sd[f"proj_gazes.0.{rest}"] = val
        elif key.startswith("proj_out."):
            new_sd[key.replace("proj_out.", "proj_outs.0.", 1)] = val
        elif any(key.startswith(bk + ".") for bk in block_keys):
            for bk in block_keys:
                if key.startswith(bk + "."):
                    rest = key[len(bk) + 1:]
                    new_sd[f"{bk}.0.{rest}"] = val
                    break
        else:
            new_sd[key] = val

    return new_sd


# ---------------------------------------------------------------------------
# Gaze-signal helper: DETR-style cosine positional encoding
# ---------------------------------------------------------------------------

def _detr_cosine_pe(sp, current_frame_time, d_model):
    """DETR-style cosine positional encoding of active fixation coordinates (G3).

    For each active fixation, encode (x, y) with d_model/2 sinusoidal frequencies
    for x and d_model/2 for y, concat → (d_model,).
    Average if multiple active fixations.
    Returns (d_model,) tensor, or None if no active fixations.
    """
    device = sp.device
    mid_t = sp[:, 2]
    dur = sp[:, 3].clamp(min=1e-6)
    start_t = mid_t - dur / 2.0
    end_t = mid_t + dur / 2.0

    active = (current_frame_time >= start_t) & (current_frame_time <= end_t)

    if not active.any():
        return None

    sp_active = sp[active]
    x_coords = sp_active[:, 0]  # (K,) in [0, 1]
    y_coords = sp_active[:, 1]  # (K,) in [0, 1]

    half_d = d_model // 2
    temperature = 10000.0
    dim_t = torch.arange(half_d, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / half_d)

    x_emb = x_coords[:, None] / dim_t[None, :]  # (K, half_d)
    y_emb = y_coords[:, None] / dim_t[None, :]  # (K, half_d)

    pe_x = torch.stack([x_emb[:, 0::2].sin(), x_emb[:, 1::2].cos()], dim=-1).flatten(-2)
    pe_y = torch.stack([y_emb[:, 0::2].sin(), y_emb[:, 1::2].cos()], dim=-1).flatten(-2)

    pe = torch.cat([pe_x, pe_y], dim=-1)  # (K, d_model)
    return pe.mean(0)  # (d_model,) — average over active fixations


# ---------------------------------------------------------------------------
# GazeLens (V-JEPA + Coord-PE + Per-layer)
# ---------------------------------------------------------------------------

class GazeLens(nn.Module):
    """
    V-JEPA 2.1-backed Voila Perceiver for LLM hidden-state modulation.

    Fixed configuration: Coord-PE gaze input, per-layer sharing (4 modules).
    """

    version = "gazelens"

    def __init__(
        self,
        d_vjepa: int = VJEPA_HIDDEN_DIM,
        d_llm: int = QWEN_LLM_HIDDEN_DIM,
        n_latents: int = 32,
        d_latent: int = 256,
        n_blocks: int = 2,
        n_layers: int = QWEN_LLM_DEPTH,
        amplitude_init: float = 1.0,
        **kwargs,  # absorb legacy config keys (gaze_input, per_layer) from old checkpoints
    ):
        super().__init__()
        self.d_vjepa = d_vjepa
        self.d_llm = d_llm
        self.n_latents = n_latents
        self.d_latent = d_latent
        self.n_blocks = n_blocks
        self.n_layers = n_layers

        n_modules = len(LLM_ACTIVE_LAYERS)  # always per-layer (4 modules)

        self.latents = nn.ParameterList([
            nn.Parameter(torch.randn(n_latents, d_latent) * 0.02) for _ in range(n_modules)
        ])

        self.proj_ins = nn.ModuleList([
            nn.Linear(d_vjepa, d_latent, bias=False) for _ in range(n_modules)
        ])

        self.q_projs = nn.ModuleList([nn.ModuleList([nn.Linear(d_latent, d_latent) for _ in range(n_blocks)]) for _ in range(n_modules)])
        self.k_projs = nn.ModuleList([nn.ModuleList([nn.Linear(d_latent, d_latent) for _ in range(n_blocks)]) for _ in range(n_modules)])
        self.g_projs = nn.ModuleList([nn.ModuleList([nn.Linear(d_latent, d_latent) for _ in range(n_blocks)]) for _ in range(n_modules)])
        self.v_projs = nn.ModuleList([nn.ModuleList([nn.Linear(d_latent, d_latent) for _ in range(n_blocks)]) for _ in range(n_modules)])
        self.ff_blocks = nn.ModuleList([nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_latent), nn.Linear(d_latent, d_latent * 4), nn.GELU(), nn.Linear(d_latent * 4, d_latent))
            for _ in range(n_blocks)
        ]) for _ in range(n_modules)])
        self.block_norms = nn.ModuleList([nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(n_blocks)]) for _ in range(n_modules)])

        self.proj_outs = nn.ModuleList()
        for _ in range(n_modules):
            p = nn.Linear(d_latent, d_llm, bias=False)
            nn.init.zeros_(p.weight)
            self.proj_outs.append(p)

        self.amplitude = nn.Parameter(torch.full((n_layers,), amplitude_init))
        self._layer_to_idx = {l: i for i, l in enumerate(LLM_ACTIVE_LAYERS)}

    def get_config(self) -> dict:
        return {k: getattr(self, k) for k in _CONFIG_KEYS}

    def forward(
        self,
        vjepa_features: torch.Tensor,
        scanpath: torch.Tensor,
        current_frame_time: float,
        layer_index: int,
        llm_H: int,
        llm_W: int,
    ) -> torch.Tensor:
        device = vjepa_features.device
        HW = llm_H * llm_W

        if scanpath.shape[0] == 0:
            return torch.zeros(HW, self.d_llm, device=device, dtype=torch.float32)

        sp = scanpath.to(device=device, dtype=torch.float32)

        m = self._layer_to_idx.get(layer_index, 0)

        # --- Step 1: Build gaze signal G (coord_pe) ---
        pe = _detr_cosine_pe(sp, current_frame_time, self.d_latent)
        if pe is None:
            pe = torch.zeros(self.d_latent, device=device, dtype=torch.float32)
        G = pe.unsqueeze(0).expand(HW, -1)

        # --- Step 2: Project backbone features ---
        X = self.proj_ins[m](vjepa_features.to(torch.float32))

        # --- Step 3: Voila Perceiver Blocks ---
        L = self.latents[m]
        pad = torch.zeros(self.n_latents, self.d_latent, device=device, dtype=torch.float32)

        for i in range(self.n_blocks):
            Q = self.q_projs[m][i](L)
            XL = torch.cat([X, L], dim=0)
            GP = torch.cat([G, pad], dim=0)
            K = self.k_projs[m][i](XL) + self.g_projs[m][i](GP)
            V = self.v_projs[m][i](XL)

            scores = torch.mm(Q, K.t()) / math.sqrt(self.d_latent)
            attn_w = F.softmax(scores, dim=-1)
            attn_out = torch.mm(attn_w, V)

            L = L + attn_out
            L = self.block_norms[m][i](L + self.ff_blocks[m][i](L))

        # --- Step 4: Cross-attend X to updated L ---
        scores_out = torch.mm(X, L.t()) / math.sqrt(self.d_latent)
        attn_out_w = F.softmax(scores_out, dim=-1)
        modulation = torch.mm(attn_out_w, L)

        # --- Step 5: Project + amplitude ---
        residual = self.proj_outs[m](modulation)
        amp = self.amplitude[layer_index]
        return residual * amp

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = _migrate_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
