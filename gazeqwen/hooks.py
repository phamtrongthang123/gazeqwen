"""
GazeLens hook system for Qwen2.5-VL-7B — LLM injection.

Registers hooks on LLM decoder layers to inject gaze-conditioned residuals
into visual-token hidden states. The residuals are produced by the GazeLens
f_theta network that takes V-JEPA features, scanpath, and frame timing as
input and outputs a per-token additive bias in LLM hidden-state space.

Usage:
    ctx = GazeLensContext(f_theta, active_layers=[6, 13, 20, 27])
    register_gazelens_hooks(model, ctx)

    with ctx.active(scanpath, frame_times, backbone_features=vjepa_feats):
        output = model(**inputs)

    remove_gazelens_hooks(model)

Architecture notes:
  - Qwen2.5-VL LLM has 28 decoder layers (hidden_dim=3584).
  - We inject into evenly-spaced 4 layers: LLM_ACTIVE_LAYERS = [6, 13, 20, 27].
  - Visual tokens (from ViT) occupy a contiguous block in the LLM sequence.
    Their positions are found by locating video_token_id in input_ids via a
    pre-hook on the full model (before embeddings are computed).
  - GazeLens computes a residual from V-JEPA features + scanpath, which is
    added to visual-token hidden states at each active LLM layer.
  - No window_index reordering needed in LLM (tokens already in spatial order).
"""

import contextlib
from typing import Optional, List

import torch
import torch.nn as nn

from gazeqwen.config import (
    QWEN_SPATIAL_MERGE_SIZE,
    LLM_ACTIVE_LAYERS,
)


# ---------------------------------------------------------------------------
# Context object
# ---------------------------------------------------------------------------

class GazeLensContext:
    """
    Mutable context shared by all hooks for a single forward pass.

    Set before each forward pass via the `active()` context manager.
    """

    def __init__(self, f_theta: nn.Module, active_layers: Optional[List[int]] = None):
        self.f_theta = f_theta

        # Subset of LLM layers that receive injection.
        # None means LLM_ACTIVE_LAYERS; pass [] to disable (ablations).
        self.active_layers: Optional[List[int]] = active_layers

        # Per-forward-pass state (set via active())
        self.scanpath: Optional[torch.Tensor] = None  # (N, 4)
        self.frame_times: Optional[List[float]] = None  # one per ViT temporal step

        # Backbone features (populated before forward pass)
        self.backbone_features: Optional[List[torch.Tensor]] = None  # list of (H*W, d_feat) per temporal step

        # Populated by outer vision-encoder pre-hook
        self.window_index: Optional[torch.Tensor] = None  # (total_groups,)
        self.llm_H: Optional[int] = None
        self.llm_W: Optional[int] = None
        self.n_temporal_steps: Optional[int] = None  # T = grid_thw[0][0]
        self.spatial_merge_unit: int = QWEN_SPATIAL_MERGE_SIZE ** 2  # 4

        # Populated by full-model pre-hook
        self.llm_visual_token_start: Optional[int] = None
        self.llm_visual_token_end: Optional[int] = None

        self._enabled: bool = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextlib.contextmanager
    def active(self, scanpath: torch.Tensor, frame_times: List[float],
               backbone_features: Optional[List[torch.Tensor]] = None):
        """
        Context manager that enables bias injection for one forward pass.

        Args:
            scanpath:       (N, 4) tensor [x, y, t, dur] on the same device as the model
            frame_times:    list of float timestamps, one per ViT temporal step
            backbone_features:  list of (H*W, d_feat) tensors, one per temporal step
        """
        self.scanpath = scanpath
        self.frame_times = frame_times
        self.backbone_features = backbone_features
        self._enabled = True
        try:
            yield self
        finally:
            self._enabled = False
            self.scanpath = None
            self.frame_times = None
            self.backbone_features = None
            self.window_index = None
            self.llm_H = None
            self.llm_W = None
            self.n_temporal_steps = None
            self.llm_visual_token_start = None
            self.llm_visual_token_end = None


# ---------------------------------------------------------------------------
# LLM injection
# ---------------------------------------------------------------------------

def _apply_llm_injection(
    ctx: GazeLensContext,
    layer_index: int,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """
    Apply GazeLens gaze injection to visual-token hidden states in an LLM layer.

    For each temporal step, passes pre-extracted V-JEPA features through f_theta
    to produce a residual in LLM hidden-state space, then adds it to the visual
    tokens. Uses torch.cat (no in-place ops) for clean autograd.

    Args:
        ctx:           GazeLensContext (enabled, with llm_H/W/T and visual positions set)
        layer_index:   LLM layer index (passed to f_theta)
        hidden_states: (batch=1, seq_len, D_llm) float tensor

    Returns:
        Modified hidden_states tensor (same shape, same device).
    """
    vis_start = ctx.llm_visual_token_start
    vis_end = ctx.llm_visual_token_end
    llm_H = ctx.llm_H
    llm_W = ctx.llm_W
    T = ctx.n_temporal_steps

    if T is None or T == 0 or vis_start is None or vis_end is None:
        return hidden_states

    expected_vis_len = T * llm_H * llm_W
    actual_vis_len = vis_end - vis_start
    if actual_vis_len != expected_vis_len:
        return hidden_states

    seq_len = hidden_states.shape[1]
    if vis_end > seq_len:
        return hidden_states

    # Build output via torch.cat — no in-place ops, clean autograd graph
    parts = [hidden_states[:, :vis_start, :]]  # prefix tokens
    for t in range(T):
        t_abs_start = vis_start + t * llm_H * llm_W
        t_abs_end = t_abs_start + llm_H * llm_W
        hs_t = hidden_states[0, t_abs_start:t_abs_end, :].to(torch.float32)
        if t < len(ctx.frame_times) and ctx.backbone_features is not None and t < len(ctx.backbone_features):
            t_frame = ctx.frame_times[t]
            feat_t = ctx.backbone_features[t].to(device=hs_t.device, dtype=torch.float32)
            res = ctx.f_theta(feat_t, ctx.scanpath, t_frame, layer_index, llm_H, llm_W)
            hs_t = hs_t + res
        parts.append(hs_t.to(hidden_states.dtype).unsqueeze(0))
    parts.append(hidden_states[:, vis_end:, :])  # suffix tokens
    return torch.cat(parts, dim=1)


def _make_llm_layer_hook(layer_index: int, ctx: GazeLensContext):
    """
    Returns a forward hook for an LLM decoder layer that applies gaze injection
    to visual-token hidden states.

    The hook is registered with model.model.layers[i].register_forward_hook().
    It fires AFTER the layer's forward pass and modifies the output hidden states.
    """
    def _hook(module, input, output):
        if isinstance(output, (tuple, list)):
            hidden_states = output[0]
        else:
            hidden_states = output

        if not ctx.enabled:
            return output
        if ctx.llm_visual_token_start is None or ctx.llm_H is None:
            return output
        if ctx.scanpath is None or ctx.frame_times is None:
            return output
        if ctx.f_theta is None:
            return output

        modified = _apply_llm_injection(ctx, layer_index, hidden_states)

        if isinstance(output, (tuple, list)):
            return type(output)([modified] + list(output[1:]))
        return modified

    return _hook


# ---------------------------------------------------------------------------
# Hook registration / removal
# ---------------------------------------------------------------------------

def register_gazelens_hooks(model, ctx: GazeLensContext) -> None:
    """
    Install GazeLens hooks on Qwen2.5-VL for LLM injection.

    Registers:
      1. Outer pre-hook on model.visual to capture grid_thw (spatial dims).
      2. Full-model pre-hook to find visual token positions in LLM sequence.
      3. Forward hooks on LLM_ACTIVE_LAYERS to inject gaze residuals.

    Args:
        model: Qwen2_5_VLForConditionalGeneration instance
        ctx:   GazeLensContext (shared mutable state)
    """
    visual = model.visual

    # --- Outer pre-hook: capture grid_thw and compute window_index ---
    def _outer_pre_hook(module, args, kwargs=None):
        if not ctx.enabled:
            return
        if len(args) >= 2:
            grid_thw = args[1]
        elif kwargs and "grid_thw" in kwargs:
            grid_thw = kwargs["grid_thw"]
        else:
            return
        if grid_thw is None or grid_thw.shape[0] == 0:
            return

        g = grid_thw[0]
        T, H, W = int(g[0]), int(g[1]), int(g[2])

        ctx.n_temporal_steps = T
        ctx.llm_H = H // QWEN_SPATIAL_MERGE_SIZE
        ctx.llm_W = W // QWEN_SPATIAL_MERGE_SIZE

        try:
            window_index, _ = module.get_window_index(grid_thw)
            ctx.window_index = window_index.to(grid_thw.device)
        except Exception:
            ctx.window_index = None

    outer_hook_handle = visual.register_forward_pre_hook(_outer_pre_hook, with_kwargs=True)
    if not hasattr(visual, "_gazelens_hooks"):
        visual._gazelens_hooks = []
    visual._gazelens_hooks.append(outer_hook_handle)

    # --- Full-model pre-hook: capture input_ids to find visual token positions ---
    def _full_model_pre_hook(module, args, kwargs=None):
        if not ctx.enabled:
            return
        input_ids = None
        if len(args) >= 1 and args[0] is not None:
            input_ids = args[0]
        elif kwargs and kwargs.get("input_ids") is not None:
            input_ids = kwargs["input_ids"]
        if input_ids is None:
            return

        video_token_id = getattr(module.config, "video_token_id", None)
        if video_token_id is None:
            return

        ids = input_ids[0]
        visual_mask = (ids == video_token_id)
        visual_positions = visual_mask.nonzero(as_tuple=False).squeeze(-1)
        if visual_positions.shape[0] == 0:
            return

        if int(visual_positions[-1].item()) - int(visual_positions[0].item()) + 1 != visual_positions.shape[0]:
            return

        ctx.llm_visual_token_start = int(visual_positions[0].item())
        ctx.llm_visual_token_end = int(visual_positions[-1].item()) + 1

    full_model_hook_handle = model.register_forward_pre_hook(
        _full_model_pre_hook, with_kwargs=True
    )
    if not hasattr(model, "_gazelens_hooks"):
        model._gazelens_hooks = []
    model._gazelens_hooks.append(full_model_hook_handle)

    # --- Register LLM layer forward hooks ---
    llm_layers = model.model.layers
    if not hasattr(model.model, "_gazelens_layer_hooks"):
        model.model._gazelens_layer_hooks = []
    for layer_index in LLM_ACTIVE_LAYERS:
        if layer_index < len(llm_layers):
            hook_fn = _make_llm_layer_hook(layer_index, ctx)
            handle = llm_layers[layer_index].register_forward_hook(hook_fn)
            model.model._gazelens_layer_hooks.append(handle)


def remove_gazelens_hooks(model) -> None:
    """
    Remove all GazeLens hooks from the model.
    """
    visual = model.visual

    # Remove ViT outer pre-hook
    if hasattr(visual, "_gazelens_hooks"):
        for handle in visual._gazelens_hooks:
            handle.remove()
        del visual._gazelens_hooks

    # Remove full-model pre-hook
    if hasattr(model, "_gazelens_hooks"):
        for handle in model._gazelens_hooks:
            handle.remove()
        del model._gazelens_hooks

    # Remove LLM layer hooks
    if hasattr(model.model, "_gazelens_layer_hooks"):
        for handle in model.model._gazelens_layer_hooks:
            handle.remove()
        del model.model._gazelens_layer_hooks
