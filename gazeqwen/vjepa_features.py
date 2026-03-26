"""
V-JEPA 2.1 spatiotemporal feature extraction for VoilaVJEPA.

Loads a frozen V-JEPA 2.1 ViT-B/16 (384px) model and extracts spatiotemporal
features from video frames. Features are temporally interpolated to match
Qwen's T steps and spatially interpolated to match (llm_H, llm_W).

V-JEPA processes 16 frames jointly via 3D patch embedding (tubelet_size=2),
producing 8 temporal tubelets × 576 spatial patches = 4608 tokens of dim 768.

Usage:
    extractor = VJEPAFeatureExtractor(device)
    features = extractor.extract_from_raw_frames(raw_frames, T, llm_H, llm_W)
    # features: list of T tensors, each (llm_H * llm_W, 768)
"""

import logging
import os
import sys
from typing import List

import torch
import torch.nn.functional as F

from gazeqwen.config import (
    VJEPA_HIDDEN_DIM,
    VJEPA_INPUT_SIZE,
    VJEPA_PATCH_SIZE,
    VJEPA_TUBELET_SIZE,
    VJEPA_NUM_FRAMES,
)

logger = logging.getLogger(__name__)

# Spatial grid: 384 / 16 = 24
_SPATIAL_GRID = VJEPA_INPUT_SIZE // VJEPA_PATCH_SIZE  # 24
# Temporal tubelets: 16 / 2 = 8
_N_TUBELETS = VJEPA_NUM_FRAMES // VJEPA_TUBELET_SIZE  # 8

# Project root for PYTHONPATH (needed to import V-JEPA 2.1 code)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VJEPA_ROOT = os.path.join(_PROJECT_ROOT, "third_party", "vjepa2")


def _load_vjepa_encoder(checkpoint_path: str, device: torch.device):
    """Load V-JEPA 2.1 ViT-B encoder from a local checkpoint file."""
    # Add vjepa2 to path so we can import the model code
    if _VJEPA_ROOT not in sys.path:
        sys.path.insert(0, _VJEPA_ROOT)

    from src.hub.backbones import _make_vjepa2_1_model

    # Load with pretrained=False, then manually load weights
    encoder, _predictor = _make_vjepa2_1_model(
        model_name="vjepa2_1_vit_base_384",
        checkpoint_key="ema_encoder",
        img_size=VJEPA_INPUT_SIZE,
        num_frames=VJEPA_NUM_FRAMES,
        predictor_depth=12,
        predictor_num_mask_tokens=8,
        n_output_distillation=1,
        return_all_tokens=True,
        teacher_embed_dim=1664,
        pretrained=False,
    )

    # Load checkpoint weights
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Clean keys (remove module. and backbone. prefixes)
    encoder_sd = {}
    for k, v in state_dict["ema_encoder"].items():
        clean_key = k.replace("module.", "").replace("backbone.", "")
        encoder_sd[clean_key] = v

    encoder.load_state_dict(encoder_sd, strict=True)
    logger.info("V-JEPA 2.1 encoder loaded from %s", checkpoint_path)

    return encoder


class VJEPAFeatureExtractor:
    """
    Frozen V-JEPA 2.1 ViT-B/16 (384px) spatiotemporal feature extractor.

    Processes 16 frames jointly to produce spatiotemporal tokens, then
    interpolates temporally and spatially to match target dimensions.
    """

    # Default checkpoint path (relative to project root)
    DEFAULT_CHECKPOINT = os.path.join(
        _PROJECT_ROOT, "checkpoints", "vjepa2_1_vitb_dist_vitG_384.pt"
    )

    def __init__(self, device: torch.device, checkpoint_path: str = None):
        self.device = device
        self.d_vjepa = VJEPA_HIDDEN_DIM  # 768
        self.input_size = VJEPA_INPUT_SIZE  # 384
        self.n_frames = VJEPA_NUM_FRAMES  # 16
        self.spatial_grid = _SPATIAL_GRID  # 24
        self.n_tubelets = _N_TUBELETS  # 8

        ckpt = checkpoint_path or self.DEFAULT_CHECKPOINT
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"V-JEPA checkpoint not found: {ckpt}\n"
                f"Download: wget https://dl.fbaipublicfiles.com/vjepa2/"
                f"vjepa2_1_vitb_dist_vitG_384.pt -P checkpoints/"
            )

        logger.info("Loading V-JEPA 2.1 ViT-B/16 (384px) ...")
        self.model = _load_vjepa_encoder(ckpt, device)
        self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("V-JEPA loaded: %dM params (frozen)", n_params // 1_000_000)

    @torch.no_grad()
    def extract_from_raw_frames(
        self,
        raw_frames: torch.Tensor,
        n_temporal_steps: int,
        llm_H: int,
        llm_W: int,
    ) -> List[torch.Tensor]:
        """
        Extract V-JEPA spatiotemporal features from raw video frames.

        Args:
            raw_frames:       (N_frames, C, H, W) float tensor from process_vision_info,
                              values in [0, 255] range
            n_temporal_steps: T from video_grid_thw (Qwen's temporal steps)
            llm_H, llm_W:    target spatial grid dimensions

        Returns:
            List of n_temporal_steps tensors, each (llm_H * llm_W, d_vjepa)
        """
        if raw_frames is None or raw_frames.shape[0] == 0:
            return []

        n_input = raw_frames.shape[0]

        # --- Step 1: Sample exactly VJEPA_NUM_FRAMES (16) frames ---
        if n_input >= self.n_frames:
            # Evenly sample 16 frames
            indices = torch.linspace(0, n_input - 1, self.n_frames).long()
        else:
            # Pad by repeating last frame
            indices = list(range(n_input))
            indices += [n_input - 1] * (self.n_frames - n_input)
            indices = torch.tensor(indices, dtype=torch.long)

        frames = raw_frames[indices].to(self.device)  # (16, C, H, W)

        # --- Step 2: Normalize to [0, 1] and resize to 384×384 ---
        frames = frames.float() / 255.0
        frames = frames.clamp(0, 1)
        frames = F.interpolate(
            frames,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        frames = (frames - mean) / std

        # --- Step 3: Reshape to V-JEPA input format: (1, 3, T, H, W) ---
        # frames: (16, 3, 384, 384) → (1, 3, 16, 384, 384)
        video_input = frames.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, 16, 384, 384)

        # --- Step 4: Forward through V-JEPA encoder ---
        features = self.model(video_input)  # (1, 4608, 768)

        # --- Step 5: Reshape to (n_tubelets, spatial_H, spatial_W, d_vjepa) ---
        # 4608 = 8 tubelets × 24 × 24
        feat = features[0]  # (4608, 768)
        feat = feat.view(self.n_tubelets, self.spatial_grid, self.spatial_grid, self.d_vjepa)
        # feat: (8, 24, 24, 768)

        # --- Step 6: Temporal interpolation: 8 tubelets → n_temporal_steps ---
        # Reshape for interpolation: (1, d_vjepa, n_tubelets, spatial_grid, spatial_grid)
        feat_5d = feat.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 768, 8, 24, 24)

        if self.n_tubelets != n_temporal_steps:
            feat_5d = F.interpolate(
                feat_5d,
                size=(n_temporal_steps, self.spatial_grid, self.spatial_grid),
                mode="trilinear",
                align_corners=False,
            )
        # feat_5d: (1, 768, T, 24, 24)

        # --- Step 7: Spatial interpolation per temporal step → (llm_H, llm_W) ---
        features_list = []
        for t in range(n_temporal_steps):
            feat_t = feat_5d[0, :, t, :, :]  # (768, spatial_H, spatial_W)
            feat_t = feat_t.unsqueeze(0)  # (1, 768, H, W)

            if feat_t.shape[2] != llm_H or feat_t.shape[3] != llm_W:
                feat_t = F.interpolate(
                    feat_t.float(),
                    size=(llm_H, llm_W),
                    mode="bilinear",
                    align_corners=False,
                )

            # (llm_H * llm_W, d_vjepa)
            feat_flat = feat_t.squeeze(0).permute(1, 2, 0).reshape(
                llm_H * llm_W, self.d_vjepa
            )
            features_list.append(feat_flat)

        return features_list
