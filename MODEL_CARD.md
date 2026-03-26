---
license: apache-2.0
language:
  - en
library_name: transformers
tags:
  - gaze
  - eye-tracking
  - video-understanding
  - video-qa
  - qwen2.5-vl
  - vjepa
  - lora
base_model: Qwen/Qwen2.5-VL-7B-Instruct
pipeline_tag: video-text-to-text
---

# GazeQwen

**Gaze-conditioned video understanding with Qwen2.5-VL-7B.**

GazeQwen injects eye-tracking scanpath information into a frozen Qwen2.5-VL-7B model via lightweight hook-based residual injection, enabling the VLM to leverage human gaze patterns for video QA tasks.

## Model Details

- **Base model**: [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (frozen)
- **Backbone**: V-JEPA 2.1 ViT-B/16 384px (frozen, 86M params)
- **GazeQwen module**: ~10.8M trainable params (Voila Perceiver with Coord-PE gaze input)
- **LoRA adapters**: rank=8, alpha=16.0 on LLM Q/V projections (~3.5M trainable params)
- **Total trainable**: ~14.3M params (0.2% of the full model)
- **Injection layers**: LLM layers [6, 13, 20, 27] out of 28

## Files

| File | Size | Description |
|------|------|-------------|
| `best_model.pt` | 51MB | GazeQwen checkpoint (f_theta weights + LoRA weights) |

The V-JEPA 2.1 backbone checkpoint must be downloaded separately:
```bash
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
```

## Checkpoint Format

```python
checkpoint = torch.load("best_model.pt", map_location="cpu", weights_only=False)
# Keys:
#   "state_dict"      - GazeQwen f_theta module weights
#   "config"          - f_theta architecture config dict
#   "lora_state_dict" - LoRA adapter weights for Qwen2.5-VL LLM
#   "lora_config"     - {"rank": 8, "alpha": 16.0}
```

## Quick Start

```python
import torch
from gazeqwen.model import GazeLens
from gazeqwen.hooks import GazeLensContext, register_gazelens_hooks
from gazeqwen.lora import apply_lora, load_lora_state_dict
from gazeqwen.vjepa_features import VJEPAFeatureExtractor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Load frozen Qwen2.5-VL-7B
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto", device_map="auto", attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Load GazeQwen checkpoint
checkpoint = torch.load("best_model.pt", map_location="cpu", weights_only=False)
f_theta = GazeLens(**checkpoint["config"])
f_theta.load_state_dict(checkpoint["state_dict"])

# Apply LoRA
lora_cfg = checkpoint["lora_config"]
apply_lora(model, rank=lora_cfg["rank"], alpha=lora_cfg["alpha"])
load_lora_state_dict(model, checkpoint["lora_state_dict"])

# Register hooks and run inference
ctx = GazeLensContext(f_theta.to(model.device))
register_gazelens_hooks(model, ctx)
vjepa = VJEPAFeatureExtractor(device=model.device)

# scanpath: (N, 4) tensor [x, y, midpoint_time, duration]
# frame_times: list of float timestamps
# features: list of (H*W, 768) tensors from vjepa.extract_from_raw_frames()
# with ctx.active(scanpath, frame_times, backbone_features=features):
#     output = model(**inputs)
```

## Architecture

```
Input Video + Eye Tracking Scanpath
         │                │
    Qwen2.5-VL        V-JEPA 2.1
    (frozen)           (frozen)
         │                │
         │          ┌─────┴─────┐
         │          │  Coord-PE │ ← Fixation (x,y) encoding
         │          │  Gaze     │
         │          └─────┬─────┘
         │                │
         │     ┌──────────┴──────────┐
         │     │  Voila Perceiver    │ ← 32 latents, 2 blocks
         │     │  (per-layer × 4)   │   4 independent modules
         │     └──────────┬──────────┘
         │                │
    LLM Layers ◄──── Residual Bias  (added at layers 6, 13, 20, 27)
         │
    + LoRA (Q/V)
         │
      Answer (A/B/C/D)
```

## Results

Evaluated on test set (EgoExo+EGTEA, 249 videos with gaze annotations):

| Task | no_gaze | GazeQwen | Delta | p-value |
|------|---------|----------|-------|---------|
| OTP (Object Transition Prediction) | 37.3% | 64.4% | +27.1% | 0.002 |
| NFI (Non-Fixated Identification) | 36.9% | 70.2% | +33.3% | <0.001 |
| FAP (Future Action Prediction) | 33.1% | 45.9% | +12.8% | 0.005 |
| OAR (Object Attribute Recognition) | 51.2% | 81.3% | +30.1% | <0.001 |
| OI-E (Object Identification Easy) | 54.8% | 62.2% | +7.4% | 0.038 |
| OI-H (Object Identification Hard) | 50.0% | 76.2% | +26.2% | <0.001 |
| GTA (Gaze-Triggered Alert) | 65.6% | 63.6% | -1.9% | 0.729 |
| OAA (Object Appearance Alert) | 51.4% | 74.3% | +22.9% | <0.001 |
| **Overall** | **50.4%** | **70.3%** | **+20.0%** | **<0.001** |

Statistically significant improvement on 7 out of 10 tasks (McNemar test, p<0.05).

## Training

- **Data**: 249 videos with eye-tracking from EgoExo and EGTEA datasets, 8521 QA pairs across 10 tasks
- **Split**: 70/15/15 by video (deterministic seed=42)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-2)
- **Schedule**: Linear warmup (20 steps), cosine decay
- **Epochs**: 3 (early stopping on validation accuracy)
- **Hardware**: 1x NVIDIA A100 80GB

## Limitations

- Requires eye-tracking/gaze data at inference time — not applicable to videos without gaze annotations
- Proactive task GTA shows no improvement over baseline (gaze signal may not help with temporal alert detection)
- GSM and SR tasks are underpowered in the test set (n=23 and n=31)

## License

Apache-2.0

## Citation

```bibtex
@software{gazeqwen2026,
  title={GazeQwen: Gaze-Conditioned Video Understanding with Qwen2.5-VL},
  author={Pham, Trong Thang},
  year={2026},
  url={https://huggingface.co/phamtrongthang/gazeqwen}
}
```
