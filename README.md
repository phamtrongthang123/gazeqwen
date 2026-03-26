# GazeQwen

Gaze-conditioned video understanding with Qwen2.5-VL. GazeQwen injects eye-tracking
scanpath information into a frozen Qwen2.5-VL-7B model via lightweight hook-based
residual injection, enabling the VLM to leverage human gaze patterns for video QA tasks.

## Architecture

GazeQwen adds a small trainable module (~5M params) on top of frozen Qwen2.5-VL-7B:

- **Backbone**: V-JEPA 2.1 ViT-B/16 (384px) extracts spatiotemporal video features
- **Gaze Input**: DETR-style cosine positional encoding of active fixation coordinates
- **Perceiver**: 32 learnable latents with cross-attention to V-JEPA features + gaze signal
- **Injection**: Per-layer residual bias added to visual tokens at LLM layers [6, 13, 20, 27]
- **LoRA**: Rank-8 adapters on LLM Q/V projections for fine-tuning

## Setup

```bash
pip install -r requirements.txt

# Download V-JEPA 2.1 ViT-B checkpoint (1.6GB)
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt -P checkpoints/

# Download GazeQwen checkpoint from HuggingFace (51MB)
huggingface-cli download phamtrongthang/gazeqwen best_model.pt --local-dir checkpoints/gazeqwen
```

## Quick Demo

```bash
# Run built-in example (OI-Easy task, outputs prediction with confidence)
export PYTHONPATH=".:third_party/vjepa2"
python demo.py

# Run on your own sample
python demo.py \
    --video /path/to/video.mp4 \
    --fixation /path/to/video_fixation_filtered.csv \
    --checkpoint checkpoints/gazeqwen/best_model.pt \
    --question "Which object is the user currently gazing at?" \
    --options "A. box" "B. spices" "C. knife" "D. jar" \
    --time 276.0 --answer C
```

Example output:
```
Video:    OP03-R01-PastaSalad.mp4
Clip:     [216s - 276s]
Gaze:     5 fixations
Question: Which object is the user currently gazing at?
       A. box
       B. spices
   >>> C. knife
       D. jar

Prediction: C (confidence: 92.7%)
Ground truth: C  CORRECT
```

## Inference (Python API)

```python
import torch
from gazeqwen.model import GazeLens
from gazeqwen.hooks import GazeLensContext, register_gazelens_hooks
from gazeqwen.lora import apply_lora, load_lora_state_dict
from gazeqwen.vjepa_features import VJEPAFeatureExtractor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 1. Load frozen Qwen2.5-VL-7B
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# 2. Load GazeQwen checkpoint
checkpoint = torch.load("checkpoints/gazeqwen/best_model.pt", map_location="cpu", weights_only=False)
f_theta = GazeLens(**checkpoint["config"])
f_theta.load_state_dict(checkpoint["state_dict"])

# 3. Apply LoRA and load weights
lora_config = checkpoint.get("lora_config", {"rank": 8, "alpha": 16.0})
apply_lora(model, rank=lora_config["rank"], alpha=lora_config["alpha"])
load_lora_state_dict(model, checkpoint["lora_state_dict"])

# 4. Register hooks
ctx = GazeLensContext(f_theta.to(model.device))
register_gazelens_hooks(model, ctx)

# 5. Extract V-JEPA features
vjepa = VJEPAFeatureExtractor(device=model.device)
# raw_frames: (N, 3, H, W) uint8 tensor from qwen-vl-utils
# features = vjepa.extract_from_raw_frames(raw_frames, T, llm_H, llm_W)

# 6. Forward pass with gaze injection
# scanpath: (N_fixations, 4) tensor [x, y, midpoint_time, duration]
# frame_times: list of float timestamps per ViT temporal step
# with ctx.active(scanpath, frame_times, backbone_features=features):
#     output = model(**inputs)
```

## Training

```bash
# Set paths
export VIDEO_ROOT=/path/to/videos
export FIXATION_ROOT="/path/to/egoexo/fixations /path/to/egtea/fixations"
export QA_DIR=/path/to/qa_files
export VJEPA_CHECKPOINT=checkpoints/vjepa2_1_vitb_dist_vitG_384.pt

# Run training
bash scripts/train.sh
```

Or directly:

```bash
export PYTHONPATH=".:third_party/vjepa2"
python -m gazeqwen.train \
    --qa_files dataset/qa/*.json \
    --video_root dataset/videos \
    --fixation_root "dataset/fixations" \
    --output_dir checkpoints/gazeqwen \
    --split_file splits/gazelens_split.json \
    --vjepa_checkpoint checkpoints/vjepa2_1_vitb_dist_vitG_384.pt \
    --lora_rank 8 --lora_alpha 16.0 --max_epochs 3
```

Requirements: 1x GPU with >= 24GB VRAM (A100 recommended).

## Evaluation

```bash
# GazeQwen evaluation
MODE=gazeqwen CHECKPOINT=checkpoints/gazeqwen/best_model.pt bash scripts/eval.sh

# Baseline (no gaze)
MODE=no_gaze bash scripts/eval.sh
```

## Data Format

**Fixation CSV** (`{video_stem}_fixation_filtered.csv`):
```csv
fixation_id,center_x,center_y,start_time_seconds,end_time_seconds,duration
1,0.42,0.58,1.2,1.8,0.6
2,0.51,0.45,1.85,2.5,0.65
```

**QA JSON** (per task):
```json
[
  {
    "video_path": "video_name.mp4",
    "response_time": "[02:22 - 13:20]",
    "questions": [
      {
        "question": "Which transition best matches the user's gaze pattern?",
        "time_stamp": "13:20",
        "answer": "C",
        "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
        "task_type": "past_gaze_sequence_matching"
      }
    ]
  }
]
```

## Tasks

GazeQwen supports 10 video QA tasks:

| Task | Type | Description |
|------|------|-------------|
| OTP | Past | Object Transition Prediction |
| NFI | Past | Non-Fixated object Identification |
| GSM | Past | Gaze Sequence Matching |
| SR | Past | Scene Recall |
| FAP | Future | Future Action Prediction |
| OAR | Present | Object Attribute Recognition |
| OI-E | Present | Object Identification (Easy) |
| OI-H | Present | Object Identification (Hard) |
| GTA | Proactive | Gaze-Triggered Alert |
| OAA | Proactive | Object Appearance Alert |

## Results

GazeQwen v2 on test set (EgoExo+EGTEA, 249 videos):

| Task | no_gaze | GazeQwen | Delta | p-value |
|------|---------|----------|-------|---------|
| OTP | 37.3% | 64.4% | +27.1% | 0.002 |
| NFI | 36.9% | 70.2% | +33.3% | <0.001 |
| FAP | 33.1% | 45.9% | +12.8% | 0.005 |
| OAR | 51.2% | 81.3% | +30.1% | <0.001 |
| OI-E | 54.8% | 62.2% | +7.4% | 0.038 |
| OI-H | 50.0% | 76.2% | +26.2% | <0.001 |
| GTA | 65.6% | 63.6% | -1.9% | 0.729 |
| OAA | 51.4% | 74.3% | +22.9% | <0.001 |
| **Overall** | **50.4%** | **70.3%** | **+20.0%** | **<0.001** |

## Third-party

- `third_party/vjepa2/`: [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2) (Meta, Apache-2.0 + CC-BY-NC-4.0)
- Base model: [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (Apache-2.0)

## License

Apache-2.0. See [LICENSE](LICENSE).
