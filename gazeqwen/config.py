# GazeLens hyperparameters

# Qwen2.5-VL-7B vision encoder constants
QWEN_PATCH_SIZE = 14
QWEN_TEMPORAL_PATCH_SIZE = 2
QWEN_SPATIAL_MERGE_SIZE = 2   # spatial_merge_unit = 4

# Qwen2.5-VL LLM constants
QWEN_LLM_DEPTH = 28
QWEN_LLM_HIDDEN_DIM = 3584
QWEN_LLM_NUM_HEADS = 28
QWEN_LLM_HEAD_DIM = 128       # 3584 // 28
LLM_ACTIVE_LAYERS = [6, 13, 20, 27]  # evenly-spaced 4 of 28 layers (0-indexed)

# V-JEPA 2.1 constants (ViT-B/16, 384px)
VJEPA_HIDDEN_DIM = 768
VJEPA_INPUT_SIZE = 384       # 384×384 → 24×24 = 576 spatial patches
VJEPA_PATCH_SIZE = 16
VJEPA_TUBELET_SIZE = 2       # temporal patch size: 2 frames per tubelet
VJEPA_NUM_FRAMES = 16        # number of input frames for V-JEPA

# Training
LR = 3e-4
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 20
GRAD_ACCUM_STEPS = 8
MAX_EPOCHS = 20

# Single mode name (for checkpoint paths, output filenames, etc.)
MODE_NAME = "gazeqwen"
