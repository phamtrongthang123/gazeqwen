#!/bin/bash
# GazeQwen training script
#
# Usage:
#   bash scripts/train.sh
#
# Environment variables (override defaults):
#   VIDEO_ROOT       - path to video files
#   FIXATION_ROOT    - space-separated fixation CSV directories
#   QA_DIR           - path to QA JSON files
#   OUTPUT_DIR       - checkpoint output directory
#   VJEPA_CHECKPOINT - path to V-JEPA 2.1 ViT-B checkpoint
#   LORA_RANK        - LoRA rank (default: 8)
#   LORA_ALPHA       - LoRA alpha (default: 16.0)
#   MAX_EPOCHS       - max training epochs (default: 3)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

VIDEO_ROOT="${VIDEO_ROOT:-${PROJECT_ROOT}/dataset/videos}"
QA_DIR="${QA_DIR:-${PROJECT_ROOT}/dataset/qa}"
FIXATION_ROOT="${FIXATION_ROOT:-${PROJECT_ROOT}/dataset/fixations}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/gazeqwen}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt}"
SPLIT_FILE="${SPLIT_FILE:-${PROJECT_ROOT}/splits/gazelens_split.json}"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"

QA_FILES=(
    "${QA_DIR}/past_gaze_sequence_matching.json"
    "${QA_DIR}/past_non_fixated_object_identification.json"
    "${QA_DIR}/past_object_transition_prediction.json"
    "${QA_DIR}/past_scene_recall.json"
    "${QA_DIR}/present_future_action_prediction.json"
    "${QA_DIR}/present_object_attribute_recognition.json"
    "${QA_DIR}/present_object_identification_easy.json"
    "${QA_DIR}/present_object_identification_hard.json"
    "${QA_DIR}/proactive_gaze_triggered_alert.json"
    "${QA_DIR}/proactive_object_appearance_alert.json"
)

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/vjepa2:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== GazeQwen Training ==="
echo "Video root:       ${VIDEO_ROOT}"
echo "Fixation root:    ${FIXATION_ROOT}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "V-JEPA checkpoint: ${VJEPA_CHECKPOINT}"
echo "LoRA: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "Max epochs:       ${MAX_EPOCHS}"

python -m gazeqwen.train \
    --qa_files ${QA_FILES[*]} \
    --video_root "${VIDEO_ROOT}" \
    --fixation_root "${FIXATION_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --split_file "${SPLIT_FILE}" \
    --vjepa_checkpoint "${VJEPA_CHECKPOINT}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --max_epochs ${MAX_EPOCHS}
