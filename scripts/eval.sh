#!/bin/bash
# GazeQwen evaluation script
#
# Usage:
#   MODE=gazeqwen CHECKPOINT=checkpoints/gazeqwen/best_model.pt bash scripts/eval.sh
#   MODE=no_gaze bash scripts/eval.sh
#
# Environment variables:
#   MODE             - "gazeqwen" or "no_gaze" (default: no_gaze)
#   CHECKPOINT       - path to GazeQwen checkpoint (required for gazeqwen mode)
#   VIDEO_ROOT       - path to video files
#   FIXATION_ROOT    - space-separated fixation CSV directories
#   QA_DIR           - path to QA JSON files
#   OUTPUT_FILE      - output JSON path (default: results/<MODE>.json)
#   VJEPA_CHECKPOINT - path to V-JEPA 2.1 checkpoint (auto-detected if not set)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODE="${MODE:-no_gaze}"
VIDEO_ROOT="${VIDEO_ROOT:-${PROJECT_ROOT}/dataset/videos}"
QA_DIR="${QA_DIR:-${PROJECT_ROOT}/dataset/qa}"
FIXATION_ROOT="${FIXATION_ROOT:-${PROJECT_ROOT}/dataset/fixations}"
RESULTS_DIR="${PROJECT_ROOT}/results"
OUTPUT_FILE="${OUTPUT_FILE:-${RESULTS_DIR}/${MODE}.json}"
SPLIT_FILE="${SPLIT_FILE:-${PROJECT_ROOT}/splits/gazelens_split.json}"
SPLIT="${SPLIT:-test}"
CHECKPOINT="${CHECKPOINT:-}"

EXTRA_ARGS=""

if [ -n "${CHECKPOINT}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --checkpoint '${CHECKPOINT}'"
fi

if [ -n "${SPLIT_FILE}" ] && [ -n "${SPLIT}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --split_file '${SPLIT_FILE}' --split ${SPLIT}"
fi

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

mkdir -p "${RESULTS_DIR}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/vjepa2:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== GazeQwen Evaluation ==="
echo "Mode:          ${MODE}"
echo "Output:        ${OUTPUT_FILE}"
echo "Video root:    ${VIDEO_ROOT}"
echo "Fixation root: ${FIXATION_ROOT}"

python -m gazeqwen.eval \
    --mode ${MODE} \
    --qa_files ${QA_FILES[*]} \
    --video_root "${VIDEO_ROOT}" \
    --fixation_root "${FIXATION_ROOT}" \
    --output_file "${OUTPUT_FILE}" \
    ${EXTRA_ARGS}
