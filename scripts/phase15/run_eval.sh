#!/bin/bash
set -euo pipefail
# Usage: bash scripts/phase15/run_eval.sh [gpu_base] [gpu_explicit] [gpu_typed]
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=/data/storage22t/lht/envs/qwen25vl/bin/python
Q=eval_results/qwen/phase15/heldout_200.jsonl
IMG=/data/lht/relsim_dataset/relsim_images
GPU_B=${1:-4}
GPU_E=${2:-3}
GPU_T=${3:-7}

mkdir -p eval_results/qwen/phase15/answers eval_results/qwen/phase15/logs

run_one () {
  local gpu=$1 run=$2 adapter=$3
  local ans=eval_results/qwen/phase15/answers/qwen25vl3b_${run}_mcq.jsonl
  local extra=()
  if [ -n "$adapter" ]; then
    extra=(--adapter_path "$adapter")
  fi
  PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 "$PY" scripts/phase15/eval_mcq.py \
    --model_path Qwen/Qwen2.5-VL-3B-Instruct \
    --question_file "$Q" \
    --image_folder "$IMG" \
    --answers_file "$ans" \
    --gpu "$gpu" \
    --max_new_tokens 8 \
    "${extra[@]}" \
    2>&1 | tee "eval_results/qwen/phase15/logs/qwen25vl3b_${run}_mcq.log"
}

run_one "$GPU_B" base ""
run_one "$GPU_E" explicit_1k checkpoints/qwen25vl3b_explicit_1k
run_one "$GPU_T" typed_1k checkpoints/qwen25vl3b_typed_1k
