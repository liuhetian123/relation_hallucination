#!/bin/bash
set -euo pipefail
# Usage: bash scripts/phase2/eval_heldout.sh <gpu> <run_name> [adapter_path]
if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/phase2/eval_heldout.sh <GPU_ID> <run_name> [adapter_path]"
  exit 1
fi
GPU_ID=$1
RUN=$2
ADAPTER=${3:-}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
MODEL=/home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
Q=eval_results/qwen/phase2/heldout_verification.jsonl
ANS=eval_results/qwen/phase2/answers/${RUN}_heldout.jsonl
MET=eval_results/qwen/phase2/metrics/${RUN}_heldout.json
mkdir -p eval_results/qwen/phase2/{answers,metrics,logs}

ADAPT_ARG=()
if [ -n "$ADAPTER" ]; then
  ADAPT_ARG=(--adapter_path "$ADAPTER")
fi

PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 /data/storage22t/lht/envs/qwen25vl/bin/python \
  scripts/qwen_eval/infer_vqa.py \
  --model_path "$MODEL" \
  --question_file "$Q" \
  --image_folder /data/lht/relsim_dataset/relsim_images \
  --answers_file "$ANS" \
  --gpu "$GPU_ID" \
  --max_new_tokens 8 \
  "${ADAPT_ARG[@]}" \
  2>&1 | tee "eval_results/qwen/phase2/logs/${RUN}_heldout_infer.log"

/data/storage22t/lht/envs/qwen25vl/bin/python scripts/phase2/score_verification.py \
  --question_file "$Q" \
  --result_file "$ANS" \
  --out_json "$MET"
