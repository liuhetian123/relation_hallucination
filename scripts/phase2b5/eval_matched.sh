#!/bin/bash
set -euo pipefail
# Usage: bash scripts/phase2b5/eval_matched.sh <gpu> <run_name> [adapter_path]
if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/phase2b5/eval_matched.sh <GPU_ID> <run_name> [adapter_path]"
  exit 1
fi
GPU_ID=$1
RUN=$2
ADAPTER=${3:-}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
MODEL=/home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
export HF_HOME=/home/lht/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p eval_results/qwen/phase2b5/{answers,logs}

ADAPT_ARG=()
if [ -n "$ADAPTER" ]; then
  ADAPT_ARG=(--adapter_path "$ADAPTER")
fi

infer_one () {
  local name=$1 q=$2 images=$3
  local ans=eval_results/qwen/phase2b5/answers/${RUN}_${name}_matched.jsonl
  /data/storage22t/lht/envs/qwen25vl/bin/python scripts/qwen_eval/infer_vqa.py \
    --model_path "$MODEL" \
    --question_file "$q" \
    --image_folder "$images" \
    --answers_file "$ans" \
    --gpu "$GPU_ID" \
    --max_new_tokens 16 \
    "${ADAPT_ARG[@]}" \
    2>&1 | tee "eval_results/qwen/phase2b5/logs/${RUN}_${name}_matched_infer.log"
}

infer_one rbench data/phase2b5/rbench_fast_matched_prompt.jsonl R-Bench/images
infer_one mmrel_adv data/phase2b5/mmrel_adv_fast_matched_prompt.jsonl MMRel
