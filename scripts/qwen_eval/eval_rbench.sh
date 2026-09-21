#!/bin/bash
set -euo pipefail
# Usage: bash scripts/qwen_eval/eval_rbench.sh <gpu> <run_name> [adapter_path]
# run_name: base | explicit_1k | typed_1k

if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/qwen_eval/eval_rbench.sh <GPU_ID> <run_name> [adapter_path]"
  exit 1
fi

GPU_ID=$1
RUN=$2
ADAPTER=${3:-}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

ANS=eval_results/qwen/rbench/answers/qwen25vl3b_${RUN}_image-level.jsonl
LOG=eval_results/qwen/rbench/logs/qwen25vl3b_${RUN}_image-level
mkdir -p eval_results/qwen/rbench/{answers,logs}

ADAPT_ARG=()
if [ -n "$ADAPTER" ]; then
  ADAPT_ARG=(--adapter_path "$ADAPTER")
fi

PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 /data/storage22t/lht/envs/qwen25vl/bin/python \
  scripts/qwen_eval/infer_vqa.py \
  --model_path /home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --question_file R-Bench/data_filterd/image-level_filterd_llava.jsonl \
  --image_folder R-Bench/images \
  --answers_file "$ANS" \
  --gpu "$GPU_ID" \
  --max_new_tokens 16 \
  "${ADAPT_ARG[@]}" \
  2>&1 | tee "${LOG}_infer.log"

/data/storage22t/lht/envs/qwen25vl/bin/python R-Bench/eval.py \
  --eval_image \
  --question-file R-Bench/data_filterd/image-level_filterd.json \
  --question-id-file R-Bench/data_filterd/nocaps_image-level_rel_ids_holder.json \
  --result-file "$ANS" \
  2>&1 | tee "${LOG}_metrics.log"
