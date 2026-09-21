#!/bin/bash
set -euo pipefail
# Usage: bash scripts/qwen_eval/eval_amber_dr.sh <gpu> <run_name> [adapter_path]

if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/qwen_eval/eval_amber_dr.sh <GPU_ID> <run_name> [adapter_path]"
  exit 1
fi

GPU_ID=$1
RUN=$2
ADAPTER=${3:-}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LLAVA_Q=eval_results/amber/questions/amber_dr_llava.jsonl
ANS=eval_results/qwen/amber/answers/qwen25vl3b_${RUN}_amber_dr.jsonl
AMBER_JSON=eval_results/qwen/amber/answers/qwen25vl3b_${RUN}_amber_dr.json
mkdir -p eval_results/qwen/amber/{answers,logs}

ADAPT_ARG=()
if [ -n "$ADAPTER" ]; then
  ADAPT_ARG=(--adapter_path "$ADAPTER")
fi

PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 /data/storage22t/lht/envs/qwen25vl/bin/python \
  scripts/qwen_eval/infer_vqa.py \
  --model_path /home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --question_file "$LLAVA_Q" \
  --image_folder AMBER/images \
  --answers_file "$ANS" \
  --gpu "$GPU_ID" \
  --max_new_tokens 16 \
  "${ADAPT_ARG[@]}" \
  2>&1 | tee eval_results/qwen/amber/logs/qwen25vl3b_${RUN}_amber_dr_infer.log

/data/storage22t/lht/envs/qwen25vl/bin/python scripts/convert/convert_llava_answer_to_amber.py \
  --llava-answer "$ANS" \
  --out-file "$AMBER_JSON"

METRICS_SRC=AMBER/data/metrics.txt
METRICS_RUN="$ROOT/eval_results/qwen/amber/logs/qwen25vl3b_${RUN}_amber_dr_metrics.txt"
cp "$METRICS_SRC" "$METRICS_RUN"

(
  cd AMBER
  /home/lht/miniconda3/envs/relhallu/bin/python inference.py \
    --inference_data "../$AMBER_JSON" \
    --evaluation_type dr \
    --metrics "$METRICS_RUN" \
    --annotation data/annotations.json \
    --word_association data/relation.json \
    --safe_words data/safe_words.txt
) 2>&1 | tee eval_results/qwen/amber/logs/qwen25vl3b_${RUN}_amber_dr_metrics.log
