#!/bin/bash
set -euo pipefail
# Usage: bash scripts/qwen_eval/eval_pope_adv.sh <gpu> <run_name> [adapter_path]

if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/qwen_eval/eval_pope_adv.sh <GPU_ID> <run_name> [adapter_path]"
  exit 1
fi

GPU_ID=$1
RUN=$2
ADAPTER=${3:-}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

ANS=eval_results/qwen/pope/answers/qwen25vl3b_${RUN}_pope_adv.jsonl
MET=eval_results/qwen/pope/logs/qwen25vl3b_${RUN}_pope_adv_metrics.json
mkdir -p eval_results/qwen/pope/{answers,logs}

ADAPT_ARG=()
if [ -n "$ADAPTER" ]; then
  ADAPT_ARG=(--adapter_path "$ADAPTER")
fi

PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 /data/storage22t/lht/envs/qwen25vl/bin/python \
  scripts/qwen_eval/infer_vqa.py \
  --model_path /home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --question_file eval_results/pope/questions/pope_adversarial.jsonl \
  --image_folder LLaVA/playground/data/eval/pope/val2014 \
  --answers_file "$ANS" \
  --gpu "$GPU_ID" \
  --max_new_tokens 16 \
  "${ADAPT_ARG[@]}" \
  2>&1 | tee eval_results/qwen/pope/logs/qwen25vl3b_${RUN}_pope_adv_infer.log

/data/storage22t/lht/envs/qwen25vl/bin/python scripts/qwen_eval/score_yesno.py \
  --question_file eval_results/pope/questions/pope_adversarial.jsonl \
  --result_file "$ANS" \
  --out_json "$MET"
