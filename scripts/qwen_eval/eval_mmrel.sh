#!/bin/bash
set -euo pipefail
# Usage: bash scripts/qwen_eval/eval_mmrel.sh <gpu> <run_name> <adv|dalle_normal> [adapter_path]

if [ $# -lt 3 ]; then
  echo "Usage: bash scripts/qwen_eval/eval_mmrel.sh <GPU_ID> <run_name> <adv|dalle_normal> [adapter_path]"
  exit 1
fi

GPU_ID=$1
RUN=$2
SPLIT=$3
ADAPTER=${4:-}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ "$SPLIT" = "adv" ]; then
  Q=eval_results/qwen/mmrel/questions/mmrel_adversarial.jsonl
elif [ "$SPLIT" = "dalle_normal" ]; then
  Q=eval_results/qwen/mmrel/questions/mmrel_dalle_normal.jsonl
else
  echo "SPLIT must be adv or dalle_normal"
  exit 1
fi

ANS=eval_results/qwen/mmrel/answers/qwen25vl3b_${RUN}_mmrel_${SPLIT}.jsonl
MET=eval_results/qwen/mmrel/logs/qwen25vl3b_${RUN}_mmrel_${SPLIT}_metrics.json
mkdir -p eval_results/qwen/mmrel/{answers,logs}

ADAPT_ARG=()
if [ -n "$ADAPTER" ]; then
  ADAPT_ARG=(--adapter_path "$ADAPTER")
fi

PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 /data/storage22t/lht/envs/qwen25vl/bin/python \
  scripts/qwen_eval/infer_vqa.py \
  --model_path /home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --question_file "$Q" \
  --image_folder MMRel \
  --answers_file "$ANS" \
  --gpu "$GPU_ID" \
  --max_new_tokens 16 \
  "${ADAPT_ARG[@]}" \
  2>&1 | tee eval_results/qwen/mmrel/logs/qwen25vl3b_${RUN}_mmrel_${SPLIT}_infer.log

/data/storage22t/lht/envs/qwen25vl/bin/python scripts/qwen_eval/score_yesno.py \
  --question_file "$Q" \
  --result_file "$ANS" \
  --out_json "$MET"
