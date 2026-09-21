#!/bin/bash
set -euo pipefail
# Extra F2 rendering of the frozen 393 held-out statements.
# Usage: bash scripts/phase4c/eval_heldout_f2.sh <GPU_ID> <run_name> [adapter_path]
if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/phase4c/eval_heldout_f2.sh <GPU_ID> <run_name> [adapter_path]"
  exit 1
fi
GPU_ID=$1
RUN=$2
ADAPTER=${3:-}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
MODEL=/home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
Q=data/phase4c/heldout_f2.jsonl
ANS=eval_results/qwen/phase4c/answers/${RUN}_heldout_f2.jsonl
MET=eval_results/qwen/phase4c/metrics/${RUN}_heldout_f2.json
mkdir -p eval_results/qwen/phase4c/{answers,metrics,logs}

if [ ! -f "$Q" ]; then
  echo "[phase4c] missing $Q; run scripts/phase4c/build_mf_train.py first"
  exit 1
fi

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
  2>&1 | tee "eval_results/qwen/phase4c/logs/${RUN}_heldout_f2_infer.log"

/data/storage22t/lht/envs/qwen25vl/bin/python scripts/phase2/score_verification.py \
  --question_file "$Q" \
  --result_file "$ANS" \
  --out_json "$MET"
