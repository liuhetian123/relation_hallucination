#!/bin/bash
set -euo pipefail
# After training: Gate 1 held-out then Gate 2 frozen fast subsets.
# Usage: bash scripts/phase3a/eval_all.sh <GPU_ID>
if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/phase3a/eval_all.sh <GPU_ID>"
  exit 1
fi
GPU_ID=$1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

declare -A CKPT=(
  [s534]=checkpoints/qwen25vl3b_phase3a_s534
  [s1500]=checkpoints/qwen25vl3b_phase3a_s1500
  [s3000]=checkpoints/qwen25vl3b_phase3a_s3000
  [s3000_stepmatched]=checkpoints/qwen25vl3b_phase3a_s3000_stepmatched
)
declare -A TRAINJSON=(
  [s534]=data/phase3a/train_s534.json
  [s1500]=data/phase3a/train_s1500.json
  [s3000]=data/phase3a/train_s3000.json
  [s3000_stepmatched]=data/phase3a/train_s3000.json
)

for run in s534 s1500 s3000 s3000_stepmatched; do
  bash scripts/phase3a/eval_heldout.sh "$GPU_ID" "$run" "${CKPT[$run]}"
  bash scripts/phase3a/eval_train.sh "$GPU_ID" "$run" "${CKPT[$run]}" "${TRAINJSON[$run]}"
  bash scripts/phase3a/eval_fast.sh "$GPU_ID" "$run" "${CKPT[$run]}"
done

/data/storage22t/lht/envs/qwen25vl/bin/python scripts/phase3a/analyze.py
