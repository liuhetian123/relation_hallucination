#!/bin/bash
set -euo pipefail
# After training: held-out, train self-check, frozen fast subsets, then report.
# Usage: bash scripts/phase4a/eval_all.sh <GPU_ID>
if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/phase4a/eval_all.sh <GPU_ID>"
  exit 1
fi
GPU_ID=$1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

declare -A CKPT=(
  [old]=checkpoints/qwen25vl3b_phase4a_old
  [clean]=checkpoints/qwen25vl3b_phase4a_clean
)
declare -A TRAINJSON=(
  [old]=data/phase4a/train_old.json
  [clean]=data/phase4a/train_clean.json
)

for run in old clean; do
  bash scripts/phase4a/eval_heldout.sh "$GPU_ID" "$run" "${CKPT[$run]}"
  bash scripts/phase4a/eval_train.sh "$GPU_ID" "$run" "${CKPT[$run]}" "${TRAINJSON[$run]}"
  bash scripts/phase4a/eval_fast.sh "$GPU_ID" "$run" "${CKPT[$run]}"
done

/data/storage22t/lht/envs/qwen25vl/bin/python scripts/phase4a/analyze.py
