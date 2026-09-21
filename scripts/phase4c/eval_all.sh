#!/bin/bash
set -euo pipefail
# After training: train self-check, frozen held-out F1, frozen fast, F2 held-out, report.
# Usage: bash scripts/phase4c/eval_all.sh <GPU_ID>
if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/phase4c/eval_all.sh <GPU_ID>"
  exit 1
fi
GPU_ID=$1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
MF=checkpoints/qwen25vl3b_phase4c_mf
CLEAN=checkpoints/qwen25vl3b_phase4a_clean

bash scripts/phase4c/eval_train.sh "$GPU_ID" mf "$MF" data/phase4c/train_mf.json
bash scripts/phase4c/eval_heldout.sh "$GPU_ID" mf "$MF"
bash scripts/phase4c/eval_fast.sh "$GPU_ID" mf "$MF"
bash scripts/phase4c/eval_heldout_f2.sh "$GPU_ID" mf "$MF"
bash scripts/phase4c/eval_heldout_f2.sh "$GPU_ID" clean "$CLEAN"

/data/storage22t/lht/envs/qwen25vl/bin/python scripts/phase4c/analyze.py
echo "[phase4c] eval pipeline done"
