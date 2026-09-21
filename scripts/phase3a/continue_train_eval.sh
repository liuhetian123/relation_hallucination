#!/bin/bash
set -euo pipefail
# Resume after master data + S534: remaining training, then Gate 1/2 eval.
# Usage: bash scripts/phase3a/continue_train_eval.sh [GPU_ID]
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=/data/storage22t/lht/envs/qwen25vl/bin/python
mkdir -p eval_results/qwen/phase3a/logs

if [ $# -ge 1 ]; then
  GPU_ID="$1"
else
  echo "[phase3a] waiting for an idle GPU for Qwen 3B LoRA (>=18GB free)"
  GPU_ID="$("$PY" scripts/phase3a/wait_gpu.py --min-free-mb 18000 --max-util 8 --poll-sec 60)"
fi
echo "[phase3a] continue GPU=$GPU_ID"
bash scripts/phase3a/train.sh "$GPU_ID"
bash scripts/phase3a/eval_all.sh "$GPU_ID"
echo "[phase3a] continue done GPU=$GPU_ID"
