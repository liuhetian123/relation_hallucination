#!/bin/bash
set -euo pipefail
# Phase 4C: build multi-format data → train → eval → report.
# Usage: bash scripts/phase4c/run.sh [GPU_ID]
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=/data/storage22t/lht/envs/qwen25vl/bin/python
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
mkdir -p eval_results/qwen/phase4c/{logs,metrics,answers,questions} data/phase4c checkpoints

echo "[phase4c] build multi-format train JSON"
"$PY" scripts/phase4c/build_mf_train.py \
  2>&1 | tee eval_results/qwen/phase4c/logs/build_mf.log

if [ $# -ge 1 ]; then
  GPU_ID="$1"
else
  echo "[phase4c] waiting for an idle GPU (>=24GB free, util<=8%)"
  GPU_ID="$("$PY" scripts/phase3a/wait_gpu.py --min-free-mb 24000 --max-util 8 --poll-sec 60 --n 1)"
fi
echo "[phase4c] using GPU $GPU_ID"

bash scripts/phase4c/train.sh "$GPU_ID"
bash scripts/phase4c/eval_all.sh "$GPU_ID"
echo "[phase4c] pipeline done"
