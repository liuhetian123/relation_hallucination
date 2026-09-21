#!/bin/bash
set -euo pipefail
# Phase 4A-Data: InternVL clean counterfactual negatives.
# Usage: bash scripts/phase4a/run_data.sh [GPU_ID]
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=/data/storage22t/lht/envs/qwen25vl/bin/python
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
mkdir -p data/phase4a/qc eval_results/qwen/phase4a/logs

if [ $# -ge 1 ]; then
  GPU_ID="$1"
else
  echo "[phase4a] waiting for an idle GPU (>=24GB free, util<=8%)"
  GPU_ID="$("$PY" scripts/phase3a/wait_gpu.py --min-free-mb 24000 --max-util 8 --poll-sec 60)"
fi
echo "[phase4a] using GPU $GPU_ID"

"$PY" scripts/phase4a/generate_and_filter.py \
  --gpu "$GPU_ID" \
  2>&1 | tee -a eval_results/qwen/phase4a/logs/generate_and_filter.log

"$PY" scripts/phase4a/assemble.py \
  2>&1 | tee -a eval_results/qwen/phase4a/logs/assemble.log

"$PY" scripts/phase4a/text_only_shortcut.py \
  --gpu "$GPU_ID" \
  2>&1 | tee -a eval_results/qwen/phase4a/logs/text_only_shortcut.log

"$PY" scripts/phase4a/report.py \
  2>&1 | tee -a eval_results/qwen/phase4a/logs/report.log

echo "[phase4a] data pipeline done GPU=$GPU_ID"
