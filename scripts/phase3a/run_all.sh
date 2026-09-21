#!/bin/bash
set -euo pipefail
# Full Phase 3A: data (InternVL) → train → eval → report.
# Usage: bash scripts/phase3a/run_all.sh [GPU_ID]
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=/data/storage22t/lht/envs/qwen25vl/bin/python
mkdir -p eval_results/qwen/phase3a/logs

if [ $# -ge 1 ]; then
  GPU_ID="$1"
else
  echo "[phase3a] waiting for an idle GPU"
  GPU_ID="$("$PY" scripts/phase3a/wait_gpu.py --min-free-mb 24000 --max-util 8 --poll-sec 60)"
fi
echo "[phase3a] pipeline GPU=$GPU_ID"
bash scripts/phase3a/run_data.sh "$GPU_ID"
bash scripts/phase3a/train.sh "$GPU_ID"
bash scripts/phase3a/eval_all.sh "$GPU_ID"
echo "[phase3a] done GPU=$GPU_ID"
