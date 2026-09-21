#!/bin/bash
set -euo pipefail
# Phase 4A Old vs Clean sanity-check: build data → train → eval → report.
# Usage: bash scripts/phase4a/run_train.sh [GPU_OLD] [GPU_CLEAN]
# If GPUs omitted, wait for idle cards (>=24GB free, util<=8%).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=/data/storage22t/lht/envs/qwen25vl/bin/python
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
mkdir -p eval_results/qwen/phase4a/logs checkpoints data/phase4a

echo "[phase4a] build matched train JSON"
"$PY" scripts/phase4a/build_train.py \
  2>&1 | tee -a eval_results/qwen/phase4a/logs/build_train.log

if [ $# -ge 2 ]; then
  GPU_OLD="$1"
  GPU_CLEAN="$2"
elif [ $# -eq 1 ]; then
  GPU_OLD="$1"
  GPU_CLEAN="$1"
else
  echo "[phase4a] trying two idle GPUs"
  GPUS="$("$PY" scripts/phase3a/wait_gpu.py --min-free-mb 24000 --max-util 8 --once --n 2)" || GPUS=""
  if [ "$(echo ${GPUS:-} | wc -w)" -ge 2 ]; then
    GPU_OLD="$(echo "$GPUS" | awk '{print $1}')"
    GPU_CLEAN="$(echo "$GPUS" | awk '{print $2}')"
  else
    echo "[phase4a] waiting for one idle GPU"
    GPU_OLD="$("$PY" scripts/phase3a/wait_gpu.py --min-free-mb 24000 --max-util 8 --poll-sec 60 --n 1)"
    GPU_CLEAN="$GPU_OLD"
  fi
fi
echo "[phase4a] train GPUs old=$GPU_OLD clean=$GPU_CLEAN"

if [ "$GPU_OLD" = "$GPU_CLEAN" ]; then
  bash scripts/phase4a/train.sh "$GPU_OLD" old
  bash scripts/phase4a/train.sh "$GPU_CLEAN" clean
  EVAL_GPU="$GPU_OLD"
else
  bash scripts/phase4a/train.sh "$GPU_OLD" old &
  PID_OLD=$!
  bash scripts/phase4a/train.sh "$GPU_CLEAN" clean &
  PID_CLEAN=$!
  wait "$PID_OLD"
  wait "$PID_CLEAN"
  EVAL_GPU="$GPU_OLD"
fi

echo "[phase4a] eval on GPU $EVAL_GPU"
bash scripts/phase4a/eval_all.sh "$EVAL_GPU"
echo "[phase4a] train pipeline done"
