#!/bin/bash
set -euo pipefail
# Build Phase 3A master data. InternVL waits for a free GPU.
# Usage: bash scripts/phase3a/run_data.sh [GPU_ID]
# If GPU_ID is omitted, wait_gpu.py polls until a card is idle.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=/data/storage22t/lht/envs/qwen25vl/bin/python
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
mkdir -p data/phase3a/qc eval_results/qwen/phase3a/logs

if [ ! -f data/phase3a/candidates.jsonl ]; then
  echo "[phase3a] building candidate pool"
  "$PY" scripts/phase3a/build_candidate_pool.py \
    2>&1 | tee eval_results/qwen/phase3a/logs/build_candidates.log
fi

if [ $# -ge 1 ]; then
  GPU_ID="$1"
else
  echo "[phase3a] waiting for an idle GPU (>=24GB free, util<=8%)"
  GPU_ID="$("$PY" scripts/phase3a/wait_gpu.py --min-free-mb 24000 --max-util 8 --poll-sec 60)"
fi
echo "[phase3a] using GPU $GPU_ID"

TARGET_OK=5000
MAX_CAND=12000
ROUND=0
while true; do
  ROUND=$((ROUND + 1))
  echo "[phase3a] round $ROUND annotate target_ok=$TARGET_OK"
  "$PY" scripts/phase3a/annotate_sro.py \
    --gpu "$GPU_ID" \
    --target-ok "$TARGET_OK" \
    --max-candidates "$MAX_CAND" \
    2>&1 | tee -a eval_results/qwen/phase3a/logs/annotate_sro.log

  "$PY" scripts/phase3a/qc_sro.py \
    2>&1 | tee -a eval_results/qwen/phase3a/logs/qc_sro.log

  "$PY" scripts/phase3a/build_negative_jobs.py \
    2>&1 | tee -a eval_results/qwen/phase3a/logs/build_negative_jobs.log

  "$PY" scripts/phase3a/filter_negatives.py \
    --gpu "$GPU_ID" \
    2>&1 | tee -a eval_results/qwen/phase3a/logs/filter_negatives.log

  set +e
  "$PY" scripts/phase3a/assemble_master.py \
    2>&1 | tee -a eval_results/qwen/phase3a/logs/assemble.log
  RC=$?
  set -e
  if [ "$RC" -eq 0 ]; then
    echo "[phase3a] master dataset ready"
    break
  fi
  if [ "$RC" -ne 2 ]; then
    echo "[phase3a] assemble failed rc=$RC"
    exit "$RC"
  fi
  if [ "$ROUND" -ge 4 ]; then
    echo "[phase3a] still short of 3000 after $ROUND rounds"
    exit 2
  fi
  TARGET_OK=$((TARGET_OK + 1500))
  echo "[phase3a] need more pairs; raising target_ok to $TARGET_OK"
done
