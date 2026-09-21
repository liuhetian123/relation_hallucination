#!/bin/bash
set -euo pipefail
# Parallel Phase 4A InternVL generate on multiple GPUs, then merge + QC.
# Usage: bash scripts/phase4a/run_parallel.sh [gpu,gpu,...]
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=/data/storage22t/lht/envs/qwen25vl/bin/python
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
mkdir -p data/phase4a/shards data/phase4a/qc eval_results/qwen/phase4a/logs

IFS=',' read -r -a GPUS <<< "${1:-1,4,5,6,7}"
N=${#GPUS[@]}
echo "[phase4a] parallel generate GPUs=${GPUS[*]} shards=$N"

PIDS=()
i=0
for gpu in "${GPUS[@]}"; do
  log="eval_results/qwen/phase4a/logs/generate_shard${i}_gpu${gpu}.log"
  echo "[phase4a] shard $i/$N -> GPU $gpu log=$log"
  "$PY" scripts/phase4a/generate_and_filter.py \
    --gpu "$gpu" \
    --shard "$i" \
    --shards "$N" \
    --skip-jsonl data/phase4a/clean_attempts.jsonl \
    --output "data/phase4a/shards/attempts.shard${i}.gpu${gpu}.jsonl" \
    > "$log" 2>&1 &
  PIDS+=($!)
  i=$((i + 1))
done
echo "[phase4a] generate pids=${PIDS[*]}"

fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "[phase4a] pid $pid failed"
    fail=1
  fi
done
if [ "$fail" -ne 0 ]; then
  echo "[phase4a] one or more shards failed"
  exit 1
fi

echo "[phase4a] merging shards"
"$PY" scripts/phase4a/merge_shards.py \
  2>&1 | tee -a eval_results/qwen/phase4a/logs/merge.log

"$PY" scripts/phase4a/assemble.py \
  2>&1 | tee -a eval_results/qwen/phase4a/logs/assemble.log

MERGE_GPU="${GPUS[0]}"
echo "[phase4a] text-only shortcut on GPU $MERGE_GPU"
"$PY" scripts/phase4a/text_only_shortcut.py \
  --gpu "$MERGE_GPU" \
  2>&1 | tee -a eval_results/qwen/phase4a/logs/text_only_shortcut.log

"$PY" scripts/phase4a/report.py \
  2>&1 | tee -a eval_results/qwen/phase4a/logs/report.log

echo "[phase4a] parallel pipeline done"
