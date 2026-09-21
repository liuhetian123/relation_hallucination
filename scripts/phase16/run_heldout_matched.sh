#!/bin/bash
set -euo pipefail
# Usage: bash scripts/phase16/run_heldout_matched.sh [gpu]
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=/data/storage22t/lht/envs/qwen25vl/bin/python
GPU=${1:-3}
MODEL=/home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

mkdir -p eval_results/qwen/phase16b/scores eval_results/qwen/phase16b/logs

"$PY" scripts/phase16/pack_heldout_matched.py

run_one () {
  local run=$1 adapter=$2
  local extra=()
  if [ -n "$adapter" ]; then
    extra=(--adapter_path "$adapter")
  fi
  "$PY" scripts/phase16/score_heldout_matched.py \
    --model_path "$MODEL" \
    --run_name "$run" \
    --gpu "$GPU" \
    "${extra[@]}" \
    2>&1 | tee "eval_results/qwen/phase16b/logs/${run}_nll.log"
}

run_one base ""
run_one explicit_1k checkpoints/qwen25vl3b_explicit_1k
run_one typed_1k checkpoints/qwen25vl3b_typed_1k

"$PY" scripts/phase16/analyze_16b.py
