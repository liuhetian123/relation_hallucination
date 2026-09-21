#!/bin/bash
set -euo pipefail
# Usage: bash scripts/phase2/eval_benchmarks.sh <gpu> <run_name> [adapter_path]
if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/phase2/eval_benchmarks.sh <GPU_ID> <run_name> [adapter_path]"
  exit 1
fi
GPU_ID=$1
RUN=$2
ADAPTER=${3:-}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Qwen2.5-VL-3B lives in the default HF hub cache, not the InternVL HF_HOME.
export HF_HOME=/home/lht/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

run_one () {
  local script=$1
  shift
  if [ -n "$ADAPTER" ]; then
    bash "$script" "$GPU_ID" "$RUN" "$@" "$ADAPTER"
  else
    bash "$script" "$GPU_ID" "$RUN" "$@"
  fi
}

run_one scripts/qwen_eval/eval_rbench.sh
run_one scripts/qwen_eval/eval_mmrel.sh adv
run_one scripts/qwen_eval/eval_mmrel.sh dalle_normal
run_one scripts/qwen_eval/eval_amber_dr.sh
run_one scripts/qwen_eval/eval_pope_adv.sh
