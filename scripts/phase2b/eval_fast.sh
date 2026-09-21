#!/bin/bash
set -euo pipefail
# Fast external gate: R-Bench-fast / MMRel-Adv-fast / AMBER-dr-fast.
# Usage: bash scripts/phase2b/eval_fast.sh <gpu> <run_name> [adapter_path]
if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/phase2b/eval_fast.sh <GPU_ID> <run_name> [adapter_path]"
  exit 1
fi
GPU_ID=$1
RUN=$2
ADAPTER=${3:-}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
MODEL=/home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
export HF_HOME=/home/lht/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p eval_results/qwen/phase2b/{answers,metrics,logs}

ADAPT_ARG=()
if [ -n "$ADAPTER" ]; then
  ADAPT_ARG=(--adapter_path "$ADAPTER")
fi

infer_score () {
  local bench=$1 q=$2 images=$3
  local ans=eval_results/qwen/phase2b/answers/${RUN}_${bench}_fast.jsonl
  local met=eval_results/qwen/phase2b/metrics/${RUN}_${bench}_fast.json
  /data/storage22t/lht/envs/qwen25vl/bin/python scripts/qwen_eval/infer_vqa.py \
    --model_path "$MODEL" \
    --question_file "$q" \
    --image_folder "$images" \
    --answers_file "$ans" \
    --gpu "$GPU_ID" \
    --max_new_tokens 16 \
    "${ADAPT_ARG[@]}" \
    2>&1 | tee "eval_results/qwen/phase2b/logs/${RUN}_${bench}_fast_infer.log"
  /data/storage22t/lht/envs/qwen25vl/bin/python scripts/phase2b/score_fast.py \
    --bench "$bench" \
    --result_file "$ans" \
    --out_json "$met"
}

infer_score rbench eval_results/qwen/phase2b/fast_subsets/rbench_questions.jsonl R-Bench/images
infer_score mmrel_adv eval_results/qwen/phase2b/fast_subsets/mmrel_adv_questions.jsonl MMRel
infer_score amber_dr eval_results/qwen/phase2b/fast_subsets/amber_dr_questions.jsonl AMBER/images
