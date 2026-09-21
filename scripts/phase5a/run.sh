#!/bin/bash
set -euo pipefail
# Phase 5A: text-only prior measurement + stratified analysis. No training.
# Usage: bash scripts/phase5a/run.sh [GPU_ID]
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=/data/storage22t/lht/envs/qwen25vl/bin/python
MODEL=/home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_HOME=/home/lht/.cache/huggingface
mkdir -p eval_results/qwen/phase5a/{answers,metrics,logs,jobs}

if [ $# -ge 1 ]; then
  GPU_ID="$1"
else
  GPU_ID="$("$PY" scripts/phase3a/wait_gpu.py --min-free-mb 10000 --max-util 90 --once --n 1)" || GPU_ID=""
  if [ -z "$GPU_ID" ]; then
    GPU_ID="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
  fi
fi
echo "[phase5a] using GPU $GPU_ID"

cat > eval_results/qwen/phase5a/jobs/base.json <<'EOF'
[
  {"question_file": "eval_results/qwen/phase2b/fast_subsets/mmrel_adv_questions.jsonl", "answers_file": "eval_results/qwen/phase5a/answers/base_mmrel_adv_fast_textonly.jsonl"},
  {"question_file": "eval_results/qwen/phase2b/fast_subsets/rbench_questions.jsonl", "answers_file": "eval_results/qwen/phase5a/answers/base_rbench_fast_textonly.jsonl"},
  {"question_file": "eval_results/qwen/phase2b/fast_subsets/amber_dr_questions.jsonl", "answers_file": "eval_results/qwen/phase5a/answers/base_amber_dr_fast_textonly.jsonl"},
  {"question_file": "data/phase2/heldout_verification.jsonl", "answers_file": "eval_results/qwen/phase5a/answers/base_heldout_textonly.jsonl"}
]
EOF

cat > eval_results/qwen/phase5a/jobs/clean_mmrel.json <<'EOF'
[
  {"question_file": "eval_results/qwen/phase2b/fast_subsets/mmrel_adv_questions.jsonl", "answers_file": "eval_results/qwen/phase5a/answers/clean_mmrel_adv_fast_textonly.jsonl"}
]
EOF

cat > eval_results/qwen/phase5a/jobs/mf_mmrel.json <<'EOF'
[
  {"question_file": "eval_results/qwen/phase2b/fast_subsets/mmrel_adv_questions.jsonl", "answers_file": "eval_results/qwen/phase5a/answers/mf_mmrel_adv_fast_textonly.jsonl"}
]
EOF

infer_jobs () {
  local name=$1 jobs=$2 extra=()
  shift 2
  extra=("$@")
  echo "[phase5a] infer $name"
  "$PY" scripts/phase5a/infer_text_only.py \
    --model_path "$MODEL" \
    --jobs "$jobs" \
    --gpu "$GPU_ID" \
    --max_new_tokens 16 \
    "${extra[@]}" \
    2>&1 | tee "eval_results/qwen/phase5a/logs/${name}_infer.log"
}

infer_jobs base eval_results/qwen/phase5a/jobs/base.json
infer_jobs clean_mmrel eval_results/qwen/phase5a/jobs/clean_mmrel.json --adapter_path checkpoints/qwen25vl3b_phase4a_clean
infer_jobs mf_mmrel eval_results/qwen/phase5a/jobs/mf_mmrel.json --adapter_path checkpoints/qwen25vl3b_phase4c_mf

echo "[phase5a] validity check / maybe fallback"
NEED_FB="$("$PY" scripts/phase5a/analyze_prior.py --validity-only || true)"
if echo "$NEED_FB" | grep -q "NEED_FALLBACK"; then
  echo "[phase5a] invalid rate >10%, rerunning with fallback prompt"
  cat > eval_results/qwen/phase5a/jobs/base_fallback.json <<'EOF'
[
  {"question_file": "eval_results/qwen/phase2b/fast_subsets/mmrel_adv_questions.jsonl", "answers_file": "eval_results/qwen/phase5a/answers/base_mmrel_adv_fast_textonly_fallback.jsonl"},
  {"question_file": "eval_results/qwen/phase2b/fast_subsets/rbench_questions.jsonl", "answers_file": "eval_results/qwen/phase5a/answers/base_rbench_fast_textonly_fallback.jsonl"},
  {"question_file": "eval_results/qwen/phase2b/fast_subsets/amber_dr_questions.jsonl", "answers_file": "eval_results/qwen/phase5a/answers/base_amber_dr_fast_textonly_fallback.jsonl"},
  {"question_file": "data/phase2/heldout_verification.jsonl", "answers_file": "eval_results/qwen/phase5a/answers/base_heldout_textonly_fallback.jsonl"}
]
EOF
  infer_jobs base_fallback eval_results/qwen/phase5a/jobs/base_fallback.json --fallback
  cat > eval_results/qwen/phase5a/jobs/clean_mmrel_fallback.json <<'EOF'
[
  {"question_file": "eval_results/qwen/phase2b/fast_subsets/mmrel_adv_questions.jsonl", "answers_file": "eval_results/qwen/phase5a/answers/clean_mmrel_adv_fast_textonly_fallback.jsonl"}
]
EOF
  cat > eval_results/qwen/phase5a/jobs/mf_mmrel_fallback.json <<'EOF'
[
  {"question_file": "eval_results/qwen/phase2b/fast_subsets/mmrel_adv_questions.jsonl", "answers_file": "eval_results/qwen/phase5a/answers/mf_mmrel_adv_fast_textonly_fallback.jsonl"}
]
EOF
  infer_jobs clean_mmrel_fallback eval_results/qwen/phase5a/jobs/clean_mmrel_fallback.json --adapter_path checkpoints/qwen25vl3b_phase4a_clean --fallback
  infer_jobs mf_mmrel_fallback eval_results/qwen/phase5a/jobs/mf_mmrel_fallback.json --adapter_path checkpoints/qwen25vl3b_phase4c_mf --fallback
fi

echo "[phase5a] analyze + report"
"$PY" scripts/phase5a/analyze_prior.py \
  2>&1 | tee eval_results/qwen/phase5a/logs/analyze.log
echo "[phase5a] done"
