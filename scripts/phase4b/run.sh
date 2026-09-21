#!/bin/bash
set -euo pipefail
# Phase 4B: frozen stratified analysis + InternVL falsity recheck on 100 pairs.
# Usage: bash scripts/phase4b/run.sh [GPU_ID]
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SPACY_PY=/home/lht/miniconda3/envs/relhallu/bin/python
QWEN_PY=/data/storage22t/lht/envs/qwen25vl/bin/python
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
mkdir -p eval_results/qwen/phase4b/{metrics,logs} data/phase4a/qc

echo "[phase4b] extract relations (spaCy via relhallu, read-only)"
"$SPACY_PY" scripts/phase4b/extract_relations.py \
  2>&1 | tee eval_results/qwen/phase4b/logs/extract_relations.log

echo "[phase4b] stratified analysis"
"$QWEN_PY" scripts/phase4b/analyze.py \
  2>&1 | tee eval_results/qwen/phase4b/logs/analyze.log

echo "[phase4b] sample 100 + HTML"
"$QWEN_PY" scripts/phase4b/make_falsity_qc.py \
  2>&1 | tee eval_results/qwen/phase4b/logs/make_falsity_qc.log

if [ $# -ge 1 ]; then
  GPU_ID="$1"
else
  echo "[phase4b] waiting for idle GPU for InternVL recheck"
  GPU_ID="$("$QWEN_PY" scripts/phase3a/wait_gpu.py --min-free-mb 24000 --max-util 8 --poll-sec 60 --n 1)"
fi
echo "[phase4b] InternVL falsity GPU=$GPU_ID"
"$QWEN_PY" scripts/phase4b/falsity_recheck.py --gpu "$GPU_ID" \
  2>&1 | tee eval_results/qwen/phase4b/logs/falsity_recheck.log

echo "[phase4b] report"
"$QWEN_PY" scripts/phase4b/report.py \
  2>&1 | tee eval_results/qwen/phase4b/logs/report.log
echo "[phase4b] done"
