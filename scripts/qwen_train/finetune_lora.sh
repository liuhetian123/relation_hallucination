#!/bin/bash
set -euo pipefail
# Matched Qwen2.5-VL-3B LoRA SFT. Do not set CUDA_VISIBLE_DEVICES; pass GPU id as $1.
# Usage: bash scripts/qwen_train/finetune_lora.sh <gpu> <explicit|typed>

if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/qwen_train/finetune_lora.sh <GPU_ID> <explicit|typed>"
  exit 1
fi

GPU_ID=$1
SPLIT=$2
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ "$SPLIT" = "explicit" ]; then
  DATA=data/phase1/qwen_explicit_1k.json
  OUT=checkpoints/qwen25vl3b_explicit_1k
elif [ "$SPLIT" = "typed" ]; then
  DATA=data/phase1/qwen_typed_1k.json
  OUT=checkpoints/qwen25vl3b_typed_1k
else
  echo "SPLIT must be explicit or typed"
  exit 1
fi

mkdir -p "$OUT" eval_results/qwen/logs
PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 /data/storage22t/lht/envs/qwen25vl/bin/python \
  scripts/qwen_train/finetune_lora_relsim.py \
  --model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --data_path "$DATA" \
  --image_root data/relsim_images \
  --output_dir "$OUT" \
  --gpu "$GPU_ID" \
  --seed 42 \
  2>&1 | tee "eval_results/qwen/logs/train_${SPLIT}_1k.log"
