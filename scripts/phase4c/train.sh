#!/bin/bash
set -euo pipefail
# Train Phase 4C multi-format LoRA. Same hparams as Phase 4A clean.
# Usage: bash scripts/phase4c/train.sh <GPU_ID>
if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/phase4c/train.sh <GPU_ID>"
  exit 1
fi
GPU_ID=$1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=/data/storage22t/lht/envs/qwen25vl/bin/python
MODEL=/home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
IMG=/data/lht/relsim_dataset/relsim_images
DATA=data/phase4c/train_mf.json
OUT=checkpoints/qwen25vl3b_phase4c_mf
mkdir -p "$OUT" eval_results/qwen/phase4c/logs

if [ -f "$OUT/adapter_model.safetensors" ]; then
  echo "[phase4c] skip existing $OUT"
  exit 0
fi

echo "[phase4c] train $OUT from $DATA gpu=$GPU_ID"
PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 "$PY" \
  scripts/qwen_train/finetune_lora_relsim.py \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --image_root "$IMG" \
  --output_dir "$OUT" \
  --gpu "$GPU_ID" \
  --seed 42 \
  --num_train_epochs 1 \
  --save_strategy epoch \
  --save_total_limit 2 \
  2>&1 | tee "eval_results/qwen/phase4c/logs/train_mf.log"
echo "[phase4c] done $OUT"
