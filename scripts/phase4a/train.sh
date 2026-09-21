#!/bin/bash
set -euo pipefail
# Train one Phase 4A LoRA: old | clean.
# Usage: bash scripts/phase4a/train.sh <GPU_ID> <old|clean>
if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/phase4a/train.sh <GPU_ID> <old|clean>"
  exit 1
fi
GPU_ID=$1
RUN=$2
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=/data/storage22t/lht/envs/qwen25vl/bin/python
MODEL=/home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
IMG=/data/lht/relsim_dataset/relsim_images

case "$RUN" in
  old) DATA=data/phase4a/train_old.json ;;
  clean) DATA=data/phase4a/train_clean.json ;;
  *) echo "RUN must be old or clean"; exit 1 ;;
esac
OUT=checkpoints/qwen25vl3b_phase4a_${RUN}
mkdir -p "$OUT" eval_results/qwen/phase4a/logs

if [ -f "$OUT/adapter_model.safetensors" ]; then
  echo "[phase4a] skip existing $OUT"
  exit 0
fi

echo "[phase4a] train $OUT from $DATA gpu=$GPU_ID"
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
  2>&1 | tee "eval_results/qwen/phase4a/logs/train_${RUN}.log"
echo "[phase4a] done $OUT"
