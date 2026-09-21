#!/bin/bash
set -euo pipefail
# Usage: bash scripts/phase2/train.sh <gpu> <v1|v2|v3>
if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/phase2/train.sh <GPU_ID> <v1|v2|v3>"
  exit 1
fi
GPU_ID=$1
SPLIT=$2
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
MODEL=/home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3

if [ "$SPLIT" = "v1" ]; then
  DATA=data/phase2/v1_positive.json
  OUT=checkpoints/qwen25vl3b_phase2_v1
elif [ "$SPLIT" = "v2" ]; then
  DATA=data/phase2/v2_pos_random.json
  OUT=checkpoints/qwen25vl3b_phase2_v2
elif [ "$SPLIT" = "v3" ]; then
  DATA=data/phase2/v3_pos_hard.json
  OUT=checkpoints/qwen25vl3b_phase2_v3
else
  echo "SPLIT must be v1, v2, or v3"
  exit 1
fi

mkdir -p "$OUT" eval_results/qwen/phase2/logs
PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 /data/storage22t/lht/envs/qwen25vl/bin/python \
  scripts/qwen_train/finetune_lora_relsim.py \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --image_root data/relsim_images \
  --output_dir "$OUT" \
  --gpu "$GPU_ID" \
  --seed 42 \
  2>&1 | tee "eval_results/qwen/phase2/logs/train_${SPLIT}.log"
