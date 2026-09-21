#!/bin/bash
set -euo pipefail
# Train nested Phase 3A LoRAs: S534-1ep, S1500-1ep, S3000-1ep, S3000-stepmatched.
# Usage: bash scripts/phase3a/train.sh <GPU_ID>
if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/phase3a/train.sh <GPU_ID>"
  exit 1
fi
GPU_ID=$1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=/data/storage22t/lht/envs/qwen25vl/bin/python
MODEL=/home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
IMG=/data/lht/relsim_dataset/relsim_images
mkdir -p eval_results/qwen/phase3a/logs checkpoints

train_one () {
  local data=$1 out=$2 extra=${3:-}
  if [ -f "$out/adapter_model.safetensors" ]; then
    echo "[phase3a] skip existing $out"
    return 0
  fi
  mkdir -p "$out"
  echo "[phase3a] train $out from $data $extra"
  PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 "$PY" \
    scripts/qwen_train/finetune_lora_relsim.py \
    --model_path "$MODEL" \
    --data_path "$data" \
    --image_root "$IMG" \
    --output_dir "$out" \
    --gpu "$GPU_ID" \
    --seed 42 \
    --num_train_epochs 1 \
    --save_strategy epoch \
    --save_total_limit 2 \
    $extra \
    2>&1 | tee "eval_results/qwen/phase3a/logs/train_$(basename "$out").log"
}

read_global_step () {
  local out=$1
  "$PY" - "$out" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
cands = [root / "trainer_state.json"]
cands.extend(sorted(root.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[-1])))
meta = root / "train_meta.json"
for p in reversed(cands):
    if p.exists():
        print(int(json.loads(p.read_text())["global_step"]))
        raise SystemExit(0)
if meta.exists():
    step = json.loads(meta.read_text()).get("global_step")
    if step:
        print(int(step))
        raise SystemExit(0)
raise SystemExit(f"no trainer_state under {root}")
PY
}

train_one data/phase3a/train_s534.json checkpoints/qwen25vl3b_phase3a_s534
S534_STEPS="$(read_global_step checkpoints/qwen25vl3b_phase3a_s534)"
echo "[phase3a] S534-1ep optimizer steps=$S534_STEPS"

train_one data/phase3a/train_s1500.json checkpoints/qwen25vl3b_phase3a_s1500
train_one data/phase3a/train_s3000.json checkpoints/qwen25vl3b_phase3a_s3000
train_one data/phase3a/train_s3000.json checkpoints/qwen25vl3b_phase3a_s3000_stepmatched \
  "--max_steps $S534_STEPS --save_strategy steps --save_steps $S534_STEPS"
