#!/bin/bash
set -euo pipefail
# Train Phase 2B balanced verification LoRA for 2 epochs, keep epoch-1 as V2B-1ep.
# Usage: bash scripts/phase2b/train.sh <GPU_ID>
if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/phase2b/train.sh <GPU_ID>"
  exit 1
fi
GPU_ID=$1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
MODEL=/home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
DATA=data/phase2b/v2b_balanced.json
OUT=checkpoints/qwen25vl3b_phase2_v2b_2ep
OUT1=checkpoints/qwen25vl3b_phase2_v2b_1ep
mkdir -p "$OUT" "$OUT1" eval_results/qwen/phase2b/logs

PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 /data/storage22t/lht/envs/qwen25vl/bin/python \
  scripts/qwen_train/finetune_lora_relsim.py \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --image_root data/relsim_images \
  --output_dir "$OUT" \
  --gpu "$GPU_ID" \
  --seed 42 \
  --num_train_epochs 2 \
  --save_strategy epoch \
  --save_total_limit 3 \
  2>&1 | tee eval_results/qwen/phase2b/logs/train_v2b_2ep.log

# Epoch-1 checkpoint is the earlier checkpoint-* directory.
python3 - <<PY
from pathlib import Path
import shutil
out = Path("$OUT")
out1 = Path("$OUT1")
ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
if not ckpts:
    raise SystemExit("No checkpoint-* found after 2-epoch training")
src = ckpts[0]
print(f"Copying epoch-1 adapter from {src} -> {out1}")
if out1.exists():
    shutil.rmtree(out1)
shutil.copytree(src, out1)
# Processor files live in the final output dir.
for name in ("preprocessor_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.json", "chat_template.jinja"):
    srcf = out / name
    if srcf.exists():
        shutil.copy2(srcf, out1 / name)
meta = (out / "train_meta.json")
if meta.exists():
    shutil.copy2(meta, out1 / "train_meta.json")
print("saved", out1)
PY
