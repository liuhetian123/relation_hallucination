#!/bin/bash
set -euo pipefail
# Usage: bash scripts/phase3a/eval_train.sh <gpu> <run_name> <adapter_path> <train_json>
if [ $# -lt 4 ]; then
  echo "Usage: bash scripts/phase3a/eval_train.sh <GPU_ID> <run_name> <adapter_path> <train_json>"
  exit 1
fi
GPU_ID=$1
RUN=$2
ADAPTER=$3
DATA=$4
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
MODEL=/home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
Q=eval_results/qwen/phase3a/questions/${RUN}_train.jsonl
ANS=eval_results/qwen/phase3a/answers/${RUN}_train.jsonl
MET=eval_results/qwen/phase3a/metrics/${RUN}_train.json
mkdir -p eval_results/qwen/phase3a/{questions,answers,metrics,logs}

/data/storage22t/lht/envs/qwen25vl/bin/python - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/phase2")
from common import load_json, verification_prompt
data = load_json(Path("$DATA"))
out = Path("$Q")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for rec in data:
        f.write(json.dumps({
            "question_id": rec["id"],
            "id": rec.get("source_id", rec["id"]),
            "image": rec["image"],
            "text": verification_prompt(rec["statement"]),
            "statement": rec["statement"],
            "label": rec["label"].lower(),
            "subset": rec.get("subset"),
        }, ensure_ascii=False) + "\n")
print("wrote", out, "n", len(data))
PY

PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 /data/storage22t/lht/envs/qwen25vl/bin/python \
  scripts/qwen_eval/infer_vqa.py \
  --model_path "$MODEL" \
  --question_file "$Q" \
  --image_folder /data/lht/relsim_dataset/relsim_images \
  --answers_file "$ANS" \
  --gpu "$GPU_ID" \
  --max_new_tokens 8 \
  --adapter_path "$ADAPTER" \
  2>&1 | tee "eval_results/qwen/phase3a/logs/${RUN}_train_infer.log"

/data/storage22t/lht/envs/qwen25vl/bin/python scripts/phase2/score_verification.py \
  --question_file "$Q" \
  --result_file "$ANS" \
  --out_json "$MET"
