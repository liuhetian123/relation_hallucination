#!/bin/bash
set -euo pipefail
# Train self-check on mixed-format prompts as stored in train_mf.json.
# Usage: bash scripts/phase4c/eval_train.sh <GPU_ID> [run_name] [adapter_path] [train_json]
if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/phase4c/eval_train.sh <GPU_ID> [run_name] [adapter_path] [train_json]"
  exit 1
fi
GPU_ID=$1
RUN=${2:-mf}
ADAPTER=${3:-checkpoints/qwen25vl3b_phase4c_mf}
DATA=${4:-data/phase4c/train_mf.json}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
MODEL=/home/lht/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
Q=eval_results/qwen/phase4c/questions/${RUN}_train.jsonl
ANS=eval_results/qwen/phase4c/answers/${RUN}_train.jsonl
MET=eval_results/qwen/phase4c/metrics/${RUN}_train.json
mkdir -p eval_results/qwen/phase4c/{questions,answers,metrics,logs}

/data/storage22t/lht/envs/qwen25vl/bin/python - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/phase4c")
from render import human_prompt_text

data = json.loads(Path("$DATA").read_text(encoding="utf-8"))
out = Path("$Q")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for rec in data:
        f.write(json.dumps({
            "question_id": rec["id"],
            "id": rec.get("source_id", rec["id"]),
            "image": rec["image"],
            "text": human_prompt_text(rec),
            "statement": rec["statement"],
            "label": rec["label"].lower(),
            "subset": rec.get("subset"),
            "prompt_format": rec.get("prompt_format"),
        }, ensure_ascii=False) + "\n")
print("wrote", out, "n", len(data), "using mixed training prompts")
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
  2>&1 | tee "eval_results/qwen/phase4c/logs/${RUN}_train_infer.log"

/data/storage22t/lht/envs/qwen25vl/bin/python scripts/phase2/score_verification.py \
  --question_file "$Q" \
  --result_file "$ANS" \
  --out_json "$MET"
