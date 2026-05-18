#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/v1_5/eval/rbench_7b_image.sh <GPU_ID>"
    echo "Example: bash scripts/v1_5/eval/rbench_7b_image.sh 0"
    exit 1
fi

GPU_ID=$1

mkdir -p ../eval_results/rbench/answers
mkdir -p ../eval_results/rbench/logs

echo "=========================================="
echo "R-Bench image-level evaluation"
echo "Model: LLaVA-1.5-7B baseline"
echo "GPU: ${GPU_ID}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ../R-Bench/data_filterd/image-level_filterd_llava.jsonl \
    --image-folder ../R-Bench/images \
    --answers-file ../eval_results/rbench/answers/llava-v1.5-7b_baseline_image-level.json \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    2>&1 | tee ../eval_results/rbench/logs/llava-v1.5-7b_baseline_image-level_infer.log
