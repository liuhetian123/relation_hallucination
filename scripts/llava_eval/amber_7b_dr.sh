#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/v1_5/eval/amber_7b_dr.sh <GPU_ID>"
    echo "Example: bash scripts/v1_5/eval/amber_7b_dr.sh 5"
    exit 1
fi

GPU_ID=$1

mkdir -p ../eval_results/amber/answers
mkdir -p ../eval_results/amber/logs

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ../eval_results/amber/questions/amber_dr_llava.jsonl \
    --image-folder ../AMBER/images \
    --answers-file ../eval_results/amber/answers/llava-v1.5-7b_baseline_amber_dr_llava.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
