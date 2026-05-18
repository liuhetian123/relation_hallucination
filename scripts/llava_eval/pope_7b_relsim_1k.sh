#!/bin/bash

mkdir -p ./playground/data/eval/pope/answers
mkdir -p ../eval_results/pope

python -m llava.eval.model_vqa_loader \
    --model-path ../checkpoints/llava15_7b_relsim_lora_1k \
    --model-base liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b-relsim-1k.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b-relsim-1k.jsonl \
    | tee ../eval_results/pope/logs/llava-v1.5-7b_relsim-1k_metrics.log

cp ./playground/data/eval/pope/answers/llava-v1.5-7b-relsim-1k.jsonl \
   ../eval_results/pope/answers/llava-v1.5-7b_relsim-1k_answers.jsonl
