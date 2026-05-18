# relation_hallucination Project Layout

## Third-party repositories

- `LLaVA/`: LLaVA codebase.
- `POPE/`: POPE benchmark and annotations.
- `R-Bench/`: R-Bench benchmark, annotations, and images.
- `AMBER/`: AMBER benchmark, queries, and images.
- `relsim/`: RelSim repository.

## Project data

- `data/relsim_images/`: downloaded RelSim images.
- `data/relsim_llava_1k.json`: RelSim 1k LLaVA-format SFT data.
- `data/relsim_llava_10k.json`: RelSim 10k LLaVA-format SFT data.
- `data/relsim_llava_50k.json`: RelSim 50k LLaVA-format SFT data.

## Checkpoints

- `checkpoints/llava15_7b_relsim_lora_1k/`: LLaVA-1.5-7B LoRA checkpoint trained on RelSim 1k.
- Future checkpoints should follow the same naming pattern.

## Evaluation results

- `eval_results/pope/answers/`: POPE model answers.
- `eval_results/pope/logs/`: POPE metric logs.
- `eval_results/pope/summaries/`: POPE comparison summaries.
- `eval_results/rbench/answers/`: R-Bench model answers.
- `eval_results/rbench/logs/`: R-Bench logs.
- `eval_results/amber/questions/`: AMBER queries converted to LLaVA format.
- `eval_results/amber/answers/`: AMBER model answers.
- `eval_results/amber/logs/`: AMBER metric logs.

## Custom scripts

- `scripts/data_prep/`: dataset preparation and validation scripts.
- `scripts/convert/`: benchmark format conversion scripts.
- `scripts/utils/`: utility scripts.

## LLaVA evaluation scripts

LLaVA-specific evaluation shell scripts are stored under:

- `LLaVA/scripts/v1_5/eval/`
