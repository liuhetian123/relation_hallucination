# Relation Hallucination Experiments

This repository contains scripts and configuration files for studying whether RelSim anonymous relational-caption supervision can reduce hallucination in LLaVA-1.5-7B.

## Repository scope

This repository does not include:

- third-party repositories: LLaVA, POPE, R-Bench, AMBER, RelSim
- downloaded images or benchmark datasets
- pretrained model weights
- fine-tuned checkpoints
- raw model answer files

Please prepare them following the instructions in `PROJECT_GUIDE.md`.

## Project layout

- `configs/`: environment files and version snapshots
- `scripts/data_prep/`: RelSim data preparation scripts
- `scripts/convert/`: benchmark format conversion scripts
- `scripts/llava_eval/`: LLaVA evaluation shell scripts
- `patches/llava_eval/`: files copied into `LLaVA/llava/eval/`
- `eval_results/`: metric logs and summaries

## Environment

```bash
conda env create -f configs/environment_relhallu.yml
conda activate relhallu
