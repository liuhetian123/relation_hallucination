# Relation Hallucination Experiments

本仓库用于复现和扩展一组关于 **关系幻觉（relation hallucination）** 的实验：我们将 RelSim anonymous relational captions 转换为 LLaVA 的视觉指令微调数据，使用 LoRA 对 LLaVA-1.5-7B 进行轻量微调，并在 POPE、R-Bench 和 AMBER 上评估微调前后的幻觉行为变化。

> 核心问题：显式的关系监督是否能缓解多模态大模型在视觉关系判断中的幻觉？

---

## 1. Repository Scope

本仓库只保存我们自己的：

- 环境配置文件；
- 数据转换脚本；
- LLaVA 评估脚本；
- 必要 patch；
- 实验指标日志和说明文档。

本仓库 **不包含**：

- 第三方仓库：`LLaVA/`、`POPE/`、`R-Bench/`、`AMBER/`、`relsim/`；
- 下载后的图像数据；
- RelSim 微调 JSON 数据；
- COCO、R-Bench、AMBER 图像；
- LLaVA 预训练模型权重；
- LoRA checkpoint；
- 原始逐条 answer 文件。

这些内容需要按照本文档单独准备。

---

## 2. Expected Project Layout

复现实验时，项目根目录推荐保持如下结构：

```text
relation_hallucination/
├── LLaVA/                         # third-party LLaVA codebase
├── POPE/                          # POPE benchmark repository / annotations
├── R-Bench/                       # R-Bench benchmark repository / data
├── AMBER/                         # AMBER benchmark repository / data
├── relsim/                        # RelSim repository, optional
├── data/                          # RelSim fine-tuning data, not tracked by git
│   ├── relsim_images/
│   ├── relsim_llava_1k.json
│   ├── relsim_llava_10k.json
│   └── relsim_llava_50k.json
├── checkpoints/                   # LoRA checkpoints, not tracked by git
│   └── llava15_7b_relsim_lora_1k/
├── eval_results/
│   ├── pope/
│   │   ├── answers/               # raw answers, not tracked by git
│   │   ├── logs/                  # metrics logs
│   │   └── summaries/
│   ├── rbench/
│   │   ├── answers/               # raw answers, not tracked by git
│   │   └── logs/                  # metrics logs
│   └── amber/
│       ├── questions/             # converted questions
│       ├── answers/               # raw answers, not tracked by git
│       └── logs/                  # metrics logs
├── scripts/
│   ├── data_prep/                 # RelSim data preparation scripts
│   ├── convert/                   # benchmark format conversion scripts
│   ├── llava_eval/                # LLaVA evaluation shell scripts copied from experiments
│   └── utils/
├── patches/
│   └── llava_eval/                # files copied into LLaVA/llava/eval/
├── configs/                       # environment and version files
├── PROJECT_GUIDE.md
├── PROJECT_LAYOUT.md
└── README.md
```

---

## 3. Environment Setup

### 3.1 Create conda environment

```bash
conda env create -f configs/environment_relhallu.yml
conda activate relhallu
```

如果 conda 环境创建失败，可以参考完整 pip 依赖：

```bash
pip install -r configs/requirements_relhallu_full.txt
```

核心版本可查看：

```bash
cat configs/version_snapshot.txt
```

### 3.2 AMBER-specific dependencies

AMBER 官方评估脚本依赖 `nltk`、`spacy` 和 `scikit-learn`。如果运行 AMBER 时缺包，可以安装：

```bash
pip install nltk spacy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果需要安装 spaCy 英文模型，推荐优先用 conda-forge：

```bash
conda install -c conda-forge spacy-model-en_core_web_lg -y
```

如果服务器无法访问 GitHub，可以在本地下载 wheel 后上传服务器安装。

---

## 4. Prepare Third-party Repositories

请在项目根目录下单独 clone 第三方仓库：

```bash
git clone https://github.com/haotian-liu/LLaVA.git
git clone https://github.com/AoiDragon/POPE.git
git clone https://github.com/mrwu-mac/R-Bench.git
git clone https://github.com/junyangwang0410/AMBER.git
```

RelSim 仓库可选，如果需要复现 RelSim 原始流程或查看原始代码，可单独 clone 到：

```text
relation_hallucination/relsim/
```

第三方仓库版本记录在：

```bash
cat configs/third_party_commits.txt
```

---

## 5. Apply Custom LLaVA Scripts and Patches

由于本仓库不上传整个 `LLaVA/`，因此需要把自定义脚本复制到 LLaVA 对应目录。

```bash
cd relation_hallucination

cp scripts/llava_eval/*.sh LLaVA/scripts/v1_5/eval/
cp patches/llava_eval/*.py LLaVA/llava/eval/
```

其中：

```text
scripts/llava_eval/
├── pope_7b.sh
├── pope_7b_relsim_1k.sh
├── rbench_7b_image.sh
├── rbench_7b_relsim_1k_image.sh
├── amber_7b_dr.sh
└── amber_7b_relsim_1k_dr.sh
```

`patches/llava_eval/` 中的文件用于支持 R-Bench 相关评估脚本。如果只使用 LLaVA 官方 `model_vqa_loader` 跑 R-Bench image-level，可以不依赖 patched `eval_rbench.py`。

---

## 6. Prepare RelSim Fine-tuning Data

### 6.1 Data format

RelSim 图像统一放在：

```text
data/relsim_images/
```

不同规模的训练数据通过不同 JSON 控制：

```text
data/relsim_llava_1k.json
data/relsim_llava_10k.json
data/relsim_llava_50k.json
data/relsim_llava_full.json
```

所有 JSON 共用同一个图片文件夹。训练时 LLaVA 根据 JSON 中的 `image` 字段去 `data/relsim_images/` 找对应图片，因此不需要为 1k、10k、50k 分别建立图片文件夹。

### 6.2 Generate RelSim LLaVA-format data

示例：准备 1k 数据。

```bash
cd relation_hallucination
conda activate relhallu

python scripts/data_prep/prepare_relsim_llava.py
```

如果使用支持断点续传或不同规模的脚本，建议命名为：

```text
scripts/data_prep/prepare_relsim_llava_resume.py
```

示例命令：

```bash
python scripts/data_prep/prepare_relsim_llava_resume.py \
    --num-samples 10000 \
    --out-json data/relsim_llava_10k.json \
    --image-dir data/relsim_images
```

### 6.3 Check JSON-image consistency

建议每次生成 JSON 后检查：

```bash
python scripts/data_prep/check_relsim_json.py \
    --json data/relsim_llava_10k.json \
    --image-dir data/relsim_images
```

要求：

```text
Missing images: 0
Broken images: 0
```

---

## 7. Fine-tuning with Different RelSim Scales

我们使用 LoRA 对 LLaVA-1.5-7B 进行轻量微调。视觉编码器保持冻结，微调主要作用于 LLM 的 LoRA adapter 和相关投影模块。

### 7.1 1k RelSim LoRA fine-tuning

在 `LLaVA/` 目录下运行：

```bash
cd relation_hallucination/LLaVA
conda activate relhallu

bash scripts/v1_5/finetune_lora_relsim_1k.sh
```

如果需要指定单张 GPU，例如物理 GPU 4：

```bash
CUDA_VISIBLE_DEVICES=4 bash scripts/v1_5/finetune_lora_relsim_1k.sh
```

输出目录：

```text
checkpoints/llava15_7b_relsim_lora_1k/
```

### 7.2 Train with 10k / 50k / full RelSim data

复制 1k 脚本并修改两个参数：

```bash
cp LLaVA/scripts/v1_5/finetune_lora_relsim_1k.sh \
   LLaVA/scripts/v1_5/finetune_lora_relsim_10k.sh
```

将脚本中的：

```bash
--data_path ../data/relsim_llava_1k.json
--output_dir ../checkpoints/llava15_7b_relsim_lora_1k
```

改成：

```bash
--data_path ../data/relsim_llava_10k.json
--output_dir ../checkpoints/llava15_7b_relsim_lora_10k
```

对于 50k：

```bash
--data_path ../data/relsim_llava_50k.json
--output_dir ../checkpoints/llava15_7b_relsim_lora_50k
```

注意：不同数据规模必须使用不同 `--output_dir`，否则会覆盖已有 checkpoint。

---

## 8. Evaluation Overview

当前评估包含三类 benchmark：

| Benchmark | Purpose | Main files |
|---|---|---|
| POPE | object hallucination / yes-bias | `LLaVA/playground/data/eval/pope/` |
| R-Bench | relation hallucination | `R-Bench/data_filterd/`, `R-Bench/images/` |
| AMBER | existence / attribute / relation hallucination | `AMBER/data/query/`, `AMBER/images/` |

推荐评估顺序：

```text
POPE → R-Bench image-level → AMBER discriminative-relation
```

---

## 9. POPE Evaluation

### 9.1 Prepare POPE data

需要准备：

```text
LLaVA/playground/data/eval/pope/val2014/
LLaVA/playground/data/eval/pope/coco/
LLaVA/playground/data/eval/pope/llava_pope_test.jsonl
```

其中 `coco/` 下应包含：

```text
coco_pope_random.json
coco_pope_popular.json
coco_pope_adversarial.json
```

可以由 POPE 仓库中的 `POPE/output/coco/` 复制而来。

### 9.2 Run baseline

```bash
cd relation_hallucination/LLaVA
conda activate relhallu

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope_7b.sh
```

### 9.3 Run RelSim-1k LoRA

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope_7b_relsim_1k.sh
```

结果保存到：

```text
eval_results/pope/answers/
eval_results/pope/logs/
```

---

## 10. R-Bench Evaluation

### 10.1 Prepare R-Bench data

需要准备：

```text
R-Bench/data_filterd/
R-Bench/images/
```

### 10.2 Convert R-Bench JSON to LLaVA JSONL

LLaVA 官方 `model_vqa_loader` 需要 JSONL 格式，而 R-Bench 原始 `image-level_filterd.json` 是 JSON list。因此需要转换：

```bash
cd relation_hallucination

python scripts/convert/convert_rbench_to_llava_jsonl.py \
    --input R-Bench/data_filterd/image-level_filterd.json \
    --output R-Bench/data_filterd/image-level_filterd_llava.jsonl
```

检查：

```bash
wc -l R-Bench/data_filterd/image-level_filterd_llava.jsonl
head -n 2 R-Bench/data_filterd/image-level_filterd_llava.jsonl
```

应约为：

```text
7787 R-Bench/data_filterd/image-level_filterd_llava.jsonl
```

### 10.3 Run R-Bench baseline inference

```bash
cd relation_hallucination/LLaVA

bash scripts/v1_5/eval/rbench_7b_image.sh 0
```

### 10.4 Run R-Bench RelSim-1k inference

```bash
bash scripts/v1_5/eval/rbench_7b_relsim_1k_image.sh 0
```

### 10.5 Evaluate R-Bench answers

注意：生成答案时使用转换后的 JSONL；官方评估时仍然使用 R-Bench 原始 JSON。

Baseline：

```bash
cd relation_hallucination/R-Bench

python eval.py \
    --annotation-dir data_filterd \
    --question-file data_filterd/image-level_filterd.json \
    --question-id-file data_filterd/nocaps_image-level_rel_ids_holder.json \
    --result-file ../eval_results/rbench/answers/llava-v1.5-7b_baseline_image-level.json \
    --eval_image \
    | tee ../eval_results/rbench/logs/llava-v1.5-7b_baseline_image-level_metrics.log
```

RelSim-1k：

```bash
python eval.py \
    --annotation-dir data_filterd \
    --question-file data_filterd/image-level_filterd.json \
    --question-id-file data_filterd/nocaps_image-level_rel_ids_holder.json \
    --result-file ../eval_results/rbench/answers/llava-v1.5-7b_relsim-1k_image-level.json \
    --eval_image \
    | tee ../eval_results/rbench/logs/llava-v1.5-7b_relsim-1k_image-level_metrics.log
```

### 10.6 Important note on empty-output bug

R-Bench demo 中的 `eval_rbench.py` 在某些 LLaVA / transformers 环境下可能生成空回答：

```json
{"text": ""}
```

这会导致 R-Bench `eval.py` 将空回答误判为 yes，从而出现虚假的：

```text
Yes ratio ≈ 100%
TN = 0
Accuracy ≈ 50%
```

因此 image-level 推荐使用 LLaVA 官方 `model_vqa_loader`，并确保输入是 JSONL。

检查 answer 是否为空：

```bash
python - <<'PY'
import json
from collections import Counter

path = "eval_results/rbench/answers/llava-v1.5-7b_baseline_image-level.json"
c = Counter()
with open(path, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        ans = x.get("text", "").strip().lower()
        if ans == "": c["empty"] += 1
        elif ans.startswith("yes"): c["yes"] += 1
        elif ans.startswith("no"): c["no"] += 1
        else: c["other"] += 1
print(c)
PY
```

---

## 11. AMBER Evaluation

### 11.1 Prepare AMBER data

需要准备：

```text
AMBER/data/query/query_discriminative-relation.json
AMBER/images/
```

### 11.2 Convert AMBER questions to LLaVA format

```bash
cd relation_hallucination

python scripts/convert/convert_amber_to_llava.py \
    --query-file AMBER/data/query/query_discriminative-relation.json \
    --image-folder AMBER/images \
    --out-file eval_results/amber/questions/amber_dr_llava.jsonl
```

检查图片是否存在：

```bash
python - <<'PY'
import json, os
qfile = "eval_results/amber/questions/amber_dr_llava.jsonl"
imgdir = "AMBER/images"
missing = []
with open(qfile, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        if not os.path.exists(os.path.join(imgdir, x["image"])):
            missing.append(x["image"])
print("missing images:", len(missing))
print(missing[:10])
PY
```

### 11.3 Run AMBER baseline inference

```bash
cd relation_hallucination/LLaVA

bash scripts/v1_5/eval/amber_7b_dr.sh 0
```

### 11.4 Run AMBER RelSim-1k inference

```bash
bash scripts/v1_5/eval/amber_7b_relsim_1k_dr.sh 0
```

### 11.5 Convert LLaVA answer to AMBER official format

```bash
cd relation_hallucination

python scripts/convert/convert_llava_answer_to_amber.py \
    --llava-answer eval_results/amber/answers/llava-v1.5-7b_baseline_amber_dr_llava.jsonl \
    --out-file eval_results/amber/answers/llava-v1.5-7b_baseline_amber_dr.json

python scripts/convert/convert_llava_answer_to_amber.py \
    --llava-answer eval_results/amber/answers/llava-v1.5-7b_relsim-1k_amber_dr_llava.jsonl \
    --out-file eval_results/amber/answers/llava-v1.5-7b_relsim-1k_amber_dr.json
```

### 11.6 Run AMBER official evaluation

```bash
cd relation_hallucination/AMBER

python inference.py \
    --inference_data ../eval_results/amber/answers/llava-v1.5-7b_baseline_amber_dr.json \
    --evaluation_type dr \
    | tee ../eval_results/amber/logs/llava-v1.5-7b_baseline_amber_dr_metrics.log

python inference.py \
    --inference_data ../eval_results/amber/answers/llava-v1.5-7b_relsim-1k_amber_dr.json \
    --evaluation_type dr \
    | tee ../eval_results/amber/logs/llava-v1.5-7b_relsim-1k_amber_dr_metrics.log
```

### 11.7 Important note on AMBER metrics

在 AMBER discriminative-relation 中，官方 `Precision` / `Recall` 是以 **No** 为正类计算的，反映模型拒绝不成立关系描述的能力。

因此：

```text
Recall ↑ 表示模型能拒绝更多不成立的关系；
Precision ↓ 可能表示模型更保守，也误拒了一些真实关系。
```

不要把 AMBER 的 relation recall 简单解释成“识别真实关系的能力增强”。

---

## 12. Current Main Results

### 12.1 POPE

| Model | Category | Acc | Precision | Recall | F1 | Yes ratio |
|---|---|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | random | 0.8957 | 0.8861 | 0.9080 | 0.8969 | 0.5123 |
| LLaVA-1.5-7B + RelSim-1k | random | 0.9017 | 0.9175 | 0.8827 | 0.8998 | 0.4810 |
| LLaVA-1.5-7B | adversarial | 0.7980 | 0.7443 | 0.9080 | 0.8180 | 0.6100 |
| LLaVA-1.5-7B + RelSim-1k | adversarial | 0.8167 | 0.7797 | 0.8827 | 0.8280 | 0.5660 |
| LLaVA-1.5-7B | popular | 0.8620 | 0.8315 | 0.9080 | 0.8681 | 0.5460 |
| LLaVA-1.5-7B + RelSim-1k | popular | 0.8753 | 0.8699 | 0.8827 | 0.8762 | 0.5073 |

Observation: RelSim-1k reduces false positives and yes ratio on POPE, improving precision and F1 while slightly reducing recall.

### 12.2 R-Bench image-level

| Model | Accuracy | Precision | Recall | F1 | Yes ratio |
|---|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | 70.79 | 64.14 | 95.61 | 76.77 | 75.27 |
| LLaVA-1.5-7B + RelSim-1k | 71.62 | 64.91 | 95.30 | 77.22 | 74.14 |

Observation: RelSim-1k reduces relation false positives and yes ratio, improving accuracy, precision, and F1 with a small recall trade-off.

### 12.3 AMBER discriminative-relation

| Model | Accuracy | Precision | Recall | F1 | Yes ratio |
|---|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | 73.9 | 69.1 | 66.6 | 67.8 | 60.10 |
| LLaVA-1.5-7B + RelSim-1k | 74.5 | 66.5 | 77.2 | 71.4 | 51.92 |

Note: AMBER relation Precision / Recall are No-oriented metrics. RelSim-1k improves No-recall and F1 while reducing Yes ratio, suggesting stronger rejection of unsupported relation claims but with a conservativeness trade-off.

---

## 13. Troubleshooting

### 13.1 Hugging Face or GitHub network error

If downloading model weights or spaCy models fails, use mirrors, local download, or upload files manually via `scp`.

### 13.2 `model_vqa_loader.py` reports `list indices must be integers`

This usually means the question file is JSON list, but `model_vqa_loader.py` expects JSONL.

Fix:

```bash
python scripts/convert/convert_rbench_to_llava_jsonl.py \
    --input R-Bench/data_filterd/image-level_filterd.json \
    --output R-Bench/data_filterd/image-level_filterd_llava.jsonl
```

### 13.3 R-Bench answer text is empty

If answer records contain:

```json
{"text": ""}
```

then the result is invalid. Use `model_vqa_loader.py` with converted JSONL input instead of relying on incompatible decoding logic.

### 13.4 AMBER `numpy.dtype size changed`

This indicates NumPy / scikit-learn binary incompatibility. Reinstall compatible versions:

```bash
pip install --force-reinstall --no-cache-dir \
  numpy==1.26.4 \
  scipy==1.11.4 \
  scikit-learn==1.3.2 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 13.5 DeepSpeed GPU slot error

When using DeepSpeed with `CUDA_VISIBLE_DEVICES`, visible GPU indices may be remapped. Prefer using scripts that explicitly set local visible device or adjust DeepSpeed include arguments carefully.

---

## 14. What Is Not Tracked by Git

The following files and directories are intentionally ignored:

```text
LLaVA/
POPE/
R-Bench/
AMBER/
relsim/
data/relsim_images/
data/*.json
data/*.jsonl
checkpoints/
outputs/
eval_results/*/answers/
*.pt
*.pth
*.bin
*.safetensors
```

If you want to reproduce our exact results without retraining, download the prepared RelSim data and LoRA checkpoint from the external link provided by the project maintainer, and place them under the expected paths.

---

## 15. Citation and Acknowledgements

This project builds on the following third-party resources:

- LLaVA
- POPE
- R-Bench
- AMBER
- RelSim / anonymous relational captions

Please follow the licenses and citation requirements of the original projects and datasets.

---

## 16. Maintainer Notes

Before pushing updates to GitHub, check staged file sizes:

```bash
git diff --cached --name-only | xargs -r du -h | sort -h | tail -n 30
```

Make sure no datasets, checkpoints, images, model weights, or raw answer files are accidentally committed.
