# relation_hallucination 快速上手说明

本文档用于帮助新同学快速接手当前实验代码。当前项目目标是：使用 RelSim 的匿名关系 caption 数据对 LLaVA-1.5-7B 做 LoRA 微调，并在 POPE、R-Bench、AMBER 等幻觉评估数据集上比较微调前后的变化。

---

## 1. 项目目录约定

项目根目录：

```bash
~/works/relation_hallucination
```

固定目录结构如下：

```bash
relation_hallucination/
├── LLaVA/              # LLaVA 代码仓库；LLaVA 专用 eval / train 脚本也放在这里
├── POPE/               # POPE benchmark 仓库与标注
├── R-Bench/            # R-Bench benchmark 仓库、标注与图片
├── AMBER/              # AMBER benchmark 仓库、query 与图片
├── relsim/             # RelSim 原始仓库
├── data/               # RelSim 微调数据
├── checkpoints/        # LoRA 微调后的 checkpoint
├── eval_results/       # 所有评估结果
├── scripts/            # 项目自定义脚本
└── outputs/            # 临时输出
```

### 1.1 RelSim 数据目录

```bash
data/
├── relsim_images/          # 所有 RelSim 图片统一放这里
├── relsim_llava_1k.json    # 1k LLaVA 格式训练数据
├── relsim_llava_10k.json   # 10k LLaVA 格式训练数据
├── relsim_llava_50k.json   # 50k LLaVA 格式训练数据
└── relsim_llava_full.json  # full 规模训练数据，可选
```

注意：**不同规模的数据集不需要分图片文件夹**。所有图片统一放在：

```bash
data/relsim_images/
```

不同训练规模只通过不同 JSON 文件控制。

### 1.2 Checkpoint 目录

```bash
checkpoints/
├── llava15_7b_relsim_lora_1k/
├── llava15_7b_relsim_lora_10k/
├── llava15_7b_relsim_lora_50k/
└── llava15_7b_relsim_lora_full/
```

不同数据规模必须使用不同的 `--output_dir`，避免覆盖已有 checkpoint。

### 1.3 评估结果目录

```bash
eval_results/
├── pope/
│   ├── answers/
│   ├── logs/
│   └── summaries/
├── rbench/
│   ├── answers/
│   └── logs/
└── amber/
    ├── questions/
    ├── answers/
    └── logs/
```

长期保存的评估结果都放到 `eval_results/`，不要只保留在 LLaVA 默认输出目录中。

### 1.4 自定义脚本目录

```bash
scripts/
├── data_prep/      # 数据准备和数据检查脚本
├── convert/        # benchmark 格式转换脚本
└── utils/          # 辅助检查脚本
```

LLaVA 专用的训练和评估 shell 脚本放在：

```bash
LLaVA/scripts/v1_5/eval/
LLaVA/scripts/v1_5/
```

---

## 2. 环境启动

进入服务器后：

```bash
conda activate relhallu
cd ~/works/relation_hallucination
```

检查 GPU：

```bash
nvidia-smi
```

如果要运行 LLaVA 相关命令，进入 LLaVA 目录：

```bash
cd ~/works/relation_hallucination/LLaVA
```

---

## 3. RelSim 数据准备

### 3.1 数据来源说明

当前使用的数据是 RelSim 发布的匿名关系 caption 数据。每条样本包含：

```text
image URL + anonymous relational caption
```

我们没有重新训练 RelSim 的 anonymous captioning model，而是直接使用数据集中已经生成好的 `caption` 字段，将其转换成 LLaVA 的视觉指令微调格式。

LLaVA 格式大致如下：

```json
{
  "id": "relsim_xxx",
  "image": "xxx.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nDescribe the underlying relational logic of this image. Focus on abstract relations rather than concrete object names, colors, or surface appearance."
    },
    {
      "from": "gpt",
      "value": "anonymous relational caption"
    }
  ]
}
```

### 3.2 下载/生成 1k 数据

当前 1k 数据已经生成并上传到服务器，路径为：

```bash
data/relsim_images/
data/relsim_llava_1k.json
```

检查数据完整性：

```bash
cd ~/works/relation_hallucination

python - <<'PY'
import json, os

json_path = "data/relsim_llava_1k.json"
image_dir = "data/relsim_images"

with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

missing = []
for x in data:
    p = os.path.join(image_dir, x["image"])
    if not os.path.exists(p):
        missing.append(x["image"])

print("json samples:", len(data))
print("missing images:", len(missing))
print("All images exist." if not missing else missing[:10])
PY
```

期望输出：

```text
json samples: 1000
missing images: 0
All images exist.
```

### 3.3 准备 10k / 50k / full 数据

推荐在本地主机下载图片，然后上传到服务器。原因是服务器常常无法访问外部图片 URL。

本地主机上继续复用已有目录：

```bash
data/relsim_images/
```

不要删除已有 1k 图片。生成更大规模 JSON 时，脚本会复用已存在图片，并继续下载新的图片。

推荐生成：

```bash
data/relsim_llava_10k.json
data/relsim_llava_50k.json
data/relsim_llava_full.json
```

如果使用 resume 脚本，建议放在：

```bash
scripts/data_prep/prepare_relsim_llava_resume.py
```

示例命令：

```bash
python scripts/data_prep/prepare_relsim_llava_resume.py \
    --num-samples 10000 \
    --out-json data/relsim_llava_10k.json
```

生成后检查：

```bash
python scripts/data_prep/check_relsim_json.py \
    --json data/relsim_llava_10k.json \
    --image-dir data/relsim_images
```

如果没有 `check_relsim_json.py`，可以使用第 3.2 节中的 Python 检查代码。

### 3.4 上传本地数据到服务器

在 Windows PowerShell 本地主机中执行，不要在已经 SSH 到服务器后执行：

```powershell
cd D:\works\relation_hallu

tar -czf relsim_10k_data.tar.gz data\relsim_images data\relsim_llava_10k.json
scp .\relsim_10k_data.tar.gz lht_8xA6000_docker:~/works/relation_hallucination/
```

服务器上解压：

```bash
cd ~/works/relation_hallucination
tar -xzf relsim_10k_data.tar.gz
```

---

## 4. LLaVA-1.5-7B LoRA 微调

### 4.1 1k 微调脚本

脚本位置：

```bash
LLaVA/scripts/v1_5/finetune_lora_relsim_1k.sh
```

运行方式：

```bash
cd ~/works/relation_hallucination/LLaVA
conda activate relhallu

bash scripts/v1_5/finetune_lora_relsim_1k.sh <GPU_ID>
```

例如使用物理 GPU 4：

```bash
bash scripts/v1_5/finetune_lora_relsim_1k.sh 4
```

注意：该训练脚本内部使用 DeepSpeed 的：

```bash
deepspeed --include localhost:${GPU_ID}
```

因此不要再额外写：

```bash
CUDA_VISIBLE_DEVICES=4 bash scripts/v1_5/finetune_lora_relsim_1k.sh
```

否则可能出现 DeepSpeed GPU slot 编号错误。

### 4.2 当前 LoRA 微调策略

当前方案是参数高效微调：

```text
Frozen:
  - CLIP visual encoder / vision tower
  - LLaVA/Vicuna base LLM 原始大部分权重

Trainable:
  - LLM LoRA adapter
  - multimodal projector / connector
```

脚本中关键参数：

```bash
--lora_enable True
--lora_r 16
--lora_alpha 32
--mm_projector_lr 1e-4
--learning_rate 2e-5
```

原始模型权重从 Hugging Face cache 读取，不会被修改：

```bash
~/.cache/huggingface/hub/
```

微调后的 LoRA checkpoint 保存到：

```bash
checkpoints/llava15_7b_relsim_lora_1k/
```

### 4.3 使用不同数量的 RelSim 数据重新微调

复制 1k 脚本并修改数据路径和输出目录。

例如 10k：

```bash
cd ~/works/relation_hallucination/LLaVA
cp scripts/v1_5/finetune_lora_relsim_1k.sh scripts/v1_5/finetune_lora_relsim_10k.sh
vim scripts/v1_5/finetune_lora_relsim_10k.sh
```

需要修改：

```bash
--data_path ../data/relsim_llava_10k.json
--output_dir ../checkpoints/llava15_7b_relsim_lora_10k
```

如果训练更大数据，建议也修改：

```bash
--num_train_epochs 1
--save_steps 500
--save_total_limit 2
```

运行：

```bash
bash scripts/v1_5/finetune_lora_relsim_10k.sh 4
```

### 4.4 检查 checkpoint

训练结束后：

```bash
cd ~/works/relation_hallucination/LLaVA
ls -lah ../checkpoints/llava15_7b_relsim_lora_1k
```

正常应看到类似：

```text
adapter_config.json
adapter_model.bin / adapter_model.safetensors
non_lora_trainables.bin
trainer_state.json
training_args.bin
```

---

## 5. POPE 评估

POPE 用于评估 object existence hallucination。

### 5.1 Baseline 评估

脚本：

```bash
LLaVA/scripts/v1_5/eval/pope_7b.sh
```

运行：

```bash
cd ~/works/relation_hallucination/LLaVA
CUDA_VISIBLE_DEVICES=<GPU_ID> bash scripts/v1_5/eval/pope_7b.sh
```

建议输出保存到：

```bash
eval_results/pope/answers/llava-v1.5-7b_baseline_answers.jsonl
eval_results/pope/logs/llava-v1.5-7b_baseline_metrics.log
```

### 5.2 RelSim-1k 微调模型评估

脚本：

```bash
LLaVA/scripts/v1_5/eval/pope_7b_relsim_1k.sh
```

运行：

```bash
cd ~/works/relation_hallucination/LLaVA
CUDA_VISIBLE_DEVICES=<GPU_ID> bash scripts/v1_5/eval/pope_7b_relsim_1k.sh
```

RelSim-1k 模型加载方式：

```bash
--model-path ../checkpoints/llava15_7b_relsim_lora_1k
--model-base liuhaotian/llava-v1.5-7b
```

输出保存到：

```bash
eval_results/pope/answers/llava-v1.5-7b_relsim-1k_answers.jsonl
eval_results/pope/logs/llava-v1.5-7b_relsim-1k_metrics.log
```

### 5.3 已有 POPE 观察

RelSim-1k 在 POPE 上的主要现象：

```text
FP 下降
Precision 上升
Yes ratio 下降
Recall 略降
```

这说明 RelSim-1k 微调后模型更保守，减少了一部分 object-level false positive，但也牺牲了一些 recall。

---

## 6. R-Bench 评估

R-Bench 用于评估 relationship hallucination。

### 6.1 重要注意事项

R-Bench 官方 demo 中的：

```bash
python -m llava.eval.eval_rbench
```

在当前环境下曾出现空输出问题：

```json
"text": ""
```

因此 image-level 评估推荐使用 LLaVA 自带的：

```bash
python -m llava.eval.model_vqa_loader
```

但是 `model_vqa_loader` 需要 JSONL 格式输入，而 R-Bench 原始文件是 JSON list。因此需要先转换：

```text
R-Bench/data_filterd/image-level_filterd.json
→ R-Bench/data_filterd/image-level_filterd_llava.jsonl
```

### 6.2 转换 R-Bench image-level 标注

转换脚本建议放在：

```bash
scripts/convert/convert_rbench_to_llava_jsonl.py
```

运行：

```bash
cd ~/works/relation_hallucination

python scripts/convert/convert_rbench_to_llava_jsonl.py \
    --input R-Bench/data_filterd/image-level_filterd.json \
    --output R-Bench/data_filterd/image-level_filterd_llava.jsonl
```

检查：

```bash
wc -l R-Bench/data_filterd/image-level_filterd_llava.jsonl
head -n 3 R-Bench/data_filterd/image-level_filterd_llava.jsonl
```

### 6.3 Baseline R-Bench image-level 推理

脚本：

```bash
LLaVA/scripts/v1_5/eval/rbench_7b_image.sh
```

运行：

```bash
cd ~/works/relation_hallucination/LLaVA
bash scripts/v1_5/eval/rbench_7b_image.sh <GPU_ID>
```

输出：

```bash
eval_results/rbench/answers/llava-v1.5-7b_baseline_image-level.json
eval_results/rbench/logs/llava-v1.5-7b_baseline_image-level_infer.log
```

### 6.4 RelSim-1k R-Bench image-level 推理

脚本：

```bash
LLaVA/scripts/v1_5/eval/rbench_7b_relsim_1k_image.sh
```

运行：

```bash
cd ~/works/relation_hallucination/LLaVA
bash scripts/v1_5/eval/rbench_7b_relsim_1k_image.sh <GPU_ID>
```

输出：

```bash
eval_results/rbench/answers/llava-v1.5-7b_relsim-1k_image-level.json
eval_results/rbench/logs/llava-v1.5-7b_relsim-1k_image-level_infer.log
```

### 6.5 检查 R-Bench 回答是否为空

```bash
cd ~/works/relation_hallucination

python - <<'PY'
import json
from collections import Counter

files = [
    "eval_results/rbench/answers/llava-v1.5-7b_baseline_image-level.json",
    "eval_results/rbench/answers/llava-v1.5-7b_relsim-1k_image-level.json",
]

for path in files:
    counter = Counter()
    examples = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            ans = x.get("text", "").strip()
            low = ans.lower()

            if ans == "":
                counter["empty"] += 1
            elif low.startswith("yes"):
                counter["yes"] += 1
            elif low.startswith("no"):
                counter["no"] += 1
            else:
                counter["other"] += 1

            if len(examples) < 5:
                examples.append(ans)

    print("=" * 80)
    print(path)
    print(counter)
    for e in examples:
        print("-", repr(e))
PY
```

如果出现：

```text
Counter({'empty': 7787})
```

说明推理结果无效，需要重新检查脚本。

### 6.6 R-Bench 指标计算

注意：计算指标时仍然使用 R-Bench 原始 JSON 标注文件，不使用转换后的 JSONL。

Baseline：

```bash
cd ~/works/relation_hallucination/R-Bench

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
cd ~/works/relation_hallucination/R-Bench

python eval.py \
    --annotation-dir data_filterd \
    --question-file data_filterd/image-level_filterd.json \
    --question-id-file data_filterd/nocaps_image-level_rel_ids_holder.json \
    --result-file ../eval_results/rbench/answers/llava-v1.5-7b_relsim-1k_image-level.json \
    --eval_image \
    | tee ../eval_results/rbench/logs/llava-v1.5-7b_relsim-1k_image-level_metrics.log
```

---

## 7. AMBER 评估

AMBER 用于更综合地评估 hallucination，包括 existence、attribute、relation。当前建议先跑：

```text
discriminative-relation, 简写 dr
```

### 7.1 AMBER 数据路径

当前固定路径：

```bash
AMBER/data/query/query_discriminative-relation.json
AMBER/images/
```

### 7.2 转换 AMBER query 为 LLaVA JSONL

转换脚本建议放在：

```bash
scripts/convert/convert_amber_to_llava.py
```

运行：

```bash
cd ~/works/relation_hallucination

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
        p = os.path.join(imgdir, x["image"])
        if not os.path.exists(p):
            missing.append(x["image"])

print("missing images:", len(missing))
if missing:
    print(missing[:10])
else:
    print("All images exist.")
PY
```

### 7.3 AMBER baseline 推理

脚本：

```bash
LLaVA/scripts/v1_5/eval/amber_7b_dr.sh
```

运行：

```bash
cd ~/works/relation_hallucination/LLaVA
bash scripts/v1_5/eval/amber_7b_dr.sh <GPU_ID>
```

输出：

```bash
eval_results/amber/answers/llava-v1.5-7b_baseline_amber_dr_llava.jsonl
```

### 7.4 AMBER RelSim-1k 推理

脚本：

```bash
LLaVA/scripts/v1_5/eval/amber_7b_relsim_1k_dr.sh
```

运行：

```bash
cd ~/works/relation_hallucination/LLaVA
bash scripts/v1_5/eval/amber_7b_relsim_1k_dr.sh <GPU_ID>
```

输出：

```bash
eval_results/amber/answers/llava-v1.5-7b_relsim-1k_amber_dr_llava.jsonl
```

### 7.5 转换 LLaVA answer 为 AMBER 官方格式

转换脚本建议放在：

```bash
scripts/convert/convert_llava_answer_to_amber.py
```

Baseline：

```bash
cd ~/works/relation_hallucination

python scripts/convert/convert_llava_answer_to_amber.py \
    --llava-answer eval_results/amber/answers/llava-v1.5-7b_baseline_amber_dr_llava.jsonl \
    --out-file eval_results/amber/answers/llava-v1.5-7b_baseline_amber_dr.json
```

RelSim-1k：

```bash
python scripts/convert/convert_llava_answer_to_amber.py \
    --llava-answer eval_results/amber/answers/llava-v1.5-7b_relsim-1k_amber_dr_llava.jsonl \
    --out-file eval_results/amber/answers/llava-v1.5-7b_relsim-1k_amber_dr.json
```

### 7.6 AMBER 官方评估

Baseline：

```bash
cd ~/works/relation_hallucination/AMBER

python inference.py \
    --inference_data ../eval_results/amber/answers/llava-v1.5-7b_baseline_amber_dr.json \
    --evaluation_type dr \
    | tee ../eval_results/amber/logs/llava-v1.5-7b_baseline_amber_dr_metrics.log
```

RelSim-1k：

```bash
cd ~/works/relation_hallucination/AMBER

python inference.py \
    --inference_data ../eval_results/amber/answers/llava-v1.5-7b_relsim-1k_amber_dr.json \
    --evaluation_type dr \
    | tee ../eval_results/amber/logs/llava-v1.5-7b_relsim-1k_amber_dr_metrics.log
```

---

## 8. 常见问题

### 8.1 DeepSpeed 报 No slot specified

错误示例：

```text
ValueError: No slot '4' specified on host 'localhost'
```

原因：同时使用了 `CUDA_VISIBLE_DEVICES=4` 和 DeepSpeed 的 GPU slot 机制。

解决：训练脚本使用：

```bash
deepspeed --include localhost:${GPU_ID}
```

运行时不要加 `CUDA_VISIBLE_DEVICES`。

### 8.2 train_mem.py 报 flash_attn 未安装

错误原因：`train_mem.py` 默认启用 FlashAttention2，但环境没有安装 `flash_attn`。

快速解决：将训练入口从：

```bash
llava/train/train_mem.py
```

改成：

```bash
llava/train/train.py
```

### 8.3 R-Bench answer 全为空

如果 answer 文件中：

```json
"text": ""
```

说明 R-Bench demo 的 `eval_rbench.py` 在当前环境下解码异常。image-level 推荐改用 LLaVA 官方：

```bash
python -m llava.eval.model_vqa_loader
```

并确保输入 question file 是 JSONL。

### 8.4 JSON 和图片是否会错位

不会依赖文件夹顺序。LLaVA 训练时通过 JSON 中的：

```json
"image": "xxx.jpg"
```

去 `--image_folder` 下查找对应图片。只要文件存在且可打开，就不会错位。

### 8.5 微调会不会修改 Hugging Face cache 中的原始模型

不会。`.cache/huggingface/hub/` 中的原始 LLaVA 权重只会被读取，不会被覆盖。LoRA 微调结果保存到：

```bash
checkpoints/llava15_7b_relsim_lora_*/
```

---

## 9. 推荐实验顺序

新同学接手时建议按以下顺序操作：

```text
1. 激活 relhallu 环境，确认 LLaVA 可推理。
2. 检查 data/relsim_llava_1k.json 和 data/relsim_images/ 是否完整。
3. 跑 LLaVA-1.5-7B baseline 的 POPE / R-Bench / AMBER。
4. 跑 RelSim-1k LoRA 微调。
5. 使用 RelSim-1k checkpoint 重新跑 POPE / R-Bench / AMBER。
6. 对比 baseline vs RelSim-1k。
7. 扩展到 RelSim-10k / 50k。
```

已有 1k 实验的核心命令：

```bash
# 微调
cd ~/works/relation_hallucination/LLaVA
bash scripts/v1_5/finetune_lora_relsim_1k.sh 4

# POPE baseline / RelSim-1k
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope_7b.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope_7b_relsim_1k.sh

# R-Bench baseline / RelSim-1k
bash scripts/v1_5/eval/rbench_7b_image.sh 0
bash scripts/v1_5/eval/rbench_7b_relsim_1k_image.sh 0

# AMBER baseline / RelSim-1k
bash scripts/v1_5/eval/amber_7b_dr.sh 0
bash scripts/v1_5/eval/amber_7b_relsim_1k_dr.sh 0
```

---

## 10. 输出命名规范

建议统一使用：

```text
模型名_实验设置_benchmark_subset.后缀
```

示例：

```bash
llava-v1.5-7b_baseline_answers.jsonl
llava-v1.5-7b_relsim-1k_answers.jsonl
llava-v1.5-7b_baseline_image-level.json
llava-v1.5-7b_relsim-1k_image-level.json
llava-v1.5-7b_baseline_amber_dr.json
llava-v1.5-7b_relsim-1k_amber_dr.json
```

这样后续加入 10k、50k 时可以自然扩展：

```bash
llava-v1.5-7b_relsim-10k_...
llava-v1.5-7b_relsim-50k_...
```
