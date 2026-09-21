# Phase 1.5：Held-out RelSim Relation MCQ Sanity Check

请基于当前已有代码仓库实现一个 **Phase 1.5 evaluation pipeline**。

本阶段不要重新训练模型，也不要修改现有 Phase 1 checkpoint。

## 0. 实验目的

Phase 1 已完成：

- Qwen2.5-VL-3B Base
- Qwen2.5-VL-3B + Explicit-1k LoRA
- Qwen2.5-VL-3B + Typed-1k LoRA

三个模型在 R-Bench、MMRel、POPE、AMBER 等 benchmark 上几乎没有差异。

Phase 1.5 只回答一个问题：

> **Explicit-1k / Typed-1k LoRA 是否真正学到了 RelSim relational supervision？**

因此，需要使用训练过程中**从未见过的 RelSim 图片**构造一个简单、可自动评分的 relation multiple-choice test。

不要使用开放式 caption generation 作为主要指标。

---

# 1. Held-out 测试集构建

## 1.1 图片来源

RelSim 图片全集路径：

```text
/data/lht/relsim_dataset/relsim_images
```

从其中选择 **200 张训练期间没有见过的图片**。

需要先找到 Phase 1 使用的原始 RelSim-1k JSON，并读取其中所有 image filename / image id。

注意：

> 排除的是原始 1k JSON 中出现过的所有图片，而不仅仅是最终 Explicit reconstruction 成功的 955 张。

流程：

```text
RelSim image pool
        ↓
排除 Phase1 RelSim-1k 中全部 image IDs
        ↓
remaining unseen images
        ↓
固定随机种子
        ↓
随机选择 200 张
```

建议：

```text
seed = 42
```

最终生成的 200 个 sample 必须固定保存，之后 Base / Explicit / Typed 三个模型完全使用同一套测试题。

---

# 2. 找到对应的 Anonymous Caption

请先检查当前仓库和 RelSim 数据，找到完整 RelSim dataset 中：

```text
image → anonymous relational caption
```

的 annotation 文件。

不要根据图片重新调用 VLM 生成 caption。

GT relation 必须来源于 RelSim 原始 relational annotation。

每个测试样本至少应该能够得到：

```json
{
  "image": "...jpg",
  "anonymous_caption": "Tiny {Animal} balancing on a person's {Body Part}."
}
```

如果数据存在多个 annotation 文件，优先使用和 Phase 1 训练数据同源、同格式的 RelSim annotation。

---

# 3. 从 Anonymous Caption 中获得 GT Relation

目标不是让模型复现完整 anonymous caption，而是提取其中的核心 relation phrase。

例如：

```text
Tiny {Animal} balancing on a person's {Body Part}.
```

对应：

```text
balancing on
```

例如：

```text
{Person} standing beside a {Vehicle}.
```

对应：

```text
standing beside
```

例如：

```text
{Object} hanging from a {Structure}.
```

对应：

```text
hanging from
```

## 3.1 优先检查现有数据

首先检查当前 RelSim annotation 中是否已经存在：

```text
relation
predicate
relation_phrase
triplet
```

等结构化字段。

如果存在，直接使用。

## 3.2 如果没有结构化 relation 字段

不要简单通过 substring 或固定位置暴力截取。

可以实现一个独立的 relation-extraction preprocessing step，将 anonymous caption 转换成：

```json
{
  "anonymous_caption": "...",
  "relation": "balancing on"
}
```

由于本阶段只有 200 条测试数据，优先保证质量，而不是追求完全自动化。

允许：

- 使用现有仓库已有 parser；
- 使用一个严格约束的文本 LLM extraction；
- 或生成候选后人工检查。

最终 200 条的 GT relation 应人工快速复核一次。

**不要让 evaluation model 本身参与 GT relation 提取。**

---

# 4. 构建 Relation Vocabulary

从完整 RelSim annotation 集中提取所有可用 relation phrase，构成：

```text
RELATION_VOCAB
```

不是只从 200 个测试样本中构建。

需要做基础 normalization，例如：

```text
strip
lowercase
去除句末标点
合并完全重复字符串
```

但：

**不要擅自把语义不同的 relation 合并。**

可以统计：

```text
relation vocabulary size
relation frequency
top frequent relations
```

并把统计结果保存下来。

---

# 5. 为每个样本构造四选一 Relation MCQ

每张图片产生一题。

统一 question：

```text
Which relation best describes the primary relation shown in the image?

A. ...
B. ...
C. ...
D. ...

Answer with A, B, C, or D only.
```

其中：

- 1 个选项 = GT relation
- 3 个选项 = 从 `RELATION_VOCAB` 中采样的错误 relation

### 5.1 Wrong relation 的基本约束

错误关系暂时**不需要专门用 LLM 生成 hard negatives**。

直接从 relation 全集中采样即可。

但至少保证：

```text
wrong_relation != gt_relation
```

并进行 basic normalization 后再次检查不重复。

同一题四个选项必须唯一。

如果明显存在完全同义或词形变化，例如：

```text
standing beside
standing next to
```

能够简单识别时应避免同时出现。

第一版不需要构建复杂 semantic-negative pipeline。

---

# 6. 正确答案在 A/B/C/D 中严格均衡

200 道题：

```text
A = 50
B = 50
C = 50
D = 50
```

不要独立随机后期待近似均衡。

应该提前构造一个 answer-position list：

```text
50 × A
50 × B
50 × C
50 × D
```

shuffle 后分配给 200 个 sample。

固定：

```text
seed = 42
```

然后再根据正确答案位置填充三个 wrong relations。

最终要求：

```text
A: 50
B: 50
C: 50
D: 50
```

严格成立。

---

# 7. 测试集最终建议 Schema

输出一个独立 JSON / JSONL，至少包含：

```json
{
  "id": "...",
  "image": "...jpg",

  "anonymous_caption":
    "Tiny {Animal} balancing on a person's {Body Part}.",

  "gt_relation":
    "balancing on",

  "question":
    "Which relation best describes the primary relation shown in the image?",

  "options": {
    "A": "holding",
    "B": "balancing on",
    "C": "standing beside",
    "D": "hanging from"
  },

  "answer": "B",

  "source": "relsim_heldout",

  "split_seed": 42
}
```

保留 `anonymous_caption` 是为了之后人工 audit。

---

# 8. 数据质量检查

生成完 200 条后，自动输出 QC summary：

```text
Total questions
Unique images
Training overlap count
Missing images
Missing captions
Unique GT relations
Answer-position distribution
Duplicate options count
GT missing from options count
```

硬性要求：

```text
training overlap = 0
missing image = 0
GT missing = 0
duplicate option within question = 0

A/B/C/D = 50/50/50/50
```

同时随机打印至少 20 道完整题目供人工检查。

---

# 9. 模型评测

使用当前已有三个模型：

## M0 Base

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

## M1 Explicit

```text
checkpoints/qwen25vl3b_explicit_1k/
```

或根据当前仓库实际 checkpoint 路径自动定位。

## M2 Typed

```text
checkpoints/qwen25vl3b_typed_1k/
```

同样以当前仓库实际路径为准。

必须确认 LoRA adapter 正确加载到同一个 base model。

---

# 10. Inference Prompt

三个模型使用完全相同 prompt。

推荐：

```text
<image>

Which relation best describes the primary relation shown in the image?

A. {relation_A}
B. {relation_B}
C. {relation_C}
D. {relation_D}

Answer with A, B, C, or D only.
```

不要：

- 告诉模型 anonymous caption；
- 告诉模型 entity placeholder；
- 提供 relation explanation；
- 使用 chain-of-thought；
- 使用不同 system prompt。

我们测的是：

> **LoRA 是否改变模型从 image 中识别 relation 的能力。**

---

# 11. Decoding 设置

必须 deterministic：

```text
do_sample = False
temperature = 0
max_new_tokens = 很小
```

只需要生成：

```text
A
B
C
D
```

解析时允许：

```text
A
A.
Answer: A
The answer is A
```

统一 parse 为 A/B/C/D。

无法解析的回答记为：

```text
invalid
```

不要自动猜测。

---

# 12. 第一主指标：MCQ Accuracy

分别得到：

```text
Base Acc
Explicit Acc
Typed Acc
```

随机水平：

```text
25%
```

输出：

| Model | Correct | Wrong | Invalid | Accuracy |
|---|---:|---:|---:|---:|
| Base | | | | |
| Explicit-1k | | | | |
| Typed-1k | | | | |

同时输出：

```text
Explicit - Base
Typed - Base
Typed - Explicit
```

本阶段最重要的是：

```text
Explicit vs Base
Typed vs Base
```

而不是 Typed vs Explicit。

---

# 13. 第二主指标：Paired Prediction Analysis

因为三个模型回答的是**完全相同的 200 道题**，请保存每道题三个模型的 prediction。

统计：

```text
Base wrong → Explicit correct
Base correct → Explicit wrong

Base wrong → Typed correct
Base correct → Typed wrong
```

例如：

| Transition | Count |
|---|---:|
| Base wrong → Explicit correct | |
| Base correct → Explicit wrong | |
| Base wrong → Typed correct | |
| Base correct → Typed wrong | |

这比只看 overall accuracy 更能判断 LoRA 是否改变行为。

建议同时计算：

```text
Base / Explicit prediction agreement
Base / Typed prediction agreement
Explicit / Typed prediction agreement
```

如果三个模型 98% 的题都输出同一个选项，那么基本说明 LoRA 并没有明显改变 relation decision boundary。

---

# 14. 第三指标：可选的 option log-probability

如果当前 Qwen evaluation pipeline 很容易获得 logits，请额外实现。

如果实现成本很高，不要阻塞 MCQ 主实验。

对于每个 option label：

```text
A
B
C
D
```

计算条件 log probability。

得到：

```text
score(A)
score(B)
score(C)
score(D)
```

定义：

```text
GT margin =
score(correct_option)
-
max(score(wrong_options))
```

输出：

| Model | Accuracy | Mean GT Margin |
|---|---:|---:|
| Base | | |
| Explicit | | |
| Typed | | |

这一指标可以检测：

> 模型虽然最终选项没变，但 LoRA 是否提高了正确 relation 的置信度。

如果 label 并非单 token，请使用完整 candidate label 的 sequence log-likelihood，而不是假设一定是单 token。

---

# 15. 简单统计

由于三个模型评的是同一套 200 题，建议至少做：

- paired bootstrap 95% CI，比较 accuracy difference；

或者：

- McNemar test。

重点比较：

```text
Explicit vs Base
Typed vs Base
```

不要因为这是 sanity check 就只报单个 accuracy。

---

# 16. Phase 1.5 判定标准

## 情况 A：训练明确生效

例如出现：

```text
Base      48%
Explicit  62%
Typed     64%
```

或者 Explicit / Typed 相比 Base 有稳定的 positive margin improvement。

则说明：

> **LoRA 确实学到了 RelSim relation supervision。**

结合 Phase 1 外部 benchmark：

```text
Base ≈ Explicit ≈ Typed
```

可以得到重要结论：

> Positive RelSim relation SFT 能提高 held-out RelSim relation recognition，
> 但这种能力没有迁移成 relation-hallucination robustness。

这时进入下一阶段：

```text
Positive + Negative relation supervision
```

尤其考虑：

```text
hard / plausible negative relations
```

---

## 情况 B：训练基本没有生效

例如：

```text
Base      50%
Explicit  51%
Typed     50%
```

且三个模型 prediction agreement 极高。

则当前 Phase 1 不足以用于否定 relational supervision hypothesis。

更可能说明：

```text
955 samples
1 epoch
30 optimizer steps
lr = 2e-5
```

训练干预过弱。

下一步应该首先增加：

```text
training epochs / steps
或
dataset scale
```

让模型在 held-out RelSim relation probe 上出现明确 adaptation，再重新研究外部 benchmark。

---

## 情况 C：Explicit / Typed 一升一不升

例如：

```text
Base      50%
Explicit  65%
Typed     51%
```

或反过来。

这种结果非常重要。

不要立即进入 Phase 2。

先检查：

- checkpoint 是否正确；
- dataset 是否完全 matched；
- prompt 是否一致；
- relation vocabulary 是否受到 caption形式影响；
- train loss；
- LoRA 参数是否正确加载；
- relation frequency distribution。

---

# 17. 本阶段明确不要做

Phase 1.5 暂时不要：

- 训练新 checkpoint；
- 跑 Qwen2.5-VL-7B；
- 生成 hard negative training data；
- 做 Fully Anonymous；
- 做 Grounded A/B；
- 做 DPO；
- 做新的 hallucination benchmark；
- 做开放式 caption semantic judge 作为主指标。

本阶段只回答：

> **Phase 1 的两个 LoRA checkpoint 是否真的学到了 RelSim relational supervision？**

---

# 18. 最终输出

完成后请输出：

1. `heldout_200` MCQ 数据；
2. 数据 QC summary；
3. Base / Explicit / Typed 三组原始 prediction；
4. accuracy 汇总；
5. paired transition / agreement；
6. 如果容易实现则输出 GT-margin；
7. 一份简短 Markdown report。

报告最后只给出三个结论之一：

```text
Phase 1.5-A:
Training clearly effective.

Phase 1.5-B:
Training intervention too weak / no measurable adaptation.

Phase 1.5-C:
Inconsistent adaptation; inspect pipeline/checkpoints.
```

不要在这一阶段进一步解释 entity abstraction 或 hallucination mechanism。