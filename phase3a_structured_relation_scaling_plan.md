# Phase 3A 实验计划：Structured Relation Verification Scaling（534 → 1.5k → 3k Image Pairs）

## 0. 阶段目标

经过 Phase 1 → 2B.5，目前已经得到比较稳定的结论：

1. Positive relational caption SFT 能被模型学习，并能在 RelSim-style matched caption task 上跨图片泛化；
2. Positive-only supervision 不足以教会 relation rejection；
3. 显式 Negative supervision 能显著增强 rejection；
4. 1:1 Positive/Negative balance 能明显改善自建 held-out verification；
5. 但在 R-Bench / MMRel 等外部 benchmark 上，当前约 534 个 image-pair 的训练仍主要表现为 rejection bias / calibration shift，而不是稳定的 relation discrimination improvement；
6. Phase 2B.5 已基本排除“只要换回训练 prompt 就能释放隐藏 relation ability”这一解释。

因此 Phase 3A 的核心问题是：

> 当 image / subject / object / relation diversity 显著扩大时，balanced relation verification 是否能从“prompt-dependent rejection policy”转变为真正具有 cross-benchmark transfer 的 relation discrimination ability？

本阶段重点研究 data scale / data diversity，暂时不重新引入 Typed vs Explicit、Hard Negative、7B、DPO、Fully Anonymous、Grounded A/B。

---

## 1. 核心假设

当前最合理的假设是：

> 534 张训练图足以让模型学到一个可泛化的 rejection tendency，但不足以覆盖足够丰富的 entity–relation combinations，因此无法稳定迁移到外部 relation-hallucination benchmark。

如果这个假设成立，随着训练数据的 image diversity、subject diversity、object diversity、relation diversity、subject–object combination diversity、entity–relation combination diversity 增加，外部 benchmark 应逐步出现：

```text
FP ↓
但 FN 不显著 ↑
Accuracy / F1 ↑
Recall 基本保持
```

而不是仅仅：

```text
Yes Ratio ↓
FP ↓
FN ↑↑
```

---

## 2. 为什么下一阶段不再依赖 caption relation-span extraction

当前旧数据存在结构性瓶颈：

```text
955 training images
↓
只有 534 张能够唯一定位 relation span
↓
421 张无法安全做字符串 relation replacement
```

Phase 3A 改用 Structured Subject–Relation–Object（SRO）supervision：

```json
{
  "subject": "person",
  "relation": "holding",
  "object": "cup"
}
```

这样：

- relation replacement 不再依赖字符串 span；
- Positive/Negative 可以严格 paired；
- 扩到 3k/5k 不受 534 上限影响；
- 以后重新做 Explicit vs Typed 也更容易。

---

## 3. 数据源：第一轮仍只使用 RelSim

Phase 3A 首轮不要混入 VG / MMRel / R-Bench 等 benchmark-related 数据，以避免 benchmark leakage 和 source-diversity confound。

候选图片从现有 RelSim image pool 中抽取，必须排除：

```text
Phase 1 / 2 training images
Phase 1.5 / 1.6 / 2 held-out images
```

如果本地已有外部 benchmark 图片，建议额外做 exact hash / perceptual hash dedup。

---

## 4. Structured SRO 标注

### 4.1 Teacher

继续优先复用：

```text
InternVL3.5-8B
```

### 4.2 SRO 标注 Prompt

```text
<image>

Identify one primary binary relation that is clearly and directly visible in the image.

Return exactly one JSON object:

{
  "subject": "...",
  "relation": "...",
  "object": "...",
  "confidence": "high|medium|low"
}

Requirements:
- subject and object must be concrete visible entities;
- relation must describe a direct visual relation between them;
- relation should be a short verb / verb phrase / spatial relation;
- do not output attributes, scene descriptions, counts, or vague arrangements;
- avoid relations requiring hidden intent or external knowledge;
- prefer the most visually salient subject-relation-object triplet;
- use concise basic-level entity names.
```

### 4.3 SRO 自动 QC

只保留 confidence=high 且满足：

```text
subject 非空
relation 非空
object 非空
subject != object
relation 长度合理
JSON 可解析
```

排除纯属性、纯场景描述、过度抽象关系和非 binary relation。

---

## 5. Positive Statement

统一使用 structured SRO 渲染，不再自由生成长 caption。

例如：

```text
subject = person
relation = holding
object = cup
```

渲染：

```text
A person is holding a cup.
```

所有 scale 必须使用同一个 renderer。

---

## 6. Negative Relation

Phase 3A 先只使用 Random Negative，不使用 Hard Negative，避免同时改变 negative difficulty。

保持 subject/object 不变，只替换 relation。

例如：

```text
Positive:
A person is holding a cup.

Negative:
A person is biting a cup.
```

Negative candidate 从 Phase 3A master relation vocabulary 中采样，并排除 exact duplicate、substring-equivalent、明显同义、morphology-only variant。

---

## 7. Negative 必须做 false-relation filtering

Open-world image 中“未标注”不等于“不存在”。

建议用 InternVL3.5-8B 做轻量过滤：

```text
<image>

Subject: {subject}
Object: {object}
Candidate relation: {negative_relation}

Is this relation clearly visually supported between the specified subject and object?

Answer:
Yes
No
Uncertain
```

只保留 `No`。

`Yes` / `Uncertain` 全部丢弃并重新采样。

每张图最多尝试固定次数，例如 5 次；找不到可靠 negative 的图片不进入 master pool。

---

## 8. Master Dataset：3000 Image Pairs

最终构造：

```text
3000 unique images
3000 Positive
3000 Negative
----------------
6000 training samples
```

Positive : Negative 严格 1:1，每张图恰好两条训练样本。

---

## 9. Diversity QC

必须统计：

```text
Unique images
Unique normalized relations
Unique subjects
Unique objects
Unique subject-object pairs
Unique subject-relation pairs
Unique relation-object pairs
Unique S-R-O triplets
```

同时输出 relation frequency distribution。

建议任何单一 normalized relation 不超过总 image-pair 的 3%–5%，避免 scaling 只是在重复 holding / standing / sitting 等高频关系。

---

## 10. 建立 Nested Scaling Splits

从同一份 3000-pair master pool 中构造三个嵌套规模：

### S534

```text
534 image pairs
534 Positive
534 Negative
1068 total
```

### S1500

```text
1500 image pairs
1500 Positive
1500 Negative
3000 total
```

且包含全部 S534。

### S3000

```text
3000 image pairs
3000 Positive
3000 Negative
6000 total
```

且包含全部 S1500。

即：

```text
S534 ⊂ S1500 ⊂ S3000
```

固定 seed=42，一旦生成永久冻结。

---

## 11. 为什么要重新训练 Structured-S534

不要直接把旧 V2B-2ep 当作正式 scaling curve 的唯一 534 baseline。

旧 V2B 的 statement 来自 Explicit caption + relation-span replacement，而新 3k 使用 structured SRO renderer。

因此建议重新训练：

```text
Structured-S534
Structured-S1500
Structured-S3000
```

旧 V2B-2ep 只作为 historical reference。

这样可以区分 pipeline change 和 scale change。

---

## 12. 训练 Prompt

完全复用 Phase 2A / 2B verification prompt：

```text
<image>

Does the image support the relation expressed in the following statement?

Judge only whether the stated relation between the referenced entities is visually supported.
Ignore minor wording or attribute details that are not relevant to the relation.

Statement:
{statement}

Answer Yes or No only.
```

Target 仅为 Yes / No。

---

## 13. 模型与 LoRA

继续：

```text
Qwen2.5-VL-3B-Instruct
```

LoRA 保持：

```text
r = 16
alpha = 32
vision frozen
q/k/v/o/gate/up/down
```

optimizer / batch / dtype 尽量沿用 Phase 2B。

暂时不上 7B。

---

## 14. 主 Scaling 训练：相同 epoch

正式 scaling curve 使用每个样本相同 exposure：

```text
S534-1ep
S1500-1ep
S3000-1ep
```

三个 run 都必须是独立的 `num_train_epochs=1`。

不能再用一个 2-epoch run 的中途 checkpoint 充当 1-epoch 主对照。

每个 run：

- 独立 scheduler；
- scheduler total steps 与各自 1 epoch 对齐；
- 相同 seed；
- 相同超参。

---

## 15. Secondary Control：S3000 Step-Matched

同 epoch 时，dataset 越大，optimizer steps 越多。因此增加一个 secondary diagnostic：

```text
S3000-stepmatched
```

设置：

```text
max_steps = S534-1ep 的 optimizer steps
```

从完整 S3000 pool 随机 shuffle 取 batch。

如果 S3000-stepmatched 已优于 S534，说明更大 pool / diversity 本身有贡献，而不只是更多 optimizer updates。

该模型只做 diagnostic，不进入主 scaling curve。

---

## 16. Training Diagnostics

每个 checkpoint 至少记录：

```text
overall train loss
positive loss
negative loss
Positive Acc
Negative Acc
Overall Acc
Yes Ratio
```

重点检查 Positive Acc 与 Negative Acc 是否严重失衡。

如果 Negative Acc 很高但 Positive Acc 明显崩溃，不能视为成功。

---

## 17. Gate 1：Frozen Custom Held-out Verification

继续使用已有冻结：

```text
193 Positive
100 Random Negative
100 Hard Negative
=
393
```

不重新生成。

报告：

```text
Acc
Pos Acc
Random Neg Acc
Hard Neg Acc
Hard FP
Yes Ratio
TP/TN/FP/FN
```

Gate 1 主要检查模型是否学坏或严重 No-bias，不作为最终 transfer claim。

---

## 18. Gate 2：Frozen External Fast Subsets

继续使用 Phase 2B 已冻结的：

```text
R-Bench-fast
MMRel-Adversarial-fast
AMBER-dr-fast
```

不换 seed，不重新抽题。

Phase 3A 的 external screening 优先使用官方 prompt。

Phase 2B.5 的 matched prompt 不作为 scaling 主评测。

---

## 19. Fast Benchmark 的核心评价原则

不能再把 FP 越低越好作为单独成功标准。

必须联合看：

```text
Accuracy ↑
F1 ↑
FP ↓
FN 不显著 ↑
Recall 基本保持
Yes Ratio 不异常漂移
```

真正 discrimination improvement：

```text
FP ↓
FN ≈
Acc ↑
```

rejection-bias improvement：

```text
FP ↓↓↓
FN ↑↑
Acc ≈ / ↓
```

后者不算成功。

---

## 20. Scale Trend 分析

建议输出：

| Scale | Image Pairs | Train Samples | R-Bench Acc | R-Bench F1 | MMRel Acc | MMRel F1 | AMBER Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| S534 | 534 | 1068 | | | | | |
| S1500 | 1500 | 3000 | | | | | |
| S3000 | 3000 | 6000 | | | | | |

同时输出 FP / FN / Yes Ratio。

---

## 21. Statistics

重点比较：

```text
S534 vs S1500
S534 vs S3000
S1500 vs S3000
```

在同一 frozen fast subset 上计算：

```text
paired bootstrap 95% CI
McNemar
```

不能只看 raw difference。

---

## 22. Fast Subset 的分辨率限制

Phase 2B 已经知道 R-Bench-fast 对 <0.5 pp 的差异分辨能力有限。

因此：

```text
+0.1 / +0.2 pp
```

不解释成 scaling success。

建议：

### Strong Go

至少两个 relation benchmark 出现：

```text
Acc / F1 >= +1.0 pp
```

或 paired CI 明显偏正，同时 FP 降、FN/Recall 不恶化。

### Weak Go

如果 +0.5～1.0 pp，但三个 benchmark 方向一致、confusion matrix 健康，也可以进入 full evaluation 验证。

---

## 23. Full Benchmark Gate

只有最好的 1–2 个 checkpoint 才跑完整：

```text
R-Bench full
MMRel-Adversarial full
MMRel normal full
AMBER-dr full
POPE-Adversarial full
```

不要所有 scale 全量跑。

POPE 继续作为 generic hallucination control。

---

## 24. Go / No-Go

### Go A：Scale 增大后外部 discrimination 明显提升

如果：

```text
S534 < S1500 < S3000
```

或至少：

```text
S3000 clearly > S534
```

同时：

```text
FP ↓
FN 不大幅 ↑
Acc / F1 ↑
Recall 稳定
```

则支持：

> 当前主要瓶颈确实是 relation/entity/image diversity 不足。

下一步可考虑 3k→5k，或者重新加入 Explicit vs Typed。

### Go B：S3000-stepmatched 已优于 S534

说明更大数据 pool / diversity 本身有贡献，而不只是更多训练 step。

### No-Go A：Custom held-out 继续提升，但 external 仍 ≈ Base

说明单纯扩大同源 RelSim verification data 仍无法解决 cross-benchmark gap。

此时不要继续 3k→5k→10k，应转向 source diversity / benchmark-independent structured relation data / grounded or counterfactual relation supervision。

### No-Go B：Scale 越大，No-bias 越强

如果 FP 降但 FN 大涨、Yes Ratio 大跌、Acc 不升，说明 scaling 只是在强化 rejection policy，应停止扩大数据。

### No-Go C：Structured-S534 与旧 V2B 差异巨大

说明 SRO pipeline / renderer 本身是强变量。先分析 teacher quality、renderer style、relation distribution、negative filtering，再解释 scaling。

---

## 25. 数据人工 QC

正式训练前至少抽查 200 image pairs。

检查：

```text
subject 是否正确
object 是否正确
positive relation 是否真实
negative relation 是否确实不成立
positive/negative 是否只差 relation
statement 是否自然且语义明确
```

高频 / 中频 / 低频 relation 都要覆盖。

---

## 26. Master Dataset 建议字段

```json
{
  "id": "scale_xxx",
  "image": "xxx.jpg",
  "subject": "person",
  "relation": "holding",
  "object": "cup",
  "positive_statement": "A person is holding a cup.",
  "negative_relation": "biting",
  "negative_statement": "A person is biting a cup.",
  "sro_teacher": "InternVL3.5-8B",
  "sro_confidence": "high",
  "negative_teacher_judgment": "No",
  "split_seed": 42
}
```

再由程序展开为一条 Positive、一条 Negative。

---

## 27. 推荐目录

```text
data/phase3a/
    master_sro_3000.jsonl
    scale_534.jsonl
    scale_1500.jsonl
    scale_3000.jsonl
    qc/

checkpoints/
    qwen25vl3b_phase3a_s534/
    qwen25vl3b_phase3a_s1500/
    qwen25vl3b_phase3a_s3000/
    qwen25vl3b_phase3a_s3000_stepmatched/

eval_results/qwen/phase3a/
    gate1/
    fast/
    full/
    metrics/

scripts/phase3a/
```

---

## 28. Cursor 顶层任务

请基于现有 Phase 2B / 2B.5 工程完成 Phase 3A：

1. 从 RelSim image pool 建立候选池；
2. 排除已有 train / held-out；
3. 用 InternVL3.5-8B 标注 structured SRO；
4. 做 SRO QC；
5. 统一 renderer 生成 Positive statement；
6. 建立 relation vocabulary；
7. 采 Random Negative；
8. 用 InternVL 过滤 candidate negative，只保留明确 `No`；
9. 构建 3000 image-pair master pool；
10. 做 diversity statistics；
11. 生成 nested S534 / S1500 / S3000；
12. 训练：
    - S534-1ep
    - S1500-1ep
    - S3000-1ep
    - S3000-stepmatched
13. 每个 checkpoint 先跑 frozen 393 held-out；
14. 通过 Gate 后跑 frozen external fast subset；
15. 做 paired statistics；
16. 只有最优 1–2 个 checkpoint 才跑 full benchmark；
17. 输出 Markdown report。

---

## 29. 本阶段最重要的科学问题

Phase 3A 最终只回答：

> 把 balanced relation verification 从约 500 image pairs 扩大到 1500 / 3000 image pairs，并显著增加 relation/entity/image diversity 后，能否把当前 prompt-dependent rejection behavior 转化成 cross-benchmark relation discrimination improvement？

真正成功的标志不是模型更爱回答 `No`，而是模型能更准确地区分哪些 relation 应该 Yes、哪些应该 No，并且这种 improvement 能出现在 R-Bench、MMRel、AMBER 等外部数据上。
