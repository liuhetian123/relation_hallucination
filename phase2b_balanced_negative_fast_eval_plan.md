# Phase 2B 实验计划：Balanced Negative Verification + Frozen Fast-Eval Subsets

## 0. 阶段目标

Phase 2A 已经得到两个清楚的结果：

1. **Negative relation supervision 有作用。**
   - 在自建 held-out verification 上，V2/V3 相比 Base / V1 明显降低 FP、提高 TN；
   - 说明模型确实可以通过显式 `No` supervision 学会 relation rejection。

2. **这种 improvement 尚未稳定迁移到外部 relation-hallucination benchmark。**
   - R-Bench / MMRel-Adversarial 只有同方向的小幅变化；
   - Hard negative 没有稳定优于 Random negative；
   - 当前 V2/V3 的 negative train accuracy 只有约 43%–49%，说明 `No` 类仍明显欠拟合。

Phase 2B 不引入新的故事线，主要解决两个实验设计问题：

> **Q1. 当前 Positive/Negative 数量不平衡是否限制了 relation rejection 的学习？**

> **Q2. 在总训练量仍保持约 1k 的前提下，使用严格 1:1 paired Positive/Negative，并适度增加训练 epoch，能否让 negative learning 更充分，并在外部 benchmark 上产生更清晰的 transfer？**

同时，为降低反复完整评测的时间成本，本阶段建立一套：

> **固定、冻结、不可后调的 benchmark fast-eval subset**

用于后续 checkpoint screening。

---

## 1. Phase 2B 的核心控制原则

本阶段暂时只使用 **Explicit + Random Negative**。

不做：

- Typed vs Explicit；
- Hard negative 新方法；
- 7B；
- 5k 扩数据；
- DPO；
- Fully Anonymous；
- Grounded Region A/B。

原因：

Phase 2A 中 Random Negative（V2）整体上并不弱于 Hard Negative（V3），而当前最明显的 confound 是：

```text
V2:
955 Positive
534 Negative

V3:
955 Positive
490 Negative
```

即训练分布明显偏向 `Yes`。

Phase 2B 首先把这个变量控制干净。

---

## 2. Negative Pool 的结构性限制

### 2.1 当前只有 534 张训练图适合安全制造 relation negative

Phase 1.6 的 relation-span extraction 对 955 张训练图片做了定位：

```text
Total train images: 955
Unique relation span located: 534
Failed / ambiguous relation span: 421
```

负例制造要求：

> **必须能在 Explicit caption 中唯一定位 GT relation span。**

例如：

```text
Positive:
A person is holding a cup.

GT relation span:
holding
```

才能安全做：

```text
Negative:
A person is biting a cup.
```

并保证：

```text
只改 relation
其他 entity / syntax / modifier 全部不动
```

### 2.2 421 张 relation span 不稳定的样本不要强行用于 negative construction

例如：

```text
Arrangement of candles on a shelf ...
```

这种 caption 没有一个稳定、唯一、适合机械替换的 binary relation phrase。

如果硬改：

- 可能替换错误词；
- 可能修改非关系成分；
- 可能出现多个候选 span；
- 会引入额外 label noise。

因此 Phase 2B **不尝试修复这 421 张数据**。

这 421 张也不进入本轮 Positive pool。

原因是：

如果：

```text
Positive = 955
Negative = 534
```

那么 Positive 和 Negative 不仅数量不平衡，而且来自不同的 caption 子分布。

Phase 2B 要求：

> **Positive 与 Negative 来自完全相同的 relation-span-eligible images。**

---

## 3. Phase 2B 训练集：严格 Paired 1:1

使用全部 534 张 relation-span-eligible images。

每张图片生成：

```text
1 个 Positive
+
1 个 Random Negative
```

得到：

```text
534 Positive
534 Negative
----------------
1068 total
```

总训练规模仍然约为 1k。

这样可以同时做到：

- Positive : Negative = 1 : 1；
- 相同图片分布；
- 相同 entity 分布；
- 相同 caption structure；
- 每张图一正一负；
- 唯一主要变量是 relation 是否正确。

---

## 4. Negative 数据优先复用 Phase 2A V2

为了最大限度隔离“class balance”的作用：

> **Phase 2B 优先直接复用 Phase 2A V2 已经生成的 534 个 Random Negative。**

不要重新采样一批新的 negative。

即：

```text
Phase 2A V2:
955 Positive + 534 Random Negative

Phase 2B:
只保留与这 534 个 Negative 对应的 534 个 Positive
+
原来的同一批 534 Random Negative
```

因此 V2 → Phase 2B 的主要变化只有：

```text
Positive 955 -> 534
Negative 534 -> 534
```

而不是：

- 换了图片；
- 换了 negative；
- 换了 relation vocabulary；
- 换了生成策略。

这使得：

> **旧 V2 vs Phase 2B Balanced**

成为一个非常干净的 class-balance ablation。

---

## 5. Phase 2B 训练 Prompt

继续完全复用 Phase 2A verification prompt。

```text
<image>

Does the image support the relation expressed in the following statement?

Judge only whether the stated relation between the referenced entities is visually supported.
Ignore minor wording or attribute details that are not relevant to the relation.

Statement:
{statement}

Answer Yes or No only.
```

Positive target：

```text
Yes
```

Negative target：

```text
No
```

不修改 prompt。

---

## 6. 第一轮训练组

### B0：Base

不训练。

```text
Qwen2.5-VL-3B-Instruct
```

### B1：Phase 1 Explicit Caption-SFT

直接复用已有 checkpoint。

用于确认：

```text
caption learning != verification learning
```

### V2-old：Phase 2A Random Negative

直接复用：

```text
955 Positive
+
534 Random Negative
```

这是 Phase 2B 最重要的历史对照。

### V2B-1ep：Balanced Random Negative，1 epoch

新训练：

```text
534 Positive
+
534 Random Negative
=
1068 total
```

训练 1 epoch。

作用：

> **只测试 balance 本身是否有帮助。**

### V2B-2ep：Balanced Random Negative，2 epochs

使用和 V2B-1ep 完全相同的数据。

训练 2 epochs。

作用：

> 检查 Phase 2A 的 negative underfitting 是否主要来自训练不充分。

### V2B-3ep：可选

只有满足以下条件才启动：

```text
V2B-2ep 的 train negative accuracy
仍明显低于 positive accuracy
且
held-out verification 仍继续改善
```

否则不跑。

---

## 7. 训练阶段必须监控的指标

每个 epoch 至少记录：

```text
overall train loss
positive loss
negative loss
positive accuracy
negative accuracy
overall accuracy
Yes prediction ratio
```

特别关注：

```text
Positive Acc
vs
Negative Acc
```

Phase 2A：

```text
V2:
Positive Acc = 84.9%
Negative Acc = 43.3%
```

Phase 2B 的目标不是追求 100% train acc，而是希望：

```text
Negative Acc 明显上升
Positive Acc 不明显崩溃
```

---

## 8. Phase 2B 的第一道 Gate：自建 Frozen Held-out Verification

继续使用 Phase 2A 已冻结的 held-out verification set。

不要重新生成。

当前：

```text
193 Positive
100 Random Negative
100 Hard Negative
=
393 questions
```

评测：

```text
Base
V2-old
V2B-1ep
V2B-2ep
```

主要指标：

```text
Accuracy
Positive Acc
Random Negative Acc
Hard Negative Acc
FP
FN
Yes Ratio
Hard FP
```

第一道 Gate 最关注：

```text
Negative Acc ↑
Hard FP ↓
同时 Positive Acc 不大幅下降
```

---

## 9. 建立 Frozen Fast-Eval Benchmark Subsets

完整评测 R-Bench / MMRel / AMBER 的耗时较高。

Phase 2B 开始建立一个：

> **固定、冻结、后续所有实验使用同一份的 fast-eval subset。**

它只用于：

```text
checkpoint screening / model selection
```

不能代替论文最终 full benchmark。

---

## 10. Fast-Eval Subset 的三个 Benchmark

优先覆盖三个 relation-focused benchmark：

```text
1. R-Bench image-level
2. MMRel-Adversarial
3. AMBER discriminative-relation
```

MMRel normal 和 POPE-Adversarial 暂时不作为每个 checkpoint 的 fast gate。

最终候选模型仍需要跑完整：

```text
R-Bench
MMRel-Adversarial
MMRel normal
AMBER-dr
POPE-Adversarial
```

---

## 11. Subset 不建议简单全局 random，而应固定 seed + stratified sampling

目标仍然是“随机子集”，但尽量保持原 benchmark 结构。

统一使用固定：

```text
seed = 42
```

一旦生成后永久冻结。

不要因为模型在 subset 上表现不好而：

- 换 seed；
- 换样本；
- 删除难题；
- 调整比例。

### 11.1 R-Bench Fast Subset

建议取完整有效评测集约：

```text
25%
```

如果 full effective questions 约 5498，则：

```text
约 1375 questions
```

尽量保持：

- GT Yes / No 比例；
- 官方 fold 结构；
- 如果 metadata 可用，relation category / source 比例。

### 11.2 MMRel-Adversarial Fast Subset

MMRel-Adversarial 有效样本本身较少，因此比例提高。

建议：

```text
50%
```

当前有效约 770：

```text
约 385 questions
```

保持：

- Yes / No 比例；
- VG / DALL-E source 比例；
- 若有 relation category，则按 category stratify。

### 11.3 AMBER-dr Fast Subset

建议：

```text
30%
```

1664 题约：

```text
约 499 questions
```

保持原始 label 比例。

注意：

AMBER-dr 的 Precision / Recall 仍按官方定义，以 `No` 为正类。

---

## 12. Fast Subset 的代表性校验

这是必须做的，但不需要重新跑旧模型推理。

因为 Base / B1 / V1 / V2 / V3 的 full prediction 已经存在。

可以：

```text
直接从旧 prediction 中
重新 score frozen subset
```

然后比较：

```text
Full benchmark
vs
Fast subset
```

至少检查：

```text
Base
V2-old
V3
```

### 12.1 代表性要求

fast subset 不要求和 full 数值完全一致。

它只需要作为 screening gate 保持：

#### A. 大体性能水平一致

例如：

```text
subset Acc 与 full Acc
差异不要异常巨大
```

#### B. model ranking 大体一致

例如 full 上：

```text
V2 >= Base
```

subset 上最好也不要长期反转成：

```text
V2 << Base
```

#### C. 关键方向一致

重点检查：

```text
Acc direction
FP direction
Yes-ratio direction
Recall direction
```

### 12.2 不允许根据模型结果挑 seed

如果 seed=42 的 subset 存在较大 sampling noise：

> 不重新换 seed 找一个“更符合预期”的 subset。

允许的修正方式只有：

```text
提高 sampling ratio
```

例如：

```text
R-Bench 25% -> 40%
```

并继续使用同一 seed、同一 deterministic ordering 扩展。

这样避免 benchmark subset selection 变成隐性调参。

---

## 13. 两级 Evaluation Gate

从 Phase 2B 开始，每个新 checkpoint 不再立即跑全部 benchmark。

### Gate 1：超便宜

固定 393 held-out verification：

```text
Positive Acc
Negative Acc
Hard FP
Yes Ratio
```

如果没有优于 V2-old：

> 不继续。

### Gate 2：Fast External Benchmark

只有通过 Gate 1 的 checkpoint 才跑：

```text
R-Bench-fast
MMRel-Adv-fast
AMBER-dr-fast
```

如果外部 fast subset 仍：

```text
≈ Base
或
明显差于 V2-old
```

不跑 full benchmark。

### Final Gate：Full benchmark

只有最好的 1–2 个 checkpoint 才跑完整：

```text
R-Bench full
MMRel-Adversarial full
MMRel normal full
AMBER-dr full
POPE-Adversarial full
```

这样把完整验证的次数压到最少。

---

## 14. Phase 2B 的核心比较

最终主要比较：

```text
V2-old
955 Pos + 534 Neg
1 epoch
```

vs

```text
V2B-1ep
534 Pos + 534 Neg
1 epoch
```

回答：

> **class balance 是否重要？**

然后：

```text
V2B-1ep
vs
V2B-2ep
```

回答：

> **negative underfitting 是否可以通过更充分训练改善？**

这两个问题没有回答清楚之前：

> 不扩大到 5k。

---

## 15. Go / No-Go 判定

### Go A：Balance 有帮助

如果：

```text
V2B-1ep
相比 V2-old
```

出现：

- train negative acc 明显提高；
- held-out negative acc 提高；
- Hard FP 降低；
- Positive Acc 基本保持；
- fast external benchmark 至少不退化。

则认为：

> Positive / Negative imbalance 是 Phase 2A 的重要限制因素。

### Go B：更多 epoch 有帮助

如果：

```text
V2B-2ep > V2B-1ep
```

且提升主要来自：

```text
negative rejection ↑
```

而不是：

```text
全部变 No
```

则保留 2 epoch 作为后续默认设置。

### No-Go A：Balance / epoch 只改善自建 test

如果：

```text
held-out verification ↑
但是
fast external benchmark ≈ Base
```

说明核心瓶颈仍是：

> **training distribution diversity / benchmark transfer**

这时才进入下一步：

```text
扩大数据到 3k / 5k
```

而不是继续调 1k epoch。

### No-Go B：Balance 后仍学不好 No

如果：

```text
V2B-2ep
train negative acc 仍明显偏低
```

需要检查：

- negative label noise；
- random negative 是否实际上也可能成立；
- prompt 设计；
- image / statement 是否有 subject-object ambiguity。

不要直接靠更多 epoch 强行拟合。

---

## 16. 对 421 个 ineligible captions 的后续处理

Phase 2B 不处理。

它们的问题不是“模型不够强”，而是：

> 当前 caption 本身不适合做安全的 relation-span substitution。

后续若要扩大到 3k / 5k，不建议继续靠修补这类 caption。

更合理的扩展方式是寻找 / 构造：

```text
明确 subject
+
明确 relation
+
明确 object
```

的数据源或新 supervision schema。

例如未来的数据最好直接有：

```text
subject
relation
object
```

结构化字段。

这样 negative relation 可以机械构造，而不依赖从自由文本 caption 中猜 relation span。

---

## 17. 如果 Phase 2B 成功，下一阶段怎么扩数据

只有 Phase 2B 证明：

```text
balanced negative supervision
确实有稳定作用
```

才进入数据 scaling。

优先：

```text
1k -> 3k
```

而不是直接 5k / 10k。

并保持：

```text
Positive : Negative = 1 : 1
```

且尽量：

```text
一张图一正一负
```

扩大的是：

- image diversity；
- relation diversity；
- subject/object diversity；

而不是单纯给同一张图重复采更多负关系。

---

## 18. 本阶段最终输出

Cursor 最终需要输出：

1. `relation-span-eligible` 534 image ID list；
2. 1068 条 balanced train data；
3. V2B-1ep checkpoint；
4. V2B-2ep checkpoint；
5. train positive / negative accuracy；
6. frozen held-out verification 结果；
7. 三个 frozen fast benchmark subset；
8. subset QC；
9. 旧模型 full-vs-subset 代表性对照；
10. 新 checkpoint fast-eval；
11. 最优 checkpoint 的 full benchmark（仅在通过 Gate 后）；
12. Phase 2B Markdown report。

---

## 19. 本阶段最终要回答的问题

Phase 2B 只回答三个问题：

```text
Q1.
Phase 2A 的 Positive/Negative imbalance
是否限制了 rejection learning？

Q2.
在总数据仍约 1k 的情况下，
1:1 paired supervision + 更充分训练
是否能让 No 类真正学扎实？

Q3.
当 training-side rejection 明显改善后，
外部 relation hallucination benchmark
是否开始出现比 Phase 2A 更清晰的 transfer？
```

如果 Q1/Q2 成立但 Q3 仍不成立：

> 下一步问题就不再是 class balance，而是 **data diversity / cross-benchmark transfer**。

此时再进入 3k / 5k 扩数据才有充分理由。
