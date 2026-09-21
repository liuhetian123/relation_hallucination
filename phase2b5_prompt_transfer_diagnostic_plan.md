# Phase 2B.5 实验计划：Prompt Transfer Diagnostic on External Relation Benchmarks

## 0. 本阶段目的

Phase 2B 已经证明：

- 1:1 Positive/Negative balance 能明显改善自建 held-out verification；
- V2B-2ep 相比 V2-old：
  - held-out Acc：64.6% → 70.2%
  - Hard FP：44% → 25%
  - Yes ratio：52.4%，接近 held-out gold Yes 比例
- 但是外部 fast benchmark 上：
  - R-Bench ≈ Base
  - MMRel-Adversarial ≈ Base
  - AMBER-dr ≈ Base / 略差

因此当前还不能直接判断：

> **外部 transfer 失败是因为训练数据分布太窄，还是因为外部 benchmark 的 prompt / task formulation 与训练 verification prompt 不一致。**

Phase 2B.5 不重新训练。

本阶段只回答：

> **当外部 benchmark 使用与 Phase 2B 训练完全一致的 verification prompt 时，V2B-2ep 是否开始明显优于 Base？**

这一步用于区分：

```text
Prompt-transfer failure
vs
Data/distribution-transfer failure
```

---

## 1. 使用模型

只评三个模型：

### M0：Base

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

### M1：V2-old

Phase 2A：

```text
955 Positive
+
534 Random Negative
1 epoch
```

### M2：V2B-2ep

Phase 2B 主 checkpoint：

```text
534 Positive
+
534 Random Negative
2 epochs
```

暂时不评：

- V1
- V3
- V2B-1ep
- Caption-SFT
- Typed
- 7B

---

## 2. 使用 benchmark

优先只使用已经冻结的 fast subset：

```text
1. R-Bench-fast
2. MMRel-Adversarial-fast
```

AMBER-dr 暂时作为可选项。

原因：

- R-Bench / MMRel 都直接测试 relation existence / correctness；
- 更容易转换为 statement-based verification；
- Phase 2B fast subset 已冻结，可以直接复用；
- 不需要重新抽样。

不要创建新的 subset。

---

## 3. 两套 Prompt

对每个 benchmark sample，必须保留完全相同的：

```text
image
relation statement / semantic content
gold label
```

只改变 prompt formulation。

### 3.1 Prompt A：Official / Existing Evaluation Prompt

直接复用 Phase 2B 当前的 benchmark evaluation prompt。

不要修改。

输出：

```text
Yes / No
```

已有结果直接复用，不重复推理。

### 3.2 Prompt B：Training-Matched Verification Prompt

使用与 Phase 2B 训练完全一致的模板：

```text
<image>

Does the image support the relation expressed in the following statement?

Judge only whether the stated relation between the referenced entities is visually supported.
Ignore minor wording or attribute details that are not relevant to the relation.

Statement:
{statement}

Answer Yes or No only.
```

Assistant 输出：

```text
Yes
```

或：

```text
No
```

---

## 4. 最关键的问题：如何构造 `{statement}`

Phase 2B.5 的科学有效性主要取决于这里。

要求：

> **只改变 prompt format，不改变 benchmark 原始 relation semantics。**

不能让 LLM 根据图片重新生成 statement。

不能改变 gold relation。

不能加入新的实体或关系信息。

---

## 5. R-Bench statement conversion

优先检查 R-Bench 原始样本结构。

如果原始问题类似：

```text
Is the man riding the horse?
```

则转换成：

```text
The man is riding the horse.
```

然后放进 training-matched prompt。

### 转换规则优先顺序

#### A. 使用 benchmark 已有 structured fields

如果 R-Bench 有：

```text
subject
relation
object
```

则程序化生成：

```text
{subject} {relation} {object}.
```

这是最优方式。

#### B. 使用已有 declarative statement / caption 字段

如果 benchmark 本身已有 relation statement：

直接使用。

#### C. 只有 yes/no question 时做 deterministic question-to-statement conversion

例如：

```text
Is the dog under the table?
```

转换：

```text
The dog is under the table.
```

不要使用自由生成 VLM。

如果必须用文本 LLM：

- 输入只包含原始 question；
- 不提供 image；
- 要求语义完全不变；
- 输出单句 declarative statement；
- 保存原始 question 与 converted statement；
- 至少人工检查 100 条。

---

## 6. MMRel-Adversarial statement conversion

优先使用 MMRel 原始结构化 relation 信息。

如果存在：

```text
subject
predicate
object
```

则直接构造：

```text
{subject} {predicate} {object}.
```

如果只有 yes/no relation question，则使用同样 deterministic conversion。

要求：

- 不改 relation predicate；
- 不改 subject；
- 不改 object；
- 不加入新的属性；
- gold label 与官方完全一致。

---

## 7. Statement Conversion QC

每个 benchmark 输出：

```text
Total
Converted successfully
Failed conversion
Missing subject/relation/object
Ambiguous conversion
Duplicate statement
```

硬要求：

```text
Gold label unchanged = 100%
Image unchanged = 100%
```

随机人工检查至少：

```text
R-Bench: 100
MMRel-Adv: 100
```

review 文件至少包含：

```json
{
  "image": "...",
  "official_question": "...",
  "matched_statement": "...",
  "gold": "Yes/No"
}
```

重点人工确认：

> Official question 和 matched statement 是否表达同一个 relation proposition。

---

## 8. 推理设置

三个模型、两种 prompt 必须：

```text
do_sample = False
temperature = 0
same max_new_tokens
same image preprocessing
same dtype
same chat template
```

要求模型：

```text
Answer Yes or No only.
```

解析：

```text
Yes
No
Invalid
```

invalid 不强行映射。

---

## 9. 核心实验设计

形成一个 2 × 3 对照：

| Prompt | Base | V2-old | V2B-2ep |
|---|---|---|---|
| Official | ✓ | ✓ | ✓ |
| Training-Matched | ✓ | ✓ | ✓ |

最重要的是看：

```text
(V2B-2ep - Base)_Official
```

和：

```text
(V2B-2ep - Base)_Matched
```

是否明显不同。

---

## 10. 指标

每个 benchmark、每种 prompt、每个模型都报告：

```text
Accuracy
Precision
Recall
F1
Yes Ratio
TP
TN
FP
FN
Invalid
```

重点关注：

```text
FP
TN
Yes Ratio
Recall
```

不能只看 Accuracy。

---

## 11. 额外必须计算：Prompt Sensitivity

对同一个模型、同一批样本，比较：

```text
Official prediction
vs
Matched prediction
```

计算：

```text
prediction agreement
Official correct -> Matched wrong
Official wrong -> Matched correct
Yes -> No
No -> Yes
```

分别对：

```text
Base
V2-old
V2B-2ep
```

这样可以判断：

> 是否只有 LoRA 对 prompt formulation 特别敏感。

---

## 12. 最重要的差分分析

定义：

```text
Delta_model(prompt)
=
Acc(LoRA, prompt) - Acc(Base, prompt)
```

以及：

```text
Delta_FP(prompt)
=
FP(LoRA, prompt) - FP(Base, prompt)
```

比较：

```text
Delta_model(Matched)
vs
Delta_model(Official)
```

如果：

```text
Matched 下 LoRA 优势明显扩大
```

说明 training-style prompt 能恢复一部分 learned rejection。

---

## 13. Paired Statistics

因为同一题在两个 prompt 下都被预测，使用 paired analysis。

至少计算：

```text
paired bootstrap 95% CI
McNemar test
```

重点比较：

```text
Base vs V2B-2ep
```

分别在：

```text
Official
Matched
```

两套 prompt 下。

不要只报 raw Acc difference。

---

## 14. 判定逻辑

### Case A：Prompt transfer 是主要问题

如果：

```text
Official:
V2B-2ep ≈ Base

Matched:
V2B-2ep clearly > Base
```

同时：

- FP 明显下降；
- TN 上升；
- Recall 没有严重崩溃；
- improvement 明显大于 Official；

则判断：

> **Negative relation verification 已经能跨图片 / benchmark relation content transfer，但对 instruction / prompt formulation 缺乏鲁棒性。**

下一步不优先扩数据。

应优先研究：

```text
multi-prompt verification training
prompt augmentation
instruction diversity
```

---

### Case B：Matched prompt 下仍然 ≈ Base

如果：

```text
Official:
V2B-2ep ≈ Base

Matched:
V2B-2ep ≈ Base
```

则判断：

> **外部失败不是主要由 prompt mismatch 导致，而更可能来自 image / entity / relation distribution diversity 不足。**

这时进入：

```text
3k paired 1:1 structured relation verification
```

下一阶段重点扩大：

- image diversity；
- subject/object diversity；
- relation diversity；
- source diversity。

---

### Case C：Matched prompt 让所有模型一起变化，但 LoRA 相对 Base 不变

例如：

```text
Base Official 69
Base Matched 72

V2B Official 69
V2B Matched 72
```

说明：

> matched prompt 本身更适合 benchmark，但没有释放 LoRA 特有的 learned rejection。

仍应归入：

```text
data/distribution transfer bottleneck
```

---

### Case D：Matched prompt 只恢复 No-bias

例如：

```text
FP ↓
但 FN ↑↑
Acc ≈ / ↓
```

则说明：

> training prompt 只是重新激活 learned rejection bias，而不是恢复真正的 relation discrimination。

不能解释为成功 transfer。

---

## 15. AMBER-dr 可选扩展

只有当 R-Bench / MMRel 中出现 Case A 倾向时，再做 AMBER-dr。

如果做：

- 保持 gold labels；
- statement conversion 规则固定；
- 报告官方 metric；
- 额外报告 Yes ratio / confusion matrix。

如果 R-Bench + MMRel 都是 Case B，则无需再跑 AMBER。

---

## 16. 本阶段不做

Phase 2B.5 明确不做：

- 新训练；
- 3k / 5k 数据；
- Typed；
- Hard Negative 新方法；
- 7B；
- DPO；
- Fully Anonymous；
- Grounded A/B；
- 修改 frozen fast subset；
- 根据结果换 seed；
- benchmark full evaluation。

这一步必须保持为纯 diagnostic。

---

## 17. 输出文件

至少输出：

1. `rbench_fast_matched_prompt.jsonl`
2. `mmrel_adv_fast_matched_prompt.jsonl`
3. statement conversion QC
4. 200 条人工 review 样本
5. Base / V2-old / V2B-2ep predictions
6. Official vs Matched metrics
7. prompt sensitivity transitions
8. paired bootstrap / McNemar
9. Phase 2B.5 Markdown report

建议路径：

```text
data/phase2b5/
eval_results/qwen/phase2b5/
scripts/phase2b5/
```

---

## 18. 最终主表

| Benchmark | Prompt | Base Acc | V2-old Acc | V2B-2ep Acc | V2B−Base | Base FP | V2B FP |
|---|---|---:|---:|---:|---:|---:|---:|
| R-Bench-fast | Official | | | | | | |
| R-Bench-fast | Matched | | | | | | |
| MMRel-Adv-fast | Official | | | | | | |
| MMRel-Adv-fast | Matched | | | | | | |

再附：

| Model | Benchmark | Prompt Agreement | Official Wrong→Matched Correct | Official Correct→Matched Wrong | Yes→No | No→Yes |
|---|---|---:|---:|---:|---:|---:|
| Base | R-Bench | | | | | |
| V2-old | R-Bench | | | | | |
| V2B-2ep | R-Bench | | | | | |
| Base | MMRel | | | | | |
| V2-old | MMRel | | | | | |
| V2B-2ep | MMRel | | | | | |

---

## 19. 最终只回答一个问题

Phase 2B.5 最终不要扩展结论。

只回答：

> **Phase 2B learned relation rejection 没有迁移到外部 benchmark，主要是因为 prompt formulation mismatch，还是因为 underlying data / relation distribution transfer 不足？**

最终结论只能归入：

```text
Phase 2B.5-A:
Prompt-transfer bottleneck is substantial.

Phase 2B.5-B:
Prompt matching does not recover LoRA advantage;
data/distribution transfer is the main bottleneck.

Phase 2B.5-C:
Matched prompt mainly induces rejection bias without
improving relation discrimination.
```

如果是 B：

> 下一步进入 3k paired 1:1 structured relation verification。

如果是 A：

> 下一步优先做 multi-prompt / instruction-diverse verification training，而不是先扩大数据规模。
