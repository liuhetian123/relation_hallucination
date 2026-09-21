# Phase 4A-Data 实验计划：InternVL 驱动的 Clean Counterfactual Negative Construction

## 0. 阶段目标

本阶段不训练模型，先解决当前 SRO3000 中 Negative Relation 质量不干净的问题。

现有 Negative 主要通过：

```text
从全局 relation vocabulary 随机抽取 relation
→ 替换 GT relation
```

得到。

这种做法会产生大量语言上明显不合理的 statement，例如：

```text
Positive:
A burger is on the building.

Dirty Negative:
A burger is inserting a building.
```

这种 Negative 存在严重 linguistic shortcut：

> 模型不需要看图，仅根据语言常识或语法异常就可以判断为 `No`。

这与后续目标冲突：减少 MLLM 对语言 shortcut 的依赖，迫使模型更多使用视觉关系证据。

因此本阶段将 Negative 构造方式改为：

> **让 InternVL3.5-8B 基于当前图像、subject、object 和真实 relation，生成“语言上合理但当前图像中不成立”的 counterfactual relation。**

---

## 1. 核心数据定义

对于真实 SRO：

```text
(s, r+, o)
```

希望构造：

```text
(s, r-, o)
```

其中 `r-` 必须同时满足两点。

### 1.1 Linguistically Plausible

```text
Plausible(s, r-, o) = True
```

要求：

- 语法自然；
- subject / relation / object 语义兼容；
- 在某个现实或视觉场景中可能成立；
- 不能仅凭常识直接排除。

例如：

```text
burger inside building
burger beside building
burger in front of building
```

都是潜在合理的。

而：

```text
burger inserting building
chair eating person
car wearing person
```

属于明显不合理，应排除。

### 1.2 Visually False

同时要求：

```text
VisualSupport(I, s, r-, o) = False
```

即该 relation 虽然语言上合理，但在当前图片里不成立。

理想 Negative 是：

```text
language plausible
+
image false
```

而不是：

```text
language absurd
+
image false
```

---

## 2. 为什么不能继续从全局 Relation Vocabulary 随机抽取

当前随机 relation replacement 的问题是 relation 与 entity type 不兼容。

例如：

```text
Food + Building
```

全局随机可能得到：

```text
inserting
wearing
riding
holding
```

这些 relation 本身就不适合当前实体组合。

这样模型容易学到：

```text
P(No | subject, relation, object)
```

而不是：

```text
P(No | image, subject, relation, object)
```

因此 Phase 4 开始后，Negative relation 不再通过全局随机采样产生。

---

## 3. 新的 Negative Construction Pipeline

```text
Image + GT SRO
      ↓
InternVL3.5-8B
      ↓
生成多个 visually-false but linguistically-plausible
counterfactual relation candidates
      ↓
规则过滤
      ↓
保留 high-confidence candidate
      ↓
Programmatic Renderer
      ↓
Clean Positive / Negative Pair
```

---

## 4. 输入数据

直接复用 Phase 3A 的 SRO3000 master pool。

每条至少包含：

```json
{
  "image": "xxx.jpg",
  "subject": "burger",
  "relation": "on",
  "object": "building",
  "positive_statement": "A burger is on the building."
}
```

本阶段只重新构造：

```text
negative_relation
negative_statement
```

不重新标注：

```text
subject
positive_relation
object
```

---

## 5. InternVL3.5-8B 的角色

InternVL 只负责：

> **根据当前图像与真实 SRO，提出合理的 counterfactual relation。**

不要让 InternVL：

- 改 subject；
- 改 object；
- 改 GT relation；
- 自由重写整条 caption；
- 添加新的实体；
- 添加 attribute；
- 添加 scene description。

最终训练 statement 仍由程序统一 renderer 生成。

---

## 6. 主 Prompt：Counterfactual Relation Candidate Generation

推荐第一版 Prompt：

```text
<image>

You are constructing counterfactual relation examples for visual relation verification.

The following relation is TRUE in the image:

Subject: {subject}
True relation: {positive_relation}
Object: {object}

Generate THREE alternative relations between exactly the SAME subject and object.

Each alternative relation must satisfy ALL of the following:

1. It must be grammatically natural with the given subject and object.
2. It must be semantically plausible in some realistic or visually possible situation.
3. It must NOT be obviously false from language or common sense alone.
4. It must NOT be visually supported in the current image.
5. It must keep exactly the same subject and object roles.
6. Do not change, add, remove, rename, or replace the subject or object.
7. Do not output a synonym, paraphrase, inverse-expression, or trivial morphological variant of the true relation.
8. Prefer relations that are visually confusable with the true relation or plausible for the same entity pair.
9. The negative should require looking at the image to determine that it is false.
10. Avoid absurd or selectionally incompatible relations.

Bad examples:
- "a burger inserting a building"
- "a chair eating a person"
- "a car wearing a person"

Return exactly one JSON object:

{
  "candidates": [
    {
      "relation": "...",
      "statement": "...",
      "linguistically_plausible": true,
      "visually_supported": false,
      "confidence": "high|medium|low"
    },
    {
      "relation": "...",
      "statement": "...",
      "linguistically_plausible": true,
      "visually_supported": false,
      "confidence": "high|medium|low"
    },
    {
      "relation": "...",
      "statement": "...",
      "linguistically_plausible": true,
      "visually_supported": false,
      "confidence": "high|medium|low"
    }
  ]
}
```

---

## 7. 为什么一次生成 3 个 Candidate

不要强迫 Teacher 每张图只生成一个 Negative。

原因：

- 单个 candidate 容易模糊；
- 某个 relation 可能实际也成立；
- 某个 relation 可能语言上合理但视觉上无法确认；
- 多 candidate 可以提高 clean negative 的成功率。

程序按顺序选择第一个满足：

```text
linguistically_plausible = true
visually_supported = false
confidence = high
```

的 candidate。

如果 3 个都不满足：

```text
重新生成一次
```

若固定重试次数后仍无可靠 candidate：

```text
drop this sample
```

不要为了保证 3000 条而强行保留低质量 Negative。

---

## 8. Negative Candidate 的禁止项

### 8.1 不允许 GT Synonym / Paraphrase

例如：

```text
GT:
next to

Negative:
beside
```

不能作为 Negative。

需要建立 relation synonym normalization，例如：

```text
next to ↔ beside
on top of ↔ on
holding ↔ grasping
looking at ↔ watching
```

### 8.2 不允许仅做 Morphological Variant

例如：

```text
hold
holding
held by
```

不能互相作为 negative。

### 8.3 不允许语义不变的 Inverse Expression

例如：

```text
person holding cup
```

不能把：

```text
cup held by person
```

当作 Negative。

### 8.4 不允许 Ambiguous Relation

例如：

```text
person near car
```

Teacher 生成：

```text
person in front of car
```

但图片中 front/behind 无法可靠判断。

这种 candidate 应标为：

```text
confidence = medium / low
```

并丢弃。

---

## 9. 推荐增加二次 Verification

第一轮可直接依赖 InternVL 同一 Prompt 的：

```text
linguistically_plausible
visually_supported
confidence
```

如果 QC 发现误判较多，再增加独立 verification pass。

### 9.1 Text-Only Plausibility Filter

不给图片。

Prompt：

```text
Subject: {subject}
Relation: {candidate_relation}
Object: {object}

Statement:
{candidate_statement}

Question:
Is this statement grammatically natural and semantically plausible in at least one realistic or visually possible situation?

Do NOT judge whether it is true in any specific image.

Answer exactly one:
Plausible
Implausible
```

只保留：

```text
Plausible
```

注意：

```text
horse inside building
```

虽然不常见，但语义上可能成立，应判断为 Plausible。

### 9.2 Visual Falsity Filter

给图片。

Prompt：

```text
<image>

Subject: {subject}
Object: {object}
Candidate relation: {candidate_relation}

Question:
Is the candidate relation clearly visually supported between the specified subject and object in this image?

Answer exactly one:
Yes
No
Uncertain
```

只保留：

```text
No
```

丢弃：

```text
Yes
Uncertain
```

---

## 10. 正式 Negative 的生成

InternVL 的 `statement` 只用于 QC。

正式训练文本不要直接使用 Teacher 自由输出。

只保存：

```text
negative_relation
```

然后通过与 Positive 完全相同的 renderer 生成：

```text
negative_statement
```

例如：

```text
subject = burger
negative_relation = inside
object = building
```

统一生成：

```text
A burger is inside the building.
```

这样 Positive 与 Negative：

- grammar style 一致；
- 信息量一致；
- lexical style 一致；
- 只有 relation 不同。

---

## 11. 建议输出字段

新的 Clean SRO 数据：

```json
{
  "id": "xxx",
  "image": "xxx.jpg",

  "subject": "burger",
  "positive_relation": "on",
  "object": "building",

  "positive_statement": "A burger is on the building.",

  "negative_relation": "inside",
  "negative_statement": "A burger is inside the building.",

  "negative_source": "InternVL3.5-8B",
  "linguistically_plausible": true,
  "visually_supported": false,
  "negative_confidence": "high",

  "candidate_rank": 1,
  "negative_candidates_raw": []
}
```

---

## 12. 人工 QC

正式重训前，至少随机抽：

```text
300 pairs
```

建议分层抽样：

```text
100 spatial relations
100 action / interaction relations
100 other relations
```

人工检查：

1. Positive relation 是否真实；
2. Negative relation 是否语言自然；
3. Negative 是否语义 plausible；
4. Negative 是否确实不成立；
5. 是否需要看图才能判断；
6. 是否为 GT synonym；
7. 是否出现 role reversal；
8. 是否存在无法确定的视觉 ambiguity。

建议统计：

```text
Naturalness pass rate
Plausibility pass rate
Visual falsity pass rate
Overall clean rate
```

目标最好：

```text
Overall clean rate >= 90%
```

如果低于 90%，先修改 prompt / filter，不进入 Phase 4A 训练。

**2026-09-16 决定：跳过本节人工 QC。** 分层抽样表 `data/phase4a/qc/review_300.tsv` 与 `review.html` 仍保留，但不填写、不作为训练门槛。后续训练以自动 filter + text-only shortcut 为准。

---

## 13. Text-Only Shortcut QC

这是本阶段非常重要的诊断。

目的：

> 判断 Negative 是否仍然可以仅靠文本被识别。

构造 balanced test：

```text
Positive statements
Negative statements
```

不提供图片。

让 text-only model 或 VLM text-only 模式判断 statement 的语言合理性 / 可发生性。

目标不是严格要求 Accuracy = 50%，但 Clean Negative 的 text-only separability 应显著低于旧 Dirty Negative。

例如理想情况：

```text
Old random negatives:
Text-only Acc = 75%+

Clean negatives:
Text-only Acc ≈ 50%–60%
```

---

## 14. Old vs Clean Negative 对比实验

进入 Consistency 方法实验之前，建议做一个便宜 sanity check。

固定：

```text
Qwen2.5-VL-3B
3000 image pairs
1:1 Positive/Negative
same LoRA
same prompt
same epoch
```

只比较：

```text
Old-S3000
= old random / dirty negatives

Clean-S3000
= InternVL plausible counterfactual negatives
```

目的不是作为主论文实验，而是确认：

> Clean Negative 是否改变 relation verification 学习质量。

评测继续用：

```text
Custom held-out
R-Bench-fast
MMRel-Adversarial-fast
AMBER-dr-fast
```

同时报告：

```text
Acc
F1
Precision
Recall
Yes Ratio
FP
FN
Text-only shortcut score
```

---

## 15. 结果解释

### Case A：Clean Negative 明显更好

如果：

```text
Text-only shortcut ↓
+
External relation benchmark ↑
+
FP/FN 更健康
```

说明：

> 旧数据中确实存在 linguistic shortcut，plausible counterfactual negative 能提高视觉 relation supervision 质量。

后续 Phase 4A 全部使用 Clean-S3000。

### Case B：Text-only shortcut 降，但 external ≈ Old

说明：

> 数据更干净，但旧 Negative 不是 external transfer 的主要瓶颈。

仍建议后续使用 Clean-S3000，因为方法实验更可解释。

### Case C：Clean Negative 使训练更难、指标下降

这不一定表示 Clean Negative 错。

可能说明：

> 新 Negative 更 hard，旧数据中存在大量 easy linguistic negatives。

需要重点看：

```text
train Positive Acc
train Negative Acc
held-out negative Acc
```

必要时增加训练 exposure，而不是立即退回 Dirty Negative。

---

## 16. 与 Typed Anonymous 的衔接

Clean Negative 完成后，再进行 Phase 4A。

例如已有：

```text
subject = burger
subject_type = Food
positive_relation = on
object = building
object_type = Building
negative_relation = inside
```

则得到：

```text
Explicit Positive:
A burger is on the building.

Typed Positive:
A {Food} is on the {Building}.

Explicit Negative:
A burger is inside the building.

Typed Negative:
A {Food} is inside the {Building}.
```

第一版 Consistency 训练仍可以只对：

```text
Explicit Positive
↔
Typed Positive
```

计算 consistency。

Negative 继续承担 relation discrimination supervision。

---

## 17. 与论文核心问题的关系

这一步不是单纯的数据清洗。

论文中的逻辑是：

```text
我们希望减少：
Entity identity → Relation prediction

因此不能同时让 Negative 数据存在：
Linguistic implausibility → No prediction
```

所以训练数据必须保证：

> **错误 relation 在语言层面同样 plausible，只能通过视觉证据区分。**

Entity abstraction 和 Clean Counterfactual Negative 是互补的：

```text
Entity Abstraction
↓
削弱 entity → relation lexical prior

Plausible Counterfactual Negative
↓
削弱 linguistic implausibility → No shortcut
```

共同目标：

> **迫使模型更多依赖 image-conditioned relation evidence。**

---

## 18. 推荐执行顺序

```text
Step 1
读取 SRO3000

Step 2
用 InternVL3.5-8B
为每条生成 3 个 counterfactual relation candidates

Step 3
规则过滤：
synonym / morphology / inverse / duplicate

Step 4
保留 high-confidence candidate

Step 5
必要时执行：
Text plausibility verification
+
Visual falsity verification

Step 6
统一 renderer 生成 Negative statement

Step 7
人工 QC 300 条（2026-09-16 已跳过）

Step 8
Text-only shortcut QC

Step 9
形成 Clean-SRO3000

Step 10
可选做 Old-S3000 vs Clean-S3000 sanity check

Step 11
进入 Phase 4A Typed + Consistency 实验
```

---

## 19. Cursor 顶层任务

请基于现有 Phase 3A SRO3000 完成 Clean Negative Reconstruction：

1. 保留现有 image / subject / positive relation / object；
2. 不再使用全局 relation vocabulary 随机 replacement；
3. 调用 InternVL3.5-8B；
4. 每条生成 3 个 linguistically plausible、visually false counterfactual relation candidates；
5. 严格 JSON parsing；
6. 排除 synonym / paraphrase / morphology variant / inverse relation；
7. 只保留 high-confidence candidate；
8. 对失败样本允许固定次数 retry；
9. 必要时增加 text-only plausibility pass；
10. 必要时增加 visual falsity pass；
11. 使用统一 renderer 重新生成 negative statement；
12. 保存原始 candidates 和最终 selected candidate；
13. 输出数据统计；
14. 人工抽样 300 条用于 QC；
15. 构建 text-only shortcut diagnostic；
16. 输出旧 Dirty Negative vs Clean Negative 的数据质量对比；
17. 最终保存 `clean_sro3000.jsonl`；
18. 输出 Markdown report。

---

## 20. 本阶段最终只回答一个问题

> **能否利用 InternVL 根据当前图像和真实 SRO，稳定地产生语言上自然、语义上合理、但当前图像中不成立的 counterfactual relation，从而消除原有 random negative 中明显的 linguistic shortcut？**

只有这一步通过后，再进入 Typed Anonymous + Entity-Invariant Consistency 的正式方法实验。
