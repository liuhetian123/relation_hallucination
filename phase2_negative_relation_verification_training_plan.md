# Phase 2 初步训练计划：Negative Relation Verification

## 0. 为什么进入这一阶段

Phase 1.6b 已经说明：

- Phase 1 的 LoRA **不是只记住训练图片**；
- 在训练时完全一致的 prompt / caption target 格式下，Explicit-LoRA 和 Typed-LoRA 都能在未见过的 RelSim 图片上降低 full-target NLL；
- matching LoRA 在 held-out relation-token NLL 上同样明显优于 Base；
- 但是这种能力没有迁移到 relation-only prompting、MCQ relation discrimination，以及 R-Bench / MMRel 等 hallucination benchmark。

因此，当前最合理的判断是：

> **RelSim positive relational caption supervision 已经表现出“同分布、同任务形式”的跨图片泛化，但它并没有自动转化成 relation verification / hallucination rejection 能力。**

这里的“泛化”需要严格限定为：

> **cross-image generalization within the RelSim relational-caption task distribution**

而不是：

> “已经泛化到通用 relation reasoning / relation hallucination”。

下一阶段的核心问题因此变为：

> **如果显式加入视觉上不成立的 negative relation，并把训练目标改成 relation verification，模型是否能学会拒绝错误关系，从而改善 relation hallucination？**

---

## 1. Phase 2A 的核心假设

当前 Phase 1 的训练形式是：

```text
Image
    ↓
Positive relational caption
```

它只告诉模型：

```text
这个 relation 是对的
```

但从未显式告诉模型：

```text
这个看起来合理的 relation 是错的
```

因此下一阶段测试：

> **Positive-only caption supervision 和 Positive+Negative verification supervision 是否存在本质差异。**

第一阶段先不要重新引入 “Typed 是否优于 Explicit” 作为核心问题。

先回答：

> **negative relation supervision 本身是否能改变 relation rejection / hallucination behavior。**

---

## 2. 第一轮只使用 Explicit 数据

Phase 2A 优先使用已经构造好的 Explicit matched data。

例如：

```text
Tiny bird balancing on a person's finger.
```

暂时不要同时展开 Typed。

原因：

1. Phase 2A 的变量应该只有“有没有 negative supervision”；
2. Explicit statement 能明确指出参与 relation 的实体，减少 Region / placeholder ambiguity；
3. Phase 1 已经证明 Typed vs Explicit 在 caption-SFT 下没有可测差异；
4. 如果 negative verification 本身有效，再在 Phase 2B 重新比较 Explicit vs Typed。

因此：

```text
Phase 2A:
先验证 negative supervision

Phase 2B:
如果有效，再研究 entity abstraction
```

---

## 3. 数据来源

### 3.1 训练图片

优先复用 Phase 1 的 matched Explicit 数据：

```text
约 955 张 image
```

每个样本已有：

```text
image
explicit_caption
typed_caption
relation phrase
```

例如：

```text
Image:
xxx.jpg

Explicit:
Tiny bird balancing on a person's finger.

GT relation:
balancing on
```

### 3.2 Held-out evaluation 图片

继续复用 Phase 1.5 / 1.6 / 1.6b 冻结的 held-out 200 张图片。

不要重新抽样。

这些图片已经保证：

```text
Phase 1 train overlap = 0
```

Phase 2 需要基于这 200 张图片额外构造一个 balanced relation-verification test。

---

## 4. 把 caption 转成 Relation Verification

不要重新自由生成 question。

训练统一使用一个固定模板。

### 推荐训练 Prompt

```text
<image>

Does the image support the relation expressed in the following statement?

Judge only whether the stated relation between the referenced entities is visually supported.
Ignore minor wording or attribute details that are not relevant to the relation.

Statement:
{statement}

Answer Yes or No only.
```

Assistant target：

```text
Yes
```

或：

```text
No
```

---

## 5. Positive 样本

Positive statement 直接使用原始 Explicit caption。

例如：

```text
Image:
xxx.jpg

Statement:
Tiny bird balancing on a person's finger.

Answer:
Yes
```

不要改写原 caption。

不要额外添加 relation explanation。

---

## 6. Negative 样本

Negative 样本必须满足：

> **保持 image、entity、句法、描述结构尽量不变，只替换 relation phrase。**

例如：

```text
Positive:
Tiny bird balancing on a person's finger.

Negative:
Tiny bird hanging from a person's finger.
```

训练：

```text
Statement:
Tiny bird hanging from a person's finger.

Answer:
No
```

目标是让模型不能仅靠：

```text
bird
finger
person
```

猜答案，而必须检查图像中真正的 relation。

---

## 7. Negative 的两种类型

第一轮建议至少区分两种 negative。

### 7.1 Random Negative

从完整 RelSim relation vocabulary 中随机抽一个错误 relation。

要求：

```text
negative != GT
```

并进行基础 normalization：

- lowercase
- strip
- 去句末标点
- 排除完全重复
- 排除明显 substring duplication

如果能够低成本识别，也避免：

```text
standing beside
standing next to
```

这类明显同义关系同时作为 GT / negative。

示例：

```text
GT:
balancing on

Random negative:
hiding behind
```

Random negative 的作用：

> 验证模型是否能够从 positive-only caption learning 转向基本 relation discrimination。

### 7.2 Hard / Plausible Negative

Hard negative 更重要。

定义：

> **对于当前实体组合而言语言上/常识上合理，但图像中实际不成立的 relation。**

例如：

```text
Image:
person pushing a car

GT:
pushing

Hard negative:
driving
```

或者：

```text
Image:
person standing beside a horse

GT:
standing beside

Hard negative:
riding
```

Hard negative 的目标是直接接近：

```text
entity–relation prior
```

和 relation hallucination failure mode。

---

## 8. Hard Negative 的生成建议

Phase 2A 不建议直接让 LLM 自由改写整句。

建议：

```text
GT relation
        ↓
产生 candidate alternative relations
        ↓
筛选 visually false candidate
        ↓
机械替换 relation span
```

可以优先使用以下方式之一：

### 方法 A：Relation vocabulary + semantic candidates

从完整 relation vocabulary 中：

1. 排除 GT；
2. 优先取语义接近、同 relation family、或常见混淆关系；
3. 形成若干 candidate；
4. 再过滤。

### 方法 B：InternVL3.5-8B 仅用于候选判断

允许复用 InternVL3.5-8B，但不要让它重写 caption。

给定：

```text
image
original statement
GT relation
candidate relation
```

只问：

```text
Is the candidate relation visually supported between the same referenced entities?
Yes / No / Uncertain
```

仅保留：

```text
No
```

作为 negative。

`Uncertain` 全部丢弃。

---

## 9. Negative label noise 控制

这一点非常重要。

错误 negative 比“negative 不够难”更危险。

例如：

```text
GT:
standing beside

candidate:
looking at
```

一张图片可能两者同时成立。

不能因为 annotation 里只有一个 relation，就自动认为其他 relation 都是假。

因此：

> **Open-world image 中“未标注”不等于“不存在”。**

第一轮 hard negative 必须过滤。

建议：

- 自动 candidate sampling；
- InternVL / 规则过滤；
- 随机人工检查至少 100 条；
- 宁可样本少，也不要保留疑似 false-negative。

---

## 10. 数据平衡

第一轮训练保持：

```text
Positive : Negative = 1 : 1
```

例如约 955 positive：

```text
955 Yes
955 No
```

总计约：

```text
1910 training samples
```

如果 Random 和 Hard 分开做实验，则分别生成：

```text
Random set:
955 positive + 955 random negative

Hard set:
955 positive + <=955 high-quality hard negative
```

如果 hard negative 无法为每张图都找到，不强行补齐。

最终训练时可以对 positive / hard-negative 做 sampling balance。

---

## 11. 第一轮训练对照组

建议 Qwen2.5-VL-3B 先做以下对照。

### B0：Base

```text
Qwen2.5-VL-3B-Instruct
```

不训练。

### B1：Caption-SFT

直接复用已有：

```text
Explicit-1k LoRA
```

代表：

> positive relational caption learning

不重新训练。

### V1：Positive-only Verification

只把原来的 positive caption 转成 verification：

```text
Statement = correct Explicit caption
Answer = Yes
```

训练约 955 条。

作用：

> 隔离“换成 Yes/No task format”本身是否有作用。

注意：

该模型只见过 Yes，因此不期待它具备良好 rejection。

它只是一个 control。

### V2：Positive + Random Negative Verification

```text
50% Positive
50% Random Negative
```

作用：

> 检查 basic negative supervision 是否提升 relation rejection。

### V3：Positive + Hard Negative Verification

```text
50% Positive
50% Hard / Plausible Negative
```

这是 Phase 2A 最重要的组。

作用：

> 检查 shortcut-aligned negative 是否比 random negative 更有效。

---

## 12. 最重要的实验比较

不是：

```text
V3 vs Base
```

而是依次看：

```text
B1 Caption-SFT
       ↓
V1 Positive Verification
       ↓
V2 Positive + Random Negative
       ↓
V3 Positive + Hard Negative
```

分别回答：

### Q1

```text
Caption → Verification
```

任务形式本身是否重要？

### Q2

```text
Positive-only → Positive+Negative
```

negative supervision 是否重要？

### Q3

```text
Random Negative → Hard Negative
```

针对 plausible relation prior 的 negative 是否更有效？

---

## 13. 训练配置

第一轮尽量复用 Phase 1 配置，减少变量：

```text
Base: Qwen2.5-VL-3B-Instruct
LoRA rank: 16
LoRA alpha: 32
vision encoder: frozen
language LoRA target modules:
q/k/v/o/gate/up/down
```

学习率、batch、optimizer 等优先沿用 Phase 1。

但由于数据从约 955 变成约 1910：

> 不要求 optimizer steps 必须与 Phase 1 完全一样。

第一轮优先：

```text
1 epoch
```

如果 train verification accuracy / NLL 明显没有学进去，再调整 epoch。

不要一开始大幅增加训练强度。

---

## 14. 训练时必须保存的指标

每组至少保存：

```text
train loss
positive loss
negative loss
Yes prediction rate
No prediction rate
```

最好额外保存：

```text
positive accuracy
negative accuracy
```

防止出现：

```text
全部回答 Yes
```

或：

```text
全部回答 No
```

的退化。

---

## 15. Phase 2 Held-out 自建 Verification Test

在 frozen held-out 200 图上构造一个新的测试集。

每张图至少：

```text
1 Positive
1 Negative
```

形成约：

```text
400 questions
```

推荐进一步分：

```text
200 Positive
100 Random Negative
100 Hard Negative
```

或者如果 hard negative 足够：

```text
200 Positive
200 Hard Negative
```

必须保证：

- 测试 negative 与训练 negative 独立采样；
- 不因为模型结果不好而重新修改测试题；
- 构造完成后冻结。

---

## 16. Held-out 测试指标

必须报告：

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
```

另外单独报告：

```text
Positive accuracy
Random-negative accuracy
Hard-negative accuracy
```

最重要的是：

```text
Hard Negative False Positive Rate
```

即：

> 图像里关系不存在，但模型仍因为 relation/entity prior 回答 Yes 的比例。

---

## 17. 外部 Benchmark

第一轮继续评：

### Relation 主指标

```text
R-Bench image-level
MMRel-Adversarial
MMRel normal
AMBER discriminative-relation
```

### Generic control

```text
POPE-Adversarial
```

每个 Yes/No benchmark 都保存：

```text
Acc
Precision
Recall
F1
Yes Ratio
FP
FN
```

不能只报 Accuracy。

---

## 18. Phase 2A 的理想结果

一个理想趋势可能是：

```text
Caption-SFT
≈ Base

Positive-only Verification
≈ Base / 更 Yes-biased

Positive + Random Negative
relation rejection ↑

Positive + Hard Negative
MMRel-Adv / R-Bench FP ↓更多
```

而同时：

```text
Recall 基本保持
POPE 不出现相同幅度变化
```

如果出现这个模式，可以支持：

> **Positive relational caption learning itself is insufficient for hallucination rejection; explicit negative relation supervision is required, and plausible hard negatives are particularly effective for shortcut-heavy relation errors.**

---

## 19. Go / No-Go

### Go A：Negative supervision 有效

满足至少：

- V2/V3 在 held-out verification 上明显优于 V1；
- negative accuracy / TN 明显提升；
- 不是简单 Yes Ratio 暴跌；
- R-Bench 或 MMRel-Adversarial 出现可测 improvement。

进入下一阶段。

### Go B：Hard Negative > Random Negative

如果：

```text
V3 > V2
```

特别是在：

```text
MMRel-Adversarial
Hard-negative held-out
R-Bench FP
```

上差距明显，则可以继续发展：

```text
shortcut-aware negative construction
```

### No-Go A：只在自建 verification test 上提升

如果：

```text
Held-out custom ↑
但
R-Bench / MMRel ≈ Base
```

说明：

> verification task 学到了，但仍存在跨 benchmark / 数据分布 transfer gap。

下一步优先扩大 data diversity，而不是立即回到 anonymity。

### No-Go B：Random / Hard 都没作用

如果：

```text
V1 ≈ V2 ≈ V3 ≈ Base
```

先检查：

- negative 质量；
- 是否训练生效；
- train positive/negative accuracy；
- prompt 是否和 benchmark 太不一致。

不要直接扩大到 7B。

---

## 20. Phase 2B：只有 Phase 2A 成立后再回到 Explicit vs Typed

如果 Positive + Hard Negative 已经证明有效，再构造 matched：

```text
Explicit Positive
Explicit Hard Negative
```

vs

```text
Typed Positive
Typed Hard Negative
```

例如：

```text
Explicit:
Is the person driving the car?
No

Typed:
Is the {Human} driving the {Vehicle}?
No
```

这时再问：

> **在真正要求模型拒绝错误 relation 的任务中，entity abstraction 是否减少 shortcut？**

这个实验比 Phase 1 的：

```text
caption target entity replacement
```

更直接。

---

## 21. Cursor 实现要求

请在现有仓库基础上完成 Phase 2A，不需要重构已有工程。

顶层需要完成：

1. 从 Phase 1 Explicit matched data 中读取：
   - image
   - explicit caption
   - GT relation
2. 构造 positive verification samples；
3. 从 RelSim relation vocabulary 构造 random negatives；
4. 构造 / 过滤 hard negatives；
5. 保证 relation replacement 不改 entity / syntax；
6. 构造 V1 / V2 / V3 三套训练数据；
7. 复用当前 Qwen2.5-VL-3B LoRA 训练框架；
8. 在 frozen held-out images 上构造 verification test；
9. 运行 R-Bench / MMRel / AMBER / POPE；
10. 输出统一 Markdown report。

不要在本阶段自行扩展：

- Fully Anonymous
- Grounded Region A/B
- DPO
- Qwen2.5-VL-7B
- 5k / 10k 扩规模
- 新外挂模块

除非 Phase 2A 已经得到明确 Go 结果。

---

## 22. 最终要回答的问题

Phase 2A 最终只回答三件事：

```text
1. Relation caption learning
   是否等价于 relation verification learning？

2. 显式 negative supervision
   是否能降低 relation false positive？

3. Hard / plausible negative
   是否比 random negative 更有助于
   shortcut-heavy relation hallucination？
```

只有这三件事得到清晰答案后，再决定是否重新进入：

```text
Explicit vs Typed
```

以及后续的：

```text
entity–relation shortcut mitigation
```
