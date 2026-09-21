# Phase 1.6 实验计划：LoRA Adaptation / Relation-Token NLL Check

## 0. 本阶段目的

Phase 1 的结果是：

- Qwen2.5-VL-3B Base
- Explicit-1k LoRA
- Typed-1k LoRA

在 R-Bench、MMRel、POPE、AMBER 上几乎重合。

Phase 1.5 又发现：

- Held-out RelSim MCQ 上三模型 Accuracy 基本一致；
- 三模型逐题预测一致率约 97%–98.5%；
- Explicit / Typed 的 GT margin 没有高于 Base；
- 部分 MCQ 本身存在“primary relation / subject-object 不明确”的题目噪声。

因此 Phase 1.6 **不再设计新的视觉 benchmark**，而是直接检查训练目标本身。

本阶段只回答一个问题：

> **Explicit-1k / Typed-1k 这两个 LoRA checkpoint 是否真正学到了 Phase 1 的 relational supervision？**

需要区分三种情况：

1. **训练干预太弱，模型基本没学到；**
2. **模型主要记住训练样本，没有泛化；**
3. **模型学到了 relation supervision 并能泛化，但这种学习没有迁移到 hallucination / relation-discrimination benchmark。**

本阶段不讨论 entity abstraction 是否有效，也不重新比较 “Typed 是否优于 Explicit” 作为主结论。

---

# 1. 已有模型与数据

## 1.1 模型

使用 Phase 1 已有三个模型，不重新训练：

- **M0 Base**：`Qwen/Qwen2.5-VL-3B-Instruct`
- **M1 Explicit-1k**
- **M2 Typed-1k**

LoRA 必须分别正确加载到同一个 Base checkpoint 上。

---

## 1.2 训练数据

使用 Phase 1 实际参与训练的 **955 个 matched samples**。

每个样本应该存在：

- image
- unified prompt
- explicit target
- typed target
- relation phrase / 或能够从 matched caption 中可靠定位 relation phrase

示例：

```text
Prompt:
Describe the primary relation shown in this image in one short sentence.

Explicit target:
Tiny bird balancing on a person's finger.

Typed target:
Tiny {Animal} balancing on a person's {Body Part}.

Relation phrase:
balancing on
```

要求 Explicit 与 Typed 使用完全相同的 955 张图。

---

## 1.3 Held-out 数据

优先复用 Phase 1.5 已冻结的 **200 张 held-out RelSim 图**：

- 与 Phase 1 的原始 1k 图像 overlap = 0；
- 已有 anonymous caption；
- 已有提取后的 GT relation phrase。

Phase 1.6 不重新挑图，避免不断调整测试集。

---

# 2. 核心思想：Teacher-Forced NLL，而不是自由生成

不再让模型生成完整 caption，也不把 MCQ accuracy 作为主指标。

对给定：

\[
(I, Q, Y)
\]

计算 assistant target 的 teacher-forced negative log-likelihood：

\[
\mathrm{NLL}(Y)
=
-\frac{1}{|Y|}
\sum_t
\log p_\theta(y_t \mid I,Q,y_{<t})
\]

需要至少同时计算两种 NLL：

1. **Full-target NLL**
2. **Relation-token-only NLL**

---

# 3. Experiment A：Training-set Full-Target NLL

这是最重要、最直接的 sanity check。

## 3.1 Explicit target

对全部 955 个训练样本，用相同 image + prompt + explicit target，分别计算：

```text
Base → Explicit target NLL
Explicit-LoRA → Explicit target NLL
Typed-LoRA → Explicit target NLL（辅助）
```

核心比较：

\[
\Delta^{train}_{explicit}
=
NLL(Base, Y_{explicit})
-
NLL(ExplicitLoRA, Y_{explicit})
\]

如果 Explicit-LoRA 真正学到了训练目标，应看到：

```text
Explicit-LoRA NLL < Base NLL
```

并且最好不是极小差异。

---

## 3.2 Typed target

同理：

```text
Base → Typed target NLL
Typed-LoRA → Typed target NLL
Explicit-LoRA → Typed target NLL（辅助）
```

核心比较：

\[
\Delta^{train}_{typed}
=
NLL(Base, Y_{typed})
-
NLL(TypedLoRA, Y_{typed})
\]

预期：

```text
Typed-LoRA NLL < Base NLL
```

---

## 3.3 为什么必须做 cross-target

建议同时保存：

| Model | Explicit-target NLL | Typed-target NLL |
|---|---:|---:|
| Base | | |
| Explicit-LoRA | | |
| Typed-LoRA | | |

cross-target 不是主判定，但可以帮助回答：

- LoRA 是只学会了输出格式 / placeholder 风格？
- 还是对 relational content 本身也有影响？

例如：

```text
Explicit-LoRA 只显著降低 Explicit full-caption NLL
Typed-LoRA 只显著降低 Typed full-caption NLL
```

可能说明大量变化来自 output-format adaptation。

因此还必须做 Relation-token-only NLL。

---

# 4. Experiment B：Training-set Relation-Token NLL

这是 Phase 1.6 最关键的 relation-specific check。

## 4.1 只对 relation phrase token 计算 loss

例如：

```text
Tiny {Animal} [balancing on] a person's {Body Part}.
```

或：

```text
Tiny bird [balancing on] a person's finger.
```

只累计：

```text
balancing
on
```

对应 token 的 NLL。

其余 target token：

- 仍然作为 teacher-forcing context；
- 但 loss mask = 0。

定义：

\[
NLL_{rel}
=
-\frac{1}{|R|}
\sum_{t\in R}
\log p_\theta(y_t \mid I,Q,y_{<t})
\]

---

## 4.2 Explicit context 与 Typed context 都要算

因为两套 caption 是 matched 的，relation phrase 应保持不变。

分别计算：

### Explicit-context Relation NLL

```text
Base
Explicit-LoRA
Typed-LoRA
```

在 Explicit caption context 下预测 relation phrase。

### Typed-context Relation NLL

```text
Base
Explicit-LoRA
Typed-LoRA
```

在 Typed caption context 下预测 relation phrase。

最终输出：

| Model | Explicit-context Relation NLL ↓ | Typed-context Relation NLL ↓ |
|---|---:|---:|
| Base | | |
| Explicit-LoRA | | |
| Typed-LoRA | | |

如果两个 LoRA 只降低 full-caption NLL，但 **relation-token NLL 几乎不变**，则说明 Phase 1 训练可能主要学习了：

- entity lexical pattern；
- placeholder output format；
- caption surface form；

而没有明显强化 relation token 本身。

---

# 5. Relation phrase 的定位

必须保证 relation token mask 正确。

优先使用 Phase 1 / RelSim 已有结构化 relation 字段。

如果没有，则利用 matched Explicit / Typed caption 的结构提取 relation phrase。

要求：

- 不要让当前评测模型生成 relation label；
- 不要基于模型 prediction 决定 mask；
- relation phrase 必须固定、可追溯。

对 955 个训练样本输出 extraction QC：

```text
Total
Successfully located relation
Failed relation location
Multi-match
Empty relation
```

无法可靠定位 relation span 的样本：

```text
skip from relation-token metric
```

但仍可参加 full-target NLL。

随机人工检查至少 50 条 relation span。

---

# 6. Experiment C：Held-out Relation Generalization

Experiment A/B 只能证明：

> 模型是否拟合训练目标。

还需要判断：

> 是否泛化到未见过的 RelSim image / relation description。

优先复用 Phase 1.5 的 200 张 held-out 图。

---

## 6.1 必做：Neutral Relation-Only NLL

对每张 held-out 图片使用统一 prompt：

```text
<image>

What is the primary relation shown in this image?
Answer only with the relation phrase.
```

target 只使用 Phase 1.5 已提取的 GT relation：

```text
balancing on
```

三个模型完全使用相同 target：

```text
Base
Explicit-LoRA
Typed-LoRA
```

计算：

```text
Held-out relation-only NLL
Held-out relation perplexity（可选）
```

优点：

- 不依赖 Explicit / Typed caption 风格；
- 不依赖 placeholder；
- 不需要自由生成；
- 三个模型面对完全相同的 target。

注意：

Phase 1.5 已发现部分图存在 “primary relation” ambiguity，因此：

> 这一指标用于比较相对变化，不把绝对准确率 / NLL 当作新 benchmark。

---

# 7. 推荐增强：Held-out Matched Explicit / Typed NLL

如果实现成本很低，建议进一步把 Phase 1.5 的 200 个 anonymous captions 使用**之前完全相同的 InternVL3.5-8B entity reconstruction pipeline**补成 explicit captions。

只处理这 200 张即可，不需要扩大数据。

例如：

```text
Typed held-out:
Tiny {Animal} balancing on a person's {Body Part}.

Explicit held-out:
Tiny bird balancing on a person's finger.
```

然后在 held-out set 上重复：

- Full-target NLL
- Relation-token NLL
- Explicit-context / Typed-context

这样可以形成真正 matched 的 train / held-out 表：

| Split | Metric | Base | Explicit-LoRA | Typed-LoRA |
|---|---|---:|---:|---:|
| Train | Explicit full NLL | | | |
| Train | Typed full NLL | | | |
| Train | Explicit relation NLL | | | |
| Train | Typed relation NLL | | | |
| Held-out | Explicit full NLL | | | |
| Held-out | Typed full NLL | | | |
| Held-out | Explicit relation NLL | | | |
| Held-out | Typed relation NLL | | | |

这一步是推荐项，但不要阻塞 Experiment A/B/C。

---

# 8. Sequence NLL 计算细节

实现时需要注意以下原则。

## 8.1 只计算 assistant target

system / user / image token 不进入 target loss。

即：

```text
labels = -100
```

覆盖：

- system prompt
- user prompt
- image token
- prompt template

只对 assistant target token 计算 NLL。

---

## 8.2 不使用生成模式

不要：

```text
model.generate(...)
```

而是标准 forward：

```text
outputs = model(...)
logits = outputs.logits
```

根据 shifted logits / labels 计算 token-level log probability。

---

## 8.3 使用平均 token NLL

因为 Explicit / Typed target 长度可能不同，主指标使用：

```text
mean NLL per target token
```

而不是整句 total NLL。

同时保存：

```text
total NLL
token count
mean NLL
```

---

## 8.4 Relation multi-token phrase

relation phrase 可能 tokenize 为多个 token。

必须对 relation span 的**所有 target tokens**求平均。

不能只取 relation 第一个 token。

---

## 8.5 模型设置

三个 checkpoint：

```text
eval mode
no grad
同一 dtype
同一 image preprocessing
同一 chat template
同一 prompt
```

不得因为模型不同调整 image resolution / max pixels。

---

# 9. 推荐额外指标：Parameter / Output Adaptation Strength

为了进一步确认 LoRA 是否实际上改变模型，可以附加两个非常轻量的分析。

## 9.1 Token-level KL divergence

在训练样本和 held-out 样本上，对 relation span 位置比较：

\[
KL(
P_{LoRA}(\cdot)
\|
P_{Base}(\cdot)
)
\]

分别计算：

```text
Explicit-LoRA vs Base
Typed-LoRA vs Base
```

如果 KL 几乎为 0，与 Phase 1.5 的 97%+ prediction agreement 一致，则进一步说明训练干预极弱。

不是主指标，容易实现再做。

---

## 9.2 Correct-relation probability shift

对 relation phrase token 记录：

\[
\Delta \log P(r)
=
\log P_{LoRA}(r)
-
\log P_{Base}(r)
\]

统计：

```text
mean
median
positive-rate
distribution
```

例如：

```text
多少样本 LoRA 提高了 GT relation likelihood？
```

相比只报平均 NLL，这能判断 improvement 是否由少量 outlier 驱动。

---

# 10. 统计分析

由于三个模型对完全相同样本进行 scoring，全部使用 paired analysis。

至少报告：

```text
Mean NLL
Median NLL
Standard deviation
Mean paired ΔNLL
Bootstrap 95% CI
```

例如：

\[
\Delta NLL_i
=
NLL_{Base,i}
-
NLL_{LoRA,i}
\]

若：

```text
ΔNLL > 0
```

表示 LoRA 提高 target likelihood。

分别计算：

```text
Explicit-LoRA vs Base
Typed-LoRA vs Base
```

---

# 11. 最重要的最终结果表

建议最终报告优先输出这张：

| Split | Metric | Base | Explicit-LoRA | Typed-LoRA |
|---|---|---:|---:|---:|
| Train | Explicit full-target NLL ↓ | | | |
| Train | Typed full-target NLL ↓ | | | |
| Train | Explicit-context relation NLL ↓ | | | |
| Train | Typed-context relation NLL ↓ | | | |
| Held-out | Relation-only NLL ↓ | | | |

如果做了 held-out matched reconstruction，再附加完整表。

---

# 12. Phase 1.6 判定标准

## Case A：Training intervention too weak

表现：

```text
Train full-target:
LoRA ≈ Base

Train relation-token:
LoRA ≈ Base

Held-out:
LoRA ≈ Base
```

或者所有 ΔNLL 都极小。

结论：

> **Phase 1 的 955 samples × 1 epoch × 30 steps × lr=2e-5 对 Qwen2.5-VL-3B 的行为干预太弱。**

下一步：

- 不讨论 anonymous hypothesis；
- 不立即构造复杂 negative dataset；
- 先加强训练强度；
- 目标是让至少 train / held-out RelSim relation NLL 出现明确 adaptation。

可以优先尝试：

```text
更多 epoch / optimizer steps
适度提高 learning rate
或扩展到 5k relation data
```

但具体超参留给下一阶段决定。

---

## Case B：Memorization only

表现：

```text
Train NLL 明显下降
Held-out relation NLL ≈ Base
```

结论：

> **LoRA 成功拟合训练数据，但 1k relational captions 没有形成可测泛化。**

下一步优先：

- 增加 relation diversity / data scale；
- 不急于比较 Typed vs Explicit；
- 先建立能泛化的 relation adaptation。

---

## Case C：Relation supervision learned and generalized

表现：

```text
Train full NLL ↓明显
Train relation NLL ↓明显
Held-out relation NLL ↓明显
```

但 Phase 1：

```text
R-Bench / MMRel / POPE / AMBER ≈ Base
```

且 Phase 1.5：

```text
MCQ behavior ≈ Base
```

则得到最重要的结论：

> **Positive relational caption supervision can be learned and generalized on RelSim, but it does not transfer into stronger relation discrimination / hallucination rejection.**

这时下一步非常明确：

## Phase 2：Positive + Negative Relation Supervision

重点加入：

```text
Positive relation
Random negative relation
Hard / plausible negative relation
```

尤其研究：

```text
visually false but entity-prior-plausible relations
```

此时再重新考虑：

```text
Explicit vs Typed
```

才有意义。

---

## Case D：Full-caption 学到了，但 Relation-token 没学到

表现：

```text
Full-target NLL ↓明显
Relation-token NLL ≈ Base
```

结论：

> 训练主要改变了 caption surface form / entity / placeholder pattern，而没有明显改变 relation prediction 本身。

下一步不应简单增加 epoch。

应重新设计 supervision，让 relation 本身成为直接学习目标，例如：

```text
relation-only generation
relation classification
positive / negative relation verification
```

---

# 13. 本阶段明确不做

Phase 1.6 不做：

- 新模型训练；
- Qwen2.5-VL-7B；
- 新 hallucination benchmark；
- 扩大 Phase 1.5 MCQ；
- Fully Anonymous；
- Grounded Region A/B；
- hard-negative training；
- DPO；
- 外挂 mitigation；
- 大规模新数据构造。

唯一目标：

> **确认 Phase 1 LoRA 在训练目标和 relation token 层面到底发生了什么。**

---

# 14. 输出物

Cursor 完成后至少输出：

1. **Train-set full-target token-level scoring**
2. **Train-set relation-token scoring**
3. **Held-out 200 relation-only scoring**
4. 每个模型逐样本 NLL / relation-NLL 原始结果
5. paired ΔNLL
6. bootstrap 95% CI
7. relation-span QC
8. 一份简短 Markdown report

最终报告明确归类为：

```text
Phase 1.6-A:
Training intervention too weak.

Phase 1.6-B:
Training fits train data but does not generalize.

Phase 1.6-C:
Relation supervision is learned and generalized,
but does not transfer to hallucination benchmarks.

Phase 1.6-D:
Surface-form adaptation without clear relation-token adaptation.
```

---

# 15. 推荐执行顺序

```text
Step 1
确认 955 train matched data 与 relation span
        ↓
Step 2
计算 Base / Explicit / Typed 的
train full-target NLL
        ↓
Step 3
计算 train relation-token NLL
        ↓
Step 4
复用 Phase1.5 held-out 200
计算 neutral relation-only NLL
        ↓
Step 5
paired ΔNLL + bootstrap
        ↓
Step 6
按照 A/B/C/D 判定
        ↓
只有判定后再决定
加强训练 / 扩数据 / 负关系 Phase 2
```

**优先级最高的是 Step 2 + Step 3。**

如果训练集本身 LoRA 与 Base 都几乎没有 NLL 差异，那么不需要先花时间做更复杂的 held-out reconstruction；可以直接判定当前 training intervention 太弱。
