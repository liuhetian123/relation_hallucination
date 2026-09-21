# Phase 1.6b 实验计划：Matched Held-out Caption NLL Check

## 0. 本阶段目的

Phase 1.6 已得到以下结果：

- 两个 LoRA 在训练集上都能明显降低 full-target NLL；
- 两个 LoRA 在训练集 relation-token NLL 上也有明显下降；
- 但在 held-out 200 图的 relation-only prompt 上，Explicit-LoRA 与 Typed-LoRA 均比 Base 更差；
- 因此目前还无法完全区分：
  1. **真正没有泛化，只拟合了训练样本；**
  2. **模型对 RelSim caption 分布有一定泛化，但 Phase 1.6 的 relation-only prompt / output format 引入了额外 distribution shift。**

Phase 1.6b 的唯一目标是：

> **在 held-out RelSim 图上，使用与训练阶段完全一致的 prompt 和 caption target format，检查 LoRA 是否能在 matched setting 下泛化。**

本阶段：

- 不重新训练；
- 不构造新的 benchmark；
- 不做自由生成；
- 不进入负关系训练；
- 不比较 7B；
- 不讨论最终 hallucination mitigation。

---

## 1. 使用已有模型

继续使用 Phase 1 的三个模型：

- **Base**：`Qwen/Qwen2.5-VL-3B-Instruct`
- **Explicit-LoRA**：Phase 1 Explicit-1k checkpoint
- **Typed-LoRA**：Phase 1 Typed-1k checkpoint

要求：

- 三个模型使用同一个 base checkpoint；
- LoRA adapter 正确加载；
- eval mode；
- no grad；
- image preprocessing / chat template / dtype 与 Phase 1.6 保持一致。

---

## 2. Held-out 数据

直接复用 Phase 1.5 / Phase 1.6 已冻结的 held-out 200 张图片。

不要重新抽样。

数据必须满足：

```text
Phase 1 training overlap = 0
```

每个样本至少应包含：

```json
{
  "image": "...jpg",
  "anonymous_caption": "...",
  "gt_relation": "..."
}
```

本阶段新增：

```text
explicit_caption
```

---

## 3. Typed Held-out Target

Typed condition 直接使用原始 RelSim anonymous caption。

例如：

```text
Tiny {Animal} balancing on a person's {Body Part}.
```

统一 prompt 使用 Phase 1 训练时的原始 prompt：

```text
Describe the primary relation shown in this image in one short sentence.
```

因此 Typed held-out evaluation 完全模拟：

```text
<image>
Describe the primary relation shown in this image in one short sentence.

Target:
Tiny {Animal} balancing on a person's {Body Part}.
```

不要使用：

```text
What is the primary relation shown in this image?
Answer only with the relation phrase.
```

Phase 1.6b 的核心就是消除这类 prompt / output-format shift。

---

## 4. Explicit Held-out Target

### 4.1 构造方式

对同一批 200 张 held-out 图片，使用与 Phase 1 Explicit-1k 完全相同的 entity reconstruction pipeline。

优先复用之前已经验证过的：

```text
InternVL3.5-8B
```

任务仍然只做 placeholder filling：

```text
Typed:
Tiny {Animal} balancing on a person's {Body Part}.

Mapping:
{Animal} -> bird
{Body Part} -> finger

Explicit:
Tiny bird balancing on a person's finger.
```

禁止重新自由 caption。

要求：

- relation phrase 不改；
- syntax 不改；
- attribute 不改；
- modifier 不改；
- 只替换 placeholder；
- 使用 basic-level concrete entity name；
- uncertain 样本单独标记。

### 4.2 Explicit reconstruction QC

输出：

```text
Total held-out samples
Reconstruction OK
UNCERTAIN
Parse failure
Missing placeholder
Relation phrase changed
Other validation failure
```

只对 reconstruction `ok` 的样本做 Explicit matched evaluation。

Typed evaluation 可以继续使用完整 200 条，但最终 Explicit vs Typed 对照时，必须同时报告：

```text
matched-OK subset
```

即只比较 Explicit reconstruction 成功的同一批图片。

---

## 5. Experiment A：Held-out Full-target NLL

这是 Phase 1.6b 的核心实验。

### 5.1 Typed target

对每个 held-out sample：

```text
Input:
image
+
Describe the primary relation shown in this image in one short sentence.

Target:
original typed anonymous caption
```

分别计算：

```text
Base -> Typed target NLL
Explicit-LoRA -> Typed target NLL
Typed-LoRA -> Typed target NLL
```

核心比较：

```text
Typed-LoRA vs Base
```

### 5.2 Explicit target

对 reconstruction 成功的 held-out samples：

```text
Input:
image
+
Describe the primary relation shown in this image in one short sentence.

Target:
reconstructed explicit caption
```

分别计算：

```text
Base -> Explicit target NLL
Explicit-LoRA -> Explicit target NLL
Typed-LoRA -> Explicit target NLL
```

核心比较：

```text
Explicit-LoRA vs Base
```

---

## 6. Experiment B：Held-out Relation-token NLL in Matched Caption Context

除了 full-target NLL，再在 matched held-out caption context 中重新计算 relation-token NLL。

例如：

```text
Typed:
Tiny {Animal} [balancing on] a person's {Body Part}.
```

Explicit：

```text
Tiny bird [balancing on] a person's finger.
```

只累计：

```text
balancing on
```

对应 token 的 loss，其余 target token 只作为 teacher-forcing context。

分别计算：

### Typed context

```text
Base
Explicit-LoRA
Typed-LoRA
```

### Explicit context

```text
Base
Explicit-LoRA
Typed-LoRA
```

这一实验用于判断：

> LoRA 是否能在未见图像上提高相同 caption distribution 中 relation phrase 的 likelihood。

---

## 7. 与 Phase 1.6 的结果联合报告

Phase 1.6 已有：

```text
Train full-target NLL
Train relation-token NLL
Held-out relation-only NLL
```

Phase 1.6b 新增：

```text
Held-out matched full-target NLL
Held-out matched relation-token NLL
```

最终建议输出总表：

| Split | Context / Target | Metric | Base | Explicit-LoRA | Typed-LoRA |
|---|---|---|---:|---:|---:|
| Train | Explicit caption | Full NLL ↓ | 2.974 | 2.743 | 2.851 |
| Train | Typed caption | Full NLL ↓ | 3.187 | 2.954 | 2.743 |
| Train | Explicit caption | Relation NLL ↓ | 4.185 | 3.856 | 3.947 |
| Train | Typed caption | Relation NLL ↓ | 3.907 | 3.524 | 3.356 |
| Held-out | Relation-only prompt | Relation NLL ↓ | 3.327 | 3.507 | 3.571 |
| Held-out | Explicit matched caption | Full NLL ↓ | | | |
| Held-out | Typed matched caption | Full NLL ↓ | | | |
| Held-out | Explicit matched caption | Relation NLL ↓ | | | |
| Held-out | Typed matched caption | Relation NLL ↓ | | | |

前五行可以直接复用 Phase 1.6 报告数值。

---

## 8. 主要统计

继续使用 paired analysis。

对每个 held-out sample 计算：

\[
\Delta NLL_i = NLL_{Base,i} - NLL_{LoRA,i}
\]

正值表示 LoRA 提高 target likelihood。

至少输出：

```text
Mean NLL
Median NLL
Mean Delta NLL
Median Delta NLL
Positive-rate
Bootstrap 95% CI
Sample count
```

其中：

```text
positive-rate
```

定义为：

```text
LoRA NLL < Base NLL
```

的样本比例。

---

## 9. Cross-target 结果仍保留，但不作为主结论

建议继续计算：

```text
Explicit-LoRA on Typed target
Typed-LoRA on Explicit target
```

目的只是辅助判断：

- 是否主要学习 output style；
- 是否有跨 caption form 的迁移。

不要把 cross-target 结果作为 Phase 1.6b Go/No-Go 主指标。

---

## 10. Phase 1.6b 判定

### Case 1：真正缺乏 held-out 泛化

如果：

```text
Train:
matching LoRA NLL << Base

Held-out matched full caption:
matching LoRA ~= Base 或 worse

Held-out matched relation-token:
matching LoRA ~= Base 或 worse
```

则可以较有把握判定：

> **Phase 1 的 1k LoRA 主要形成 training-set specialization / memorization，没有可测的 cross-image relational generalization。**

下一步：

```text
1k -> 5k data scaling
```

优先扩大 relation diversity。

此时先不做：

- Explicit vs Typed mechanism story；
- negative relation training；
- 7B；
- Fully Anonymous；
- Grounded A/B。

建议先只训练一个：

```text
Typed-5k
```

检查 held-out matched-caption NLL 是否开始改善。

---

### Case 2：Matched caption 能泛化，但 relation-only prompt 不泛化

如果：

```text
Held-out matched full-caption:
LoRA clearly better than Base

Held-out matched relation-token:
LoRA clearly better than Base

但 Phase 1.6 relation-only:
LoRA worse than Base
```

则说明：

> **LoRA 学会并泛化了 RelSim-style relational caption distribution，但这种能力不能自然 transfer 到新的 relation-only instruction / discrimination format。**

下一步不应该简单扩大 caption 数据，而应该考虑把训练目标从：

```text
image -> relational caption
```

改成：

```text
image + relation query -> relation answer
```

或：

```text
positive / negative relation verification
```

即进入更 task-aligned 的 relation supervision。

---

### Case 3：Full-caption 泛化，但 relation-token 不泛化

如果：

```text
Held-out full-target NLL ↓
Held-out relation-token NLL ~= Base
```

则说明：

> **模型主要学到了 RelSim caption surface distribution，而没有明显增强 relation phrase 本身的预测。**

下一步优先转向：

```text
relation-only generation
relation classification
relation verification
```

而不是继续增加普通 caption SFT。

---

### Case 4：Typed 与 Explicit 泛化模式明显不同

例如：

```text
Explicit-LoRA held-out improvement > 0
Typed-LoRA held-out ~= / worse
```

或者反过来。

先不要立刻解释为 entity abstraction effect。

首先检查：

- reconstruction quality；
- caption length；
- relation frequency；
- placeholder distribution；
- relation extraction coverage；
- train target token distribution。

Phase 1.6b 的主要目的仍然是：

> **确认是否存在 matched held-out generalization。**

---

## 11. 本阶段明确不做

Phase 1.6b 不做：

- 新模型训练；
- 5k 训练；
- Qwen2.5-VL-7B；
- hard-negative generation；
- negative relation SFT；
- DPO；
- Fully Anonymous；
- Grounded Region A/B；
- 新 hallucination benchmark；
- 扩大 Phase 1.5 MCQ；
- 自由生成 caption 作为主指标。

---

## 12. 最终输出物

至少输出：

1. held-out 200 Explicit reconstruction 数据；
2. reconstruction QC；
3. Base / Explicit-LoRA / Typed-LoRA 的：
   - held-out Explicit full-target NLL
   - held-out Typed full-target NLL
   - held-out Explicit-context relation NLL
   - held-out Typed-context relation NLL
4. paired Delta NLL；
5. positive-rate；
6. bootstrap 95% CI；
7. 逐样本 scoring JSONL；
8. 一份 Markdown report。

最终报告结论只归类为：

```text
Phase 1.6b-A:
No matched held-out generalization.

Phase 1.6b-B:
Matched relational-caption generalization exists,
but does not transfer to relation-only prompting.

Phase 1.6b-C:
Surface-form generalization without clear relation-token generalization.

Phase 1.6b-D:
Inconsistent Explicit/Typed generalization; inspect data factors.
```

---

## 13. 推荐执行顺序

```text
Step 1
复用 frozen held-out 200
        ↓
Step 2
Typed target 直接使用原 anonymous caption
        ↓
Step 3
用同一 InternVL3.5-8B pipeline
回填 held-out Explicit caption
        ↓
Step 4
做 Explicit reconstruction QC
        ↓
Step 5
Base / Explicit / Typed
计算 held-out full-target NLL
        ↓
Step 6
计算 matched-context relation-token NLL
        ↓
Step 7
与 Phase 1.6 train / relation-only 结果联合分析
        ↓
Step 8
判定：
数据规模问题
or
task-format transfer 问题
```

**Phase 1.6b 最重要的比较不是 Typed vs Explicit，而是：**

```text
matching LoRA vs Base
```

即：

```text
Explicit-LoRA vs Base on held-out Explicit captions
Typed-LoRA vs Base on held-out Typed captions
```

只有先判断这种 matched held-out 泛化是否存在，才应该决定下一步是扩大到 5k，还是直接改变 relation supervision 的任务形式。
