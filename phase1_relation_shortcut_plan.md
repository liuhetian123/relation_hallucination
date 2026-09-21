# Phase 1 实验计划：验证 Entity Abstraction 是否带来独立的 Relation Hallucination 改善

## 0. 本阶段目标

本阶段不是验证“RelSim 微调是否有效”，因为已有 LLaVA-1.5-7B 的 pilot 已经观察到 RelSim-1k 会改变模型的 hallucination behavior；但该结果同时伴随 Yes ratio 普遍下降，因此不能排除“模型整体变保守”的解释。

**Phase 1 的核心目标只有一个：**

> 在完全 matched 的图像、关系、训练预算和提示模板下，验证 **Typed Anonymous relational supervision** 是否比 **Explicit relational supervision** 带来额外的 relation-specific 改善。

如果成立，才能进一步支持如下假设：

> 具体 subject/object identity 会给模型提供 entity–relation shortcut；适度抽象实体信息可以减少这一 shortcut，使模型更依赖视觉关系证据。

本阶段不追求完整方法，不引入 bbox/Region A-B、counterfactual、DPO 等复杂设计。先判断核心现象是否存在。

---

## 1. 核心研究问题与假设

### RQ1：RelSim 的收益是否只是 relation exposure？

比较：

- Explicit relational supervision
- Typed Anonymous relational supervision

如果两者效果接近，则当前 RelSim 收益更可能来自“增加了 relation-specific training data”。

如果 Typed Anonymous 稳定优于 Explicit，则说明 **supervision form 本身具有额外作用**。

### RQ2：Typed Anonymous 的收益是否只是让模型更保守？

需要同时检查：

- FP 是否下降；
- Recall / TP 是否基本保持；
- Yes ratio 是否出现异常整体下降；
- POPE 等非 relation benchmark 是否同步获得类似幅度提升。

只有 relation benchmark 上的增益明显强于 generic hallucination control，才能认为改善具有 relation specificity。

### RQ3：这种差异是否在 shortcut-heavy relation 上更明显？

重点看：

- MMRel Adversarial；
- R-Bench 中高共现/高先验关系；
- 后续可补 rare / counter-intuitive relation 分组。

期望现象：

> Typed Anonymous 相比 Explicit 的优势，在 adversarial / counter-intuitive relation 上大于 normal relation。

---

## 2. 模型准备

### 2.1 主模型：Qwen2.5-VL-3B

用途：

- 第一阶段所有 discovery 实验；
- 完整 matched comparison；
- 后续如需要，可继续承担 mechanism analysis 和消融。

第一轮只在 3B 上训练，降低试错成本。

### 2.2 验证模型：Qwen2.5-VL-7B

第一轮先完成 Base benchmark，不立即做全部训练。

只有当 Qwen2.5-VL-3B 上观察到明确的 `Typed > Explicit` 趋势后，再启动：

- Qwen2.5-VL-7B + Explicit-1k
- Qwen2.5-VL-7B + Typed-1k

7B 的作用是验证现象是否随模型规模增加仍然存在，而不是承担完整消融。

### 2.3 已有 LLaVA-1.5-7B 结果的定位

已有 LLaVA RelSim-1k 结果作为 preliminary evidence 保留，不继续大规模扩展。

其作用是说明：

> 少量 RelSim relational supervision 已经能改变 hallucination behavior，但由于 Yes ratio 普遍下降，仍需要新的 matched experiment 区分 relation abstraction 与 generic conservativeness。

---

## 3. 数据准备

### 3.1 基础数据

第一轮使用同一批 **RelSim-1k images**。

所有训练组必须满足：

- 完全相同的图片；
- 完全相同的样本数；
- 完全相同的 relation；
- 尽可能相同的句法结构；
- 相同训练 epoch / step / effective batch；
- 相同 prompt template。

核心原则：**只改变 entity identity 的暴露程度。**

---

## 4. 对照组设计

### B0：Base

不做任何微调。

作用：

- 提供原始能力基线；
- 计算不同训练方式相对 Base 的增益；
- 观察 3B / 7B 的 benchmark headroom。

---

### B1：Generic / Normal Caption Control（建议保留）

使用同一批 RelSim-1k images，但 target 为普通、简短、具体的图像描述，不刻意强调 relation abstraction。

示例：

```text
A small bird is on a person's finger.
```

作用：

- 排除“只要再训练 1k 张图片就会改善”的可能；
- 区分 generic image SFT 与 relation-specific SFT。

该组不是最核心的第一对比，如果数据生成成本过高，可以延后到 Phase 1b。

---

### B2：Explicit Relational Supervision【核心对照】

把当前 RelSim Typed caption 中的 placeholder 替换回具体可见实体，但**不改写其他内容**。

Typed：

```text
Tiny {Animal} balancing on a person's {Body Part}.
```

Explicit：

```text
Tiny bird balancing on a person's finger.
```

要求：

- relation predicate 不变；
- syntax 尽量不变；
- attribute 尽量不变；
- 只替换 entity placeholder；
- 不允许 teacher 自由扩写。

作用：

> 控制“relation exposure”。

如果 Explicit 与 Typed 都提升，则说明 relation-specific supervision 本身有效；只有 Typed 进一步超过 Explicit，才说明 abstraction 可能具有额外价值。

---

### B3：Typed Anonymous Relational Supervision【核心实验组】

直接使用当前 RelSim caption：

```text
Tiny {Animal} balancing on a person's {Body Part}.
```

保留 coarse entity type，去掉具体 entity identity。

这是第一阶段最重要的实验组。

核心比较：

```text
B2 Explicit  vs  B3 Typed Anonymous
```

而不是只看：

```text
Base  vs  Typed Anonymous
```

---

### B4：Fully Anonymous【Phase 1b，可选，不作为第一枪】

示例：

```text
{Entity A} balancing on {Entity B}.
```

该组用于后续研究 abstraction granularity，但第一轮不作为必须项。

原因：Fully Anonymous 会引入新的变量——A/B 对应关系可能不明确，因此若效果变差，很难区分：

- anonymity 过强；
- 还是 grounding ambiguity。

只有 Explicit vs Typed 已经出现现象后，再加入该组研究：

```text
Explicit → Typed → Fully Anonymous
```

是否存在最佳 abstraction granularity。

---

## 5. Prompt 统一原则

训练 prompt 必须使用中性模板，不能继续使用明确要求“abstract relation / avoid concrete object names”的 instruction，否则 Explicit 组和 instruction 会冲突。

建议统一成：

```text
<image>
Describe the primary relation shown in this image in one short sentence.
```

所有组使用完全相同的 human prompt，只改变 assistant target。

---

## 6. 第一轮训练矩阵

### Phase 1a：最小 Go / No-Go

| Run | Model | Data | Target | 必须执行 |
|---|---|---|---|---|
| B0 | Qwen2.5-VL-3B | — | Base | 是 |
| E1 | Qwen2.5-VL-3B | RelSim-1k | Explicit | 是 |
| E2 | Qwen2.5-VL-3B | RelSim-1k | Typed Anonymous | 是 |
| C1 | Qwen2.5-VL-3B | RelSim-1k | Generic Caption | 建议 |

第一轮只跑 1 seed，用于快速判断是否存在明显趋势。

### Phase 1b：确认现象

只有 Phase 1a 出现 `Typed > Explicit` 后执行：

| Run | Model | Data | Target |
|---|---|---|---|
| E1-s2/s3 | Qwen2.5-VL-3B | RelSim-1k | Explicit，补 seed |
| E2-s2/s3 | Qwen2.5-VL-3B | RelSim-1k | Typed，补 seed |
| E3 | Qwen2.5-VL-3B | RelSim-1k | Fully Anonymous |
| E4 | Qwen2.5-VL-7B | RelSim-1k | Explicit |
| E5 | Qwen2.5-VL-7B | RelSim-1k | Typed Anonymous |

---

## 7. 训练变量控制

所有 matched runs 必须锁死：

- image set；
- sample count；
- epoch；
- optimizer steps；
- LoRA target modules；
- LoRA rank / alpha；
- learning rate；
- batch size / gradient accumulation；
- random seed；
- image resolution；
- decoding / evaluation prompt。

第一阶段原则：

> 不追求最优超参，只追求控制变量干净。

只要 Base / Explicit / Typed 的训练设置不同，结果就难以解释。

---

## 8. Evaluation 设计

### 8.1 MMRel Normal

作用：

- 测普通 relation understanding；
- 判断 relation SFT 是否带来基础能力提升。

主要观察：

- Accuracy；
- relation category breakdown。

---

### 8.2 MMRel Adversarial【第一优先级】

这是第一阶段最重要的 benchmark。

原因：该 subset 更接近：

```text
language / commonsense prior
vs
actual visual relation
```

理想现象：

```text
Typed - Explicit gain on MMRel-Adv
>
Typed - Explicit gain on MMRel-Normal
```

这将直接支持“abstraction 对 shortcut-heavy relation 更有帮助”。

---

### 8.3 R-Bench image-level【核心】

用于直接评估 relation hallucination。

必须保存完整指标：

- Accuracy；
- Precision；
- Recall；
- F1；
- TP / TN / FP / FN；
- Yes ratio。

不能只看 Accuracy / F1。

最关键观察：

> Typed 是否比 Explicit 进一步降低 FP，同时 Recall 基本保持。

---

### 8.4 POPE Adversarial【Conservativeness Control】

POPE 不是主任务，而是控制实验。

用途：判断 Typed 是否只是让模型整体更少回答 Yes。

理想情况：

```text
Typed > Explicit   在 relation benchmarks 上明显
Typed ≈ Explicit   在 POPE 上接近
```

如果 Typed 在 POPE 上也出现同幅度的 Yes ratio 下降和 Accuracy 上升，则要警惕 generic conservativeness。

---

### 8.5 AMBER Discriminative-Relation【建议作为辅助】

保留作为与已有 LLaVA pilot 的连续性验证。

特别关注：

- Yes ratio；
- No Precision；
- No Recall；
- 是否出现“更敢说 No，但误拒真实关系”的现象。

该 benchmark 用于辅助判断 response bias，不作为 Phase 1 唯一主结论来源。

---

## 9. 第一阶段核心结果表

最终至少形成一张统一表：

| Model | MMRel | MMRel-Adv | R-Bench Acc | R-Bench FP | R-Bench Recall | R-Bench Yes | POPE-Adv | AMBER-Rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B Base | | | | | | | | |
| + Generic-1k | | | | | | | | |
| + Explicit-1k | | | | | | | | |
| + Typed-1k | | | | | | | | |

重点计算：

```text
ΔExplicit = Explicit - Base
ΔTyped    = Typed - Base
ΔAbstraction = Typed - Explicit
```

真正决定故事是否成立的是：

```text
ΔAbstraction
```

而不是 `Typed - Base`。

---

## 10. Go / No-Go 判定

### Go：进入下一阶段

如果出现以下信号中的至少两个：

1. Typed 在 MMRel-Adversarial 上明显优于 Explicit；
2. Typed 在 R-Bench 上 FP 更低，同时 Recall 基本保持；
3. `Typed - Explicit` 在 relation benchmark 上的增益明显大于 POPE；
4. Typed 的优势在 adversarial relation 上明显大于 normal relation；
5. 补 seed 后总体方向保持稳定。

则认为：

> **存在初步 evidence：entity abstraction 对 relation hallucination 的改善超出了普通 relation exposure。**

下一阶段再研究：

- 1k → 5k data scaling；
- 3B → 7B model scaling；
- Fully Anonymous；
- image-dependency / entity-prior mechanism test；
- 最后才考虑 Grounded A/B 或 counterfactual enhancement。

---

### No-Go A：Explicit ≈ Typed

如果：

```text
Base < Explicit ≈ Typed
```

说明当前收益更可能来自：

> relation-specific SFT / relation exposure。

此时不要继续扩展 anonymous anti-shortcut 故事，应重新分析 hypothesis。

---

### No-Go B：Typed 只是更保守

如果：

```text
FP ↓
Yes ratio ↓↓↓
Recall / TP ↓明显
```

并且 POPE 等 generic benchmark 同样出现类似变化，则说明：

> 当前 effect 更接近 response conservativeness，而不是 relation-specific grounding 改善。

此时应优先分析 decision bias，而不是进入 Grounded / DPO 方法开发。

---

### No-Go C：Typed 比 Explicit 更差

需要首先检查：

- Explicit/Typed 数据是否真正 matched；
- typed placeholder 是否破坏 Qwen 的语言分布；
- prompt 是否存在冲突；
- target 是否过于抽象；
- RelSim caption 本身是否存在噪声。

如果数据与训练无明显问题，则说明“删除具体 entity identity 有利于 relation learning”的核心假设可能不成立。

---

## 11. 本阶段明确不做的内容

为了快速观察现象，Phase 1 暂时不做：

- full RelSim 114k；
- 10k / 50k 大规模训练；
- bbox / Region A-B；
- Grounded Anonymous；
- Counterfactual negative；
- DPO / GRPO；
- 大量 LoRA 超参搜索；
- 所有模型完整 3 seeds；
- 7B 全消融。

这些都必须建立在 `Typed > Explicit` 这一核心现象已经出现的前提上。

---

## 12. 最小执行顺序

```text
Step 0
Qwen2.5-VL-3B / 7B Base evaluation
        ↓
确认 benchmark pipeline 与 headroom
        ↓
Step 1
构造 matched RelSim-1k
Explicit / Typed
        ↓
Step 2
Qwen2.5-VL-3B
Explicit-1k vs Typed-1k
        ↓
Step 3
MMRel-Adv + R-Bench + POPE-Adv
        ↓
Typed > Explicit ?
      /        \
    NO          YES
    │            │
停止扩大      补 seeds
分析原因         │
                 ↓
           Qwen2.5-VL-7B
          Explicit vs Typed
                 │
                 ↓
        进入 Phase 2：5k + mechanism
```

---

## 13. 本阶段最终希望得到的结论

第一阶段**不要求证明完整的 entity–relation shortcut causal mechanism**。

只需要得到一个足以支持后续研究的初步结果：

> **Under matched relational supervision, reducing entity-specific lexical information produces larger gains than explicit relation supervision on shortcut-sensitive relation benchmarks, without merely inducing a global conservative response bias.**

中文对应：

> **在控制图像、关系内容和训练预算后，降低关系监督中的具体实体身份信息，相比显式实体关系监督，在对语言先验更敏感的 relation benchmark 上产生额外收益，并且这种收益不能简单由“模型整体更保守”解释。**

只要这个结论能够在 Qwen2.5-VL-3B 上稳定观察到，并在 Qwen2.5-VL-7B 上复现方向，后续的 abstraction granularity、mechanism analysis 和正式方法才值得继续展开。
