# Phase 1.6 LoRA Adaptation / Relation-Token NLL

不重新训练。用 teacher-forced mean token NLL 检查 Phase 1 的两个 LoRA 是否学到了 RelSim relational supervision。

ΔNLL = NLL(Base) − NLL(LoRA)。**正值 = LoRA 提高了该 target 的 likelihood。**

---

## Relation-span QC（训练 955）

| 项 | 值 |
|---|---:|
| Total | 955 |
| Successfully located | 534 |
| Failed / empty | 421 |
| Multi-match | 0 |

Relation phrase 从 Typed caption 的 placeholder 结构抽取，不经过评测模型。无法定位的样本仍计入 full-target NLL，不计入 relation-token NLL。50 条抽查：`eval_results/qwen/phase16/review_50_spans.jsonl`。

---

## 主表

| Split | Metric | Base | Explicit-LoRA | Typed-LoRA |
|---|---|---:|---:|---:|
| Train | Explicit full-target NLL ↓ | 2.974 | **2.743** | 2.851 |
| Train | Typed full-target NLL ↓ | 3.187 | 2.954 | **2.743** |
| Train | Explicit-context relation NLL ↓ | 4.185 | **3.856** | 3.947 |
| Train | Typed-context relation NLL ↓ | 3.907 | 3.524 | **3.356** |
| Held-out | Relation-only NLL ↓ | **3.327** | 3.507 | 3.571 |

---

## Paired ΔNLL vs Base（bootstrap 95% CI）

| 比较 | ΔNLL | 95% CI | positive-rate | n |
|---|---:|---|---:|---:|
| Train Explicit full / Explicit-LoRA | +0.231 | [0.222, 0.240] | 95.8% | 955 |
| Train Typed full / Typed-LoRA | +0.444 | [0.433, 0.456] | 99.6% | 955 |
| Train Explicit relation / Explicit-LoRA | +0.329 | [0.281, 0.377] | 77.2% | 534 |
| Train Typed relation / Typed-LoRA | +0.550 | [0.484, 0.617] | 78.3% | 534 |
| Held-out relation-only / Explicit-LoRA | **−0.181** | [−0.219, −0.142] | 24.0% | 200 |
| Held-out relation-only / Typed-LoRA | **−0.244** | [−0.295, −0.196] | 20.5% | 200 |

训练上匹配 target 的 NLL 明确下降（CI 不含 0），relation token 同样下降。Held-out 上两个 LoRA 都比 Base **更差**。

逐样本：`eval_results/qwen/phase16/paired_nll.jsonl`、`eval_results/qwen/phase16/scores/`。

---

## Decision

```text
Phase 1.6-B:
Training fits train data but does not generalize.
```

955 × 1 epoch 足以拟合训练 caption 和其中的 relation token，但没有形成对未见 RelSim 图的 relation-phrase 泛化。按计划：下一步应先增加 relation diversity / data scale，而不是比较 Typed vs Explicit，也先不进入负关系 Phase 2。
