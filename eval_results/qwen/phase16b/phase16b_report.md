# Phase 1.6b Matched Held-out Caption NLL

不重新训练。在 Phase 1.5 冻结的 held-out 200 图上，使用与训练完全相同的 prompt 和 caption 格式做 teacher-forced NLL。

ΔNLL = NLL(Base) − NLL(LoRA)。**正值 = LoRA 提高了该 target 的 likelihood。**

主比较是 matching LoRA vs Base：Explicit-LoRA 对 explicit caption，Typed-LoRA 对 typed caption。

## Reconstruction QC

| 项 | 值 |
|---|---:|
| Total held-out | 200 |
| Reconstruction OK | 193 |
| UNCERTAIN | 7 |
| Parse failure | 0 |
| Missing placeholder | 0 |
| Relation phrase changed | 0 |
| Other validation failure | 0 |
| Explicit eval subset | 193 |

Explicit NLL 只计入 reconstruction `ok` 且 relation phrase 未改的样本。Typed 用完整 200；对照时另报 matched-OK subset。

## 联合总表

| Split | Context / Target | Metric | Base | Explicit-LoRA | Typed-LoRA |
|---|---|---|---:|---:|---:|
| Train | Explicit caption | Full NLL ↓ | 2.974 | **2.743** | 2.851 |
| Train | Typed caption | Full NLL ↓ | 3.187 | 2.954 | **2.743** |
| Train | Explicit caption | Relation NLL ↓ | 4.185 | **3.856** | 3.947 |
| Train | Typed caption | Relation NLL ↓ | 3.907 | 3.524 | **3.356** |
| Held-out | Relation-only prompt | Relation NLL ↓ | **3.327** | 3.507 | 3.571 |
| Held-out | Explicit matched caption | Full NLL ↓ | 2.911 | **2.699** | 2.802 |
| Held-out | Typed matched caption | Full NLL ↓ | 3.216 | 2.998 | **2.765** |
| Held-out | Explicit matched caption | Relation NLL ↓ | 4.137 | **3.709** | 3.781 |
| Held-out | Typed matched caption | Relation NLL ↓ | 3.836 | 3.405 | **3.196** |

Typed matched-OK subset（与 Explicit 同一批图，n=193） full NLL：Base 3.218 / Explicit-LoRA 3.001 / Typed-LoRA 2.766。 Median full NLL：Typed Base 3.234 / Typed-LoRA 2.750；Explicit Base 2.953 / Explicit-LoRA 2.766。

## Paired ΔNLL vs Base（bootstrap 95% CI）

| 比较 | Mean Δ | Median Δ | 95% CI | positive-rate | n |
|---|---:|---:|---|---:|---:|
| Held-out Typed full / Typed-LoRA | +0.451 | +0.438 | [0.426, 0.476] | 100.0% | 200 |
| Held-out Explicit full / Explicit-LoRA | +0.212 | +0.219 | [0.193, 0.231] | 94.8% | 193 |
| Held-out Typed relation / Typed-LoRA | +0.639 | +0.617 | [0.532, 0.751] | 79.0% | 200 |
| Held-out Explicit relation / Explicit-LoRA | +0.429 | +0.305 | [0.355, 0.506] | 82.9% | 193 |
| Held-out Typed full / Explicit-LoRA (cross) | +0.218 | +0.203 | [0.204, 0.233] | 99.5% | 200 |
| Held-out Explicit full / Typed-LoRA (cross) | +0.109 | +0.141 | [0.075, 0.141] | 66.3% | 193 |
| Held-out Typed matched-OK full / Typed-LoRA | +0.452 | +0.438 | [0.427, 0.478] | 100.0% | 193 |

Held-out matching Δ 与训练集几乎同量级：Typed full +0.451 vs train +0.444；Explicit full +0.212 vs train +0.231。
因此 1k LoRA 学到的是 RelSim caption 分布，而不是逐图记忆。Phase 1.6 的 relation-only 变差来自 prompt / output-format shift，不是“完全没有泛化”。

Phase 1.6 对照：Train matching LoRA 明确下降；Held-out relation-only Explicit −0.181 / Typed −0.244。

逐样本：`eval_results/qwen/phase16b/paired_nll.jsonl`、`eval_results/qwen/phase16b/scores/`。

## Decision

```text
Phase 1.6b-B:
Matched relational-caption generalization exists,
but does not transfer to relation-only prompting.
```

下一步不应只扩大 caption 数据，而应把训练目标改成 relation-only / verification 等与评测更对齐的任务形式。

