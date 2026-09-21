# Phase 1 结果：No-Go A（Explicit ≈ Typed ≈ Base）

对应计划：`phase1_relation_shortcut_plan.md`。

核心问题：在 matched 图像 / 关系 / 训练预算 / prompt 下，**Typed Anonymous** 是否比 **Explicit** 带来额外的 relation-hallucination 改善（`ΔAbstraction = Typed − Explicit`）。

判定：**No-Go A**。三个模型在所有已评 benchmark 上几乎重合；abstraction 没有独立收益。计划中的 No-Go A 写的是 `Base < Explicit ≈ Typed`（收益来自 relation exposure）。本轮更强：连 relation exposure 的增益也没有，`Base ≈ Explicit ≈ Typed`。

未做（按计划，只有出现 `Typed > Explicit` 才做）：C1 Generic caption、补 seed、Fully Anonymous、Qwen2.5-VL-7B 微调。

---

## 1. 实验设置

| 项 | 取值 |
|---|---|
| 基座 | `Qwen/Qwen2.5-VL-3B-Instruct` |
| B0 | 不微调 |
| E1 / E2 | Explicit-1k / Typed-1k LoRA |
| 训练样本 | 955（Explicit reconstruction `ok` 的 matched ID；同源 RelSim-1k 图） |
| 统一 prompt | `Describe the primary relation shown in this image in one short sentence.` |
| 唯一变量 | assistant target：具体实体 vs typed placeholder（如 `{Animal}`） |
| LoRA | r=16, α=32；只打 language_model 的 q/k/v/o/gate/up/down；vision 冻结 |
| 训练 | 1 epoch，30 step，bs=2，accum=16，lr=2e-5，cosine，seed=42 |
| train_loss | Explicit 2.836；Typed 2.913 |

权重：`checkpoints/qwen25vl3b_{explicit,typed}_1k/`。答案与指标：`eval_results/qwen/`。

---

## 2. R-Bench image-level

官方 `R-Bench/eval.py` 五折平均。每折评 5498 题；答案文件 7787 行。FP 为五折均值。

| 模型 | Acc | Precision | Recall | F1 | Yes | FP |
|---|---:|---:|---:|---:|---:|---:|
| Base | 81.14 | 78.24 | 86.80 | 82.30 | 56.02 | 670.4 |
| Explicit-1k | 81.13 | 78.18 | 86.89 | 82.30 | 56.12 | 673.4 |
| Typed-1k | 81.20 | 78.33 | 86.77 | 82.33 | 55.93 | 666.4 |

`ΔAbstraction`：Acc **+0.07**，Yes **−0.19**，FP **−7**。Typed 没有更低的幻觉、也没有更保守。主结论来源。

---

## 3. POPE-Adversarial

3000 题；`scripts/qwen_eval/score_yesno.py`（跳过空答）。用于检查 generic object hallucination / Yes-bias，不是 relation 主结论。

| 模型 | Acc | Precision | Recall | F1 | Yes | FP |
|---|---:|---:|---:|---:|---:|---:|
| Base | 85.33 | 87.48 | 82.47 | 84.90 | 47.13 | 177 |
| Explicit-1k | 85.27 | 87.04 | 82.87 | 84.90 | 47.60 | 185 |
| Typed-1k | 85.37 | 87.33 | 82.73 | 84.97 | 47.37 | 180 |

三组 Acc 差在 0.1 个百分点内。没有“Typed 更保守”的迹象（否决 No-Go B）。

---

## 4. MMRel-Adversarial

有效 770 题（VG 671 + Dall-E 99）。SPEC 179 题缺图，已跳过空答，不计入。

| 模型 | Acc | Precision | Recall | F1 | Yes | FP |
|---|---:|---:|---:|---:|---:|---:|
| Base | 69.09 | 63.70 | 91.33 | 75.05 | 72.99 | 204 |
| Explicit-1k | 68.96 | 63.59 | 91.33 | 74.97 | 73.12 | 205 |
| Typed-1k | 68.96 | 63.54 | 91.58 | 75.03 | 73.38 | 206 |

shortcut-heavy split 上 Typed 不优于 Explicit。Go 条件 1、4 不成立。

---

## 5. MMRel Dall-E normal

2393 题，无空答。对照 adversarial，看 Typed 优势是否只出现在 hard split。

| 模型 | Acc | Precision | Recall | F1 | Yes | FP |
|---|---:|---:|---:|---:|---:|---:|
| Base | 68.28 | 63.35 | 89.67 | 74.24 | 72.17 | 633 |
| Explicit-1k | 68.24 | 63.33 | 89.59 | 74.20 | 72.13 | 633 |
| Typed-1k | 68.03 | 63.19 | 89.34 | 74.02 | 72.09 | 635 |

normal 与 adversarial 一样平坦。Typed 在 normal 上甚至略差于 Explicit（Acc −0.21）。

---

## 6. AMBER discriminative-relation

官方 `AMBER/inference.py`，1664 题。**Precision / Recall 以 No 为正类**：Recall↑ 表示更能拒绝不成立关系。

| 模型 | Acc | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Base | 81.0 | 71.1 | 91.1 | 79.9 |
| Explicit-1k | 80.9 | 71.4 | 90.0 | 79.6 |
| Typed-1k | 81.6 | 72.3 | 90.1 | 80.2 |

唯一 Typed 略高于 Explicit 的表：Acc **+0.7**，F1 **+0.6**。Recall 仍低于 Base（90.1 vs 91.1），幅度不足以构成 Go 信号。

---

## 7. ΔAbstraction 汇总

`ΔAbstraction = Typed − Explicit`。Acc / Yes 为百分点；FP 为计数差。

| Benchmark | ΔAcc | ΔYes | ΔFP | 备注 |
|---|---:|---:|---:|---|
| R-Bench image-level | +0.07 | −0.19 | −7 | 主指标，噪声量级 |
| POPE-Adv | +0.10 | −0.23 | −5 | generic，无差异 |
| MMRel-Adv | 0.00 | +0.26 | +1 | adversarial 无 Typed 优势 |
| MMRel Dall-E normal | −0.21 | −0.04 | +2 | Typed 略差 |
| AMBER-dr | +0.7 | — | — | 本轮最大正向差，仍很小 |

相对 Base 同样接近零：1k matched SFT 几乎没有改写 Qwen2.5-VL-3B 的 Yes/No 行为。

---

## 8. 判定

对照 `phase1_relation_shortcut_plan.md` §10：

| 条件 | 结果 |
|---|---|
| Go：Typed 在 MMRel-Adv 明显优于 Explicit | 否 |
| Go：R-Bench FP↓ 且 Recall 保持，且差幅可观 | 否（FP 仅 −7） |
| Go：relation 增益明显大于 POPE | 否（两边都是零） |
| Go：adversarial 优势大于 normal | 否 |
| **No-Go A：Explicit ≈ Typed** | **是**（且 Base 也 ≈） |
| No-Go B：Typed 只是更保守 | 否（Yes / Recall 未系统性下降） |
| No-Go C：Typed 明显更差 | 否 |

**结论：** 当前 RelSim-1k 设定下，entity abstraction 没有带来独立的 relation-hallucination 改善。不宜继续扩展 anonymous anti-shortcut 故事，也不按计划启动 7B / Fully Anonymous / Grounded A-B。

若继续，应先改 hypothesis（例如数据量、难负例、或 3B 头寸已饱和），而不是加大同一 Typed vs Explicit 设计。
