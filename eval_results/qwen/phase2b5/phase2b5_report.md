# Phase 2B.5 Prompt Transfer Diagnostic

对应计划：`phase2b5_prompt_transfer_diagnostic_plan.md`。

本阶段不训练。只回答：

> 外部 transfer 失败，主要是 prompt / task formulation mismatch，还是 data / relation distribution transfer 不足？

未做：新训练、3k、Typed、Hard 新方法、7B、DPO、改 fast subset、full benchmark、AMBER-dr（未出现 Case A）。

---

## 1. 设置

| 项 | 取值 |
|---|---|
| 模型 | Base；V2-old（Phase 2A）；V2B-2ep（Phase 2B 主 checkpoint） |
| Bench | 冻结的 R-Bench-fast（1924）与 MMRel-Adv-fast（384） |
| Prompt A | 官方评测题，**不重新推理** |
| Prompt B | Phase 2B 训练 verification 模板 + 从原题确定性转换的 statement |
| 解码 | `do_sample=False`，`max_new_tokens=16`，与 2B fast-eval 相同 |
| 解析 | Matched：句首 Yes/No，其余记 Invalid（本轮 Invalid=0）。Official：复用 R-Bench `no/not` 规则，以便与 2B 对照 |

权重：`checkpoints/qwen25vl3b_phase2_v2`、`checkpoints/qwen25vl3b_phase2_v2b_2ep`。

---

## 2. Statement conversion

R-Bench 无 subject/relation/object 字段；MMRel adversarial JSON 同样只有 yes/no 问句。全部用**确定性**规则，不用 VLM 看图生成。

| Bench | 方法 | 成功 | 失败 |
|---|---|---:|---:|
| R-Bench-fast | spaCy 只定位 nsubj，用原文字符 span 还原；失败则正则回退 | 1924 / 1924 | 0 |
| MMRel-Adv-fast | 两种官方模板正则：`Is there X REL to Y` / `Does X VERB Y` | 384 / 384 | 0 |

Gold label 与 image **100% 未改**。R-Bench 有 211 条重复 statement 文本（不同图、同类问句），不是转换崩溃。

抽查：`eval_results/qwen/phase2b5/reviews/{rbench,mmrel}_review_100.tsv`（seed=42，各 100 条）。

人工核对结论：Official question 与 matched statement 表达同一 relation proposition。已知残余：

- `Are there any X?` → `There are any X.`（awkward，命题不变）
- MMRel 保留官方不自然介词（`above to` / `on to`），**不改 predicate**
- 个别处所+形容词（如 *Is the door inside the house open?*）语序仍略歪，数量可忽略

QC：`eval_results/qwen/phase2b5/conversion_qc.json`。  
题面：`data/phase2b5/{rbench_fast,mmrel_adv_fast}_matched_prompt.jsonl`。

---

## 3. 主表

R-Bench 为与 Phase 2B 相同的五折平均；MMRel 为 384 有效题。V2B−Base 的显著性见第 5 节 unique-item paired 统计。

| Benchmark | Prompt | Base Acc | V2-old Acc | V2B-2ep Acc | V2B−Base | Base FP | V2B FP |
|---|---|---:|---:|---:|---:|---:|---:|
| R-Bench-fast | Official | 81.12 | 80.87 | 80.94 | −0.18 | 176.4 | 171.4 |
| R-Bench-fast | Matched | 82.59 | 81.12 | 80.61 | **−1.98** | 131.4 | **74.4** |
| MMRel-Adv-fast | Official | 69.01 | 69.01 | 69.27 | +0.26 | 101 | 97 |
| MMRel-Adv-fast | Matched | 69.53 | 74.22 | **75.78** | **+6.25** | 77 | **21** |

Official 与 Phase 2B 公布的 fast 表相差 ≤0.4 Acc（同一批预测、同一 `no/not` 规则）。Matched 全部 Invalid=0。

### 配套指标（Matched）

| Bench | 模型 | P | R | F1 | Yes | TP | TN | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R-Bench 五折 | Base | 81.60 | 84.47 | 83.01 | 52.14 | 583.2 | 548.6 | 131.4 | 107.2 |
| R-Bench 五折 | V2-old | 84.44 | 76.64 | 80.34 | 45.71 | 529.0 | 582.6 | 97.4 | 161.4 |
| R-Bench 五折 | V2B-2ep | 87.02 | 72.29 | 78.97 | 41.84 | 499.0 | 605.6 | 74.4 | 191.4 |
| MMRel | Base | 66.95 | 79.59 | 72.73 | 60.68 | 156 | 111 | 77 | 40 |
| MMRel | V2-old | 76.22 | 71.94 | 74.02 | 48.18 | 141 | 144 | 44 | 55 |
| MMRel | V2B-2ep | 85.52 | 63.27 | 72.73 | 37.76 | 124 | 167 | 21 | 72 |

Gold Yes：R-Bench-fast 约 65%；MMRel-fast 196/384 = 51.0%。V2B matched 的 Yes 41.8% / 37.8%，明显低于 gold。

---

## 4. Prompt sensitivity（同一题 Official vs Matched）

| Model | Benchmark | Agreement | Official Wrong→Matched Correct | Official Correct→Matched Wrong | Yes→No | No→Yes |
|---|---|---:|---:|---:|---:|---:|
| Base | R-Bench | 91.3 | 87 | 80 | 125 | 42 |
| V2-old | R-Bench | 87.9 | 93 | 140 | 218 | 15 |
| V2B-2ep | R-Bench | 85.0 | 103 | 186 | **283** | 6 |
| Base | MMRel | 80.7 | 38 | 36 | 60 | 14 |
| V2-old | MMRel | 74.5 | 59 | 39 | 93 | 5 |
| V2B-2ep | MMRel | 65.4 | 79 | 54 | **130** | 3 |

LoRA 对训练模板更敏感，而且几乎是单向的 Yes→No。不是“只有 Base 换 prompt 才会动”。

---

## 5. Paired 统计（unique items，Base vs V2B-2ep）

| Bench | Prompt | ΔAcc | 95% CI | McNemar p | ΔFP |
|---|---|---:|---|---:|---:|
| R-Bench | Official | −0.47 | [−1.09, +0.21] | 0.20 | −5 |
| R-Bench | Matched | **−5.15** | [−6.60, −3.69] | 1.9×10⁻¹¹ | −57 |
| MMRel | Official | +0.26 | [−1.30, +1.82] | 1.00 | −4 |
| MMRel | Matched | **+6.25** | [+1.56, +10.94] | 0.014 | −56 |

Official：V2B ≈ Base，CI 含 0。  
Matched：R-Bench 上 V2B **显著更差**；MMRel 上 Acc 显著更好，但伴随 FN 18→72、Recall 91→63。

---

## 6. 判定

对照计划四类：

- **不是 A。** 换成训练 prompt 之后，R-Bench 上 LoRA 相对 Base 的 Acc 优势没有出现，反而显著变差。
- **不是单纯 B。** Matched 并非“大家都差不多”；prompt 明显改变了 LoRA 行为。
- **不是 C。** 不是三种模型一起平移、相对差距不变。Base matched Acc 略升，V2B 在 R-Bench 下降、在 MMRel 靠拒识拉升。
- **是 D / Phase 2B.5-C。** 训练模板重新激活了 learned rejection：FP↓、TN↑、Yes↓，但 FN 明显上升。R-Bench Acc 下降；MMRel Acc 上升来自把大量题打成 No（Yes 37.8% vs gold 51%），不是更准的 relation discrimination。

一句话：

> **外部失败主要不是“换回训练 prompt 就能释放已学会的关系判定”。Prompt matching 主要诱导 No-bias，不能当成成功的 cross-benchmark transfer。**

因此下一步仍应按 2B No-Go A：**3k paired 1:1、带明确 subject/relation/object 的数据扩展**，而不是先做 multi-prompt / instruction-diverse training（那是 Case A 的下一步）。

---

## 7. 产物

| 产物 | 路径 |
|---|---|
| Matched 题 | `data/phase2b5/` |
| Conversion QC + 200 条 review | `eval_results/qwen/phase2b5/{conversion_qc.json,reviews/}` |
| 预测 | `eval_results/qwen/phase2b5/answers/` |
| 分析（含 bootstrap / McNemar） | `eval_results/qwen/phase2b5/metrics/phase2b5_analysis.json` |
| 脚本 | `scripts/phase2b5/` |
