# Phase 1.5：Held-out RelSim Relation MCQ

不重新训练、不改 Phase 1 checkpoint。只检查 Explicit-1k / Typed-1k LoRA 是否在**未见过的 RelSim 图**上学会了关系识别。

---

## 数据

- 训练排除：`data/relsim_llava_1k.json` 的全部 **1000** 张图（不是只排除 955）。
- Caption：`/data/lht/relsim_dataset/relsim_llava_100k.json`（65946 条，与 Phase 1 同源 LLaVA 格式）。
- 图片：`/data/lht/relsim_dataset/relsim_images`
- GT relation：从 anonymous caption **前两个 placeholder 之间**抽取动词短语（如 `balancing on`），不用评测模型生成。
- 错误选项：从全集 relation vocab（3078）中采样，排除同动词 / 子串近义。
- seed=42；答案位置 **A/B/C/D = 50/50/50/50**。

冻结题目：`eval_results/qwen/phase15/heldout_200.jsonl`  
人工抽查：`eval_results/qwen/phase15/review_20.jsonl`

### QC

| 检查项 | 值 |
|---|---:|
| Total questions | 200 |
| Unique images | 200 |
| Training overlap | 0 |
| Missing images | 0 |
| Missing captions | 0 |
| Unique GT relations | 120 |
| Duplicate options | 0 |
| GT missing from options | 0 |
| A/B/C/D | 50/50/50/50 |
| Relation vocab size | 3078 |

---

## Accuracy

三个模型同一 prompt、greedy、`max_new_tokens=8`。随机水平 25%。

| Model | Correct | Wrong | Invalid | Accuracy | Mean GT Margin |
|---|---:|---:|---:|---:|---:|
| Base | 153 | 47 | 0 | 76.5% | 1.245 |
| Explicit-1k | 152 | 48 | 0 | 76.0% | 1.226 |
| Typed-1k | 154 | 46 | 0 | 77.0% | 1.179 |

- Explicit − Base = **−0.5%**
- Typed − Base = **+0.5%**
- Typed − Explicit = **+1.0%**

GT margin = `logp(正确选项字母) − max logp(错误选项字母)`。LoRA 没有提高正确关系的置信度。

原始预测：

- `eval_results/qwen/phase15/answers/qwen25vl3b_base_mcq.jsonl`
- `eval_results/qwen/phase15/answers/qwen25vl3b_explicit_1k_mcq.jsonl`
- `eval_results/qwen/phase15/answers/qwen25vl3b_typed_1k_mcq.jsonl`

---

## Paired transitions

| Transition | Count |
|---|---:|
| Base wrong → Explicit correct | 2 |
| Base correct → Explicit wrong | 3 |
| Base wrong → Typed correct | 2 |
| Base correct → Typed wrong | 1 |

## Prediction agreement

| Pair | Agreement |
|---|---:|
| Base / Explicit | 97.5% |
| Base / Typed | 98.5% |
| Explicit / Typed | 97.0% |

## 统计检验（同一 200 题）

- **Explicit vs Base**：ΔAcc=−0.5%（bootstrap 95% CI [−3.0%, +1.5%]）；McNemar n01=2, n10=3, p=1.00
- **Typed vs Base**：ΔAcc=+0.5%（bootstrap 95% CI [−1.0%, +2.5%]）；McNemar n01=2, n10=1, p=1.00
- **Typed vs Explicit**：ΔAcc=+1.0%（bootstrap 95% CI [−1.5%, +3.5%]）；McNemar n01=4, n10=2, p=0.68

差异均包含 0，不能拒绝「与 Base 相同」。

---

## Decision

```text
Phase 1.5-B:
Training intervention too weak / no measurable adaptation.
```

Base 已在该 probe 上达到 76.5%，两个 LoRA 与 Base 的预测一致率 ≥97%，Acc / GT-margin 都没有可测提升。按计划，下一步应先加强训练（更多 epoch / step 或更大数据），让 held-out RelSim probe 出现明确 adaptation，再谈外部 hallucination benchmark。
