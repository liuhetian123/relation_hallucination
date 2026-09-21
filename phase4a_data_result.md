# Phase 4A-Data 结果：Clean Counterfactual Negative Construction

本阶段不训练模型。用 InternVL3.5-8B 为 Phase 3A SRO3000 重写 linguistically plausible、visually false 的 counterfactual negative。

## 产量

- 输入 attempts：3000
- Clean 保留：2156
- 丢弃：844
- Keep rate：71.9%

丢弃原因：

| reason | n |
|---|---:|
| no_high_confidence_candidate | 844 |

- unique clean negative relations：359
- unique old negative relations：420
- clean 与 old 相同：3
- old grammar-smell rate：15.9%
- clean grammar-smell rate：0.6%

Candidate rank（选中第几个 InternVL candidate）：

| rank | n |
|---|---:|
| 0 | 1061 |
| 1 | 688 |
| 2 | 407 |

Top clean negative relations：

| relation | n |
|---|---:|
| looking at | 166 |
| leaning on | 121 |
| attached to | 93 |
| pushing | 84 |
| standing on | 69 |
| leaning against | 65 |
| jumping from | 56 |
| playing with | 54 |
| in front of | 53 |
| carrying | 52 |
| holding | 46 |
| sitting on | 40 |
| beside | 39 |
| covering | 32 |
| landing on | 31 |
| lying on | 29 |
| chasing | 28 |
| flying over | 28 |
| touching | 28 |
| hanging from | 28 |

## Text-only shortcut QC

InternVL text-only：statement 是否 linguistically/semantically plausible（不看图）。
将 Implausible 视为预测 Negative、Plausible 视为 Positive。

- Positive implausible rate：11.0%
- Old negative implausible rate：66.2%
- Clean negative implausible rate：0.0%
- Paired old text-only Acc：77.6% (n=1000)
- Paired clean text-only Acc：44.5% (n=1000)

计划期望：old Acc 明显更高（75%+），clean 接近 50%–60%。

## 人工 QC

分层抽样表：`data/phase4a/qc/review_300.tsv`（100 spatial / 100 contact / 100 other，按 GT positive relation family）；浏览页：`data/phase4a/qc/review.html`。

**2026-09-16：跳过人工填写。** 不统计 overall clean rate，不作为训练门槛。后续以自动 filter + text-only shortcut 为准。

## 本阶段问题

> 能否利用 InternVL 根据当前图像和真实 SRO，稳定产生语言上自然、语义上合理、但当前图像中不成立的 counterfactual relation，从而消除原有 random negative 中明显的 linguistic shortcut？

训练对照（Old-S3000 vs Clean-S3000）按计划为可选 sanity check，本报告不包含训练数字。

