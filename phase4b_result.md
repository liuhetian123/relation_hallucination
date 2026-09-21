# Phase 4B 结果：外部平坦的归因诊断（OOD 覆盖度 vs 能力不足）

不训练、不对 base/s3000/old/clean 做新推理。任务 A 在冻结 unique questions 上按训练 relation 词表分层；任务 B 对 100 条 clean negative 做独立 InternVL visual falsity 复验。

## 对账

重算 unique-question Acc，须与 `phase4a_train_result.md` / `phase3a_result.md` Confusion 表相差 ≤ 0.05 pp。

| Model | R-Bench Acc | 发布值 | MMRel Acc | 发布值 |
|---|---:|---:|---:|---:|
| base | 83.1081081081081 | 83.11 | 69.01041666666666 | 69.01 |
| s3000 | 83.57588357588358 | 83.58 | 71.09375 | 71.09 |
| old | 83.26403326403326 | 83.26 | 70.3125 | 70.31 |
| clean | 82.95218295218295 | 82.95 | 70.83333333333334 | 70.83 |

## Relation 抽取

- R-Bench n=1924，unparsed=442 (22.97%)
- MMRel n=384，unparsed=0 (0.00%)
- old/s3000 词表大小：424；clean 词表：535
- lemma-seen（old 词表）：R-Bench 1132 / MMRel 258
- lemma-seen（clean 词表）：R-Bench 1119 / MMRel 261

AMBER-dr-fast 跳过：题面几乎全是 “Is there direct contact between X and Y?”，无法抽取多样 relation。

Parse reasons：

```json
{
  "rbench": {
    "verb": 1295,
    "unparsed": 29,
    "prep": 187,
    "existence": 257,
    "attribute": 156
  },
  "mmrel_adv": {
    "mmrel_action": 175,
    "mmrel_spatial_to": 182,
    "mmrel_spatial_prep": 27
  }
}
```

R-Bench unparsed > 15%，抽样 20 条：

| reason | statement |
|---|---|
| existence | There is a sheep near the toy tractor. |
| existence | There are any books on the shelves that are not in the same language as the label on the shelf. |
| existence | There are any sliced vegetables that are not laying on the plate. |
| existence | There are any people standing on the balcony in the image. |
| existence | There are any chairs inside the building. |
| existence | There is a fountain behind the skunk. |
| existence | There are any gumdrops in the image. |
| attribute | All three knives of are the same size. |
| existence | There is a building in the image. |
| attribute | The shower curtain in the image is white. |
| attribute | The car in the image is a classic car. |
| existence | There are any wrenches in the tool belt. |
| attribute | The drink in a is clear cup. |
| existence | There are any people on the rafts in the image. |
| attribute | The bread in the image is still in the oven. |
| existence | There are any computers on the wall in the image. |
| attribute | The person in the image is a woman wearing makeup. |
| existence | There are people on horses in the image. |
| existence | There are any tools or accessories in the image that are not related to sewing. |
| attribute | The sewing machine in the image is a modern electronic device. |

## 主分析（lemma seen/unseen）

seen = 外部题 relation 在对应训练词表上 exact 或 lemma 命中。s3000/old 用 old 词表，clean 用 clean 词表。

### rbench

#### s3000_vs_base（vocab=old）

| layer | n | Acc_A | Acc_B | ΔAcc | 95% CI | McNemar p | Yes_A | Yes_B | FP/FN_A | FP/FN_B |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| seen | 1132 | 85.34 | 84.98 | +0.35 | [-0.80, +1.41] | 0.6352562959972483 | 72.88 | 70.94 | 94/72 | 85/85 |
| unseen | 350 | 82.00 | 82.29 | -0.29 | [-2.57, +2.00] | 1.0 | 66.29 | 64.29 | 42/21 | 38/24 |
| unparsed | 442 | 81.22 | 79.41 | +1.81 | [-0.45, +4.30] | 0.20124262095772402 | 58.14 | 52.71 | 53/30 | 45/46 |

DiD Δ(seen)−Δ(unseen) = +0.64 pp，CI [-1.89, +3.17]；n_seen=1132 n_unseen=350。

敏感性（exact / synonym-family）：

| level | seen n | seen ΔAcc | seen CI | unseen n | unseen ΔAcc | unseen CI | DiD | DiD CI |
|---|---:|---:|---|---:|---:|---|---:|---|
| exact | 1104 | +0.27 | [-0.82, +1.36] | 378 | +0.00 | [-2.12, +2.12] | +0.27 | [-2.20, +2.76] |
| lemma | 1132 | +0.35 | [-0.80, +1.41] | 350 | -0.29 | [-2.57, +2.00] | +0.64 | [-1.89, +3.17] |
| synonym-family | 1134 | +0.35 | [-0.79, +1.50] | 348 | -0.29 | [-2.59, +2.01] | +0.64 | [-1.88, +3.27] |

#### old_vs_base（vocab=old）

| layer | n | Acc_A | Acc_B | ΔAcc | 95% CI | McNemar p | Yes_A | Yes_B | FP/FN_A | FP/FN_B |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| seen | 1132 | 85.60 | 84.98 | +0.62 | [-0.44, +1.77] | 0.3366683676100388 | 72.79 | 70.94 | 92/71 | 85/85 |
| unseen | 350 | 80.57 | 82.29 | -1.71 | [-4.00, +0.57] | 0.2112995473337105 | 66.57 | 64.29 | 45/23 | 38/24 |
| unparsed | 442 | 80.32 | 79.41 | +0.90 | [-1.58, +3.39] | 0.5838824207703651 | 58.60 | 52.71 | 56/31 | 45/46 |

DiD Δ(seen)−Δ(unseen) = +2.33 pp，CI [-0.11, +4.86]；n_seen=1132 n_unseen=350。

敏感性（exact / synonym-family）：

| level | seen n | seen ΔAcc | seen CI | unseen n | unseen ΔAcc | unseen CI | DiD | DiD CI |
|---|---:|---:|---|---:|---:|---|---:|---|
| exact | 1104 | +0.54 | [-0.54, +1.63] | 378 | -1.32 | [-3.44, +0.79] | +1.87 | [-0.53, +4.34] |
| lemma | 1132 | +0.62 | [-0.44, +1.77] | 350 | -1.71 | [-4.00, +0.57] | +2.33 | [-0.11, +4.86] |
| synonym-family | 1134 | +0.62 | [-0.44, +1.76] | 348 | -1.72 | [-4.02, +0.57] | +2.34 | [-0.11, +4.90] |

#### clean_vs_base（vocab=clean）

| layer | n | Acc_A | Acc_B | ΔAcc | 95% CI | McNemar p | Yes_A | Yes_B | FP/FN_A | FP/FN_B |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| seen | 1119 | 84.63 | 84.72 | -0.09 | [-0.89, +0.71] | 1.0 | 70.96 | 70.51 | 89/83 | 86/85 |
| unseen | 363 | 82.92 | 83.20 | -0.28 | [-1.93, +1.38] | 1.0 | 65.56 | 65.84 | 37/25 | 37/24 |
| unparsed | 442 | 79.64 | 79.41 | +0.23 | [-1.58, +2.04] | 1.0 | 54.75 | 52.71 | 49/41 | 45/46 |

DiD Δ(seen)−Δ(unseen) = +0.19 pp，CI [-1.64, +2.02]；n_seen=1119 n_unseen=363。

敏感性（exact / synonym-family）：

| level | seen n | seen ΔAcc | seen CI | unseen n | unseen ΔAcc | unseen CI | DiD | DiD CI |
|---|---:|---:|---|---:|---:|---|---:|---|
| exact | 1081 | -0.19 | [-1.02, +0.65] | 401 | +0.00 | [-1.50, +1.50] | -0.19 | [-1.96, +1.62] |
| lemma | 1119 | -0.09 | [-0.89, +0.71] | 363 | -0.28 | [-1.93, +1.38] | +0.19 | [-1.64, +2.02] |
| synonym-family | 1121 | -0.09 | [-0.89, +0.71] | 361 | -0.28 | [-1.94, +1.39] | +0.19 | [-1.64, +2.04] |

#### clean_vs_old（vocab=clean）

| layer | n | Acc_A | Acc_B | ΔAcc | 95% CI | McNemar p | Yes_A | Yes_B | FP/FN_A | FP/FN_B |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| seen | 1119 | 84.63 | 84.99 | -0.36 | [-1.25, +0.54] | 0.5402913746074199 | 70.96 | 72.39 | 89/83 | 95/73 |
| unseen | 363 | 82.92 | 82.64 | +0.28 | [-1.38, +2.20] | 1.0 | 65.56 | 68.04 | 37/25 | 42/21 |
| unparsed | 442 | 79.64 | 80.32 | -0.68 | [-2.71, +1.36] | 0.6625205835400575 | 54.75 | 58.60 | 49/41 | 56/31 |

DiD Δ(seen)−Δ(unseen) = -0.63 pp，CI [-2.64, +1.37]；n_seen=1119 n_unseen=363。

敏感性（exact / synonym-family）：

| level | seen n | seen ΔAcc | seen CI | unseen n | unseen ΔAcc | unseen CI | DiD | DiD CI |
|---|---:|---:|---|---:|---:|---|---:|---|
| exact | 1081 | -0.37 | [-1.30, +0.46] | 401 | +0.25 | [-1.50, +1.75] | -0.62 | [-2.49, +1.22] |
| lemma | 1119 | -0.36 | [-1.25, +0.54] | 363 | +0.28 | [-1.38, +2.20] | -0.63 | [-2.64, +1.37] |
| synonym-family | 1121 | -0.36 | [-1.25, +0.54] | 361 | +0.28 | [-1.66, +2.22] | -0.63 | [-2.63, +1.37] |

### mmrel_adv

#### s3000_vs_base（vocab=old）

| layer | n | Acc_A | Acc_B | ΔAcc | 95% CI | McNemar p | Yes_A | Yes_B | FP/FN_A | FP/FN_B |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| seen | 258 | 67.44 | 64.73 | +2.71 | [-0.39, +5.81] | 0.14561009539686695 | 70.16 | 75.19 | 67/17 | 77/14 |
| unseen | 126 | 78.57 | 77.78 | +0.79 | [-2.38, +3.97] | 1.0 | 63.49 | 67.46 | 21/6 | 24/4 |
| unparsed **功效不足** | 0 | 0.00 | 0.00 | +0.00 | [+0.00, +0.00] | 1.0 | 0.00 | 0.00 | 0/0 | 0/0 |

DiD Δ(seen)−Δ(unseen) = +1.92 pp，CI [-2.81, +6.61]；n_seen=258 n_unseen=126。

敏感性（exact / synonym-family）：

| level | seen n | seen ΔAcc | seen CI | unseen n | unseen ΔAcc | unseen CI | DiD | DiD CI |
|---|---:|---:|---|---:|---:|---|---:|---|
| exact | 153 | +5.23 | [+1.31, +9.80] | 231 | +0.00 | [-2.60, +2.60] | +5.23 | [+0.23, +10.45] |
| lemma | 258 | +2.71 | [-0.39, +5.81] | 126 | +0.79 | [-2.38, +3.97] | +1.92 | [-2.81, +6.61] |
| synonym-family | 258 | +2.71 | [-0.39, +5.81] | 126 | +0.79 | [-2.38, +3.97] | +1.92 | [-2.81, +6.61] |

#### old_vs_base（vocab=old）

| layer | n | Acc_A | Acc_B | ΔAcc | 95% CI | McNemar p | Yes_A | Yes_B | FP/FN_A | FP/FN_B |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| seen | 258 | 66.67 | 64.73 | +1.94 | [-0.78, +5.04] | 0.30169958247834794 | 70.93 | 75.19 | 69/17 | 77/14 |
| unseen | 126 | 77.78 | 77.78 | +0.00 | [-3.17, +3.17] | 0.6170750774519738 | 64.29 | 67.46 | 22/6 | 24/4 |
| unparsed **功效不足** | 0 | 0.00 | 0.00 | +0.00 | [+0.00, +0.00] | 1.0 | 0.00 | 0.00 | 0/0 | 0/0 |

DiD Δ(seen)−Δ(unseen) = +1.94 pp，CI [-2.38, +6.26]；n_seen=258 n_unseen=126。

敏感性（exact / synonym-family）：

| level | seen n | seen ΔAcc | seen CI | unseen n | unseen ΔAcc | unseen CI | DiD | DiD CI |
|---|---:|---:|---|---:|---:|---|---:|---|
| exact | 153 | +3.92 | [+0.00, +7.84] | 231 | -0.43 | [-3.03, +2.16] | +4.35 | [-0.21, +9.14] |
| lemma | 258 | +1.94 | [-0.78, +5.04] | 126 | +0.00 | [-3.17, +3.17] | +1.94 | [-2.38, +6.26] |
| synonym-family | 258 | +1.94 | [-0.78, +5.04] | 126 | +0.00 | [-3.17, +3.17] | +1.94 | [-2.38, +6.26] |

#### clean_vs_base（vocab=clean）

| layer | n | Acc_A | Acc_B | ΔAcc | 95% CI | McNemar p | Yes_A | Yes_B | FP/FN_A | FP/FN_B |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| seen | 261 | 68.20 | 65.52 | +2.68 | [+0.00, +5.36] | 0.0960923294556733 | 67.43 | 72.41 | 66/17 | 76/14 |
| unseen | 123 | 76.42 | 76.42 | +0.00 | [-3.25, +3.25] | 0.6170750774519738 | 69.92 | 73.17 | 23/6 | 25/4 |
| unparsed **功效不足** | 0 | 0.00 | 0.00 | +0.00 | [+0.00, +0.00] | 1.0 | 0.00 | 0.00 | 0/0 | 0/0 |

DiD Δ(seen)−Δ(unseen) = +2.68 pp，CI [-1.34, +6.75]；n_seen=261 n_unseen=123。

敏感性（exact / synonym-family）：

| level | seen n | seen ΔAcc | seen CI | unseen n | unseen ΔAcc | unseen CI | DiD | DiD CI |
|---|---:|---:|---|---:|---:|---|---:|---|
| exact | 152 | +4.61 | [+1.32, +8.55] | 232 | +0.00 | [-2.59, +2.59] | +4.61 | [+0.23, +9.19] |
| lemma | 261 | +2.68 | [+0.00, +5.36] | 123 | +0.00 | [-3.25, +3.25] | +2.68 | [-1.34, +6.75] |
| synonym-family | 261 | +2.68 | [+0.00, +5.36] | 123 | +0.00 | [-3.25, +3.25] | +2.68 | [-1.34, +6.75] |

#### clean_vs_old（vocab=clean）

| layer | n | Acc_A | Acc_B | ΔAcc | 95% CI | McNemar p | Yes_A | Yes_B | FP/FN_A | FP/FN_B |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| seen | 261 | 68.20 | 67.43 | +0.77 | [-1.15, +2.68] | 0.6830913983096087 | 67.43 | 68.20 | 66/17 | 68/17 |
| unseen | 123 | 76.42 | 76.42 | +0.00 | [-2.44, +2.44] | 0.4795001221869535 | 69.92 | 69.92 | 23/6 | 23/6 |
| unparsed **功效不足** | 0 | 0.00 | 0.00 | +0.00 | [+0.00, +0.00] | 1.0 | 0.00 | 0.00 | 0/0 | 0/0 |

DiD Δ(seen)−Δ(unseen) = +0.77 pp，CI [-2.06, +3.59]；n_seen=261 n_unseen=123。

敏感性（exact / synonym-family）：

| level | seen n | seen ΔAcc | seen CI | unseen n | unseen ΔAcc | unseen CI | DiD | DiD CI |
|---|---:|---:|---|---:|---:|---|---:|---|
| exact | 152 | +0.66 | [-1.97, +3.29] | 232 | +0.43 | [-0.86, +2.16] | +0.23 | [-3.04, +3.52] |
| lemma | 261 | +0.77 | [-1.15, +2.68] | 123 | +0.00 | [-2.44, +2.44] | +0.77 | [-2.06, +3.59] |
| synonym-family | 261 | +0.77 | [-1.15, +2.68] | 123 | +0.00 | [-2.44, +2.44] | +0.77 | [-2.06, +3.59] |

### MMRel 图片领域（lemma）

**clean_vs_base / vg**

| layer | n | Acc_base | Acc_clean | ΔAcc | 95% CI |
|---|---:|---:|---:|---:|---|
| seen | 226 | 65.04 | 67.26 | +2.21 | [-0.44, +5.31] |
| unseen | 109 | 78.90 | 78.90 | +0.00 | [-2.75, +2.75] |

**clean_vs_base / dalle**

| layer | n | Acc_base | Acc_clean | ΔAcc | 95% CI |
|---|---:|---:|---:|---:|---|
| seen | 35 | 68.57 | 74.29 | +5.71 | [+0.00, +14.29] |
| unseen | 14 | 57.14 | 57.14 | +0.00 | [-21.43, +21.43] |

## 任务 A 判定

各对比 lemma 标签：s3000_vs_base/rbench=B, s3000_vs_base/mmrel_adv=A, old_vs_base/rbench=B, old_vs_base/mmrel_adv=B, clean_vs_base/rbench=B, clean_vs_base/mmrel_adv=A

**Case B。** 量级/能力问题。下一步优先 Qwen2.5-VL-7B 或加训练 exposure（2 epoch），不再叠数据技巧。

R-Bench 上 seen/unseen 的 ΔAcc 与 DiD 的 CI 均含 0。MMRel 上 clean/s3000 vs base 的 seen 点估计约 +2.7 pp（unseen ≈ 0），但 seen CI 下界触及 0，且 unseen 层 n≈123，功效不足，达不到 Case A 的“CI 不含 0”标准。不把 MMRel 的点估计当成覆盖度主证据。

## 任务 B：Clean Negative 标签噪声（100 条）

- n=100
- InternVL Yes（负例其实成立）：0.00%
- Uncertain：0.00%
- No（负例确实不成立）：100.00%
- 噪声上限（Yes 或 Uncertain）：0.00%
- 自动判定：`clean_ok_leq_5pct`

二次 pass 使用与生成时相同的 InternVL3.5-8B，新对话、不带原 candidates。100/100 判 No，表示教师对自己当初的 visual-falsity 过滤高度一致，不是人工金标。

人工列留空，不阻塞本报告。页面：`data/phase4a/qc/review_100_falsity.html`。

噪声率 ≤ 5%：clean 数据质量可信，train neg Acc 83.5% 主要反映难度。

## 本阶段问题

> 微调带来的关系判别增益，在外部 benchmark 上是否集中于训练 relation 词表覆盖到的题目？（顺带：clean negative 的视觉假性标签噪声有多大？）

**回答：Case B。** 量级/能力问题。下一步优先 Qwen2.5-VL-7B 或加训练 exposure（2 epoch），不再叠数据技巧。

