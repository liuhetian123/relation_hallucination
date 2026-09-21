# Phase 5A 结果：语言先验冲突分解

本阶段不训练。用 base 的 text-only 回答把每道题切成 prior-consistent / prior-conflict，再在冻结看图预测上分层。

> 必须看图才能答对的先验冲突子集，是否既是 base 幻觉的集中地，也是 mf 增益的集中地？

**判定：Case C。** 先验测不出：MMRel 生成式层差 -2.19 pp（< 5 pp）；logprob 第二轮层差 +4.09 pp。 mf 增益在 consistent +6.90、conflict +3.87，不集中在冲突层。 AMBER 生成式层差 +15.43 pp，但不是 mf 增益所在题集。 R-Bench 拒答率/logprob 单边，测量失败。

## 有效性检查

| Bench | source | n | invalid% | Yes | No | degenerate | mode | failed |
|---|---|---:|---:|---:|---:|---|---|---|
| mmrel_adv | main | 384 | 0.00 | 48.70 | 51.30 | False | main | False |
| mmrel_adv (logprob orig) | main | 384 | 0.00 | 56.51 | 43.49 | False | logprob |  |
| rbench | main | 1924 | 0.00 | 93.04 | 6.96 | False | logprob | True |
| rbench (main raw) | main | 1924 | 79.05 | 59.55 | 40.45 | False | text |  |
| rbench (fallback raw) | fallback | 1924 | 0.00 | 1.14 | 98.86 | True | text |  |
| amber_dr | main | 499 | 4.61 | 26.26 | 73.74 | False | main | False |
| amber_dr (logprob orig) | main | 499 | 0.00 | 34.07 | 65.93 | False | logprob |  |
| heldout | main | 393 | 0.00 | 47.33 | 52.67 | False | main | False |
| heldout (logprob orig) | main | 393 | 0.00 | 56.49 | 43.51 | False | logprob |  |

无效/拒答率阈值 10%；Yes/No 单边阈值 95%。分层用上表 `mode` 列。`failed=True` 的题集先验测量不可用，层结果只作附录。

## 对账（冻结看图 unique Acc，必须与已发布数字一致 ±0.05 pp）

| Model | Bench | 已发布 | 复算 | match |
|---|---|---:|---:|---|
| base | rbench | 83.11 | 83.11 | True |
| base | mmrel_adv | 69.01 | 69.01 | True |
| base | heldout | 58.27 | 58.27 | True |
| base | amber_dr | 79.56 | 79.56 | True |
| clean | rbench | 82.95 | 82.95 | True |
| clean | mmrel_adv | 70.83 | 70.83 | True |
| clean | heldout | 67.18 | 67.18 | True |
| clean | amber_dr | 80.16 | 80.16 | True |
| mf | rbench | 82.85 | 82.85 | True |
| mf | mmrel_adv | 74.48 | 74.48 | True |
| mf | heldout | 65.90 | 65.90 | True |
| mf | amber_dr | 79.16 | 79.16 | True |
| s3000 | rbench | 83.58 | 83.58 | True |
| s3000 | mmrel_adv | 71.09 | 71.09 | True |
| s3000 | heldout | 72.77 | 72.77 | True |

## 层规模（由 base text-only 定义）

| Bench | slice | n | gold Yes | prior Yes |
|---|---|---:|---:|---:|
| mmrel_adv | all | 384 | 51.04 | 48.70 |
| mmrel_adv | consistent | 203 | 49.75 | 49.75 |
| mmrel_adv | conflict | 181 | 52.49 | 47.51 |
| mmrel_adv | invalid | 0 | 0.00 | 0.00 |
| rbench | all | 1482 | 68.76 | 93.72 |
| rbench | consistent | 1020 | 95.39 | 95.39 |
| rbench | conflict | 462 | 9.96 | 90.04 |
| rbench | invalid | 0 | 0.00 | 0.00 |
| rbench_full | all | 1924 | 65.18 | 93.04 |
| rbench_full | consistent | 1270 | 94.09 | 94.09 |
| rbench_full | conflict | 654 | 9.02 | 90.98 |
| rbench_full | invalid | 0 | 0.00 | 0.00 |
| amber_dr | all | 499 | 58.52 | 25.05 |
| amber_dr | consistent | 250 | 35.20 | 35.20 |
| amber_dr | conflict | 226 | 83.63 | 16.37 |
| amber_dr | invalid | 23 | 65.22 | 0.00 |
| heldout | all | 393 | 49.11 | 47.33 |
| heldout | consistent | 234 | 47.01 | 47.01 |
| heldout | conflict | 159 | 52.20 | 47.80 |
| heldout | invalid | 0 | 0.00 | 0.00 |

R-Bench 主分析只用 relation-only（verb+prep）；`rbench_full` 仅附录。

## 诊断一：Base 看图 Acc 分层 + 先验锁定率

| Bench | consistent Acc | conflict Acc | 层差 | 95% CI | lock rate | n_cons | n_conf |
|---|---:|---:|---:|---|---:|---:|---:|
| mmrel_adv | 67.98 | 70.17 | -2.19 pp | [-11.43, +6.85] | 50.00 | 203 | 181 |
| rbench | 89.41 | 72.94 | +16.47 pp | [+12.07, +20.93] | 69.97 | 1020 | 462 |
| rbench_full | 87.95 | 73.70 | +14.25 pp | [+10.43, +18.04] | 67.00 | 1270 | 654 |
| amber_dr | 88.00 | 72.57 | +15.43 pp | [+8.46, +22.57] | 59.24 | 250 | 226 |
| heldout | 61.97 | 52.83 | +9.14 pp | [-1.00, +19.14] | 55.98 | 234 | 159 |

层差 = consistent Acc − conflict Acc。锁定率 = 看图预测与 text-only 预测相同的比例（valid 题）。R-Bench 若 `failed=True`，该行只作附录，不进入 Case 判定。

## 诊断二：微调增益分层（主假设：mf 的 MMRel +5.47 pp 集中在 conflict）

### mmrel_adv

| Contrast | layer | ΔAcc | 95% CI | McNemar p | Yes_a | Yes_b | FP/FN_a | FP/FN_b | n |
|---|---|---:|---|---:|---:|---:|---|---|---:|
| mf_vs_base | conflict | +3.87 pp | [-2.21, +9.94] | 0.2812 | 70.17 | 53.04 | 43/11 | 24/23 | 181 |
| mf_vs_base | consistent | +6.90 pp | [+0.99, +12.81] | 0.03496 | 74.88 | 56.16 | 58/7 | 32/19 | 203 |
| mf_vs_base | DiD=Δconf−Δcons | -3.03 pp | [-11.40, +5.20] |  |  |  |  |  |  |
| clean_vs_base | conflict | +0.55 pp | [-2.21, +3.31] | 1 | 70.17 | 66.30 | 43/11 | 39/14 | 181 |
| clean_vs_base | consistent | +2.96 pp | [+0.00, +5.91] | 0.1138 | 74.88 | 69.95 | 58/7 | 50/9 | 203 |
| clean_vs_base | DiD=Δconf−Δcons | -2.40 pp | [-6.58, +1.72] |  |  |  |  |  |  |
| s3000_vs_base | conflict | +1.10 pp | [-1.66, +3.87] | 0.6831 | 70.17 | 66.85 | 43/11 | 39/13 | 181 |
| s3000_vs_base | consistent | +2.96 pp | [-0.99, +6.90] | 0.2113 | 74.88 | 68.97 | 58/7 | 49/10 | 203 |
| s3000_vs_base | DiD=Δconf−Δcons | -1.85 pp | [-6.46, +2.64] |  |  |  |  |  |  |

### rbench

| Contrast | layer | ΔAcc | 95% CI | McNemar p | Yes_a | Yes_b | FP/FN_a | FP/FN_b | n |
|---|---|---:|---|---:|---:|---:|---|---|---:|
| mf_vs_base | conflict | +3.25 pp | [+0.65, +6.06] | 0.02878 | 30.95 | 24.68 | 111/14 | 89/21 | 462 |
| mf_vs_base | consistent | -2.16 pp | [-3.43, -0.98] | 0.001194 | 86.76 | 84.41 | 10/98 | 9/121 | 1020 |
| mf_vs_base | DiD=Δconf−Δcons | +5.40 pp | [+2.45, +8.39] |  |  |  |  |  |  |
| clean_vs_base | conflict | -0.65 pp | [-2.38, +1.08] | 0.6056 | 30.95 | 31.17 | 111/14 | 113/15 | 462 |
| clean_vs_base | consistent | -0.10 pp | [-0.88, +0.59] | 1 | 86.76 | 87.06 | 10/98 | 12/97 | 1020 |
| clean_vs_base | DiD=Δconf−Δcons | -0.55 pp | [-2.38, +1.24] |  |  |  |  |  |  |
| s3000_vs_base | conflict | -2.38 pp | [-4.55, -0.22] | 0.0455 | 30.95 | 32.47 | 111/14 | 120/16 | 462 |
| s3000_vs_base | consistent | +1.18 pp | [+0.10, +2.25] | 0.05183 | 86.76 | 88.92 | 10/98 | 15/81 | 1020 |
| s3000_vs_base | DiD=Δconf−Δcons | -3.56 pp | [-6.00, -1.27] |  |  |  |  |  |  |

### rbench_full

| Contrast | layer | ΔAcc | 95% CI | McNemar p | Yes_a | Yes_b | FP/FN_a | FP/FN_b | n |
|---|---|---:|---|---:|---:|---:|---|---|---:|
| mf_vs_base | conflict | +2.75 pp | [+0.61, +4.89] | 0.0184 | 29.20 | 24.01 | 152/20 | 126/28 | 654 |
| mf_vs_base | consistent | -1.81 pp | [-2.99, -0.63] | 0.004181 | 84.25 | 82.28 | 14/139 | 13/163 | 1270 |
| mf_vs_base | DiD=Δconf−Δcons | +4.56 pp | [+2.10, +7.03] |  |  |  |  |  |  |
| clean_vs_base | conflict | -0.76 pp | [-2.14, +0.61] | 0.4042 | 29.20 | 29.97 | 152/20 | 157/20 | 654 |
| clean_vs_base | consistent | +0.16 pp | [-0.63, +0.94] | 0.8383 | 84.25 | 84.88 | 14/139 | 17/134 | 1270 |
| clean_vs_base | DiD=Δconf−Δcons | -0.92 pp | [-2.61, +0.69] |  |  |  |  |  |  |
| s3000_vs_base | conflict | -2.60 pp | [-4.43, -0.76] | 0.008529 | 29.20 | 31.50 | 152/20 | 168/21 | 654 |
| s3000_vs_base | consistent | +2.05 pp | [+1.02, +3.15] | 0.000407 | 84.25 | 87.24 | 14/139 | 20/107 | 1270 |
| s3000_vs_base | DiD=Δconf−Δcons | -4.65 pp | [-6.78, -2.57] |  |  |  |  |  |  |

### amber_dr

| Contrast | layer | ΔAcc | 95% CI | McNemar p | Yes_a | Yes_b | FP/FN_a | FP/FN_b | n |
|---|---|---:|---|---:|---:|---:|---|---|---:|
| mf_vs_base | conflict | -3.54 pp | [-7.08, +0.00] | 0.09896 | 59.73 | 56.19 | 4/58 | 4/66 | 226 |
| mf_vs_base | consistent | +0.00 pp | [-2.40, +2.00] | 0.7237 | 33.60 | 31.20 | 13/17 | 10/20 | 250 |
| mf_vs_base | DiD=Δconf−Δcons | -3.54 pp | [-7.88, +0.72] |  |  |  |  |  |  |
| clean_vs_base | conflict | -0.44 pp | [-3.10, +2.21] | 1 | 59.73 | 60.18 | 4/58 | 5/58 | 226 |
| clean_vs_base | consistent | -0.40 pp | [-2.40, +1.60] | 1 | 33.60 | 33.20 | 13/17 | 13/18 | 250 |
| clean_vs_base | DiD=Δconf−Δcons | -0.04 pp | [-3.41, +3.33] |  |  |  |  |  |  |
| s3000_vs_base | conflict | +0.44 pp | [-3.54, +4.42] | 1 | 59.73 | 61.95 | 4/58 | 6/55 | 226 |
| s3000_vs_base | consistent | +0.80 pp | [-1.60, +3.20] | 0.7237 | 33.60 | 34.40 | 13/17 | 13/15 | 250 |
| s3000_vs_base | DiD=Δconf−Δcons | -0.36 pp | [-5.01, +4.15] |  |  |  |  |  |  |

### heldout

| Contrast | layer | ΔAcc | 95% CI | McNemar p | Yes_a | Yes_b | FP/FN_a | FP/FN_b | n |
|---|---|---:|---|---:|---:|---:|---|---|---:|
| mf_vs_base | conflict | +1.89 pp | [-3.14, +6.92] | 0.6276 | 81.76 | 72.33 | 61/14 | 52/20 | 159 |
| mf_vs_base | consistent | +11.54 pp | [+6.84, +16.24] | 1.917e-05 | 76.50 | 60.68 | 79/10 | 47/15 | 234 |
| mf_vs_base | DiD=Δconf−Δcons | -9.65 pp | [-16.57, -2.48] |  |  |  |  |  |  |
| clean_vs_base | conflict | +6.29 pp | [+1.89, +11.32] | 0.02445 | 81.76 | 74.21 | 61/14 | 50/15 | 159 |
| clean_vs_base | consistent | +10.68 pp | [+6.41, +14.96] | 1.629e-05 | 76.50 | 64.10 | 79/10 | 52/12 | 234 |
| clean_vs_base | DiD=Δconf−Δcons | -4.39 pp | [-10.96, +2.17] |  |  |  |  |  |  |
| s3000_vs_base | conflict | +8.81 pp | [+2.52, +15.09] | 0.01079 | 81.76 | 79.25 | 61/14 | 52/9 | 159 |
| s3000_vs_base | consistent | +18.38 pp | [+13.25, +23.93] | 1.973e-09 | 76.50 | 64.96 | 79/10 | 44/2 | 234 |
| s3000_vs_base | DiD=Δconf−Δcons | -9.57 pp | [-17.62, -1.30] |  |  |  |  |  |  |

conflict 层 gold 与先验反向，「无脑翻转先验」也能涨分；须同时看 consistent 层是否对称下降，以及 Yes ratio / FP/FN。

## 第二轮：原题 Yes/No logprob margin（不新做推理）

生成式先验在 MMRel 上有效但层差 < 5 pp，按方案改用首 token logprob 再切一次层。

| Bench | logprob Yes | consistent Acc | conflict Acc | 层差 | 95% CI | n_cons | n_conf |
|---|---:|---:|---:|---:|---|---:|---:|
| mmrel_adv | 56.51 | 70.94 | 66.85 | +4.09 pp | [-5.25, +13.13] | 203 | 181 |
| rbench | 93.72 | 89.41 | 72.94 | +16.47 pp | [+12.07, +20.93] | 1020 | 462 |
| rbench_full | 93.04 | 87.95 | 73.70 | +14.25 pp | [+10.43, +18.04] | 1270 | 654 |
| amber_dr | 34.07 | 85.45 | 74.11 | +11.35 pp | [+4.20, +18.61] | 275 | 224 |
| heldout | 56.49 | 65.09 | 48.45 | +16.64 pp | [+6.69, +26.44] | 232 | 161 |

## 次要：微调是否改写了先验（MMRel text-only）

| Contrast | n | 一致率 |
|---|---:|---:|
| clean_vs_base | 384 | 93.49 |
| mf_vs_base | 384 | 63.80 |

## 交叉：conflict × Phase 4B lemma seen/unseen（n<50 只描述）

| Contrast | Bench | cell | n | ΔAcc | 检验 |
|---|---|---|---:|---:|---|
| mf_vs_base | mmrel_adv | conflict×seen | 112 | +8.04 pp | p=0.09529283802345662 |
| mf_vs_base | mmrel_adv | conflict×unseen | 69 | -2.90 pp | p=0.7236736098317631 |
| mf_vs_base | mmrel_adv | consistent×seen | 149 | +6.71 pp | p=0.06619257972219343 |
| mf_vs_base | mmrel_adv | consistent×unseen | 54 | +7.41 pp | p=0.4226780741706354 |
| clean_vs_base | mmrel_adv | conflict×seen | 112 | +0.89 pp | p=1.0 |
| clean_vs_base | mmrel_adv | conflict×unseen | 69 | +0.00 pp | p=0.4795001221869535 |
| clean_vs_base | mmrel_adv | consistent×seen | 149 | +4.03 pp | p=0.07709987174354177 |
| clean_vs_base | mmrel_adv | consistent×unseen | 54 | +0.00 pp | p=0.4795001221869535 |
| s3000_vs_base | mmrel_adv | conflict×seen | 112 | +2.68 pp | p=0.37109336952269756 |
| s3000_vs_base | mmrel_adv | conflict×unseen | 69 | -1.45 pp | p=1.0 |
| s3000_vs_base | mmrel_adv | consistent×seen | 146 | +2.74 pp | p=0.3864762307712327 |
| s3000_vs_base | mmrel_adv | consistent×unseen | 57 | +3.51 pp | p=0.6170750774519738 |
| mf_vs_base | rbench | conflict×seen | 336 | +2.68 pp | p=0.13739482585580087 |
| mf_vs_base | rbench | conflict×unseen | 126 | +4.76 pp | p=0.14891467317876567 |
| mf_vs_base | rbench | consistent×seen | 783 | -2.30 pp | p=0.0019107751373644405 |
| mf_vs_base | rbench | consistent×unseen | 237 | -1.69 pp | p=0.3864762307712327 |
| clean_vs_base | rbench | conflict×seen | 336 | -1.19 pp | p=0.3864762307712327 |
| clean_vs_base | rbench | conflict×unseen | 126 | +0.79 pp | p=1.0 |
| clean_vs_base | rbench | consistent×seen | 783 | +0.38 pp | p=0.5049850750938458 |
| clean_vs_base | rbench | consistent×unseen | 237 | -1.69 pp | p=0.22067136191984682 |
| s3000_vs_base | rbench | conflict×seen | 333 | -3.00 pp | p=0.033894853524689274 |
| s3000_vs_base | rbench | conflict×unseen | 129 | -0.78 pp | p=1.0 |
| s3000_vs_base | rbench | consistent×seen | 799 | +1.50 pp | p=0.019016473672300544 |
| s3000_vs_base | rbench | consistent×unseen | 221 | +0.00 pp | p=0.7518296340458492 |

## 本阶段问题

> **Base 模型的关系幻觉是否集中在 prior-conflict 子集？Phase 4C 的 mf 增益是否也集中在该子集？**

Case C → 生成式与 logprob 两轮都失败。若还要保留该框架，只剩对比式先验（同题正反两问）；否则不要把 conflict 子集 Acc 当作主轴。

