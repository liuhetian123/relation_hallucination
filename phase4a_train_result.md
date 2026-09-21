# Phase 4A 训练结果：Old vs Clean Negative

Qwen2.5-VL-3B LoRA（r=16, α=32），1 epoch，同一 2156 张图 / 1:1 Positive–Negative。Old 用 Phase 3A dirty random negative；Clean 用 InternVL counterfactual negative。s3000 是 Phase 3A 全量 3000 对 dirty 训练，仅作 historical reference（图集更大）。

人工 QC 已于 2026-09-16 跳过。

## Text-only shortcut（数据侧，训练前）

- Old text-only Acc：77.60 (n=1000)
- Clean text-only Acc：44.50 (n=1000)

## 训练自检

| Model | Acc | Pos Acc | Neg Acc | Yes | n |
|---|---:|---:|---:|---:|---:|
| old | 92.60 | 92.07 | 93.14 | 49.47 | 4312 |
| clean | 79.66 | 75.83 | 83.49 | 46.17 | 4312 |

## Gate 1：冻结 393 held-out verification

| Model | Acc | Pos Acc | Random Neg Acc | Hard Neg Acc | Hard FP | Yes | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 58.27 | 87.56 | 21.00 | 39.00 | 61.00 | 78.63 | 393 |
| s3000 | 72.77 | 94.30 | 50.00 | 54.00 | 46.00 | 70.74 | 393 |
| old | 65.65 | 93.78 | 33.00 | 44.00 | 56.00 | 77.35 | 393 |
| clean | 67.18 | 86.01 | 41.00 | 57.00 | 43.00 | 68.19 | 393 |

## Gate 2：冻结 external fast subsets（官方 prompt）

| Model | R-Bench Acc | R-Bench F1 | MMRel Acc | MMRel F1 | AMBER Acc |
|---|---:|---:|---:|---:|---:|
| base | 81.54 | 82.72 | 69.01 | 74.95 | 79.56 |
| s3000 | 81.30 | 82.95 | 71.09 | 75.71 | 81.36 |
| old | 81.01 | 82.73 | 70.31 | 75.22 | 81.96 |
| clean | 81.39 | 82.73 | 70.83 | 75.55 | 80.16 |

### Confusion（unique questions）

| Model | Bench | Acc | FP | FN | Recall | Yes |
|---|---|---:|---:|---:|---:|---:|
| base | rbench | 83.11 | 166 | 159 | 87.32 | 65.54 |
| base | mmrel_adv | 69.01 | 101 | 18 | 90.82 | 72.66 |
| s3000 | rbench | 83.58 | 188 | 128 | 89.79 | 68.30 |
| s3000 | mmrel_adv | 71.09 | 88 | 23 | 88.27 | 67.97 |
| old | rbench | 83.26 | 192 | 130 | 89.63 | 68.40 |
| old | mmrel_adv | 70.31 | 91 | 23 | 88.27 | 68.75 |
| clean | rbench | 82.95 | 174 | 154 | 87.72 | 66.22 |
| clean | mmrel_adv | 70.83 | 89 | 23 | 88.27 | 68.23 |

## Paired tests（same frozen items）

### rbench

| Contrast | ΔAcc | 95% CI | McNemar p | n |
|---|---:|---|---:|---:|
| clean_vs_old | -0.31 pp | [-1.04, +0.47] | 0.504 | 1924 |
| old_vs_base | +0.16 pp | [-0.78, +1.09] | 0.8283 | 1924 |
| clean_vs_base | -0.16 pp | [-0.83, +0.52] | 0.7705 | 1924 |
| clean_vs_s3000 | -0.62 pp | [-1.35, +0.16] | 0.1416 | 1924 |
| old_vs_s3000 | -0.31 pp | [-0.73, +0.05] | 0.1814 | 1924 |

### mmrel_adv

| Contrast | ΔAcc | 95% CI | McNemar p | n |
|---|---:|---|---:|---:|
| clean_vs_old | +0.52 pp | [-0.78, +2.08] | 0.7237 | 384 |
| old_vs_base | +1.30 pp | [-0.78, +3.39] | 0.3588 | 384 |
| clean_vs_base | +1.82 pp | [-0.26, +3.91] | 0.1456 | 384 |
| clean_vs_s3000 | -0.26 pp | [-1.82, +1.30] | 1 | 384 |
| old_vs_s3000 | -0.78 pp | [-2.08, +0.26] | 0.3711 | 384 |

## 计划第 15 节判定

**Case B：Text-only shortcut 已降，但 external ≈ Old。** 数据更干净，旧 Negative 不是外部 transfer 的主要瓶颈。后续仍建议使用 Clean-S3000。

Clean − Old ΔAcc (pp): rbench=+0.38, mmrel_adv=+0.52, amber_dr=-1.80；mean=-0.30
FP/FN Δ (Clean−Old): rbench FP=-18 FN=+24, mmrel_adv FP=-2 FN=+0
Train negative Acc: old=93.14  clean=83.49

