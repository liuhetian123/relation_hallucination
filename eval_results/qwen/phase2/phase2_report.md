# Phase 2A Negative Relation Verification

对应计划：`phase2_negative_relation_verification_training_plan.md`。

本阶段只回答三件事：

1. Caption SFT 是否等价于 relation verification？
2. 显式 negative supervision 能否降低 relation false positive？
3. Hard / plausible negative 是否比 random negative 更有效？

未做：Typed、7B、5k、DPO、Fully Anonymous。

---

## 1. 设置

| 项 | 取值 |
|---|---|
| 基座 | Qwen2.5-VL-3B-Instruct |
| B0 | 不训练 |
| B1 | Phase 1 Explicit-1k caption LoRA（不重训） |
| V1 | Positive-only verification（955 Yes） |
| V2 | Positive + random negative（955 Yes + 534 No） |
| V3 | Positive + InternVL-filtered hard negative（955 Yes + 490 No） |
| LoRA | r=16, α=32；vision 冻结；q/k/v/o/gate/up/down |
| 训练 | 1 epoch，bs=2，accum=16，lr=2e-5，seed=42 |
| Prompt | 固定 verification 模板；target 仅为 `Yes` / `No` |
| 图 | 训练 `data/relsim_images`；held-out `/data/lht/relsim_dataset/relsim_images` |

权重：`checkpoints/qwen25vl3b_phase2_{v1,v2,v3}/`。

train_loss：V1 **0.263** / V2 **0.310** / V3 **0.291**（V1 只学 Yes，loss 更低是预期）。

---

## 2. 数据 QC

Relation 只能在 Phase 1.6 已定位的 span 上机械替换（534/955 训练图、193/200 held-out explicit-OK）。

| 项 | 值 |
|---|---:|
| RelSim vocab（freq≥2） | 1754 |
| V1 | 955 Yes |
| V2 | 1489（955 Yes + 534 random No） |
| V3 | 1445（955 Yes + 490 hard No） |
| InternVL train judgments | No 490 / Yes 807 / Uncertain 3 |
| InternVL held-out judgments | No 183 / Yes 246 / Uncertain 2 |
| Held-out test（冻结） | 193 Yes + 100 random No + 100 hard No = **393** |

Hard candidate 只替换 relation phrase，不重写 caption。InternVL3.5-8B 只判 `Yes/No/Uncertain`，**只保留 No**。人工抽查表：`eval_results/qwen/phase2/review_100.tsv`。

---

## 3. 训练后自检（防止全 Yes / 全 No）

| 模型 | n | Acc | Yes | Pos Acc | Neg Acc |
|---|---:|---:|---:|---:|---:|
| V1 | 955 | 96.1 | 96.1 | 96.1 | — |
| V2 | 1489 | 70.0 | 74.8 | 84.9 | 43.3（random） |
| V3 | 1445 | 74.9 | 75.5 | 88.2 | 49.2（hard） |

V1 几乎只会答 Yes。V2/V3 没有塌成全 Yes 或全 No，但对 **No 的拟合仍然偏弱**（训练集 negative acc 只有 43–49%）。1 epoch 够学到 verification 格式，不够把 rejection 学扎实。

---

## 4. Held-out verification（主表）

冻结 Phase 1.5/1.6b 的 200 图；测试 negative 独立采样（seed=123）。Yes 为正类。

| 模型 | Acc | P | R | F1 | Yes | TP | TN | FP | FN | Pos Acc | Random Acc | Hard Acc | Hard FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 Base | 58.3 | 54.7 | 87.6 | 67.3 | 78.6 | 169 | 60 | 140 | 24 | 87.6 | 21.0 | 39.0 | **61.0** |
| B1 Caption-SFT | 58.0 | 54.5 | 87.0 | 67.1 | 78.4 | 168 | 60 | 140 | 25 | 87.0 | 22.0 | 38.0 | 62.0 |
| V1 Pos-only | 51.7 | 50.4 | **94.8** | 65.8 | **92.4** | 183 | 20 | 180 | 10 | **94.8** | 6.0 | 14.0 | 86.0 |
| **V2 Pos+Random** | **64.6** | **60.5** | 80.3 | 69.0 | 65.1 | 155 | **99** | **101** | 38 | 80.3 | **43.0** | **56.0** | **44.0** |
| V3 Pos+Hard | 63.4 | 58.8 | 84.5 | **69.4** | 70.5 | 163 | 86 | 114 | 30 | 84.5 | 38.0 | 48.0 | 52.0 |

Gold：193 Yes / 200 No，故 Acc 随机基线约 50%。

---

## 5. 三个问题的答案（held-out）

### Q1. Caption → Verification 本身？

**否。** B1 ≈ Base。V1 把任务改成 Yes/No 后更差：Yes 升到 92.4%，Hard FP 升到 86%。只见过 Yes 的 verification 会强化 Yes-bias，而不是教模型检查关系。

### Q2. 要不要 explicit negative？

**要。** V2/V3 相对 V1：Acc +12–13 个百分点，TN 从 20 升到 86–99，Hard FP 从 86% 降到 44–52%。相对 Base：Acc +5–6 个百分点，Hard FP 61% → 44–52%。Yes 从 79% 降到 65–70%，**不是**简单全答 No（Gold Yes 约 49%）。

### Q3. Hard > Random？

**本轮不成立。** V2 在 Acc、Random Acc、Hard Acc、Hard FP 上全面略优于 V3。Hard 候选经过 InternVL 过滤，但 1 epoch 对 No 拟合不足；random negative 反而带来更稳的 rejection shift。不要把 V3 解释成 shortcut-aware 机制已经赢了。

---

## 6. 外部 Benchmark

B0 / B1 复用 Phase 1。R-Bench 为官方五折平均，FP 为五折均值。AMBER-dr 的 Precision/Recall 以 **No 为正类**。

### R-Bench image-level

| 模型 | Acc | Precision | Recall | F1 | Yes | FP |
|---|---:|---:|---:|---:|---:|---:|
| B0 Base | 81.14 | 78.24 | 86.80 | 82.30 | 56.02 | 670.4 |
| B1 Caption-SFT | 81.13 | 78.18 | 86.89 | 82.30 | 56.12 | 673.4 |
| V1 Pos-only | 80.65 | 76.53 | **88.97** | 82.28 | **58.70** | 757.4 |
| V2 Pos+Random | 81.22 | **78.52** | 86.45 | 82.30 | 55.59 | **656.4** |
| V3 Pos+Hard | **81.33** | 78.46 | 86.89 | **82.46** | 55.92 | 662.4 |

V2 vs Base：Acc **+0.08**，FP **−14**。V3 Acc **+0.19**，FP **−8**。与 Phase 1 Typed−Explicit 的噪声量级相同。V1 更 Yes-biased（Yes +2.7，FP +87）。

### MMRel-Adversarial

有效 770 题（SPEC 179 缺图，跳过空答）。

| 模型 | Acc | Precision | Recall | F1 | Yes | FP |
|---|---:|---:|---:|---:|---:|---:|
| B0 Base | 69.09 | 63.70 | 91.33 | 75.05 | 72.99 | 204 |
| B1 Caption-SFT | 68.96 | 63.59 | 91.33 | 74.97 | 73.12 | 205 |
| V1 Pos-only | 67.53 | 62.16 | **92.60** | 74.39 | **75.84** | 221 |
| V2 Pos+Random | **69.35** | **64.08** | 90.56 | 75.05 | 71.95 | **199** |
| V3 Pos+Hard | **69.35** | 63.98 | 91.07 | **75.16** | 72.47 | 201 |

V2/V3 vs Base：Acc **+0.26**，FP **−5 / −3**。方向对，但幅度不够构成 Go。

### MMRel Dall-E normal

| 模型 | Acc | Precision | Recall | F1 | Yes | FP |
|---|---:|---:|---:|---:|---:|---:|
| B0 Base | 68.28 | 63.35 | 89.67 | 74.24 | 72.17 | 633 |
| B1 Caption-SFT | 68.24 | 63.33 | 89.59 | 74.20 | 72.13 | 633 |
| V1 Pos-only | 64.94 | 60.20 | **92.13** | 72.82 | **78.02** | 743 |
| V2 Pos+Random | 68.66 | **64.02** | 87.95 | 74.10 | 70.04 | **603** |
| V3 Pos+Hard | **68.83** | 64.01 | 88.77 | **74.38** | 70.71 | 609 |

这是外部里 FP 降幅最大的表：V2 **−30**，V3 **−24**。Acc 仍只 +0.4–0.6。

### AMBER discriminative-relation

| 模型 | Acc | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| B0 Base | 81.0 | 71.1 | 91.1 | 79.9 |
| B1 Caption-SFT | 80.9 | 71.4 | 90.0 | 79.6 |
| V1 Pos-only | **82.9** | **75.0** | 88.1 | **81.0** |
| V2 Pos+Random | 80.6 | 70.3 | 91.9 | 79.7 |
| V3 Pos+Hard | 80.8 | 70.6 | **92.2** | 80.0 |

V1 Acc 最高，但是 No-Recall 从 91.1 降到 88.1，更不会拒绝错误关系。V2/V3 的 No-Recall 略高于 Base（91.9 / 92.2），Acc 基本持平。

### POPE-Adversarial（generic control）

| 模型 | Acc | Precision | Recall | F1 | Yes | FP |
|---|---:|---:|---:|---:|---:|---:|
| B0 Base | 85.33 | 87.48 | 82.47 | 84.90 | 47.13 | 177 |
| B1 Caption-SFT | 85.27 | 87.04 | 82.87 | 84.90 | 47.60 | 185 |
| V1 Pos-only | 85.20 | 86.31 | **83.67** | 84.97 | **48.47** | 199 |
| V2 Pos+Random | **85.77** | **88.29** | 82.47 | **85.28** | 46.70 | **164** |
| V3 Pos+Hard | 85.53 | 87.86 | 82.47 | 85.08 | 46.93 | 171 |

POPE 与 relation bench 同量级小幅移动（V2 Acc +0.44，FP −13），没有出现“只在 relation 上大变、POPE 不动”的分离。

---

## 7. 判定

```text
No-Go A:
Held-out custom verification ↑
but
R-Bench / MMRel-Adv ≈ Base
```

自建 verification 上 negative supervision 有效（V2/V3 > V1 > Base 的 rejection）。外部 hallucination benchmark 只有同方向、噪声量级的变化，不能支持“已经缓解 relation hallucination”。

三个问题：

1. **Caption SFT ≠ verification。** B1≈Base；V1 更 Yes-biased，外部也更差。
2. **显式 negative 能降低自建测试的 FP**，但对 R-Bench / MMRel-Adv 几乎传不过去。
3. **Hard 并不比 Random 更有效。** 外部同样是 V2 ≈ V3，V2 的 FP 还略低。

按计划下一步优先 **扩大 data diversity / 加强 No 拟合（更多 epoch 或更多负例）**，不要立刻回到 Typed vs Explicit，也不上 7B。

逐样本：`eval_results/qwen/phase2/`；训练数据：`data/phase2/`。
