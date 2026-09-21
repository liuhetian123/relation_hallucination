# Phase 2B Balanced Negative Verification + Frozen Fast-Eval

对应计划：`phase2b_balanced_negative_fast_eval_plan.md`。

本阶段只回答三件事：

1. Phase 2A 的 Positive/Negative 数量不平衡，是否限制了 relation rejection？
2. 在总数据仍约 1k 时，1:1 paired + 更充分训练，能否把 No 类学扎实？
3. 自建 rejection 明显改善后，外部 relation-hallucination benchmark 是否出现比 2A 更清晰的 transfer？

未做：Typed vs Explicit、Hard 新方法、7B、5k、DPO、Fully Anonymous、V2B-3ep、完整外部 benchmark。

---

## 1. 设置

| 项 | 取值 |
|---|---|
| 基座 | Qwen2.5-VL-3B-Instruct |
| B0 | 不训练（复用 Phase 2A） |
| B1 | Phase 1 Explicit-1k caption LoRA（不重训） |
| V2-old | Phase 2A：955 Yes + 534 Random No，1 epoch |
| V2B-1ep | 534 Yes + 534 Random No；同一 2-epoch 训练的 epoch-1 快照（`checkpoint-34`） |
| V2B-2ep | 同一 run 训满 2 epoch |
| LoRA | r=16, α=32；vision 冻结；与 2A 相同 |
| 训练 | bs=2，accum=16，lr=2e-5，seed=42，2 epoch cosine（68 step，warmup=1） |
| Prompt | 完全复用 Phase 2A verification 模板；target 仅为 `Yes` / `No` |
| 负例 | **原样复用** V2 的 534 条 random negative，不重新采样 |

权重：`checkpoints/qwen25vl3b_phase2_v2b_{1ep,2ep}/`。

整体 `train_loss`（2 epoch 平均）**0.284**。Epoch 1 末附近 logged loss ≈ 0.203；epoch 2 末附近 ≈ 0.215。

V2B-1ep 不是另一次独立 1-epoch 训练，而是 2-epoch cosine 中途存盘。这样 Q2（多训一个 epoch 有没有用）是同一条优化轨迹上的对比。

---

## 2. 数据

Relation span 可安全替换的训练图：**534 / 955**。421 张 ineligible caption 不进入本轮 Positive，也不强行造负例。

| 项 | 值 |
|---|---:|
| Eligible images | 534 |
| Positive | 534（V2 中与负例同图的 Yes） |
| Random Negative | 534（**与 V2 完全同一批**） |
| 总计 | **1068** |
| 图分布 | 一张图一正一负 |

ID 列表：`data/phase2b/eligible_image_ids.json`。  
训练 JSON：`data/phase2b/v2b_balanced.json`。

相对 V2-old 的唯一训练变量：Positive 955 → 534。图片、负例、关系词表、生成策略都不变。

Held-out verification **不重新生成**：仍是冻结的 193 Yes + 100 random No + 100 hard No = 393。

---

## 3. 训练后自检（1068 条 balanced train）

V2-old 的数字是用已有 `v2_train` 回答，在同一 1068 条 ID 上重打分（负例集合相同，正例从 955 收到 534）。

| 模型 | Acc | Yes | Pos Acc | Neg Acc |
|---|---:|---:|---:|---:|
| V2-old（同 1068） | 63.2 | 69.9 | 83.1 | **43.3** |
| V2B-1ep | 68.4 | 40.2 | 58.6 | **78.3** |
| V2B-2ep | 67.8 | 56.4 | 74.2 | **61.4** |

Balance 本身就把 train negative acc 从 43% 拉到 61–78%。1 epoch 明显偏向 No（Yes 只有 40%）；第 2 epoch 把 Positive Acc 从 59% 拉回 74%，Negative Acc 回落到 61%，不再是“全答 No”。

没有跑 V2B-3ep：2 epoch 的 held-out **rejection 没有继续变好**（见下一节），不满足计划里的启动条件。

---

## 4. Gate 1：冻结 Held-out Verification

Yes 为正类。Gold：193 Yes / 200 No。

| 模型 | Acc | Pos Acc | Random Acc | Hard Acc | Hard FP | Yes |
|---|---:|---:|---:|---:|---:|---:|
| B0 Base | 58.3 | 87.6 | 21.0 | 39.0 | 61.0 | 78.6 |
| V2-old | 64.6 | 80.3 | 43.0 | 56.0 | 44.0 | 65.1 |
| V2B-1ep | 68.4 | 53.4 | **78.0** | **88.0** | **12.0** | 34.9 |
| **V2B-2ep** | **70.2** | 73.1 | 60.0 | 75.0 | 25.0 | 52.4 |

相对 V2-old：

- **V2B-1ep**：Acc +3.8，Hard FP 44→12，Random Acc 43→78；但 Positive Acc 80→53，Yes 掉到 35%。rejection 很强，校准崩了。
- **V2B-2ep**：Acc +5.6，Hard FP 44→25，Random Acc 43→60；Positive Acc 只掉 7 个百分点（80→73），Yes 52% 接近 gold 49%。

Gate 1：**通过**，主 checkpoint 是 V2B-2ep。1ep 也优于 V2-old 的 overall Acc / Hard FP，但属于 No-bias overshoot，不单独作为“成功的 1:1 1-epoch 模型”。

---

## 5. Frozen Fast-Eval Subsets

seed=**42**，生成后冻结。不因模型表现改 seed / 换题。

| Bench | 规则 | 规模 |
|---|---|---|
| R-Bench image-level | 5 折 union 上按 Yes/No × qtype 分层抽 25%，再与各折求交 | unique 1924；每折 1346–1400（均值 1370） |
| MMRel-Adversarial | 去掉缺图 SPEC 179 后，按 label × VG/DALL-E 抽 50% | 384 / 770 |
| AMBER-dr | 按 Yes/No × type 抽 30% | 499 / 1664 |

路径：`eval_results/qwen/phase2b/fast_subsets/`。

### 5.1 旧模型 Full vs Subset QC（不重新推理）

| Bench | 模型 | Full Acc | Subset Acc | Δ |
|---|---|---:|---:|---:|
| R-Bench | Base | 81.14 | 81.54 | +0.40 |
| R-Bench | V2-old | 81.22 | 81.25 | +0.03 |
| R-Bench | V3 | 81.33 | 81.42 | +0.09 |
| MMRel-Adv | Base | 69.09 | 69.01 | −0.08 |
| MMRel-Adv | V2-old | 69.35 | 69.01 | −0.34 |
| AMBER-dr | Base | 81.0 | 79.6 | −1.4 |
| AMBER-dr | V2-old | 80.6 | 79.2 | −1.4 |

水平一致：R-Bench / MMRel 差在 0.5 个百分点以内；AMBER 子集略难约 1.4。

方向一致：V1 仍是更强 Yes-bias（R-Bench Yes 59.1 vs Base 56.5；MMRel Acc 最低）。

排序噪声：R-Bench 上 V2/V3 与 Base 的真实差距只有 0.08–0.19，subset 出现 0.3 个百分点翻转（Base 略高于 V2）。这不是 `V2 << Base`，不足以触发“提高 sampling ratio”。**subset 分辨不了 <0.5 Acc 的差异**，只用于 screening。

---

## 6. Gate 2：Fast External Benchmark

V2-old 的 subset 分从已有 full 预测重打；V2B 在 subset 上新推理。

### R-Bench-fast（五折平均）

| 模型 | Acc | P | R | F1 | Yes | FP mean |
|---|---:|---:|---:|---:|---:|---:|
| Base | 81.54 | 78.24 | 87.76 | 82.72 | 56.49 | 168.4 |
| V2-old | 81.25 | 78.13 | 87.18 | 82.40 | 56.20 | 168.4 |
| V2B-1ep | 81.47 | 78.81 | 86.45 | 82.45 | 55.25 | **160.4** |
| V2B-2ep | 81.25 | 78.35 | 86.74 | 82.33 | 55.76 | 165.4 |

### MMRel-Adv-fast

| 模型 | Acc | P | R | F1 | Yes | FP |
|---|---:|---:|---:|---:|---:|---:|
| Base | 69.01 | 63.80 | 90.82 | 74.95 | 72.66 | 101 |
| V2-old | 69.01 | 64.10 | 89.29 | 74.63 | 71.09 | 98 |
| V2B-1ep | **69.27** | 64.44 | 88.78 | 74.68 | 70.31 | **96** |
| V2B-2ep | **69.27** | 64.34 | 89.29 | 74.79 | 70.83 | 97 |

### AMBER-dr-fast（官方：No 为正类）

| 模型 | Acc | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Base | 79.56 | 69.09 | 91.79 | 78.84 |
| V2-old | 79.16 | 68.33 | 92.75 | 78.69 |
| V2B-1ep | 78.36 | 67.13 | **93.72** | 78.23 |
| V2B-2ep | 78.96 | 67.96 | 93.24 | 78.62 |

外部 fast：**≈ Base**。R-Bench Acc 变化 ≤0.3；MMRel Acc +0.26、FP −4～−5，与 Phase 2A full 的量级相同；AMBER Acc 略降、No-Recall 略升，是更多答 No 的权衡，不是 transfer。

按计划 **不跑 full R-Bench / MMRel / AMBER / POPE**。

---

## 7. 三个问题的答案

### Q1. Imbalance 是否限制了 rejection？

**在自建 verification 上，是。** 只把 Positive 从 955 收到与 Negative 配对的 534，train/held-out 的 No acc 都大幅上升，Hard FP 明显下降。这是一次干净的 class-balance ablation（负例未重采样）。

### Q2. 1:1 + 更多 epoch 能否把 No 学扎实？

**部分能，但不该用 1 epoch 停在 No-bias。** 1 epoch 把 rejection 推过头（held-out Yes 35%，Pos Acc 53%）。2 epoch 校准回来，Neg Acc 仍是 61% 对 Pos 74%，No 类好于 2A 的 43%，但谈不上学死。再加第 3 epoch 没有依据：held-out rejection 已经从 1ep 回落。

### Q3. 外部是否开始有更清晰的 transfer？

**没有。** 自建 Acc 从 64.6 到 70.2，外部 fast 仍在 Base 噪声带里。瓶颈不是 1k 上的 class balance / epoch，而是 **training distribution diversity / cross-benchmark transfer**。

---

## 8. 判定

```text
No-Go A:
held-out verification ↑
但是
fast external benchmark ≈ Base
```

Go A 只在自建测试上成立（balance 确实限制了 2A 的 No 学习）。  
Go B 不成立：多一个 epoch 改善的是校准，不是继续加强 rejection，更不是外部 transfer。

按计划，下一步应进入 **3k paired 1:1 扩数据**（明确 subject / relation / object，而不是修补那 421 条 ineligible caption），而不是继续调 1k epoch，也不回到 Typed vs Explicit。

---

## 9. 产物

| 产物 | 路径 |
|---|---|
| Eligible 534 image IDs | `data/phase2b/eligible_image_ids.json` |
| 1068 balanced train | `data/phase2b/v2b_balanced.json` |
| V2B-1ep / 2ep | `checkpoints/qwen25vl3b_phase2_v2b_{1ep,2ep}/` |
| Held-out / train 指标 | `eval_results/qwen/phase2b/metrics/` |
| Fast subsets + QC | `eval_results/qwen/phase2b/fast_subsets/` |
| 脚本 | `scripts/phase2b/` |

逐样本回答：`eval_results/qwen/phase2b/answers/`。
