# Phase 4C 实验计划：多格式 Prompt 渲染训练（快速对照）

## 0. 背景与本阶段唯一要回答的问题

Phase 4A 确认：clean 数据在内部 held-out 上有效（hard neg 44→57），但外部冻结 benchmark 上与 Base 无显著差异。Phase 2B.5 已证明**评测侧**把官方题改写成训练模板不可行（诱导单向 Yes→No 偏移）。但**训练侧**的 prompt 形式多样化从未测试过。

现有训练数据全部使用单一模板：

```text
<image>
Does the image support the relation expressed in the following statement?
...
Statement:
A car is driving over a buildings.

Answer Yes or No only.
```

而外部 benchmark 是直接疑问句（R-Bench：`Does the man in the image wear a helmet?`；MMRel：`Does an elephant support a seat in the image? Please answer with one word.`）。Phase 2B.5 的 prompt sensitivity 数据显示微调模型在两种形式下一致率仅 65–85%，且分歧几乎全为单向 Yes→No——形式绑定的旁证很强。

本阶段回答：

> **把同一批 clean 训练对渲染成多种提问形式混合训练，能否解锁在官方 prompt 下的外部迁移？**

这与 Phase 4B（relation 内容覆盖度分析）正交：4B 查"内容"，本阶段查"形式"。两者可并行。

---

## 1. 实验设计

### 1.1 对照结构

| Arm | 数据 | 状态 |
|---|---|---|
| base | 不微调 | 已有全部冻结预测 |
| clean | `data/phase4a/train_clean.json`（单一模板，4312 样本） | 已训练，`checkpoints/qwen25vl3b_phase4a_clean` |
| **mf（本阶段新增）** | 同一 2156 对 / 4312 样本，多格式渲染 | 需训练 1 个 LoRA |

唯一变量是 prompt 形式：图片、statement 内容、正负样本、样本总数、LoRA 配置、epoch、seed 全部与 clean 一致。

### 1.2 三种渲染格式

内容层保持 renderer 输出的 `A {subject} is {relation} a {object}` 语序（包括已有的 `a buildings` 类语法瑕疵，**不修**，保证与 clean arm 只差形式不差内容）。三种格式：

**F1 — 现有 statement-verification 模板**（与 `train_clean.json` 完全一致，直接复制）。

**F2 — 直接疑问句（R-Bench 风格）**。从渲染 statement 做确定性变换：`A car is driving over a buildings.` → 

```text
<image>
Is a car driving over a buildings?
Answer Yes or No.
```

变换规则：statement 固定为 `A/An {s} is {rel-phrase} a/an {o}.` 结构，问句 = `Is a/an {s} {rel-phrase} a/an {o}?`。不做动词变形（避免 lemmatization 错误），纯字符串重排。

**F3 — MMRel 风格后缀**。同 F2 问句主体，改为：

```text
<image>
Is a car driving over a buildings in the image? Please answer with one word.
```

Assistant target 三种格式统一为 `Yes` / `No`。

### 1.3 格式分配

- 每个样本（4312 条）确定性分配一种格式：`hash(sample_id) % 3`，seed 无关、可复现；
- 分配后自检：三种格式各约 1/3，且每种格式内 positive/negative 比例保持约 1:1（偏差 > 3 pp 则改用分层轮转分配）；
- **同一对的 pos 和 neg 允许不同格式**（按 id 独立分配），不做强制配对；
- 输出 `data/phase4c/train_mf.json`（llava 格式，字段与 `train_clean.json` 一致，额外加 `prompt_format: F1|F2|F3`）与 `data/phase4c/mf_build_stats.json`。

### 1.4 渲染 QC（训练前必做）

- 随机抽 30 条（seed=42，F2/F3 各 15 条）打印到 `eval_results/qwen/phase4c/mf_render_sample.txt`，报告中附上供人扫一眼；
- 自动检查：问句以 `Is a`/`Is an` 开头、以 `?` 结尾、包含完整 subject/relation/object 子串；失败样本数 > 0 则修 renderer 后重建。

---

## 2. 训练

复制 `scripts/phase4a/train.sh` 的方式，新建 `scripts/phase4c/train.sh`：

- 入口 `scripts/qwen_train/finetune_lora_relsim.py`，参数与 Phase 4A 完全一致（LoRA r=16 α=32、1 epoch、seed=42、bs/accum 沿用脚本默认）；
- `--data_path data/phase4c/train_mf.json`，`--output_dir checkpoints/qwen25vl3b_phase4c_mf`；
- 环境：`/data/storage22t/lht/envs/qwen25vl/bin/python`，模型走本地 HF 缓存 + `HF_HUB_OFFLINE=1`；
- GPU：运行前用 `nvidia-smi` 现场选卡。2026-09-16 16:30 快照：0 号和 4 号卡利用率 0% 但显存被占 ~39GB——**先确认占显存的进程是否可忽略/已死**，不可用就等或选其他空闲卡。显存需求参考 Phase 4A 训练日志。
- 已知坑：不要设 `CUDA_VISIBLE_DEVICES` 与 deepspeed 混用的问题只涉及 LLaVA 训练脚本，本 Qwen 脚本用 `--gpu` 参数即可。

---

## 3. 评估（全部复用 Phase 4A 冻结框架）

新建 `scripts/phase4c/` 下对应脚本（或给 Phase 4A 脚本加输出目录参数），产物写 `eval_results/qwen/phase4c/`：

1. **训练自检**：复用 `scripts/phase4a/eval_train.sh` 逻辑，在 `train_mf.json` 自身上评，报 Acc / Pos Acc / Neg Acc / Yes。**注意**：自检用训练时的混合格式原样评。
2. **冻结 393 held-out**：复用 `eval_heldout.sh`（held-out 题面保持原 F1 模板不变，这是冻结集，不允许改）。
3. **冻结外部 fast subsets（官方 prompt，主判据）**：复用 `eval_fast.sh` 三个 benchmark。
4. **配对统计**：复用 `scripts/phase4a/analyze.py` 的 paired 机制，对比 `mf_vs_clean`、`mf_vs_base`、`clean_vs_base`（后者直接引用 Phase 4A 已发布数字对账）。
5. **格式内分解（附加诊断）**：held-out 之外，把 393 held-out 的同一批 statement 额外渲染成 F2 一份（`data/phase4c/heldout_f2.jsonl`），mf 与 clean 两个模型都评，看 clean 模型在 F2 下是否崩、mf 是否两种格式都稳。这一步是新增推理但很便宜（393×2×2）。

解码与判读全部沿用 Phase 4A（`max_new_tokens=16`，official 判读用 `scripts/phase2b/score_fast.py`）。

---

## 4. 判定

主判据：mf vs clean 在 R-Bench-fast 与 MMRel-Adv-fast 官方 prompt 下的配对 ΔAcc 与 Yes ratio。

| Case | 现象 | 结论 |
|---|---|---|
| **A：形式绑定成立** | mf_vs_clean 在至少一个外部 bench ΔAcc ≥ +1 pp 且 McNemar p < 0.05，Yes ratio 更接近 gold，held-out 不明显退化 | 单一模板确实锁死了能力。后续所有训练（含 Phase 4C 之后的 Typed + Consistency）改用多格式渲染；并与 Phase 4B 的内容覆盖结论合并归因 |
| **B：形式不是瓶颈** | mf ≈ clean（外部 CI 含 0），且第 3.5 节显示 mf 在 F2 held-out 上明显好于 clean | 模型能学会跨格式，但外部仍不迁移 → 瓶颈在内容/量级，强化 Phase 4B 的 H1/H2 归因路线 |
| **C：mf 更差** | mf 外部或 held-out 明显低于 clean | 格式混合稀释了每种格式的曝光（每种只剩 ~1437 条）。记录后不回退结论，考虑"多格式 + 2 epoch"补一枪再判 |

注意防坑（Phase 2B.5 教训）：任何 Acc 提升必须同时检查 Yes ratio 与 FP/FN，排除"靠 No-bias 涨分"的假阳性。

---

## 5. 产物

| 产物 | 路径 |
|---|---|
| 构建脚本 + 渲染器 | `scripts/phase4c/build_mf_train.py` |
| 训练/评估脚本 | `scripts/phase4c/{train.sh,eval_*.sh}` |
| 训练数据 | `data/phase4c/train_mf.json` |
| LoRA 权重 | `checkpoints/qwen25vl3b_phase4c_mf/` |
| 指标与日志 | `eval_results/qwen/phase4c/{metrics,logs}/`（answers 不入库） |
| 详细报告 | `eval_results/qwen/phase4c/phase4c_report.md` |
| 根目录摘要 | `phase4c_result.md`（含 Case 判定） |

## 6. 预算与顺序

总预算 ≈ 1 次 LoRA 训练（Phase 4A 单次约 30–60 min）+ 3 个 fast subset 推理 + held-out×3 份。半天内可完成。执行顺序：build → 渲染 QC → train → 自检对账（clean_vs_base 数字必须与 `phase4a_train_result.md` 一致）→ 评估 → 分析 → 报告。

## 7. 本阶段最终只回答一个问题

> **训练 prompt 形式的单一性，是否是 clean 数据外部迁移失败的原因之一？**
