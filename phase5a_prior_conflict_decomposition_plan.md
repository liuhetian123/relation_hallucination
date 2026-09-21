# Phase 5A 实验计划：语言先验冲突分解（Prior-Conflict Decomposition）

## 0. 背景与本阶段唯一要回答的问题

项目的机制假设：MLLM 回答关系问题时依赖语言先验 P(relation | subject, object)，而非视觉证据 P(relation | image)。此前所有阶段都在间接绕这个假设（匿名化、clean negative、多格式），但**从未直接测量过先验本身**。

已有铺垫：

- Phase 4B：微调增益集中在训练词表覆盖（seen）的 MMRel 题目；R-Bench 全平。
- Phase 4C：多格式训练 mf 在 MMRel 上取得项目首个显著外部增益（mf_vs_base +5.47 pp，p=0.016，Yes ratio 向 gold 校准）。
- 人工 QC：clean negative 干净率仅 50%（19% 标签错误 + 31% 模糊），RelSim 作为主训练源被否定。

本阶段把"先验"操作化为**模型在不给图片时对同一道题的回答**，用它把外部 benchmark 切成两个子集：

```text
prior-consistent：text-only 回答 == gold（不看图就能答对）
prior-conflict： text-only 回答 != gold（必须看图才能答对）
```

回答一个问题：

> **Base 模型的关系幻觉是否集中在 prior-conflict 子集？Phase 4C 的 mf 增益是否也集中在该子集？**

若成立，"先验冲突子集 Acc"将成为后续所有实验（Phase 5B：PSG 数据重建 + 先验分歧采样训练）的主评测轴，替代对干预不敏感的整体 Acc。

本阶段**不训练**。唯一的新推理是 text-only（无图），成本极低。

---

## 1. Text-only 先验测量

### 1.1 推理脚本

新建 `scripts/phase5a/infer_text_only.py`（可参考 `scripts/qwen_eval/infer_vqa.py`，去掉图片输入，纯文本 chat）：

- 模型：Qwen2.5-VL-3B-Instruct（本地缓存，`HF_HUB_OFFLINE=1`），环境 `/data/storage22t/lht/envs/qwen25vl/bin/python`；
- 输入：官方题面原文（去掉 `<image>`，其余一字不改）；
- 解码：`do_sample=False`，`max_new_tokens=16`，与历次 fast-eval 一致；
- 支持 `--adapter_path`（1.3 节次要分析用）。

### 1.2 主测量：base 的先验

对以下题集跑 base text-only：

| 题集 | 题面文件 | n |
|---|---|---:|
| MMRel-Adv-fast（主） | `eval_results/qwen/phase2b/fast_subsets/mmrel_adv_questions.jsonl` | 384 |
| R-Bench-fast | `eval_results/qwen/phase2b/fast_subsets/rbench_questions.jsonl` | 1924 |
| AMBER-dr-fast | `eval_results/qwen/phase2b/fast_subsets/amber_dr_questions.jsonl` | 全量 |
| 冻结 393 held-out | `data/phase2/heldout_verification.jsonl` | 393 |

判读规则与对应 benchmark 的官方/既有规则一致（`scripts/phase2b/score_fast.py` 的判读逻辑）。

**有效性检查（做任何分层前必须通过）**：

1. 无效/拒答率（无法判读为 Yes/No，包括"看不到图片"类回答）≤ 10%。若超标，启用 fallback prompt：在题面后追加一行 `There is no image. Answer based on what is typically most plausible. Answer Yes or No.`，主/fallback 两版都报告，分层用 fallback 版；
2. text-only Yes ratio 不得是退化的全 Yes / 全 No（若 > 95% 单边，说明测的是回答偏置而非先验，需改用首 token Yes/No logprob margin 作为先验强度，重新定义 conflict：margin 方向与 gold 相反）。

### 1.3 次要测量：微调是否改变了先验本身

对 MMRel-Adv-fast 额外跑 clean 与 mf 的 text-only（各 384 条）：

- 若微调模型的 text-only 回答与 base 高度一致 → 微调没有改写先验，增益（如有）来自视觉整合，机制故事更干净；
- 若 text-only 大幅漂移 → 增益可能部分来自先验本身被改写，解释时须注明。

---

## 2. 先验冲突分层分析

### 2.1 分层

以 **base 的 text-only 回答**定义每道题的层：`consistent`（= gold）/ `conflict`（≠ gold）/ `invalid`（无法判读，单列不进主分析）。

报告每个 benchmark 的层规模、层内 gold Yes ratio、base text-only Yes ratio。

### 2.2 诊断一：Base 的幻觉在哪里

对每个 benchmark，报告 base **看图**预测（已有冻结答案，禁止重跑）在两层的 Acc：

- 预测：conflict 层 Acc 显著低于 consistent 层。层差就是"先验依赖度"的直接测量；
- 同时报告看图预测与 text-only 预测的一致率（多少题模型看了图答案也不动——先验锁定率）。

### 2.3 诊断二：微调增益集中在哪里

对已有冻结预测做配对分层（复用 Phase 4B 的 bootstrap / McNemar / DiD 代码，`scripts/phase4b/`）：

| 对比 | 预测文件 |
|---|---|
| mf_vs_base（主） | `eval_results/qwen/phase4c/answers/`（mf）；base 同 Phase 4B 来源 |
| clean_vs_base | `eval_results/qwen/phase4a/answers/` |
| s3000_vs_base | `eval_results/qwen/phase3a/answers/` |

每层报告 ΔAcc、95% CI、McNemar p、Yes ratio、FP/FN；DiD = Δ(conflict) − Δ(consistent)。

主假设检验：**mf 在 MMRel 的 +5.47 pp 应集中在 conflict 层**。

### 2.4 交叉与对账

- 对账：合并两层后的整体 Acc 必须与 `phase4c_result.md` / `phase4a_train_result.md` 已发布数字一致（±0.05 pp），不一致先停；
- 可选交叉：conflict 层 × Phase 4B 的 seen/unseen（`eval_results/qwen/phase4b/metrics/relation_extraction.jsonl` 已有每题 relation 与 seen 标记）。若单元格 n < 50 只报描述性数字，不做检验；
- R-Bench 只用 relation-only 子集（Phase 4B 抽取的 verb + prep，n=1482）做分层，整体 R-Bench 不再作为判定依据；
- 393 held-out 同样做 conflict/consistent 分解（base/clean/mf 三模型），检验内部增益的先验结构。

---

## 3. 判定

| Case | 现象 | 结论与下一步 |
|---|---|---|
| **A：框架成立** | base 层差明显（conflict Acc 比 consistent 低 ≥ 15 pp），且 mf_vs_base 在 MMRel conflict 层 ΔAcc 为正、DiD 同号（CI 不含 0，或点估计 ≥ +5 pp 且方向跨对比一致） | 采纳"conflict 子集 Acc"为主评测轴。启动 Phase 5B：PSG 数据重建 + 按先验分歧采样训练样本 |
| **B：先验依赖存在，但增益不集中** | base 层差明显，mf 增益两层均匀或在 consistent 层 | 诊断有效但现有训练没有专门打先验冲突 → Phase 5B 直接按先验分歧采样设计训练数据（这正是缺的东西） |
| **C：先验测不出** | base 层差 < 5 pp，或 text-only 退化（全 Yes 等）且 logprob 版仍无层差 | 生成式先验测量失效。改用 logprob margin 或对比式先验（同题正反两问）再试一轮；两轮都失败才放弃该框架 |

防坑（沿用 Phase 2B.5 教训）：conflict 层的 ΔAcc 必须伴随 Yes ratio 与 FP/FN 检查——conflict 层里 gold 与先验反向，"无脑翻转先验"也能涨分，需确认 consistent 层没有对称的下降。

---

## 4. 产物

| 产物 | 路径 |
|---|---|
| text-only 推理脚本 | `scripts/phase5a/infer_text_only.py` |
| 分层分析脚本 | `scripts/phase5a/analyze_prior.py`（复用 phase4b 统计函数） |
| text-only 答案 | `eval_results/qwen/phase5a/answers/`（不入库） |
| 分层指标 | `eval_results/qwen/phase5a/metrics/phase5a_prior_analysis.json` |
| 每题先验标注 | `eval_results/qwen/phase5a/metrics/prior_labels.jsonl`（question_id, benchmark, prior_answer, gold, layer——Phase 5B 复用） |
| 详细报告 | `eval_results/qwen/phase5a/phase5a_report.md` |
| 根目录摘要 | `phase5a_result.md`（含 Case 判定） |

## 5. 约束

1. 不训练；不对任何模型重跑看图推理，冻结答案原样复用；
2. text-only 推理量：384×3（MMRel 三模型）+ 1924 + AMBER + 393 ≈ 4500 条纯文本生成，单卡 1 小时内；GPU 现场用 `nvidia-smi` 选空闲卡；
3. NumPy 保持 1.26.4；不动 `relhallu` 环境；
4. 所有中间检查（无效率、Yes ratio、对账）写进报告，不通过就停。

## 6. 本阶段最终只回答一个问题

> **"必须看图才能答对"的先验冲突子集，是否既是 base 幻觉的集中地，也是 mf 增益的集中地？**

Case A → Phase 5B（PSG 重建 + 先验分歧采样）以此为主轴；Case B → Phase 5B 把先验分歧直接设计进训练数据；Case C → 换先验测量方式重试一轮后再定去留。
