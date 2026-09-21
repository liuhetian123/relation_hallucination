# Phase 4B 实验计划：外部平坦的归因诊断（OOD 覆盖度 vs 能力不足）

## 0. 背景与本阶段唯一要回答的问题

Phase 3A / 4A 的一致现象：

- 内部冻结 393 held-out verification：Base 58.3 → s3000 72.8 → clean 67.2，提升明显；
- 外部冻结 fast subsets（R-Bench / MMRel-Adv / AMBER-dr）：所有微调模型相对 Base 的 ΔAcc 均在噪声内（配对 McNemar 全部 p > 0.09）。

Phase 2B.5 已排除"纯 prompt mismatch"解释（prompt matching 只诱导 No-bias）。剩下两个候选解释：

> **H1（覆盖度 / OOD）**：模型学到的 relation 判别能力是真的，但训练 relation / 图片领域覆盖不到外部 benchmark 的分布，增益只出现在重叠部分。
>
> **H2（能力 / 量级）**：3B + LoRA + 2~3k 样本的干预量级不足，或学到的是绑定训练分布的判别规则，重叠部分也没有增益。

本阶段**不训练模型、不跑新的模型推理（除任务 B 的 InternVL 复验外）**，只做已有冻结预测的分层分析 + 数据质量 QC，回答：

> **微调模型相对 Base 的增益，是否集中在"训练 relation 词表覆盖到的"外部题目上？**

结果直接决定下一步走"扩数据多样性"（H1）还是"换 7B / 加训练 exposure"（H2），之后才进入 Phase 4C（Typed + Consistency 正式实验）。

---

## 1. 任务 A：Relation 重叠分层分析

### 1.1 对比的模型与冻结预测文件

全部复用已有预测，**禁止重新推理**：

| 模型 | R-Bench-fast | MMRel-Adv-fast |
|---|---|---|
| base | 从全量答案切子集：`eval_results/qwen/rbench/answers/qwen25vl3b_base_image-level.jsonl`（切分逻辑复用 Phase 2B 的 from_full 方法，见 `eval_results/qwen/phase2b/metrics/base_rbench_fast_from_full.json` 的生成脚本，位于 `scripts/phase2b/`） | `eval_results/qwen/mmrel/answers/qwen25vl3b_base_mmrel_adv.jsonl` 切子集 |
| s3000 | `eval_results/qwen/phase3a/answers/s3000_rbench_fast.jsonl` | `eval_results/qwen/phase3a/answers/s3000_mmrel_adv_fast.jsonl` |
| old | `eval_results/qwen/phase4a/answers/old_rbench_fast.jsonl` | `eval_results/qwen/phase4a/answers/old_mmrel_adv_fast.jsonl` |
| clean | `eval_results/qwen/phase4a/answers/clean_rbench_fast.jsonl` | `eval_results/qwen/phase4a/answers/clean_mmrel_adv_fast.jsonl` |

答案判读规则与 Phase 3A/4A 分析脚本保持一致（复用 `scripts/phase3a/` 或 `scripts/phase4a/` 中已有的 official 判读逻辑），保证每个模型在每道 unique question 上得到与已发布 paired tests 相同的 correct/incorrect 判定。**自检：重算出的整体 Acc 必须与 `phase4a_train_result.md` / `phase3a_result.md` 的 Confusion 表一致（±0.05 pp 内），不一致先停下排查。**

AMBER-dr-fast 可选：若 relation 抽取（1.2 节）在 AMBER 题面上可靠，则同样纳入；否则跳过并在报告中说明。

### 1.2 外部题目的 relation 抽取

不重新发明转写。直接复用 Phase 2B.5 的确定性转写产物：

- `data/phase2b5/rbench_fast_matched_prompt.jsonl`（字段 `matched_statement`、`conversion_tag`，1924 条全部转写成功）
- `data/phase2b5/mmrel_adv_fast_matched_prompt.jsonl`（384 条，`conversion_tag` 为 `mmrel_action` / 空间模板，正则本身能给出 relation 短语）

从 `matched_statement` 中抽取 relation 短语：

- MMRel：官方模板正则直接捕获 REL（`Is there X REL to Y` / `Does X VERB Y`），预期成功率 ~100%；
- R-Bench：用 spaCy（与 Phase 2B.5 相同环境）在 statement 上取主谓结构的 predicate（动词 + 介词短语），失败样本落入 `unparsed` 桶单独统计，**不强行猜**。unparsed 比例若超过 15%，需在报告中给出 20 条抽样供人工检查抽取质量。

### 1.3 训练 relation 词表与匹配层级

两套训练词表分别构建（因为 old/s3000 与 clean 的 negative 词表不同）：

- **s3000 / old 词表**：`data/phase4a/train_old.json` 中出现的全部 positive + negative relation（等价于 Phase 3A master pool 词表）；
- **clean 词表**：`data/phase4a/clean_sro3000.jsonl` 的 `positive_relation` ∪ `negative_relation`（359 + positive 词表）。

外部题 relation 与训练词表的匹配分三个层级，逐级报告：

1. **exact**：小写、去冠词后字符串相等；
2. **lemma**：spaCy lemmatize 后相等（`wears` → `wear`，`holding` → `hold`）;
3. **synonym-family**：复用 `scripts/phase4a/` 中已有的 relation synonym normalization 规则（`next to ↔ beside` 等），归一化到同族后相等。

主分析用 **lemma** 层级定义 seen/unseen；exact 与 synonym-family 作为敏感性分析同表报告。

### 1.4 分层统计

对每个 benchmark、每个模型对比（s3000 vs base、old vs base、clean vs base、clean vs old），在 unique questions 上：

- 按 **seen / unseen relation** 分两层，各层报告：n、各模型 Acc、ΔAcc、bootstrap 95% CI（10000 次重采样）、McNemar p；
- **交互作用（difference-in-differences）**：Δ(seen) − Δ(unseen)，bootstrap 95% CI。这是主判据；
- 第二维度分层：MMRel 按图片领域（`vg/` 真实照片 vs `dalle/` 合成图，从 image 路径前缀判断）重复上述分析；R-Bench 无此维度；
- 报告每层的 Yes ratio 与 FP/FN，防止"增益"其实是 No-bias 在某层的副作用（Phase 2B.5 教训）。

注意功效：MMRel-fast 只有 384 unique 题，分层后每层可能不足 200，**所有结论必须带 CI，不允许只报点估计**。若 seen 层不足 100 题，在报告中明确标注功效不足。

### 1.5 判定标准

| Case | 现象 | 结论与下一步 |
|---|---|---|
| **A（支持 H1）** | seen 层 ΔAcc 明显为正（CI 不含 0，或 ≥ +2 pp 且 DiD 同号），unseen 层 ≈ 0 | 覆盖度问题。下一步做数据多样性扩展（新图源 + relation 词表对齐外部分布），再进 Phase 4C |
| **B（支持 H2）** | 两层都平坦（ΔAcc 与 DiD 的 CI 均含 0 且点估计 < 1 pp） | 量级/能力问题。下一步优先 Qwen2.5-VL-7B 或加训练 exposure（2 epoch），不再叠数据技巧 |
| **C（异常）** | unseen 层增益反而大于 seen 层 | 优先怀疑 relation 抽取或 synonym 匹配有 bug，抽 30 条人工核对后重跑；若确认无 bug，说明增益与 relation 词汇无关（可能是格式/先验效应），按 H2 处理 |

---

## 2. 任务 B：Clean Negative 标签噪声估计（100 条 QC）

目的：Clean 训练自检 negative Acc 只有 83.5%，需要区分"更 hard"与"标签错误（candidate relation 在图中其实成立）"。

### 2.1 抽样

从 `data/phase4a/qc/review_300.tsv` 中按原分层（spatial / contact / other 各取前 1/3，共 100 条，seed=42）抽取。图片已在 `data/phase4a/qc/images/`。

### 2.2 自动复验（InternVL visual falsity 二次 pass）

对这 100 条跑 Phase 4A 计划 9.2 节的 visual falsity prompt（`<image> + subject + object + candidate relation → Yes / No / Uncertain`），用 InternVL3.5-8B，**独立于原生成 pass**（新对话、不带原 candidates 上下文）。推理脚本复用 `scripts/phase4a/` 中的 InternVL 调用封装；环境与 Phase 4A-Data 相同。

- 二次 pass 判 `Yes`（即 relation 其实成立）或 `Uncertain` 的比例 = 自动估计的标签噪声上限。

### 2.3 人工页

生成 `data/phase4a/qc/review_100_falsity.html`：每条展示图片 + positive statement + negative statement，人工只填一列——**negative relation 在图中是否成立**（`成立 / 不成立 / 无法判断`）。同时输出 `review_100_falsity.tsv` 供填写。

人工填写不阻塞任务 A 的结论；执行 agent 只负责准备页面与自动复验，报告中把自动复验结果作为主数字、人工列留空待填。

### 2.4 判定

- 自动复验噪声率 ≤ 5%：clean 数据质量可信，83.5% 的 train neg Acc 主要反映难度；
- 5%–15%：可用，但 Phase 4C 报告中必须引用该噪声率作为解释边界；
- \> 15%：Phase 4C 前先加一轮全量 visual falsity filter 重新过滤 clean_sro3000。

---

## 3. 产物与目录

| 产物 | 路径 |
|---|---|
| 分析脚本 | `scripts/phase4b/` |
| 分层指标（JSON） | `eval_results/qwen/phase4b/metrics/phase4b_stratified_analysis.json` |
| relation 抽取结果 + unparsed 清单 | `eval_results/qwen/phase4b/metrics/relation_extraction.jsonl` |
| QC 自动复验结果 | `eval_results/qwen/phase4b/metrics/falsity_recheck_100.json` |
| 人工 QC 页 | `data/phase4a/qc/review_100_falsity.{html,tsv}` |
| 详细报告 | `eval_results/qwen/phase4b/phase4b_report.md` |
| 根目录结果摘要 | `phase4b_result.md`（含 Case A/B/C 判定与下一步建议） |

`eval_results/**/answers/` 惯例不入库，`metrics/`、`logs/` 入库。

---

## 4. 约束与已知坑

1. **不训练、不对四个对比模型做任何新推理**；只允许任务 B 的 InternVL 复验推理（100 张图，量很小）。
2. 环境：分析脚本用 `qwen25vl` 环境（有 spaCy 的环境以 Phase 2B.5 脚本实际使用者为准，先 `python -c "import spacy"` 确认）；不要动 `relhallu` 环境。NumPy 保持 1.26.4。
3. base 的 fast-subset 切分必须与 Phase 2B/3A 使用的完全相同（同一批 question_id），否则 paired 统计无效。
4. R-Bench fast 评测涉及五折平均，但本阶段 paired 分析在 unique questions 上做（与 `phase4a_train_result.md` 的 Paired tests 一致，n=1924），不要混用两种口径。
5. 所有中间判定（relation 抽取成功率、seen/unseen 划分比例、与已发布指标的对账）写进报告；对不上就停，不要带着不一致继续算。

---

## 5. 本阶段最终只回答一个问题

> **微调带来的关系判别增益，在外部 benchmark 上是否集中于训练 relation 词表覆盖到的题目？（顺带：clean negative 的视觉假性标签噪声有多大？）**

Case A → 扩数据多样性；Case B → 上 7B / 加 exposure；两种情况都在归因明确后才进入 Phase 4C（Typed + Consistency）。
