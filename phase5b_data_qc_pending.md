# Phase 5B 阶段结果：InternVL 验证已完成，等待人工 QC

## 当前结论

异族 InternVL3.5-8B 已对先前选出的 3,500 组候选做完视觉验证（7 号卡，约 3.6 小时，0 失败）。严格三项通过率只有 **6.20%（217/3500）**，因此 **不能按计划凑齐 3,000 / 500**。已按验证结果重筛，未用 Qwen 自验，也未训练。

重建后规模：

- train：212（目标 3,000）
- held-out：5（目标 500；仅 5 个 predicate 有 ≥10 条通过样本可预留）
- 图交集：0
- 泄漏：0

这 217 组都满足 InternVL：`factual=Yes`、`relation-CF=No`、`visual-CF=No`。当前 QC 页只从这批抽取，**可以开始人工填写**。但 212/5 远不够可行性训练与内部机制评测，QC 只用来判断构造质量，不作为开训许可。

## InternVL 判断分布（n=3,500）

| 分支 | Yes | No | Uncertain |
|---|---:|---:|---:|
| factual（应为 Yes） | 3077 (87.9%) | 401 | 22 |
| relation-CF（应为 No） | 2608 (74.5%) | 877 | 15 |
| visual-CF（应为 No） | 2424 (69.3%) | 1063 | 13 |

最常见组合是 `yes/yes/yes`（1,795，51.3%）：InternVL 认为正例成立，同时认为关系反事实和换图反事实也成立。PSG GT 正例大体可信，失败主要在两类负例不够“假”。

按 6.2% 通过率外推，要凑满约 3,500 组通过样本，还需验证大约 5–6 万组（数十小时级）。未验证的 221,528 条仍算 `missing_verification`，没有拿来凑数。

## 人工 QC 操作

打开：

- 页面：`data/phase5b/qc/review_100.html`
- 填写：`data/phase5b/qc/review_100.tsv`

为每行填写五列，值只用 `1`（通过）或 `0`（不通过）：

1. `positive_clear`
2. `relation_cf_clear_negative`
3. `visual_cf_clear_negative`
4. `reference_clear`
5. `grammar_natural`

整组 clean 当且仅当五列全为 1：`clean_rate = 五列全为 1 的行数 / 100`。

本轮 QC 的目的：确认 InternVL 筛出来的 217 组是否真干净。即使 clean rate ≥ 85%，也还不能按原计划开训，因为规模和 held-out 都不够。

## 下一步（需你决定）

1. 先填这 100 条 QC，看 InternVL 通过样本的真实干净率。
2. 若人工也觉得负例普遍不够假：回头改 relation-CF / visual-CF 采样规则，而不是继续盲验 225k。
3. 若人工觉得这 217 组确实干净：再考虑扩大 InternVL 验证覆盖，或把可行性实验缩小到几百组（需同时改 held-out 设计）。

产物：`eval_results/qwen/phase5b/metrics/verify_stats.json`，判断文件 `/data/storage22t/lht/datasets/phase5b_work/verify_judgements.jsonl`。
