# Phase 5B 实验计划：视觉反事实关系 Grounding 可行性验证

## 0. 阶段定位

此前实验不再支持“实体匿名化 → 削弱文本先验 → 缓解关系幻觉”作为主线：

- Typed ≈ Explicit；
- RelSim 人工 QC 仅 50% negative 明确不成立；
- Phase 5A 中 MMRel 的 prior-conflict 层差很小，mf 增益也不集中于 conflict；
- Phase 4C 证明多格式训练有用，但 mf 的 text-only 行为相对 base 大幅漂移，不能据此证明视觉 grounding 增强。

因此，本阶段不再把匿名化作为主方法，而检验更直接的机制：

> **保持关系问题文本不变，只改变视觉证据，并显式要求模型的 Yes/No margin 随视觉证据变化。**

暂称方法为 **Visual Counterfactual Grounding（VCG）**。匿名化仅保留为后续消融，不进入本轮最小可行实验。

本阶段是带止损线的 feasibility test，不追求最终论文规模。

---

## 1. 核心研究问题

1. 从泄漏安全的 PSG train 数据能否构造人工可接受的关系正例与视觉反事实负例？
2. 与普通多格式 SFT 相比，VCG 是否让模型更依赖图像证据，而不是仅改变 Yes/No 输出偏置？
3. 机制性改善能否迁移到冻结的 Reefknot YESNO 和 MMRel？

只有“机制指标改善 + 至少一个外部 benchmark 改善”同时成立，才继续扩规模。

---

## 2. 数据边界与泄漏控制

### 2.1 已有资源

- PSG 标注：`/data/storage22t/lht/datasets/OpenPSG_annotations`
- COCO 2017：`/data/storage22t/lht/datasets/COCO2017`
- Reefknot：`/data/storage22t/lht/datasets/Reefknot`
- Reefknot 图像：`/data/storage22t/lht/datasets/Reefknot_images`
- 泄漏安全 PSG train ID：
  `data_manifests/psg_train_excluding_reefknot_image_ids.txt`

所有训练图片必须来自该安全 ID 清单。不得使用 PSG validation/test，也不得使用与 Reefknot 重叠的 4,796 张图片。

### 2.2 数据规模

先构建：

- train：3,000 个正关系实例；
- internal held-out：500 个关系实例，图像与 train 不重叠；
- QC：从 train 候选中分层抽 100 组。

按 PSG 的 56 个 predicate 分层采样，尽量避免高频关系垄断。每类上限为总训练实例的 10%；低频类全部保留后再补高频类。

### 2.3 实体定位

PSG 的 relation 是 `(subject segment index, object segment index, predicate index)`。构建时必须保存：

- subject/object 类别与实例 ID；
- subject/object bounding box；
- predicate；
-原始 image/coco ID；
- PSG 原始 relation 索引。

第一版训练仍使用整图，但内部机制评测需使用 box 生成遮挡图。不要依赖模型自由识别“图中的某个人”。

---

## 3. 三类配对样本

对每个真实三元组 `(I+, s, r+, o)` 构造：

### 3.1 Factual positive

```text
Image: I+
Question: Is the {subject} {relation} the {object}?
Target: Yes
```

### 3.2 Relation counterfactual

图像不变，关系改变：

```text
Image: I+
Question: Is the {subject} {r-} the {object}?
Target: No
```

`r-` 必须满足：

1. 不在该图该 subject/object 实例对的全部 GT relations 中；
2. 不是 `r+` 的 synonym、inverse expression 或蕴含关系；
3. subject/relation/object 在语言上合理；
4. 与 `r+` 尽量属于同一关系族，形成 hard negative；
5. 经独立视觉验证器判为 `No`，`Uncertain` 丢弃。

关系族至少分为 spatial、contact、action/interaction、other。规则表必须保存为可审计 JSON/YAML。

### 3.3 Visual counterfactual

文本保持 factual positive 完全不变，替换为 `I-`：

```text
Image: I-
Question: Is the {subject} {relation} the {object}?
Target: No
```

`I-` 的选择要求：

1. 与 `I+` 不同；
2. 图中同时有相同 subject/object 类别；
3. PSG 标注中不存在该 relation；
4. 优先选择存在同一实体对、但关系为同族其他 predicate 的图；
5. 独立视觉验证器必须判 `No`；`Yes/Uncertain` 丢弃。

注意：PSG 标注“不存在”不等于视觉上一定不成立，所以独立验证与人工 QC 是硬门槛。

每个数据单元保存 `(positive, relation_cf, visual_cf)` 三元组，训练时不可拆散到不同 split。

---

## 4. Prompt 格式

沿用 Phase 4C 已验证的多格式原则，同一三元组确定性分配 F1/F2/F3：

- F1：statement verification；
- F2：直接 Yes/No 疑问句；
- F3：MMRel 风格 one-word 后缀。

渲染器必须按 predicate 模板处理语法，不允许再次产生 `A buildings`、`is supports` 等错误。先为 56 个 PSG predicate 建立模板与 inverse/同义关系表。

assistant target 统一为 `Yes` / `No`。

---

## 5. 人工 QC Gate（训练前硬门槛）

生成 `data/phase5b/qc/review_100.html` 和 TSV，每组同时展示：

- factual positive；
- relation counterfactual；
- visual counterfactual 的两张图；
- subject/object box 可视化；
- 自动验证器判断。

人工分别标注：

1. positive 是否明确成立；
2. relation counterfactual 是否明确不成立；
3. visual counterfactual 是否明确不成立；
4. subject/object 指代是否清楚；
5. 题面语法是否自然。

定义整组 clean：五项全部通过。

- clean rate ≥ 85%：进入训练；
- 70%–85%：修订采样/规则/验证后重新抽 100 条；
- < 70%：停止 Phase 5B，不训练。

禁止再次用生成负例的同一模型做唯一复验。若可用，生成器与验证器使用不同模型族；人工 QC 是最终门槛。

---

## 6. 模型与训练臂

模型统一为 Qwen2.5-VL-3B-Instruct，LoRA 配置沿用 Phase 4C（r=16、alpha=32、vision encoder 冻结、seed=42）。

### 6.1 B0：Base

不训练。Reefknot YESNO 基线已完成：Acc 67.04%。

### 6.2 B1：PSG-MF-SFT

普通多格式监督学习。对每组使用：

- factual positive：Yes；
- relation counterfactual：No；
- visual counterfactual：No。

总训练 exposure 与 B2 完全一致。

### 6.3 B2：PSG-VCG（主方法）

在 B1 的 token-level CE 之外，对配对样本加入 image-conditioned margin loss。

定义 `m(I,q) = log P(Yes|I,q) - log P(No|I,q)`，要求：

```text
m(I+, q+) >= m(I-, q+) + gamma_visual
m(I+, q+) >= m(I+, q-) + gamma_relation
```

总损失：

```text
L = L_CE
  + lambda_visual * max(0, gamma_visual - m(I+,q+) + m(I-,q+))
  + lambda_relation * max(0, gamma_relation - m(I+,q+) + m(I+,q-))
```

首轮固定：

- `lambda_visual = 0.5`
- `lambda_relation = 0.5`
- `gamma_visual = 1.0`
- `gamma_relation = 1.0`

不在本轮扫参。训练实现应以最小侵入方式扩展现有 Qwen trainer，并添加单 batch smoke test，确认：

- 三个分支图像/文本对应正确；
- Yes/No token 定位正确；
- margin loss 非 NaN 且可反传；
- lambda=0 时退化为 B1 等价 CE。

训练 1 epoch；若 B1/B2 的 train Acc 都低于 75%，才允许补 2 epoch，且必须两臂同时补，禁止只补主方法。

---

## 7. 内部机制评测

使用 500 个 image-disjoint held-out 单元，评估：

1. factual positive Acc；
2. relation-CF Acc；
3. visual-CF Acc；
4. triplet all-correct rate；
5. visual pair ordering：
   `m(I+,q+) > m(I-,q+)` 的比例；
6. relation pair ordering：
   `m(I+,q+) > m(I+,q-)` 的比例；
7. visual margin 与 relation margin 的均值/中位数；
8. Yes ratio、FP/FN。

### 7.1 遮挡敏感性

基于 PSG box 生成：

- subject 被遮挡；
- object 被遮挡；
- subject+object 都被遮挡；
- 等面积随机区域遮挡对照。

对 factual positive 测量 Yes margin 下降：

```text
delta_subject = m(original,q) - m(mask_subject,q)
delta_object  = m(original,q) - m(mask_object,q)
delta_random  = m(original,q) - m(mask_random,q)
```

主机制指标：

- VCG 的 visual pair ordering 高于 B1；
- `delta_subject/object` 高于 `delta_random`；
- 不是所有遮挡都统一诱导 No。

---

## 8. 外部冻结评测

### 8.1 Reefknot YESNO（主）

使用完整 9,740 题、官方题面，不修改 prompt。报告：

- Overall Acc/F1/Precision/Recall/Yes ratio；
- perception/cognitive 分层；
- paired bootstrap CI 与 McNemar（B2 vs B1、B2 vs Base）；
- 每图聚类 bootstrap 作为敏感性分析，避免同图多题被当作完全独立。

### 8.2 MMRel

优先使用全量 adversarial 题；若当前仅 fast 可复现，先报告 fast，并明确功效限制。报告同样的 paired 指标与 Yes ratio/FP/FN。

### 8.3 AMBER

只作不退化 guard。其 direct-contact 任务与 PSG 56 类关系并不完全同构。

### 8.4 R-Bench

不再作主 benchmark。可选仅报告 Phase 4B 定义的 relation-only subset 作为附录 guard。

---

## 9. 判定与止损

### Go

必须同时满足：

1. QC clean rate ≥ 85%；
2. B2 vs B1 的 internal visual pair ordering 提升 ≥ 5 pp，且 bootstrap CI 不含 0；
3. subject/object 遮挡的 margin 下降显著大于随机遮挡；
4. Reefknot 或 MMRel 至少一个 benchmark 上 B2 vs B1 ΔAcc ≥ 1 pp，方向一致且无明显 Yes/No 偏置恶化；
5. 另一个外部 benchmark 不显著退化。

满足后再扩到 10k、7B，并加入 Anonymous 作为消融。

### Mechanism-only

内部机制指标明确改善，但外部无提升：

- 不宣称缓解关系幻觉；
- 先检查 relation 覆盖、规模与 grounded-to-ungrounded transfer；
- 只允许再做一次 10k 扩规模验证。

### No-Go

以下任一成立即停止该方向：

- QC 两轮仍 < 85%；
- B2 相对 B1 的视觉机制指标不改善；
- 外部提升仅由 Yes ratio 大幅漂移造成；
- 10k 补实验后仍只有内部改善、外部完全平坦。

---

## 10. 产物

- 数据构建：`scripts/phase5b/build_psg_vcg.py`
- 关系规则：`scripts/phase5b/psg_relation_rules.yaml`
- QC：`data/phase5b/qc/`
- 训练数据：`data/phase5b/train_triplets.jsonl`
- held-out：`data/phase5b/heldout_triplets.jsonl`
- 训练器：`scripts/phase5b/train_vcg.py`
- 训练脚本：`scripts/phase5b/train_{mf_sft,vcg}.sh`
- 评测脚本：`scripts/phase5b/eval_*.sh`
- checkpoints：
  - `checkpoints/qwen25vl3b_phase5b_psg_mf_sft`
  - `checkpoints/qwen25vl3b_phase5b_psg_vcg`
- 结果：`eval_results/qwen/phase5b/`
- 根目录摘要：`phase5b_result.md`

大型图像不得复制进项目。answers 继续不入库，metrics/logs/report 入库。

---

## 11. 执行顺序

1. 审计 PSG 标注与 56 类 predicate；
2. 建关系模板、同义/inverse/蕴含规则；
3. 构建 3k train + 500 held-out 候选；
4. 异族模型自动验证；
5. 生成 100 组人工 QC 页面并暂停；
6. 人工 QC 达标后训练 B1/B2；
7. 先跑内部机制评测；
8. 机制指标通过后才跑 Reefknot/MMRel 全量；
9. 写 paired 统计与 Case 判定。

**执行 agent 必须在第 5 步暂停，等待人工 QC 文件填完；不得绕过 QC 直接训练。**

