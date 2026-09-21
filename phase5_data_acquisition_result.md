# Phase 5 数据获取与 Reefknot 基线结果

## 1. 结论

- Reefknot 官方三份标注、11,084 张实际所需 VG 图像均已就绪；JPEG 全量可解码检查无坏图，缺图率为 0。
- OpenPSG 官方代码和完整 PSG JSON 标注已就绪；COCO 2017 train/val 图像包已从官方地址下载、校验并解压。
- Qwen2.5-VL-3B-Instruct base 已按 Reefknot 官方 YESNO 题面完成 9,740 题基线。
- **重要泄漏修正**：PSG 不是与 VG 无交集的纯 COCO 数据。论文和官方 README 明确说明 PSG 使用 COCO 与 VG 的交集图像。PSG 候选训练图中有 4,796 张与 Reefknot 图像重叠，不能用于未来训练。排除 PSG test 和全部 Reefknot 图像后，安全训练候选为 41,767 张。

## 2. 来源、版本和访问情况

### Reefknot

- 官方代码/标注：https://github.com/JackChen-seu/Reefknot
- 本地提交：`310ec0c468f7763440ca4ffb53f1fe00d9a3deb7`
- 官方论文：https://aclanthology.org/2025.findings-acl.322/
- 官方图像说明：https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
- 最小图像镜像：https://huggingface.co/datasets/MM-Hallu/Reefknot

GitHub 仓库只附三份 JSONL 标注，不附图片。官方 README 要求下载完整 VG `images.zip/images2.zip`，约 15 GB。为避免盲目下载完整 VG，本次采用 MM-Hallu 发布的 Reefknot Hugging Face 镜像；它将三种题型对应图像嵌入 Parquet。按 `image_id` 去重后恰为 11,084 张，与镜像数据卡声明一致。

Reefknot 仓库没有 `LICENSE` 文件，故不能把代码或新增标注视为已有明确开源许可证；论文可引用，但再分发前应向作者确认。Visual Genome 通常标为 CC BY 4.0，使用时仍应保留归属和许可证说明，不应未经核实重新分发底层图片。

### OpenPSG / PSG

- 官方代码与数据入口：https://github.com/Jingkang50/OpenPSG
- 本地提交：`34b2a892f7441966265e3d60ad01ee8eeae89041`
- 官方完整数据入口：OpenPSG README 中的 NTU SharePoint 链接
- JSON 备用镜像：https://huggingface.co/datasets/HarborYuan/OpenPSG
- COCO train：http://images.cocodataset.org/zips/train2017.zip
- COCO val：http://images.cocodataset.org/zips/val2017.zip

官方 SharePoint 页面可访问，但自动直链返回页面/重定向标记，不能可靠脚本化下载。因此 PSG JSON 使用完整镜像下载，并以文件哈希固定。OpenPSG **代码**是 MIT；这不自动覆盖 PSG 标注和底图。COCO 标注通常按 CC BY 4.0 发布，图片分别受原 Flickr 图片许可证约束。

## 3. 实际路径与大小

- Reefknot 官方仓库：`/data/storage22t/lht/datasets/Reefknot`，8.8 MB
- Reefknot Parquet 镜像：`/data/storage22t/lht/datasets/Reefknot_hf`，1.5 GB
- Reefknot 去重图像：`/data/storage22t/lht/datasets/Reefknot_images`，1.1 GB，11,084 张
- OpenPSG 官方仓库：`/data/storage22t/lht/datasets/OpenPSG`，14 MB
- PSG JSON：`/data/storage22t/lht/datasets/OpenPSG_annotations`，249 MB
- COCO 2017：`/data/storage22t/lht/datasets/COCO2017`，56 GB
  - `train2017/`：118,287 张，19 GB
  - `val2017/`：5,000 张，788 MB
  - `archives/train2017.zip`：19,336,861,798 bytes
  - `archives/val2017.zip`：815,585,330 bytes
  - 另有 `train2017.zip.wget-partial`；网络回退进程最终也完成，MD5 与正式包相同。遵守“不删除任何现有数据”要求，未删除这份约 19 GB 的冗余副本。

COCO 校验：

- train MD5 `cced6f7f71b7629ddf16f17bbcfab6b2`，SHA256 `69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929`
- val MD5 `442b8da7639aecaf257c1dceb8ba8c80`，SHA256 `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05`
- 两个 ZIP 均通过 `unzip -t`。

## 4. Reefknot 标注统计

- `YESNO.jsonl`：9,740 题；4,592 张图；yes/no 各 4,870；perception 4,300，cognitive 5,440。
- `Multichoice.jsonl`：6,950 题；5,695 张图；A/B/C/D 分别 1,735/1,676/1,763/1,776；perception 2,150，cognitive 4,800。
- `VQA.jsonl`：4,870 题；4,639 张图；perception 2,150，cognitive 2,720；4,329 个不同答案字符串。
- 合计 21,560 题、11,084 个不同 `image_id`。同图多题是数据设计，不是重复行错误。
- 官方论文 Table 2 曾把 perception/cognition 数量写反；作者在 GitHub Issue #2 确认发布 JSONL 标签正确。

真实题面示例：

- YESNO：`Is the bus driver on bus in this photo? Please answer yes or no.`，label=`yes`
- YESNO 负问：`Is the bus driver off the bus in this photo? Please answer yes or no.`，label=`no`
- Multichoice：`What is the relation with knife and apple in this photo? A. behind B. onto C. on D. into, please choose.`，label=`D`
- VQA：`What is the relation with bag and ground in this photo? Please answer in the following format:bag is <relation> ground.`，label=`bag is on ground.`

Reefknot 确实适配“关系幻觉”叙事：问题直接围绕实体对关系，并区分 perception/cognitive，且 YESNO 有平衡正负例。但它不是纯粹的局部视觉空间关系集；cognitive 占多数，可能包含常识/语义推理成分。因此报告时应始终同时给整体和 perception/cognitive 分层结果。

## 5. PSG 标注统计与训练建议

`psg.json`：

- 48,749 张图，275,371 条关系，875 张无关系；
- 80 个 thing 类、53 个 stuff 类、56 个 predicate 类；
- 图像路径来源：46,563 个 `train2017`，2,186 个 `val2017`；
- 样本字段：`file_name, height, width, image_id, coco_image_id, pan_seg_file_name, segments_info, relations, annotations`。

官方竞争文件：

- `psg_train_val.json`：46,697 条，即 45,697 train + 1,000 validation GT；
- `psg_val_test.json`：2,177 条；官方说明由 1,000 validation GT 和 1,177 test 无 GT 构成。

未来训练不应使用 PSG validation/test。更严格地，使用 `psg.json` 中不在 `test_image_ids` 的 46,563 张后，还必须排除与 Reefknot 的 4,796 张重叠图，得到 41,767 张泄漏安全候选。安全 ID 已写入：

`data_manifests/psg_train_excluding_reefknot_image_ids.txt`

PSG 只作为训练底料，因为其关系是基于 COCO panoptic mask 的 scene graph 标注，不是专为“模型是否幻觉”设计的问答评测；需要后续从安全 train 候选构造正负关系问题。Reefknot 则保持冻结评测用途。

## 6. Qwen base Reefknot YESNO 基线

模型：本地缓存 `Qwen/Qwen2.5-VL-3B-Instruct`；环境 `/data/storage22t/lht/envs/qwen25vl/bin/python`；贪心解码，`max_new_tokens=16`；官方题面未改写。

10 题 smoke：Acc 70.00%，F1 72.73%，无无效回答。

全量 9,740 题：

- Overall：Acc **67.04%**，F1 **71.55%**，Precision **62.94%**，Recall **82.87%**，Yes ratio **65.83%**；TP/TN/FP/FN = 4036/2494/2376/834。
- Perception（n=4,300）：Acc **64.84%**，F1 **70.43%**，Precision **60.76%**，Recall **83.77%**，Yes ratio **68.93%**。
- Cognitive（n=5,440）：Acc **68.79%**，F1 **72.47%**，Precision **64.82%**，Recall **82.17%**，Yes ratio **63.38%**。
- 无效/空回答：0。

答案位于 `eval_results/qwen/reefknot/answers/`，指标位于 `eval_results/qwen/reefknot/metrics/`，评分日志位于 `eval_results/qwen/reefknot/logs/`。

## 7. 可重跑性与清单

入口：`scripts/phase5_data/prepare_phase5_data.py`

脚本支持：

- clone/update 两个官方仓库；
- Hugging Face 快照断点续传；
- Reefknot 嵌入图像按 `image_id` 去重提取；
- COCO `.part` 断点续传、ZIP CRC 检查和解压；
- JSON/JSONL 条数、字段、分布、SHA256 与缺图检查；
- 自动生成排除 Reefknot 重叠图后的 PSG 训练 ID。

机器可读清单：`data_manifests/phase5_data_manifest.json`。新增 Python 脚本均通过 `py_compile` 和 IDE lint；Reefknot 11,084 张 JPEG 全量通过 PIL `verify()`。

## 8. 阻塞与注意事项

- 数据和 YESNO 基线无阻塞。
- PSG 官方 SharePoint 不能稳定非交互下载；当前 JSON 镜像内容完整且已固定 SHA256，但来源链应在论文发布前人工归档确认。
- 未下载 COCO panoptic mask/annotations；当前目标是后续把 PSG scene graph 转为 VLM 关系问答，RGB 图和 PSG 自带 box/segment metadata 已足够开始设计。如后续必须做像素 mask 可视化，再补官方 panoptic annotations。
- 不应再使用“PSG/COCO 与 Reefknot/VG 天然隔离”的假设；两者真实重叠 4,796 张训练候选图，必须使用已生成的安全 ID 文件。
