请基于当前项目已有代码和数据结构，实现一个“RelSim anonymous caption → reconstructed explicit caption”的数据生成脚本。

先阅读并理解当前仓库中：
1. RelSim-1k 数据的实际存储格式；
2. 图片路径组织方式；
3. 已有的数据读取/保存工具；
4. 当前 Python / transformers 环境。

不要重构整个项目，不要修改原始 RelSim 数据，只在已有基础上增加最小必要代码。

==================================================
一、任务目标
==================================================

我们现在要做一个 controlled experiment。

现有 RelSim 样本类似：

{
  "id": "relsim_xxx",
  "image": "xxx.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nDescribe the underlying relational logic of this image. Focus on abstract relations rather than concrete object names, colors, or surface appearance."
    },
    {
      "from": "gpt",
      "value": "Tiny {Animal} balancing on a person's {Body Part}."
    }
  ]
}

我们要利用本地已经下载好的：

InternVL3.5-8B

根据“图片内容 + anonymous caption”，识别每个 placeholder 对应的具体可见实体。

例如：

anonymous:
Tiny {Animal} balancing on a person's {Body Part}.

模型只需要判断：

{
  "{Animal}": "bird",
  "{Body Part}": "finger"
}

然后由 Python 程序机械替换，得到：

Tiny bird balancing on a person's finger.

注意：

我们的实验目的是比较：

Typed Anonymous:
Tiny {Animal} balancing on a person's {Body Part}.

vs

Explicit:
Tiny bird balancing on a person's finger.

因此必须尽可能保证除了 entity identity 之外，relation、syntax、attribute、modifier、word order 全部保持不变。

==================================================
二、非常重要：禁止让 VLM 重写 caption
==================================================

InternVL 不能直接生成新的 caption。

不要让模型输出：

"A small bird is perched on a man's finger."

这种结果不可接受，因为它改变了：
- relation wording
- syntax
- attributes
- sentence length
- description style

InternVL 只能完成：

placeholder → concrete entity

最终 explicit caption 必须由代码使用字符串替换生成。

==================================================
三、placeholder 提取
==================================================

自动从 assistant/gpt caption 中提取所有形如：

{Animal}
{Body Part}
{Object}
{Person}
{Furniture}
...

的 placeholder。

不要提前写死 placeholder 类别。

建议使用通用正则，例如：

\{[^{}]+\}

需要正确处理：
- 一个 placeholder
- 多个 placeholder
- 同一个 placeholder 重复出现
- caption 中没有 placeholder

如果没有 placeholder：
标记为 no_placeholder，
不要调用模型。

==================================================
四、给 InternVL 的固定 Prompt
==================================================

为每个样本构造固定 prompt。

核心内容使用下面的英文模板，不要自由修改任务定义：

You are given an image and an anonymous relational caption.

Your ONLY task is to identify the concrete visible entity corresponding to each placeholder in the caption.

Anonymous caption:
"{ANONYMOUS_CAPTION}"

Placeholders:
{PLACEHOLDER_LIST}

Rules:
1. Do NOT rewrite the caption.
2. Do NOT change, infer, paraphrase, or complete the relation.
3. Do NOT add any new attributes or descriptions.
4. Only identify the concrete visible entity corresponding to each placeholder.
5. Use a simple, common, basic-level English category name.
6. Prefer "bird" instead of "sparrow", "dog" instead of "golden retriever", "car" instead of a specific car model.
7. The replacement must match the semantic type indicated by the placeholder.
8. Use the image as the primary evidence.
9. If a placeholder cannot be identified confidently from the image, return "UNCERTAIN" for that placeholder.
10. Do not guess an entity only because it is statistically likely to participate in the described relation.
11. Return JSON only. Do not provide explanations.

Expected output format:
{
  "replacements": {
    "{Animal}": "bird",
    "{Body Part}": "finger"
  },
  "status": "ok"
}

If any placeholder is uncertain:
{
  "replacements": {
    "{Animal}": "UNCERTAIN",
    "{Body Part}": "finger"
  },
  "status": "uncertain"
}

尤其保留第 10 条：

Do not guess an entity only because it is statistically likely to participate in the described relation.

因为我们后面研究的恰好是 entity-relation shortcut，不希望 teacher 只根据 caption 猜 entity。

==================================================
五、InternVL3.5-8B 推理设置
==================================================

当前服务器 GPU 为 A6000 48GB，但当前最空闲 GPU 大约只有 30GB 可用，而且显存可能被其他任务动态占用。

第一版以稳定为主。

建议默认：

- 单 GPU
- batch_size = 1
- bf16
- do_sample = False
- temperature 不启用随机采样
- max_new_tokens = 64
- 动态图片 tile 数先设置较保守值，例如 max_num = 4
- inference_mode / no_grad
- 不保存 gradient
- 不训练任何参数

如果当前项目/InternVL3.5-HF 的接口与这些参数名称不同，请根据实际模型接口适配，但保持：
“低显存、确定性、短输出”的原则。

模型路径必须做成 CLI 参数，例如：

--model_path /path/to/InternVL3.5-8B

GPU 也必须可以指定，例如：

CUDA_VISIBLE_DEVICES=...

不要在代码里写死 GPU 编号或模型路径。

如果 bf16 在当前实际环境中 OOM：
第一优先级降低 image tile / max_num，
不要直接改成 4bit。

暂时不要主动做量化，先测试正常 bf16。

==================================================
六、处理流程
==================================================

每条样本执行：

RelSim sample
    ↓
读取 image
    ↓
读取原始 anonymous caption
    ↓
提取 placeholders
    ↓
image + anonymous caption + placeholders
    ↓
InternVL3.5-8B
    ↓
保存 raw response
    ↓
解析 JSON
    ↓
检查所有 placeholder 是否都有 replacement
    ↓
检查是否存在 UNCERTAIN
    ↓
如果全部有效
    ↓
Python 机械字符串替换
    ↓
生成 explicit_caption
    ↓
保存新样本

绝对不要覆盖原始数据。

==================================================
七、输出结果需要保留的信息
==================================================

尽量保留原始样本的全部字段，并额外增加 reconstruction 信息。

例如：

{
  "id": "...",
  "image": "...",

  "anonymous_caption":
    "Tiny {Animal} balancing on a person's {Body Part}.",

  "entity_mapping": {
    "{Animal}": "bird",
    "{Body Part}": "finger"
  },

  "explicit_caption":
    "Tiny bird balancing on a person's finger.",

  "reconstruction_status": "ok",

  "teacher_model": "InternVL3.5-8B",

  "teacher_raw_output": "...原始模型输出..."
}

status 至少区分：

- ok
- uncertain
- json_parse_error
- missing_placeholder
- invalid_mapping
- image_error
- inference_error
- no_placeholder

如果项目当前 conversation schema 需要保留，则可以同时生成一个用于后续训练的 explicit conversations 版本，但不要丢掉上述 metadata。

==================================================
八、自动质量检查
==================================================

对 status=ok 的数据至少进行以下自动检查：

1. 所有原 placeholder 均被替换；
2. explicit caption 中不应再存在这些 placeholder；
3. 未出现额外 placeholder；
4. entity replacement 非空；
5. replacement 不是 UNCERTAIN；
6. anonymous caption 除 placeholder 替换部分外，不应该被修改；
7. caption 的 relation wording 必须保持不变；
8. 不允许直接采用 VLM 自己重写的完整 caption。

最重要的一点：

explicit_caption 必须由：

anonymous_caption + entity_mapping

确定性生成。

不能直接使用 teacher 输出的一句话。

==================================================
九、必须支持断点续跑
==================================================

服务器 GPU 是共享的，因此脚本必须支持断点续跑。

要求：

- 已成功处理的 id 再次运行时自动 skip；
- 每处理一条或少量样本就持久化结果，不要等 1000 条全部结束才保存；
- inference error 不应导致整个程序停止；
- 错误样本记录下来继续跑；
- Ctrl+C / GPU 被占用 / 程序意外终止以后可以继续；
- 最终能够统计：
  total
  processed
  ok
  uncertain
  failed
  no_placeholder

最好使用 JSONL 作为中间增量结果，如果当前项目已有更合适的格式则优先复用已有格式。

==================================================
十、先实现 dry-run，不要直接跑 1000 条
==================================================

需要提供类似：

--limit 20
--limit 50

的参数。

第一步先跑 20 条。

确认：
- 模型能正常加载；
- 峰值显存不会逼近 30GB 上限；
- JSON 输出稳定；
- entity 粒度合理。

然后跑 50 条供人工检查。

人工主要检查：

{Animal} → bird
而不是 → sparrow / robin / exotic bird

{Vehicle} → car
而不是 → Tesla Model 3

{Body Part} → finger
如果图里实际 relation argument 确实是 finger。

如果 50 条质量合格，再运行完整 1k。

==================================================
十一、增加一个方便人工审核的输出
==================================================

除了正式数据文件，再生成一个简单的 review 文件，方便快速人工 QC。

每条包含：

id
image path
anonymous_caption
entity_mapping
explicit_caption
status

例如 TSV / CSV / JSONL 都可以。

目标是我可以快速浏览 100~200 条结果，判断实体回填是否正确。

如果方便，可以额外输出：

needs_review = true / false

规则例如：
- UNCERTAIN
- JSON 修复过
- replacement 含多个词且明显细粒度
- mapping 异常

这些全部标记 needs_review=true。

==================================================
十二、需要输出统计摘要
==================================================

脚本完成后输出类似：

Total samples: 1000
With placeholders: 923
No placeholders: 77

OK: 861
Uncertain: 42
Parse error: 8
Inference error: 12

Success rate: 93.3%

Placeholder frequency:
{Animal}: ...
{Object}: ...
{Person}: ...
...

还可以统计：
- 每条平均 placeholder 数量
- replacement 最常见 entity
- UNCERTAIN 最常出现在哪些 placeholder type

方便后续判断 RelSim 数据本身的匿名化情况。

==================================================
十三、当前不要做的事情
==================================================

这一阶段只做数据 reconstruction。

不要：
- 开始 Qwen LoRA 训练；
- 修改 benchmark；
- 构造 Fully Anonymous；
- 构造 bbox / Region A/B；
- 做 counterfactual negatives；
- 用 LLM 重写 caption；
- 自动扩充 RelSim 数据；
- 删除低质量样本原始记录。

这些属于后续实验。

==================================================
十四、验收标准
==================================================

实现完成后，我希望能够执行类似：

python <script>.py \
    --input <relsim_1k_data> \
    --image_root <image_root> \
    --model_path <InternVL3.5-8B-path> \
    --output <output_file> \
    --limit 20

完成第一轮测试。

然后支持去掉 --limit 跑完整 1k。

请先完成代码，并告诉我：

1. 新增/修改了哪些文件；
2. 如何运行 20 条 dry-run；
3. 如何指定 GPU；
4. 如何运行完整 1k；
5. 输出文件是什么；
6. 如何断点续跑；
7. 当前 InternVL 推理使用的 dtype / image tile / generation 参数；
8. 如果出现 OOM，优先调整哪个参数；
9. 给出 2~3 条实际处理结果示例。

不要直接启动完整 1000 条任务。
先让我确认 20 条 dry-run 的质量后再继续。