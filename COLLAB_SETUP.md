# relation_hallucination 同机双目录协作

套自 token_compress 的 worktree 方案。约束：同一服务器、同一 Linux 用户 `lht`、同一份数据和 conda，不占额外磁盘。

Git 只管代码和文档。模型、数据集、第三方仓、conda **只留一份**，两边用同一路径读取。

---

## 1. 目标布局

```text
/data/storage22t/lht/works/                    # /home/lht/works 是这里的软链
├── relation_hallucination/                    # 你：分支 main，唯一 .git
└── relation_hallucination-collab/             # 师弟：分支 collab（worktree）

共享资产（不进 Git、不拷贝）：
/data/lht/relsim_dataset/                      # RelSim 图 (~6.6 万张) + LLaVA/CLIP 权重
/data/storage22t/lht/datasets/                 # COCO / PSG / Reefknot / phase5b_work
/data/storage22t/lht/envs/qwen25vl             # Qwen2.5-VL 环境
/home/lht/miniconda3/envs/relhallu             # LLaVA 环境
/home/lht/.cache/huggingface                   # Qwen 基座
/data/storage22t/lht/hf_cache                  # InternVL 基座
```

GitHub：https://github.com/liuhetian123/relation_hallucination （当前公开）

| 工作区 | 分支 | 跟踪 |
|---|---|---|
| `.../relation_hallucination` | `main` | `origin/main` |
| `.../relation_hallucination-collab` | `collab` | `origin/collab` |

远程用 SSH，且必须走 **Host 别名**（22 端口不通；默认 `github.com` 仍绑定旧钥匙 `3166596380`）：

```text
git@github.com-liuhetian123:liuhetian123/relation_hallucination.git
```

路径由 `configs/paths.env` 和 `scripts/utils/paths.py` 集中管理，可用 `RH_*` 覆盖。

---

## 2. Git 身份（已完成，2026-09-21）

旧钥匙 `~/.ssh/id_ed25519` 属于 `3166596380`，GitHub 不允许同一把公钥挂两个账号。本仓使用单独钥匙 `~/.ssh/id_ed25519_liuhetian123`。

`~/.ssh/config`：

```text
Host github.com
  HostName ssh.github.com
  User git
  Port 443
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes

Host github.com-liuhetian123
  HostName ssh.github.com
  User git
  Port 443
  IdentityFile ~/.ssh/id_ed25519_liuhetian123
  IdentitiesOnly yes
```

验证：

```bash
ssh -T git@github.com-liuhetian123
# 应看到 Hi liuhetian123!
```

`origin` 已是 SSH，`main` 已推到 `8f58680`。下一步可以建 collab worktree。

---

## 3. Git 身份通了之后：worktree

gitignore 的大目录（`LLaVA/`、`checkpoints/`、`R-Bench/` 等）**不会**出现在第二个 worktree 里，必须先在主目录做成指向共享位置的软链，或在 collab 目录再链一次。先建 worktree，再补链：

```bash
cd /data/storage22t/lht/works/relation_hallucination
git worktree add -b collab ../relation_hallucination-collab
git push -u origin collab
git worktree list
```

同一分支不能同时在两个 worktree 检出。你固定 `main`，对方固定 `collab`。对方目录里的 `.git` 只是指针。

主目录里需要两边共用、且被 gitignore 的目录：

| 仓内路径 | 约大小 | 处理 |
|---|---|---|
| `LLaVA/` | 13G | collab 目录软链到主目录这份 |
| `checkpoints/` | 7.4G | 同上；师弟新实验写 `checkpoints/collab/` |
| `R-Bench/` | 2.8G | 软链 |
| `MMRel/` | 2.2G | 软链 |
| `AMBER/` | 812M | 软链 |
| `relsim/` | 211M | 软链 |
| `POPE/` | 66M | 软链 |
| `data/relsim_images/` | 仓内仅 1000 张 | 建议改链到 `$RH_RELSIM_IMAGES` |

示例（在 collab 目录执行）：

```bash
cd /data/storage22t/lht/works/relation_hallucination-collab
MAIN=/data/storage22t/lht/works/relation_hallucination
for d in LLaVA checkpoints R-Bench MMRel AMBER relsim POPE; do
  ln -sfn "$MAIN/$d" "$d"
done
mkdir -p data
ln -sfn /data/lht/relsim_dataset/relsim_images data/relsim_images
```

新实验输出不要覆盖对方文件：师弟用 `checkpoints/collab/`、`eval_results/qwen/collab/`。

---

## 4. 环境与数据（两边共用，不复制）

```bash
conda activate qwen25vl    # 或 relhallu；不要 conda create 第二份
export CUDA_VISIBLE_DEVICES=0    # 先 nvidia-smi
set -a && source configs/paths.env && set +a
```

Python 脚本：

```python
from scripts.utils.paths import RELSIM_IMAGES, QWEN_MODEL, QWEN_PY
```

（若从仓库根以外 import，把 `REPO_ROOT` 加进 `sys.path`，或按现有 phase 脚本那样用绝对路径跑。）

默认值见 `configs/paths.env`。

---

## 5. 要改成 RH_* 的脚本清单

硬编码集中在三类：`RH_QWEN_PY`、`RH_QWEN_MODEL` / `RH_HF_HOME`、`RH_RELSIM_IMAGES`。改的时候 **先改正在用的入口**，旧 phase 可以后补。

### 优先（训练 / 评测入口，两边都会跑）

| 文件 | 硬编码 |
|---|---|
| `scripts/qwen_train/finetune_lora.sh` | `RH_QWEN_PY`；`--image_root` 现为仓内 `data/relsim_images` |
| `scripts/qwen_eval/eval_rbench.sh` | `RH_QWEN_PY`、`RH_QWEN_MODEL` |
| `scripts/qwen_eval/eval_pope_adv.sh` | 同上 |
| `scripts/qwen_eval/eval_amber_dr.sh` | 同上，另有 `RH_LLAVA_PY` |
| `scripts/qwen_eval/eval_mmrel.sh` | `RH_QWEN_PY`、`RH_QWEN_MODEL` |

### 其次（phase 5 / 4c，当前主线）

| 文件 | 硬编码 |
|---|---|
| `scripts/phase5a/run.sh` | `RH_QWEN_PY`、`RH_QWEN_MODEL`、`RH_HF_HOME` |
| `scripts/phase5_data/acquire_phase5_data.py` | `--root` → `RH_DATASETS` |
| `scripts/phase5_data/prepare_phase5_data.py` | `--root` → `RH_DATASETS` |
| `scripts/phase5b/build_psg_vcg.py` | PSG / COCO / work-dir → `RH_DATASETS` |
| `scripts/phase4c/train.sh` | `RH_QWEN_PY`、`RH_QWEN_MODEL`、`RH_RELSIM_IMAGES` |
| `scripts/phase4c/eval_{train,heldout,heldout_f2,fast,all}.sh` `run.sh` | 同上 |

### 其余（可后补，模式相同）

- **train / eval shell**：`scripts/phase2/train.sh`、`eval_train.sh`、`eval_heldout.sh`、`eval_benchmarks.sh`；`phase2b/train.sh`、`eval_*.sh`；`phase2b5/eval_matched.sh`；`phase3a/train.sh`、`eval_*.sh`、`run_*.sh`、`continue_train_eval.sh`；`phase4a/train.sh`、`eval_*.sh`、`run_*.sh`；`phase15/run_eval.sh`；`phase16/run_heldout_matched.sh`；`phase4b/run.sh`（还用 `RH_LLAVA_PY`）
- **Python 默认参数**：`phase3a/common.py`（图、100k json、InternVL）；`phase2/filter_hard.py`；`phase2/build_data.py`（还有一份 `/home/yy/...` 的死路径）；`phase15/build_heldout_mcq.py`、`eval_mcq.py`；`phase16/score_nll.py`、`score_heldout_matched.py`；`phase2b/qc_fast_subsets.py`
- **两套 RelSim 图**：部分脚本用仓内 `data/relsim_images`（1000 张），held-out / 4c 用 `/data/lht/relsim_dataset/relsim_images`（6.6 万张）。收口后一律 `RH_RELSIM_IMAGES`，仓内目录改成软链。

---

## 6. 两个分支怎么同步

两边共享 `.git`，本地提交立刻可见，不用先 fetch。GitHub 只作备份。

你改完、对方要用：

```bash
cd /data/storage22t/lht/works/relation_hallucination
git add -A && git commit -m "..." && git push

cd /data/storage22t/lht/works/relation_hallucination-collab
git merge main
git push
```

对方改完、你收进主线：在主目录 `git merge collab && git push`。

不要：在 collab 目录 `git switch main`；不要 `cp -r` 覆盖整个工作区。

---

## 7. 日常命令

你：

```bash
cd /data/storage22t/lht/works/relation_hallucination
conda activate qwen25vl    # 或 relhallu
git status
git add -A && git commit -m "..." && git push
```

师弟：

```bash
cd /data/storage22t/lht/works/relation_hallucination-collab
conda activate qwen25vl
git merge main
git add -A && git commit -m "..." && git push
```

---

## 8. 不要做的事

- 不要把 `LLaVA/`、`checkpoints/`、COCO、HF 权重、`MMRel/` 放进 Git 或再下一份。
- 不要为师弟再 `conda create` 一套 `relhallu` / `qwen25vl`。
- 不要两个人在同一工作目录改文件。
- 不要用 HTTPS 当 Git 远程。
- 改环境包装包前先商量。
- 推这个仓时不要用 `git@github.com:...`（会走旧钥匙，变成 3166596380）。

---

## 9. 账号与钥匙（本机核实，2026-09-21）

| 项 | 值 |
|---|---|
| 仓库 | 公开 `liuhetian123/relation_hallucination` |
| Git 作者 | `liuhetian123` `<doushabao24816@163.com>` |
| 本仓 SSH 私钥 | `~/.ssh/id_ed25519_liuhetian123` |
| 另一账号钥匙 | `~/.ssh/id_ed25519`（3166596380，不要删） |
| `ssh -T git@github.com-liuhetian123` | `Hi liuhetian123!` |
| origin | `git@github.com-liuhetian123:liuhetian123/relation_hallucination.git` |
| `main` | 已推送 `8f58680` |
| worktree | 尚未建立 |
