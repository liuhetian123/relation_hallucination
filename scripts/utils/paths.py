"""Central paths for relation_hallucination. Override with RH_* env vars."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _p(key: str, default: str | Path) -> Path:
    return Path(os.environ.get(key, str(default)))


RELSIM = _p("RH_RELSIM", "/data/lht/relsim_dataset")
RELSIM_IMAGES = _p("RH_RELSIM_IMAGES", RELSIM / "relsim_images")
RELSIM_100K = _p("RH_RELSIM_100K", RELSIM / "relsim_llava_100k.json")

DATASETS = _p("RH_DATASETS", "/data/storage22t/lht/datasets")
COCO2017 = DATASETS / "COCO2017"
PSG_JSON = DATASETS / "OpenPSG_annotations" / "psg.json"
PHASE5B_WORK = DATASETS / "phase5b_work"

QWEN_PY = _p("RH_QWEN_PY", "/data/storage22t/lht/envs/qwen25vl/bin/python")
LLAVA_PY = _p("RH_LLAVA_PY", "/home/lht/miniconda3/envs/relhallu/bin/python")

HF_HOME = _p("RH_HF_HOME", Path.home() / ".cache/huggingface")
HF_CACHE_ALT = _p("RH_HF_CACHE_ALT", "/data/storage22t/lht/hf_cache")
QWEN_MODEL = _p(
    "RH_QWEN_MODEL",
    HF_HOME
    / "hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots"
    / "66285546d2b821cf421d4f5eb2576359d3770cd3",
)
INTERNVL_MODEL = _p(
    "RH_INTERNVL_MODEL",
    HF_CACHE_ALT
    / "models--OpenGVLab--InternVL3_5-8B-HF/snapshots"
    / "741a7d03020411e666c6109218ab71e08151ef86",
)


def as_env() -> dict[str, str]:
    return {
        "RH_RELSIM": str(RELSIM),
        "RH_RELSIM_IMAGES": str(RELSIM_IMAGES),
        "RH_RELSIM_100K": str(RELSIM_100K),
        "RH_DATASETS": str(DATASETS),
        "RH_QWEN_PY": str(QWEN_PY),
        "RH_LLAVA_PY": str(LLAVA_PY),
        "RH_HF_HOME": str(HF_HOME),
        "RH_HF_CACHE_ALT": str(HF_CACHE_ALT),
        "RH_QWEN_MODEL": str(QWEN_MODEL),
        "RH_INTERNVL_MODEL": str(INTERNVL_MODEL),
    }
