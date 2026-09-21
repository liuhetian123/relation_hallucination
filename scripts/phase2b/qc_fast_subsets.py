#!/usr/bin/env python3
"""Rescore Phase 2A full predictions on the frozen fast subsets (no new inference)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = "/data/storage22t/lht/envs/qwen25vl/bin/python"
OUT = ROOT / "eval_results/qwen/phase2b/metrics"
SCORE = ROOT / "scripts/phase2b/score_fast.py"

RUNS = {
    "base": {
        "rbench": "eval_results/qwen/rbench/answers/qwen25vl3b_base_image-level.jsonl",
        "mmrel_adv": "eval_results/qwen/mmrel/answers/qwen25vl3b_base_mmrel_adv.jsonl",
        "amber_dr": "eval_results/qwen/amber/answers/qwen25vl3b_base_amber_dr.json",
    },
    "explicit_1k": {
        "rbench": "eval_results/qwen/rbench/answers/qwen25vl3b_explicit_1k_image-level.jsonl",
        "mmrel_adv": "eval_results/qwen/mmrel/answers/qwen25vl3b_explicit_1k_mmrel_adv.jsonl",
        "amber_dr": "eval_results/qwen/amber/answers/qwen25vl3b_explicit_1k_amber_dr.json",
    },
    "v1": {
        "rbench": "eval_results/qwen/rbench/answers/qwen25vl3b_v1_image-level.jsonl",
        "mmrel_adv": "eval_results/qwen/mmrel/answers/qwen25vl3b_v1_mmrel_adv.jsonl",
        "amber_dr": "eval_results/qwen/amber/answers/qwen25vl3b_v1_amber_dr.json",
    },
    "v2": {
        "rbench": "eval_results/qwen/rbench/answers/qwen25vl3b_v2_image-level.jsonl",
        "mmrel_adv": "eval_results/qwen/mmrel/answers/qwen25vl3b_v2_mmrel_adv.jsonl",
        "amber_dr": "eval_results/qwen/amber/answers/qwen25vl3b_v2_amber_dr.json",
    },
    "v3": {
        "rbench": "eval_results/qwen/rbench/answers/qwen25vl3b_v3_image-level.jsonl",
        "mmrel_adv": "eval_results/qwen/mmrel/answers/qwen25vl3b_v3_mmrel_adv.jsonl",
        "amber_dr": "eval_results/qwen/amber/answers/qwen25vl3b_v3_amber_dr.json",
    },
}

FULL = {
    "rbench": {
        "base": {"Accuracy": 81.14, "Precision": 78.24, "Recall": 86.80, "F1": 82.30, "Yes_ratio": 56.02},
        "explicit_1k": {"Accuracy": 81.13, "Precision": 78.18, "Recall": 86.89, "F1": 82.30, "Yes_ratio": 56.12},
        "v1": {"Accuracy": 80.65, "Precision": 76.53, "Recall": 88.97, "F1": 82.28, "Yes_ratio": 58.70},
        "v2": {"Accuracy": 81.22, "Precision": 78.52, "Recall": 86.45, "F1": 82.30, "Yes_ratio": 55.59},
        "v3": {"Accuracy": 81.33, "Precision": 78.46, "Recall": 86.89, "F1": 82.46, "Yes_ratio": 55.92},
    },
    "mmrel_adv": {
        "base": {"Accuracy": 69.09, "Precision": 63.70, "Recall": 91.33, "F1": 75.05, "Yes_ratio": 72.99},
        "explicit_1k": {"Accuracy": 68.96, "Precision": 63.59, "Recall": 91.33, "F1": 74.97, "Yes_ratio": 73.12},
        "v1": {"Accuracy": 67.53, "Precision": 62.16, "Recall": 92.60, "F1": 74.39, "Yes_ratio": 75.84},
        "v2": {"Accuracy": 69.35, "Precision": 64.08, "Recall": 90.56, "F1": 75.05, "Yes_ratio": 71.95},
        "v3": {"Accuracy": 69.35, "Precision": 63.98, "Recall": 91.07, "F1": 75.16, "Yes_ratio": 72.47},
    },
    "amber_dr": {
        "base": {"Accuracy": 81.0, "Precision": 71.1, "Recall": 91.1, "F1": 79.9},
        "explicit_1k": {"Accuracy": 80.9, "Precision": 71.4, "Recall": 90.0, "F1": 79.6},
        "v1": {"Accuracy": 82.9, "Precision": 75.0, "Recall": 88.1, "F1": 81.0},
        "v2": {"Accuracy": 80.6, "Precision": 70.3, "Recall": 91.9, "F1": 79.7},
        "v3": {"Accuracy": 80.8, "Precision": 70.6, "Recall": 92.2, "F1": 80.0},
    },
}


def to_pct(rec: dict, key: str) -> float | None:
    if not rec or rec.get(key) is None:
        return None
    val = rec[key]
    if isinstance(val, float) and val <= 1.0 and key in {"Accuracy", "Precision", "Recall", "F1", "Yes_ratio"}:
        return 100.0 * val
    return float(val)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = {}
    for run, files in RUNS.items():
        table[run] = {}
        for bench, path in files.items():
            out_json = OUT / f"{run}_{bench}_fast_from_full.json"
            cmd = [
                PY,
                str(SCORE),
                "--bench",
                bench,
                "--result_file",
                str(ROOT / path),
                "--out_json",
                str(out_json),
            ]
            print(" ".join(cmd), flush=True)
            subprocess.check_call(cmd, cwd=ROOT)
            rec = json.loads(out_json.read_text(encoding="utf-8"))
            full = FULL[bench][run]
            row = {
                "subset": {k: to_pct(rec, k) for k in ("Accuracy", "Precision", "Recall", "F1", "Yes_ratio")},
                "full": full,
                "delta_acc": None,
            }
            sub_acc = row["subset"]["Accuracy"]
            if sub_acc is not None:
                row["delta_acc"] = round(sub_acc - full["Accuracy"], 2)
            table[run][bench] = row
    ranking = {}
    for bench in ("rbench", "mmrel_adv", "amber_dr"):
        full_order = sorted(FULL[bench], key=lambda r: FULL[bench][r]["Accuracy"], reverse=True)
        sub_order = sorted(table, key=lambda r: table[r][bench]["subset"]["Accuracy"] or -1, reverse=True)
        ranking[bench] = {"full": full_order, "subset": sub_order}
    blob = {"runs": table, "ranking": ranking}
    out = ROOT / "eval_results/qwen/phase2b/fast_subsets/qc_full_vs_subset.json"
    out.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(blob, indent=2))


if __name__ == "__main__":
    main()
