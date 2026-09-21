#!/usr/bin/env python3
"""Write Phase 2A report from held-out / benchmark metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Phase 1 already-published numbers for B0 / B1 (Explicit-1k caption SFT)
B0_B1 = {
    "rbench": {
        "base": {"Accuracy": 81.14, "Precision": 78.24, "Recall": 86.80, "F1": 82.30, "Yes_ratio": 56.02, "FP": 670.4},
        "explicit_1k": {"Accuracy": 81.13, "Precision": 78.18, "Recall": 86.89, "F1": 82.30, "Yes_ratio": 56.12, "FP": 673.4},
    },
    "pope_adv": {
        "base": {"Accuracy": 85.33, "Precision": 87.48, "Recall": 82.47, "F1": 84.90, "Yes_ratio": 47.13, "FP": 177},
        "explicit_1k": {"Accuracy": 85.27, "Precision": 87.04, "Recall": 82.87, "F1": 84.90, "Yes_ratio": 47.60, "FP": 185},
    },
    "mmrel_adv": {
        "base": {"Accuracy": 69.09, "Precision": 63.70, "Recall": 91.33, "F1": 75.05, "Yes_ratio": 72.99, "FP": 204},
        "explicit_1k": {"Accuracy": 68.96, "Precision": 63.59, "Recall": 91.33, "F1": 74.97, "Yes_ratio": 73.12, "FP": 205},
    },
    "mmrel_dalle_normal": {
        "base": {"Accuracy": 68.28, "Precision": 63.35, "Recall": 89.67, "F1": 74.24, "Yes_ratio": 72.17, "FP": 633},
        "explicit_1k": {"Accuracy": 68.24, "Precision": 63.33, "Recall": 89.59, "F1": 74.20, "Yes_ratio": 72.13, "FP": 633},
    },
    "amber_dr": {
        "base": {"Accuracy": 81.0, "Precision": 71.1, "Recall": 91.1, "F1": 79.9},
        "explicit_1k": {"Accuracy": 80.9, "Precision": 71.4, "Recall": 90.0, "F1": 79.6},
    },
}

RUNS = ("base", "explicit_1k", "v1", "v2", "v3")
RUN_LABEL = {
    "base": "B0 Base",
    "explicit_1k": "B1 Caption-SFT",
    "v1": "V1 Pos-only Verif",
    "v2": "V2 Pos+Random",
    "v3": "V3 Pos+Hard",
}


def loadj(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x, nd=2):
    if x is None:
        return "—"
    v = x * 100 if 0 <= x <= 1 else x
    return f"{v:.{nd}f}"


def cell(rec, key, as_pct=True):
    if not rec or rec.get(key) is None:
        return "—"
    val = rec[key]
    if as_pct and key in {"Accuracy", "Precision", "Recall", "F1", "Yes_ratio"} and isinstance(val, float) and val <= 1:
        return f"{100 * val:.2f}"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def decide(heldout: dict) -> str:
    def acc(run, subset="all"):
        rec = (heldout.get(run) or {}).get(subset) or {}
        return rec.get("Accuracy")

    v1, v2, v3 = acc("v1"), acc("v2"), acc("v3")
    if v1 is None or v2 is None:
        return "Incomplete: missing held-out metrics."
    yes_v2 = ((heldout.get("v2") or {}).get("all") or {}).get("Yes_ratio")
    yes_v3 = ((heldout.get("v3") or {}).get("all") or {}).get("Yes_ratio")
    hard_v3 = ((heldout.get("v3") or {}).get("hard_negative") or {}).get("hard_fp_rate")
    hard_v2 = ((heldout.get("v2") or {}).get("hard_negative") or {}).get("hard_fp_rate")
    if v2 > v1 + 0.03 or (v3 is not None and v3 > v1 + 0.03):
        if v3 is not None and v2 is not None and v3 > v2 + 0.02:
            return "Go B:\nHard negative > random negative on held-out verification."
        return "Go A (held-out only until external benchmarks):\nNegative supervision improves custom verification."
    if abs((v2 or 0) - (v1 or 0)) < 0.02 and (v3 is None or abs(v3 - v1) < 0.02):
        return "No-Go B:\nRandom / Hard did not change held-out verification beyond V1."
    return "No-Go A candidate:\nCheck external transfer after benchmarks."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_dir", default=str(ROOT / "eval_results/qwen/phase2/metrics"))
    parser.add_argument("--qc", default=str(ROOT / "data/phase2/assemble_qc.json"))
    parser.add_argument("--out", default=str(ROOT / "phase2_result.md"))
    args = parser.parse_args()
    mdir = Path(args.metrics_dir)
    qc = loadj(Path(args.qc)) or {}

    heldout = {}
    train = {}
    for run in RUNS:
        heldout[run] = loadj(mdir / f"{run}_heldout.json")
        if run.startswith("v"):
            train[run] = loadj(mdir / f"{run}_train.json")

    ext = {}
    for bench, fname in [
        ("mmrel_adv", "mmrel_adv"),
        ("mmrel_dalle_normal", "mmrel_dalle_normal"),
        ("pope_adv", "pope_adv"),
    ]:
        ext[bench] = {}
        for run in RUNS:
            if run in {"base", "explicit_1k"}:
                ext[bench][run] = B0_B1.get(bench, {}).get(run)
            else:
                ext[bench][run] = loadj(ROOT / f"eval_results/qwen/mmrel/logs/qwen25vl3b_{run}_mmrel_{'adv' if bench=='mmrel_adv' else 'dalle_normal'}_metrics.json") if "mmrel" in bench else loadj(ROOT / f"eval_results/qwen/pope/logs/qwen25vl3b_{run}_pope_adv_metrics.json")

    lines = [
        "# Phase 2A Negative Relation Verification",
        "",
        "只训练 Explicit verification。B0 / B1 复用 Phase 1；V1/V2/V3 为新 LoRA。",
        "",
        "## 数据 QC",
        "",
        f"- V1 positives: {qc.get('n_v1')}",
        f"- V2 samples: {qc.get('n_v2')}",
        f"- V3 samples: {qc.get('n_v3')} (hard negatives {qc.get('n_train_hard_images')})",
        f"- Held-out test: pos {qc.get('heldout_positive')} / random {qc.get('heldout_random_selected')} / hard {qc.get('heldout_hard_selected')} / total {qc.get('heldout_total')}",
        f"- InternVL judgments: {qc.get('internvl_status_counts')}",
        "",
        "## Held-out verification",
        "",
        "| 模型 | Acc | P | R | F1 | Yes | Pos Acc | Random Acc | Hard Acc | Hard FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in RUNS:
        rec = heldout.get(run) or {}
        allr = rec.get("all") or {}
        pos = rec.get("positive") or {}
        rnd = rec.get("random_negative") or {}
        hard = rec.get("hard_negative") or {}
        lines.append(
            f"| {RUN_LABEL[run]} | {cell(allr,'Accuracy')} | {cell(allr,'Precision')} | "
            f"{cell(allr,'Recall')} | {cell(allr,'F1')} | {cell(allr,'Yes_ratio')} | "
            f"{cell(pos,'Accuracy')} | {cell(rnd,'Accuracy')} | {cell(hard,'Accuracy')} | "
            f"{cell(hard,'hard_fp_rate')} |"
        )
    lines += ["", "## Decision", "", "```text", decide(heldout), "```", ""]
    report = "\n".join(lines)
    Path(args.out).write_text(report + "\n", encoding="utf-8")
    (ROOT / "eval_results/qwen/phase2/phase2_report.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
