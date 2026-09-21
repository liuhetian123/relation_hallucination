#!/usr/bin/env python3
"""Paired NLL analysis and Phase 1.6 A/B/C/D decision."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SPLITS = ("train_explicit", "train_typed", "heldout_rel")
RUNS = ("base", "explicit_1k", "typed_1k")
RUN_LABEL = {
    "base": "Base",
    "explicit_1k": "Explicit-LoRA",
    "typed_1k": "Typed-LoRA",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def index_scores(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        if r.get("error") or r.get("full_mean_nll") is None:
            continue
        out[(r["id"], r["split"])] = r
    return out


def summarize(vals: np.ndarray) -> dict:
    vals = np.asarray(vals, dtype=float)
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()) if vals.size else None,
        "median": float(np.median(vals)) if vals.size else None,
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
    }


def bootstrap_mean(vals: np.ndarray, seed: int = 42, iters: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    n = len(vals)
    if n == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    boots = np.empty(iters)
    for i in range(iters):
        boots[i] = vals[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(vals.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def paired_delta(base: np.ndarray, lora: np.ndarray) -> dict:
    # ΔNLL = NLL_base - NLL_lora  (>0 means LoRA more likely)
    delta = base - lora
    rec = bootstrap_mean(delta)
    rec["positive_rate"] = float((delta > 0).mean()) if delta.size else None
    rec["median"] = float(np.median(delta)) if delta.size else None
    rec["std"] = float(delta.std(ddof=1)) if delta.size > 1 else 0.0
    rec["n"] = int(delta.size)
    return rec


def decide(table: dict, deltas: dict) -> str:
    def mean_of(split, metric, run):
        return table[split][metric][run]["mean"]

    def d(split, metric, run):
        return deltas[split][metric][run]

    te_full = d("train_explicit", "full", "explicit_1k")
    tt_full = d("train_typed", "full", "typed_1k")
    te_rel = d("train_explicit", "rel", "explicit_1k")
    tt_rel = d("train_typed", "rel", "typed_1k")
    ho = d("heldout_rel", "full", "explicit_1k")
    ho_t = d("heldout_rel", "full", "typed_1k")

    def clear_gain(rec):
        return rec and rec["mean"] is not None and rec["ci95_low"] > 0.03 and rec["mean"] > 0.05

    def none_gain(rec):
        if not rec or rec["mean"] is None:
            return True
        return rec["ci95_high"] < 0.05 and rec["mean"] < 0.05

    train_full_ok = clear_gain(te_full) or clear_gain(tt_full)
    train_rel_ok = clear_gain(te_rel) or clear_gain(tt_rel)
    train_full_none = none_gain(te_full) and none_gain(tt_full)
    train_rel_none = none_gain(te_rel) and none_gain(tt_rel)
    ho_ok = clear_gain(ho) or clear_gain(ho_t)
    ho_none = none_gain(ho) and none_gain(ho_t)

    if train_full_ok and train_rel_none:
        return "Phase 1.6-D:\nSurface-form adaptation without clear relation-token adaptation."
    if train_full_ok and train_rel_ok and ho_ok:
        return (
            "Phase 1.6-C:\nRelation supervision is learned and generalized,\n"
            "but does not transfer to hallucination benchmarks."
        )
    if train_full_ok and ho_none:
        return "Phase 1.6-B:\nTraining fits train data but does not generalize."
    if train_full_none and train_rel_none and ho_none:
        return "Phase 1.6-A:\nTraining intervention too weak."
    if train_full_ok and train_rel_ok and ho_none:
        return "Phase 1.6-B:\nTraining fits train data but does not generalize."
    return "Phase 1.6-A:\nTraining intervention too weak."


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_dir", default="eval_results/qwen/phase16/scores")
    parser.add_argument("--out_dir", default="eval_results/qwen/phase16")
    parser.add_argument("--qc_spans", default="eval_results/qwen/phase16/qc_spans.json")
    args = parser.parse_args()

    scores = {}
    for run in RUNS:
        path = Path(args.score_dir) / f"{run}.jsonl"
        scores[run] = index_scores(load_jsonl(path))

    # intersect ids per split
    table = {}
    deltas = {}
    per_sample = []
    for split in SPLITS:
        keys = None
        for run in RUNS:
            ks = {k for k in scores[run] if k[1] == split}
            keys = ks if keys is None else keys & ks
        keys = sorted(keys)
        full = {run: [] for run in RUNS}
        rel = {run: [] for run in RUNS}
        for key in keys:
            recs = {run: scores[run][key] for run in RUNS}
            for run in RUNS:
                full[run].append(recs[run]["full_mean_nll"])
            if all(recs[run].get("rel_located_tokens") and recs[run].get("rel_mean_nll") is not None for run in RUNS):
                for run in RUNS:
                    rel[run].append(recs[run]["rel_mean_nll"])
            per_sample.append(
                {
                    "id": key[0],
                    "split": split,
                    **{f"{run}_full": recs[run]["full_mean_nll"] for run in RUNS},
                    **{
                        f"{run}_rel": recs[run].get("rel_mean_nll")
                        for run in RUNS
                    },
                }
            )
        table[split] = {
            "full": {run: summarize(np.array(full[run])) for run in RUNS},
            "rel": {run: summarize(np.array(rel[run])) for run in RUNS},
        }
        deltas[split] = {
            "full": {
                "explicit_1k": paired_delta(np.array(full["base"]), np.array(full["explicit_1k"])),
                "typed_1k": paired_delta(np.array(full["base"]), np.array(full["typed_1k"])),
            },
            "rel": {
                "explicit_1k": paired_delta(np.array(rel["base"]), np.array(rel["explicit_1k"]))
                if rel["base"]
                else {"mean": None, "ci95_low": None, "ci95_high": None, "positive_rate": None, "n": 0},
                "typed_1k": paired_delta(np.array(rel["base"]), np.array(rel["typed_1k"]))
                if rel["base"]
                else {"mean": None, "ci95_low": None, "ci95_high": None, "positive_rate": None, "n": 0},
            },
        }

    decision = decide(table, deltas)
    qc = {}
    qc_path = Path(args.qc_spans)
    if qc_path.exists():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = {"table": table, "deltas": deltas, "decision": decision, "span_qc": qc}
    (out_dir / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "paired_nll.jsonl").open("w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row) + "\n")

    lines = [
        "# Phase 1.6 LoRA Adaptation / Relation-Token NLL",
        "",
        "Teacher-forced mean token NLL。ΔNLL = NLL(Base) − NLL(LoRA)，**正值表示 LoRA 提高了 target likelihood**。",
        "",
        "## Relation-span QC（训练 955）",
        "",
    ]
    if qc:
        lines += [
            f"- Total: {qc.get('total')}",
            f"- Successfully located: {qc.get('successfully_located_relation')}",
            f"- Failed: {qc.get('failed_relation_location')}",
            f"- Multi-match: {qc.get('multi_match')}",
            f"- Empty: {qc.get('empty_relation')}",
            "",
            "无法定位的样本仍计入 full-target NLL，不计入 relation-token NLL。",
            "",
        ]
    lines += [
        "## 主表",
        "",
        "| Split | Metric | Base | Explicit-LoRA | Typed-LoRA |",
        "|---|---|---:|---:|---:|",
        "| Train | Explicit full-target NLL ↓ | "
        f"{fmt(table['train_explicit']['full']['base']['mean'])} | "
        f"{fmt(table['train_explicit']['full']['explicit_1k']['mean'])} | "
        f"{fmt(table['train_explicit']['full']['typed_1k']['mean'])} |",
        "| Train | Typed full-target NLL ↓ | "
        f"{fmt(table['train_typed']['full']['base']['mean'])} | "
        f"{fmt(table['train_typed']['full']['explicit_1k']['mean'])} | "
        f"{fmt(table['train_typed']['full']['typed_1k']['mean'])} |",
        "| Train | Explicit-context relation NLL ↓ | "
        f"{fmt(table['train_explicit']['rel']['base']['mean'])} | "
        f"{fmt(table['train_explicit']['rel']['explicit_1k']['mean'])} | "
        f"{fmt(table['train_explicit']['rel']['typed_1k']['mean'])} |",
        "| Train | Typed-context relation NLL ↓ | "
        f"{fmt(table['train_typed']['rel']['base']['mean'])} | "
        f"{fmt(table['train_typed']['rel']['explicit_1k']['mean'])} | "
        f"{fmt(table['train_typed']['rel']['typed_1k']['mean'])} |",
        "| Held-out | Relation-only NLL ↓ | "
        f"{fmt(table['heldout_rel']['full']['base']['mean'])} | "
        f"{fmt(table['heldout_rel']['full']['explicit_1k']['mean'])} | "
        f"{fmt(table['heldout_rel']['full']['typed_1k']['mean'])} |",
        "",
        "## Paired ΔNLL vs Base（bootstrap 95% CI）",
        "",
    ]
    for split, metric, run, label in [
        ("train_explicit", "full", "explicit_1k", "Train Explicit full / Explicit-LoRA"),
        ("train_typed", "full", "typed_1k", "Train Typed full / Typed-LoRA"),
        ("train_explicit", "rel", "explicit_1k", "Train Explicit relation / Explicit-LoRA"),
        ("train_typed", "rel", "typed_1k", "Train Typed relation / Typed-LoRA"),
        ("heldout_rel", "full", "explicit_1k", "Held-out relation-only / Explicit-LoRA"),
        ("heldout_rel", "full", "typed_1k", "Held-out relation-only / Typed-LoRA"),
    ]:
        rec = deltas[split][metric][run]
        lines.append(
            f"- **{label}**: Δ={fmt(rec.get('mean'))} "
            f"[{fmt(rec.get('ci95_low'))}, {fmt(rec.get('ci95_high'))}] "
            f"median={fmt(rec.get('median'))} positive-rate={fmt(rec.get('positive_rate'), 3)} n={rec.get('n')}"
        )
    lines += ["", "## Decision", "", "```text", decision, "```", ""]
    report = "\n".join(lines)
    (out_dir / "phase16_report.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
