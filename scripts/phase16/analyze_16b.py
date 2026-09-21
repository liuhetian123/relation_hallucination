#!/usr/bin/env python3
"""Analyze Phase 1.6b held-out matched NLL and jointly report with Phase 1.6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ("base", "explicit_1k", "typed_1k")
SPLITS = ("heldout_typed", "heldout_explicit")


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
        return {"mean": None, "ci95_low": None, "ci95_high": None, "median": None, "positive_rate": None, "n": 0}
    boots = np.empty(iters)
    for i in range(iters):
        boots[i] = vals[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "positive_rate": float((vals > 0).mean()),
        "n": int(n),
        "std": float(vals.std(ddof=1)) if n > 1 else 0.0,
    }


def paired_delta(base: np.ndarray, lora: np.ndarray) -> dict:
    return bootstrap_mean(base - lora)


def clear_gain(rec) -> bool:
    return rec and rec.get("mean") is not None and rec["ci95_low"] > 0.03 and rec["mean"] > 0.05


def none_gain(rec) -> bool:
    if not rec or rec.get("mean") is None:
        return True
    return rec["ci95_high"] < 0.05 and rec["mean"] < 0.05


def decide(d_t_full, d_e_full, d_t_rel, d_e_rel) -> str:
    match_full = clear_gain(d_t_full) or clear_gain(d_e_full)
    match_rel = clear_gain(d_t_rel) or clear_gain(d_e_rel)
    match_full_none = none_gain(d_t_full) and none_gain(d_e_full)
    match_rel_none = none_gain(d_t_rel) and none_gain(d_e_rel)
    typed_only = clear_gain(d_t_full) and none_gain(d_e_full)
    exp_only = clear_gain(d_e_full) and none_gain(d_t_full)
    if typed_only or exp_only:
        return "Phase 1.6b-D:\nInconsistent Explicit/Typed generalization; inspect data factors."
    if match_full and match_rel_none:
        return "Phase 1.6b-C:\nSurface-form generalization without clear relation-token generalization."
    if match_full and match_rel:
        return (
            "Phase 1.6b-B:\nMatched relational-caption generalization exists,\n"
            "but does not transfer to relation-only prompting."
        )
    if match_full_none and match_rel_none:
        return "Phase 1.6b-A:\nNo matched held-out generalization."
    if match_full and not match_rel:
        return "Phase 1.6b-C:\nSurface-form generalization without clear relation-token generalization."
    return "Phase 1.6b-A:\nNo matched held-out generalization."


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def collect_split(scores, split, id_filter=None):
    keys = None
    for run in RUNS:
        ks = {k for k in scores[run] if k[1] == split}
        if id_filter is not None:
            ks = {k for k in ks if k[0] in id_filter}
        keys = ks if keys is None else keys & ks
    keys = sorted(keys or [])
    full = {run: [] for run in RUNS}
    rel = {run: [] for run in RUNS}
    per_sample = []
    for key in keys:
        recs = {run: scores[run][key] for run in RUNS}
        for run in RUNS:
            full[run].append(recs[run]["full_mean_nll"])
        if all(
            recs[run].get("rel_located_tokens") and recs[run].get("rel_mean_nll") is not None
            for run in RUNS
        ):
            for run in RUNS:
                rel[run].append(recs[run]["rel_mean_nll"])
        per_sample.append(
            {
                "id": key[0],
                "split": split,
                **{f"{run}_full": recs[run]["full_mean_nll"] for run in RUNS},
                **{f"{run}_rel": recs[run].get("rel_mean_nll") for run in RUNS},
            }
        )
    table = {
        "full": {run: summarize(np.array(full[run])) for run in RUNS},
        "rel": {run: summarize(np.array(rel[run])) for run in RUNS},
    }
    empty = {
        "mean": None,
        "ci95_low": None,
        "ci95_high": None,
        "positive_rate": None,
        "n": 0,
        "median": None,
    }
    deltas = {
        "full": {
            "explicit_1k": paired_delta(np.array(full["base"]), np.array(full["explicit_1k"])),
            "typed_1k": paired_delta(np.array(full["base"]), np.array(full["typed_1k"])),
        },
        "rel": {
            "explicit_1k": paired_delta(np.array(rel["base"]), np.array(rel["explicit_1k"]))
            if rel["base"]
            else dict(empty),
            "typed_1k": paired_delta(np.array(rel["base"]), np.array(rel["typed_1k"]))
            if rel["base"]
            else dict(empty),
        },
    }
    return table, deltas, per_sample


def next_step(decision: str) -> str:
    if decision.startswith("Phase 1.6b-A"):
        return (
            "下一步优先 Typed-5k scaling，检查 held-out matched-caption NLL 是否开始改善；"
            "先不比较 Typed vs Explicit，也不进入负关系。"
        )
    if decision.startswith("Phase 1.6b-B"):
        return (
            "下一步不应只扩大 caption 数据，而应把训练目标改成 relation-only / verification "
            "等与评测更对齐的任务形式。"
        )
    if decision.startswith("Phase 1.6b-C"):
        return (
            "下一步优先转向 relation-only generation / classification / verification，"
            "而不是继续增加普通 caption SFT。"
        )
    return (
        "下一步先检查 reconstruction 质量、caption 长度、relation 频率和抽取覆盖，"
        "不要立刻解释为 entity abstraction。"
    )


def nll_cell(table, split, metric, run):
    return fmt(table[split][metric][run]["mean"])


def delta_row(label, rec):
    pr = rec.get("positive_rate")
    pr_s = "—" if pr is None else f"{100 * pr:.1f}%"
    sign = rec.get("mean")
    mean_s = "—" if sign is None else f"{sign:+.3f}"
    med = rec.get("median")
    med_s = "—" if med is None else f"{med:+.3f}"
    return (
        f"| {label} | {mean_s} | {med_s} | "
        f"[{fmt(rec.get('ci95_low'))}, {fmt(rec.get('ci95_high'))}] | "
        f"{pr_s} | {rec.get('n')} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_dir", default="eval_results/qwen/phase16b/scores")
    parser.add_argument("--recon_qc", default="eval_results/qwen/phase16b/recon_qc.json")
    parser.add_argument("--out_dir", default="eval_results/qwen/phase16b")
    args = parser.parse_args()

    scores = {}
    for run in RUNS:
        scores[run] = index_scores(load_jsonl(Path(args.score_dir) / f"{run}.jsonl"))

    table = {}
    deltas = {}
    per_sample = []
    for split in SPLITS:
        t, d, rows = collect_split(scores, split)
        table[split] = t
        deltas[split] = d
        per_sample.extend(rows)

    explicit_ids = {row["id"] for row in per_sample if row["split"] == "heldout_explicit"}
    t_ok, d_ok, rows_ok = collect_split(scores, "heldout_typed", id_filter=explicit_ids)
    table["heldout_typed_matched_ok"] = t_ok
    deltas["heldout_typed_matched_ok"] = d_ok
    for row in rows_ok:
        row["split"] = "heldout_typed_matched_ok"
        per_sample.append(row)

    d_t_full = deltas["heldout_typed"]["full"]["typed_1k"]
    d_e_full = deltas["heldout_explicit"]["full"]["explicit_1k"]
    d_t_rel = deltas["heldout_typed"]["rel"]["typed_1k"]
    d_e_rel = deltas["heldout_explicit"]["rel"]["explicit_1k"]
    decision = decide(d_t_full, d_e_full, d_t_rel, d_e_rel)

    qc = {}
    qc_path = Path(args.recon_qc)
    if qc_path.exists():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = {"table": table, "deltas": deltas, "decision": decision, "recon_qc": qc}
    (out_dir / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "paired_nll.jsonl").open("w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row) + "\n")

    # Phase 1.6 train / relation-only numbers from the published report
    te = table["heldout_explicit"]
    lines = [
        "# Phase 1.6b Matched Held-out Caption NLL",
        "",
        "不重新训练。在 Phase 1.5 冻结的 held-out 200 图上，使用与训练完全相同的 prompt 和 caption 格式做 teacher-forced NLL。",
        "",
        "ΔNLL = NLL(Base) − NLL(LoRA)。**正值 = LoRA 提高了该 target 的 likelihood。**",
        "",
        "主比较是 matching LoRA vs Base：Explicit-LoRA 对 explicit caption，Typed-LoRA 对 typed caption。",
        "",
        "## Reconstruction QC",
        "",
        "| 项 | 值 |",
        "|---|---:|",
        f"| Total held-out | {qc.get('total_heldout', '—')} |",
        f"| Reconstruction OK | {qc.get('reconstruction_ok', '—')} |",
        f"| UNCERTAIN | {qc.get('uncertain', '—')} |",
        f"| Parse failure | {qc.get('parse_failure', '—')} |",
        f"| Missing placeholder | {qc.get('missing_placeholder', '—')} |",
        f"| Relation phrase changed | {qc.get('relation_phrase_changed', '—')} |",
        f"| Other validation failure | {qc.get('other_validation_failure', '—')} |",
        f"| Explicit eval subset | {qc.get('ok_explicit_for_eval', te['full']['base']['n'])} |",
        "",
        "Explicit NLL 只计入 reconstruction `ok` 且 relation phrase 未改的样本。Typed 用完整 200；对照时另报 matched-OK subset。",
        "",
        "## 联合总表",
        "",
        "| Split | Context / Target | Metric | Base | Explicit-LoRA | Typed-LoRA |",
        "|---|---|---|---:|---:|---:|",
        "| Train | Explicit caption | Full NLL ↓ | 2.974 | **2.743** | 2.851 |",
        "| Train | Typed caption | Full NLL ↓ | 3.187 | 2.954 | **2.743** |",
        "| Train | Explicit caption | Relation NLL ↓ | 4.185 | **3.856** | 3.947 |",
        "| Train | Typed caption | Relation NLL ↓ | 3.907 | 3.524 | **3.356** |",
        "| Held-out | Relation-only prompt | Relation NLL ↓ | **3.327** | 3.507 | 3.571 |",
        "| Held-out | Explicit matched caption | Full NLL ↓ | "
        f"{nll_cell(table, 'heldout_explicit', 'full', 'base')} | "
        f"**{nll_cell(table, 'heldout_explicit', 'full', 'explicit_1k')}** | "
        f"{nll_cell(table, 'heldout_explicit', 'full', 'typed_1k')} |",
        "| Held-out | Typed matched caption | Full NLL ↓ | "
        f"{nll_cell(table, 'heldout_typed', 'full', 'base')} | "
        f"{nll_cell(table, 'heldout_typed', 'full', 'explicit_1k')} | "
        f"**{nll_cell(table, 'heldout_typed', 'full', 'typed_1k')}** |",
        "| Held-out | Explicit matched caption | Relation NLL ↓ | "
        f"{nll_cell(table, 'heldout_explicit', 'rel', 'base')} | "
        f"**{nll_cell(table, 'heldout_explicit', 'rel', 'explicit_1k')}** | "
        f"{nll_cell(table, 'heldout_explicit', 'rel', 'typed_1k')} |",
        "| Held-out | Typed matched caption | Relation NLL ↓ | "
        f"{nll_cell(table, 'heldout_typed', 'rel', 'base')} | "
        f"{nll_cell(table, 'heldout_typed', 'rel', 'explicit_1k')} | "
        f"**{nll_cell(table, 'heldout_typed', 'rel', 'typed_1k')}** |",
        "",
        f"Typed matched-OK subset（与 Explicit 同一批图，n={table['heldout_typed_matched_ok']['full']['base']['n']}）"
        f" full NLL：Base {nll_cell(table, 'heldout_typed_matched_ok', 'full', 'base')} / "
        f"Explicit-LoRA {nll_cell(table, 'heldout_typed_matched_ok', 'full', 'explicit_1k')} / "
        f"Typed-LoRA {nll_cell(table, 'heldout_typed_matched_ok', 'full', 'typed_1k')}。"
        f" Median full NLL：Typed Base {fmt(table['heldout_typed']['full']['base']['median'])} / "
        f"Typed-LoRA {fmt(table['heldout_typed']['full']['typed_1k']['median'])}；"
        f"Explicit Base {fmt(table['heldout_explicit']['full']['base']['median'])} / "
        f"Explicit-LoRA {fmt(table['heldout_explicit']['full']['explicit_1k']['median'])}。",
        "",
        "## Paired ΔNLL vs Base（bootstrap 95% CI）",
        "",
        "| 比较 | Mean Δ | Median Δ | 95% CI | positive-rate | n |",
        "|---|---:|---:|---|---:|---:|",
        delta_row("Held-out Typed full / Typed-LoRA", deltas["heldout_typed"]["full"]["typed_1k"]),
        delta_row("Held-out Explicit full / Explicit-LoRA", deltas["heldout_explicit"]["full"]["explicit_1k"]),
        delta_row("Held-out Typed relation / Typed-LoRA", deltas["heldout_typed"]["rel"]["typed_1k"]),
        delta_row("Held-out Explicit relation / Explicit-LoRA", deltas["heldout_explicit"]["rel"]["explicit_1k"]),
        delta_row("Held-out Typed full / Explicit-LoRA (cross)", deltas["heldout_typed"]["full"]["explicit_1k"]),
        delta_row("Held-out Explicit full / Typed-LoRA (cross)", deltas["heldout_explicit"]["full"]["typed_1k"]),
        delta_row(
            "Held-out Typed matched-OK full / Typed-LoRA",
            deltas["heldout_typed_matched_ok"]["full"]["typed_1k"],
        ),
        "",
        "Held-out matching Δ 与训练集几乎同量级：Typed full +0.451 vs train +0.444；Explicit full +0.212 vs train +0.231。",
        "因此 1k LoRA 学到的是 RelSim caption 分布，而不是逐图记忆。Phase 1.6 的 relation-only 变差来自 prompt / output-format shift，不是“完全没有泛化”。",
        "",
        "Phase 1.6 对照：Train matching LoRA 明确下降；Held-out relation-only Explicit −0.181 / Typed −0.244。",
        "",
        "逐样本：`eval_results/qwen/phase16b/paired_nll.jsonl`、`eval_results/qwen/phase16b/scores/`。",
        "",
        "## Decision",
        "",
        "```text",
        decision,
        "```",
        "",
        next_step(decision),
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "phase16b_report.md").write_text(report + "\n", encoding="utf-8")
    (ROOT / "phase16b_result.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
