#!/usr/bin/env python3
"""Score Phase 1.5 MCQ predictions: accuracy, transitions, agreement, McNemar, bootstrap."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

LETTERS = ("A", "B", "C", "D")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_questions(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("questions") or data.get("data")


def index_preds(rows: list[dict]) -> dict:
    return {r["question_id"]: r for r in rows}


def acc_stats(rows: list[dict], questions: list[dict]) -> dict:
    gold = {q["id"]: q["answer"] for q in questions}
    correct = wrong = invalid = 0
    margins = []
    parsed = {}
    for q in questions:
        rec = rows.get(q["id"]) if isinstance(rows, dict) else None
        if rec is None:
            invalid += 1
            parsed[q["id"]] = "invalid"
            continue
        choice = rec.get("parsed") or "invalid"
        parsed[q["id"]] = choice
        if choice == "invalid":
            invalid += 1
        elif choice == gold[q["id"]]:
            correct += 1
        else:
            wrong += 1
        if rec.get("gt_margin") is not None:
            margins.append(rec["gt_margin"])
    n = len(questions)
    return {
        "n": n,
        "correct": correct,
        "wrong": wrong,
        "invalid": invalid,
        "accuracy": correct / n if n else 0.0,
        "mean_gt_margin": float(np.mean(margins)) if margins else None,
        "parsed": parsed,
        "ok": np.array([parsed[q["id"]] == gold[q["id"]] for q in questions], dtype=bool),
        "labels": [parsed[q["id"]] for q in questions],
    }


def mcnemar(a_ok: np.ndarray, b_ok: np.ndarray) -> dict:
    """McNemar test: b better than a is n01 vs n10."""
    n01 = int((~a_ok & b_ok).sum())  # a wrong, b correct
    n10 = int((a_ok & ~b_ok).sum())
    n = n01 + n10
    if n == 0:
        p = 1.0
        chi2 = 0.0
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / n  # continuity correction
        # two-sided exact binomial if n small; chi2 otherwise
        # Survival of chi2_1
        p = math.erfc(math.sqrt(chi2 / 2))
    return {"n01_a_wrong_b_correct": n01, "n10_a_correct_b_wrong": n10, "chi2": chi2, "p": p}


def bootstrap_diff(a_ok: np.ndarray, b_ok: np.ndarray, seed: int = 42, iters: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    n = len(a_ok)
    diffs = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        diffs[i] = b_ok[idx].mean() - a_ok[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": float(b_ok.mean() - a_ok.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def agreement(a: list[str], b: list[str]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a) if a else 0.0


def decide(base_acc: float, exp_acc: float, typed_acc: float, agree: dict) -> str:
    d_e = exp_acc - base_acc
    d_t = typed_acc - base_acc
    mean_agree = np.mean(list(agree.values()))
    # Clear gain vs random-ish movement
    if (d_e >= 0.08 and d_t >= 0.08) or (
        (d_e >= 0.10 or d_t >= 0.10) and min(d_e, d_t) >= 0.04
    ):
        return "Phase 1.5-A:\nTraining clearly effective."
    if (d_e >= 0.08) != (d_t >= 0.08) and abs(d_e - d_t) >= 0.08:
        return "Phase 1.5-C:\nInconsistent adaptation; inspect pipeline/checkpoints."
    if abs(d_e) < 0.04 and abs(d_t) < 0.04 and mean_agree >= 0.90:
        return "Phase 1.5-B:\nTraining intervention too weak / no measurable adaptation."
    if abs(d_e) < 0.04 and abs(d_t) < 0.04:
        return "Phase 1.5-B:\nTraining intervention too weak / no measurable adaptation."
    if (d_e > 0.05 and d_t < 0.02) or (d_t > 0.05 and d_e < 0.02):
        return "Phase 1.5-C:\nInconsistent adaptation; inspect pipeline/checkpoints."
    if d_e >= 0.05 and d_t >= 0.05:
        return "Phase 1.5-A:\nTraining clearly effective."
    return "Phase 1.5-B:\nTraining intervention too weak / no measurable adaptation."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--explicit", required=True)
    parser.add_argument("--typed", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    preds = {
        "Base": index_preds(load_jsonl(Path(args.base))),
        "Explicit-1k": index_preds(load_jsonl(Path(args.explicit))),
        "Typed-1k": index_preds(load_jsonl(Path(args.typed))),
    }
    stats = {name: acc_stats(rows, questions) for name, rows in preds.items()}

    paired_rows = []
    for q in questions:
        paired_rows.append(
            {
                "id": q["id"],
                "image": q["image"],
                "gold": q["answer"],
                "gt_relation": q["gt_relation"],
                "base": stats["Base"]["parsed"][q["id"]],
                "explicit": stats["Explicit-1k"]["parsed"][q["id"]],
                "typed": stats["Typed-1k"]["parsed"][q["id"]],
            }
        )

    b_ok, e_ok, t_ok = stats["Base"]["ok"], stats["Explicit-1k"]["ok"], stats["Typed-1k"]["ok"]
    transitions = {
        "Base wrong → Explicit correct": int((~b_ok & e_ok).sum()),
        "Base correct → Explicit wrong": int((b_ok & ~e_ok).sum()),
        "Base wrong → Typed correct": int((~b_ok & t_ok).sum()),
        "Base correct → Typed wrong": int((b_ok & ~t_ok).sum()),
    }
    agree = {
        "Base / Explicit": agreement(stats["Base"]["labels"], stats["Explicit-1k"]["labels"]),
        "Base / Typed": agreement(stats["Base"]["labels"], stats["Typed-1k"]["labels"]),
        "Explicit / Typed": agreement(stats["Explicit-1k"]["labels"], stats["Typed-1k"]["labels"]),
    }
    tests = {
        "Explicit vs Base": {
            "mcnemar": mcnemar(b_ok, e_ok),
            "bootstrap_acc_diff": bootstrap_diff(b_ok, e_ok),
        },
        "Typed vs Base": {
            "mcnemar": mcnemar(b_ok, t_ok),
            "bootstrap_acc_diff": bootstrap_diff(b_ok, t_ok),
        },
        "Typed vs Explicit": {
            "mcnemar": mcnemar(e_ok, t_ok),
            "bootstrap_acc_diff": bootstrap_diff(e_ok, t_ok),
        },
    }

    summary = {
        "accuracy": {
            name: {
                "correct": s["correct"],
                "wrong": s["wrong"],
                "invalid": s["invalid"],
                "accuracy": s["accuracy"],
                "mean_gt_margin": s["mean_gt_margin"],
            }
            for name, s in stats.items()
        },
        "deltas": {
            "Explicit - Base": stats["Explicit-1k"]["accuracy"] - stats["Base"]["accuracy"],
            "Typed - Base": stats["Typed-1k"]["accuracy"] - stats["Base"]["accuracy"],
            "Typed - Explicit": stats["Typed-1k"]["accuracy"] - stats["Explicit-1k"]["accuracy"],
        },
        "transitions": transitions,
        "agreement": agree,
        "tests": tests,
    }
    decision = decide(
        stats["Base"]["accuracy"],
        stats["Explicit-1k"]["accuracy"],
        stats["Typed-1k"]["accuracy"],
        agree,
    )
    summary["decision"] = decision

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "paired_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in paired_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def pct(x):
        return f"{100 * x:.1f}%"

    lines = [
        "# Phase 1.5 Held-out RelSim Relation MCQ",
        "",
        "200 道 held-out RelSim 关系四选一。三个模型使用同一套题目、同一 prompt、greedy decoding。",
        "",
        "## Accuracy",
        "",
        "| Model | Correct | Wrong | Invalid | Accuracy | Mean GT Margin |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("Base", "Explicit-1k", "Typed-1k"):
        s = stats[name]
        margin = "—" if s["mean_gt_margin"] is None else f"{s['mean_gt_margin']:.3f}"
        lines.append(
            f"| {name} | {s['correct']} | {s['wrong']} | {s['invalid']} | {pct(s['accuracy'])} | {margin} |"
        )
    lines += [
        "",
        f"- Explicit − Base = {pct(summary['deltas']['Explicit - Base'])}",
        f"- Typed − Base = {pct(summary['deltas']['Typed - Base'])}",
        f"- Typed − Explicit = {pct(summary['deltas']['Typed - Explicit'])}",
        "",
        "随机水平：25%。",
        "",
        "## Paired transitions",
        "",
        "| Transition | Count |",
        "|---|---:|",
    ]
    for k, v in transitions.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Prediction agreement",
        "",
        "| Pair | Agreement |",
        "|---|---:|",
    ]
    for k, v in agree.items():
        lines.append(f"| {k} | {pct(v)} |")
    lines += ["", "## Explicit vs Base / Typed vs Base", ""]
    for pair, rec in tests.items():
        bd = rec["bootstrap_acc_diff"]
        mc = rec["mcnemar"]
        lines.append(
            f"- **{pair}**: ΔAcc={pct(bd['diff'])} "
            f"(bootstrap 95% CI [{pct(bd['ci95_low'])}, {pct(bd['ci95_high'])}]); "
            f"McNemar n01={mc['n01_a_wrong_b_correct']}, n10={mc['n10_a_correct_b_wrong']}, p={mc['p']:.4f}"
        )
    lines += ["", "## Decision", "", "```text", decision, "```", ""]
    report = "\n".join(lines)
    (out_dir / "phase15_report.md").write_text(report + "\n", encoding="utf-8")
    Path(args.out_dir).joinpath("..")
    # Also write a repo-root copy via caller; print path here.
    print(report)
    print(f"\nWrote {out_dir / 'metrics.json'}")
    print(f"Wrote {out_dir / 'phase15_report.md'}")


if __name__ == "__main__":
    main()
