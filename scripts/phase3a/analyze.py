#!/usr/bin/env python3
"""Phase 3A Gate 1 / Gate 2 metrics, paired tests, and Markdown report."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/qwen_eval"))
sys.path.insert(0, str(ROOT / "scripts/phase2b"))
from score_fast import score_amber, score_mmrel, score_rbench  # noqa: E402
from score_yesno import gold_yes, load_json_or_jsonl, pred_yes  # noqa: E402

OUT = ROOT / "eval_results/qwen/phase3a"
FAST = ROOT / "eval_results/qwen/phase2b/fast_subsets"
N_BOOT = 2000
SEED = 42
RUNS = ("s534", "s1500", "s3000", "s3000_stepmatched")
HIST = {
    "base": ROOT / "eval_results/qwen/phase2/metrics/base_heldout.json",
    "v2b_2ep": ROOT / "eval_results/qwen/phase2b/metrics/v2b_2ep_heldout.json",
}
BASE_FAST = {
    "rbench": ROOT / "eval_results/qwen/rbench/answers/qwen25vl3b_base_image-level.jsonl",
    "mmrel_adv": ROOT / "eval_results/qwen/mmrel/answers/qwen25vl3b_base_mmrel_adv.jsonl",
    "amber_dr": ROOT / "eval_results/qwen/amber/answers/qwen25vl3b_base_amber_dr.jsonl",
}
V2B_FAST = {
    "rbench": ROOT / "eval_results/qwen/phase2b/answers/v2b_2ep_rbench_fast.jsonl",
    "mmrel_adv": ROOT / "eval_results/qwen/phase2b/answers/v2b_2ep_mmrel_adv_fast.jsonl",
    "amber_dr": ROOT / "eval_results/qwen/phase2b/answers/v2b_2ep_amber_dr_fast.jsonl",
}


def loadj(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def qid_aliases(qid):
    yield qid
    if isinstance(qid, int):
        yield str(qid)
    elif isinstance(qid, str) and qid.isdigit():
        yield int(qid)


def answer_preds(path: Path) -> dict:
    amap = {}
    if not path.exists():
        return amap
    for row in load_json_or_jsonl(path):
        qid = row.get("question_id", row.get("id"))
        text = row.get("text") or row.get("response") or ""
        pred = 1 if pred_yes(text) else 0
        for key in qid_aliases(qid):
            amap[key] = pred
    return amap


def lookup(amap: dict, qid):
    for key in qid_aliases(qid):
        if key in amap:
            return amap[key]
    return None


def mcnemar(a_ok: list[int], b_ok: list[int]) -> dict:
    b = c = 0
    for x, y in zip(a_ok, b_ok):
        if x and not y:
            b += 1
        elif (not x) and y:
            c += 1
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2_cc": 0.0, "p_value": 1.0, "n_discordant": 0}
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2))
    return {"b": b, "c": c, "chi2_cc": chi2, "p_value": p, "n_discordant": n}


def bootstrap_delta(a_ok: list[int], b_ok: list[int], n_boot=N_BOOT, seed=SEED) -> dict:
    rng = random.Random(seed)
    n = len(a_ok)
    if not n:
        return {"delta": 0.0, "ci95": [0.0, 0.0], "n": 0}
    acc_a = sum(a_ok) / n
    acc_b = sum(b_ok) / n
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a_ok[i] for i in idx) / n
        db = sum(b_ok[i] for i in idx) / n
        deltas.append(db - da)
    deltas.sort()
    return {
        "a_acc": acc_a,
        "b_acc": acc_b,
        "delta": acc_b - acc_a,
        "ci95": [deltas[int(0.025 * n_boot)], deltas[int(0.975 * n_boot)]],
        "n": n,
        "n_boot": n_boot,
    }


def item_ok_lists(questions: list[dict], pred_a: dict, pred_b: dict):
    a_ok, b_ok = [], []
    for q in questions:
        qid = q.get("question_id", q.get("id"))
        pa, pb = lookup(pred_a, qid), lookup(pred_b, qid)
        if pa is None or pb is None:
            continue
        gold = 1 if gold_yes(q.get("label")) else 0
        a_ok.append(int(pa == gold))
        b_ok.append(int(pb == gold))
    return a_ok, b_ok


def confusion(questions: list[dict], preds: dict) -> dict:
    tp = fp = tn = fn = 0
    for q in questions:
        qid = q.get("question_id", q.get("id"))
        p = lookup(preds, qid)
        if p is None:
            continue
        g = 1 if gold_yes(q.get("label")) else 0
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif (not p) and (not g):
            tn += 1
        else:
            fn += 1
    n = tp + fp + tn + fn
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": n,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": (tp + tn) / n if n else 0.0,
        "Recall": rec,
        "Precision": prec,
        "F1": f1,
        "Yes_ratio": (tp + fp) / n if n else 0.0,
    }


def pct(x) -> str:
    if x is None:
        return ""
    return f"{100.0 * float(x):.2f}"


def heldout_row(name: str, blob: dict | None) -> str:
    if not blob:
        return f"| {name} | | | | | | | |"
    a = blob["all"]
    pos = blob.get("positive") or {}
    rnd = blob.get("random_negative") or {}
    hard = blob.get("hard_negative") or {}
    return (
        f"| {name} | {pct(a.get('Accuracy'))} | {pct(pos.get('Accuracy'))} | "
        f"{pct(rnd.get('Accuracy'))} | {pct(hard.get('Accuracy'))} | "
        f"{pct(hard.get('hard_fp_rate'))} | {pct(a.get('Yes_ratio'))} | {a.get('n','')} |"
    )


def go_nogo(blob: dict) -> str:
    fast = blob.get("fast") or {}
    s534 = fast.get("s534") or {}
    s3000 = fast.get("s3000") or {}
    base = fast.get("base") or {}
    lines = ["## Go / No-Go", ""]
    if not (s534 and s3000 and base):
        lines.append("结果不完整，暂不判定。")
        return "\n".join(lines)

    def acc(run, bench):
        rec = (fast.get(run) or {}).get(bench) or {}
        if bench == "rbench":
            return rec.get("Accuracy")
        return rec.get("Accuracy")

    def cm(run, bench):
        return ((blob.get("fast_cm") or {}).get(run) or {}).get(bench) or {}

    benches = ("rbench", "mmrel_adv", "amber_dr")
    deltas = []
    healthy = 0
    bias = 0
    for b in benches:
        a0 = acc("s534", b)
        a1 = acc("s3000", b)
        if a0 is None or a1 is None:
            continue
        d = 100 * (a1 - a0)
        deltas.append((b, d))
        c0, c1 = cm("s534", b), cm("s3000", b)
        fp_ok = c1.get("FP", 0) <= c0.get("FP", 0)
        fn_bad = c1.get("FN", 0) > c0.get("FN", 0) + max(3, 0.15 * max(c0.get("FN", 1), 1))
        if d >= 0.5 and fp_ok and not fn_bad:
            healthy += 1
        if fp_ok and fn_bad and d < 0.5:
            bias += 1

    n_ge_1 = sum(1 for _b, d in deltas if d >= 1.0)
    n_ge_05 = sum(1 for _b, d in deltas if d >= 0.5)
    s3000_vs_s534 = [d for _b, d in deltas]
    mean_d = sum(s3000_vs_s534) / len(s3000_vs_s534) if s3000_vs_s534 else 0.0

    if n_ge_1 >= 2 and healthy >= 2:
        verdict = "Go A：scale 增大后外部 discrimination 明显提升。"
    elif n_ge_05 >= 3 and healthy >= 2:
        verdict = "Weak Go：方向一致但幅度有限，可对最优 checkpoint 跑 full bench 验证。"
    elif bias >= 2:
        verdict = "No-Go B：scale 越大 No-bias 越强（FP 降但 FN 大涨）。"
    else:
        # vs base
        vs_base = []
        for b in benches:
            a0 = acc("base", b)
            a1 = acc("s3000", b)
            if a0 is not None and a1 is not None:
                vs_base.append(100 * (a1 - a0))
        if vs_base and max(abs(x) for x in vs_base) < 0.5:
            verdict = "No-Go A：custom held-out 可能提升，但 external 仍 ≈ Base。"
        else:
            verdict = f"暂判观察：S3000-S534 平均 ΔAcc={mean_d:.2f} pp。对照计划第 24 节人工确认。"
    lines.append(verdict)
    lines.append("")
    lines.append("S3000 − S534 ΔAcc (pp): " + ", ".join(f"{b}={d:+.2f}" for b, d in deltas))
    return "\n".join(lines)


def main() -> None:
    rbench_q = load_json_or_jsonl(FAST / "rbench_questions.jsonl")
    mmrel_q = load_json_or_jsonl(FAST / "mmrel_adv_questions.jsonl")
    blob = {"heldout": {}, "train": {}, "fast": {}, "fast_cm": {}, "paired": {}}

    for name, path in HIST.items():
        blob["heldout"][name] = loadj(path)

    for run in RUNS:
        blob["heldout"][run] = loadj(OUT / "metrics" / f"{run}_heldout.json")
        blob["train"][run] = loadj(OUT / "metrics" / f"{run}_train.json")

    def add_fast(name: str, paths: dict):
        rec = {}
        cm = {}
        if paths["rbench"].exists():
            rec["rbench"] = score_rbench(paths["rbench"])
            cm["rbench"] = confusion(rbench_q, answer_preds(paths["rbench"]))
        if paths["mmrel_adv"].exists():
            rec["mmrel_adv"] = score_mmrel(paths["mmrel_adv"])
            cm["mmrel_adv"] = confusion(mmrel_q, answer_preds(paths["mmrel_adv"]))
        if paths["amber_dr"].exists():
            rec["amber_dr"] = score_amber(paths["amber_dr"])
        blob["fast"][name] = rec
        blob["fast_cm"][name] = cm

    add_fast("base", BASE_FAST)
    add_fast("v2b_2ep", V2B_FAST)
    for run in RUNS:
        add_fast(
            run,
            {
                "rbench": OUT / "answers" / f"{run}_rbench_fast.jsonl",
                "mmrel_adv": OUT / "answers" / f"{run}_mmrel_adv_fast.jsonl",
                "amber_dr": OUT / "answers" / f"{run}_amber_dr_fast.jsonl",
            },
        )

    preds = {
        "base": {b: answer_preds(p) for b, p in BASE_FAST.items()},
        "v2b_2ep": {b: answer_preds(p) for b, p in V2B_FAST.items()},
    }
    for run in RUNS:
        preds[run] = {
            "rbench": answer_preds(OUT / "answers" / f"{run}_rbench_fast.jsonl"),
            "mmrel_adv": answer_preds(OUT / "answers" / f"{run}_mmrel_adv_fast.jsonl"),
        }

    pairs = [
        ("s534", "s1500"),
        ("s534", "s3000"),
        ("s1500", "s3000"),
        ("base", "s534"),
        ("base", "s3000"),
        ("s534", "s3000_stepmatched"),
    ]
    for bench, questions in (("rbench", rbench_q), ("mmrel_adv", mmrel_q)):
        blob["paired"][bench] = {}
        for a, b in pairs:
            pa, pb = preds.get(a, {}).get(bench, {}), preds.get(b, {}).get(bench, {})
            a_ok, b_ok = item_ok_lists(questions, pa, pb)
            blob["paired"][bench][f"{b}_vs_{a}"] = {
                "mcnemar": mcnemar(a_ok, b_ok),
                "bootstrap": bootstrap_delta(a_ok, b_ok),
            }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics").mkdir(exist_ok=True)
    slim_path = OUT / "metrics" / "phase3a_analysis.json"

    def drop_folds(obj):
        if isinstance(obj, dict):
            return {k: drop_folds(v) for k, v in obj.items() if k != "folds"}
        if isinstance(obj, list):
            return [drop_folds(x) for x in obj]
        return obj

    slim = drop_folds(blob)
    slim_path.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")

    md = []
    md.append("# Phase 3A 结果：Structured Relation Verification Scaling")
    md.append("")
    md.append("嵌套规模 S534 ⊂ S1500 ⊂ S3000，均为 1:1 Positive/Negative，官方 prompt 评测冻结 fast subset。")
    md.append("")
    md.append("## Gate 1：冻结 393 held-out verification")
    md.append("")
    md.append("| Model | Acc | Pos Acc | Random Neg Acc | Hard Neg Acc | Hard FP | Yes | n |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("base", "v2b_2ep", *RUNS):
        md.append(heldout_row(name, blob["heldout"].get(name)))
    md.append("")
    md.append("## Gate 2：冻结 external fast subsets（官方 prompt）")
    md.append("")
    md.append("| Scale | R-Bench Acc | R-Bench F1 | MMRel Acc | MMRel F1 | AMBER Acc |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for name in ("base", "v2b_2ep", *RUNS):
        rec = blob["fast"].get(name) or {}
        rb, mm, am = rec.get("rbench") or {}, rec.get("mmrel_adv") or {}, rec.get("amber_dr") or {}
        md.append(
            f"| {name} | {pct(rb.get('Accuracy'))} | {pct(rb.get('F1'))} | "
            f"{pct(mm.get('Accuracy'))} | {pct(mm.get('F1'))} | {pct(am.get('Accuracy'))} |"
        )
    md.append("")
    md.append("### Confusion（unique questions）")
    md.append("")
    md.append("| Model | Bench | Acc | FP | FN | Recall | Yes |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for name in ("base", "v2b_2ep", *RUNS):
        for bench in ("rbench", "mmrel_adv"):
            c = (blob["fast_cm"].get(name) or {}).get(bench) or {}
            if not c:
                continue
            md.append(
                f"| {name} | {bench} | {pct(c.get('Accuracy'))} | {c.get('FP','')} | "
                f"{c.get('FN','')} | {pct(c.get('Recall'))} | {pct(c.get('Yes_ratio'))} |"
            )
    md.append("")
    md.append("## Paired tests（same frozen items）")
    md.append("")
    for bench, recs in blob["paired"].items():
        md.append(f"### {bench}")
        md.append("")
        md.append("| Contrast | ΔAcc | 95% CI | McNemar p | n |")
        md.append("|---|---:|---|---:|---:|")
        for k, v in recs.items():
            bs = v["bootstrap"]
            p = v["mcnemar"]["p_value"]
            ci = bs["ci95"]
            md.append(
                f"| {k} | {100*bs['delta']:+.2f} pp | "
                f"[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] | {p:.4g} | {bs['n']} |"
            )
        md.append("")
    md.append(go_nogo(blob))
    md.append("")
    md.append("旧 V2B-2ep 只作 historical reference，不进入 scaling curve。S3000-stepmatched 只作 diagnostic。")
    md.append("")
    text = "\n".join(md) + "\n"
    (OUT / "phase3a_report.md").write_text(text, encoding="utf-8")
    (ROOT / "phase3a_result.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {slim_path}")
    print(f"wrote {OUT / 'phase3a_report.md'}")


if __name__ == "__main__":
    main()
