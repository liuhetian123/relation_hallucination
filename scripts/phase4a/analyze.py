#!/usr/bin/env python3
"""Phase 4A Old vs Clean Negative sanity-check report."""

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

OUT = ROOT / "eval_results/qwen/phase4a"
P3 = ROOT / "eval_results/qwen/phase3a"
FAST = ROOT / "eval_results/qwen/phase2b/fast_subsets"
N_BOOT = 2000
SEED = 42
RUNS = ("old", "clean")
HIST = {
    "base": ROOT / "eval_results/qwen/phase2/metrics/base_heldout.json",
    "v2b_2ep": ROOT / "eval_results/qwen/phase2b/metrics/v2b_2ep_heldout.json",
    "s3000": P3 / "metrics/s3000_heldout.json",
}
BASE_FAST = {
    "rbench": ROOT / "eval_results/qwen/rbench/answers/qwen25vl3b_base_image-level.jsonl",
    "mmrel_adv": ROOT / "eval_results/qwen/mmrel/answers/qwen25vl3b_base_mmrel_adv.jsonl",
    "amber_dr": ROOT / "eval_results/qwen/amber/answers/qwen25vl3b_base_amber_dr.jsonl",
}
S3000_FAST = {
    "rbench": P3 / "answers/s3000_rbench_fast.jsonl",
    "mmrel_adv": P3 / "answers/s3000_mmrel_adv_fast.jsonl",
    "amber_dr": P3 / "answers/s3000_amber_dr_fast.jsonl",
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


def train_neg(blob: dict | None) -> dict:
    if not blob:
        return {}
    return blob.get("clean_negative") or blob.get("random_negative") or {}


def case_verdict(blob: dict) -> str:
    fast = blob.get("fast") or {}
    cm = blob.get("fast_cm") or {}
    old_f, clean_f = fast.get("old") or {}, fast.get("clean") or {}
    if not old_f or not clean_f:
        return "结果不完整，暂不判定。"
    benches = ("rbench", "mmrel_adv", "amber_dr")
    deltas = []
    for b in benches:
        a0 = (old_f.get(b) or {}).get("Accuracy")
        a1 = (clean_f.get(b) or {}).get("Accuracy")
        if a0 is None or a1 is None:
            continue
        deltas.append((b, 100 * (a1 - a0)))
    if not deltas:
        return "外部 fast subset 结果不完整，暂不判定。"
    mean_d = sum(d for _b, d in deltas) / len(deltas)
    n_up = sum(1 for _b, d in deltas if d >= 0.5)
    n_down = sum(1 for _b, d in deltas if d <= -0.5)
    fp_fn = []
    for b in ("rbench", "mmrel_adv"):
        c0, c1 = (cm.get("old") or {}).get(b) or {}, (cm.get("clean") or {}).get(b) or {}
        if c0 and c1:
            fp_fn.append((b, c1.get("FP", 0) - c0.get("FP", 0), c1.get("FN", 0) - c0.get("FN", 0)))
    train = blob.get("train") or {}
    old_tr, clean_tr = train.get("old") or {}, train.get("clean") or {}
    old_neg = (train_neg(old_tr) or {}).get("Accuracy")
    clean_neg = (train_neg(clean_tr) or {}).get("Accuracy")
    lines = []
    if n_up >= 2 and mean_d >= 0.5:
        lines.append("**Case A：Clean Negative 明显更好。** 外部 relation benchmark 相对 Old 上升，后续 Phase 4A 使用 Clean-S3000。")
    elif n_down >= 2 and mean_d <= -0.5:
        lines.append("**Case C：Clean Negative 使指标下降。** 新负例更 hard；先看 train Positive/Negative Acc 与 held-out negative Acc，必要时加 exposure，不立刻退回 Dirty Negative。")
    else:
        lines.append("**Case B：Text-only shortcut 已降，但 external ≈ Old。** 数据更干净，旧 Negative 不是外部 transfer 的主要瓶颈。后续仍建议使用 Clean-S3000。")
    lines.append("")
    lines.append("Clean − Old ΔAcc (pp): " + ", ".join(f"{b}={d:+.2f}" for b, d in deltas) + f"；mean={mean_d:+.2f}")
    if fp_fn:
        lines.append("FP/FN Δ (Clean−Old): " + ", ".join(f"{b} FP={dfp:+d} FN={dfn:+d}" for b, dfp, dfn in fp_fn))
    if old_neg is not None and clean_neg is not None:
        lines.append(f"Train negative Acc: old={100*old_neg:.2f}  clean={100*clean_neg:.2f}")
    return "\n".join(lines)


def main() -> None:
    rbench_q = load_json_or_jsonl(FAST / "rbench_questions.jsonl")
    mmrel_q = load_json_or_jsonl(FAST / "mmrel_adv_questions.jsonl")
    blob = {"heldout": {}, "train": {}, "fast": {}, "fast_cm": {}, "paired": {}}
    text = loadj(ROOT / "data/phase4a/text_only_metrics.json")
    build = loadj(ROOT / "data/phase4a/train_build_stats.json")

    for name, path in HIST.items():
        blob["heldout"][name] = loadj(path)
    for run in RUNS:
        blob["heldout"][run] = loadj(OUT / "metrics" / f"{run}_heldout.json")
        blob["train"][run] = loadj(OUT / "metrics" / f"{run}_train.json")

    def add_fast(name: str, paths: dict):
        rec, cm = {}, {}
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
    add_fast("s3000", S3000_FAST)
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
        "s3000": {b: answer_preds(p) for b, p in S3000_FAST.items()},
    }
    for run in RUNS:
        preds[run] = {
            "rbench": answer_preds(OUT / "answers" / f"{run}_rbench_fast.jsonl"),
            "mmrel_adv": answer_preds(OUT / "answers" / f"{run}_mmrel_adv_fast.jsonl"),
        }
    pairs = [("old", "clean"), ("base", "old"), ("base", "clean"), ("s3000", "clean"), ("s3000", "old")]
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
    slim_path = OUT / "metrics" / "phase4a_train_analysis.json"

    def drop_folds(obj):
        if isinstance(obj, dict):
            return {k: drop_folds(v) for k, v in obj.items() if k != "folds"}
        if isinstance(obj, list):
            return [drop_folds(x) for x in obj]
        return obj

    slim = drop_folds(blob)
    slim["text_only"] = text
    slim["train_build"] = build
    slim_path.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")

    n_pairs = (build or {}).get("n_pairs", 2156)
    md = []
    md.append("# Phase 4A 训练结果：Old vs Clean Negative")
    md.append("")
    md.append(
        f"Qwen2.5-VL-3B LoRA（r=16, α=32），1 epoch，同一 {n_pairs} 张图 / 1:1 Positive–Negative。"
        "Old 用 Phase 3A dirty random negative；Clean 用 InternVL counterfactual negative。"
        "s3000 是 Phase 3A 全量 3000 对 dirty 训练，仅作 historical reference（图集更大）。"
    )
    md.append("")
    md.append("人工 QC 已于 2026-09-16 跳过。")
    md.append("")
    if text:
        md.append("## Text-only shortcut（数据侧，训练前）")
        md.append("")
        md.append(
            f"- Old text-only Acc：{pct((text.get('paired_old_textonly_acc') or {}).get('acc'))} "
            f"(n={(text.get('paired_old_textonly_acc') or {}).get('n')})"
        )
        md.append(
            f"- Clean text-only Acc：{pct((text.get('paired_clean_textonly_acc') or {}).get('acc'))} "
            f"(n={(text.get('paired_clean_textonly_acc') or {}).get('n')})"
        )
        md.append("")
    md.append("## 训练自检")
    md.append("")
    md.append("| Model | Acc | Pos Acc | Neg Acc | Yes | n |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for name in RUNS:
        rec = blob["train"].get(name) or {}
        allm = rec.get("all") or {}
        pos = rec.get("positive") or {}
        neg = train_neg(rec)
        md.append(
            f"| {name} | {pct(allm.get('Accuracy'))} | {pct(pos.get('Accuracy'))} | "
            f"{pct(neg.get('Accuracy'))} | {pct(allm.get('Yes_ratio'))} | {allm.get('n','')} |"
        )
    md.append("")
    md.append("## Gate 1：冻结 393 held-out verification")
    md.append("")
    md.append("| Model | Acc | Pos Acc | Random Neg Acc | Hard Neg Acc | Hard FP | Yes | n |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("base", "s3000", *RUNS):
        md.append(heldout_row(name, blob["heldout"].get(name)))
    md.append("")
    md.append("## Gate 2：冻结 external fast subsets（官方 prompt）")
    md.append("")
    md.append("| Model | R-Bench Acc | R-Bench F1 | MMRel Acc | MMRel F1 | AMBER Acc |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for name in ("base", "s3000", *RUNS):
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
    for name in ("base", "s3000", *RUNS):
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
    md.append("## 计划第 15 节判定")
    md.append("")
    md.append(case_verdict(blob))
    md.append("")
    text_md = "\n".join(md) + "\n"
    (OUT / "phase4a_train_report.md").write_text(text_md, encoding="utf-8")
    (ROOT / "phase4a_train_result.md").write_text(text_md, encoding="utf-8")
    print(text_md)
    print(f"wrote {slim_path}")
    print(f"wrote {OUT / 'phase4a_train_report.md'}")


if __name__ == "__main__":
    main()
