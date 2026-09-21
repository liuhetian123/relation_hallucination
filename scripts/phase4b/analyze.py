#!/usr/bin/env python3
"""Phase 4B Task A: seen/unseen relation stratification on frozen predictions."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/qwen_eval"))
from score_yesno import gold_yes, load_json_or_jsonl, pred_yes  # noqa: E402

OUT = ROOT / "eval_results/qwen/phase4b"
FAST = ROOT / "eval_results/qwen/phase2b/fast_subsets"
N_BOOT = 10000
SEED = 42
PUBLISHED = {
    "base": {"rbench": 83.11, "mmrel_adv": 69.01},
    "s3000": {"rbench": 83.58, "mmrel_adv": 71.09},
    "old": {"rbench": 83.26, "mmrel_adv": 70.31},
    "clean": {"rbench": 82.95, "mmrel_adv": 70.83},
}
ANSWERS = {
    "base": {
        "rbench": ROOT / "eval_results/qwen/rbench/answers/qwen25vl3b_base_image-level.jsonl",
        "mmrel_adv": ROOT / "eval_results/qwen/mmrel/answers/qwen25vl3b_base_mmrel_adv.jsonl",
    },
    "s3000": {
        "rbench": ROOT / "eval_results/qwen/phase3a/answers/s3000_rbench_fast.jsonl",
        "mmrel_adv": ROOT / "eval_results/qwen/phase3a/answers/s3000_mmrel_adv_fast.jsonl",
    },
    "old": {
        "rbench": ROOT / "eval_results/qwen/phase4a/answers/old_rbench_fast.jsonl",
        "mmrel_adv": ROOT / "eval_results/qwen/phase4a/answers/old_mmrel_adv_fast.jsonl",
    },
    "clean": {
        "rbench": ROOT / "eval_results/qwen/phase4a/answers/clean_rbench_fast.jsonl",
        "mmrel_adv": ROOT / "eval_results/qwen/phase4a/answers/clean_mmrel_adv_fast.jsonl",
    },
}
CONTRASTS = (
    ("base", "s3000", "old"),
    ("base", "old", "old"),
    ("base", "clean", "clean"),
    ("old", "clean", "clean"),
)
LEVELS = ("exact", "lemma", "synonym-family")


def qid_aliases(qid):
    yield qid
    if isinstance(qid, int):
        yield str(qid)
    elif isinstance(qid, str) and qid.isdigit():
        yield int(qid)


def answer_preds(path: Path) -> dict:
    amap = {}
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


def loadj(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gold_bin(label) -> int:
    return 1 if gold_yes(label) else 0


def confusion_from_ok(preds: list[int], golds: list[int]) -> dict:
    tp = fp = tn = fn = 0
    for p, g in zip(preds, golds):
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
        return {"delta": 0.0, "ci95": [0.0, 0.0], "n": 0, "a_acc": 0.0, "b_acc": 0.0}
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


def bootstrap_did(seen_a, seen_b, unseen_a, unseen_b, n_boot=N_BOOT, seed=SEED) -> dict:
    rng = random.Random(seed)
    ns, nu = len(seen_a), len(unseen_a)
    if not ns or not nu:
        return {"did": None, "ci95": [None, None], "n_seen": ns, "n_unseen": nu}
    d_seen = (sum(seen_b) / ns) - (sum(seen_a) / ns)
    d_unseen = (sum(unseen_b) / nu) - (sum(unseen_a) / nu)
    draws = []
    for _ in range(n_boot):
        is_ = [rng.randrange(ns) for _ in range(ns)]
        iu = [rng.randrange(nu) for _ in range(nu)]
        ds = sum(seen_b[i] for i in is_) / ns - sum(seen_a[i] for i in is_) / ns
        du = sum(unseen_b[i] for i in iu) / nu - sum(unseen_a[i] for i in iu) / nu
        draws.append(ds - du)
    draws.sort()
    return {
        "did": d_seen - d_unseen,
        "delta_seen": d_seen,
        "delta_unseen": d_unseen,
        "ci95": [draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]],
        "n_seen": ns,
        "n_unseen": nu,
        "n_boot": n_boot,
    }


def seen_at(level: str, match: str) -> bool | None:
    if match == "unparsed":
        return None
    rank = {"exact": 0, "lemma": 1, "synonym-family": 2, "unseen": 3}
    cut = {"exact": 0, "lemma": 1, "synonym-family": 2}[level]
    return rank[match] <= cut


def layer_stats(items: list[dict], pred_a: dict, pred_b: dict, name_a: str, name_b: str) -> dict:
    a_ok, b_ok, golds, preds_a, preds_b = [], [], [], [], []
    missing = 0
    for it in items:
        pa, pb = lookup(pred_a, it["question_id"]), lookup(pred_b, it["question_id"])
        if pa is None or pb is None:
            missing += 1
            continue
        g = gold_bin(it["gold"])
        a_ok.append(int(pa == g))
        b_ok.append(int(pb == g))
        golds.append(g)
        preds_a.append(pa)
        preds_b.append(pb)
    rec = {
        "n": len(a_ok),
        "missing": missing,
        "low_power": len(a_ok) < 100,
        name_a: confusion_from_ok(preds_a, golds),
        name_b: confusion_from_ok(preds_b, golds),
        "delta": bootstrap_delta(a_ok, b_ok),
        "mcnemar": mcnemar(a_ok, b_ok),
    }
    return rec


def reconcile(questions: list[dict], preds: dict, published: float, label: str) -> float:
    golds, ps = [], []
    miss = 0
    for q in questions:
        p = lookup(preds, q.get("question_id", q.get("id")))
        if p is None:
            miss += 1
            continue
        golds.append(gold_bin(q.get("label")))
        ps.append(p)
    acc = 100.0 * confusion_from_ok(ps, golds)["Accuracy"]
    gap = abs(acc - published)
    print(f"[reconcile] {label} acc={acc:.2f} published={published:.2f} gap={gap:.3f} miss={miss}", flush=True)
    if gap > 0.05:
        raise SystemExit(f"Acc mismatch {label}: {acc:.2f} vs published {published:.2f} (>{0.05} pp)")
    return acc


def main() -> None:
    extracted = load_jsonl(OUT / "metrics/relation_extraction.jsonl")
    meta = loadj(OUT / "metrics/relation_extraction_meta.json")
    rbench_q = load_jsonl(FAST / "rbench_questions.jsonl")
    mmrel_q = load_jsonl(FAST / "mmrel_adv_questions.jsonl")
    by_bench = {
        "rbench": [r for r in extracted if r["benchmark"] == "rbench"],
        "mmrel_adv": [r for r in extracted if r["benchmark"] == "mmrel_adv"],
    }
    preds = {name: {b: answer_preds(p) for b, p in paths.items()} for name, paths in ANSWERS.items()}

    recon = {}
    for model in ("base", "s3000", "old", "clean"):
        recon[model] = {}
        recon[model]["rbench"] = reconcile(rbench_q, preds[model]["rbench"], PUBLISHED[model]["rbench"], f"{model}/rbench")
        recon[model]["mmrel_adv"] = reconcile(
            mmrel_q, preds[model]["mmrel_adv"], PUBLISHED[model]["mmrel_adv"], f"{model}/mmrel"
        )

    blob = {
        "reconcile": recon,
        "extraction_meta": meta,
        "amber_dr": "skipped: AMBER-dr-fast 几乎全是 'direct contact' 二元接触问句，无法抽取多样 relation。",
        "contrasts": {},
    }

    for bench, items in by_bench.items():
        blob["contrasts"][bench] = {}
        for name_a, name_b, vocab in CONTRASTS:
            match_key = "match_old" if vocab == "old" else "match_clean"
            pred_a, pred_b = preds[name_a][bench], preds[name_b][bench]
            contrast = {"vocab": vocab, "levels": {}}
            for level in LEVELS:
                seen_items, unseen_items, unparsed_items = [], [], []
                for it in items:
                    flag = seen_at(level, it[match_key])
                    if flag is None:
                        unparsed_items.append(it)
                    elif flag:
                        seen_items.append(it)
                    else:
                        unseen_items.append(it)
                seen = layer_stats(seen_items, pred_a, pred_b, name_a, name_b)
                unseen = layer_stats(unseen_items, pred_a, pred_b, name_a, name_b)
                unp = layer_stats(unparsed_items, pred_a, pred_b, name_a, name_b)
                sa, sb, ua, ub = [], [], [], []
                for it in seen_items:
                    pa, pb = lookup(pred_a, it["question_id"]), lookup(pred_b, it["question_id"])
                    if pa is None or pb is None:
                        continue
                    g = gold_bin(it["gold"])
                    sa.append(int(pa == g))
                    sb.append(int(pb == g))
                for it in unseen_items:
                    pa, pb = lookup(pred_a, it["question_id"]), lookup(pred_b, it["question_id"])
                    if pa is None or pb is None:
                        continue
                    g = gold_bin(it["gold"])
                    ua.append(int(pa == g))
                    ub.append(int(pb == g))
                contrast["levels"][level] = {
                    "seen": seen,
                    "unseen": unseen,
                    "unparsed": unp,
                    "did": bootstrap_did(sa, sb, ua, ub),
                }
            if bench == "mmrel_adv":
                contrast["domain"] = {}
                for domain in ("vg", "dalle"):
                    d_items = [it for it in items if it.get("domain") == domain]
                    contrast["domain"][domain] = {}
                    for level in ("lemma",):
                        seen_items = [it for it in d_items if seen_at(level, it[match_key]) is True]
                        unseen_items = [it for it in d_items if seen_at(level, it[match_key]) is False]
                        sa, sb, ua, ub = [], [], [], []
                        for bucket, accs in ((seen_items, (sa, sb)), (unseen_items, (ua, ub))):
                            for it in bucket:
                                pa, pb = lookup(pred_a, it["question_id"]), lookup(pred_b, it["question_id"])
                                if pa is None or pb is None:
                                    continue
                                g = gold_bin(it["gold"])
                                accs[0].append(int(pa == g))
                                accs[1].append(int(pb == g))
                        contrast["domain"][domain][level] = {
                            "seen": layer_stats(seen_items, pred_a, pred_b, name_a, name_b),
                            "unseen": layer_stats(unseen_items, pred_a, pred_b, name_a, name_b),
                            "did": bootstrap_did(sa, sb, ua, ub),
                        }
            blob["contrasts"][bench][f"{name_b}_vs_{name_a}"] = contrast

    # top relations in seen/unseen for lemma, clean vs base on rbench
    top = defaultdict(Counter)
    for it in by_bench["rbench"]:
        if not it["parse_ok"]:
            continue
        bucket = "seen" if seen_at("lemma", it["match_clean"]) else "unseen"
        top[bucket][it["relation_lemma"] or it["relation"]] += 1
    blob["rbench_clean_lemma_top"] = {
        k: c.most_common(15) for k, c in top.items()
    }

    outp = OUT / "metrics/phase4b_stratified_analysis.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
