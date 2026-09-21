#!/usr/bin/env python3
"""Score Official vs Training-Matched prompts; prompt sensitivity; McNemar; bootstrap."""

from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAST = ROOT / "eval_results/qwen/phase2b/fast_subsets"
OUT = ROOT / "eval_results/qwen/phase2b5"
N_BOOT = 2000
SEED = 42

RUNS = {
    "base": {
        "adapter": None,
        "official_rbench": ROOT / "eval_results/qwen/rbench/answers/qwen25vl3b_base_image-level.jsonl",
        "official_mmrel": ROOT / "eval_results/qwen/mmrel/answers/qwen25vl3b_base_mmrel_adv.jsonl",
    },
    "v2_old": {
        "adapter": "v2",
        "official_rbench": ROOT / "eval_results/qwen/rbench/answers/qwen25vl3b_v2_image-level.jsonl",
        "official_mmrel": ROOT / "eval_results/qwen/mmrel/answers/qwen25vl3b_v2_mmrel_adv.jsonl",
    },
    "v2b_2ep": {
        "adapter": "v2b_2ep",
        "official_rbench": ROOT / "eval_results/qwen/phase2b/answers/v2b_2ep_rbench_fast.jsonl",
        "official_mmrel": ROOT / "eval_results/qwen/phase2b/answers/v2b_2ep_mmrel_adv_fast.jsonl",
    },
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def qid_aliases(qid):
    yield qid
    if isinstance(qid, int):
        yield str(qid)
    elif isinstance(qid, str) and qid.isdigit():
        yield int(qid)


def answer_map(path: Path) -> dict:
    amap = {}
    for row in load_jsonl(path):
        qid = row.get("question_id", row.get("id"))
        for key in qid_aliases(qid):
            amap[key] = row
    return amap


def lookup(amap: dict, qid):
    for key in qid_aliases(qid):
        if key in amap:
            return amap[key]
    return None


def parse_pred(text: str, official_fallback: bool = False) -> str:
    """Yes / No / Invalid. Invalid is not force-mapped unless official_fallback."""
    t = (text or "").strip()
    if not t:
        return "Invalid"
    first = t.split("\n")[0].strip()
    if "." in first:
        first = first.split(".")[0].strip()
    first = first.replace(",", " ")
    m = re.match(r"^\s*(yes|no)\b", first, flags=re.I)
    if m:
        return "Yes" if m.group(1).lower() == "yes" else "No"
    if official_fallback:
        words = first.split()
        if any(w in {"No", "not", "no"} for w in words):
            return "No"
        return "Yes"
    return "Invalid"


def gold_yes(label) -> bool:
    s = str(label or "").strip().lower()
    return "no" not in s


def metrics(preds: list[str], golds: list[str]) -> dict:
    tp = fp = tn = fn = invalid = 0
    for pred, gold in zip(preds, golds):
        g = gold == "Yes"
        if pred == "Invalid":
            invalid += 1
            if g:
                fn += 1
            else:
                fp += 1
            continue
        p = pred == "Yes"
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif (not p) and (not g):
            tn += 1
        else:
            fn += 1
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / n if n else 0.0
    valid = n - invalid
    yes_n = sum(1 for p in preds if p == "Yes")
    return {
        "n": n,
        "valid": valid,
        "Invalid": invalid,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Yes_ratio": yes_n / n if n else 0.0,
    }


def rbench_fold_ids() -> list[list]:
    folds = []
    for i in range(1, 6):
        folds.append(json.loads((FAST / f"rbench_fold_{i}.json").read_text(encoding="utf-8")))
    return folds


def score_rbench(pred_by_qid: dict[object, str], gold_by_qid: dict) -> dict:
    fold_recs = []
    for i, qids in enumerate(rbench_fold_ids(), 1):
        preds, golds = [], []
        missing = 0
        for qid in qids:
            pred = None
            for key in qid_aliases(qid):
                if key in pred_by_qid:
                    pred = pred_by_qid[key]
                    break
            gold = None
            for key in qid_aliases(qid):
                if key in gold_by_qid:
                    gold = gold_by_qid[key]
                    break
            if pred is None or gold is None:
                missing += 1
                continue
            preds.append(pred)
            golds.append(gold)
        rec = metrics(preds, golds)
        rec["fold"] = i
        rec["missing"] = missing
        fold_recs.append(rec)
    keys = ["Accuracy", "Precision", "Recall", "F1", "Yes_ratio"]
    avg = {k: sum(r[k] for r in fold_recs) / 5 for k in keys}
    avg.update(
        {
            "n_mean": sum(r["n"] for r in fold_recs) / 5,
            "FP_mean": sum(r["FP"] for r in fold_recs) / 5,
            "FN_mean": sum(r["FN"] for r in fold_recs) / 5,
            "TP_mean": sum(r["TP"] for r in fold_recs) / 5,
            "TN_mean": sum(r["TN"] for r in fold_recs) / 5,
            "Invalid_mean": sum(r["Invalid"] for r in fold_recs) / 5,
            "folds": fold_recs,
        }
    )
    return avg


def score_items(items: list[dict], pred_by_qid: dict) -> dict:
    preds, golds, missing = [], [], 0
    for it in items:
        qid = it["question_id"]
        pred = None
        for key in qid_aliases(qid):
            if key in pred_by_qid:
                pred = pred_by_qid[key]
                break
        if pred is None:
            missing += 1
            continue
        preds.append(pred)
        golds.append(it["gold"] if it.get("gold") in {"Yes", "No"} else ("No" if not gold_yes(it.get("label")) else "Yes"))
    rec = metrics(preds, golds)
    rec["missing"] = missing
    return rec


def preds_from_answers(path: Path, official_fallback: bool = False) -> dict:
    amap = answer_map(path)
    out = {}
    for k, row in amap.items():
        out[k] = parse_pred(row.get("text") or "", official_fallback=official_fallback)
    return out


def gold_maps():
    rbench_q = load_jsonl(ROOT / "data/phase2b5/rbench_fast_matched_prompt.jsonl")
    mmrel_q = load_jsonl(ROOT / "data/phase2b5/mmrel_adv_fast_matched_prompt.jsonl")
    r_gold = {}
    for q in rbench_q:
        for k in qid_aliases(q["question_id"]):
            r_gold[k] = q["gold"]
    m_items = mmrel_q
    return rbench_q, mmrel_q, r_gold, m_items


def sensitivity(off_preds: dict, mat_preds: dict, items: list[dict]) -> dict:
    agr = ow_mc = oc_mw = y2n = n2y = both = 0
    for it in items:
        qid = it["question_id"]
        o = m = None
        for k in qid_aliases(qid):
            if o is None and k in off_preds:
                o = off_preds[k]
            if m is None and k in mat_preds:
                m = mat_preds[k]
        if o is None or m is None:
            continue
        gold = it["gold"]
        both += 1
        if o == m:
            agr += 1
        if o != gold and m == gold:
            ow_mc += 1
        if o == gold and m != gold:
            oc_mw += 1
        if o == "Yes" and m == "No":
            y2n += 1
        if o == "No" and m == "Yes":
            n2y += 1
    return {
        "n": both,
        "agreement": agr / both if both else 0.0,
        "official_wrong_matched_correct": ow_mc,
        "official_correct_matched_wrong": oc_mw,
        "Yes_to_No": y2n,
        "No_to_Yes": n2y,
    }


def mcnemar(base_ok: list[int], lora_ok: list[int]) -> dict:
    b = c = 0  # b: base correct lora wrong; c: base wrong lora correct
    for x, y in zip(base_ok, lora_ok):
        if x and not y:
            b += 1
        elif (not x) and y:
            c += 1
    n_disc = b + c
    if n_disc == 0:
        p = 1.0
        chi2 = 0.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n_disc  # continuity correction
        # chi-square survival with 1 df
        p = math.erfc(math.sqrt(chi2 / 2))
    return {"b_base_correct_lora_wrong": b, "c_base_wrong_lora_correct": c, "chi2_cc": chi2, "p_value": p, "n_discordant": n_disc}


def bootstrap_delta(base_ok: list[int], lora_ok: list[int], n_boot=N_BOOT, seed=SEED) -> dict:
    rng = random.Random(seed)
    n = len(base_ok)
    acc_b = sum(base_ok) / n
    acc_l = sum(lora_ok) / n
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        db = sum(base_ok[i] for i in idx) / n
        dl = sum(lora_ok[i] for i in idx) / n
        deltas.append(dl - db)
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot)]
    return {
        "base_acc": acc_b,
        "lora_acc": acc_l,
        "delta": acc_l - acc_b,
        "ci95": [lo, hi],
        "n": n,
        "n_boot": n_boot,
    }


def item_correct(pred: str, gold: str) -> int:
    return int(pred == gold)


def paired_lists(base_preds, lora_preds, items):
    b_ok, l_ok = [], []
    for it in items:
        qid = it["question_id"]
        bp = lp = None
        for k in qid_aliases(qid):
            if bp is None and k in base_preds:
                bp = base_preds[k]
            if lp is None and k in lora_preds:
                lp = lora_preds[k]
        if bp is None or lp is None:
            continue
        gold = it["gold"]
        b_ok.append(item_correct(bp, gold))
        l_ok.append(item_correct(lp, gold))
    return b_ok, l_ok


def fp_delta(base_preds, lora_preds, items):
    def fps(preds):
        fp = 0
        n = 0
        for it in items:
            qid = it["question_id"]
            pred = None
            for k in qid_aliases(qid):
                if k in preds:
                    pred = preds[k]
                    break
            if pred is None:
                continue
            n += 1
            if it["gold"] == "No" and pred == "Yes":
                fp += 1
        return fp, n
    b, n = fps(base_preds)
    l, n2 = fps(lora_preds)
    return {"base_fp": b, "lora_fp": l, "delta_fp": l - b, "n": n}


def main() -> None:
    rbench_items, mmrel_items, r_gold, _ = gold_maps()
    blob = {"runs": {}, "sensitivity": {}, "paired": {}, "deltas": {}}
    store = {}
    for run, spec in RUNS.items():
        off_r = preds_from_answers(spec["official_rbench"], official_fallback=True)
        off_m = preds_from_answers(spec["official_mmrel"], official_fallback=True)
        mat_r = preds_from_answers(OUT / f"answers/{run}_rbench_matched.jsonl", official_fallback=False)
        mat_m = preds_from_answers(OUT / f"answers/{run}_mmrel_adv_matched.jsonl", official_fallback=False)
        store[run] = {"off_r": off_r, "off_m": off_m, "mat_r": mat_r, "mat_m": mat_m}
        blob["runs"][run] = {
            "rbench": {
                "official": score_rbench(off_r, r_gold),
                "matched": score_rbench(mat_r, r_gold),
            },
            "mmrel_adv": {
                "official": score_items(mmrel_items, off_m),
                "matched": score_items(mmrel_items, mat_m),
            },
        }
        blob["sensitivity"][run] = {
            "rbench": sensitivity(off_r, mat_r, rbench_items),
            "mmrel_adv": sensitivity(off_m, mat_m, mmrel_items),
        }

    for bench, items, key in (
        ("rbench", rbench_items, "r"),
        ("mmrel_adv", mmrel_items, "m"),
    ):
        blob["paired"][bench] = {}
        blob["deltas"][bench] = {}
        for prompt, pkey in (("official", f"off_{key}"), ("matched", f"mat_{key}")):
            b_ok, l_ok = paired_lists(store["base"][pkey], store["v2b_2ep"][pkey], items)
            rec = {
                "v2b_vs_base": {
                    "mcnemar": mcnemar(b_ok, l_ok),
                    "bootstrap": bootstrap_delta(b_ok, l_ok),
                    "fp": fp_delta(store["base"][pkey], store["v2b_2ep"][pkey], items),
                }
            }
            b_ok2, l_ok2 = paired_lists(store["base"][pkey], store["v2_old"][pkey], items)
            rec["v2old_vs_base"] = {
                "mcnemar": mcnemar(b_ok2, l_ok2),
                "bootstrap": bootstrap_delta(b_ok2, l_ok2),
                "fp": fp_delta(store["base"][pkey], store["v2_old"][pkey], items),
            }
            blob["paired"][bench][prompt] = rec
            blob["deltas"][bench][prompt] = {
                "acc_v2b_minus_base": rec["v2b_vs_base"]["bootstrap"]["delta"],
                "fp_v2b_minus_base": rec["v2b_vs_base"]["fp"]["delta_fp"],
            }

    out = OUT / "metrics" / "phase2b5_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def drop_folds(obj):
        if isinstance(obj, dict):
            return {k: drop_folds(v) for k, v in obj.items() if k != "folds"}
        if isinstance(obj, list):
            return [drop_folds(x) for x in obj]
        return obj

    slim = drop_folds(blob)
    out.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
