#!/usr/bin/env python3
"""Score R-Bench / MMRel-Adv / AMBER-dr on the frozen Phase 2B fast subsets.

Can rescore existing full-benchmark predictions (QC) or new subset-only answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/qwen_eval"))
from score_yesno import gold_yes, load_json_or_jsonl, metrics, pred_yes  # noqa: E402

FAST = ROOT / "eval_results/qwen/phase2b/fast_subsets"
ANN = ROOT / "AMBER/data/annotations.json"


def qid_aliases(qid):
    yield qid
    if isinstance(qid, int):
        yield str(qid)
    elif isinstance(qid, str) and qid.isdigit():
        yield int(qid)


def answer_map(rows: list[dict]) -> dict:
    amap = {}
    for row in rows:
        qid = row.get("question_id", row.get("id"))
        for key in qid_aliases(qid):
            amap[key] = row
    return amap


def lookup(amap: dict, qid):
    for key in qid_aliases(qid):
        if key in amap:
            return amap[key]
    return None


def rbench_fold_metrics(answers: dict, labels: dict, qids: list) -> dict:
    preds, golds = [], []
    skipped = 0
    for qid in qids:
        text = lookup(answers, qid)
        if text is None:
            skipped += 1
            continue
        text = text or ""
        preds.append(1 if pred_yes(text) else 0)
        lab = labels.get(qid, "")
        golds.append(0 if (lab and "no" in str(lab).lower()) else 1)
    rec = metrics(preds, golds)
    rec["skipped_missing"] = skipped
    return rec


def score_rbench(result_file: Path) -> dict:
    questions = json.loads((ROOT / "R-Bench/data_filterd/image-level_filterd.json").read_text(encoding="utf-8"))
    labels = {q["question_id"]: q["label"] for q in questions}
    rows = load_json_or_jsonl(result_file)
    amap = answer_map(rows)
    answers = {qid: (row.get("text") if row else None) for qid, row in amap.items()}
    fold_recs = []
    for i in range(1, 6):
        qids = json.loads((FAST / f"rbench_fold_{i}.json").read_text(encoding="utf-8"))
        rec = rbench_fold_metrics(answers, labels, qids)
        rec["fold"] = i
        rec["fold_n_ids"] = len(qids)
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
            "folds": fold_recs,
        }
    )
    return avg


def score_mmrel(result_file: Path) -> dict:
    questions = load_json_or_jsonl(FAST / "mmrel_adv_questions.jsonl")
    rows = load_json_or_jsonl(result_file)
    amap = answer_map(rows)
    preds, golds = [], []
    skipped_empty = 0
    missing = 0
    for q in questions:
        qid = q.get("question_id", q.get("id"))
        ans = lookup(amap, qid)
        if ans is None:
            missing += 1
            continue
        text = (ans.get("text") or "").strip()
        if ans.get("error") or not text:
            skipped_empty += 1
            continue
        golds.append(1 if gold_yes(q.get("label")) else 0)
        preds.append(1 if pred_yes(text) else 0)
    rec = metrics(preds, golds)
    rec["missing"] = missing
    rec["skipped_empty"] = skipped_empty
    return rec


def normalize_amber_response(text: str) -> str:
    text = (text or "").strip()
    lower = text.lower()
    if lower.startswith("yes"):
        return "Yes"
    if lower.startswith("no"):
        return "No"
    if "yes" in lower and "no" not in lower:
        return "Yes"
    if "no" in lower and "yes" not in lower:
        return "No"
    return "No"


def score_amber(result_file: Path) -> dict:
    """Official AMBER-dr: No is the positive class for Precision/Recall."""
    ids = json.loads((FAST / "amber_dr_ids.json").read_text(encoding="utf-8"))
    id_set = set(ids)
    ann = {x["id"]: x for x in json.loads(ANN.read_text(encoding="utf-8"))}
    raw = load_json_or_jsonl(result_file)
    responses = {}
    if raw and isinstance(raw, list) and "response" in raw[0]:
        for row in raw:
            for key in qid_aliases(row["id"]):
                responses[key] = row["response"]
    else:
        for row in raw:
            qid = row.get("question_id", row.get("id"))
            val = normalize_amber_response(row.get("text") or row.get("response") or "")
            for key in qid_aliases(qid):
                responses[key] = val

    correct = pred_no = gold_no_n = tp_no = 0
    n = 0
    missing = 0
    for qid in ids:
        resp = lookup(responses, qid)
        if resp is None:
            missing += 1
            continue
        truth = str(ann[qid]["truth"]).strip().lower()
        n += 1
        if truth == "yes":
            if resp == "Yes":
                correct += 1
        else:
            gold_no_n += 1
            if resp == "No":
                correct += 1
                tp_no += 1
        if resp == "No":
            pred_no += 1
    acc = correct / n if n else 0.0
    precision = tp_no / pred_no if pred_no else 0.0
    recall = tp_no / gold_no_n if gold_no_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": n,
        "missing": missing,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "gold_no": gold_no_n,
        "pred_no": pred_no,
        "tp_no": tp_no,
        "Yes_ratio": 1.0 - (pred_no / n if n else 0.0),
        "id_set_n": len(id_set),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", required=True, choices=("rbench", "mmrel_adv", "amber_dr"))
    p.add_argument("--result_file", required=True)
    p.add_argument("--out_json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.result_file)
    if args.bench == "rbench":
        rec = score_rbench(path)
    elif args.bench == "mmrel_adv":
        rec = score_mmrel(path)
    else:
        rec = score_amber(path)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
