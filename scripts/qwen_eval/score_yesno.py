#!/usr/bin/env python3
"""Yes/No scoring with Acc / P / R / F1 / TP TN FP FN / Yes ratio."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def pred_yes(text: str) -> bool:
    text = (text or "").strip()
    if "." in text:
        text = text.split(".")[0]
    text = text.replace(",", " ")
    words = text.split()
    if any(w in {"No", "not", "no"} for w in words):
        return False
    return True


def gold_yes(label) -> bool:
    if isinstance(label, bool):
        return label
    s = str(label).strip().lower()
    if s in {"no", "false", "0", "n"}:
        return False
    return True


def metrics(preds: list[int], labels: list[int]) -> dict:
    tp = fp = tn = fn = 0
    for pred, label in zip(preds, labels):
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / total if total else 0.0
    yes_ratio = sum(preds) / len(preds) if preds else 0.0
    return {
        "n": total,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Yes_ratio": yes_ratio,
    }


def load_json_or_jsonl(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question_file", required=True)
    parser.add_argument("--result_file", required=True)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--label_key", default="label")
    args = parser.parse_args()

    questions = load_json_or_jsonl(Path(args.question_file))
    if isinstance(questions, dict):
        questions = list(questions.values())
    if args.category:
        questions = [
            q
            for q in questions
            if q.get("category") == args.category or q.get("split") == args.category
        ]
    qmap = {}
    for q in questions:
        qid = q.get("question_id", q.get("id"))
        qmap[qid] = q

    answers = load_json_or_jsonl(Path(args.result_file))
    amap = {a.get("question_id", a.get("id")): a for a in answers}

    preds, labels, missing, skipped_empty = [], [], 0, 0
    for qid, q in qmap.items():
        if qid not in amap:
            missing += 1
            continue
        ans = amap[qid]
        text = ans.get("text", "") or ""
        if ans.get("error") or not text.strip():
            skipped_empty += 1
            continue
        gold = q.get(args.label_key, q.get("answer", q.get("gt")))
        if gold is None:
            continue
        labels.append(1 if gold_yes(gold) else 0)
        preds.append(1 if pred_yes(text) else 0)

    result = metrics(preds, labels)
    result["missing"] = missing
    result["skipped_empty"] = skipped_empty
    result["category"] = args.category
    print(json.dumps(result, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
