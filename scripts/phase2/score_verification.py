#!/usr/bin/env python3
"""Score Phase 2 held-out / train verification answers with subset breakdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/qwen_eval"))
from score_yesno import gold_yes, load_json_or_jsonl, metrics, pred_yes  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--question_file", required=True)
    p.add_argument("--result_file", required=True)
    p.add_argument("--out_json", required=True)
    return p.parse_args()


def score_subset(questions, amap, subset=None):
    preds, labels = [], []
    skipped = 0
    for q in questions:
        if subset and q.get("subset") != subset:
            continue
        qid = q.get("question_id", q.get("id"))
        ans = amap.get(qid)
        if not ans or ans.get("error") or not (ans.get("text") or "").strip():
            skipped += 1
            continue
        labels.append(1 if gold_yes(q.get("label")) else 0)
        preds.append(1 if pred_yes(ans.get("text")) else 0)
    rec = metrics(preds, labels)
    rec["skipped_empty"] = skipped
    rec["subset"] = subset or "all"
    if subset == "hard_negative" and rec["n"]:
        rec["hard_fp_rate"] = rec["FP"] / rec["n"]
        rec["hard_negative_accuracy"] = rec["TN"] / rec["n"]
    return rec


def main() -> None:
    args = parse_args()
    questions = load_json_or_jsonl(Path(args.question_file))
    answers = load_json_or_jsonl(Path(args.result_file))
    amap = {a.get("question_id", a.get("id")): a for a in answers}
    subsets = sorted({q.get("subset") for q in questions if q.get("subset")})
    blob = {"all": score_subset(questions, amap, None)}
    for sub in subsets:
        blob[sub] = score_subset(questions, amap, sub)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(blob, indent=2))


if __name__ == "__main__":
    main()
