#!/usr/bin/env python3
"""Score Reefknot YESNO answers overall and by relation_type."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def classify(text: str) -> int | None:
    match = re.search(r"\b(yes|no)\b", (text or "").lower())
    if not match:
        return None
    return int(match.group(1) == "yes")


def metrics(rows: list[tuple[int, int]]) -> dict:
    tp = sum(pred == gold == 1 for pred, gold in rows)
    tn = sum(pred == gold == 0 for pred, gold in rows)
    fp = sum(pred == 1 and gold == 0 for pred, gold in rows)
    fn = sum(pred == 0 and gold == 1 for pred, gold in rows)
    n = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": n,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": (tp + tn) / n if n else 0.0,
        "Precision": precision,
        "Recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "Yes_ratio": (tp + fp) / n if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers_file", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    invalid: dict[str, int] = defaultdict(int)
    for line in Path(args.answers_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        group = row["relation_type"]
        prediction = classify(row.get("text", ""))
        if prediction is None or row.get("error"):
            invalid[group] += 1
            invalid["overall"] += 1
            continue
        gold = int(str(row["label"]).strip().lower() == "yes")
        groups[group].append((prediction, gold))
        groups["overall"].append((prediction, gold))
    result = {group: {**metrics(rows), "invalid": invalid[group]} for group, rows in groups.items()}
    for group in invalid:
        result.setdefault(group, {**metrics([]), "invalid": invalid[group]})
    destination = Path(args.out_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
