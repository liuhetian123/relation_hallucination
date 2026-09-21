#!/usr/bin/env python3
"""Build matched Old vs Clean verification JSON from clean_sro3000.jsonl.

Same 2156 images / positives; only the negative statement changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, dump_json, llava_verify_sample


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", default=str(DATA_DIR / "clean_sro3000.jsonl"))
    p.add_argument("--out-old", default=str(DATA_DIR / "train_old.json"))
    p.add_argument("--out-clean", default=str(DATA_DIR / "train_clean.json"))
    p.add_argument("--out-stats", default=str(DATA_DIR / "train_build_stats.json"))
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pair_samples(rec: dict, neg_statement: str, neg_relation: str, subset: str, prefix: str) -> list[dict]:
    pos = llava_verify_sample(
        f"{rec['id']}__pos",
        rec["image"],
        rec["positive_statement"],
        "Yes",
        source_id=rec["id"],
        split="train",
        subset="positive",
        subject=rec["subject"],
        object=rec["object"],
        gt_relation=rec["positive_relation"],
        scale_prefix=prefix,
    )
    neg = llava_verify_sample(
        f"{rec['id']}__neg",
        rec["image"],
        neg_statement,
        "No",
        source_id=rec["id"],
        split="train",
        subset=subset,
        subject=rec["subject"],
        object=rec["object"],
        gt_relation=rec["positive_relation"],
        neg_relation=neg_relation,
        scale_prefix=prefix,
    )
    return [pos, neg]


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.clean))
    old, clean = [], []
    skipped = 0
    for rec in rows:
        old_stmt = (rec.get("old_negative_statement") or "").strip()
        old_rel = (rec.get("old_negative_relation") or "").strip()
        clean_stmt = (rec.get("negative_statement") or "").strip()
        clean_rel = (rec.get("negative_relation") or "").strip()
        if not old_stmt or not clean_stmt:
            skipped += 1
            continue
        old.extend(pair_samples(rec, old_stmt, old_rel, "random_negative", "old"))
        clean.extend(pair_samples(rec, clean_stmt, clean_rel, "clean_negative", "clean"))
    dump_json(Path(args.out_old), old)
    dump_json(Path(args.out_clean), clean)
    stats = {
        "n_source": len(rows),
        "n_pairs": len(old) // 2,
        "n_samples": len(old),
        "skipped_incomplete": skipped,
        "same_length": len(old) == len(clean),
        "same_ids": [r["id"] for r in old] == [r["id"] for r in clean],
    }
    dump_json(Path(args.out_stats), stats)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
