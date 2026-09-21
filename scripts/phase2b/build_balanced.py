#!/usr/bin/env python3
"""Build Phase 2B 1:1 paired train data by dropping unmatched V2 positives.

Keeps the original 534 V2 random negatives and the 534 positives from the
same relation-span-eligible images. Does not resample negatives.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data/phase2/v2_pos_random.json"
OUT_DIR = ROOT / "data/phase2b"
OUT_DATA = OUT_DIR / "v2b_balanced.json"
OUT_IDS = OUT_DIR / "eligible_image_ids.json"


def main() -> None:
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    pos = [x for x in v2 if x.get("label") == "Yes"]
    neg = [x for x in v2 if x.get("label") == "No"]
    if len(neg) != 534:
        raise SystemExit(f"Expected 534 V2 negatives, got {len(neg)}")

    pos_by_src = {x["source_id"]: x for x in pos}
    missing = [n["source_id"] for n in neg if n["source_id"] not in pos_by_src]
    if missing:
        raise SystemExit(f"Negatives without matching positive: {missing[:5]}")

    paired_pos = [pos_by_src[n["source_id"]] for n in neg]
    if len({p["image"] for p in paired_pos}) != 534:
        raise SystemExit("Paired positives are not 534 unique images")
    if any(p["image"] != n["image"] for p, n in zip(paired_pos, neg)):
        raise SystemExit("Positive/negative image mismatch")

    # Keep original records; order is all positives then all negatives.
    # Trainer shuffles with data_seed=42.
    data = paired_pos + neg
    eligible = [
        {
            "source_id": n["source_id"],
            "image": n["image"],
            "gt_relation": n.get("gt_relation"),
            "neg_relation": n.get("neg_relation"),
            "positive_id": p["id"],
            "negative_id": n["id"],
        }
        for p, n in zip(paired_pos, neg)
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_IDS.write_text(json.dumps(eligible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "n": len(data),
                "labels": dict(Counter(x["label"] for x in data)),
                "subsets": dict(Counter(x.get("subset") for x in data)),
                "unique_images": len({x["image"] for x in data}),
                "out_data": str(OUT_DATA),
                "out_ids": str(OUT_IDS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
