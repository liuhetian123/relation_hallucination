#!/usr/bin/env python3
"""Locate relation phrases in the 955 matched Phase 1 train captions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/phase15"))

from relation_extract import extract_relation, normalize_relation, strip_caption  # noqa: E402


def find_all(text: str, phrase: str) -> list[int]:
    tl, pl = text.lower(), phrase.lower()
    out, i = [], 0
    while True:
        j = tl.find(pl, i)
        if j < 0:
            break
        out.append(j)
        i = j + 1
    return out


def caption_of(sample: dict) -> str:
    return strip_caption(sample["conversations"][-1]["value"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit", default=str(ROOT / "data/phase1/qwen_explicit_1k.json"))
    parser.add_argument("--typed", default=str(ROOT / "data/phase1/qwen_typed_1k.json"))
    parser.add_argument("--out_dir", default=str(ROOT / "eval_results/qwen/phase16"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    explicit = json.loads(Path(args.explicit).read_text(encoding="utf-8"))
    typed = json.loads(Path(args.typed).read_text(encoding="utf-8"))
    if len(explicit) != len(typed):
        raise SystemExit("Explicit/Typed length mismatch")

    rows = []
    n_success = n_fail = n_empty = n_multi = 0
    for exp, typ in zip(explicit, typed):
        if exp["id"] != typ["id"] or exp["image"] != typ["image"]:
            raise SystemExit(f"Unmatched pair {exp['id']} vs {typ['id']}")
        cap_e = caption_of(exp)
        cap_t = caption_of(typ)
        rel, src = extract_relation(cap_t, between_only=True)
        if not rel:
            rel, src = extract_relation(cap_t, between_only=False)
        rec = {
            "id": exp["id"],
            "image": exp["image"],
            "explicit_caption": cap_e,
            "typed_caption": cap_t,
            "relation": rel,
            "relation_source": src if rel else "fail",
            "explicit_starts": [],
            "typed_starts": [],
            "located": False,
            "multi_match": False,
        }
        if not rel:
            n_fail += 1
            n_empty += 1
            rows.append(rec)
            continue
        rec["explicit_starts"] = find_all(cap_e, rel)
        rec["typed_starts"] = find_all(cap_t, rel)
        if len(rec["explicit_starts"]) > 1 or len(rec["typed_starts"]) > 1:
            rec["multi_match"] = True
            n_multi += 1
        if rec["explicit_starts"] and rec["typed_starts"]:
            rec["located"] = True
            n_success += 1
        else:
            n_fail += 1
        rows.append(rec)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spans_path = out / "train_relation_spans.jsonl"
    with spans_path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    qc = {
        "total": len(rows),
        "successfully_located_relation": n_success,
        "failed_relation_location": n_fail,
        "multi_match": n_multi,
        "empty_relation": n_empty,
        "used_for_relation_token_nll": n_success,
        "used_for_full_target_nll": len(rows),
    }
    (out / "qc_spans.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")

    located = [r for r in rows if r["located"]]
    rng = random.Random(args.seed)
    review = rng.sample(located, min(50, len(located)))
    review.sort(key=lambda r: r["id"])
    with (out / "review_50_spans.jsonl").open("w", encoding="utf-8") as f:
        for rec in review:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("QC")
    for k, v in qc.items():
        print(f"  {k}: {v}")
    print("\nReview 8:")
    for rec in review[:8]:
        print(f"- {rec['relation']}")
        print(f"  T: {rec['typed_caption']}")
        print(f"  E: {rec['explicit_caption']}")
    print(f"\nWrote {spans_path}")


if __name__ == "__main__":
    main()
