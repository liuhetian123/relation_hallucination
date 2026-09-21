#!/usr/bin/env python3
"""Sample up to N random negative relations per QC-ok SRO image."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    NEG_TRIES,
    ROOT,
    SEED,
    VISUAL_RELATIONS,
    dump_json,
    dump_jsonl,
    load_json,
    normalize_relation,
    render_statement,
    too_close_negative,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sro-ok", default=str(ROOT / "data/phase3a/sro_ok.jsonl"))
    p.add_argument("--out-jobs", default=str(ROOT / "data/phase3a/negative_jobs.jsonl"))
    p.add_argument("--out-vocab", default=str(ROOT / "data/phase3a/relation_vocab.json"))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tries", type=int, default=NEG_TRIES)
    return p.parse_args()


def sample_negs(gt: str, vocab: list[str], rng: random.Random, n: int) -> list[str]:
    pool = [r for r in vocab if not too_close_negative(gt, r)]
    rng.shuffle(pool)
    out = []
    seen = set()
    for r in pool:
        nr = normalize_relation(r)
        if nr in seen:
            continue
        seen.add(nr)
        out.append(r)
        if len(out) >= n:
            break
    return out


def main() -> None:
    args = parse_args()
    rows = load_json(Path(args.sro_ok))
    freq = Counter(normalize_relation(r["relation"]) for r in rows)
    vocab = []
    seen = set()
    for rel, _c in freq.most_common():
        if rel and rel not in seen:
            vocab.append(rel)
            seen.add(rel)
    for rel in VISUAL_RELATIONS:
        nr = normalize_relation(rel)
        if nr and nr not in seen:
            vocab.append(nr)
            seen.add(nr)
    dump_json(
        Path(args.out_vocab),
        {"n": len(vocab), "from_sro": len(freq), "items": freq.most_common(2000)},
    )

    rng = random.Random(args.seed)
    jobs = []
    for rec in rows:
        cands = sample_negs(rec["relation"], vocab, rng, args.tries)
        for i, neg in enumerate(cands):
            jobs.append(
                {
                    "job_id": f"{rec['id']}__neg{i}__{normalize_relation(neg).replace(' ', '_')}",
                    "id": rec["id"],
                    "image": rec["image"],
                    "subject": rec["subject"],
                    "object": rec["object"],
                    "gt_relation": rec["relation"],
                    "negative_relation": neg,
                    "rank": i,
                    "positive_statement": render_statement(rec["subject"], rec["relation"], rec["object"]),
                    "candidate_statement": render_statement(rec["subject"], neg, rec["object"]),
                }
            )
    dump_jsonl(Path(args.out_jobs), jobs)
    print(
        json.dumps(
            {
                "n_sro_ok": len(rows),
                "vocab": len(vocab),
                "n_jobs": len(jobs),
                "tries": args.tries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
