#!/usr/bin/env python3
"""Build Phase 2A V1/V2 data, random/hard candidates, and InternVL filter jobs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    ROOT,
    VISUAL_RELATIONS,
    build_relation_vocab,
    caption_of,
    dump_json,
    dump_jsonl,
    hard_score,
    llava_verify_sample,
    load_json,
    normalize_relation,
    replace_relation,
    too_close_for_hard,
    too_close_for_random,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--explicit", default=str(ROOT / "data/phase1/qwen_explicit_1k.json"))
    p.add_argument("--spans", default=str(ROOT / "eval_results/qwen/phase16/train_relation_spans.jsonl"))
    p.add_argument("--heldout", default=str(ROOT / "eval_results/qwen/phase16b/heldout_matched.jsonl"))
    p.add_argument("--relsim_1k", default=str(ROOT / "data/relsim_llava_1k.json"))
    p.add_argument("--relsim_100k", default="/home/yy/relation_hallucination/data/relsim_llava_100k.json")
    p.add_argument("--out_dir", default=str(ROOT / "data/phase2"))
    p.add_argument("--train_seed", type=int, default=42)
    p.add_argument("--heldout_seed", type=int, default=123)
    p.add_argument("--n_hard_cand", type=int, default=5)
    p.add_argument("--min_freq", type=int, default=2)
    return p.parse_args()


def load_vocab(args) -> list[str]:
    captions = []
    for path in (Path(args.relsim_1k), Path(args.relsim_100k)):
        if not path.exists():
            print(f"skip missing {path}")
            continue
        data = load_json(path)
        captions.extend(caption_of(s) for s in data)
    items = build_relation_vocab(captions, min_freq=args.min_freq)
    return [r for r, _c in items], items


def sample_random(gt: str, vocab: list[str], rng: random.Random, n: int = 1) -> list[str]:
    pool = [r for r in vocab if not too_close_for_random(gt, r)]
    if not pool:
        pool = [r for r in vocab if normalize_relation(r) != normalize_relation(gt)]
    rng.shuffle(pool)
    out = []
    for r in pool:
        if r not in out:
            out.append(r)
        if len(out) >= n:
            break
    return out


def rank_hard(gt: str, vocab: list[str], k: int) -> list[str]:
    pool = list(dict.fromkeys(list(VISUAL_RELATIONS) + vocab))
    scored = []
    for r in pool:
        s = hard_score(gt, r)
        if s > 0:
            scored.append((s, r))
    scored.sort(key=lambda x: (-x[0], x[1]))
    seen_head = Counter()
    out = []
    for _s, r in scored:
        head = r.split()[0]
        if seen_head[head] >= 2:
            continue
        seen_head[head] += 1
        out.append(r)
        if len(out) >= k:
            break
    if len(out) < k:
        for _s, r in scored:
            if r not in out:
                out.append(r)
            if len(out) >= k:
                break
    return out


def make_negative(row: dict, new_rel: str, kind: str, split: str) -> dict | None:
    statement = replace_relation(row["explicit_caption"], row["relation"], new_rel)
    if not statement:
        return None
    sid = f"{row['id']}__{kind}__{normalize_relation(new_rel).replace(' ', '_')}"
    rec = llava_verify_sample(
        sid,
        row["image"],
        statement,
        "No",
        source_id=row["id"],
        split=split,
        subset=kind,
        gt_relation=row["relation"],
        neg_relation=new_rel,
        explicit_caption=row["explicit_caption"],
    )
    return rec


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_rng = random.Random(args.train_seed)
    ho_rng = random.Random(args.heldout_seed)

    vocab, vocab_items = load_vocab(args)
    dump_json(out / "relation_vocab.json", {"min_freq": args.min_freq, "n": len(vocab), "items": vocab_items[:2000]})
    print(f"vocab={len(vocab)}")

    explicit = {s["id"]: s for s in load_json(Path(args.explicit))}
    spans = load_json(Path(args.spans))
    heldout = load_json(Path(args.heldout))

    positives = []
    for sample in explicit.values():
        cap = caption_of(sample)
        positives.append(
            llava_verify_sample(
                f"{sample['id']}__pos",
                sample["image"],
                cap,
                "Yes",
                source_id=sample["id"],
                split="train",
                subset="positive",
                explicit_caption=cap,
            )
        )
    dump_json(out / "v1_positive.json", positives)
    print(f"V1 positives={len(positives)}")

    train_loc = []
    for sp in spans:
        if not sp.get("located") or not sp.get("relation"):
            continue
        if sp["id"] not in explicit:
            continue
        train_loc.append(
            {
                "id": sp["id"],
                "image": sp["image"],
                "explicit_caption": sp["explicit_caption"],
                "relation": normalize_relation(sp["relation"]),
                "split": "train",
            }
        )

    ho_loc = []
    for h in heldout:
        if not h.get("ok_explicit") or not h.get("rel_in_explicit"):
            continue
        ho_loc.append(
            {
                "id": h["id"],
                "image": h["image"],
                "explicit_caption": h["explicit_caption"],
                "relation": normalize_relation(h["gt_relation"]),
                "split": "heldout",
            }
        )
    print(f"replaceable train={len(train_loc)} heldout={len(ho_loc)}")

    train_random, ho_random = [], []
    n_fail = 0
    for row in train_loc:
        picked = sample_random(row["relation"], vocab, train_rng, n=1)
        rec = make_negative(row, picked[0], "random_negative", "train") if picked else None
        if rec:
            train_random.append(rec)
        else:
            n_fail += 1
    for row in ho_loc:
        picked = sample_random(row["relation"], vocab, ho_rng, n=1)
        rec = make_negative(row, picked[0], "random_negative", "heldout") if picked else None
        if rec:
            ho_random.append(rec)
        else:
            n_fail += 1

    v2 = positives + train_random
    train_rng.shuffle(v2)
    dump_json(out / "v2_pos_random.json", v2)
    dump_jsonl(out / "train_random_negatives.jsonl", train_random)
    dump_jsonl(out / "heldout_random_negatives.jsonl", ho_random)
    print(f"V2 n={len(v2)} train_random={len(train_random)} heldout_random={len(ho_random)} replace_fail={n_fail}")

    jobs = []
    for row in train_loc + ho_loc:
        cands = rank_hard(row["relation"], vocab, args.n_hard_cand)
        for rank, cand in enumerate(cands, 1):
            stmt = replace_relation(row["explicit_caption"], row["relation"], cand)
            if not stmt:
                continue
            jobs.append(
                {
                    "job_id": f"{row['id']}__hard__{rank}__{cand.replace(' ', '_')}",
                    "source_id": row["id"],
                    "image": row["image"],
                    "split": row["split"],
                    "gt_relation": row["relation"],
                    "candidate_relation": cand,
                    "rank": rank,
                    "hard_score": hard_score(row["relation"], cand),
                    "original_statement": row["explicit_caption"],
                    "candidate_statement": stmt,
                }
            )
    dump_jsonl(out / "hard_candidate_jobs.jsonl", jobs)
    meta = {
        "n_positive": len(positives),
        "n_train_replaceable": len(train_loc),
        "n_heldout_replaceable": len(ho_loc),
        "n_train_random": len(train_random),
        "n_heldout_random": len(ho_random),
        "n_hard_jobs": len(jobs),
        "n_hard_jobs_train": sum(1 for j in jobs if j["split"] == "train"),
        "n_hard_jobs_heldout": sum(1 for j in jobs if j["split"] == "heldout"),
        "vocab_size": len(vocab),
        "train_seed": args.train_seed,
        "heldout_seed": args.heldout_seed,
    }
    dump_json(out / "build_meta.json", meta)
    print("meta", json.dumps(meta, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
