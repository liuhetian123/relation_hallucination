#!/usr/bin/env python3
"""Assemble 3000 nested image-pair splits from InternVL-filtered SRO + No negatives."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    IMAGE_ROOT,
    MASTER_N,
    MAX_REL_FRAC,
    ROOT,
    SEED,
    TEACHER_NAME,
    diversity_stats,
    dump_json,
    dump_jsonl,
    llava_verify_sample,
    load_json,
    normalize_relation,
    render_statement,
)

SCALES = (534, 1500, 3000)


def first_no_by_image(rows: list[dict]) -> dict[str, dict]:
    best = {}
    for rec in rows:
        if rec.get("judgment") != "no":
            continue
        sid = rec["id"]
        rank = rec.get("rank", 99)
        if sid not in best or rank < best[sid].get("rank", 99):
            best[sid] = rec
    return best


def greedy_select(paired: list[dict], n: int, seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    order = list(paired)
    rng.shuffle(order)
    selected: list[dict] = []
    leftover: list[dict] = []
    rel_count: Counter[str] = Counter()
    relaxed_from = None
    caps = [MAX_REL_FRAC, 0.08, 0.12, 1.0]
    remaining = order
    for frac in caps:
        cap = n if frac >= 1 else max(1, int(n * frac))
        still = []
        for rec in remaining:
            if len(selected) >= n:
                still.append(rec)
                continue
            rel = rec["relation_norm"]
            if rel_count[rel] < cap:
                selected.append(rec)
                rel_count[rel] += 1
                if frac > MAX_REL_FRAC and relaxed_from is None:
                    relaxed_from = frac
            else:
                still.append(rec)
        remaining = still
        if len(selected) >= n:
            break
    leftover = remaining
    meta = {
        "requested": n,
        "selected": len(selected),
        "leftover": len(leftover),
        "cap_start": MAX_REL_FRAC,
        "relaxed_to": relaxed_from,
        "max_relation_count": max(rel_count.values()) if rel_count else 0,
    }
    return selected[:n], meta


def expand_train(pairs: list[dict], prefix: str) -> list[dict]:
    out = []
    for rec in pairs:
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
            gt_relation=rec["relation"],
            scale_prefix=prefix,
        )
        neg = llava_verify_sample(
            f"{rec['id']}__neg",
            rec["image"],
            rec["negative_statement"],
            "No",
            source_id=rec["id"],
            split="train",
            subset="random_negative",
            subject=rec["subject"],
            object=rec["object"],
            gt_relation=rec["relation"],
            neg_relation=rec["negative_relation"],
            scale_prefix=prefix,
        )
        out.append(pos)
        out.append(neg)
    return out


def write_review(pairs: list[dict], path: Path, n: int = 200, seed: int = SEED) -> None:
    freq = Counter(p["relation_norm"] for p in pairs)
    ranked = sorted(pairs, key=lambda p: (-freq[p["relation_norm"]], p["id"]))
    tert = max(len(ranked) // 3, 1)
    buckets = [ranked[:tert], ranked[tert : 2 * tert], ranked[2 * tert :]]
    rng = random.Random(seed)
    picked = []
    per = n // 3
    for bucket in buckets:
        b = list(bucket)
        rng.shuffle(b)
        picked.extend(b[:per])
    if len(picked) < n:
        rest = [p for p in pairs if p["id"] not in {x["id"] for x in picked}]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "id",
                "image_path",
                "subject",
                "relation",
                "object",
                "positive_statement",
                "negative_relation",
                "negative_statement",
                "relation_freq_in_master",
                "subject_ok",
                "object_ok",
                "positive_true",
                "negative_false",
                "only_relation_differs",
                "statement_natural",
            ]
        )
        for rec in picked[:n]:
            w.writerow(
                [
                    rec["id"],
                    str(IMAGE_ROOT / rec["image"]),
                    rec["subject"],
                    rec["relation"],
                    rec["object"],
                    rec["positive_statement"],
                    rec["negative_relation"],
                    rec["negative_statement"],
                    freq[rec["relation_norm"]],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sro-ok", default=str(ROOT / "data/phase3a/sro_ok.jsonl"))
    p.add_argument("--judgements", default=str(ROOT / "data/phase3a/negative_judgements.jsonl"))
    p.add_argument("--out-dir", default=str(ROOT / "data/phase3a"))
    p.add_argument("--n", type=int, default=MASTER_N)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qc").mkdir(exist_ok=True)

    sro = {r["id"]: r for r in load_json(Path(args.sro_ok))}
    nos = first_no_by_image(load_json(Path(args.judgements)))
    paired = []
    for sid, neg in nos.items():
        src = sro.get(sid)
        if not src:
            continue
        rec = {
            "id": sid,
            "image": src["image"],
            "subject": src["subject"],
            "relation": src["relation"],
            "object": src["object"],
            "relation_norm": normalize_relation(src["relation"]),
            "subject_norm": normalize_relation(src["subject"]),
            "object_norm": normalize_relation(src["object"]),
            "positive_statement": render_statement(src["subject"], src["relation"], src["object"]),
            "negative_relation": neg["negative_relation"],
            "negative_statement": render_statement(src["subject"], neg["negative_relation"], src["object"]),
            "sro_teacher": src.get("sro_teacher") or TEACHER_NAME,
            "sro_confidence": src.get("confidence") or "high",
            "negative_teacher_judgment": "No",
            "negative_rank": neg.get("rank"),
            "split_seed": args.seed,
        }
        paired.append(rec)

    selected, sel_meta = greedy_select(paired, args.n, args.seed)
    for i, rec in enumerate(selected):
        rec["master_index"] = i
        rec["id_scale"] = f"scale_{i:04d}"

    dump_jsonl(out / "master_sro_3000.jsonl", selected)
    dump_json(out / "paired_pool_stats.json", {"n_paired": len(paired), "select": sel_meta, **diversity_stats(paired)})
    dump_json(out / "master_diversity.json", diversity_stats(selected) | {"select": sel_meta})

    for n in SCALES:
        subset = selected[:n]
        dump_jsonl(out / f"scale_{n}.jsonl", subset)
        train = expand_train(subset, f"s{n}")
        dump_json(out / f"train_s{n}.json", train)
        dump_json(out / f"scale_{n}_diversity.json", diversity_stats(subset))
        print(f"scale_{n}: pairs={len(subset)} train={len(train)}")

    write_review(selected, out / "qc/review_200.tsv", n=200, seed=args.seed)

    blob = {
        "n_sro_ok": len(sro),
        "n_images_with_no": len(nos),
        "n_paired": len(paired),
        "n_master": len(selected),
        "select": sel_meta,
        "diversity": diversity_stats(selected),
        "nested": "S534 ⊂ S1500 ⊂ S3000",
        "seed": args.seed,
    }
    dump_json(out / "assemble_meta.json", blob)
    print(json.dumps({k: v for k, v in blob.items() if k != "diversity"}, indent=2))
    print("diversity", json.dumps(blob["diversity"], indent=2))
    if len(selected) < args.n:
        print(f"NEED_MORE: selected {len(selected)} < {args.n}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
