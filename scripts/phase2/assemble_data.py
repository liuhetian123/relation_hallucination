#!/usr/bin/env python3
"""Assemble V3 training data and frozen held-out verification test after InternVL filter."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ROOT,
    dump_json,
    dump_jsonl,
    llava_verify_sample,
    load_json,
    verification_prompt,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=str(ROOT / "data/phase2"))
    p.add_argument("--heldout_n_random", type=int, default=100)
    p.add_argument("--heldout_n_hard", type=int, default=100)
    p.add_argument("--review_n", type=int, default=100)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def first_no_per_image(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: (r["split"], r["source_id"], r.get("rank", 99)))
    picked = {}
    for rec in rows:
        if rec.get("judgment") != "no":
            continue
        key = (rec["split"], rec["source_id"])
        if key not in picked:
            picked[key] = rec
    return picked


def to_neg_sample(rec: dict, split: str) -> dict:
    sid = f"{rec['source_id']}__hard__{rec['candidate_relation'].replace(' ', '_')}"
    return llava_verify_sample(
        sid,
        rec["image"],
        rec["candidate_statement"],
        "No",
        source_id=rec["source_id"],
        split=split,
        subset="hard_negative",
        gt_relation=rec["gt_relation"],
        neg_relation=rec["candidate_relation"],
        explicit_caption=rec["original_statement"],
        internvl_judgment=rec.get("judgment"),
        internvl_raw=rec.get("teacher_raw_output"),
        hard_rank=rec.get("rank"),
    )


def to_question(rec: dict, qid: str) -> dict:
    label = rec["label"]
    gold = "yes" if str(label).lower() == "yes" else "no"
    return {
        "question_id": qid,
        "id": rec.get("source_id") or rec["id"],
        "image": rec["image"],
        "text": verification_prompt(rec["statement"]),
        "statement": rec["statement"],
        "label": gold,
        "subset": rec.get("subset"),
        "gt_relation": rec.get("gt_relation"),
        "neg_relation": rec.get("neg_relation"),
        "split": rec.get("split"),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    rng = random.Random(args.seed)

    positives = load_json(out / "v1_positive.json")
    train_random = load_json(out / "train_random_negatives.jsonl")
    ho_random = load_json(out / "heldout_random_negatives.jsonl")
    judgements = load_json(out / "hard_candidate_judgements.jsonl")

    status = defaultdict(int)
    for rec in judgements:
        status[(rec["split"], rec.get("judgment", "missing"))] += 1
    picked = first_no_per_image(judgements)
    train_hard = [to_neg_sample(v, "train") for k, v in picked.items() if k[0] == "train"]
    ho_hard_all = [to_neg_sample(v, "heldout") for k, v in picked.items() if k[0] == "heldout"]

    v3 = positives + train_hard
    rng.shuffle(v3)
    dump_json(out / "v3_pos_hard.json", v3)
    dump_jsonl(out / "train_hard_negatives.jsonl", train_hard)

    pos_by_id = {p["source_id"]: p for p in positives}
    ho_pos = []
    ho_matched = load_json(ROOT / "eval_results/qwen/phase16b/heldout_matched.jsonl")
    for h in ho_matched:
        if not h.get("ok_explicit"):
            continue
        ho_pos.append(
            llava_verify_sample(
                f"{h['id']}__pos",
                h["image"],
                h["explicit_caption"],
                "Yes",
                source_id=h["id"],
                split="heldout",
                subset="positive",
                gt_relation=h.get("gt_relation"),
                explicit_caption=h["explicit_caption"],
            )
        )

    n_rand = min(args.heldout_n_random, len(ho_random))
    n_hard = min(args.heldout_n_hard, len(ho_hard_all))
    ho_rand_sel = sorted(ho_random, key=lambda r: r["id"])
    rng.shuffle(ho_rand_sel)
    ho_rand_sel = ho_rand_sel[:n_rand]
    ho_hard_sel = sorted(ho_hard_all, key=lambda r: r["id"])
    rng.shuffle(ho_hard_sel)
    ho_hard_sel = ho_hard_sel[:n_hard]

    questions = []
    for rec in ho_pos:
        questions.append(to_question(rec, rec["id"]))
    for rec in ho_rand_sel:
        questions.append(to_question(rec, rec["id"]))
    for rec in ho_hard_sel:
        questions.append(to_question(rec, rec["id"]))
    questions.sort(key=lambda q: (q["subset"], q["question_id"]))
    dump_jsonl(out / "heldout_verification.jsonl", questions)
    dump_jsonl(out / "heldout_hard_negatives.jsonl", ho_hard_all)

    # frozen copy under eval_results
    eval_dir = ROOT / "eval_results/qwen/phase2"
    dump_jsonl(eval_dir / "heldout_verification.jsonl", questions)

    review_pool = train_random + train_hard
    n_each = min(args.review_n // 2, len(train_random), len(train_hard) if train_hard else 0)
    review = []
    if train_random:
        review.extend(rng.sample(train_random, min(n_each or args.review_n // 2, len(train_random))))
    if train_hard:
        review.extend(rng.sample(train_hard, min(n_each or args.review_n // 2, len(train_hard))))
    review.sort(key=lambda r: r["id"])
    tsv = eval_dir / "review_100.tsv"
    tsv.parent.mkdir(parents=True, exist_ok=True)
    with tsv.open("w", encoding="utf-8") as f:
        f.write("id\tsubset\timage\tgt_relation\tneg_relation\tstatement\toriginal\n")
        for rec in review:
            f.write(
                "\t".join(
                    [
                        rec["id"],
                        rec.get("subset", ""),
                        rec.get("image", ""),
                        rec.get("gt_relation") or "",
                        rec.get("neg_relation") or "",
                        (rec.get("statement") or "").replace("\t", " "),
                        (rec.get("explicit_caption") or "").replace("\t", " "),
                    ]
                )
                + "\n"
            )

    qc = {
        "internvl_status_counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(status.items())},
        "n_train_hard_images": len(train_hard),
        "n_heldout_hard_images": len(ho_hard_all),
        "n_v1": len(positives),
        "n_v2": len(positives) + len(train_random),
        "n_v3": len(v3),
        "heldout_positive": len(ho_pos),
        "heldout_random_selected": len(ho_rand_sel),
        "heldout_hard_selected": len(ho_hard_sel),
        "heldout_total": len(questions),
        "review_n": len(review),
    }
    dump_json(out / "assemble_qc.json", qc)
    dump_json(eval_dir / "assemble_qc.json", qc)
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
