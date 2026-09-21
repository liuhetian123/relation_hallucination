#!/usr/bin/env python3
"""Assemble Clean-SRO3000, diversity stats, and stratified 300-pair QC table."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    DATA_DIR,
    IMAGE_ROOT,
    SEED,
    TEACHER_NAME,
    dump_json,
    dump_jsonl,
    load_json,
    qc_family,
    render_statement,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempts", default=str(DATA_DIR / "clean_attempts.jsonl"))
    p.add_argument("--out-clean", default=str(DATA_DIR / "clean_sro3000.jsonl"))
    p.add_argument("--out-dropped", default=str(DATA_DIR / "dropped.jsonl"))
    p.add_argument("--out-stats", default=str(DATA_DIR / "assemble_stats.json"))
    p.add_argument("--out-qc", default=str(DATA_DIR / "qc/review_300.tsv"))
    p.add_argument("--qc-n", type=int, default=300)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def compact_candidates(attempts: list) -> list[dict]:
    out = []
    for att in attempts or []:
        for cand in att.get("candidates") or []:
            out.append(
                {
                    "attempt": att.get("attempt"),
                    "rank": cand.get("rank"),
                    "relation": cand.get("relation"),
                    "status": cand.get("status"),
                    "confidence": cand.get("confidence"),
                    "linguistically_plausible": cand.get("linguistically_plausible"),
                    "visually_supported": cand.get("visually_supported"),
                    "text_judgment": cand.get("text_judgment"),
                    "visual_judgment": cand.get("visual_judgment"),
                }
            )
    return out


def to_clean_row(rec: dict) -> dict:
    subject = rec["subject"]
    obj = rec["object"]
    pos_rel = rec["positive_relation"]
    neg_rel = rec["negative_relation"]
    return {
        "id": rec["id"],
        "image": rec["image"],
        "subject": subject,
        "positive_relation": pos_rel,
        "object": obj,
        "positive_statement": rec.get("positive_statement")
        or render_statement(subject, pos_rel, obj),
        "negative_relation": neg_rel,
        "negative_statement": rec.get("negative_statement")
        or render_statement(subject, neg_rel, obj),
        "negative_source": rec.get("negative_source", TEACHER_NAME),
        "linguistically_plausible": True,
        "visually_supported": False,
        "negative_confidence": rec.get("negative_confidence", "high"),
        "candidate_rank": rec.get("candidate_rank"),
        "attempt": rec.get("attempt"),
        "old_negative_relation": rec.get("old_negative_relation"),
        "old_negative_statement": rec.get("old_negative_statement"),
        "text_judgment": rec.get("text_judgment"),
        "visual_judgment": rec.get("visual_judgment"),
        "master_index": rec.get("master_index"),
        "id_scale": rec.get("id_scale"),
        "negative_candidates_raw": compact_candidates(rec.get("attempts")),
    }


def stratified_qc(rows: list[dict], n: int, seed: int) -> list[dict]:
    buckets = {"spatial": [], "contact": [], "other": []}
    for rec in rows:
        fam = qc_family(rec["positive_relation"])
        buckets[fam].append(rec)
    rng = random.Random(seed)
    for fam in buckets:
        rng.shuffle(buckets[fam])
    target = {"spatial": n // 3, "contact": n // 3, "other": n - 2 * (n // 3)}
    picked = []
    leftover = []
    for fam, k in target.items():
        take = buckets[fam][:k]
        picked.extend(take)
        leftover.extend(buckets[fam][k:])
    if len(picked) < n:
        rng.shuffle(leftover)
        picked.extend(leftover[: n - len(picked)])
    return picked[:n]


def grammar_smell(rel: str) -> bool:
    """Crude cue that 'A X is {rel} a Y' is ungrammatical (finite 3sg, not participle/prep)."""
    head = (rel or "").strip().split()[0].lower() if rel else ""
    if not head:
        return True
    if head.endswith("ing") or head.endswith("ed"):
        return False
    preps = {
        "on",
        "in",
        "at",
        "by",
        "near",
        "over",
        "under",
        "behind",
        "beside",
        "inside",
        "above",
        "below",
        "against",
        "around",
        "across",
        "along",
        "with",
        "from",
        "into",
        "onto",
        "upon",
        "atop",
        "between",
        "among",
    }
    if head in preps:
        return False
    if len(head) > 3 and head.endswith("s") and not head.endswith("ss"):
        return True
    return False


def main() -> None:
    args = parse_args()
    attempts = load_json(Path(args.attempts))
    ok = [r for r in attempts if r.get("qc_status") == "ok" and r.get("negative_relation")]
    dropped = [r for r in attempts if r.get("qc_status") != "ok"]
    clean = [to_clean_row(r) for r in ok]
    dump_jsonl(Path(args.out_clean), clean)
    dump_jsonl(Path(args.out_dropped), dropped)

    drop_reasons = Counter(r.get("drop_reason") or "unknown" for r in dropped)
    rank = Counter(r.get("candidate_rank") for r in clean)
    fam = Counter(qc_family(r["positive_relation"]) for r in clean)
    old_smell = sum(1 for r in clean if grammar_smell(r.get("old_negative_relation") or ""))
    new_smell = sum(1 for r in clean if grammar_smell(r.get("negative_relation") or ""))
    same_as_old = sum(
        1
        for r in clean
        if (r.get("old_negative_relation") or "").strip().lower()
        == (r.get("negative_relation") or "").strip().lower()
    )
    stats = {
        "n_attempts": len(attempts),
        "n_clean": len(clean),
        "n_dropped": len(dropped),
        "keep_rate": (len(clean) / len(attempts)) if attempts else 0.0,
        "drop_reasons": dict(drop_reasons),
        "candidate_rank": {str(k): v for k, v in rank.items()},
        "positive_relation_family": dict(fam),
        "unique_negative_relations": len({r["negative_relation"] for r in clean}),
        "unique_old_negative_relations": len(
            {r.get("old_negative_relation") for r in clean if r.get("old_negative_relation")}
        ),
        "n_negative_equals_old": same_as_old,
        "old_negative_grammar_smell": old_smell,
        "clean_negative_grammar_smell": new_smell,
        "old_negative_grammar_smell_rate": old_smell / len(clean) if clean else 0.0,
        "clean_negative_grammar_smell_rate": new_smell / len(clean) if clean else 0.0,
        "top_negative_relations": Counter(r["negative_relation"] for r in clean).most_common(20),
        "top_old_negative_relations": Counter(
            r.get("old_negative_relation") or "" for r in clean
        ).most_common(20),
    }
    dump_json(Path(args.out_stats), stats)

    qc = stratified_qc(clean, args.qc_n, args.seed)
    qc_path = Path(args.out_qc)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    with qc_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "id",
                "image_path",
                "family",
                "subject",
                "positive_relation",
                "object",
                "positive_statement",
                "negative_relation",
                "negative_statement",
                "old_negative_statement",
                "candidate_rank",
                "pos_true",
                "neg_natural",
                "neg_plausible",
                "neg_visually_false",
                "needs_image",
                "is_synonym",
                "role_reversal",
                "ambiguous",
                "overall_clean",
                "notes",
            ]
        )
        for rec in qc:
            w.writerow(
                [
                    rec["id"],
                    str(IMAGE_ROOT / rec["image"]),
                    qc_family(rec["positive_relation"]),
                    rec["subject"],
                    rec["positive_relation"],
                    rec["object"],
                    rec["positive_statement"],
                    rec["negative_relation"],
                    rec["negative_statement"],
                    rec.get("old_negative_statement") or "",
                    rec.get("candidate_rank"),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
    print(
        f"clean={len(clean)} dropped={len(dropped)} keep_rate={stats['keep_rate']:.3f} "
        f"qc={len(qc)} -> {args.out_clean}",
        flush=True,
    )


if __name__ == "__main__":
    main()
