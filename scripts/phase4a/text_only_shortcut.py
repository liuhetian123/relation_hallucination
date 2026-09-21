#!/usr/bin/env python3
"""Text-only shortcut diagnostic: old dirty negatives vs clean counterfactuals."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    DATA_DIR,
    INTERNVL_PATH,
    PLAUSIBLE_PROMPT,
    SEED,
    append_jsonl,
    dump_json,
    load_json,
    load_jsonl_map,
    parse_plausible,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase3a"))
from internvl_engine import load_internvl  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", default=str(DATA_DIR / "clean_sro3000.jsonl"))
    p.add_argument("--output", default=str(DATA_DIR / "text_only_judgements.jsonl"))
    p.add_argument("--metrics", default=str(DATA_DIR / "text_only_metrics.json"))
    p.add_argument("--model-path", default=str(INTERNVL_PATH))
    p.add_argument("--gpu", default=None)
    p.add_argument("--n", type=int, default=500, help="Number of image pairs to test")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--limit", type=int, default=None, help="Cap new judgments this run")
    return p.parse_args()


def jobs_for(rec: dict) -> list[dict]:
    sid = rec["id"]
    rows = [
        {
            "job_id": f"{sid}__pos",
            "id": sid,
            "subset": "positive",
            "gold_split": "positive",
            "subject": rec["subject"],
            "object": rec["object"],
            "relation": rec["positive_relation"],
            "statement": rec["positive_statement"],
        },
        {
            "job_id": f"{sid}__clean_neg",
            "id": sid,
            "subset": "clean_negative",
            "gold_split": "negative",
            "subject": rec["subject"],
            "object": rec["object"],
            "relation": rec["negative_relation"],
            "statement": rec["negative_statement"],
        },
    ]
    if rec.get("old_negative_statement"):
        rows.append(
            {
                "job_id": f"{sid}__old_neg",
                "id": sid,
                "subset": "old_negative",
                "gold_split": "negative",
                "subject": rec["subject"],
                "object": rec["object"],
                "relation": rec.get("old_negative_relation"),
                "statement": rec["old_negative_statement"],
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    by = {}
    for rec in rows:
        by.setdefault(rec["subset"], []).append(rec)

    def rate(items, key, val):
        if not items:
            return None
        return sum(1 for x in items if x.get("judgment") == val) / len(items)

    out = {"n_total": len(rows)}
    for subset, items in by.items():
        impl = rate(items, "judgment", "implausible")
        pl = rate(items, "judgment", "plausible")
        # Predict Negative if implausible, Positive if plausible.
        pred_neg = [x for x in items if x.get("judgment") == "implausible"]
        pred_pos = [x for x in items if x.get("judgment") == "plausible"]
        gold_neg = subset.endswith("negative")
        n = len(items)
        correct = 0
        for x in items:
            gold = "negative" if x["gold_split"] == "negative" else "positive"
            pred = None
            if x.get("judgment") == "implausible":
                pred = "negative"
            elif x.get("judgment") == "plausible":
                pred = "positive"
            if pred == gold:
                correct += 1
        out[subset] = {
            "n": n,
            "plausible_rate": pl,
            "implausible_rate": impl,
            "parse_error_rate": rate(items, "judgment", "parse_error"),
            "shortcut_acc_implausible_means_negative": correct / n if n else None,
            "n_pred_neg": len(pred_neg),
            "n_pred_pos": len(pred_pos),
            "gold_is_negative": gold_neg,
        }

    # Balanced old vs clean: pos + old_neg, pos + clean_neg
    pos = by.get("positive") or []
    old = by.get("old_negative") or []
    clean = by.get("clean_negative") or []
    pos_by = {x["id"]: x for x in pos}

    def pair_acc(negs):
        n = 0
        c = 0
        for neg in negs:
            pos_row = pos_by.get(neg["id"])
            if not pos_row:
                continue
            for row, gold in ((pos_row, "positive"), (neg, "negative")):
                pred = None
                if row.get("judgment") == "implausible":
                    pred = "negative"
                elif row.get("judgment") == "plausible":
                    pred = "positive"
                n += 1
                if pred == gold:
                    c += 1
        return {"n": n, "acc": (c / n) if n else None}

    out["paired_old_textonly_acc"] = pair_acc(old)
    out["paired_clean_textonly_acc"] = pair_acc(clean)
    return out


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    clean = load_json(Path(args.clean))
    rng = random.Random(args.seed)
    order = list(clean)
    rng.shuffle(order)
    sample = order[: args.n]
    jobs = []
    for rec in sample:
        jobs.extend(jobs_for(rec))

    out_path = Path(args.output)
    existing = load_jsonl_map(out_path, "job_id")
    remaining = [j for j in jobs if j["job_id"] not in existing]
    if args.limit is not None:
        remaining = remaining[: args.limit]
    print(
        f"pairs={len(sample)} jobs={len(jobs)} already={len(existing)} todo={len(remaining)}",
        flush=True,
    )
    if remaining:
        infer = load_internvl(args.model_path, gpu=None, max_patches=1)
        for i, job in enumerate(remaining, start=1):
            prompt = PLAUSIBLE_PROMPT.format(
                subject=job["subject"],
                candidate_relation=job["relation"],
                object=job["object"],
                candidate_statement=job["statement"],
            )
            try:
                raw = infer.generate(prompt, image=None, max_new_tokens=16)
                job["teacher_raw_output"] = raw
                job["judgment"] = parse_plausible(raw)
            except Exception as exc:
                job["teacher_raw_output"] = repr(exc)
                job["judgment"] = "error"
            append_jsonl(out_path, job)
            if i % 20 == 0 or i == 1:
                print(
                    f"[{i}/{len(remaining)}] {job['job_id']} -> {job.get('judgment')}",
                    flush=True,
                )

    all_rows = list(load_jsonl_map(out_path, "job_id").values())
    metrics = summarize(all_rows)
    dump_json(Path(args.metrics), metrics)
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
