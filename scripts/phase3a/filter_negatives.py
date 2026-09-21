#!/usr/bin/env python3
"""Filter random-negative candidates with InternVL3.5-8B. Keep only explicit No."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    IMAGE_ROOT,
    INTERNVL_PATH,
    NEG_PROMPT,
    ROOT,
    append_jsonl,
    load_json,
    parse_yes_no_uncertain,
)
from internvl_engine import load_internvl, open_rgb


def load_existing(path: Path) -> dict:
    done = {}
    if not path.exists():
        return done
    import json

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                done[rec["job_id"]] = rec
    return done


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", default=str(ROOT / "data/phase3a/negative_jobs.jsonl"))
    p.add_argument("--output", default=str(ROOT / "data/phase3a/negative_judgements.jsonl"))
    p.add_argument("--image-root", default=str(IMAGE_ROOT))
    p.add_argument("--model-path", default=str(INTERNVL_PATH))
    p.add_argument("--gpu", default=None)
    p.add_argument("--max-patches", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--early-stop", action="store_true", default=True)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    early = False if args.no_early_stop else args.early_stop

    jobs = load_json(Path(args.jobs))
    jobs.sort(key=lambda j: (j["id"], j.get("rank", 99)))
    out_path = Path(args.output)
    existing = load_existing(out_path)
    got_no = set()
    for rec in existing.values():
        if rec.get("judgment") == "no":
            got_no.add(rec["id"])

    remaining = []
    for job in jobs:
        if job["job_id"] in existing:
            continue
        if early and job["id"] in got_no:
            continue
        remaining.append(job)
    print(
        f"jobs={len(jobs)} already={len(existing)} remaining={len(remaining)} "
        f"images_with_no={len(got_no)} early_stop={early}",
        flush=True,
    )
    if not remaining:
        return

    infer = load_internvl(args.model_path, gpu=None, max_patches=args.max_patches)
    image_root = Path(args.image_root)
    processed = 0
    for job in remaining:
        if early and job["id"] in got_no:
            continue
        if args.limit is not None and processed >= args.limit:
            break
        rec = dict(job)
        path = image_root / job["image"]
        try:
            image = open_rgb(path)
            prompt = NEG_PROMPT.format(
                subject=job["subject"],
                object=job["object"],
                negative_relation=job["negative_relation"],
            )
            raw = infer(image, prompt, max_new_tokens=args.max_new_tokens)
            rec["teacher_raw_output"] = raw
            rec["judgment"] = parse_yes_no_uncertain(raw)
        except Exception as exc:
            rec["teacher_raw_output"] = repr(exc)
            rec["judgment"] = "error"
        append_jsonl(out_path, rec)
        existing[job["job_id"]] = rec
        processed += 1
        if rec.get("judgment") == "no":
            got_no.add(job["id"])
        if processed % 10 == 0 or processed == 1:
            print(
                f"[{processed}/{len(remaining)}] {job['id']} {job['negative_relation']} -> {rec.get('judgment')} "
                f"images_with_no={len(got_no)}",
                flush=True,
            )
    print(f"Wrote {out_path} new={processed} images_with_no={len(got_no)}", flush=True)


if __name__ == "__main__":
    main()
