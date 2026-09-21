#!/usr/bin/env python3
"""Annotate RelSim images with InternVL3.5-8B structured SRO JSON. Resumable JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    IMAGE_ROOT,
    INTERNVL_PATH,
    ROOT,
    SRO_PROMPT,
    TEACHER_NAME,
    append_jsonl,
    extract_json_object,
    load_json,
    load_jsonl_map,
    qc_sro,
)
from internvl_engine import load_internvl, open_rgb


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", default=str(ROOT / "data/phase3a/candidates.jsonl"))
    p.add_argument("--output", default=str(ROOT / "data/phase3a/sro_raw.jsonl"))
    p.add_argument("--image-root", default=str(IMAGE_ROOT))
    p.add_argument("--model-path", default=str(INTERNVL_PATH))
    p.add_argument("--gpu", default=None)
    p.add_argument("--max-patches", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--target-ok", type=int, default=5000)
    p.add_argument("--max-candidates", type=int, default=12000)
    p.add_argument("--limit", type=int, default=None, help="Process at most N new images this run")
    return p.parse_args()


def count_ok(path: Path) -> int:
    n = 0
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("qc_status") == "ok":
                n += 1
    return n


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cands = load_json(Path(args.candidates))[: args.max_candidates]
    out_path = Path(args.output)
    existing = load_jsonl_map(out_path, "id")
    n_ok = sum(1 for r in existing.values() if r.get("qc_status") == "ok")
    print(f"candidates={len(cands)} already={len(existing)} ok={n_ok} target_ok={args.target_ok}", flush=True)
    if n_ok >= args.target_ok:
        print("target already met")
        return

    infer = load_internvl(args.model_path, gpu=None, max_patches=args.max_patches)
    image_root = Path(args.image_root)
    processed = 0
    for rec in cands:
        if n_ok >= args.target_ok:
            break
        sid = rec["id"]
        if sid in existing:
            continue
        if args.limit is not None and processed >= args.limit:
            break
        path = image_root / rec["image"]
        row = {
            "id": sid,
            "image": rec["image"],
            "rank": rec.get("rank"),
            "sro_teacher": TEACHER_NAME,
        }
        try:
            image = open_rgb(path)
            raw = infer(image, SRO_PROMPT, max_new_tokens=args.max_new_tokens)
            parsed, repaired = extract_json_object(raw)
            status, fields = qc_sro(parsed)
            row.update(
                {
                    "teacher_raw_output": raw,
                    "json_repaired": repaired,
                    "qc_status": status,
                    **fields,
                }
            )
        except Exception as exc:
            row["teacher_raw_output"] = repr(exc)
            row["qc_status"] = "inference_error"
        append_jsonl(out_path, row)
        existing[sid] = row
        processed += 1
        if row.get("qc_status") == "ok":
            n_ok += 1
        if processed % 10 == 0 or processed == 1 or row.get("qc_status") == "ok":
            print(
                f"[{processed} new] {sid} status={row.get('qc_status')} "
                f"ok_total={n_ok} {row.get('subject')} | {row.get('relation')} | {row.get('object')}",
                flush=True,
            )
    print(f"done new={processed} ok_total={n_ok} output={out_path}", flush=True)


if __name__ == "__main__":
    main()
