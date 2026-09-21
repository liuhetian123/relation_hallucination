#!/usr/bin/env python3
"""Score held-out matched Explicit/Typed captions with teacher-forced NLL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_nll import (  # noqa: E402
    TRAIN_PROMPT,
    load_existing,
    load_json,
    score_one,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--run_name", required=True)
    p.add_argument("--gpu", default=None)
    p.add_argument(
        "--matched_jsonl",
        default=str(ROOT / "eval_results/qwen/phase16b/heldout_matched.jsonl"),
    )
    p.add_argument("--image_root", default="/data/lht/relsim_dataset/relsim_images")
    p.add_argument("--out_dir", default=str(ROOT / "eval_results/qwen/phase16b/scores"))
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    rows = load_json(Path(args.matched_jsonl))
    if args.limit:
        rows = rows[: args.limit]
    image_root = Path(args.image_root)

    jobs = []
    for r in rows:
        rel = r.get("gt_relation") or None
        jobs.append(
            {
                "id": r["id"],
                "split": "heldout_typed",
                "image": image_root / Path(r["image"]).name,
                "target": r["anonymous_caption"],
                "relation": rel if r.get("rel_in_typed") else None,
            }
        )
        if r.get("ok_explicit") and r.get("explicit_caption"):
            jobs.append(
                {
                    "id": r["id"],
                    "split": "heldout_explicit",
                    "image": image_root / Path(r["image"]).name,
                    "target": r["explicit_caption"],
                    "relation": rel if r.get("rel_in_explicit") else None,
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_name}.jsonl"
    done = load_existing(out_path)
    remaining = [j for j in jobs if (j["id"], j["split"]) not in done]
    print(f"jobs={len(jobs)} already={len(done)} remaining={len(remaining)}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        print(f"Loaded LoRA {args.adapter_path}")
    model.eval()
    device = model.device

    with out_path.open("a", encoding="utf-8") as fout:
        for i, job in enumerate(remaining, 1):
            try:
                image = Image.open(job["image"]).convert("RGB")
            except Exception as exc:
                rec = {"id": job["id"], "split": job["split"], "run": args.run_name, "error": repr(exc)}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                continue
            stats = score_one(
                model, processor, image, TRAIN_PROMPT, job["target"], job["relation"], device
            )
            rec = {
                "id": job["id"],
                "split": job["split"],
                "run": args.run_name,
                "target": job["target"],
                "relation": job["relation"],
            }
            if stats is None:
                rec["error"] = "empty_target"
            else:
                rec.update(stats)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 20 == 0 or i == 1:
                print(
                    f"[{i}/{len(remaining)}] {job['split']} {job['id']} "
                    f"full={rec.get('full_mean_nll')} rel={rec.get('rel_mean_nll')}",
                    flush=True,
                )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
