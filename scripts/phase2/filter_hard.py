#!/usr/bin/env python3
"""Filter hard-negative candidates with InternVL3.5-8B (Yes / No / Uncertain)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, dump_jsonl, load_json, parse_yes_no_uncertain

from PIL import Image

JUDGE_PROMPT = """You are given an image and two relational statements about the same visible entities.

Original statement:
"{original}"

Candidate statement:
"{candidate}"

Task: decide whether the CANDIDATE relation is visually supported between the same referenced entities.

Rules:
1. Judge only the relation, not wording style.
2. If the candidate relation is clearly visible, answer Yes.
3. If the candidate relation is clearly not visible, answer No.
4. If both could be true, the image is too ambiguous, or you cannot tell, answer Uncertain.
5. Do not rewrite the statements.
6. Reply with exactly one word: Yes, No, or Uncertain.
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", default=str(ROOT / "data/phase2/hard_candidate_jobs.jsonl"))
    p.add_argument("--output", default=str(ROOT / "data/phase2/hard_candidate_judgements.jsonl"))
    p.add_argument("--train_image_root", default=str(ROOT / "data/relsim_images"))
    p.add_argument("--heldout_image_root", default="/data/lht/relsim_dataset/relsim_images")
    p.add_argument(
        "--model_path",
        default="/data/storage22t/lht/hf_cache/models--OpenGVLab--InternVL3_5-8B-HF/snapshots/741a7d03020411e666c6109218ab71e08151ef86",
    )
    p.add_argument("--gpu", default="3")
    p.add_argument("--max_patches", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--early_stop_per_image", action="store_true", default=True)
    p.add_argument("--no_early_stop", action="store_true")
    return p.parse_args()


def load_existing(path: Path) -> dict:
    done = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                done[rec["job_id"]] = rec
    return done


def image_path(job: dict, train_root: Path, ho_root: Path) -> Path:
    name = Path(job["image"]).name
    root = train_root if job["split"] == "train" else ho_root
    return root / name


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    early = False if args.no_early_stop else args.early_stop_per_image

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    jobs = load_json(Path(args.jobs))
    jobs.sort(key=lambda j: (j["split"], j["source_id"], j.get("rank", 99)))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(out_path)

    got_no = set()
    for rec in existing.values():
        if rec.get("judgment") == "no":
            got_no.add((rec["split"], rec["source_id"]))

    remaining = []
    for job in jobs:
        if job["job_id"] in existing:
            continue
        key = (job["split"], job["source_id"])
        if early and key in got_no:
            continue
        remaining.append(job)
    print(f"jobs={len(jobs)} already={len(existing)} remaining={len(remaining)} early_stop={early}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    device = next(model.parameters()).device
    train_root, ho_root = Path(args.train_image_root), Path(args.heldout_image_root)

    def infer(image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc_kwargs = {
            "text": [text],
            "images": [image],
            "return_tensors": "pt",
            "crop_to_patches": True,
            "max_patches": args.max_patches,
        }
        inputs = processor(**proc_kwargs)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
        gen = output_ids[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    with out_path.open("a", encoding="utf-8") as fout:
        for i, job in enumerate(remaining, 1):
            key = (job["split"], job["source_id"])
            if early and key in got_no:
                continue
            path = image_path(job, train_root, ho_root)
            rec = dict(job)
            try:
                image = Image.open(path).convert("RGB")
                prompt = JUDGE_PROMPT.format(
                    original=job["original_statement"],
                    candidate=job["candidate_statement"],
                )
                raw = infer(image, prompt)
                rec["teacher_raw_output"] = raw
                rec["judgment"] = parse_yes_no_uncertain(raw)
            except Exception as exc:
                rec["teacher_raw_output"] = repr(exc)
                rec["judgment"] = "error"
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if rec.get("judgment") == "no":
                got_no.add(key)
            if i % 10 == 0 or i == 1:
                print(
                    f"[{i}/{len(remaining)}] {job['split']} {job['source_id']} "
                    f"{job['candidate_relation']} -> {rec.get('judgment')}",
                    flush=True,
                )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
