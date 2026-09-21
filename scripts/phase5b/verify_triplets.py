#!/usr/bin/env python3
"""Verify PSG VCG candidates with an independent InternVL model family."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = str(Path.home() / ".cache/huggingface/hub/models--OpenGVLab--InternVL3_5-8B-HF")


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int)
    return p.parse_args()


def resolve_model(path: str) -> str:
    root = Path(path)
    snaps = list((root / "snapshots").glob("*")) if (root / "snapshots").is_dir() else []
    return str(snaps[0] if snaps else root)


def mark(path: str, subject_box: list, object_box: list) -> Image.Image:
    image = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(image)
    width = max(4, image.width // 180)
    d.rectangle(subject_box, outline="#00ff00", width=width)
    d.rectangle(object_box, outline="#ff2020", width=width)
    return image


def parse(text: str) -> str:
    match = re.search(r"\b(yes|no|uncertain)\b", text.lower())
    return match.group(1) if match else "uncertain"


def main() -> None:
    a = args()
    model_path = resolve_model(a.model)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=a.device, local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    rows = [json.loads(x) for x in a.input.read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.limit:
        rows = rows[:a.limit]
    done = {}
    if a.output.exists():
        done = {r["id"]: r for x in a.output.read_text(encoding="utf-8").splitlines() if x.strip() for r in [json.loads(x)]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("a", encoding="utf-8") as out, torch.inference_mode():
        for i, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            checks = {}
            specs = [
                ("factual", row["image"], row["subject_box"], row["object_box"], row["predicate"]),
                ("relation_cf", row["image"], row["subject_box"], row["object_box"], row["relation_cf_predicate"]),
                ("visual_cf", row["visual_cf"]["image"], row["visual_cf"]["subject_box"],
                 row["visual_cf"]["object_box"], row["predicate"]),
            ]
            for key, path, sb, ob, predicate in specs:
                image = mark(path, sb, ob)
                prompt = (
                    "Inspect the marked entities: the SUBJECT is inside the green box and the OBJECT is inside "
                    f"the red box. Is it visually clear that the SUBJECT ({row['subject']}) {predicate} "
                    f"the OBJECT ({row['object']})? Reply with exactly Yes, No, or Uncertain. "
                    "Use Uncertain if the boxes, relation, or evidence are ambiguous."
                )
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                    return_tensors="pt",
                ).to(a.device)
                generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                generated = generated[:, inputs["input_ids"].shape[1]:]
                raw = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
                checks[key] = {"judgment": parse(raw), "raw": raw}
            record = {
                "id": row["id"], "verifier_model": "OpenGVLab/InternVL3.5-8B-HF",
                "generator": "PSG ground-truth plus deterministic annotation rules", **checks,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{i}/{len(rows)} {row['id']} " + " ".join(f"{k}={v['judgment']}" for k, v in checks.items()), flush=True)


if __name__ == "__main__":
    main()
