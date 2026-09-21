#!/usr/bin/env python3
"""Greedy VQA inference for Qwen2.5-VL, with optional LoRA adapter and JSONL resume."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_existing(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec.get("question_id", rec.get("id"))
            if qid is not None:
                done.add(qid)
    return done


def load_questions(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions") or data.get("data") or list(data.values())
    return data


def question_text(item: dict) -> str:
    return (
        item.get("text")
        or item.get("question")
        or item.get("query")
        or ""
    ).strip()


def image_name(item: dict) -> str:
    return item.get("image") or item.get("image_path") or item.get("img") or ""


def question_id(item: dict, idx: int):
    if "question_id" in item:
        return item["question_id"]
    if "id" in item:
        return item["id"]
    return idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--question_file", required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--answers_file", required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", default=None, help="Keep only this POPE/MMRel category if present")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    questions = load_questions(Path(args.question_file))
    if args.category:
        questions = [q for q in questions if q.get("category") == args.category or q.get("split") == args.category]
    if args.limit is not None:
        questions = questions[: args.limit]

    out_path = Path(args.answers_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_existing(out_path)
    remaining = [q for i, q in enumerate(questions) if question_id(q, i) not in done]
    print(f"Questions={len(questions)} already={len(done)} remaining={len(remaining)}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model_id = Path(args.adapter_path).name if args.adapter_path else Path(args.model_path).name
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    image_folder = Path(args.image_folder)
    with out_path.open("a", encoding="utf-8") as fout:
        for i, item in enumerate(remaining, 1):
            qid = question_id(item, i)
            prompt = question_text(item)
            img_name = image_name(item)
            img_path = image_folder / img_name
            if not img_path.exists():
                # Some AMBER files already include a folder prefix.
                alt = image_folder / Path(img_name).name
                img_path = alt if alt.exists() else img_path
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as exc:
                rec = {
                    "question_id": qid,
                    "prompt": prompt,
                    "text": "",
                    "model_id": model_id,
                    "error": repr(exc),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{i}/{len(remaining)}] {qid} image_error {exc}", flush=True)
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            vision = process_vision_info(messages)
            image_inputs = vision[0] if vision else [image]
            video_inputs = vision[1] if vision and len(vision) > 1 else None
            size = getattr(processor.image_processor, "size", None)
            if size is not None:
                size["longest_edge"] = args.max_pixels
                size["shortest_edge"] = args.min_pixels
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )
            gen = output_ids[:, inputs["input_ids"].shape[1] :]
            answer = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
            rec = {
                "question_id": qid,
                "prompt": prompt,
                "text": answer,
                "model_id": model_id,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 20 == 0 or i == 1:
                print(f"[{i}/{len(remaining)}] {qid} {answer[:80]!r}", flush=True)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
