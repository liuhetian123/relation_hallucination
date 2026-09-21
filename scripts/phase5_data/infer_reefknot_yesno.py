#!/usr/bin/env python3
"""Greedy Qwen2.5-VL inference on the official Reefknot YESNO prompts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--question_file", required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--answers_file", required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    questions = [
        json.loads(line)
        for line in Path(args.question_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        questions = questions[: args.limit]
    output = Path(args.answers_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        done = {
            json.loads(line)["question_id"]
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    ).eval()
    image_root = Path(args.image_folder)
    with output.open("a", encoding="utf-8") as stream:
        for index, item in enumerate(questions):
            question_id = index
            if question_id in done:
                continue
            image_id = str(item["image_id"])
            image_path = image_root / f"{image_id}.jpg"
            record = {
                "question_id": question_id,
                "image_id": image_id,
                "prompt": item["query_prompt"],
                "label": item["label"],
                "relation_type": item["relation_type"],
                "model_id": Path(args.model_path).name,
            }
            try:
                image = Image.open(image_path).convert("RGB")
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": item["query_prompt"]},
                    ],
                }]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
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
                inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
                generated = generated[:, inputs["input_ids"].shape[1]:]
                record["text"] = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            except Exception as exc:
                record.update(text="", error=repr(exc))
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if index == 0 or (index + 1) % 20 == 0:
                print(f"[{index + 1}/{len(questions)}] {record.get('text', '')!r}", flush=True)


if __name__ == "__main__":
    main()
