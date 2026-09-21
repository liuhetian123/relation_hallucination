#!/usr/bin/env python3
"""Greedy MCQ inference for Qwen2.5-VL with optional LoRA and option log-probs."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

LETTERS = ("A", "B", "C", "D")


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


def parse_choice(text: str) -> str:
    """Map a generation to A/B/C/D. Do not search letters inside option words."""
    raw = (text or "").strip()
    if not raw:
        return "invalid"
    m = re.match(
        r"(?:the\s+answer\s+is\s+|answer\s*(?:is)?\s*[:\-]?\s*)?\(?([ABCD])\)?(?:[.\):]|\s|$)",
        raw,
        re.I,
    )
    if m:
        return m.group(1).upper()
    return "invalid"


def prompt_of(item: dict) -> str:
    if item.get("text"):
        return item["text"].strip()
    options = item.get("options") or {}
    lines = [
        item.get("question") or "Which relation best describes the primary relation shown in the image?",
        "",
    ]
    for letter in LETTERS:
        lines.append(f"{letter}. {options[letter]}")
    lines.extend(["", "Answer with A, B, C, or D only."])
    return "\n".join(lines)


def letter_token_ids(tokenizer, letter: str) -> list[int]:
    variants = [letter, f" {letter}"]
    for text in variants:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            return ids
    raise RuntimeError(f"Cannot tokenize choice letter {letter}")


def score_letters(model, tokenizer, inputs: dict) -> dict[str, float]:
    """Sequence log-likelihood of each choice letter given the same prompt prefix."""
    with torch.inference_mode():
        out = model(**inputs)
        next_logits = out.logits[:, -1, :]
        logp = F.log_softmax(next_logits, dim=-1)[0]
    scores = {}
    for letter in LETTERS:
        ids = letter_token_ids(tokenizer, letter)
        if len(ids) == 1:
            scores[letter] = float(logp[ids[0]].item())
            continue
        # Rare multi-token label: teacher-force the continuation.
        extra = torch.tensor([ids], device=inputs["input_ids"].device)
        full_ids = torch.cat([inputs["input_ids"], extra], dim=1)
        attn = torch.cat(
            [inputs["attention_mask"], torch.ones_like(extra)],
            dim=1,
        )
        forced = {k: v for k, v in inputs.items() if k not in {"input_ids", "attention_mask"}}
        forced["input_ids"] = full_ids
        forced["attention_mask"] = attn
        with torch.inference_mode():
            logits = model(**forced).logits
        prompt_len = inputs["input_ids"].shape[1]
        step_logp = F.log_softmax(logits[0, prompt_len - 1 : prompt_len - 1 + len(ids), :], dim=-1)
        total = 0.0
        for i, tok in enumerate(ids):
            total += float(step_logp[i, tok].item())
        scores[letter] = total
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--question_file", required=True)
    parser.add_argument("--image_folder", default="/data/lht/relsim_dataset/relsim_images")
    parser.add_argument("--answers_file", required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--no_logprob", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    questions = load_questions(Path(args.question_file))
    out_path = Path(args.answers_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_existing(out_path)
    remaining = [q for q in questions if q.get("id", q.get("question_id")) not in done]
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
        print(f"Loaded LoRA adapter from {args.adapter_path}")
    model.eval()
    tokenizer = processor.tokenizer

    image_folder = Path(args.image_folder)
    with out_path.open("a", encoding="utf-8") as fout:
        for i, item in enumerate(remaining, 1):
            qid = item.get("id", item.get("question_id"))
            prompt = prompt_of(item)
            img_path = image_folder / Path(item.get("image") or "").name
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as exc:
                rec = {
                    "question_id": qid,
                    "prompt": prompt,
                    "text": "",
                    "parsed": "invalid",
                    "model_id": model_id,
                    "error": repr(exc),
                    "gold": item.get("answer"),
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

            scores = None
            if not args.no_logprob:
                scores = score_letters(model, tokenizer, inputs)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )
            gen = output_ids[:, inputs["input_ids"].shape[1] :]
            answer = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
            parsed = parse_choice(answer)
            gold = item.get("answer")
            rec = {
                "question_id": qid,
                "prompt": prompt,
                "text": answer,
                "parsed": parsed,
                "gold": gold,
                "correct": parsed == gold if parsed != "invalid" else False,
                "model_id": model_id,
            }
            if scores is not None:
                rec["option_logprob"] = scores
                wrong = [scores[l] for l in LETTERS if l != gold]
                rec["gt_margin"] = scores[gold] - max(wrong) if gold in scores and wrong else None
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 20 == 0 or i == 1:
                print(f"[{i}/{len(remaining)}] {qid} {answer!r} -> {parsed}", flush=True)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
