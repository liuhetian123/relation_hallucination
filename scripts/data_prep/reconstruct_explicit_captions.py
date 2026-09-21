#!/usr/bin/env python3
"""Reconstruct explicit captions from RelSim anonymous captions.

InternVL only maps placeholders to visible entities. The explicit caption is
always produced by deterministic string replacement in Python.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import traceback
from collections import Counter
from pathlib import Path


PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
TEACHER_NAME = "InternVL3.5-8B"

PROMPT_TEMPLATE = """You are given an image and an anonymous relational caption.

Your ONLY task is to identify the concrete visible entity corresponding to each placeholder in the caption.

Anonymous caption:
"{anonymous_caption}"

Placeholders:
{placeholder_list}

Rules:
1. Do NOT rewrite the caption.
2. Do NOT change, infer, paraphrase, or complete the relation.
3. Do NOT add any new attributes or descriptions.
4. Only identify the concrete visible entity corresponding to each placeholder.
5. Use a simple, common, basic-level English category name.
6. Prefer "bird" instead of "sparrow", "dog" instead of "golden retriever", "car" instead of a specific car model.
7. The replacement must match the semantic type indicated by the placeholder.
8. Use the image as the primary evidence.
9. If a placeholder cannot be identified confidently from the image, return "UNCERTAIN" for that placeholder.
10. Do not guess an entity only because it is statistically likely to participate in the described relation.
11. Return JSON only. Do not provide explanations.

Expected output format:
{{
  "replacements": {{
    "{example_a}": "bird",
    "{example_b}": "finger"
  }},
  "status": "ok"
}}

If any placeholder is uncertain:
{{
  "replacements": {{
    "{example_a}": "UNCERTAIN",
    "{example_b}": "finger"
  }},
  "status": "uncertain"
}}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="RelSim JSON list, e.g. data/relsim_llava_1k.json")
    parser.add_argument("--image_root", required=True, help="Directory containing RelSim images")
    parser.add_argument("--model_path", required=True, help="InternVL3.5-8B-HF path or HF repo id")
    parser.add_argument("--output", required=True, help="Incremental JSONL output path")
    parser.add_argument("--review_file", default=None, help="TSV for human QC. Default: <output>.review.tsv")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N input samples")
    parser.add_argument("--gpu", default=None, help="Physical GPU id; sets CUDA_VISIBLE_DEVICES")
    parser.add_argument("--max_patches", type=int, default=4, help="InternVL image tiles. Use 1 to disable cropping.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Generation cap. 64 is often too short for JSON.")
    parser.add_argument("--retry_failed", action="store_true", help="Retry ids whose previous status is not ok/no_placeholder")
    return parser


def extract_placeholders(caption: str) -> list[str]:
    seen = []
    for match in PLACEHOLDER_RE.findall(caption):
        if match not in seen:
            seen.append(match)
    return seen


def get_anonymous_caption(sample: dict) -> str:
    if "conversations" in sample:
        for turn in reversed(sample["conversations"]):
            if turn.get("from") in {"gpt", "assistant"}:
                return turn.get("value", "")
    return sample.get("caption") or sample.get("anonymous_caption") or ""


def build_prompt(anonymous_caption: str, placeholders: list[str]) -> str:
    example_a = placeholders[0] if placeholders else "{Animal}"
    example_b = placeholders[1] if len(placeholders) > 1 else "{Body Part}"
    placeholder_list = "\n".join(f"- {p}" for p in placeholders)
    return PROMPT_TEMPLATE.format(
        anonymous_caption=anonymous_caption,
        placeholder_list=placeholder_list,
        example_a=example_a,
        example_b=example_b,
    )


def extract_json_object(text: str) -> tuple[dict | None, bool]:
    """Return (parsed, repaired). repaired=True if we stripped fences or surrounding text."""
    raw = text.strip()
    repaired = False
    fence = JSON_FENCE_RE.search(raw)
    if fence:
        raw = fence.group(1).strip()
        repaired = True
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed, repaired
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        repaired = True
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return parsed, repaired
        except json.JSONDecodeError:
            return None, True
    return None, repaired


def apply_replacements(caption: str, mapping: dict[str, str]) -> str:
    result = caption
    for placeholder in sorted(mapping, key=len, reverse=True):
        result = result.replace(placeholder, mapping[placeholder])
    return result


def literal_segments(caption: str, placeholders: list[str]) -> list[str]:
    if not placeholders:
        return [caption]
    pattern = "(" + "|".join(re.escape(p) for p in sorted(placeholders, key=len, reverse=True)) + ")"
    parts = re.split(pattern, caption)
    return [p for p in parts if p not in placeholders]


def is_fine_grained(value: str) -> bool:
    tokens = value.strip().split()
    if len(tokens) >= 3:
        return True
    if any(ch.isdigit() for ch in value):
        return True
    lowered = value.lower()
    markers = ("model", "tesla", "sparrow", "retriever", "robin", "golden", "siamese")
    return any(m in lowered for m in markers)


def validate_mapping(placeholders: list[str], mapping: dict, teacher_status: str) -> tuple[str, dict[str, str]]:
    cleaned: dict[str, str] = {}
    for ph in placeholders:
        if ph not in mapping:
            return "missing_placeholder", {k: str(v) for k, v in mapping.items()}
        value = mapping[ph]
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if not value:
            return "invalid_mapping", cleaned
        if value.startswith("{") and value.endswith("}"):
            return "invalid_mapping", cleaned
        cleaned[ph] = value

    if any(v.upper() == "UNCERTAIN" for v in cleaned.values()) or teacher_status == "uncertain":
        return "uncertain", cleaned
    return "ok", cleaned


def quality_ok(anonymous: str, explicit: str, placeholders: list[str], mapping: dict[str, str]) -> bool:
    if apply_replacements(anonymous, mapping) != explicit:
        return False
    if any(ph in explicit for ph in placeholders):
        return False
    if literal_segments(anonymous, placeholders) != literal_segments(explicit, list(mapping.values())):
        # Values can collide with literals; the deterministic apply_replacements check above is the source of truth.
        pass
    leftover = [m for m in PLACEHOLDER_RE.findall(explicit) if m in placeholders]
    return not leftover


def needs_review_flag(status: str, mapping: dict[str, str], json_repaired: bool) -> bool:
    if status != "ok":
        return True
    if json_repaired:
        return True
    return any(is_fine_grained(v) for v in mapping.values())


def load_existing(path: Path, retry_failed: bool) -> dict[str, dict]:
    existing = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            existing[rec["id"]] = rec
    if not retry_failed:
        return existing
    keep = {}
    for sid, rec in existing.items():
        if rec.get("reconstruction_status") in {"ok", "no_placeholder"}:
            keep[sid] = rec
    return keep


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_review_tsv(jsonl_path: Path, review_path: Path, image_root: Path) -> None:
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("r", encoding="utf-8") as src, review_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst, delimiter="\t")
        writer.writerow(
            [
                "id",
                "image_path",
                "anonymous_caption",
                "entity_mapping",
                "explicit_caption",
                "status",
                "needs_review",
            ]
        )
        for line in src:
            rec = json.loads(line)
            image_path = str(image_root / rec.get("image", ""))
            writer.writerow(
                [
                    rec.get("id", ""),
                    image_path,
                    rec.get("anonymous_caption", "").replace("\t", " ").replace("\n", " "),
                    json.dumps(rec.get("entity_mapping", {}), ensure_ascii=False),
                    (rec.get("explicit_caption") or "").replace("\t", " ").replace("\n", " "),
                    rec.get("reconstruction_status", ""),
                    str(rec.get("needs_review", True)).lower(),
                ]
            )


def print_summary(records: list[dict], placeholders_all: Counter, uncertain_ph: Counter, entity_counter: Counter) -> None:
    total = len(records)
    status_counts = Counter(r.get("reconstruction_status", "unknown") for r in records)
    n_ok = status_counts.get("ok", 0)
    n_no = status_counts.get("no_placeholder", 0)
    n_uncertain = status_counts.get("uncertain", 0)
    n_failed = total - n_ok - n_no - n_uncertain
    with_ph = total - n_no
    success_denom = with_ph if with_ph else 1

    print("\n========== Reconstruction summary ==========")
    print(f"Total samples: {total}")
    print(f"With placeholders: {with_ph}")
    print(f"No placeholders: {n_no}")
    print()
    print(f"OK: {n_ok}")
    print(f"Uncertain: {n_uncertain}")
    print(f"Parse error: {status_counts.get('json_parse_error', 0)}")
    print(f"Missing placeholder: {status_counts.get('missing_placeholder', 0)}")
    print(f"Invalid mapping: {status_counts.get('invalid_mapping', 0)}")
    print(f"Image error: {status_counts.get('image_error', 0)}")
    print(f"Inference error: {status_counts.get('inference_error', 0)}")
    print()
    print(f"Success rate: {100.0 * n_ok / success_denom:.1f}%  (ok / with_placeholders)")
    if placeholders_all:
        ph_per_sample = sum(placeholders_all.values()) / max(with_ph, 1)
        print(f"Avg placeholders per caption: {ph_per_sample:.2f}")
        print("\nPlaceholder frequency:")
        for k, v in placeholders_all.most_common(20):
            print(f"  {k}: {v}")
    if entity_counter:
        print("\nMost common replacements:")
        for k, v in entity_counter.most_common(15):
            print(f"  {k}: {v}")
    if uncertain_ph:
        print("\nUNCERTAIN by placeholder:")
        for k, v in uncertain_ph.most_common(15):
            print(f"  {k}: {v}")
    print(f"Failed (non-ok/uncertain/no_placeholder): {n_failed}")
    print("============================================")


def make_record(sample: dict, **extra) -> dict:
    rec = dict(sample)
    rec.update(extra)
    rec["teacher_model"] = TEACHER_NAME
    return rec


def run(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    input_path = Path(args.input)
    image_root = Path(args.image_root)
    output_path = Path(args.output)
    review_path = (
        Path(args.review_file)
        if args.review_file
        else output_path.with_name(output_path.stem + ".review.tsv")
    )

    with input_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    if args.limit is not None:
        samples = samples[: args.limit]

    existing = load_existing(output_path, retry_failed=args.retry_failed)
    if args.retry_failed and output_path.exists():
        kept = [rec for rec in existing.values()]
        with output_path.open("w", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Loaded {len(samples)} input samples; {len(existing)} already in output (will skip).")
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"GPU {torch.cuda.get_device_name(0)} free_gb={free_gb:.2f}")
    print(
        f"dtype=bfloat16 max_patches={args.max_patches} "
        f"max_new_tokens={args.max_new_tokens} do_sample=False"
    )

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    device = next(model.parameters()).device

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
        }
        if args.max_patches <= 1:
            proc_kwargs["crop_to_patches"] = False
        else:
            proc_kwargs["crop_to_patches"] = True
            proc_kwargs["max_patches"] = args.max_patches
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

    processed_this_run = 0
    try:
        for sample in samples:
            sid = sample.get("id") or sample.get("image")
            if sid in existing:
                continue

            anonymous = get_anonymous_caption(sample)
            placeholders = extract_placeholders(anonymous)
            image_name = sample.get("image", "")
            image_path = image_root / image_name

            base_extra = {
                "anonymous_caption": anonymous,
                "entity_mapping": {},
                "explicit_caption": None,
                "teacher_raw_output": "",
                "json_repaired": False,
                "needs_review": True,
            }

            def persist(status: str, **overrides) -> dict:
                rec = make_record(sample, **{**base_extra, "reconstruction_status": status, **overrides})
                append_jsonl(output_path, rec)
                existing[sid] = rec
                return rec

            if not placeholders:
                processed_this_run += 1
                persist(
                    "no_placeholder",
                    explicit_caption=anonymous,
                    needs_review=False,
                )
                print(f"[{processed_this_run}] {sid} no_placeholder", flush=True)
                continue

            if not image_path.exists():
                processed_this_run += 1
                persist("image_error", teacher_raw_output=f"missing image: {image_path}")
                print(f"[{processed_this_run}] {sid} image_error", flush=True)
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                processed_this_run += 1
                persist("image_error", teacher_raw_output=repr(exc))
                print(f"[{processed_this_run}] {sid} image_error {exc}", flush=True)
                continue

            prompt = build_prompt(anonymous, placeholders)
            try:
                raw = infer(image, prompt)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                processed_this_run += 1
                persist(
                    "inference_error",
                    teacher_raw_output="CUDA OOM. Reduce --max_patches first (try 1).",
                )
                print(f"[{processed_this_run}] {sid} inference_error OOM", flush=True)
                continue
            except Exception:
                processed_this_run += 1
                persist("inference_error", teacher_raw_output=traceback.format_exc())
                print(f"[{processed_this_run}] {sid} inference_error", flush=True)
                continue

            parsed, repaired = extract_json_object(raw)
            if parsed is None:
                processed_this_run += 1
                persist(
                    "json_parse_error",
                    teacher_raw_output=raw,
                    json_repaired=repaired,
                )
                print(f"[{processed_this_run}] {sid} json_parse_error", flush=True)
                continue

            mapping_in = parsed.get("replacements", parsed)
            if not isinstance(mapping_in, dict):
                processed_this_run += 1
                persist(
                    "json_parse_error",
                    teacher_raw_output=raw,
                    json_repaired=repaired,
                )
                print(f"[{processed_this_run}] {sid} json_parse_error (no replacements)", flush=True)
                continue

            teacher_status = str(parsed.get("status", "ok")).lower()
            status, mapping = validate_mapping(placeholders, mapping_in, teacher_status)
            explicit = None
            if status == "ok":
                explicit = apply_replacements(anonymous, mapping)
                if not quality_ok(anonymous, explicit, placeholders, mapping):
                    status = "invalid_mapping"
                    explicit = None

            conversations_explicit = None
            if status == "ok" and "conversations" in sample:
                conversations_explicit = []
                for turn in sample["conversations"]:
                    if turn.get("from") in {"gpt", "assistant"}:
                        conversations_explicit.append({**turn, "value": explicit})
                    else:
                        conversations_explicit.append(dict(turn))

            extra_ok = {}
            if conversations_explicit is not None:
                extra_ok["conversations_explicit"] = conversations_explicit
            persist(
                status,
                entity_mapping=mapping,
                explicit_caption=explicit,
                teacher_raw_output=raw,
                json_repaired=repaired,
                needs_review=needs_review_flag(status, mapping, repaired),
                **extra_ok,
            )
            processed_this_run += 1
            print(f"[{processed_this_run}] {sid} {status} mapping={mapping}", flush=True)
    finally:
        if output_path.exists():
            all_records = []
            placeholders_all = Counter()
            uncertain_ph = Counter()
            entity_counter = Counter()
            with output_path.open("r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    all_records.append(rec)
                    anon = rec.get("anonymous_caption") or ""
                    phs = extract_placeholders(anon)
                    placeholders_all.update(phs)
                    mapping = rec.get("entity_mapping") or {}
                    if rec.get("reconstruction_status") == "ok":
                        entity_counter.update(v.lower() for v in mapping.values())
                    if rec.get("reconstruction_status") == "uncertain":
                        for k, v in mapping.items():
                            if str(v).upper() == "UNCERTAIN":
                                uncertain_ph[k] += 1
            write_review_tsv(output_path, review_path, image_root)
            print_summary(all_records, placeholders_all, uncertain_ph, entity_counter)
            print(f"JSONL: {output_path}")
            print(f"Review TSV: {review_path}")
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Peak CUDA allocated: {peak:.2f} GB")


def main() -> None:
    args = build_parser().parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Partial JSONL is already saved; rerun the same command to resume.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
