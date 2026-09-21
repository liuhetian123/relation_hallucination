#!/usr/bin/env python3
"""Teacher-forced mean token NLL for Phase 1.6 (full target + relation span)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/phase15"))
from relation_extract import strip_caption  # noqa: E402

TRAIN_PROMPT = "Describe the primary relation shown in this image in one short sentence."
HELDOUT_PROMPT = (
    "What is the primary relation shown in this image?\n"
    "Answer only with the relation phrase."
)
MAX_PIXELS = 256 * 28 * 28
MIN_PIXELS = 16 * 28 * 28


def load_json(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                done.add((rec["id"], rec["split"]))
    return done


def find_subseq(hay: list[int], needle: list[int]) -> int | None:
    if not needle:
        return None
    n, m = len(hay), len(needle)
    for i in range(n - m + 1):
        if hay[i : i + m] == needle:
            return i
    return None


def relation_positions(tokenizer, caption: str, relation: str, target_ids: list[int]) -> list[int]:
    """Return indices into target_ids that overlap the relation phrase."""
    if not relation:
        return []
    caption = strip_caption(caption)
    start = caption.lower().find(relation.lower())
    if start < 0:
        return []
    end = start + len(relation)
    encoded = tokenizer(caption, add_special_tokens=False, return_offsets_mapping=True)
    cap_ids = list(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    rel_local = [
        i
        for i, (s, e) in enumerate(offsets)
        if e > start and s < end and e > s
    ]
    if not rel_local:
        return []
    pos = find_subseq(target_ids, cap_ids)
    if pos is None:
        encoded_sp = tokenizer(" " + caption, add_special_tokens=False, return_offsets_mapping=True)
        cap_ids = list(encoded_sp["input_ids"])
        offsets = encoded_sp["offset_mapping"]
        # offsets include the extra leading space as char 0
        rel_local = [
            i
            for i, (s, e) in enumerate(offsets)
            if e > start + 1 and s < end + 1 and e > s
        ]
        pos = find_subseq(target_ids, cap_ids)
    if pos is None:
        return []
    return [pos + i for i in rel_local]


def build_inputs(processor, image: Image.Image, user_text: str, target: str, max_pixels: int, min_pixels: int):
    user_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": user_text},
    ]
    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": target},
    ]
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    vision = process_vision_info(full_messages)
    image_inputs = vision[0] if vision else [image]
    video_inputs = vision[1] if vision and len(vision) > 1 else None
    size = getattr(processor.image_processor, "size", None)
    if size is not None:
        size["longest_edge"] = max_pixels
        size["shortest_edge"] = min_pixels
    full = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    )
    prompt = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    )
    return full, int(prompt["attention_mask"].sum().item())


def score_one(model, processor, image, user_text, target, relation, device):
    full, prompt_len = build_inputs(
        processor, image, user_text, target, MAX_PIXELS, MIN_PIXELS
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in full.items()}
    input_ids = inputs["input_ids"]
    with torch.inference_mode():
        logits = model(**inputs).logits
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = input_ids[:, 1:]
    token_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
    nll = -token_logp

    seq = input_ids[0].tolist()
    target_ids = seq[prompt_len:]
    # Shifted NLL index i corresponds to predicting token i+1, i.e. seq[i+1]
    # Token at sequence position t (t>=prompt_len) is predicted at nll index t-1.
    target_nll_idx = list(range(prompt_len - 1, len(seq) - 1))
    if len(target_nll_idx) != len(target_ids):
        # last token may be truncated from shift
        n = min(len(target_nll_idx), len(target_ids))
        target_nll_idx = target_nll_idx[:n]
        target_ids = target_ids[:n]

    if not target_nll_idx:
        return None
    full_vals = nll[target_nll_idx]
    out = {
        "n_tokens": int(full_vals.numel()),
        "full_total_nll": float(full_vals.sum().item()),
        "full_mean_nll": float(full_vals.mean().item()),
        "rel_n_tokens": 0,
        "rel_total_nll": None,
        "rel_mean_nll": None,
        "rel_located_tokens": False,
    }
    rel_pos = relation_positions(processor.tokenizer, target, relation or "", target_ids)
    rel_idx = []
    for p in rel_pos:
        # p is index into target_ids; nll index = prompt_len - 1 + p
        ni = prompt_len - 1 + p
        if 0 <= ni < nll.numel():
            rel_idx.append(ni)
    if rel_idx:
        rel_vals = nll[rel_idx]
        out["rel_n_tokens"] = int(rel_vals.numel())
        out["rel_total_nll"] = float(rel_vals.sum().item())
        out["rel_mean_nll"] = float(rel_vals.mean().item())
        out["rel_located_tokens"] = True
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--run_name", required=True, help="base | explicit_1k | typed_1k")
    p.add_argument("--gpu", default=None)
    p.add_argument("--out_dir", default=str(ROOT / "eval_results/qwen/phase16"))
    p.add_argument("--explicit_json", default=str(ROOT / "data/phase1/qwen_explicit_1k.json"))
    p.add_argument("--typed_json", default=str(ROOT / "data/phase1/qwen_typed_1k.json"))
    p.add_argument("--spans_jsonl", default=str(ROOT / "eval_results/qwen/phase16/train_relation_spans.jsonl"))
    p.add_argument("--heldout_jsonl", default=str(ROOT / "eval_results/qwen/phase15/heldout_200.jsonl"))
    p.add_argument("--train_image_root", default=str(ROOT / "data/relsim_images"))
    p.add_argument("--heldout_image_root", default="/data/lht/relsim_dataset/relsim_images")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    spans = {r["id"]: r for r in load_json(Path(args.spans_jsonl))}
    explicit = load_json(Path(args.explicit_json))
    typed = load_json(Path(args.typed_json))
    heldout = load_json(Path(args.heldout_jsonl))
    if args.limit:
        explicit = explicit[: args.limit]
        typed = typed[: args.limit]
        heldout = heldout[: args.limit]

    jobs = []
    train_root = Path(args.train_image_root)
    for sample in explicit:
        span = spans[sample["id"]]
        cap = strip_caption(sample["conversations"][-1]["value"])
        jobs.append(
            {
                "id": sample["id"],
                "split": "train_explicit",
                "image": train_root / sample["image"],
                "prompt": TRAIN_PROMPT,
                "target": cap,
                "relation": span["relation"] if span.get("located") else None,
            }
        )
    for sample in typed:
        span = spans[sample["id"]]
        cap = strip_caption(sample["conversations"][-1]["value"])
        jobs.append(
            {
                "id": sample["id"],
                "split": "train_typed",
                "image": train_root / sample["image"],
                "prompt": TRAIN_PROMPT,
                "target": cap,
                "relation": span["relation"] if span.get("located") else None,
            }
        )
    ho_root = Path(args.heldout_image_root)
    for sample in heldout:
        jobs.append(
            {
                "id": sample["id"],
                "split": "heldout_rel",
                "image": ho_root / Path(sample["image"]).name,
                "prompt": HELDOUT_PROMPT,
                "target": sample["gt_relation"],
                "relation": sample["gt_relation"],
            }
        )

    out_dir = Path(args.out_dir) / "scores"
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
                rec = {
                    "id": job["id"],
                    "split": job["split"],
                    "run": args.run_name,
                    "error": repr(exc),
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                continue
            stats = score_one(
                model,
                processor,
                image,
                job["prompt"],
                job["target"],
                job["relation"],
                device,
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
            if i % 50 == 0 or i == 1:
                rel = rec.get("rel_mean_nll")
                print(
                    f"[{i}/{len(remaining)}] {job['split']} {job['id']} "
                    f"full={rec.get('full_mean_nll')} rel={rel}",
                    flush=True,
                )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
