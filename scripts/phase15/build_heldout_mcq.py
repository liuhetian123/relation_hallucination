#!/usr/bin/env python3
"""Build a fixed RelSim held-out 200-way relation MCQ for Phase 1.5."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from relation_extract import (  # noqa: E402
    extract_relation,
    normalize_relation,
    relations_too_similar,
    strip_caption,
)

QUESTION = "Which relation best describes the primary relation shown in the image?"
LETTERS = ("A", "B", "C", "D")


def load_llava_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data") or data.get("samples") or list(data.values())
    return data


def caption_of(sample: dict) -> str:
    conv = sample.get("conversations") or []
    if conv:
        return strip_caption(conv[-1].get("value", ""))
    return strip_caption(sample.get("caption") or sample.get("anonymous_caption") or "")


def image_of(sample: dict) -> str:
    name = sample.get("image") or sample.get("image_path") or ""
    return Path(str(name)).name


def build_caption_index(samples: list[dict]) -> dict[str, dict]:
    index = {}
    for sample in samples:
        image = image_of(sample)
        if not image:
            continue
        cap = caption_of(sample)
        if image not in index:
            index[image] = {
                "id": sample.get("id") or f"relsim_{Path(image).stem}",
                "image": image,
                "anonymous_caption": cap,
            }
    return index


def format_question(options: dict[str, str]) -> str:
    lines = [QUESTION, ""]
    for letter in LETTERS:
        lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append("Answer with A, B, C, or D only.")
    return "\n".join(lines)


def sample_wrongs(gt: str, vocab: list[str], rng: random.Random, n: int = 3) -> list[str]:
    pool = [r for r in vocab if not relations_too_similar(r, gt)]
    rng.shuffle(pool)
    chosen: list[str] = []
    for cand in pool:
        if any(relations_too_similar(cand, prev) for prev in chosen):
            continue
        chosen.append(cand)
        if len(chosen) == n:
            return chosen
    raise RuntimeError(f"Could not sample {n} distinct wrong relations for '{gt}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_json", default=str(ROOT / "data/relsim_llava_1k.json"))
    parser.add_argument(
        "--caption_json",
        default="/data/lht/relsim_dataset/relsim_llava_100k.json",
    )
    parser.add_argument("--image_dir", default="/data/lht/relsim_dataset/relsim_images")
    parser.add_argument("--out_dir", default=str(ROOT / "eval_results/qwen/phase15"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_samples = load_llava_json(Path(args.train_json))
    train_images = {image_of(s) for s in train_samples if image_of(s)}
    caption_index = build_caption_index(load_llava_json(Path(args.caption_json)))
    image_dir = Path(args.image_dir)

    freq: Counter[str] = Counter()
    for rec in caption_index.values():
        rel, _src = extract_relation(rec["anonymous_caption"], between_only=False)
        if rel:
            freq[rel] += 1

    vocab_all = sorted(freq)
    vocab_distractor = [r for r, c in freq.items() if c >= 2]
    if len(vocab_distractor) < 50:
        vocab_distractor = list(vocab_all)

    pool = []
    missing_caption = 0
    missing_image = 0
    no_relation = 0
    for image, rec in caption_index.items():
        if image in train_images:
            continue
        img_path = image_dir / image
        if not img_path.is_file():
            missing_image += 1
            continue
        if not rec["anonymous_caption"]:
            missing_caption += 1
            continue
        rel, src = extract_relation(rec["anonymous_caption"], between_only=True)
        if not rel or src != "between":
            no_relation += 1
            continue
        pool.append({**rec, "gt_relation": rel})

    if len(pool) < args.n:
        raise SystemExit(f"Held-out pool too small: {len(pool)} < {args.n}")

    rng = random.Random(args.seed)
    pool.sort(key=lambda x: x["image"])
    selected = rng.sample(pool, args.n)
    selected.sort(key=lambda x: x["image"])

    positions = [letter for letter in LETTERS for _ in range(args.n // 4)]
    if len(positions) != args.n:
        raise SystemExit("n must be divisible by 4 for balanced A/B/C/D")
    rng.shuffle(positions)

    items = []
    for rec, answer in zip(selected, positions):
        wrongs = sample_wrongs(rec["gt_relation"], vocab_distractor, rng, n=3)
        options = {}
        wi = 0
        for letter in LETTERS:
            if letter == answer:
                options[letter] = rec["gt_relation"]
            else:
                options[letter] = wrongs[wi]
                wi += 1
        if len(set(options.values())) != 4:
            raise SystemExit(f"Duplicate options for {rec['image']}")
        if options[answer] != rec["gt_relation"]:
            raise SystemExit(f"GT missing from options for {rec['image']}")
        items.append(
            {
                "id": rec["id"],
                "image": rec["image"],
                "anonymous_caption": rec["anonymous_caption"],
                "gt_relation": rec["gt_relation"],
                "question": QUESTION,
                "text": format_question(options),
                "options": options,
                "answer": answer,
                "source": "relsim_heldout",
                "split_seed": args.seed,
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "heldout_200.json"
    jsonl_path = out_dir / "heldout_200.jsonl"
    json_path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    vocab_path = out_dir / "relation_vocab.json"
    vocab_path.write_text(
        json.dumps(
            {
                "size": len(freq),
                "size_freq_ge2": len(vocab_distractor),
                "top50": freq.most_common(50),
                "counts": dict(freq.most_common()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pos_dist = Counter(x["answer"] for x in items)
    dup_opt = sum(1 for x in items if len(set(x["options"].values())) != 4)
    gt_missing = sum(1 for x in items if x["options"][x["answer"]] != x["gt_relation"])
    overlap = sum(1 for x in items if x["image"] in train_images)
    missing_imgs = [x["image"] for x in items if not (image_dir / x["image"]).is_file()]
    missing_caps = [x["id"] for x in items if not x["anonymous_caption"]]
    unique_images = len({x["image"] for x in items})
    unique_gt = len({x["gt_relation"] for x in items})

    qc = {
        "total_questions": len(items),
        "unique_images": unique_images,
        "training_overlap_count": overlap,
        "missing_images": len(missing_imgs),
        "missing_captions": len(missing_caps),
        "unique_gt_relations": unique_gt,
        "answer_position_distribution": dict(pos_dist),
        "duplicate_options_count": dup_opt,
        "gt_missing_from_options_count": gt_missing,
        "heldout_pool_size": len(pool),
        "train_images_excluded": len(train_images),
        "caption_index_size": len(caption_index),
        "relation_vocab_size": len(freq),
        "pool_skipped_no_relation": no_relation,
        "hard_constraints_ok": (
            overlap == 0
            and len(missing_imgs) == 0
            and gt_missing == 0
            and dup_opt == 0
            and pos_dist["A"] == pos_dist["B"] == pos_dist["C"] == pos_dist["D"] == args.n // 4
        ),
        "train_json": str(Path(args.train_json).resolve()),
        "caption_json": str(Path(args.caption_json).resolve()),
        "image_dir": str(image_dir.resolve()),
        "seed": args.seed,
    }
    (out_dir / "qc_summary.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")

    review = items[:20] if args.n >= 20 else items
    # deterministic 20 for human audit, sampled from the frozen 200
    review_rng = random.Random(args.seed)
    review = review_rng.sample(items, min(20, len(items)))
    review.sort(key=lambda x: x["id"])
    with (out_dir / "review_20.jsonl").open("w", encoding="utf-8") as f:
        for item in review:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("QC summary")
    for k, v in qc.items():
        print(f"  {k}: {v}")
    print("\nReview 20 questions:")
    for i, item in enumerate(review, 1):
        print(f"\n[{i}] {item['id']}  image={item['image']}")
        print(f"  caption: {item['anonymous_caption']}")
        print(f"  gt: {item['gt_relation']}  answer={item['answer']}")
        print(item["text"])

    if not qc["hard_constraints_ok"]:
        raise SystemExit("QC hard constraints failed")
    print(f"\nWrote {jsonl_path}")


if __name__ == "__main__":
    main()
