#!/usr/bin/env python3
"""Build RelSim candidate images for Phase 3A, excluding train/held-out and bench hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMAGE_ROOT, ROOT, SEED, dump_json, dump_jsonl

TRAIN_JSONS = [
    ROOT / "data/relsim_llava_1k.json",
    ROOT / "data/phase1/qwen_explicit_1k.json",
    ROOT / "data/phase1/qwen_typed_1k.json",
    ROOT / "data/phase2/v1_positive.json",
    ROOT / "data/phase2b/v2b_balanced.json",
]
HELDOUT_JSONS = [
    ROOT / "eval_results/qwen/phase15/heldout_200.json",
    ROOT / "eval_results/qwen/phase16b/heldout_matched.jsonl",
    ROOT / "data/phase2/heldout_verification.jsonl",
    ROOT / "eval_results/qwen/phase2/heldout_verification.jsonl",
]
BENCH_IMAGE_DIRS = [
    ROOT / "R-Bench/images",
    ROOT / "AMBER/images",
    ROOT / "MMRel",
]


def image_name(rec: dict) -> str:
    name = rec.get("image") or rec.get("image_path") or rec.get("img") or ""
    return Path(str(name)).name


def load_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    if path.suffix == ".jsonl":
        data = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data") or data.get("questions") or list(data.values())
    names = set()
    for rec in data:
        if isinstance(rec, dict):
            n = image_name(rec)
            if n:
                names.add(n)
    return names


def iter_images(root: Path):
    if not root.exists():
        return
    if root.is_file():
        return
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            yield p


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-root", default=str(IMAGE_ROOT))
    p.add_argument("--out-dir", default=str(ROOT / "data/phase3a"))
    p.add_argument("--n-queue", type=int, default=12000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip-bench-hash", action="store_true")
    args = p.parse_args()

    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ex, ho_ex = set(), set()
    for path in TRAIN_JSONS:
        train_ex |= load_names(path)
    for path in HELDOUT_JSONS:
        ho_ex |= load_names(path)
    used = train_ex | ho_ex

    pool = sorted(p.name for p in image_root.glob("*.jpg"))
    unused = [n for n in pool if n not in used]
    rng = random.Random(args.seed)
    rng.shuffle(unused)

    bench_stems: set[str] = set()
    bench_md5: set[str] = set()
    n_bench_files = 0
    if not args.skip_bench_hash:
        for d in BENCH_IMAGE_DIRS:
            for path in iter_images(d):
                n_bench_files += 1
                bench_stems.add(path.stem.lower())
                bench_stems.add(path.name.lower())
                try:
                    bench_md5.add(file_md5(path))
                except OSError:
                    continue

    kept = []
    n_hash_skip = 0
    for name in unused:
        stem = Path(name).stem.lower()
        if stem in bench_stems or name.lower() in bench_stems:
            n_hash_skip += 1
            continue
        path = image_root / name
        if bench_md5:
            try:
                digest = file_md5(path)
            except OSError:
                continue
            if digest in bench_md5:
                n_hash_skip += 1
                continue
        else:
            digest = None
        kept.append(
            {
                "id": f"relsim_{Path(name).stem}",
                "image": name,
                "image_md5": digest,
                "rank": len(kept),
            }
        )
        if len(kept) >= args.n_queue:
            break

    dump_jsonl(out_dir / "candidates.jsonl", kept)
    meta = {
        "image_root": str(image_root),
        "pool_n": len(pool),
        "exclude_train": len(train_ex),
        "exclude_heldout": len(ho_ex),
        "exclude_union": len(used),
        "unused_n": len(unused),
        "bench_files_hashed": n_bench_files,
        "bench_hash_skipped": n_hash_skip,
        "queue_n": len(kept),
        "seed": args.seed,
        "n_queue": args.n_queue,
    }
    dump_json(out_dir / "candidate_meta.json", meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
