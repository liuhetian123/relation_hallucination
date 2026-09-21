#!/usr/bin/env python3
"""Build Phase 4C multi-format train JSON from the frozen Phase 4A clean set."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import (  # noqa: E402
    f2_prompt,
    f3_prompt,
    heldout_to_is_question,
    human_prompt_text,
    parse_renderer_statement,
    qc_is_question,
)

FORMATS = ("F1", "F2", "F3")
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=str(ROOT / "data/phase4a/train_clean.json"))
    p.add_argument("--heldout", default=str(ROOT / "eval_results/qwen/phase2/heldout_verification.jsonl"))
    p.add_argument("--out", default=str(ROOT / "data/phase4c/train_mf.json"))
    p.add_argument("--out-heldout-f2", default=str(ROOT / "data/phase4c/heldout_f2.jsonl"))
    p.add_argument("--out-stats", default=str(ROOT / "data/phase4c/mf_build_stats.json"))
    p.add_argument("--qc-sample", default=str(ROOT / "eval_results/qwen/phase4c/mf_render_sample.txt"))
    return p.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_positive(rec: dict) -> bool:
    return str(rec.get("label", "")).strip().lower() in {"yes", "true", "1", "y"}


def format_index(sample_id: str) -> int:
    digest = hashlib.md5(str(sample_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % 3


def pos_rates(samples: list[dict], assigned: list[str]) -> dict[str, dict]:
    out = {}
    for fmt in FORMATS:
        rows = [s for s, a in zip(samples, assigned) if a == fmt]
        n = len(rows)
        n_pos = sum(1 for r in rows if is_positive(r))
        n_neg = n - n_pos
        pos_rate = n_pos / n if n else 0.0
        out[fmt] = {
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_rate": pos_rate,
            "neg_rate": n_neg / n if n else 0.0,
            "pos_pp_from_50": abs(pos_rate - 0.5) * 100,
        }
    return out


def needs_stratified(rates: dict[str, dict], max_pp: float = 3.0) -> bool:
    if any(v["n"] == 0 for v in rates.values()):
        return True
    return any(v["pos_pp_from_50"] > max_pp for v in rates.values())


def hash_assign(samples: list[dict]) -> list[str]:
    return [FORMATS[format_index(rec["id"])] for rec in samples]


def stratified_assign(samples: list[dict]) -> list[str]:
    """Round-robin inside positive and negative groups, sorted by id (seed-free)."""
    assigned = [""] * len(samples)
    for flag in (True, False):
        idxs = [i for i, rec in enumerate(samples) if is_positive(rec) is flag]
        idxs.sort(key=lambda i: str(samples[i]["id"]))
        for k, i in enumerate(idxs):
            assigned[i] = FORMATS[k % 3]
    return assigned


def apply_format(rec: dict, fmt: str) -> dict:
    out = copy.deepcopy(rec)
    out["prompt_format"] = fmt
    if fmt == "F1":
        return out
    if fmt == "F2":
        text, mode = f2_prompt(rec["statement"], allow_fallback=False)
    elif fmt == "F3":
        text, mode = f3_prompt(rec["statement"], allow_fallback=False)
    else:
        raise ValueError(fmt)
    out["prompt_parse_mode"] = mode
    out["conversations"] = [
        {"from": "human", "value": f"<image>\n{text}"},
        {"from": "gpt", "value": rec["label"]},
    ]
    return out


def qc_row(rec: dict) -> list[str]:
    fmt = rec["prompt_format"]
    if fmt == "F1":
        text = human_prompt_text(rec)
        if not text.startswith("Does the image support the relation"):
            return ["f1_not_verification_template"]
        return []
    text = human_prompt_text(rec)
    first = text.split("\n", 1)[0].strip()
    return qc_is_question(first, rec)


def write_qc_sample(rows: list[dict], path: Path, seed: int = SEED) -> list[str]:
    rng = random.Random(seed)
    f2 = [r for r in rows if r["prompt_format"] == "F2"]
    f3 = [r for r in rows if r["prompt_format"] == "F3"]
    if len(f2) < 15 or len(f3) < 15:
        raise SystemExit(f"not enough F2/F3 samples: F2={len(f2)} F3={len(f3)}")
    picked = rng.sample(f2, 15) + rng.sample(f3, 15)
    lines = [
        "Phase 4C multi-format render QC sample",
        f"seed={seed}; 15 F2 + 15 F3 from train_mf.json",
        "",
    ]
    ids = []
    for rec in picked:
        ids.append(rec["id"])
        human = human_prompt_text(rec)
        lines += [
            f"===== {rec['prompt_format']}  {rec['id']}  label={rec['label']}  subset={rec.get('subset')} =====",
            f"statement: {rec['statement']}",
            f"subject={rec.get('subject')}  relation={rec.get('neg_relation') or rec.get('gt_relation')}  object={rec.get('object')}",
            "prompt:",
            human,
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ids


def build_heldout_f2(src: list[dict]) -> tuple[list[dict], dict]:
    out = []
    modes = Counter()
    n_fail = 0
    for rec in src:
        q, mode = heldout_to_is_question(rec["statement"])
        modes[mode] += 1
        text = f"{q}\nAnswer Yes or No."
        row = dict(rec)
        row["text"] = text
        row["prompt_format"] = "F2"
        row["prompt_parse_mode"] = mode
        if mode != "canonical":
            n_fail += 1
        out.append(row)
    stats = {
        "n": len(out),
        "parse_mode": dict(modes),
        "n_noncanonical": n_fail,
        "note": (
            "Held-out captions are original RelSim text, not renderer A/An {s} is {rel} a/an {o}. "
            "F2 wrapping never drops an internal copula; A/An captions use fallback_article, others fallback_prefix."
        ),
    }
    return out, stats


def main() -> None:
    args = parse_args()
    src = load_json(Path(args.src))
    if len(src) != 4312:
        raise SystemExit(f"expected 4312 clean samples, got {len(src)}")

    n_parse_fail = 0
    parse_fail_ids = []
    for rec in src:
        if parse_renderer_statement(rec["statement"]) is None:
            n_parse_fail += 1
            parse_fail_ids.append(rec["id"])
    if n_parse_fail:
        raise SystemExit(f"renderer parse failed on {n_parse_fail} train statements: {parse_fail_ids[:5]}")

    hash_fmt = hash_assign(src)
    hash_rates = pos_rates(src, hash_fmt)
    used = "hash_md5_mod3"
    assigned = hash_fmt
    if needs_stratified(hash_rates):
        assigned = stratified_assign(src)
        used = "stratified_round_robin"
    rates = pos_rates(src, assigned)

    out_rows = [apply_format(rec, fmt) for rec, fmt in zip(src, assigned)]
    qc_failures = []
    for rec in out_rows:
        reasons = qc_row(rec)
        if reasons:
            qc_failures.append({"id": rec["id"], "format": rec["prompt_format"], "reasons": reasons})
    if qc_failures:
        raise SystemExit(f"QC failed on {len(qc_failures)} samples: {qc_failures[:5]}")

    # F1 conversations must be byte-identical to clean.
    src_by_id = {r["id"]: r for r in src}
    for rec in out_rows:
        if rec["prompt_format"] != "F1":
            continue
        orig = src_by_id[rec["id"]]
        if rec["conversations"] != orig["conversations"]:
            raise SystemExit(f"F1 conversation mutated: {rec['id']}")
        if rec["statement"] != orig["statement"] or rec["image"] != orig["image"]:
            raise SystemExit(f"F1 content mutated: {rec['id']}")

    dump_json(Path(args.out), out_rows)
    sample_ids = write_qc_sample(out_rows, Path(args.qc_sample))

    heldout = load_jsonl(Path(args.heldout))
    heldout_f2, heldout_stats = build_heldout_f2(heldout)
    dump_jsonl(Path(args.out_heldout_f2), heldout_f2)

    fmt_labels = defaultdict(Counter)
    for rec in out_rows:
        fmt_labels[rec["prompt_format"]][rec["label"]] += 1

    stats = {
        "n_src": len(src),
        "n_out": len(out_rows),
        "n_pairs": len(out_rows) // 2,
        "same_ids_order": [r["id"] for r in out_rows] == [r["id"] for r in src],
        "assignment": used,
        "hash_rates": hash_rates,
        "final_rates": rates,
        "format_counts": dict(Counter(r["prompt_format"] for r in out_rows)),
        "format_labels": {k: dict(v) for k, v in fmt_labels.items()},
        "n_qc_fail": 0,
        "qc_sample_ids": sample_ids,
        "heldout_f2": heldout_stats,
    }
    dump_json(Path(args.out_stats), stats)
    print(json.dumps(stats, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {args.out_heldout_f2}", flush=True)
    print(f"wrote {args.qc_sample}", flush=True)


if __name__ == "__main__":
    main()
