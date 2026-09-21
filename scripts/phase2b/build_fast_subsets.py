#!/usr/bin/env python3
"""Create frozen Phase 2B fast-eval subsets. seed=42, stratified, never retuned."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval_results/qwen/phase2b/fast_subsets"
SEED = 42
RBENCH_RATIO = 0.25
MMREL_RATIO = 0.50
AMBER_RATIO = 0.30


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gold_no(label) -> bool:
    return "no" in str(label or "").strip().lower()


def stratified_sample(items: list, key_fn, ratio: float, rng: random.Random) -> list:
    buckets: dict[tuple, list] = defaultdict(list)
    for item in items:
        buckets[key_fn(item)].append(item)
    picked = []
    for key in sorted(buckets):
        group = list(buckets[key])
        group.sort(key=lambda x: str(x.get("question_id", x) if isinstance(x, dict) else x))
        n = int(round(len(group) * ratio))
        n = min(len(group), max(0, n))
        if n == 0:
            continue
        if n == len(group):
            chosen = group
        else:
            chosen = rng.sample(group, n)
        picked.extend(chosen)
    return picked


def build_rbench(rng: random.Random) -> dict:
    qdir = ROOT / "R-Bench/data_filterd"
    questions = json.loads((qdir / "image-level_filterd.json").read_text(encoding="utf-8"))
    qmap = {q["question_id"]: q for q in questions}
    llava = {q["question_id"]: q for q in load_jsonl(qdir / "image-level_filterd_llava.jsonl")}
    folds = []
    for i in range(1, 6):
        folds.append(json.loads((qdir / f"nocaps_image-level_rel_ids_{i}.json").read_text(encoding="utf-8")))
    union = sorted(set().union(*[set(f) for f in folds]))
    items = []
    for qid in union:
        q = qmap[qid]
        qtype = str(q.get("qtype") or "unknown").strip().lower()
        if qtype.startswith("positive"):
            qtype = "positive"
        elif qtype.startswith("opposite"):
            qtype = "opposite"
        elif qtype.startswith("random"):
            qtype = "random"
        items.append(
            {
                "question_id": qid,
                "label": "no" if gold_no(q.get("label")) else "yes",
                "qtype": qtype,
            }
        )
    sampled = stratified_sample(
        items,
        lambda x: (x["label"], x["qtype"]),
        RBENCH_RATIO,
        rng,
    )
    sampled_ids = {x["question_id"] for x in sampled}
    fast_folds = []
    for fold in folds:
        fast_folds.append([qid for qid in fold if qid in sampled_ids])
    subset_qs = [llava[qid] for qid in sorted(sampled_ids) if qid in llava]
    dump_json(OUT / "rbench_ids.json", sorted(sampled_ids))
    for i, fold in enumerate(fast_folds, 1):
        dump_json(OUT / f"rbench_fold_{i}.json", fold)
    dump_jsonl(OUT / "rbench_questions.jsonl", subset_qs)
    return {
        "union_unique": len(union),
        "sampled_unique": len(sampled_ids),
        "sampled_in_llava": len(subset_qs),
        "per_fold": [len(f) for f in fast_folds],
        "label": dict(Counter(x["label"] for x in sampled)),
        "qtype": dict(Counter(x["qtype"] for x in sampled)),
        "full_label": dict(Counter(x["label"] for x in items)),
        "full_qtype": dict(Counter(x["qtype"] for x in items)),
    }


def build_mmrel(rng: random.Random) -> dict:
    qs = load_jsonl(ROOT / "eval_results/qwen/mmrel/questions/mmrel_adversarial.jsonl")
    effective = []
    skipped_spec = 0
    for q in qs:
        src = q["image"].split("/")[0]
        if src == "SPEC":
            skipped_spec += 1
            continue
        source = "dalle" if src.lower().startswith("dall") else "vg"
        rec = dict(q)
        rec["_source"] = source
        rec["_label"] = "no" if gold_no(q.get("label")) else "yes"
        effective.append(rec)
    sampled = stratified_sample(
        effective,
        lambda x: (x["_label"], x["_source"]),
        MMREL_RATIO,
        rng,
    )
    out_rows = []
    for rec in sampled:
        row = {k: v for k, v in rec.items() if not k.startswith("_")}
        out_rows.append(row)
    out_rows.sort(key=lambda x: str(x["question_id"]))
    dump_jsonl(OUT / "mmrel_adv_questions.jsonl", out_rows)
    dump_json(OUT / "mmrel_adv_ids.json", [r["question_id"] for r in out_rows])
    return {
        "full": len(qs),
        "effective": len(effective),
        "skipped_spec": skipped_spec,
        "sampled": len(out_rows),
        "full_label": dict(Counter(x["_label"] for x in effective)),
        "full_source": dict(Counter(x["_source"] for x in effective)),
        "sampled_label": dict(Counter(("no" if gold_no(x["label"]) else "yes") for x in out_rows)),
        "sampled_source": dict(
            Counter(("dalle" if x["image"].split("/")[0].lower().startswith("dall") else "vg") for x in out_rows)
        ),
    }


def build_amber(rng: random.Random) -> dict:
    qs = load_jsonl(ROOT / "eval_results/amber/questions/amber_dr_llava.jsonl")
    ann = {x["id"]: x for x in json.loads((ROOT / "AMBER/data/annotations.json").read_text(encoding="utf-8"))}
    items = []
    for q in qs:
        a = ann[q["question_id"]]
        rec = dict(q)
        rec["label"] = a.get("truth")
        rec["amber_type"] = a.get("type")
        items.append(rec)
    sampled = stratified_sample(
        items,
        lambda x: (str(x.get("label")).lower(), str(x.get("amber_type"))),
        AMBER_RATIO,
        rng,
    )
    sampled.sort(key=lambda x: int(x["question_id"]))
    dump_jsonl(OUT / "amber_dr_questions.jsonl", sampled)
    dump_json(OUT / "amber_dr_ids.json", [x["question_id"] for x in sampled])
    return {
        "full": len(items),
        "sampled": len(sampled),
        "full_label": dict(Counter(str(x.get("label")).lower() for x in items)),
        "sampled_label": dict(Counter(str(x.get("label")).lower() for x in sampled)),
        "full_type": dict(Counter(x.get("amber_type") for x in items)),
        "sampled_type": dict(Counter(x.get("amber_type") for x in sampled)),
    }


def main() -> None:
    rng = random.Random(SEED)
    meta = {
        "seed": SEED,
        "ratios": {"rbench": RBENCH_RATIO, "mmrel_adv": MMREL_RATIO, "amber_dr": AMBER_RATIO},
        "rbench": build_rbench(rng),
        "mmrel_adv": build_mmrel(rng),
        "amber_dr": build_amber(rng),
        "note": "Frozen at creation. Do not resample because a model looks weak on this subset.",
    }
    dump_json(OUT / "subset_meta.json", meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
