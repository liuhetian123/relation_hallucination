#!/usr/bin/env python3
"""Build leakage-safe PSG visual-counterfactual triplets and Phase 5B QC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
PRED_CAP = 0.10
FORMATS = ("F1", "F2", "F3")


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--psg", type=Path, default=Path("/data/storage22t/lht/datasets/OpenPSG_annotations/psg.json"))
    p.add_argument("--coco", type=Path, default=Path("/data/storage22t/lht/datasets/COCO2017"))
    p.add_argument("--safe-ids", type=Path, default=ROOT / "data_manifests/psg_train_excluding_reefknot_image_ids.txt")
    p.add_argument("--rules", type=Path, default=Path(__file__).with_name("psg_relation_rules.yaml"))
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/phase5b")
    p.add_argument("--work-dir", type=Path, default=Path("/data/storage22t/lht/datasets/phase5b_work"))
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--heldout-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verified", type=Path, help="Verifier JSONL; if supplied, retain only exact expected judgments")
    p.add_argument("--qc-size", type=int, default=100)
    return p.parse_args()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def article(noun: str) -> str:
    return "an" if noun[:1].lower() in "aeiou" else "a"


def clean_category(name: str) -> str:
    aliases = {
        "door-stuff": "door", "floor-wood": "wooden floor", "mirror-stuff": "mirror",
        "tree-merged": "tree", "fence-merged": "fence", "ceiling-merged": "ceiling",
        "sky-other-merged": "sky", "cabinet-merged": "cabinet", "table-merged": "table",
        "floor-other-merged": "floor", "pavement-merged": "pavement", "mountain-merged": "mountain",
        "grass-merged": "grass", "dirt-merged": "dirt", "paper-merged": "paper",
        "food-other-merged": "food", "building-other-merged": "building",
        "rock-merged": "rock", "wall-other-merged": "wall", "rug-merged": "rug",
    }
    if name in aliases:
        return aliases[name]
    return name.replace("-merged", "").replace("-other", "").replace("-", " ")


def statement(subject: str, predicate: str, obj: str, rules: dict) -> str:
    phrase = rules["rules"][predicate]["phrase"]
    return f"{article(subject).capitalize()} {subject} {phrase} {article(obj)} {obj}."


def prompts(subject: str, predicate: str, obj: str, rules: dict) -> dict:
    stmt = statement(subject, predicate, obj, rules)
    phrase = rules["rules"][predicate]["phrase"]
    question = f"Is {article(subject)} {subject} {phrase[3:]} {article(obj)} {obj}?"
    return {
        "statement": stmt,
        "F1": f"Does the image support this statement?\nStatement: {stmt}\nAnswer Yes or No only.",
        "F2": f"{question}\nAnswer Yes or No.",
        "F3": f"{question[:-1]} in the image? Please answer with one word.",
    }


def image_path(row: dict, coco: Path) -> Path:
    return coco / row["file_name"]


def audit(psg: dict, safe: set[str], rules: dict, coco: Path) -> dict:
    predicates = psg["predicate_classes"]
    rule_keys = set(rules["rules"])
    family_members = [x for values in rules["families"].values() for x in values]
    safe_rows = [x for x in psg["data"] if str(x["image_id"]) in safe]
    counts = Counter(predicates[r[2]] for row in safe_rows for r in row["relations"])
    return {
        "psg_images": len(psg["data"]),
        "safe_manifest_ids": len(safe),
        "safe_rows_found": len(safe_rows),
        "safe_images_with_relations": sum(bool(x["relations"]) for x in safe_rows),
        "safe_relations": sum(len(x["relations"]) for x in safe_rows),
        "predicate_count": len(predicates),
        "predicate_order": predicates,
        "predicate_frequency_safe": dict(counts),
        "rule_keys_exact": rule_keys == set(predicates),
        "missing_rules": sorted(set(predicates) - rule_keys),
        "extra_rules": sorted(rule_keys - set(predicates)),
        "family_members_exact": Counter(family_members) == Counter(predicates),
        "missing_rgb": sum(not image_path(x, coco).is_file() for x in safe_rows),
        "psg_test_ids_in_safe_manifest": len(set(map(str, psg["test_image_ids"])) & safe),
    }


def build_candidates(psg: dict, safe: set[str], rules: dict, coco: Path, seed: int) -> list[dict]:
    preds = psg["predicate_classes"]
    classes = [clean_category(x) for x in psg["thing_classes"] + psg["stuff_classes"]]
    family = {p: fam for fam, values in rules["families"].items() for p in values}
    rows = [x for x in psg["data"] if str(x["image_id"]) in safe and x["relations"]]
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    records = []
    for row in rows:
        pair_gt = defaultdict(set)
        for s, o, p in row["relations"]:
            pair_gt[(s, o)].add(preds[p])
        for rel_idx, (s, o, p) in enumerate(row["relations"]):
            if s >= len(row["annotations"]) or o >= len(row["annotations"]):
                continue
            sa, oa = row["annotations"][s], row["annotations"][o]
            rec = {
                "image_id": str(row["image_id"]), "coco_image_id": str(row["coco_image_id"]),
                "image": str(image_path(row, coco)), "file_name": row["file_name"],
                "subject_index": s, "object_index": o,
                "subject_instance_id": row["segments_info"][s]["id"],
                "object_instance_id": row["segments_info"][o]["id"],
                "subject": classes[sa["category_id"]], "object": classes[oa["category_id"]],
                "subject_category_id": sa["category_id"], "object_category_id": oa["category_id"],
                "subject_box": sa["bbox"], "object_box": oa["bbox"],
                "predicate": preds[p], "predicate_index": p, "relation_index": rel_idx,
                "pair_gt_relations": sorted(pair_gt[(s, o)]), "family": family[preds[p]],
            }
            records.append(rec)
            by_pair[(rec["subject"], rec["object"])].append(rec)
    observed = defaultdict(Counter)
    for r in records:
        observed[(r["subject"], r["object"])][r["predicate"]] += 1
    rng = random.Random(seed)
    rng.shuffle(records)
    out = []
    for r in records:
        blocked = set(r["pair_gt_relations"]) | set(rules["rules"][r["predicate"]]["entails"])
        choices = [
            (n, p) for p, n in observed[(r["subject"], r["object"])].items()
            if p not in blocked and p != r["predicate"] and family[p] == r["family"]
        ]
        if not choices:
            choices = [
                (n, p) for p, n in observed[(r["subject"], r["object"])].items()
                if p not in blocked and p != r["predicate"]
            ]
        if not choices:
            continue
        choices.sort(reverse=True)
        max_n = choices[0][0]
        relation_cf = rng.choice([p for n, p in choices if n == max_n])
        visual_pool = by_pair[(r["subject"], r["object"])]
        # Very common pairs (for example person→person) contain tens of thousands
        # of records. A deterministic random probe is sufficient and avoids O(n²).
        probe = rng.sample(visual_pool, min(256, len(visual_pool)))
        visual = [
            x for x in probe
            if x["image_id"] != r["image_id"] and r["predicate"] not in x["pair_gt_relations"]
        ]
        same_family = [x for x in visual if x["family"] == r["family"] and x["predicate"] != r["predicate"]]
        if same_family:
            visual = same_family
        if not visual:
            continue
        v = rng.choice(visual)
        fmt = FORMATS[int(hashlib.md5(f"{r['image_id']}:{r['relation_index']}".encode()).hexdigest(), 16) % 3]
        unit = dict(r)
        unit.update({
            "id": f"psg_{r['image_id']}_{r['relation_index']}", "prompt_format": fmt,
            "relation_cf_predicate": relation_cf,
            "visual_cf": {
                "image_id": v["image_id"], "coco_image_id": v["coco_image_id"], "image": v["image"],
                "file_name": v["file_name"], "subject_index": v["subject_index"], "object_index": v["object_index"],
                "subject_instance_id": v["subject_instance_id"], "object_instance_id": v["object_instance_id"],
                "subject_box": v["subject_box"], "object_box": v["object_box"],
                "pair_gt_relations": v["pair_gt_relations"], "selected_gt_predicate": v["predicate"],
            },
            "factual": {"prompts": prompts(r["subject"], r["predicate"], r["object"], rules), "target": "Yes"},
            "relation_cf": {"prompts": prompts(r["subject"], relation_cf, r["object"], rules), "target": "No"},
            "visual_cf_prompt_identical_to_factual": True,
            "visual_cf_target": "No",
        })
        out.append(unit)
    return out


def load_verified(path: Path | None) -> dict[str, dict]:
    if not path:
        return {}
    return {r["id"]: r for line in path.read_text(encoding="utf-8").splitlines() if line.strip() for r in [json.loads(line)]}


def select(candidates: list[dict], n_train: int, n_held: int, verified: dict, seed: int) -> tuple[list[dict], list[dict], dict]:
    expected = {"factual": "yes", "relation_cf": "no", "visual_cf": "no"}
    failures = Counter()
    eligible = []
    for row in candidates:
        if verified:
            v = verified.get(row["id"])
            if not v:
                failures["missing_verification"] += 1
                continue
            bad = [k for k, val in expected.items() if str(v.get(k, {}).get("judgment", "")).lower() != val]
            if bad:
                for k in bad:
                    failures[f"{k}_{v.get(k, {}).get('judgment', 'missing')}"] += 1
                continue
            row["automatic_verification"] = v
        eligible.append(row)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    # One factual relation per image improves diversity and makes image-level split auditing unambiguous.
    one_per_image = {}
    for row in eligible:
        one_per_image.setdefault(row["image_id"], row)
    pool = list(one_per_image.values())
    rng.shuffle(pool)
    train_images, held_images = set(), set()
    train, held = [], []

    def round_robin_pick(target: int, forbidden: set[str], used: set[str], cap: int) -> list[dict]:
        buckets = defaultdict(list)
        for item in pool:
            buckets[item["predicate"]].append(item)
        order = sorted(buckets, key=lambda key: (len(buckets[key]), key))
        picked, counts = [], Counter()
        while len(picked) < target:
            progressed = False
            for pred in order:
                while buckets[pred]:
                    item = buckets[pred].pop()
                    involved = {item["image_id"], item["visual_cf"]["image_id"]}
                    if involved & forbidden or involved & used or counts[pred] >= cap:
                        continue
                    picked.append(item)
                    used |= involved
                    counts[pred] += 1
                    progressed = True
                    break
                if len(picked) == target:
                    break
            if not progressed:
                break
        return picked

    # Reserve one held-out example for predicates with enough alternatives,
    # without sacrificing the only viable examples of extremely rare classes.
    reserve_buckets = defaultdict(list)
    for item in pool:
        reserve_buckets[item["predicate"]].append(item)
    held = []
    for pred in sorted(reserve_buckets, key=lambda key: (len(reserve_buckets[key]), key)):
        if len(reserve_buckets[pred]) < 10:
            continue
        for item in reserve_buckets[pred]:
            involved = {item["image_id"], item["visual_cf"]["image_id"]}
            if not involved & held_images:
                held.append(item)
                held_images |= involved
                break
    train = round_robin_pick(n_train, held_images, train_images, max(1, int(n_train * PRED_CAP)))
    held_counts = Counter(x["predicate"] for x in held)
    for item in pool:
        if len(held) == n_held:
            break
        involved = {item["image_id"], item["visual_cf"]["image_id"]}
        if involved & train_images or involved & held_images or held_counts[item["predicate"]] >= max(1, int(n_held * PRED_CAP)):
            continue
        held.append(item)
        held_images |= involved
        held_counts[item["predicate"]] += 1
    stats = {
        "candidate_triplets": len(candidates), "verified_records": len(verified),
        "eligible_after_verification": len(eligible), "unique_factual_images_eligible": len(pool),
        "train": len(train), "heldout": len(held), "train_predicates": dict(Counter(x["predicate"] for x in train)),
        "heldout_predicates": dict(Counter(x["predicate"] for x in held)),
        "train_images_all_branches": len(train_images), "heldout_images_all_branches": len(held_images),
        "split_image_overlap": len(train_images & held_images), "verification_failures": dict(failures),
        "requested": {"train": n_train, "heldout": n_held},
    }
    return train, held, stats


def boxed(src: str, sb: list, ob: list, dst: Path) -> None:
    with Image.open(src) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        draw.rectangle(sb, outline="#00ff00", width=max(3, im.width // 250))
        draw.rectangle(ob, outline="#ff3030", width=max(3, im.width // 250))
        draw.text((max(0, sb[0]), max(0, sb[1] - 14)), "SUBJECT", fill="#00ff00")
        draw.text((max(0, ob[0]), max(0, ob[1] - 14)), "OBJECT", fill="#ff3030")
        im.thumbnail((900, 900))
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=82, optimize=True)


def make_qc(train: list[dict], out: Path, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    by_pred = defaultdict(list)
    for x in train:
        by_pred[x["predicate"]].append(x)
    chosen = []
    while len(chosen) < min(n, len(train)):
        progressed = False
        for pred in sorted(by_pred):
            if by_pred[pred] and len(chosen) < n:
                chosen.append(rng.choice(by_pred[pred]))
                by_pred[pred].remove(chosen[-1])
                progressed = True
        if not progressed:
            break
    qdir = out / "qc"
    image_dir = qdir / "images"
    qdir.mkdir(parents=True, exist_ok=True)
    fields = ["id", "positive_clear", "relation_cf_clear_negative", "visual_cf_clear_negative",
              "reference_clear", "grammar_natural", "notes"]
    with (qdir / "review_100.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in chosen:
            w.writerow({"id": r["id"]})
    cards = []
    for i, r in enumerate(chosen, 1):
        p = image_dir / f"{i:03d}_positive.jpg"
        v = image_dir / f"{i:03d}_visual_cf.jpg"
        boxed(r["image"], r["subject_box"], r["object_box"], p)
        boxed(r["visual_cf"]["image"], r["visual_cf"]["subject_box"], r["visual_cf"]["object_box"], v)
        av = r.get("automatic_verification", {"status": "BLOCKED_UNAVAILABLE"})
        cards.append(f"""<section><h2>{i:03d} · {html.escape(r['id'])} · {html.escape(r['predicate'])} · {r['prompt_format']}</h2>
<div class="imgs"><figure><img src="images/{p.name}"><figcaption>I+ (green=subject, red=object)</figcaption></figure>
<figure><img src="images/{v.name}"><figcaption>I− (green=subject, red=object)</figcaption></figure></div>
<p><b>Entities:</b> {html.escape(r['subject'])} → {html.escape(r['object'])}</p>
<ol><li><b>Factual / Yes:</b> {html.escape(r['factual']['prompts'][r['prompt_format']])}<br>auto={html.escape(str(av.get('factual', {})))}</li>
<li><b>Relation CF / No:</b> {html.escape(r['relation_cf']['prompts'][r['prompt_format']])}<br>auto={html.escape(str(av.get('relation_cf', {})))}</li>
<li><b>Visual CF / No; text identical to factual:</b> {html.escape(r['factual']['prompts'][r['prompt_format']])}<br>auto={html.escape(str(av.get('visual_cf', {})))}</li></ol></section>""")
    page = """<!doctype html><meta charset="utf-8"><title>Phase 5B QC</title>
<style>body{font:15px system-ui;max-width:1400px;margin:auto;padding:20px}section{border-top:2px solid #555;padding:16px}.imgs{display:flex;gap:16px}figure{width:48%;margin:0}img{max-width:100%;max-height:620px}li{margin:10px 0;white-space:pre-wrap}</style>
<h1>Phase 5B — 100 triplet human QC</h1><p>Do not train before all five TSV columns are filled with 1/0.</p>""" + "\n".join(cards)
    (qdir / "review_100.html").write_text(page, encoding="utf-8")
    return [x["id"] for x in chosen]


def main() -> None:
    a = args()
    psg = json.loads(a.psg.read_text(encoding="utf-8"))
    rules = json.loads(a.rules.read_text(encoding="utf-8"))
    safe = {x.strip() for x in a.safe_ids.read_text(encoding="utf-8").splitlines() if x.strip()}
    audit_stats = audit(psg, safe, rules, a.coco)
    if not audit_stats["rule_keys_exact"] or not audit_stats["family_members_exact"] or audit_stats["psg_test_ids_in_safe_manifest"]:
        raise SystemExit(f"audit failed: {audit_stats}")
    dump_json(a.out_dir / "psg_audit.json", audit_stats)
    candidates = build_candidates(psg, safe, rules, a.coco, a.seed)
    dump_jsonl(a.work_dir / "candidate_triplets.jsonl", candidates)
    verified = load_verified(a.verified)
    train, held, stats = select(candidates, a.train_size, a.heldout_size, verified, a.seed)
    stats["audit"] = audit_stats
    stats["candidate_file"] = str(a.work_dir / "candidate_triplets.jsonl")
    stats["automatic_verification_status"] = (
        "completed_independent_family" if verified else "blocked_unavailable_not_self_verified"
    )
    dump_jsonl(a.out_dir / "train_triplets.jsonl", train)
    dump_jsonl(a.out_dir / "heldout_triplets.jsonl", held)
    if train and a.qc_size > 0:
        stats["qc_ids"] = make_qc(train, a.out_dir, a.qc_size, a.seed)
    dump_json(a.out_dir / "build_stats.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
