#!/usr/bin/env python3
"""Extract relation phrases from Phase 2B.5 matched statements / MMRel templates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_p2 = _load("phase2_common", ROOT / "scripts/phase2/common.py")
_p4 = _load("phase4a_common", ROOT / "scripts/phase4a/common.py")
sys.path.insert(0, str(ROOT / "scripts/phase2b5"))
from convert_statements import MMREL_DOES, MMREL_PREP, MMREL_TO, strip_question  # noqa: E402

SYNONYM_GROUPS = _p2.SYNONYM_GROUPS
are_synonyms = _p2.are_synonyms
normalize_relation = _p2.normalize_relation
strip_caption = _p2.strip_caption
EXTRA_SYNONYM_GROUPS = _p4.EXTRA_SYNONYM_GROUPS
extra_synonyms = _p4.extra_synonyms
lemma_token = _p4.lemma_token
strip_article = _p4.strip_article

OUT = ROOT / "eval_results/qwen/phase4b/metrics"
SEED = 42
TRAIL_IMAGE = re.compile(r"\s+in the image\.?\s*$", re.I)
COMPOUND_PREPS = [
    "in front of",
    "on top of",
    "next to",
    "out of",
    "inside of",
    "outside of",
    "on top",
    "in front",
]
SINGLE_PREPS = {
    "onto",
    "into",
    "upon",
    "atop",
    "from",
    "with",
    "beside",
    "between",
    "among",
    "through",
    "across",
    "along",
    "around",
    "against",
    "under",
    "over",
    "behind",
    "inside",
    "outside",
    "above",
    "below",
    "within",
    "toward",
    "towards",
    "near",
    "off",
    "out",
    "on",
    "in",
    "at",
    "to",
    "by",
    "of",
    "over",
    "underneath",
    "beneath",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def exact_key(rel: str) -> str:
    return normalize_relation(strip_article(strip_caption(rel)))


def lemma_key(rel: str, nlp) -> str:
    rel = exact_key(rel)
    if not rel:
        return ""
    if nlp is None:
        return " ".join(lemma_token(t) for t in rel.split())
    doc = nlp(rel)
    parts = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        lem = (tok.lemma_ or tok.text).lower()
        if lem in {"-pron-"}:
            lem = tok.text.lower()
        parts.append(lemma_token(lem) if lem else tok.text.lower())
    return " ".join(parts).strip()


def leading_prep(text: str) -> str:
    low = (text or "").strip().lower()
    low = re.sub(r"^(a|an|the)\s+", "", low)
    for phrase in COMPOUND_PREPS:
        if low == phrase or low.startswith(phrase + " "):
            return phrase
    first = re.split(r"[^A-Za-z]+", low, maxsplit=1)[0] if low else ""
    if first in SINGLE_PREPS:
        return first
    return ""


def extract_mmrel(official: str, tag: str) -> tuple[str | None, str]:
    q = strip_question(official)
    m = MMREL_TO.match(q)
    if m:
        return exact_key(m.group(3)), "mmrel_spatial_to"
    m = MMREL_PREP.match(q)
    if m:
        return exact_key(m.group(3)), "mmrel_spatial_prep"
    m = MMREL_DOES.match(q)
    if m:
        return exact_key(m.group(3)), "mmrel_action"
    return None, f"mmrel_fail:{tag}"


def extract_rbench(statement: str, nlp) -> tuple[str | None, str]:
    text = TRAIL_IMAGE.sub("", strip_caption(statement)).strip().rstrip(".")
    if not text:
        return None, "empty"
    doc = nlp(text)
    if not len(doc):
        return None, "empty"
    if doc[0].text.lower() == "there":
        return None, "existence"
    roots = [t for t in doc if t.dep_ == "ROOT"]
    root = roots[0] if roots else doc[0]
    rest = text[root.idx + len(root.text) :].strip()
    rest = TRAIL_IMAGE.sub("", rest).strip(" .")

    if root.pos_ == "VERB":
        prep = leading_prep(rest)
        rel = f"{root.text} {prep}".strip() if prep else root.text
        return exact_key(rel), "verb"
    if root.lemma_ == "be" or root.pos_ in {"AUX"}:
        prep = leading_prep(rest)
        if prep:
            return exact_key(prep), "prep"
        acomp = [c for c in root.children if c.dep_ in {"acomp", "attr", "acomp"}]
        if acomp and acomp[0].pos_ == "ADJ":
            return None, "attribute"
        if rest:
            first = rest.split()[0].lower() if rest.split() else ""
            if first in {"a", "an", "the"}:
                return None, "attribute"
        return None, "attribute"
    if root.pos_ == "ADJ":
        return None, "attribute"
    prep = leading_prep(text)
    if prep:
        return exact_key(prep), "prep_fallback"
    return None, "unparsed"


def mmrel_domain(image: str) -> str:
    p = (image or "").replace("\\", "/").lower()
    if p.startswith("vg/") or "/vg_" in p or p.startswith("vg_"):
        return "vg"
    if "dall-e" in p or "dalle" in p:
        return "dalle"
    return "other"


def vocab_from_old(path: Path) -> set[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = set()
    for rec in rows:
        for key in ("gt_relation", "neg_relation"):
            v = exact_key(rec.get(key) or "")
            if v:
                out.add(v)
    return out


def vocab_from_clean(path: Path) -> set[str]:
    out = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for key in ("positive_relation", "negative_relation"):
                v = exact_key(rec.get(key) or "")
                if v:
                    out.add(v)
    return out


def synonym_hit(rel: str, vocab: set[str]) -> bool:
    e = exact_key(rel)
    if e in vocab:
        return True
    for v in vocab:
        if are_synonyms(e, v) or extra_synonyms(e, v):
            return True
        for group in SYNONYM_GROUPS:
            if e in group and v in group:
                return True
        for group in EXTRA_SYNONYM_GROUPS:
            if e in group and v in group:
                return True
    return False


def match_level(rel: str, vocab_exact: set[str], vocab_lemma: set[str], nlp) -> str:
    e = exact_key(rel)
    if e in vocab_exact:
        return "exact"
    if lemma_key(rel, nlp) in vocab_lemma:
        return "lemma"
    if synonym_hit(rel, vocab_exact):
        return "synonym-family"
    return "unseen"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(OUT / "relation_extraction.jsonl"))
    p.add_argument("--out-meta", default=str(OUT / "relation_extraction_meta.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import spacy

    nlp = spacy.load("en_core_web_lg")
    rbench = load_jsonl(ROOT / "data/phase2b5/rbench_fast_matched_prompt.jsonl")
    mmrel = load_jsonl(ROOT / "data/phase2b5/mmrel_adv_fast_matched_prompt.jsonl")
    vocab_old = vocab_from_old(ROOT / "data/phase4a/train_old.json")
    vocab_clean = vocab_from_clean(ROOT / "data/phase4a/clean_sro3000.jsonl")
    lemma_old = {lemma_key(v, nlp) for v in vocab_old}
    lemma_clean = {lemma_key(v, nlp) for v in vocab_clean}

    rows = []
    unparsed = []
    reasons = {"rbench": Counter(), "mmrel_adv": Counter()}

    for rec in rbench:
        rel, why = extract_rbench(rec["matched_statement"], nlp)
        reasons["rbench"][why] += 1
        ok = rel is not None
        item = {
            "benchmark": "rbench",
            "question_id": rec["question_id"],
            "image": rec["image"],
            "gold": rec.get("gold") or rec.get("label"),
            "official_question": rec["official_question"],
            "matched_statement": rec["matched_statement"],
            "conversion_tag": rec.get("conversion_tag"),
            "relation": rel,
            "relation_exact": exact_key(rel) if rel else "",
            "relation_lemma": lemma_key(rel, nlp) if rel else "",
            "parse_ok": ok,
            "parse_reason": why,
            "domain": "",
            "match_old": match_level(rel, vocab_old, lemma_old, nlp) if rel else "unparsed",
            "match_clean": match_level(rel, vocab_clean, lemma_clean, nlp) if rel else "unparsed",
        }
        rows.append(item)
        if not ok:
            unparsed.append(item)

    for rec in mmrel:
        rel, why = extract_mmrel(rec["official_question"], rec.get("conversion_tag") or "")
        reasons["mmrel_adv"][why] += 1
        ok = rel is not None
        item = {
            "benchmark": "mmrel_adv",
            "question_id": rec["question_id"],
            "image": rec["image"],
            "gold": rec.get("gold") or rec.get("label"),
            "official_question": rec["official_question"],
            "matched_statement": rec["matched_statement"],
            "conversion_tag": rec.get("conversion_tag"),
            "relation": rel,
            "relation_exact": exact_key(rel) if rel else "",
            "relation_lemma": lemma_key(rel, nlp) if rel else "",
            "parse_ok": ok,
            "parse_reason": why,
            "domain": mmrel_domain(rec["image"]),
            "match_old": match_level(rel, vocab_old, lemma_old, nlp) if rel else "unparsed",
            "match_clean": match_level(rel, vocab_clean, lemma_clean, nlp) if rel else "unparsed",
        }
        rows.append(item)
        if not ok:
            unparsed.append(item)

    dump_jsonl(Path(args.out), rows)
    n_rb = sum(1 for r in rows if r["benchmark"] == "rbench")
    n_mm = sum(1 for r in rows if r["benchmark"] == "mmrel_adv")
    n_rb_fail = sum(1 for r in rows if r["benchmark"] == "rbench" and not r["parse_ok"])
    n_mm_fail = sum(1 for r in rows if r["benchmark"] == "mmrel_adv" and not r["parse_ok"])
    meta = {
        "n_rbench": n_rb,
        "n_mmrel": n_mm,
        "rbench_unparsed": n_rb_fail,
        "rbench_unparsed_rate": n_rb_fail / n_rb if n_rb else 0,
        "mmrel_unparsed": n_mm_fail,
        "mmrel_unparsed_rate": n_mm_fail / n_mm if n_mm else 0,
        "parse_reasons": {k: dict(v) for k, v in reasons.items()},
        "vocab_old_n": len(vocab_old),
        "vocab_clean_n": len(vocab_clean),
        "lemma_old_n": len(lemma_old),
        "lemma_clean_n": len(lemma_clean),
        "seen_lemma_old": {
            "rbench": sum(1 for r in rows if r["benchmark"] == "rbench" and r["match_old"] in {"exact", "lemma"}),
            "mmrel_adv": sum(1 for r in rows if r["benchmark"] == "mmrel_adv" and r["match_old"] in {"exact", "lemma"}),
        },
        "seen_lemma_clean": {
            "rbench": sum(1 for r in rows if r["benchmark"] == "rbench" and r["match_clean"] in {"exact", "lemma"}),
            "mmrel_adv": sum(1 for r in rows if r["benchmark"] == "mmrel_adv" and r["match_clean"] in {"exact", "lemma"}),
        },
    }
    Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_meta).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    print(f"wrote {args.out} n={len(rows)} unparsed={len(unparsed)}", flush=True)


if __name__ == "__main__":
    main()
