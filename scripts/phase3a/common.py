"""Shared helpers for Phase 3A structured SRO scaling data."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import importlib.util

_spec = importlib.util.spec_from_file_location("phase2_common", ROOT / "scripts/phase2/common.py")
_p2 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_p2)
META_REL_HEADS = _p2.META_REL_HEADS
SYNONYM_GROUPS = _p2.SYNONYM_GROUPS
VISUAL_RELATIONS = _p2.VISUAL_RELATIONS
are_synonyms = _p2.are_synonyms
is_meta_relation = _p2.is_meta_relation
llava_verify_sample = _p2.llava_verify_sample
normalize_relation = _p2.normalize_relation
parse_yes_no_uncertain = _p2.parse_yes_no_uncertain
stem = _p2.stem
strip_caption = _p2.strip_caption
too_close_for_hard = _p2.too_close_for_hard
too_close_for_random = _p2.too_close_for_random
verification_prompt = _p2.verification_prompt

IMAGE_ROOT = Path("/data/lht/relsim_dataset/relsim_images")
RELSIM_100K = Path("/data/lht/relsim_dataset/relsim_llava_100k.json")
INTERNVL_PATH = Path(
    "/data/storage22t/lht/hf_cache/models--OpenGVLab--InternVL3_5-8B-HF"
    "/snapshots/741a7d03020411e666c6109218ab71e08151ef86"
)
TEACHER_NAME = "InternVL3.5-8B"
SEED = 42
MASTER_N = 3000
MAX_REL_FRAC = 0.05
NEG_TRIES = 5

SRO_PROMPT = """Identify one primary binary relation that is clearly and directly visible in the image.

Return exactly one JSON object:

{
  "subject": "...",
  "relation": "...",
  "object": "...",
  "confidence": "high|medium|low"
}

Requirements:
- subject and object must be concrete visible entities;
- relation must describe a direct visual relation between them;
- relation should be a short verb / verb phrase / spatial relation;
- do not output attributes, scene descriptions, counts, or vague arrangements;
- avoid relations requiring hidden intent or external knowledge;
- prefer the most visually salient subject-relation-object triplet;
- use concise basic-level entity names.
"""

NEG_PROMPT = """Subject: {subject}
Object: {object}
Candidate relation: {negative_relation}

Is this relation clearly visually supported between the specified subject and object?

Answer:
Yes
No
Uncertain
"""

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.I)
NON_ENTITY = {
    "scene",
    "image",
    "photo",
    "picture",
    "background",
    "foreground",
    "something",
    "someone",
    "object",
    "thing",
    "entity",
    "area",
    "view",
    "composition",
}
ATTR_REL_HEADS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "seem",
    "seems",
    "appear",
    "appears",
    "look",
    "looks",
    "colored",
    "coloured",
}


def load_json(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_jsonl_map(path: Path, key: str = "id") -> dict:
    done = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                done[rec[key]] = rec
    return done


def extract_json_object(text: str) -> tuple[dict | None, bool]:
    raw = (text or "").strip()
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


def strip_article(text: str) -> str:
    return ARTICLE_RE.sub("", strip_caption(text)).strip()


def article(noun: str) -> str:
    w = (noun or "").strip().split()[0].lower() if (noun or "").strip() else "x"
    return "an" if w[:1] in "aeiou" else "a"


def render_statement(subject: str, relation: str, obj: str) -> str:
    s = strip_article(subject)
    r = strip_caption(relation)
    o = strip_article(obj)
    stmt = f"{article(s)} {s} is {r} {article(o)} {o}."
    return stmt[0].upper() + stmt[1:]


def clean_span(text: str) -> str:
    t = strip_caption(str(text or ""))
    t = t.replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip(" .,;:!?\"'`")
    return t


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def qc_sro(parsed: dict | None) -> tuple[str, dict]:
    if not parsed:
        return "json_parse_error", {}
    subject = clean_span(parsed.get("subject", ""))
    relation = clean_span(parsed.get("relation", ""))
    obj = clean_span(parsed.get("object", ""))
    conf = str(parsed.get("confidence", "")).strip().lower()
    fields = {
        "subject": subject,
        "relation": relation,
        "object": obj,
        "confidence": conf,
        "subject_norm": normalize_relation(subject),
        "relation_norm": normalize_relation(relation),
        "object_norm": normalize_relation(obj),
    }
    if conf != "high":
        return "low_confidence", fields
    if not subject or not relation or not obj:
        return "empty_field", fields
    if fields["subject_norm"] == fields["object_norm"]:
        return "subject_eq_object", fields
    if not (2 <= len(relation) <= 40):
        return "relation_len", fields
    if _word_count(relation) > 5 or _word_count(subject) > 4 or _word_count(obj) > 4:
        return "too_long", fields
    if any(ch.isdigit() for ch in f"{subject}{relation}{obj}"):
        return "has_digit", fields
    if "," in subject or "," in obj or ";" in relation:
        return "not_binary", fields
    head = fields["relation_norm"].split()[0] if fields["relation_norm"] else ""
    if head in ATTR_REL_HEADS or is_meta_relation(relation) or head in META_REL_HEADS:
        return "meta_or_attribute", fields
    for token in fields["subject_norm"].split() + fields["object_norm"].split():
        if token in NON_ENTITY:
            return "non_entity", fields
    if fields["relation_norm"] in {"of", "and", "with", "for", "to", "from"}:
        return "vague_relation", fields
    return "ok", fields


def too_close_negative(gt: str, cand: str) -> bool:
    if too_close_for_random(gt, cand) or too_close_for_hard(gt, cand):
        return True
    g, c = normalize_relation(gt), normalize_relation(cand)
    if not g or not c:
        return True
    if stem(g.split()[0]) == stem(c.split()[0]) and len(stem(g.split()[0])) >= 3:
        return True
    return are_synonyms(g, c)


def diversity_stats(rows: list[dict]) -> dict:
    def vals(key):
        return [normalize_relation(r.get(key) or "") for r in rows]

    rels = vals("relation")
    subs = vals("subject")
    objs = vals("object")
    so = [f"{s}||{o}" for s, o in zip(subs, objs)]
    sr = [f"{s}||{r}" for s, r in zip(subs, rels)]
    ro = [f"{r}||{o}" for r, o in zip(rels, objs)]
    sro = [f"{s}||{r}||{o}" for s, r, o in zip(subs, rels, objs)]
    rel_freq = Counter(rels)
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "unique_images": len({Path(str(r.get("image", ""))).name for r in rows}),
        "unique_relations": len(set(rels)),
        "unique_subjects": len(set(subs)),
        "unique_objects": len(set(objs)),
        "unique_subject_object": len(set(so)),
        "unique_subject_relation": len(set(sr)),
        "unique_relation_object": len(set(ro)),
        "unique_sro": len(set(sro)),
        "max_relation_frac": (rel_freq.most_common(1)[0][1] / n) if rel_freq else 0.0,
        "top_relations": rel_freq.most_common(30),
    }
