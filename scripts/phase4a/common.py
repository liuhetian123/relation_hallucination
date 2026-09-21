"""Shared helpers for Phase 4A clean counterfactual negatives."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_p3 = _load("phase3a_common", ROOT / "scripts/phase3a/common.py")
_p2 = _load("phase2_common", ROOT / "scripts/phase2/common.py")

IMAGE_ROOT = _p3.IMAGE_ROOT
INTERNVL_PATH = _p3.INTERNVL_PATH
TEACHER_NAME = _p3.TEACHER_NAME
META_REL_HEADS = _p3.META_REL_HEADS
NON_ENTITY = _p3.NON_ENTITY
append_jsonl = _p3.append_jsonl
article = _p3.article
clean_span = _p3.clean_span
dump_json = _p3.dump_json
dump_jsonl = _p3.dump_jsonl
extract_json_object = _p3.extract_json_object
is_meta_relation = _p3.is_meta_relation
llava_verify_sample = _p3.llava_verify_sample
load_json = _p3.load_json
load_jsonl_map = _p3.load_jsonl_map
normalize_relation = _p3.normalize_relation
parse_yes_no_uncertain = _p3.parse_yes_no_uncertain
render_statement = _p3.render_statement
strip_article = _p3.strip_article
strip_caption = _p3.strip_caption
too_close_for_hard = _p3.too_close_for_hard
relation_family = _p2.relation_family
stem = _p2.stem

SEED = 42
MAX_ATTEMPTS = 3
DATA_DIR = ROOT / "data/phase4a"
MASTER_PATH = ROOT / "data/phase3a/master_sro_3000.jsonl"

CF_PROMPT = """You are constructing counterfactual relation examples for visual relation verification.

The following relation is TRUE in the image:

Subject: {subject}
True relation: {positive_relation}
Object: {object}

Generate THREE alternative relations between exactly the SAME subject and object.

Each alternative relation must satisfy ALL of the following:

1. It must be grammatically natural with the given subject and object.
2. It must be semantically plausible in some realistic or visually possible situation.
3. It must NOT be obviously false from language or common sense alone.
4. It must NOT be visually supported in the current image.
5. It must keep exactly the same subject and object roles.
6. Do not change, add, remove, rename, or replace the subject or object.
7. Do not output a synonym, paraphrase, inverse-expression, or trivial morphological variant of the true relation.
8. Prefer relations that are visually confusable with the true relation or plausible for the same entity pair.
9. The negative should require looking at the image to determine that it is false.
10. Avoid absurd or selectionally incompatible relations.

Field rules:
- "relation" MUST be only the short relation phrase, such as "parked near", "inside", or "holding".
- Do NOT put the subject name or object name in the "relation" field.
- "statement" is a full sentence for QC only, of the form "A {{subject}} is {{relation}} a {{object}}."

Bad examples:
- "a burger inserting a building"
- "a chair eating a person"
- "a car wearing a person"

Output JSON only. Do not write analysis, chain-of-thought, or markdown fences.

Return exactly one JSON object:

{{
  "candidates": [
    {{
      "relation": "...",
      "statement": "...",
      "linguistically_plausible": true,
      "visually_supported": false,
      "confidence": "high|medium|low"
    }},
    {{
      "relation": "...",
      "statement": "...",
      "linguistically_plausible": true,
      "visually_supported": false,
      "confidence": "high|medium|low"
    }},
    {{
      "relation": "...",
      "statement": "...",
      "linguistically_plausible": true,
      "visually_supported": false,
      "confidence": "high|medium|low"
    }}
  ]
}}
"""

RETRY_NOTE = """

Your previous output was not usable. Output JSON only, with no analysis. The "relation" field must be a short relation phrase, not the subject or object name.
"""

PLAUSIBLE_PROMPT = """Subject: {subject}
Relation: {candidate_relation}
Object: {object}

Statement:
{candidate_statement}

Question:
Is this statement grammatically natural and semantically plausible in at least one realistic or visually possible situation?

Do NOT judge whether it is true in any specific image.

Answer exactly one:
Plausible
Implausible
"""

VISUAL_PROMPT = """Subject: {subject}
Object: {object}
Candidate relation: {candidate_relation}

Question:
Is the candidate relation clearly visually supported between the specified subject and object in this image?

Answer exactly one:
Yes
No
Uncertain
"""

LEMMA = {
    "held": "hold",
    "holding": "hold",
    "holds": "hold",
    "worn": "wear",
    "wearing": "wear",
    "wears": "wear",
    "ridden": "ride",
    "riding": "ride",
    "rides": "ride",
    "sat": "sit",
    "sitting": "sit",
    "sits": "sit",
    "stood": "stand",
    "standing": "stand",
    "stands": "stand",
    "lay": "lie",
    "lying": "lie",
    "lain": "lie",
    "seen": "see",
    "seeing": "see",
    "sees": "see",
    "watched": "watch",
    "watching": "watch",
    "watches": "watch",
    "grasped": "grasp",
    "grasping": "grasp",
    "clutched": "clutch",
    "clutching": "clutch",
    "carried": "carry",
    "carrying": "carry",
    "ate": "eat",
    "eaten": "eat",
    "eating": "eat",
    "driven": "drive",
    "driving": "drive",
}

EXTRA_SYNONYM_GROUPS = [
    {"on", "upon", "on top of", "on top", "atop"},
    {"in", "inside", "within", "into"},
    {"near", "nearby", "close to", "next to", "beside", "adjacent to"},
    {"under", "underneath", "beneath", "below"},
    {"over", "above", "over top of"},
    {"looking at", "watching", "gazing at", "staring at", "looking towards"},
    {"cutting", "tearing", "slicing", "chopping", "snipping"},
    {"touching", "contacting"},
]


def qc_family(rel: str) -> str:
    fam = relation_family(rel)
    if fam in {"spatial", "contact"}:
        return fam
    r = normalize_relation(rel)
    tokens = set(r.split())
    spatial_tokens = {
        "on",
        "in",
        "at",
        "near",
        "over",
        "under",
        "behind",
        "beside",
        "inside",
        "outside",
        "above",
        "below",
        "against",
        "around",
        "across",
        "along",
        "between",
        "among",
        "atop",
        "upon",
        "front",
        "top",
    }
    if tokens & spatial_tokens or "front of" in r or "top of" in r or "next to" in r:
        return "spatial"
    head = r.split()[0] if r else ""
    if head.endswith("ing") or head.endswith("ed"):
        return "contact"
    return "other"


def lemma_token(token: str) -> str:
    t = normalize_relation(token)
    if t in LEMMA:
        return LEMMA[t]
    return stem(t)


def extra_synonyms(a: str, b: str) -> bool:
    a, b = normalize_relation(a), normalize_relation(b)
    for group in EXTRA_SYNONYM_GROUPS:
        if a in group and b in group:
            return True
    return False


def is_inverse_voice(gt: str, cand: str) -> bool:
    g, c = normalize_relation(gt), normalize_relation(cand)
    for src, other in ((g, c), (c, g)):
        if " by" in f" {other} ":
            head = other.split(" by", 1)[0].strip()
            if head and (
                too_close_for_hard(src, head)
                or lemma_token(src.split()[0]) == lemma_token(head.split()[0] if head else "")
            ):
                return True
        if other.endswith(" by"):
            head = other[: -len(" by")].strip()
            if head and lemma_token(src.split()[0]) == lemma_token(head.split()[0]):
                return True
    return False


def morphological_variant(gt: str, cand: str) -> bool:
    g, c = normalize_relation(gt), normalize_relation(cand)
    if not g or not c:
        return True
    gw, cw = g.split(), c.split()
    if lemma_token(gw[0]) == lemma_token(cw[0]) and len(lemma_token(gw[0])) >= 3:
        return True
    return False


def relation_mentions_entity(rel: str, subject: str, obj: str) -> bool:
    r = f" {normalize_relation(rel)} "
    for ent in (subject, obj):
        e = normalize_relation(strip_article(ent))
        if e and f" {e} " in r:
            return True
    return False


def as_bool(value, default=None):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in {"true", "yes", "1"}:
        return True
    if s in {"false", "no", "0"}:
        return False
    return default


def parse_plausible(text: str) -> str:
    raw = (text or "").strip()
    first = re.split(r"[\s,.:;!?]+", raw, maxsplit=1)[0].lower()
    if first.startswith("implausible"):
        return "implausible"
    if first.startswith("plausible"):
        return "plausible"
    low = raw.lower()
    if "implausible" in low and "plausible" not in low.replace("implausible", ""):
        return "implausible"
    if re.search(r"\bimplausible\b", low):
        return "implausible"
    if re.search(r"\bplausible\b", low):
        return "plausible"
    return "parse_error"


def qc_relation_span(rel: str) -> str | None:
    r = clean_span(rel)
    if not r:
        return "empty_relation"
    if not (2 <= len(r) <= 40):
        return "relation_len"
    if len(r.split()) > 5:
        return "too_long"
    if any(ch.isdigit() for ch in r):
        return "has_digit"
    if "," in r or ";" in r:
        return "not_binary"
    if re.search(r"\b(a|an|the)\b", normalize_relation(r)):
        return "has_article"
    head = normalize_relation(r).split()[0] if r else ""
    if head in META_REL_HEADS or is_meta_relation(r):
        return "meta_relation"
    if head in NON_ENTITY:
        return "non_entity"
    return None


def rule_reject(gt: str, cand: str, subject: str, obj: str) -> str | None:
    reason = qc_relation_span(cand)
    if reason:
        return reason
    if too_close_for_hard(gt, cand) or extra_synonyms(gt, cand):
        return "synonym_or_paraphrase"
    if morphological_variant(gt, cand):
        return "morphology"
    if is_inverse_voice(gt, cand):
        return "inverse"
    if relation_mentions_entity(cand, subject, obj):
        return "mentions_entity"
    return None


def normalize_candidate(raw: dict) -> dict | None:
    if not isinstance(raw, dict):
        return None
    rel = clean_span(raw.get("relation", ""))
    if not rel:
        return None
    return {
        "relation": rel,
        "relation_norm": normalize_relation(rel),
        "teacher_statement": clean_span(raw.get("statement", "")),
        "linguistically_plausible": as_bool(raw.get("linguistically_plausible")),
        "visually_supported": as_bool(raw.get("visually_supported")),
        "confidence": str(raw.get("confidence", "")).strip().lower(),
    }


def extract_candidates(parsed: dict | None) -> list[dict]:
    if not parsed:
        return []
    rows = parsed.get("candidates")
    if isinstance(rows, list):
        out = []
        for row in rows:
            cand = normalize_candidate(row)
            if cand:
                out.append(cand)
        return out
    cand = normalize_candidate(parsed)
    return [cand] if cand else []


def recover_relation(rel: str, statement: str, subject: str, obj: str) -> str:
    rel = clean_span(rel)
    stmt = clean_span(statement)
    ents = {
        normalize_relation(strip_article(subject)),
        normalize_relation(strip_article(obj)),
    }

    def is_entity_span(text: str) -> bool:
        n = normalize_relation(text)
        return bool(n) and n in ents

    if rel and not is_entity_span(rel) and not relation_mentions_entity(rel, subject, obj):
        return rel
    t = normalize_relation(stmt)
    s = normalize_relation(strip_article(subject))
    o = normalize_relation(strip_article(obj))
    if s and t.startswith(s + " "):
        t = t[len(s) :].strip()
    if o and t.endswith(" " + o):
        t = t[: -len(o)].strip()
    t = clean_span(t)
    if t and not is_entity_span(t):
        return t
    return rel


def parse_teacher_json(text: str):
    raw = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.I).strip()
    return extract_json_object(raw)


def teacher_flags_ok(cand: dict) -> str | None:
    if cand.get("linguistically_plausible") is not True:
        return "teacher_not_plausible"
    if cand.get("visually_supported") is not False:
        return "teacher_not_visually_false"
    conf = cand.get("confidence")
    if conf == "high":
        return None
    if conf == "medium":
        return None
    return "teacher_not_high"
