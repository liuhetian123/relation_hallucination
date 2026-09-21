"""Shared helpers for Phase 2A relation verification data."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/phase15"))
from relation_extract import (  # noqa: E402
    extract_relation,
    normalize_relation,
    relations_too_similar,
    strip_caption,
)

VERIFY_PROMPT = (
    "Does the image support the relation expressed in the following statement?\n"
    "\n"
    "Judge only whether the stated relation between the referenced entities is visually supported.\n"
    "Ignore minor wording or attribute details that are not relevant to the relation.\n"
    "\n"
    "Statement:\n"
    "{statement}\n"
    "\n"
    "Answer Yes or No only."
)

SYNONYM_GROUPS = [
    {"standing beside", "standing next to", "standing by", "standing near", "beside", "next to"},
    {"sitting on", "sitting upon", "seated on"},
    {"on top of", "on top", "atop"},
    {"in front of", "in front"},
    {"holding", "grasping", "clutching"},
    {"looking at", "looking towards", "gazing at", "watching"},
    {"interacting with", "engaging with"},
    {"placed on", "resting on", "lying on"},
    {"crafted with", "crafted from"},
    {"shaped like", "shaped into"},
    {"merged with", "combined with", "fused with"},
    {"balancing on", "balanced on", "balanced on top of"},
    {"submerged in", "immersed in", "sunk in"},
]

VISUAL_RELATIONS = [
    "holding",
    "wearing",
    "riding",
    "driving",
    "pushing",
    "pulling",
    "carrying",
    "touching",
    "hugging",
    "kissing",
    "biting",
    "feeding",
    "eating",
    "drinking",
    "chasing",
    "catching",
    "throwing",
    "catching",
    "sitting on",
    "standing on",
    "standing beside",
    "lying on",
    "hanging from",
    "leaning on",
    "sleeping on",
    "walking on",
    "jumping on",
    "jumping from",
    "climbing on",
    "looking at",
    "pointing at",
    "talking to",
    "playing with",
    "hiding behind",
    "hiding under",
    "in front of",
    "behind",
    "next to",
    "on top of",
    "under",
    "inside",
    "surrounded by",
    "wrapped around",
    "attached to",
    "tied to",
    "flying over",
    "swimming in",
    "floating on",
    "falling from",
    "coming out of",
    "going into",
    "covering",
    "blocking",
    "supporting",
    "crushing",
    "hugging",
]

META_REL_HEADS = {
    "based",
    "added",
    "aligned",
    "appeared",
    "anthropomorphized",
    "accentuated",
    "captured",
    "illustrated",
    "featuring",
    "showing",
    "depicting",
    "representing",
    "portraying",
    "displaying",
    "showcasing",
    "including",
    "consisting",
    "creating",
    "making",
    "describing",
    "presenting",
    "combining",
    "blending",
    "inspired",
    "designed",
    "arranged",
    "composed",
    "containing",
    "having",
    "being",
    "emphasizing",
    "symbolizing",
    "corresponding",
    "remaining",
    "existing",
    "related",
    "themed",
    "refined",
    "represented",
    "highlighted",
    "expressed",
    "associated",
    "connected",
    "linked",
    "derived",
    "inspired",
    "defined",
    "characterized",
    "known",
    "used",
    "seen",
    "shown",
    "found",
    "located",
}

SPATIAL_KEYS = {
    "beside",
    "behind",
    "under",
    "over",
    "near",
    "around",
    "against",
    "inside",
    "above",
    "below",
    "along",
    "across",
    "among",
    "next to",
    "in front of",
    "on top of",
    "standing on",
    "sitting on",
    "lying on",
    "hanging from",
    "standing beside",
    "resting on",
    "leaning on",
    "walking on",
    "hiding behind",
    "floating on",
    "swimming in",
}
CONTACT_KEYS = {
    "holding",
    "touching",
    "grabbing",
    "hugging",
    "kissing",
    "wearing",
    "carrying",
    "pulling",
    "pushing",
    "riding",
    "driving",
    "feeding",
    "eating",
    "drinking",
    "biting",
    "catching",
    "chasing",
    "throwing",
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


def verification_prompt(statement: str) -> str:
    return VERIFY_PROMPT.format(statement=strip_caption(statement))


def llava_verify_sample(sample_id: str, image: str, statement: str, answer: str, **extra) -> dict:
    rec = {
        "id": sample_id,
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\n{verification_prompt(statement)}"},
            {"from": "gpt", "value": answer},
        ],
        "statement": statement,
        "label": answer,
    }
    rec.update(extra)
    return rec


def replace_relation(caption: str, relation: str, new_relation: str) -> str | None:
    caption = strip_caption(caption)
    rel = strip_caption(relation)
    new = strip_caption(new_relation)
    if not caption or not rel or not new:
        return None
    low = caption.lower()
    needle = rel.lower()
    start = low.find(needle)
    if start < 0:
        return None
    if low.find(needle, start + 1) >= 0:
        return None
    out = caption[:start] + new + caption[start + len(rel) :]
    if normalize_relation(out) == normalize_relation(caption):
        return None
    return out


def are_synonyms(a: str, b: str) -> bool:
    a, b = normalize_relation(a), normalize_relation(b)
    if not a or not b:
        return False
    if a == b:
        return True
    for group in SYNONYM_GROUPS:
        if a in group and b in group:
            return True
    return False


def too_close_for_random(gt: str, cand: str) -> bool:
    return relations_too_similar(gt, cand) or are_synonyms(gt, cand)


def stem(token: str) -> str:
    t = normalize_relation(token)
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def is_meta_relation(rel: str) -> bool:
    head = normalize_relation(rel).split()[0]
    return head in META_REL_HEADS


def too_close_for_hard(gt: str, cand: str) -> bool:
    gt, cand = normalize_relation(gt), normalize_relation(cand)
    if not gt or not cand or gt == cand:
        return True
    if are_synonyms(gt, cand):
        return True
    if gt in cand or cand in gt:
        return True
    gw, cw = gt.split(), cand.split()
    if stem(gw[0]) == stem(cw[0]) and len(stem(gw[0])) >= 4:
        return True
    return False


def relation_family(rel: str) -> str:
    r = normalize_relation(rel)
    if r in SPATIAL_KEYS:
        return "spatial"
    if r in CONTACT_KEYS or r.split()[0] in CONTACT_KEYS:
        return "contact"
    return "other"


def hard_score(gt: str, cand: str) -> float:
    if too_close_for_hard(gt, cand) or is_meta_relation(cand):
        return -1.0
    gw, cw = normalize_relation(gt).split(), normalize_relation(cand).split()
    score = 0.0
    if cand in VISUAL_RELATIONS:
        score += 4.0
    if gw[0] == cw[0]:
        score += 5.0
    if relation_family(gt) == relation_family(cand) and relation_family(gt) != "other":
        score += 3.0
    jac = jaccard_tokens(gt, cand)
    if 0.15 <= jac <= 0.6:
        score += 1.0
    if abs(len(gt) - len(cand)) <= 8:
        score += 0.5
    return score


def jaccard_tokens(a: str, b: str) -> float:
    wa, wb = set(normalize_relation(a).split()), set(normalize_relation(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def parse_yes_no_uncertain(text: str) -> str:
    raw = (text or "").strip()
    first = re.split(r"[\s,.:;!?]+", raw, maxsplit=1)[0].lower()
    if first in {"yes", "y"}:
        return "yes"
    if first in {"no", "n"}:
        return "no"
    if first.startswith("uncertain") or first in {"unknown", "unsure"}:
        return "uncertain"
    low = raw.lower()
    if re.search(r"\bno\b", low) and not re.search(r"\byes\b", low):
        return "no"
    if re.search(r"\byes\b", low) and not re.search(r"\bno\b", low):
        return "yes"
    if "uncertain" in low:
        return "uncertain"
    return "parse_error"


def caption_of(sample: dict) -> str:
    for turn in reversed(sample.get("conversations", [])):
        if turn.get("from") in {"gpt", "assistant"}:
            return strip_caption(turn.get("value") or "")
    return strip_caption(sample.get("caption") or sample.get("anonymous_caption") or "")


def build_relation_vocab(captions: list[str], min_freq: int = 2) -> list[tuple[str, int]]:
    freq: Counter[str] = Counter()
    for cap in captions:
        rel, src = extract_relation(cap, between_only=True)
        if not rel:
            rel, src = extract_relation(cap, between_only=False)
        if rel:
            freq[normalize_relation(rel)] += 1
    items = [(r, c) for r, c in freq.items() if c >= min_freq and len(r) >= 3]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items
