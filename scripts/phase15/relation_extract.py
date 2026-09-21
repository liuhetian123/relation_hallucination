"""Extract a short relation phrase from a RelSim anonymous caption.

GT relations are taken from the span *between the first two placeholders*
so we do not use the evaluation VLM and do not blindly slice by character index.
"""

from __future__ import annotations

import re

PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")

META_HEADS = {
    "showing",
    "featuring",
    "depicting",
    "illustrated",
    "highlighting",
    "representing",
    "portraying",
    "displaying",
    "showcasing",
    "including",
    "consisting",
    "using",
    "creating",
    "making",
    "looking",
    "seeing",
    "describing",
    "presenting",
    "combining",
    "blending",
    "merging",
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
    "interesting",
    "amazing",
    "unexpected",
    "striking",
    "setting",
    "organized",
    "detailed",
    "related",
    "colored",
    "patterned",
    "stacked",
    "themed",
    "surrounded",
}

COMPOUND_PREPS = [
    "in front of",
    "on top of",
    "next to",
    "out of",
    "because of",
    "on top",
    "in front",
]
SINGLE_PREPS = sorted(
    {
        "onto",
        "into",
        "upon",
        "atop",
        "from",
        "with",
        "beside",
        "between",
        "among",
        "amidst",
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
        "without",
        "toward",
        "towards",
        "near",
        "together",
        "apart",
        "off",
        "out",
        "on",
        "in",
        "at",
        "to",
        "by",
        "as",
        "for",
        "like",
        "via",
        "of",
    },
    key=len,
    reverse=True,
)

VERB_RE = re.compile(r"\b([A-Za-z][A-Za-z\-]*(?:ing|ed))\b")


def normalize_relation(text: str) -> str:
    text = (text or "").strip().strip("\"'").lower()
    text = re.sub(r"\s+", " ", text).strip(" .,;:!?")
    return text


def strip_caption(text: str) -> str:
    return (text or "").strip().strip("\"'").strip()


def consume_prep(text: str) -> str | None:
    t = text.lstrip()
    low = t.lower()
    for phrase in COMPOUND_PREPS:
        if low.startswith(phrase) and (len(t) == len(phrase) or not t[len(phrase)].isalpha()):
            return phrase
    for phrase in SINGLE_PREPS:
        if low.startswith(phrase) and (len(t) == len(phrase) or not t[len(phrase)].isalpha()):
            return phrase
    return None


def extract_from_span(span: str) -> str | None:
    span = (span or "").strip(" \t\n\"'.,;:()")
    for match in VERB_RE.finditer(span):
        head = match.group(1).lower()
        if head in META_HEADS or head in {"is", "are", "been", "being", "has", "have"}:
            continue
        prefix = span[: match.start()].strip().lower().strip(" ,;:")
        # "{X} in daring {Y}" — adjective after a preposition, not a predicate.
        if prefix in {p.lower() for p in SINGLE_PREPS + COMPOUND_PREPS}:
            continue
        prep = consume_prep(span[match.end() :])
        words = [head] + ([prep] if prep else [])
        rel = normalize_relation(" ".join(words))
        if len(rel) < 3:
            continue
        return rel
    return None


def extract_relation(caption: str, between_only: bool = False) -> tuple[str | None, str]:
    """Return (relation, source) where source is between / later / fail."""
    caption = strip_caption(caption)
    parts = PLACEHOLDER_RE.split(caption)
    n_ph = len(PLACEHOLDER_RE.findall(caption))
    if n_ph >= 2 and len(parts) >= 3:
        rel = extract_from_span(parts[1])
        if rel:
            return rel, "between"
    if between_only:
        return None, "fail"
    for span in parts[1:]:
        rel = extract_from_span(span)
        if rel:
            return rel, "later"
    return None, "fail"


def relations_too_similar(a: str, b: str) -> bool:
    a = normalize_relation(a)
    b = normalize_relation(b)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True
    wa, wb = a.split(), b.split()
    if wa[0] == wb[0]:
        return True
    inter = set(wa) & set(wb)
    union = set(wa) | set(wb)
    return bool(inter) and len(inter) / len(union) >= 0.5
