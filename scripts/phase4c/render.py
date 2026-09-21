#!/usr/bin/env python3
"""Deterministic statement → F2/F3 prompt transforms for Phase 4C.

Training statements are renderer output: ``A/An {s} is {rel-phrase} a/an {o}.``
No verb inflection: rearrange the existing tokens only.
"""

from __future__ import annotations

import re

STMT_RE = re.compile(r"^(An?)\s+(.+?)\s+is\s+(.+?)\.?\s*$", re.I)
ARTICLE_RE = re.compile(r"^(An?)\s+(.+?)\.?\s*$", re.I)

F1_VERIFY_PREFIX = "Does the image support the relation expressed in the following statement?"


def strip_caption(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def parse_renderer_statement(statement: str) -> dict | None:
    s = strip_caption(statement)
    m = STMT_RE.match(s)
    if not m:
        return None
    article = m.group(1).lower()
    subject = m.group(2).strip()
    rest = m.group(3).rstrip(" .").strip()
    if not subject or not rest:
        return None
    return {"article": article, "subject": subject, "rest": rest, "statement": s}


def statement_to_is_question(statement: str) -> tuple[str, str]:
    """Return (question, mode). mode is canonical | fallback_article | fallback_prefix."""
    parsed = parse_renderer_statement(statement)
    if parsed:
        q = f"Is {parsed['article']} {parsed['subject']} {parsed['rest']}?"
        return q, "canonical"
    s = strip_caption(statement)
    m = ARTICLE_RE.match(s)
    if m:
        art = m.group(1).lower()
        rest = m.group(2).rstrip(" .").strip()
        return f"Is {art} {rest}?", "fallback_article"
    body = s.rstrip(" .").strip()
    if body:
        body = body[0].lower() + body[1:]
    return f"Is {body}?", "fallback_prefix"


def heldout_to_is_question(statement: str) -> tuple[str, str]:
    """Wrap a frozen held-out caption as a direct question without dropping tokens.

    Held-out captions are original RelSim text, not renderer output. Do not use
    the training canonical rewrite (it would steal a later copula such as
    ``as it is being sketched``).
    """
    s = strip_caption(statement)
    m = ARTICLE_RE.match(s)
    if m:
        art = m.group(1).lower()
        rest = m.group(2).rstrip(" .").strip()
        return f"Is {art} {rest}?", "fallback_article"
    body = s.rstrip(" .").strip()
    if body:
        body = body[0].lower() + body[1:]
    return f"Is {body}?", "fallback_prefix"


def f2_prompt(statement: str, allow_fallback: bool = False) -> tuple[str, str]:
    q, mode = statement_to_is_question(statement)
    if mode != "canonical" and not allow_fallback:
        raise ValueError(f"non-canonical statement: {statement!r}")
    return f"{q}\nAnswer Yes or No.", mode


def f3_prompt(statement: str, allow_fallback: bool = False) -> tuple[str, str]:
    q, mode = statement_to_is_question(statement)
    if mode != "canonical" and not allow_fallback:
        raise ValueError(f"non-canonical statement: {statement!r}")
    if not q.endswith("?"):
        raise ValueError(f"question missing '?': {q!r}")
    body = q[:-1] + " in the image? Please answer with one word."
    return body, mode


def human_prompt_text(rec: dict) -> str:
    for turn in rec.get("conversations") or []:
        if turn.get("from") in {"human", "user"}:
            return re.sub(r"<image>\s*", "", turn.get("value") or "", flags=re.I).strip()
    return (rec.get("text") or "").strip()


def qc_is_question(question: str, rec: dict) -> list[str]:
    """Return a list of QC failure reasons (empty = pass)."""
    q = (question or "").strip()
    first = q.split("\n", 1)[0].strip()
    is_q = first
    suffix = " Please answer with one word."
    if is_q.endswith(suffix) or is_q.endswith(suffix.rstrip(".")):
        is_q = is_q[: is_q.lower().rfind("please answer with one word")].strip()
    reasons = []
    if not re.match(r"^Is an?\s", is_q):
        reasons.append("not_is_a_an")
    if not is_q.endswith("?"):
        reasons.append("no_question_mark")
    blob = is_q.lower()
    for key, field in (("subject", "subject"), ("object", "object")):
        span = (rec.get(field) or "").strip()
        if span and span.lower() not in blob:
            reasons.append(f"missing_{key}")
    rel = (rec.get("neg_relation") or rec.get("gt_relation") or "").strip()
    if rec.get("subset") in {"clean_negative", "random_negative", "hard_negative"}:
        rel = (rec.get("neg_relation") or rel).strip()
    elif rec.get("subset") == "positive":
        rel = (rec.get("gt_relation") or rel).strip()
    if rel and rel.lower() not in blob:
        reasons.append("missing_relation")
    return reasons
