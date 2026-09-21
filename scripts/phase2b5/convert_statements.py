#!/usr/bin/env python3
"""Deterministic question → statement conversion for Phase 2B.5.

MMRel-Adv uses regex over the two official templates.
R-Bench uses spaCy only to locate nsubj; the statement is rebuilt from
original character spans (no paraphrasing, no image-conditioned generation).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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

IRREG_3SG = {
    "have": "has",
    "do": "does",
    "be": "is",
    "go": "goes",
    "wear": "wears",
}


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strip_question(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s*Please answer with one word\.?\s*$", "", text, flags=re.I)
    return text.rstrip("?").strip()


def capitalize_sent(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def ensure_period(text: str) -> str:
    text = text.rstrip(" .") + "."
    return text


def inflect_3sg(verb: str) -> str:
    raw = verb
    v = verb.lower()
    if v in IRREG_3SG:
        out = IRREG_3SG[v]
    elif v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        out = v[:-1] + "ies"
    elif v.endswith(("s", "sh", "ch", "x", "z", "o")):
        out = v + "es"
    else:
        out = v + "s"
    if raw[:1].isupper():
        return out[:1].upper() + out[1:]
    return out


def gold_label(label) -> str:
    s = str(label or "").strip().lower()
    if s.startswith("no") or "no" in s:
        return "No"
    return "Yes"


# --- MMRel ---
MMREL_TO = re.compile(
    r"^Is there (an?|the) (.+?) (above|on|left|right|under|in) to (an?|the) (.+) in the image$",
    re.I,
)
MMREL_PREP = re.compile(
    r"^Is there (an?|the) (.+?) (in|on|above|under|left|right|at|near|over|by) (an?|the) (.+) in the image$",
    re.I,
)
MMREL_DOES = re.compile(
    r"^Does (an?|the) (.+) ([A-Za-z-]+) (an?|the) (.+) in the image$",
    re.I,
)


def convert_mmrel(question: str) -> tuple[str | None, str]:
    q = strip_question(question)
    m = MMREL_TO.match(q)
    if m:
        det_s, subj, rel, det_o, obj = m.groups()
        sent = f"{det_s} {subj} is {rel} to {det_o} {obj}."
        return capitalize_sent(sent), "mmrel_spatial_to"
    m = MMREL_PREP.match(q)
    if m:
        det_s, subj, prep, det_o, obj = m.groups()
        sent = f"{det_s} {subj} is {prep} {det_o} {obj}."
        return capitalize_sent(sent), "mmrel_spatial_prep"
    m = MMREL_DOES.match(q)
    if m:
        det_s, subj, verb, det_o, obj = m.groups()
        sent = f"{det_s} {subj} {inflect_3sg(verb)} {det_o} {obj}."
        return capitalize_sent(sent), "mmrel_action"
    return None, "mmrel_fail"


# --- R-Bench via spaCy nsubj + original char spans ---
def _doc_slice(doc, start_i: int, end_i: int) -> str:
    if end_i <= start_i:
        return ""
    start = doc[start_i].idx
    end = doc[end_i - 1].idx + len(doc[end_i - 1])
    return doc.text[start:end]


def _find_nsubj(doc):
    roots = [t for t in doc if t.dep_ == "ROOT"]
    root = roots[0] if roots else doc[0]
    for t in doc:
        if t.dep_ in {"nsubj", "nsubjpass"} and t.head == root:
            return t
    for t in doc:
        if t.dep_ in {"nsubj", "nsubjpass"}:
            return t
    return None


def _is_prep(tok) -> bool:
    return tok.text.lower() in {
        "in",
        "on",
        "at",
        "of",
        "with",
        "from",
        "by",
        "near",
        "over",
        "under",
        "above",
        "below",
        "against",
        "along",
        "around",
        "between",
        "beside",
        "inside",
        "outside",
        "behind",
        "across",
        "onto",
        "into",
        "next",
    }


def _looks_pred_verb(doc, i: int) -> bool:
    if i >= len(doc):
        return False
    tok = doc[i]
    if tok.pos_ in {"VERB", "AUX"}:
        return True
    if tok.tag_ in {"VBG", "VBN", "VBZ", "VBD", "JJ", "JJR", "JJS"}:
        return True
    return False


def _looks_predicate(doc, i: int) -> bool:
    if i >= len(doc):
        return False
    tok = doc[i]
    if _looks_pred_verb(doc, i):
        return True
    if tok.text.lower() in {"a", "an"}:
        return True
    if tok.text.lower() in {"the"} and i + 1 < len(doc) and doc[i + 1].text.lower() in {"same", "only", "first", "second"}:
        return True
    return False


_PARTICIPLES = {
    "made",
    "located",
    "filled",
    "cut",
    "numbered",
    "positioned",
    "surrounded",
    "aimed",
    "cooked",
    "covered",
    "attached",
    "connected",
    "painted",
    "dressed",
    "seated",
    "placed",
    "used",
    "shown",
    "hidden",
    "parked",
    "closed",
    "opened",
}
_LOCS = (
    "in front of",
    "next to",
    "on top of",
    "out of",
    "inside",
    "outside",
    "behind",
    "under",
    "above",
    "below",
    "against",
    "along",
    "around",
    "between",
    "beside",
    "near",
    "over",
    "onto",
    "into",
    "across",
    "atop",
    "on",
    "in",
    "at",
    "by",
)
_DOES_VERBS = {
    "wear",
    "have",
    "play",
    "hold",
    "contain",
    "go",
    "pull",
    "stand",
    "smile",
    "drink",
    "jump",
    "guide",
    "shop",
    "drive",
    "ride",
}


_FALSE_ING = {
    "ceiling",
    "parking",
    "something",
    "anything",
    "everything",
    "nothing",
    "morning",
    "evening",
    "king",
    "ring",
    "wing",
    "spring",
    "string",
    "pudding",
    "sibling",
    "dumpling",
    "sterling",
    "building",
    "according",
}


def _gerund_split(rest: str) -> int | None:
    tokens = rest.split()
    for i, tok in enumerate(tokens):
        if i == 0:
            continue
        stem = re.sub(r"[^A-Za-z]", "", tok).lower()
        if len(stem) > 4 and stem.endswith("ing") and stem not in _FALSE_ING:
            return i
    return None


def _split_trailing_image(rest: str) -> tuple[str, str]:
    m = re.search(r"\s+in the image$", rest, flags=re.I)
    if m:
        return rest[: m.start()].strip(), " in the image"
    return rest, ""


def _pack_invert(subj: str, aux: str, pred: str, trail: str) -> str:
    sent = " ".join(x for x in (subj.strip(), aux, pred.strip()) if x)
    if trail and not sent.lower().endswith("in the image"):
        sent = sent.rstrip(".") + trail
    return ensure_period(capitalize_sent(sent))


def fallback_does(q: str) -> tuple[str | None, str]:
    m = re.match(r"^(Does|Do)\s+(.*)$", q, flags=re.I)
    if not m:
        return None, "does_fail"
    aux, rest = m.group(1).lower(), m.group(2).strip()
    tokens = rest.split()
    verb_i = None
    for i, tok in enumerate(tokens):
        stem = re.sub(r"[^A-Za-z]", "", tok).lower()
        if stem in _DOES_VERBS or (i > 0 and stem.isalpha() and stem not in {
            "the", "a", "an", "in", "on", "of", "image", "and", "or", "with", "from", "to",
            "his", "her", "their", "any", "some", "blue", "red", "black", "white", "yellow",
            "green", "pink", "gray", "grey", "brown",
        }):
            # first content verb after at least one token
            if i == 0:
                continue
            verb_i = i
            break
    if verb_i is None:
        return None, "does_fail"
    left = " ".join(tokens[:verb_i])
    verb = inflect_3sg(tokens[verb_i]) if aux == "does" else tokens[verb_i]
    right = " ".join(tokens[verb_i + 1 :])
    sent = " ".join(x for x in (left, verb, right) if x)
    return ensure_period(capitalize_sent(sent)), "rbench_does_fallback"


def fallback_is_are(q: str) -> tuple[str | None, str]:
    m = re.match(r"^(Is|Are|Was|Were)\s+(.*)$", q, flags=re.I)
    if not m:
        return None, "is_are_fail"
    aux, rest = m.group(1).lower(), m.group(2).strip()
    rest, trail = _split_trailing_image(rest)
    gi = _gerund_split(rest)
    if gi is not None:
        toks = rest.split()
        return _pack_invert(" ".join(toks[:gi]), aux, " ".join(toks[gi:]), trail), "rbench_fallback_ing"
    m = re.search(
        r"\b(" + "|".join(sorted(_PARTICIPLES, key=len, reverse=True)) + r")\b",
        rest,
        flags=re.I,
    )
    if m and m.start() > 0:
        return _pack_invert(rest[: m.start()], aux, rest[m.start() :], trail), "rbench_fallback_ed"
    m = re.search(r"\b(a|an|the same)\b", rest, flags=re.I)
    if m and m.start() > 0:
        return _pack_invert(rest[: m.start()], aux, rest[m.start() :], trail), "rbench_fallback_comp"
    # locative PP; if leftover after PP looks like ADJ/verb, keep PP in subject
    loc_pat = re.compile(
        r"\b(" + "|".join(re.escape(x) for x in sorted(_LOCS, key=len, reverse=True)) + r")\b",
        flags=re.I,
    )
    m = loc_pat.search(rest)
    if m and m.start() > 0:
        after = rest[m.start() :].strip()
        leftover = rest[m.end() :].strip()
        leftover_toks = leftover.split()
        if leftover_toks and leftover_toks[-1].lower() in {
            "open", "closed", "black", "white", "red", "full", "empty", "visible", "on", "off",
            "asleep", "awake", "together",
        } and not leftover_toks[0].lower() in {"the", "a", "an"}:
            # e.g. "inside the house open"
            # find last adjective
            pred = leftover_toks[-1]
            mid = rest[: m.start()] + rest[m.start() : rest.rfind(pred)].rstrip()
            return _pack_invert(mid, aux, pred, trail), "rbench_fallback_loc_adj"
        return _pack_invert(rest[: m.start()], aux, rest[m.start() :], trail), "rbench_fallback_loc"
    toks = rest.split()
    if len(toks) >= 2:
        return _pack_invert(" ".join(toks[:-1]), aux, toks[-1], trail), "rbench_fallback_last"
    return None, "is_are_fail"


def _bad_aux_final(sent: str) -> bool:
    return bool(re.search(r"\b(is|are|was|were)\.$", sent or "", flags=re.I))


def convert_rbench(question: str, nlp) -> tuple[str | None, str]:
    q = strip_question(question)
    doc = nlp(q)
    if not len(doc):
        return None, "empty"
    first = doc[0]
    low = first.text.lower()

    if low in {"is", "are"} and len(doc) > 1 and doc[1].text.lower() == "there":
        rest = _doc_slice(doc, 2, len(doc)).strip()
        aux = "is" if low == "is" else "are"
        sent = capitalize_sent(f"There {aux} {rest}")
        return ensure_period(sent), "rbench_there"

    if low in {"does", "do"}:
        root = [t for t in doc if t.dep_ == "ROOT"]
        root = root[0] if root else None
        if root is None or root.i == 0:
            return fallback_does(q)
        left = _doc_slice(doc, 1, root.i).strip()
        verb = inflect_3sg(root.text) if low == "does" else root.text
        right = _doc_slice(doc, root.i + 1, len(doc)).strip()
        sent = " ".join(x for x in (left, verb, right) if x)
        return ensure_period(capitalize_sent(sent)), "rbench_does"

    if low not in {"is", "are", "was", "were"}:
        return fallback_is_are(q)

    subj = _find_nsubj(doc)
    if subj is None or subj.i == 0:
        return fallback_is_are(q)
    # subject starts after aux and includes left dependents through nsubj head
    subj_start = 1
    i = subj.i + 1
    # NP-internal "of ..."
    while i < len(doc) and doc[i].text.lower() == "of":
        i += 1
        while i < len(doc) and not _is_prep(doc[i]) and not _looks_predicate(doc, i):
            i += 1
    # "in the image"
    if i + 2 < len(doc) and [doc[j].text.lower() for j in range(i, i + 3)] == ["in", "the", "image"]:
        i += 3
    # further PPs only if a predicate remains after them
    while i < len(doc) and _is_prep(doc[i]):
        j = i + 1
        if doc[i].text.lower() == "next" and j < len(doc) and doc[j].text.lower() == "to":
            j += 1
        if doc[i].text.lower() == "in" and j < len(doc) and doc[j].text.lower() == "front":
            j += 1
            if j < len(doc) and doc[j].text.lower() == "of":
                j += 1
        while j < len(doc) and not _is_prep(doc[j]) and not _looks_pred_verb(doc, j):
            j += 1
        if _looks_pred_verb(doc, j) or (
            j < len(doc)
            and j == len(doc) - 1
            and doc[j].text.lower() in {"open", "closed", "black", "white", "full", "empty", "visible", "on", "off"}
        ):
            i = j
        else:
            break
    subj_txt = _doc_slice(doc, subj_start, i).strip()
    pred_txt = _doc_slice(doc, i, len(doc)).strip()
    if not subj_txt:
        return None, "empty_subj"
    aux = first.text.lower()
    if pred_txt:
        sent = f"{subj_txt} {aux} {pred_txt}"
    else:
        sent = f"{subj_txt} {aux}"
    sent = ensure_period(capitalize_sent(sent))
    if _bad_aux_final(sent) or not pred_txt or re.match(r"^(The|A|An) is\b", sent):
        fb, tag = fallback_is_are(q)
        if fb:
            return fb, tag
    return sent, "rbench_is_are"


def make_row(src: dict, statement: str, tag: str, bench: str) -> dict:
    gold = gold_label(src.get("label"))
    return {
        "question_id": src["question_id"],
        "image": src["image"],
        "official_question": src["text"],
        "matched_statement": statement,
        "conversion_tag": tag,
        "gold": gold,
        "label": gold.lower(),
        "text": VERIFY_PROMPT.format(statement=statement),
        "benchmark": bench,
    }


def convert_bench(questions: list[dict], bench: str, nlp=None) -> tuple[list[dict], dict]:
    rows, failed = [], []
    tags = Counter()
    statements = []
    for q in questions:
        if bench == "mmrel_adv":
            sent, tag = convert_mmrel(q["text"])
        else:
            sent, tag = convert_rbench(q["text"], nlp)
        tags[tag] += 1
        if not sent:
            failed.append({"question_id": q.get("question_id"), "official_question": q["text"], "tag": tag})
            continue
        rows.append(make_row(q, sent, tag, bench))
        statements.append(sent)
    dup = len(statements) - len(set(statements))
    qc = {
        "total": len(questions),
        "converted_successfully": len(rows),
        "failed_conversion": len(failed),
        "failed_examples": failed[:20],
        "tags": dict(tags),
        "duplicate_statement": dup,
        "gold_unchanged": True,
        "image_unchanged": True,
    }
    return rows, qc


def sample_review(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    picked = rows if len(rows) <= n else rng.sample(rows, n)
    picked = sorted(picked, key=lambda r: str(r["question_id"]))
    return [
        {
            "image": r["image"],
            "official_question": r["official_question"],
            "matched_statement": r["matched_statement"],
            "gold": r["gold"],
            "question_id": r["question_id"],
            "conversion_tag": r["conversion_tag"],
        }
        for r in picked
    ]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=str(ROOT / "data/phase2b5"))
    p.add_argument("--eval_dir", default=str(ROOT / "eval_results/qwen/phase2b5"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--review_n", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import spacy

    nlp = spacy.load("en_core_web_lg")
    out = Path(args.out_dir)
    ev = Path(args.eval_dir)
    out.mkdir(parents=True, exist_ok=True)
    (ev / "reviews").mkdir(parents=True, exist_ok=True)

    rbench_q = load_jsonl(ROOT / "eval_results/qwen/phase2b/fast_subsets/rbench_questions.jsonl")
    mmrel_q = load_jsonl(ROOT / "eval_results/qwen/phase2b/fast_subsets/mmrel_adv_questions.jsonl")
    rbench_rows, rbench_qc = convert_bench(rbench_q, "rbench", nlp)
    mmrel_rows, mmrel_qc = convert_bench(mmrel_q, "mmrel_adv", None)

    dump_jsonl(out / "rbench_fast_matched_prompt.jsonl", rbench_rows)
    dump_jsonl(out / "mmrel_adv_fast_matched_prompt.jsonl", mmrel_rows)
    # also copy into eval_results for the plan's named outputs
    dump_jsonl(ev / "rbench_fast_matched_prompt.jsonl", rbench_rows)
    dump_jsonl(ev / "mmrel_adv_fast_matched_prompt.jsonl", mmrel_rows)

    rbench_rev = sample_review(rbench_rows, args.review_n, args.seed)
    mmrel_rev = sample_review(mmrel_rows, args.review_n, args.seed)
    dump_jsonl(ev / "reviews/rbench_review_100.jsonl", rbench_rev)
    dump_jsonl(ev / "reviews/mmrel_review_100.jsonl", mmrel_rev)
    # TSV for easier manual inspection
    for name, rev in ("rbench", rbench_rev), ("mmrel", mmrel_rev):
        tsv = ev / "reviews" / f"{name}_review_100.tsv"
        with tsv.open("w", encoding="utf-8") as f:
            f.write("gold\tofficial_question\tmatched_statement\timage\tquestion_id\n")
            for r in rev:
                f.write(
                    f"{r['gold']}\t{r['official_question']}\t{r['matched_statement']}\t{r['image']}\t{r['question_id']}\n"
                )

    qc = {"rbench": rbench_qc, "mmrel_adv": mmrel_qc, "seed": args.seed}
    (out / "conversion_qc.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    (ev / "conversion_qc.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
