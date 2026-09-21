#!/usr/bin/env python3
"""Phase 5A: prior-conflict decomposition on frozen vision answers + text-only priors."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/qwen_eval"))
sys.path.insert(0, str(ROOT / "scripts/phase2b"))
from score_fast import score_amber  # noqa: E402
from score_yesno import gold_yes, load_json_or_jsonl, pred_yes  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts/phase4b"))
from analyze import (  # noqa: E402
    bootstrap_delta,
    bootstrap_did,
    confusion_from_ok,
    lookup,
    mcnemar,
    qid_aliases,
    seen_at,
)

OUT = ROOT / "eval_results/qwen/phase5a"
FAST = ROOT / "eval_results/qwen/phase2b/fast_subsets"
N_BOOT = 10000
SEED = 42
INVALID_MAX = 0.10
YES_RATIO_MAX = 0.95
LAYER_GAP_A = 0.15
LAYER_GAP_C = 0.05
CROSS_MIN_N = 50

PUBLISHED_UNIQUE = {
    "base": {"rbench": 83.11, "mmrel_adv": 69.01, "heldout": 58.27, "amber_dr": 79.56},
    "clean": {"rbench": 82.95, "mmrel_adv": 70.83, "heldout": 67.18, "amber_dr": 80.16},
    "mf": {"rbench": 82.85, "mmrel_adv": 74.48, "heldout": 65.90, "amber_dr": 79.16},
    "s3000": {"rbench": 83.58, "mmrel_adv": 71.09, "heldout": 72.77},
}
VISION = {
    "base": {
        "rbench": ROOT / "eval_results/qwen/rbench/answers/qwen25vl3b_base_image-level.jsonl",
        "mmrel_adv": ROOT / "eval_results/qwen/mmrel/answers/qwen25vl3b_base_mmrel_adv.jsonl",
        "amber_dr": ROOT / "eval_results/qwen/amber/answers/qwen25vl3b_base_amber_dr.jsonl",
        "heldout": ROOT / "eval_results/qwen/phase2/answers/base_heldout.jsonl",
    },
    "clean": {
        "rbench": ROOT / "eval_results/qwen/phase4a/answers/clean_rbench_fast.jsonl",
        "mmrel_adv": ROOT / "eval_results/qwen/phase4a/answers/clean_mmrel_adv_fast.jsonl",
        "amber_dr": ROOT / "eval_results/qwen/phase4a/answers/clean_amber_dr_fast.jsonl",
        "heldout": ROOT / "eval_results/qwen/phase4a/answers/clean_heldout.jsonl",
    },
    "mf": {
        "rbench": ROOT / "eval_results/qwen/phase4c/answers/mf_rbench_fast.jsonl",
        "mmrel_adv": ROOT / "eval_results/qwen/phase4c/answers/mf_mmrel_adv_fast.jsonl",
        "amber_dr": ROOT / "eval_results/qwen/phase4c/answers/mf_amber_dr_fast.jsonl",
        "heldout": ROOT / "eval_results/qwen/phase4c/answers/mf_heldout.jsonl",
    },
    "s3000": {
        "rbench": ROOT / "eval_results/qwen/phase3a/answers/s3000_rbench_fast.jsonl",
        "mmrel_adv": ROOT / "eval_results/qwen/phase3a/answers/s3000_mmrel_adv_fast.jsonl",
        "amber_dr": ROOT / "eval_results/qwen/phase3a/answers/s3000_amber_dr_fast.jsonl",
        "heldout": ROOT / "eval_results/qwen/phase3a/answers/s3000_heldout.jsonl",
    },
}
TEXTONLY = {
    "base": {
        "mmrel_adv": OUT / "answers/base_mmrel_adv_fast_textonly.jsonl",
        "rbench": OUT / "answers/base_rbench_fast_textonly.jsonl",
        "amber_dr": OUT / "answers/base_amber_dr_fast_textonly.jsonl",
        "heldout": OUT / "answers/base_heldout_textonly.jsonl",
    },
    "clean": {"mmrel_adv": OUT / "answers/clean_mmrel_adv_fast_textonly.jsonl"},
    "mf": {"mmrel_adv": OUT / "answers/mf_mmrel_adv_fast_textonly.jsonl"},
}
TEXTONLY_FB = {
    "base": {
        "mmrel_adv": OUT / "answers/base_mmrel_adv_fast_textonly_fallback.jsonl",
        "rbench": OUT / "answers/base_rbench_fast_textonly_fallback.jsonl",
        "amber_dr": OUT / "answers/base_amber_dr_fast_textonly_fallback.jsonl",
        "heldout": OUT / "answers/base_heldout_textonly_fallback.jsonl",
    },
    "clean": {"mmrel_adv": OUT / "answers/clean_mmrel_adv_fast_textonly_fallback.jsonl"},
    "mf": {"mmrel_adv": OUT / "answers/mf_mmrel_adv_fast_textonly_fallback.jsonl"},
}
CONTRASTS = (
    ("base", "mf", "mf_vs_base"),
    ("base", "clean", "clean_vs_base"),
    ("base", "s3000", "s3000_vs_base"),
)
REFUSE_RE = re.compile(
    r"no image|without (an |the )?image|cannot see|can't see|unable to (see|answer)|"
    r"not (provided|available|given)|no visual|need(s)? (an |the )?image|"
    r"requires (an |the )?image|missing image|don'?t have (an |the )?image|"
    r"do not have (an |the )?image|i (do not|don't|cannot|can't) (see|view|access)|"
    r"can't (view|access)|cannot (view|access)|no picture|without (a |the )?picture|"
    r"there is no (image|picture)|image is missing",
    re.I,
)
YESNO_FIRST = re.compile(r"^[\s\"'`]*([Yy]es|[Nn]o|[Yy]|[Nn])(\b|[.,:;!?])")


def gold_bin(label) -> int:
    return 1 if gold_yes(label) else 0


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


def parse_prior_text(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return "invalid"
    if REFUSE_RE.search(raw):
        m = YESNO_FIRST.match(raw)
        if m:
            return "yes" if m.group(1).lower().startswith("y") else "no"
        return "invalid"
    parsed = parse_yes_no_uncertain(raw)
    if parsed in {"yes", "no"}:
        return parsed
    return "invalid"


def load_rows(path: Path) -> dict:
    amap = {}
    if not path.exists():
        return amap
    for row in load_json_or_jsonl(path):
        qid = row.get("question_id", row.get("id"))
        for key in qid_aliases(qid):
            amap[key] = row
    return amap


def answer_preds(path: Path) -> dict:
    amap = {}
    if not path.exists():
        return amap
    for row in load_json_or_jsonl(path):
        qid = row.get("question_id", row.get("id"))
        text = row.get("text") or row.get("response") or ""
        pred = 1 if pred_yes(text) else 0
        for key in qid_aliases(qid):
            amap[key] = pred
    return amap


def load_questions(bench: str) -> list[dict]:
    if bench == "rbench":
        return load_json_or_jsonl(FAST / "rbench_questions.jsonl")
    if bench == "mmrel_adv":
        return load_json_or_jsonl(FAST / "mmrel_adv_questions.jsonl")
    if bench == "amber_dr":
        return load_json_or_jsonl(FAST / "amber_dr_questions.jsonl")
    if bench == "heldout":
        return load_json_or_jsonl(ROOT / "data/phase2/heldout_verification.jsonl")
    raise KeyError(bench)


def pct(x, digits=2) -> str:
    if x is None:
        return ""
    return f"{100.0 * float(x):.{digits}f}"


def pp(x) -> str:
    if x is None:
        return ""
    return f"{100.0 * float(x):+.2f}"


def ci_str(ci) -> str:
    if not ci or ci[0] is None:
        return ""
    return f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]"


def ci_contains_zero(ci) -> bool:
    if not ci or ci[0] is None:
        return True
    return ci[0] <= 0.0 <= ci[1]


def unique_acc(questions: list[dict], preds: dict) -> dict:
    golds, ps = [], []
    miss = 0
    for q in questions:
        p = lookup(preds, q.get("question_id", q.get("id")))
        if p is None:
            miss += 1
            continue
        golds.append(gold_bin(q.get("label")))
        ps.append(p)
    rec = confusion_from_ok(ps, golds)
    rec["missing"] = miss
    rec["acc_pp"] = 100.0 * rec["Accuracy"]
    return rec


def bootstrap_gap(ok_hi: list[int], ok_lo: list[int], n_boot=N_BOOT, seed=SEED) -> dict:
    rng = random.Random(seed)
    n_hi, n_lo = len(ok_hi), len(ok_lo)
    if not n_hi or not n_lo:
        return {"gap": None, "ci95": [None, None], "n_hi": n_hi, "n_lo": n_lo}
    acc_hi = sum(ok_hi) / n_hi
    acc_lo = sum(ok_lo) / n_lo
    draws = []
    for _ in range(n_boot):
        ih = [rng.randrange(n_hi) for _ in range(n_hi)]
        il = [rng.randrange(n_lo) for _ in range(n_lo)]
        draws.append(sum(ok_hi[i] for i in ih) / n_hi - sum(ok_lo[i] for i in il) / n_lo)
    draws.sort()
    return {
        "gap": acc_hi - acc_lo,
        "acc_hi": acc_hi,
        "acc_lo": acc_lo,
        "ci95": [draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]],
        "n_hi": n_hi,
        "n_lo": n_lo,
        "n_boot": n_boot,
    }


def prior_from_row(row: dict | None, mode: str) -> tuple[str | None, str]:
    if not row:
        return None, "missing"
    if mode == "logprob":
        yes_lp = row.get("yes_logprob")
        no_lp = row.get("no_logprob")
        if yes_lp is None or no_lp is None:
            return None, "missing_logprob"
        return ("yes" if yes_lp >= no_lp else "no"), "logprob"
    parsed = parse_prior_text(row.get("text") or "")
    if parsed == "invalid":
        return "invalid", "text"
    return parsed, "text"


def bench_validity(questions: list[dict], rows: dict, mode: str = "text") -> dict:
    n = len(questions)
    invalid = missing = yes_n = no_n = 0
    parsed = []
    for q in questions:
        qid = q.get("question_id", q.get("id"))
        row = lookup(rows, qid)
        ans, src = prior_from_row(row, mode)
        if ans is None:
            missing += 1
            continue
        if ans == "invalid":
            invalid += 1
            continue
        parsed.append(ans)
        if ans == "yes":
            yes_n += 1
        else:
            no_n += 1
    valid_n = yes_n + no_n
    yes_ratio = yes_n / valid_n if valid_n else 0.0
    no_ratio = no_n / valid_n if valid_n else 0.0
    invalid_rate = (invalid + missing) / n if n else 1.0
    degenerate = valid_n > 0 and max(yes_ratio, no_ratio) > YES_RATIO_MAX
    return {
        "n": n,
        "valid": valid_n,
        "invalid": invalid,
        "missing": missing,
        "invalid_rate": invalid_rate,
        "yes": yes_n,
        "no": no_n,
        "yes_ratio": yes_ratio,
        "no_ratio": no_ratio,
        "degenerate": degenerate,
        "mode": mode,
        "need_fallback": invalid_rate > INVALID_MAX,
        "need_logprob": degenerate,
    }


def choose_prior_mode(questions: list[dict], main_rows: dict, fb_rows: dict) -> dict:
    main = bench_validity(questions, main_rows, "text")
    rec = {
        "main": main,
        "fallback": None,
        "logprob": None,
        "logprob_main": bench_validity(questions, main_rows, "logprob"),
        "used": "main",
        "rows": "main",
        "measurement_failed": False,
    }
    chosen = main
    if main["need_fallback"] and fb_rows:
        fb = bench_validity(questions, fb_rows, "text")
        rec["fallback"] = fb
        rec["used"] = "fallback"
        rec["rows"] = "fallback"
        chosen = fb
    if chosen["need_logprob"]:
        # Logprob of the original question is the language prior; fallback instruction
        # ("typically most plausible") rewrites the prior and can collapse to all-No.
        lp = rec["logprob_main"]
        rec["logprob"] = lp
        rec["used"] = "logprob"
        rec["rows"] = "main"
        chosen = lp
    rec["measurement_failed"] = bool(chosen.get("degenerate"))
    if rec["used"] == "logprob" and max(chosen.get("yes_ratio") or 0.0, chosen.get("no_ratio") or 0.0) >= 0.90:
        rec["measurement_failed"] = True
    if rec["used"] in {"main", "fallback"} and chosen.get("need_fallback"):
        rec["measurement_failed"] = True
    return rec


def layer_of(prior: str | None, gold: int) -> str:
    if prior not in {"yes", "no"}:
        return "invalid"
    return "consistent" if int(prior == "yes") == gold else "conflict"


def item_records(bench: str, questions: list[dict], prior_rows: dict, mode: str, vision: dict[str, dict]) -> list[dict]:
    items = []
    for q in questions:
        qid = q.get("question_id", q.get("id"))
        gold = gold_bin(q.get("label"))
        row = lookup(prior_rows, qid)
        prior, src = prior_from_row(row, mode)
        layer = layer_of(prior, gold)
        rec = {
            "benchmark": bench,
            "question_id": qid,
            "gold": gold,
            "gold_label": "yes" if gold else "no",
            "prior_answer": prior if prior is not None else "missing",
            "prior_source": src,
            "layer": layer,
            "text": (row or {}).get("text") or "",
            "yes_logprob": (row or {}).get("yes_logprob"),
            "no_logprob": (row or {}).get("no_logprob"),
            "logprob_margin": (row or {}).get("logprob_margin"),
        }
        for model, preds in vision.items():
            p = lookup(preds, qid)
            rec[f"vision_{model}"] = p
            rec[f"vision_{model}_ok"] = None if p is None else int(p == gold)
        items.append(rec)
    return items


def subset(items: list[dict], layer: str | None = None, extra=None) -> list[dict]:
    out = items
    if layer:
        out = [x for x in out if x["layer"] == layer]
    if extra:
        out = [x for x in out if extra(x)]
    return out


def layer_summary(items: list[dict]) -> dict:
    n = len(items)
    gold_yes_n = sum(1 for x in items if x["gold"] == 1)
    prior_yes_n = sum(1 for x in items if x["prior_answer"] == "yes")
    return {
        "n": n,
        "gold_yes_ratio": gold_yes_n / n if n else 0.0,
        "prior_yes_ratio": prior_yes_n / n if n else 0.0,
    }


def vision_layer_stats(items: list[dict], model: str) -> dict:
    preds, golds, ok = [], [], []
    miss = 0
    for x in items:
        p = x.get(f"vision_{model}")
        if p is None:
            miss += 1
            continue
        preds.append(p)
        golds.append(x["gold"])
        ok.append(int(p == x["gold"]))
    rec = confusion_from_ok(preds, golds)
    rec["missing"] = miss
    rec["ok"] = ok
    return rec


def lock_rate(items: list[dict], model: str) -> dict:
    n = locked = miss = skip = 0
    for x in items:
        if x["layer"] == "invalid":
            skip += 1
            continue
        p = x.get(f"vision_{model}")
        prior = x["prior_answer"]
        if p is None or prior not in {"yes", "no"}:
            miss += 1
            continue
        n += 1
        if p == int(prior == "yes"):
            locked += 1
    return {"n": n, "locked": locked, "rate": locked / n if n else 0.0, "missing": miss, "invalid_skipped": skip}


def contrast_on_layer(items: list[dict], a: str, b: str) -> dict:
    a_ok, b_ok, golds, pa, pb = [], [], [], [], []
    miss = 0
    for x in items:
        va, vb = x.get(f"vision_{a}"), x.get(f"vision_{b}")
        if va is None or vb is None:
            miss += 1
            continue
        a_ok.append(int(va == x["gold"]))
        b_ok.append(int(vb == x["gold"]))
        golds.append(x["gold"])
        pa.append(va)
        pb.append(vb)
    return {
        "n": len(a_ok),
        "missing": miss,
        "low_power": len(a_ok) < 100,
        a: confusion_from_ok(pa, golds),
        b: confusion_from_ok(pb, golds),
        "delta": bootstrap_delta(a_ok, b_ok, n_boot=N_BOOT, seed=SEED),
        "mcnemar": mcnemar(a_ok, b_ok),
        "a_ok": a_ok,
        "b_ok": b_ok,
    }


def prior_agreement(items_a: dict, items_b: dict) -> dict:
    n = agree = miss = inv = 0
    for qid, xa in items_a.items():
        xb = items_b.get(qid)
        if xb is None:
            miss += 1
            continue
        if xa["prior_answer"] not in {"yes", "no"} or xb["prior_answer"] not in {"yes", "no"}:
            inv += 1
            continue
        n += 1
        if xa["prior_answer"] == xb["prior_answer"]:
            agree += 1
    return {"n": n, "agree": agree, "rate": agree / n if n else 0.0, "missing": miss, "invalid": inv}


def relation_only_ids(extract: list[dict]) -> set:
    ids = set()
    for row in extract:
        if row.get("benchmark") != "rbench":
            continue
        if row.get("parse_reason") in {"verb", "prep"}:
            for key in qid_aliases(row["question_id"]):
                ids.add(key)
    return ids


def extract_index(extract: list[dict], bench: str) -> dict:
    idx = {}
    for row in extract:
        if row.get("benchmark") != bench:
            continue
        rec = {
            "match_old": row.get("match_old"),
            "match_clean": row.get("match_clean"),
            "parse_reason": row.get("parse_reason"),
            "relation": row.get("relation"),
        }
        for key in qid_aliases(row["question_id"]):
            idx[key] = rec
    return idx


def match_key_for(contrast: str) -> str:
    if contrast == "s3000_vs_base":
        return "match_old"
    return "match_clean"


def write_md(blob: dict) -> str:
    md = []
    md.append("# Phase 5A 结果：语言先验冲突分解")
    md.append("")
    md.append("本阶段不训练。用 base 的 text-only 回答把每道题切成 prior-consistent / prior-conflict，再在冻结看图预测上分层。")
    md.append("")
    md.append("> 必须看图才能答对的先验冲突子集，是否既是 base 幻觉的集中地，也是 mf 增益的集中地？")
    md.append("")
    md.append(f"**判定：Case {blob['case']}。** {blob['case_reason']}")
    md.append("")
    md.append("## 有效性检查")
    md.append("")
    md.append("| Bench | source | n | invalid% | Yes | No | degenerate | mode | failed |")
    md.append("|---|---|---:|---:|---:|---:|---|---|---|")
    for bench, rec in blob["validity"].items():
        used = rec["used"]
        src = rec.get(used) or rec.get("main") or {}
        md.append(
            f"| {bench} | {rec.get('rows')} | {src.get('n','')} | {pct(src.get('invalid_rate'))} | "
            f"{pct(src.get('yes_ratio'))} | {pct(src.get('no_ratio'))} | {src.get('degenerate')} | {used} | "
            f"{rec.get('measurement_failed')} |"
        )
        if rec.get("main") and used != "main":
            m = rec["main"]
            md.append(
                f"| {bench} (main raw) | main | {m.get('n','')} | {pct(m.get('invalid_rate'))} | "
                f"{pct(m.get('yes_ratio'))} | {pct(m.get('no_ratio'))} | {m.get('degenerate')} | text |  |"
            )
        if rec.get("fallback") and used != "fallback":
            fb = rec["fallback"]
            md.append(
                f"| {bench} (fallback raw) | fallback | {fb.get('n','')} | {pct(fb.get('invalid_rate'))} | "
                f"{pct(fb.get('yes_ratio'))} | {pct(fb.get('no_ratio'))} | {fb.get('degenerate')} | text |  |"
            )
        lm = rec.get("logprob_main")
        if lm and used != "logprob":
            md.append(
                f"| {bench} (logprob orig) | main | {lm.get('n','')} | {pct(lm.get('invalid_rate'))} | "
                f"{pct(lm.get('yes_ratio'))} | {pct(lm.get('no_ratio'))} | {lm.get('degenerate')} | logprob |  |"
            )
    md.append("")
    md.append("无效/拒答率阈值 10%；Yes/No 单边阈值 95%。分层用上表 `mode` 列。`failed=True` 的题集先验测量不可用，层结果只作附录。")
    md.append("")
    md.append("## 对账（冻结看图 unique Acc，必须与已发布数字一致 ±0.05 pp）")
    md.append("")
    md.append("| Model | Bench | 已发布 | 复算 | match |")
    md.append("|---|---|---:|---:|---|")
    for model, benches in blob["reconcile"].items():
        for bench, rec in benches.items():
            md.append(
                f"| {model} | {bench} | {rec['published']:.2f} | {rec['recomputed']:.2f} | {rec['match']} |"
            )
    md.append("")
    md.append("## 层规模（由 base text-only 定义）")
    md.append("")
    md.append("| Bench | slice | n | gold Yes | prior Yes |")
    md.append("|---|---|---:|---:|---:|")
    for bench, rec in blob["layers"].items():
        for name in ("all", "consistent", "conflict", "invalid"):
            s = rec.get(name) or {}
            md.append(
                f"| {bench} | {name} | {s.get('n','')} | {pct(s.get('gold_yes_ratio'))} | {pct(s.get('prior_yes_ratio'))} |"
            )
    md.append("")
    md.append("R-Bench 主分析只用 relation-only（verb+prep）；`rbench_full` 仅附录。")
    md.append("")
    md.append("## 诊断一：Base 看图 Acc 分层 + 先验锁定率")
    md.append("")
    md.append("| Bench | consistent Acc | conflict Acc | 层差 | 95% CI | lock rate | n_cons | n_conf |")
    md.append("|---|---:|---:|---:|---|---:|---:|---:|")
    for bench, rec in blob["diag1"].items():
        gap = rec.get("gap") or {}
        lock = rec.get("lock") or {}
        md.append(
            f"| {bench} | {pct(rec.get('consistent_acc'))} | {pct(rec.get('conflict_acc'))} | "
            f"{pp(gap.get('gap'))} pp | {ci_str(gap.get('ci95'))} | {pct(lock.get('rate'))} | "
            f"{rec.get('n_consistent','')} | {rec.get('n_conflict','')} |"
        )
    md.append("")
    md.append("层差 = consistent Acc − conflict Acc。锁定率 = 看图预测与 text-only 预测相同的比例（valid 题）。R-Bench 若 `failed=True`，该行只作附录，不进入 Case 判定。")
    md.append("")
    md.append("## 诊断二：微调增益分层（主假设：mf 的 MMRel +5.47 pp 集中在 conflict）")
    md.append("")
    for bench, contrasts in blob["diag2"].items():
        md.append(f"### {bench}")
        md.append("")
        md.append("| Contrast | layer | ΔAcc | 95% CI | McNemar p | Yes_a | Yes_b | FP/FN_a | FP/FN_b | n |")
        md.append("|---|---|---:|---|---:|---:|---:|---|---|---:|")
        for cname, crec in contrasts.items():
            did = crec.get("did") or {}
            for layer in ("conflict", "consistent"):
                rec = crec.get(layer) or {}
                if not rec:
                    continue
                a = rec.get(crec["a"]) or {}
                b = rec.get(crec["b"]) or {}
                d = rec.get("delta") or {}
                md.append(
                    f"| {cname} | {layer} | {pp(d.get('delta'))} pp | {ci_str(d.get('ci95'))} | "
                    f"{(rec.get('mcnemar') or {}).get('p_value', 0):.4g} | {pct(a.get('Yes_ratio'))} | "
                    f"{pct(b.get('Yes_ratio'))} | {a.get('FP','')}/{a.get('FN','')} | "
                    f"{b.get('FP','')}/{b.get('FN','')} | {rec.get('n','')} |"
                )
            md.append(
                f"| {cname} | DiD=Δconf−Δcons | {pp(did.get('did'))} pp | {ci_str(did.get('ci95'))} |  |  |  |  |  |  |"
            )
        md.append("")
    md.append("conflict 层 gold 与先验反向，「无脑翻转先验」也能涨分；须同时看 consistent 层是否对称下降，以及 Yes ratio / FP/FN。")
    md.append("")
    md.append("## 第二轮：原题 Yes/No logprob margin（不新做推理）")
    md.append("")
    md.append("生成式先验在 MMRel 上有效但层差 < 5 pp，按方案改用首 token logprob 再切一次层。")
    md.append("")
    md.append("| Bench | logprob Yes | consistent Acc | conflict Acc | 层差 | 95% CI | n_cons | n_conf |")
    md.append("|---|---:|---:|---:|---:|---|---:|---:|")
    for bench, rec in (blob.get("logprob_round") or {}).items():
        gap = rec.get("gap") or {}
        md.append(
            f"| {bench} | {pct(rec.get('prior_yes_ratio'))} | {pct(rec.get('consistent_acc'))} | "
            f"{pct(rec.get('conflict_acc'))} | {pp(gap.get('gap'))} pp | {ci_str(gap.get('ci95'))} | "
            f"{rec.get('n_consistent','')} | {rec.get('n_conflict','')} |"
        )
    md.append("")
    md.append("## 次要：微调是否改写了先验（MMRel text-only）")
    md.append("")
    md.append("| Contrast | n | 一致率 |")
    md.append("|---|---:|---:|")
    for name, rec in blob.get("prior_drift", {}).items():
        md.append(f"| {name} | {rec.get('n','')} | {pct(rec.get('rate'))} |")
    md.append("")
    md.append("## 交叉：conflict × Phase 4B lemma seen/unseen（n<50 只描述）")
    md.append("")
    md.append("| Contrast | Bench | cell | n | ΔAcc | 检验 |")
    md.append("|---|---|---|---:|---:|---|")
    for rec in blob.get("cross", []):
        test = "描述性" if rec.get("descriptive_only") else f"p={rec.get('p')}"
        md.append(
            f"| {rec['contrast']} | {rec['bench']} | {rec['cell']} | {rec['n']} | {pp(rec.get('delta'))} pp | {test} |"
        )
    md.append("")
    md.append("## 本阶段问题")
    md.append("")
    md.append("> **Base 模型的关系幻觉是否集中在 prior-conflict 子集？Phase 4C 的 mf 增益是否也集中在该子集？**")
    md.append("")
    md.append(blob["case_next"])
    md.append("")
    return "\n".join(md) + "\n"


def classify(blob: dict) -> tuple[str, str, str]:
    d1 = blob["diag1"].get("mmrel_adv") or {}
    gap = (d1.get("gap") or {}).get("gap")
    lp_gap = ((blob.get("logprob_round") or {}).get("mmrel_adv") or {}).get("gap") or {}
    gap2 = lp_gap.get("gap")
    mm = (blob["diag2"].get("mmrel_adv") or {}).get("mf_vs_base") or {}
    d_conf = ((mm.get("conflict") or {}).get("delta") or {}).get("delta")
    d_cons = ((mm.get("consistent") or {}).get("delta") or {}).get("delta")
    did = (mm.get("did") or {}).get("did")
    did_ci = (mm.get("did") or {}).get("ci95") or [None, None]
    val = blob["validity"].get("mmrel_adv") or {}
    amber_gap = ((blob.get("diag1") or {}).get("amber_dr") or {}).get("gap") or {}
    failed_rbench = (blob.get("validity") or {}).get("rbench", {}).get("measurement_failed")
    if gap is None:
        return (
            "C",
            "先验测不出：无法计算 MMRel base 层差。",
            "Case C → 对比式正反两问若仍无层差，则放弃 prior-conflict 作为主评测轴。",
        )
    both_flat = abs(gap) < LAYER_GAP_C and (gap2 is None or abs(gap2) < LAYER_GAP_C)
    if both_flat:
        extra = f"；logprob 第二轮层差 {pp(gap2)} pp" if gap2 is not None else ""
        amber = ""
        if amber_gap.get("gap") is not None:
            amber = f" AMBER 生成式层差 {pp(amber_gap.get('gap'))} pp，但不是 mf 增益所在题集。"
        rnote = " R-Bench 拒答率/logprob 单边，测量失败。" if failed_rbench else ""
        return (
            "C",
            (
                f"先验测不出：MMRel 生成式层差 {100 * gap:+.2f} pp（< 5 pp）{extra}。"
                f" mf 增益在 consistent {pp(d_cons)}、conflict {pp(d_conf)}，不集中在冲突层。"
                f"{amber}{rnote}"
            ),
            "Case C → 生成式与 logprob 两轮都失败。若还要保留该框架，只剩对比式先验（同题正反两问）；否则不要把 conflict 子集 Acc 当作主轴。",
        )
    concentrated = False
    if d_conf is not None and d_conf > 0 and did is not None:
        ci_ok = did_ci[0] is not None and not ci_contains_zero(did_ci)
        point_ok = did >= 0.05
        signs = []
        for cname in ("mf_vs_base", "clean_vs_base", "s3000_vs_base"):
            d = ((blob["diag2"].get("mmrel_adv") or {}).get(cname) or {}).get("did") or {}
            if d.get("did") is not None:
                signs.append(d["did"] > 0)
        same_sign = bool(signs) and all(signs)
        concentrated = ci_ok or (point_ok and same_sign)
    if gap >= LAYER_GAP_A and concentrated:
        return (
            "A",
            f"框架成立：MMRel base 层差 {100 * gap:+.2f} pp，mf_vs_base conflict ΔAcc={pp(d_conf)}，DiD={pp(did)}。",
            "Case A → 采纳 conflict 子集 Acc 为主评测轴。启动 Phase 5B：PSG 数据重建 + 按先验分歧采样训练样本。",
        )
    if gap >= LAYER_GAP_A:
        return (
            "B",
            f"先验依赖存在（MMRel 层差 {100 * gap:+.2f} pp），但 mf 增益未集中在 conflict（Δconf={pp(d_conf)}，DiD={pp(did)}）。",
            "Case B → 诊断有效但现有训练没有专门打先验冲突。Phase 5B 直接按先验分歧采样设计训练数据。",
        )
    return (
        "B",
        f"先验依赖可见但未达 Case A 门槛（MMRel 层差 {100 * gap:+.2f} pp，介于 5–15）。mf Δconf={pp(d_conf)}，DiD={pp(did)}。",
        "Case B → 先验信号存在但不够强；Phase 5B 仍可按先验分歧采样设计训练数据，同时保留整体 Acc。",
    )


def validity_only() -> int:
    need = False
    for bench, path in TEXTONLY["base"].items():
        questions = load_questions(bench)
        rows = load_rows(path)
        rec = bench_validity(questions, rows, "text")
        print(
            f"[validity] {bench} invalid_rate={rec['invalid_rate']:.3f} yes_ratio={rec['yes_ratio']:.3f} "
            f"degenerate={rec['degenerate']} n={rec['n']} valid={rec['valid']}",
            flush=True,
        )
        if rec["need_fallback"]:
            need = True
            print(f"NEED_FALLBACK {bench}", flush=True)
    return 0 if not need else 2


def drop_ok_lists(obj):
    if isinstance(obj, dict):
        return {k: drop_ok_lists(v) for k, v in obj.items() if k not in {"ok", "a_ok", "b_ok"}}
    if isinstance(obj, list):
        return [drop_ok_lists(x) for x in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validity-only", action="store_true")
    args = parser.parse_args()
    if args.validity_only:
        raise SystemExit(validity_only())

    questions = {b: load_questions(b) for b in ("mmrel_adv", "rbench", "amber_dr", "heldout")}
    extract_path = ROOT / "eval_results/qwen/phase4b/metrics/relation_extraction.jsonl"
    extract = load_json_or_jsonl(extract_path) if extract_path.exists() else []
    rel_ids = relation_only_ids(extract)
    rbench_rel_q = [q for q in questions["rbench"] if any(k in rel_ids for k in qid_aliases(q.get("question_id")))]
    print(f"[rbench] unique={len(questions['rbench'])} relation-only={len(rbench_rel_q)}", flush=True)

    vision_preds = {m: {b: answer_preds(p) for b, p in paths.items()} for m, paths in VISION.items()}

    # Reconcile vision Acc before any layering.
    reconcile = {}
    for model, pubs in PUBLISHED_UNIQUE.items():
        reconcile[model] = {}
        for bench, published in pubs.items():
            if bench == "amber_dr":
                rec = score_amber(VISION[model][bench])
                acc = 100.0 * rec["Accuracy"]
            elif bench == "heldout":
                rec = unique_acc(questions["heldout"], vision_preds[model]["heldout"])
                acc = rec["acc_pp"]
            elif bench == "rbench":
                rec = unique_acc(questions["rbench"], vision_preds[model]["rbench"])
                acc = rec["acc_pp"]
            else:
                rec = unique_acc(questions["mmrel_adv"], vision_preds[model]["mmrel_adv"])
                acc = rec["acc_pp"]
            ok = abs(acc - published) <= 0.05
            reconcile[model][bench] = {"published": published, "recomputed": round(acc, 2), "match": ok}
            print(f"[reconcile] {model}/{bench} {acc:.2f} vs {published:.2f} match={ok}", flush=True)
            if not ok:
                raise SystemExit(f"Acc mismatch {model}/{bench}: {acc:.2f} vs published {published:.2f}")

    # Choose prior source per bench (base defines layers).
    validity = {}
    prior_rows = {}
    prior_mode = {}
    for bench, path in TEXTONLY["base"].items():
        main_rows = load_rows(path)
        fb_path = TEXTONLY_FB["base"].get(bench)
        fb_rows = load_rows(fb_path) if fb_path and fb_path.exists() else {}
        choice = choose_prior_mode(questions[bench], main_rows, fb_rows)
        validity[bench] = choice
        prior_mode[bench] = "logprob" if choice["used"] == "logprob" else "text"
        prior_rows[bench] = fb_rows if choice["rows"] == "fallback" else main_rows
        print(
            f"[prior] {bench} used={choice['used']} rows={choice['rows']} "
            f"invalid={(choice.get(choice['used']) or choice['main']).get('invalid_rate')}",
            flush=True,
        )

    slices = {
        "mmrel_adv": questions["mmrel_adv"],
        "rbench": rbench_rel_q,
        "rbench_full": questions["rbench"],
        "amber_dr": questions["amber_dr"],
        "heldout": questions["heldout"],
    }
    bench_of = {
        "mmrel_adv": "mmrel_adv",
        "rbench": "rbench",
        "rbench_full": "rbench",
        "amber_dr": "amber_dr",
        "heldout": "heldout",
    }

    all_items = []
    grouped = {}
    for slice_name, qs in slices.items():
        bench = bench_of[slice_name]
        vis = {m: vision_preds[m][bench] for m in vision_preds}
        items = item_records(bench, qs, prior_rows[bench], prior_mode[bench], vis)
        grouped[slice_name] = items
        if slice_name != "rbench_full":
            all_items.extend(items)

    layers = {}
    diag1 = {}
    for slice_name, items in grouped.items():
        layers[slice_name] = {
            "all": layer_summary(items),
            "consistent": layer_summary(subset(items, "consistent")),
            "conflict": layer_summary(subset(items, "conflict")),
            "invalid": layer_summary(subset(items, "invalid")),
        }
        cons = subset(items, "consistent")
        conf = subset(items, "conflict")
        cons_s = vision_layer_stats(cons, "base")
        conf_s = vision_layer_stats(conf, "base")
        gap = bootstrap_gap(cons_s.get("ok") or [], conf_s.get("ok") or [])
        diag1[slice_name] = {
            "n_consistent": len(cons),
            "n_conflict": len(conf),
            "n_invalid": layers[slice_name]["invalid"]["n"],
            "consistent_acc": cons_s.get("Accuracy"),
            "conflict_acc": conf_s.get("Accuracy"),
            "consistent": {k: v for k, v in cons_s.items() if k != "ok"},
            "conflict": {k: v for k, v in conf_s.items() if k != "ok"},
            "gap": {k: v for k, v in gap.items() if k not in {"n_boot"}},
            "lock": lock_rate(items, "base"),
        }

    diag2 = {}
    for slice_name, items in grouped.items():
        diag2[slice_name] = {}
        for a, b, cname in CONTRASTS:
            cons = contrast_on_layer(subset(items, "consistent"), a, b)
            conf = contrast_on_layer(subset(items, "conflict"), a, b)
            did_cf = bootstrap_did(
                conf.get("a_ok") or [],
                conf.get("b_ok") or [],
                cons.get("a_ok") or [],
                cons.get("b_ok") or [],
                n_boot=N_BOOT,
                seed=SEED,
            )
            diag2[slice_name][cname] = {
                "a": a,
                "b": b,
                "consistent": {k: v for k, v in cons.items() if k not in {"a_ok", "b_ok"}},
                "conflict": {k: v for k, v in conf.items() if k not in {"a_ok", "b_ok"}},
                "did": {
                    "did": did_cf.get("did"),
                    "delta_conflict": did_cf.get("delta_seen"),
                    "delta_consistent": did_cf.get("delta_unseen"),
                    "ci95": did_cf.get("ci95"),
                    "n_conflict": did_cf.get("n_seen"),
                    "n_consistent": did_cf.get("n_unseen"),
                },
            }

    # Secondary: prior drift on MMRel.
    mm_items = {str(x["question_id"]): x for x in grouped["mmrel_adv"]}
    prior_drift = {}
    for model in ("clean", "mf"):
        path = TEXTONLY[model]["mmrel_adv"]
        fb = TEXTONLY_FB[model]["mmrel_adv"]
        rows = load_rows(path)
        used_rows = rows
        mode = prior_mode["mmrel_adv"]
        if validity["mmrel_adv"]["rows"] == "fallback" and fb.exists():
            used_rows = load_rows(fb)
        other = item_records("mmrel_adv", questions["mmrel_adv"], used_rows, mode, {"base": vision_preds["base"]["mmrel_adv"]})
        other_map = {str(x["question_id"]): x for x in other}
        prior_drift[f"{model}_vs_base"] = prior_agreement(mm_items, other_map)

    # Cross conflict × seen/unseen.
    cross = []
    ext = {
        "mmrel_adv": extract_index(extract, "mmrel_adv"),
        "rbench": extract_index(extract, "rbench"),
    }
    for slice_name in ("mmrel_adv", "rbench"):
        items = grouped[slice_name]
        idx = ext[slice_name]
        for a, b, cname in CONTRASTS:
            mkey = match_key_for(cname)
            for layer in ("conflict", "consistent"):
                for tag, pred in (("seen", True), ("unseen", False)):
                    cell_items = []
                    for x in subset(items, layer):
                        info = lookup(idx, x["question_id"]) or {}
                        seen = seen_at("lemma", info.get(mkey) or "unparsed")
                        if seen is None:
                            continue
                        if seen is pred:
                            cell_items.append(x)
                    rec = contrast_on_layer(cell_items, a, b)
                    d = (rec.get("delta") or {}).get("delta")
                    p = (rec.get("mcnemar") or {}).get("p_value")
                    n = rec.get("n") or 0
                    cross.append(
                        {
                            "contrast": cname,
                            "bench": slice_name,
                            "cell": f"{layer}×{tag}",
                            "n": n,
                            "delta": d,
                            "p": None if n < CROSS_MIN_N else p,
                            "descriptive_only": n < CROSS_MIN_N,
                            "ci95": None if n < CROSS_MIN_N else (rec.get("delta") or {}).get("ci95"),
                        }
                    )

    logprob_round = {}
    for slice_name, qs in slices.items():
        bench = bench_of[slice_name]
        vis = {m: vision_preds[m][bench] for m in vision_preds}
        main_r = load_rows(TEXTONLY["base"][bench])
        items = item_records(bench, qs, main_r, "logprob", vis)
        cons = subset(items, "consistent")
        conf = subset(items, "conflict")
        cons_s = vision_layer_stats(cons, "base")
        conf_s = vision_layer_stats(conf, "base")
        gap = bootstrap_gap(cons_s.get("ok") or [], conf_s.get("ok") or [])
        logprob_round[slice_name] = {
            "prior_yes_ratio": layer_summary(items).get("prior_yes_ratio"),
            "n_consistent": len(cons),
            "n_conflict": len(conf),
            "consistent_acc": cons_s.get("Accuracy"),
            "conflict_acc": conf_s.get("Accuracy"),
            "gap": {k: v for k, v in gap.items() if k not in {"n_boot"}},
        }

    blob = {
        "validity": validity,
        "prior_mode": prior_mode,
        "reconcile": reconcile,
        "layers": layers,
        "diag1": diag1,
        "diag2": drop_ok_lists(diag2),
        "logprob_round": logprob_round,
        "prior_drift": prior_drift,
        "cross": cross,
        "n_boot": N_BOOT,
        "rbench_relation_only_n": len(rbench_rel_q),
    }
    case, reason, nxt = classify(blob)
    blob["case"] = case
    blob["case_reason"] = reason
    blob["case_next"] = nxt

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics").mkdir(exist_ok=True)
    slim = drop_ok_lists(blob)
    # validity contains nested objects with used flags; fine.
    (OUT / "metrics/phase5a_prior_analysis.json").write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")

    labels_path = OUT / "metrics/prior_labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as fout:
        for slice_name, bench_name in (
            ("mmrel_adv", "mmrel_adv"),
            ("rbench_full", "rbench"),
            ("amber_dr", "amber_dr"),
            ("heldout", "heldout"),
        ):
            for x in grouped[slice_name]:
                fout.write(
                    json.dumps(
                        {
                            "question_id": x["question_id"],
                            "benchmark": bench_name,
                            "prior_answer": x["prior_answer"],
                            "gold": x["gold_label"],
                            "layer": x["layer"],
                            "prior_source": x["prior_source"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    text_md = write_md(blob)
    (OUT / "phase5a_report.md").write_text(text_md, encoding="utf-8")
    (ROOT / "phase5a_result.md").write_text(text_md, encoding="utf-8")
    print(text_md)
    print(f"wrote {OUT / 'metrics/phase5a_prior_analysis.json'}")
    print(f"wrote {labels_path}")
    print(f"wrote {OUT / 'phase5a_report.md'}")


if __name__ == "__main__":
    main()
