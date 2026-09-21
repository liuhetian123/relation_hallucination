#!/usr/bin/env python3
"""Generate InternVL counterfactual negatives, rule-filter, then verify.

Resumable JSONL. Drops samples with no high-confidence clean candidate.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    CF_PROMPT,
    DATA_DIR,
    IMAGE_ROOT,
    INTERNVL_PATH,
    MASTER_PATH,
    MAX_ATTEMPTS,
    PLAUSIBLE_PROMPT,
    RETRY_NOTE,
    TEACHER_NAME,
    VISUAL_PROMPT,
    append_jsonl,
    extract_candidates,
    load_json,
    load_jsonl_map,
    normalize_relation,
    parse_plausible,
    parse_teacher_json,
    parse_yes_no_uncertain,
    recover_relation,
    render_statement,
    rule_reject,
    teacher_flags_ok,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase3a"))
from internvl_engine import load_internvl, open_rgb  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master", default=str(MASTER_PATH))
    p.add_argument("--output", default=str(DATA_DIR / "clean_attempts.jsonl"))
    p.add_argument("--image-root", default=str(IMAGE_ROOT))
    p.add_argument("--model-path", default=str(INTERNVL_PATH))
    p.add_argument("--gpu", default=None)
    p.add_argument("--max-patches", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-jsonl", action="append", default=[], help="Extra JSONL whose ids are skipped")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--skip-text-filter", action="store_true")
    p.add_argument("--skip-visual-filter", action="store_true")
    return p.parse_args()


def process_one(infer, image, rec: dict, args) -> dict:
    subject = rec["subject"]
    obj = rec["object"]
    pos_rel = rec.get("relation") or rec.get("positive_relation")
    prompt = CF_PROMPT.format(
        subject=subject,
        positive_relation=pos_rel,
        object=obj,
    )
    attempts = []
    selected = None
    drop_reason = "no_high_confidence_candidate"

    for attempt in range(args.max_attempts):
        use_prompt = prompt if attempt == 0 else prompt + RETRY_NOTE
        try:
            raw = infer(
                image,
                use_prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=attempt > 0,
                temperature=0.7 if attempt > 0 else None,
            )
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "teacher_raw_output": repr(exc),
                    "parse_ok": False,
                    "candidates": [],
                    "error": "inference_error",
                }
            )
            drop_reason = "inference_error"
            continue
        parsed, repaired = parse_teacher_json(raw)
        cands = extract_candidates(parsed)
        att = {
            "attempt": attempt,
            "teacher_raw_output": raw,
            "json_repaired": repaired,
            "parse_ok": bool(cands),
            "candidates": [],
        }
        if not cands:
            att["error"] = "json_parse_error"
            attempts.append(att)
            drop_reason = "json_parse_error"
            continue

        pending = []
        seen = set()
        for rank, cand in enumerate(cands):
            item = dict(cand)
            item["rank"] = rank
            rel = recover_relation(cand["relation"], cand.get("teacher_statement") or "", subject, obj)
            item["relation_recovered_from"] = cand["relation"] if rel != cand["relation"] else None
            item["relation"] = rel
            item["relation_norm"] = normalize_relation(rel)
            rn = item["relation_norm"]
            cand["relation"] = rel
            cand["relation_norm"] = rn
            if rn in seen:
                item["status"] = "duplicate"
                att["candidates"].append(item)
                continue
            seen.add(rn)
            rule = rule_reject(pos_rel, rel, subject, obj)
            if rule:
                item["status"] = rule
                att["candidates"].append(item)
                continue
            flags = teacher_flags_ok(cand)
            if flags:
                item["status"] = flags
                att["candidates"].append(item)
                continue
            item["status"] = "pending_verify"
            pending.append(item)

        pending.sort(key=lambda x: (0 if x.get("confidence") == "high" else 1, x["rank"]))
        for item in pending:
            rel = item["relation"]
            rn = item["relation_norm"]
            stmt = render_statement(subject, rel, obj)
            item["rendered_statement"] = stmt
            if not args.skip_text_filter:
                t_prompt = PLAUSIBLE_PROMPT.format(
                    subject=subject,
                    candidate_relation=rel,
                    object=obj,
                    candidate_statement=stmt,
                )
                t_raw = infer.generate(t_prompt, image=None, max_new_tokens=16)
                item["text_raw"] = t_raw
                item["text_judgment"] = parse_plausible(t_raw)
                if item["text_judgment"] != "plausible":
                    item["status"] = f"text_{item['text_judgment']}"
                    att["candidates"].append(item)
                    continue
            if not args.skip_visual_filter:
                v_prompt = VISUAL_PROMPT.format(
                    subject=subject,
                    object=obj,
                    candidate_relation=rel,
                )
                v_raw = infer(image, v_prompt, max_new_tokens=16)
                item["visual_raw"] = v_raw
                item["visual_judgment"] = parse_yes_no_uncertain(v_raw)
                if item["visual_judgment"] != "no":
                    item["status"] = f"visual_{item['visual_judgment']}"
                    att["candidates"].append(item)
                    continue
            item["status"] = "selected"
            att["candidates"].append(item)
            selected = {
                "negative_relation": rel,
                "negative_relation_norm": rn,
                "candidate_rank": item["rank"],
                "attempt": attempt,
                "linguistically_plausible": True,
                "visually_supported": False,
                "negative_confidence": item.get("confidence") or "high",
                "text_judgment": item.get("text_judgment", "skipped"),
                "visual_judgment": item.get("visual_judgment", "skipped"),
                "teacher_statement": item.get("teacher_statement", ""),
            }
            break
        attempts.append(att)
        if selected:
            drop_reason = None
            break
        drop_reason = "no_high_confidence_candidate"

    row = {
        "id": rec["id"],
        "image": rec["image"],
        "subject": subject,
        "positive_relation": pos_rel,
        "object": obj,
        "old_negative_relation": rec.get("negative_relation"),
        "old_negative_statement": rec.get("negative_statement"),
        "positive_statement_src": rec.get("positive_statement"),
        "sro_teacher": rec.get("sro_teacher", TEACHER_NAME),
        "negative_source": TEACHER_NAME,
        "qc_status": "ok" if selected else "dropped",
        "drop_reason": drop_reason,
        "n_attempts": len(attempts),
        "attempts": attempts,
        "master_index": rec.get("master_index"),
        "id_scale": rec.get("id_scale"),
        "split_seed": rec.get("split_seed"),
    }
    if selected:
        row.update(selected)
        row["positive_statement"] = render_statement(subject, pos_rel, obj)
        row["negative_statement"] = render_statement(subject, selected["negative_relation"], obj)
    return row


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    master = load_json(Path(args.master))
    out_path = Path(args.output)
    existing = load_jsonl_map(out_path, "id")
    for extra in args.skip_jsonl:
        existing.update(load_jsonl_map(Path(extra), "id"))
    n_ok = sum(1 for r in existing.values() if r.get("qc_status") == "ok")
    n_drop = sum(1 for r in existing.values() if r.get("qc_status") != "ok")
    remaining = [r for r in master if r["id"] not in existing]
    if args.shards > 1:
        if not (0 <= args.shard < args.shards):
            raise SystemExit(f"shard must be in [0, {args.shards})")
        remaining = remaining[args.shard :: args.shards]
    if args.limit is not None:
        remaining = remaining[: args.limit]
    print(
        f"master={len(master)} already={len(existing)} ok={n_ok} dropped={n_drop} "
        f"shard={args.shard}/{args.shards} todo={len(remaining)} "
        f"text_filter={not args.skip_text_filter} visual_filter={not args.skip_visual_filter}",
        flush=True,
    )
    if not remaining:
        return

    infer = load_internvl(args.model_path, gpu=None, max_patches=args.max_patches)
    image_root = Path(args.image_root)
    t0 = time.time()
    for i, rec in enumerate(remaining, start=1):
        path = image_root / rec["image"]
        try:
            image = open_rgb(path)
            row = process_one(infer, image, rec, args)
        except Exception as exc:
            row = {
                "id": rec["id"],
                "image": rec["image"],
                "qc_status": "dropped",
                "drop_reason": "inference_error",
                "error": repr(exc),
            }
        append_jsonl(out_path, row)
        if row.get("qc_status") == "ok":
            n_ok += 1
        else:
            n_drop += 1
        elapsed = time.time() - t0
        rate = elapsed / i
        eta = rate * (len(remaining) - i)
        print(
            f"[{i}/{len(remaining)}] {rec['id']} status={row.get('qc_status')} "
            f"neg={row.get('negative_relation')} reason={row.get('drop_reason')} "
            f"ok={n_ok} drop={n_drop} eta_min={eta/60:.1f}",
            flush=True,
        )
    print(f"done ok={n_ok} dropped={n_drop} output={out_path}", flush=True)


if __name__ == "__main__":
    main()
