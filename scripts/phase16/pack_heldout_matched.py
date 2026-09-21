#!/usr/bin/env python3
"""Pack held-out matched Typed/Explicit captions and reconstruction QC."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/phase15"))
from relation_extract import strip_caption  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heldout", default=str(ROOT / "eval_results/qwen/phase15/heldout_200.jsonl"))
    p.add_argument("--recon", default=str(ROOT / "eval_results/qwen/phase16b/heldout_200_explicit.jsonl"))
    p.add_argument("--out", default=str(ROOT / "eval_results/qwen/phase16b/heldout_matched.jsonl"))
    args = p.parse_args()

    heldout = {r["id"]: r for r in load_jsonl(Path(args.heldout))}
    recon = {r["id"]: r for r in load_jsonl(Path(args.recon))}
    status = Counter(r.get("reconstruction_status", "missing") for r in recon.values())

    n_rel_changed = 0
    n_ok_matched = 0
    rows = []
    for sid, h in heldout.items():
        rec = recon.get(sid, {})
        typed = strip_caption(h.get("anonymous_caption") or "")
        explicit = strip_caption(rec.get("explicit_caption") or "")
        rel = (h.get("gt_relation") or "").strip()
        st = rec.get("reconstruction_status", "missing")
        rel_in_typed = bool(rel) and rel.lower() in typed.lower()
        rel_in_exp = bool(rel) and rel.lower() in explicit.lower() if explicit else False
        if st == "ok" and rel and not rel_in_exp:
            n_rel_changed += 1
            st_eval = "relation_phrase_changed"
        else:
            st_eval = st
        ok_explicit = st == "ok" and bool(explicit) and (not rel or rel_in_exp)
        if ok_explicit:
            n_ok_matched += 1
        rows.append(
            {
                "id": sid,
                "image": h["image"],
                "anonymous_caption": typed,
                "explicit_caption": explicit or None,
                "gt_relation": rel,
                "reconstruction_status": st,
                "eval_status": st_eval,
                "ok_explicit": ok_explicit,
                "rel_in_typed": rel_in_typed,
                "rel_in_explicit": rel_in_exp,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    qc = {
        "total_heldout": len(heldout),
        "reconstruction_ok": status.get("ok", 0),
        "uncertain": status.get("uncertain", 0),
        "parse_failure": status.get("json_parse_error", 0) + status.get("parse_error", 0),
        "missing_placeholder": status.get("no_placeholder", 0),
        "relation_phrase_changed": n_rel_changed,
        "other_validation_failure": sum(
            v for k, v in status.items() if k not in {"ok", "uncertain", "json_parse_error", "parse_error", "no_placeholder"}
        ),
        "status_counts": dict(status),
        "ok_explicit_for_eval": n_ok_matched,
        "typed_eval_n": len(heldout),
        "missing_recon": len(heldout) - len(recon),
    }
    Path(args.out).with_name("recon_qc.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print("QC")
    for k, v in qc.items():
        print(f"  {k}: {v}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
