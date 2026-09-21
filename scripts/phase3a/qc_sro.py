#!/usr/bin/env python3
"""Re-run SRO QC on raw InternVL JSONL and write sro_ok.jsonl plus stats."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, dump_json, dump_jsonl, extract_json_object, load_json, qc_sro


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default=str(ROOT / "data/phase3a/sro_raw.jsonl"))
    p.add_argument("--out-ok", default=str(ROOT / "data/phase3a/sro_ok.jsonl"))
    p.add_argument("--out-stats", default=str(ROOT / "data/phase3a/sro_qc_stats.json"))
    args = p.parse_args()

    rows = load_json(Path(args.raw))
    ok = []
    status = Counter()
    for rec in rows:
        parsed = None
        if rec.get("subject") and rec.get("relation") and rec.get("object") and rec.get("confidence"):
            parsed = rec
        else:
            parsed, _ = extract_json_object(rec.get("teacher_raw_output") or "")
        qc, fields = qc_sro(parsed)
        rec = {**rec, **fields, "qc_status": qc}
        status[qc] += 1
        if qc == "ok":
            ok.append(rec)
    dump_jsonl(Path(args.out_ok), ok)
    blob = {"n_raw": len(rows), "n_ok": len(ok), "status": dict(status)}
    dump_json(Path(args.out_stats), blob)
    print(json.dumps(blob, indent=2))


if __name__ == "__main__":
    main()
