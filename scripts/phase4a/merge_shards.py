#!/usr/bin/env python3
"""Merge shard JSONL into data/phase4a/clean_attempts.jsonl without duplicating ids."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, append_jsonl, dump_json, load_jsonl_map


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main", default=str(DATA_DIR / "clean_attempts.jsonl"))
    p.add_argument("--shard-dir", default=str(DATA_DIR / "shards"))
    p.add_argument("--stats", default=str(DATA_DIR / "merge_stats.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    main_path = Path(args.main)
    existing = load_jsonl_map(main_path, "id")
    n_before = len(existing)
    added = 0
    shard_dir = Path(args.shard_dir)
    files = sorted(shard_dir.glob("attempts.shard*.jsonl")) if shard_dir.exists() else []
    for path in files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                import json

                rec = json.loads(line)
                sid = rec.get("id")
                if not sid or sid in existing:
                    continue
                append_jsonl(main_path, rec)
                existing[sid] = rec
                added += 1
    n_ok = sum(1 for r in existing.values() if r.get("qc_status") == "ok")
    stats = {
        "n_before": n_before,
        "n_added": added,
        "n_after": len(existing),
        "n_ok": n_ok,
        "n_dropped": len(existing) - n_ok,
        "shard_files": [str(p) for p in files],
    }
    dump_json(Path(args.stats), stats)
    print(stats, flush=True)


if __name__ == "__main__":
    main()
