#!/usr/bin/env python3
"""Independent InternVL visual-falsity pass on the 100 QC pairs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_p4 = _load("phase4a_common", ROOT / "scripts/phase4a/common.py")
INTERNVL_PATH = _p4.INTERNVL_PATH
VISUAL_PROMPT = _p4.VISUAL_PROMPT
parse_yes_no_uncertain = _p4.parse_yes_no_uncertain
sys.path.insert(0, str(ROOT / "scripts/phase3a"))
from internvl_engine import load_internvl, open_rgb  # noqa: E402

QC = ROOT / "data/phase4a/qc"
OUT = ROOT / "eval_results/qwen/phase4b/metrics"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", default=str(QC / "review_100_falsity.tsv"))
    p.add_argument("--gpu", default=None)
    p.add_argument("--model-path", default=str(INTERNVL_PATH))
    p.add_argument("--out", default=str(OUT / "falsity_recheck_100.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.tsv).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    infer = load_internvl(args.model_path, gpu=args.gpu)
    results = []
    counts = Counter()
    for i, rec in enumerate(rows, start=1):
        img = open_rgb(Path(rec["image_path"]))
        prompt = VISUAL_PROMPT.format(
            subject=rec["subject"],
            object=rec["object"],
            candidate_relation=rec["negative_relation"],
        )
        raw = infer(img, prompt, max_new_tokens=16)
        judgment = parse_yes_no_uncertain(raw)
        counts[judgment] += 1
        rec = dict(rec)
        rec["internvl_raw"] = raw
        rec["internvl_judgment"] = judgment
        results.append(rec)
        print(f"[{i}/{len(rows)}] {rec['id']} {judgment} raw={raw!r}", flush=True)

    n = len(results) or 1
    yes = counts.get("yes", 0)
    unc = counts.get("uncertain", 0)
    no = counts.get("no", 0)
    other = n - yes - unc - no
    noise_upper = (yes + unc) / n
    blob = {
        "n": len(results),
        "counts": dict(counts),
        "yes_rate": yes / n,
        "uncertain_rate": unc / n,
        "no_rate": no / n,
        "other_rate": other / n,
        "noise_upper_yes_or_uncertain": noise_upper,
        "verdict": (
            "clean_ok_leq_5pct"
            if noise_upper <= 0.05
            else "usable_5_to_15pct"
            if noise_upper <= 0.15
            else "filter_before_phase4c"
        ),
        "rows": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fields = list(rows[0].keys())
    if "internvl_judgment" not in fields:
        fields += ["internvl_judgment", "internvl_raw"]
    with Path(args.tsv).open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for rec in results:
            f.write("\t".join((str(rec.get(k) or "")).replace("\t", " ").replace("\n", " ") for k in fields) + "\n")

    html_path = QC / "review_100_falsity.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        start = html.find("const ITEMS = ")
        end = html.find(";\nconst KEY")
        if start >= 0 and end > start:
            payload = []
            for rec in results:
                payload.append(
                    {
                        "id": rec["id"],
                        "image_path": rec["image_path"],
                        "image_file": Path(rec["image_path"]).name,
                        "family": rec.get("family") or "",
                        "subject": rec.get("subject") or "",
                        "object": rec.get("object") or "",
                        "positive_relation": rec.get("positive_relation") or "",
                        "positive_statement": rec.get("positive_statement") or "",
                        "negative_relation": rec.get("negative_relation") or "",
                        "negative_statement": rec.get("negative_statement") or "",
                        "internvl_judgment": rec.get("internvl_judgment") or "",
                        "internvl_raw": rec.get("internvl_raw") or "",
                        "human_falsity": rec.get("human_falsity") or "",
                    }
                )
            html = html[: start + len("const ITEMS = ")] + json.dumps(payload, ensure_ascii=False) + html[end:]
            html_path.write_text(html, encoding="utf-8")
    print(json.dumps({k: blob[k] for k in blob if k != "rows"}, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
