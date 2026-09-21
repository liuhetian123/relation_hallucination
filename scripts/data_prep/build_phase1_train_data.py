#!/usr/bin/env python3
"""Build matched Typed / Explicit RelSim-1k SFT data with a shared prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TRAIN_PROMPT = (
    "Describe the primary relation shown in this image in one short sentence."
)


def strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def llava_sample(sample_id: str, image: str, caption: str) -> dict:
    return {
        "id": sample_id,
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\n{TRAIN_PROMPT}"},
            {"from": "gpt", "value": caption},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed_input", default="data/relsim_llava_1k.json")
    parser.add_argument(
        "--explicit_jsonl",
        default="outputs/relsim_explicit/relsim_llava_1k_explicit.jsonl",
    )
    parser.add_argument("--out_dir", default="data/phase1")
    args = parser.parse_args()

    typed_raw = json.loads(Path(args.typed_input).read_text(encoding="utf-8"))
    typed_by_id = {s["id"]: s for s in typed_raw}

    explicit_ok = {}
    status_counts = {}
    with Path(args.explicit_jsonl).open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            status = rec.get("reconstruction_status", "")
            status_counts[status] = status_counts.get(status, 0) + 1
            if status != "ok" or not rec.get("explicit_caption"):
                continue
            explicit_ok[rec["id"]] = rec

    matched_ids = [s["id"] for s in typed_raw if s["id"] in explicit_ok]
    missing_typed = [sid for sid in explicit_ok if sid not in typed_by_id]
    if missing_typed:
        raise SystemExit(f"Explicit ids missing from typed data: {missing_typed[:5]}")

    typed_out = []
    explicit_out = []
    for sid in matched_ids:
        src = typed_by_id[sid]
        exp = explicit_ok[sid]
        image = src["image"]
        typed_cap = strip_wrapping_quotes(src["conversations"][-1]["value"])
        explicit_cap = strip_wrapping_quotes(exp["explicit_caption"])
        typed_out.append(llava_sample(sid, image, typed_cap))
        explicit_out.append(llava_sample(sid, image, explicit_cap))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    typed_path = out_dir / "qwen_typed_1k.json"
    explicit_path = out_dir / "qwen_explicit_1k.json"
    ids_path = out_dir / "matched_ids.json"
    meta_path = out_dir / "build_meta.json"

    typed_path.write_text(json.dumps(typed_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    explicit_path.write_text(json.dumps(explicit_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ids_path.write_text(json.dumps(matched_ids, indent=2) + "\n", encoding="utf-8")
    meta = {
        "train_prompt": TRAIN_PROMPT,
        "n_typed_source": len(typed_raw),
        "n_explicit_ok": len(explicit_ok),
        "n_matched": len(matched_ids),
        "explicit_status_counts": status_counts,
        "typed_path": str(typed_path),
        "explicit_path": str(explicit_path),
        "example_typed": typed_out[0] if typed_out else None,
        "example_explicit": explicit_out[0] if explicit_out else None,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Matched samples: {len(matched_ids)}")
    print(f"Explicit status counts: {status_counts}")
    print(f"Wrote {typed_path}")
    print(f"Wrote {explicit_path}")
    if typed_out:
        print("Typed example:", typed_out[0]["conversations"][1]["value"])
        print("Explicit example:", explicit_out[0]["conversations"][1]["value"])


if __name__ == "__main__":
    main()
