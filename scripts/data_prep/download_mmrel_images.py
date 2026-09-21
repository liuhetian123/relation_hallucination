#!/usr/bin/env python3
"""Download MMRel images needed for Phase 1 eval (Dall-E from HF mirror, VG from Stanford)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/lht/works/relation_hallucination/MMRel")
HF = "https://hf-mirror.com/datasets/jiahaonie/MMRel/resolve/main"
VG_BASE = "https://cs.stanford.edu/people/rak248"


def curl(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.part")
    cmd = [
        "curl",
        "-fL",
        "-A",
        "Mozilla/5.0",
        "--retry",
        "3",
        "--max-time",
        "60",
        "-o",
        str(tmp),
        url,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 100:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            return False
        os.replace(tmp, dest)
        return dest.exists() and dest.stat().st_size > 0
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


def download_vg(rel: str) -> bool:
    dest = ROOT / rel
    if dest.exists() and dest.stat().st_size > 0:
        return True
    filename = Path(rel).name
    folder = Path(rel).parts[1] if len(Path(rel).parts) > 1 else "VG_100K_2"
    alt = "VG_100K" if folder == "VG_100K_2" else "VG_100K_2"
    for part in (folder, alt):
        if curl(f"{VG_BASE}/{part}/{filename}", dest):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vg-only", action="store_true")
    parser.add_argument("--dalle-only", action="store_true")
    args = parser.parse_args()

    img_list = Path("/tmp/mmrel_images.txt")
    dalle = [ln.strip() for ln in img_list.read_text().splitlines() if ln.strip().startswith("Dall-E_generated")]
    vg = [ln.strip() for ln in Path("/tmp/mmrel_vg_relpaths.txt").read_text().splitlines() if ln.strip()]
    print(f"Dall-E images: {len(dalle)}; VG images: {len(vg)}")

    ok = fail = 0
    if not args.vg_only:
        for i, rel in enumerate(dalle, 1):
            dest = ROOT / rel
            if curl(f"{HF}/{rel}", dest):
                ok += 1
            else:
                fail += 1
                print(f"FAIL dalle {rel}", flush=True)
            if i % 50 == 0:
                print(f"dalle {i}/{len(dalle)} ok={ok} fail={fail}", flush=True)

    vg_ok = vg_fail = 0
    if not args.dalle_only:
        for i, rel in enumerate(vg, 1):
            if download_vg(rel):
                vg_ok += 1
            else:
                vg_fail += 1
                print(f"FAIL vg {rel}", flush=True)
            if i % 20 == 0:
                print(f"vg {i}/{len(vg)} ok={vg_ok} fail={vg_fail}", flush=True)

    print(f"DONE dalle ok={ok} fail={fail}; vg ok={vg_ok} fail={vg_fail}")


if __name__ == "__main__":
    sys.exit(main() or 0)
