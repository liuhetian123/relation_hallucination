#!/usr/bin/env python3
"""Download, extract, and verify Reefknot/OpenPSG data without using project storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

REEFKNOT_GIT = "https://github.com/JackChen-seu/Reefknot.git"
OPENPSG_GIT = "https://github.com/Jingkang50/OpenPSG.git"
REEFKNOT_HF = "MM-Hallu/Reefknot"
OPENPSG_HF = "HarborYuan/OpenPSG"
COCO_URLS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
}


def run(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


def clone_or_update(url: str, path: Path) -> None:
    if (path / ".git").is_dir():
        run("git", "-C", str(path), "pull", "--ff-only")
    else:
        run("git", "clone", "--filter=blob:none", url, str(path))


def snapshot(repo: str, path: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub before downloading dataset mirrors") from exc
    snapshot_download(repo, repo_type="dataset", local_dir=path, max_workers=8)


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"Already present, skipping download: {path}")
        return
    partial = path.with_suffix(path.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(request) as response, partial.open("ab" if start else "wb") as out:
        if start and response.status != 206:
            out.close()
            partial.unlink()
            return download(url, path)
        while chunk := response.read(8 * 1024 * 1024):
            out.write(chunk)
    partial.replace(path)


def extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Corrupt ZIP member in {archive}: {bad}")
        zf.extractall(destination)


def extract_reefknot_images(parquet_root: Path, image_dir: Path) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Install pyarrow before extracting Reefknot image parquet files") from exc
    image_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    conflicts = 0
    for parquet in sorted(parquet_root.glob("*/*.parquet")):
        table = pq.read_table(parquet, columns=["image", "image_id"])
        for row in table.to_pylist():
            image_id = str(row["image_id"])
            payload = row["image"]["bytes"]
            suffix = Path(row["image"].get("path") or "").suffix or ".jpg"
            digest = hashlib.sha256(payload).hexdigest()
            if image_id in seen:
                conflicts += seen[image_id] != digest
                continue
            destination = image_dir / f"{image_id}{suffix}"
            if not destination.exists() or destination.stat().st_size != len(payload):
                destination.write_bytes(payload)
            seen[image_id] = digest
    return {"unique_images": len(seen), "conflicting_duplicate_images": conflicts}


def jsonl_stats(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "rows": len(rows),
        "fields": sorted(set().union(*(row.keys() for row in rows))),
        "unique_image_ids": len({str(row["image_id"]) for row in rows}),
        "label": dict(Counter(str(row.get("label")) for row in rows)),
        "relation_type": dict(Counter(str(row.get("relation_type")) for row in rows)),
    }


def file_record(path: Path) -> dict:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def verify(root: Path, manifest_path: Path) -> None:
    reef_annotations = root / "Reefknot" / "Dataset"
    reef_images = root / "Reefknot_images"
    psg_root = root / "OpenPSG_annotations"
    coco_root = root / "COCO2017"
    reef_files = [reef_annotations / name for name in ("YESNO.jsonl", "Multichoice.jsonl", "VQA.jsonl")]
    psg_files = [psg_root / name for name in ("psg.json", "psg_train_val.json", "psg_val_test.json")]
    missing = [str(path) for path in reef_files + psg_files if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")
    all_ids = set()
    reef_stats = {}
    for path in reef_files:
        reef_stats[path.name] = jsonl_stats(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_ids.add(str(json.loads(line)["image_id"]))
    available = {path.stem for path in reef_images.glob("*.jpg")}
    psg = json.loads((psg_root / "psg.json").read_text(encoding="utf-8"))
    test_ids = {str(value) for value in psg["test_image_ids"]}
    psg_train_ids = {
        str(row["image_id"])
        for row in psg["data"]
        if str(row["image_id"]) not in test_ids
    }
    safe_psg_train_ids = sorted(psg_train_ids - all_ids)
    exclusion_path = manifest_path.with_name("psg_train_excluding_reefknot_image_ids.txt")
    exclusion_path.parent.mkdir(parents=True, exist_ok=True)
    exclusion_path.write_text("\n".join(safe_psg_train_ids) + "\n", encoding="utf-8")
    manifest = {
        "root": str(root),
        "reefknot": {
            "annotations": reef_stats,
            "all_unique_image_ids": len(all_ids),
            "available_images": len(available),
            "missing_images": len(all_ids - available),
            "files": [file_record(path) for path in reef_files],
        },
        "openpsg": {
            "samples": len(psg["data"]),
            "train_samples_excluding_test_ids": len(psg_train_ids),
            "test_ids": len(test_ids),
            "predicate_classes": len(psg["predicate_classes"]),
            "reefknot_overlap_in_train": len(psg_train_ids & all_ids),
            "leakage_safe_train_samples": len(safe_psg_train_ids),
            "leakage_safe_image_ids_file": str(exclusion_path),
            "files": [file_record(path) for path in psg_files],
        },
        "coco2017": {
            "train_images": len(list((coco_root / "train2017").glob("*.jpg"))),
            "val_images": len(list((coco_root / "val2017").glob("*.jpg"))),
            "archives": [
                file_record(coco_root / "archives" / name)
                for name in COCO_URLS
                if (coco_root / "archives" / name).exists()
            ],
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/storage22t/lht/datasets"))
    parser.add_argument("--manifest", type=Path, default=Path("data_manifests/phase5_data_manifest.json"))
    parser.add_argument("--clone", action="store_true", help="Clone/update official code and annotation repositories")
    parser.add_argument("--download-reefknot-images", action="store_true", help="Download minimal HF parquet mirror")
    parser.add_argument("--extract-reefknot-images", action="store_true")
    parser.add_argument("--download-openpsg-annotations", action="store_true", help="Download PSG JSON mirror")
    parser.add_argument("--download-coco", action="store_true")
    parser.add_argument("--extract-coco", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.clone:
        clone_or_update(REEFKNOT_GIT, args.root / "Reefknot")
        clone_or_update(OPENPSG_GIT, args.root / "OpenPSG")
    if args.download_reefknot_images:
        snapshot(REEFKNOT_HF, args.root / "Reefknot_hf")
    if args.extract_reefknot_images:
        print(extract_reefknot_images(args.root / "Reefknot_hf", args.root / "Reefknot_images"))
    if args.download_openpsg_annotations:
        snapshot(OPENPSG_HF, args.root / "OpenPSG_annotations")
    if args.download_coco:
        for name, url in COCO_URLS.items():
            download(url, args.root / "COCO2017" / "archives" / name)
    if args.extract_coco:
        for name in COCO_URLS:
            extract_zip(args.root / "COCO2017" / "archives" / name, args.root / "COCO2017")
    if args.verify:
        verify(args.root, args.manifest)
    if not any(vars(args)[key] for key in vars(args) if key not in {"root", "manifest"}):
        print("No action selected; use --help.", file=sys.stderr)


if __name__ == "__main__":
    main()
