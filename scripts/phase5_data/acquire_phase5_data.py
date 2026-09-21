#!/usr/bin/env python3
"""可重跑地下载、解包并核验 Phase 5 的 Reefknot、OpenPSG 与 COCO 数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

HF_REPOS = {
    "reefknot": ("MM-Hallu/Reefknot", "Reefknot_hf"),
    "openpsg": ("HarborYuan/OpenPSG", "OpenPSG_annotations"),
}
COCO_URLS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
}


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_hf(repo_id: str, destination: Path) -> None:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=destination,
        max_workers=8,
    )


def extract_reefknot(root: Path) -> None:
    import pyarrow.parquet as pq

    source = root / "Reefknot_hf"
    image_dir = root / "Reefknot_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    for config in ("yesno", "multichoice", "vqa"):
        for parquet in sorted((source / config).glob("*.parquet")):
            table = pq.read_table(parquet)
            for row in table.to_pylist():
                image = row["image"]
                image_id = str(row["image_id"])
                name = Path(image.get("path") or f"{image_id}.jpg").name
                payload = image["bytes"]
                digest = hashlib.sha256(payload).hexdigest()
                if image_id in seen and seen[image_id] != digest:
                    raise RuntimeError(f"同一 image_id 内容冲突: {image_id}")
                seen[image_id] = digest
                target = image_dir / name
                if not target.exists() or target.stat().st_size != len(payload):
                    target.write_bytes(payload)
    print(f"Reefknot images: {len(seen)} -> {image_dir}")


def download_coco(root: Path) -> None:
    archive_dir = root / "COCO2017" / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name, url in COCO_URLS.items():
        subprocess.run(["wget", "-c", url, "-O", str(archive_dir / name)], check=True)


def extract_coco(root: Path) -> None:
    coco = root / "COCO2017"
    for name in COCO_URLS:
        archive = coco / "archives" / name
        if not zipfile.is_zipfile(archive):
            raise RuntimeError(f"ZIP 不完整: {archive}")
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"ZIP CRC 失败: {archive}: {bad}")
            zf.extractall(coco)


def verify(root: Path, manifest_path: Path) -> None:
    reef_repo = root / "Reefknot" / "Dataset"
    result: dict = {"root": str(root), "files": {}, "reefknot": {}, "openpsg": {}}
    all_image_ids: set[str] = set()
    for path in sorted(reef_repo.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        ids = [str(row["image_id"]) for row in rows]
        all_image_ids.update(ids)
        result["reefknot"][path.name] = {
            "rows": len(rows),
            "unique_image_ids": len(set(ids)),
            "labels": Counter(str(row["label"]) for row in rows),
            "relation_types": Counter(str(row["relation_type"]) for row in rows),
        }
        result["files"][str(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    image_dir = root / "Reefknot_images"
    available = {p.stem for p in image_dir.glob("*.jpg")}
    result["reefknot"]["images"] = {
        "required_unique": len(all_image_ids),
        "available_required": len(all_image_ids & available),
        "missing": sorted(all_image_ids - available),
        "extra": len(available - all_image_ids),
    }

    for path in sorted((root / "OpenPSG_annotations").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("data", [])
        result["openpsg"][path.name] = {
            "rows": len(rows),
            "predicate_classes": len(data.get("predicate_classes", [])),
            "test_image_ids": len(data.get("test_image_ids", [])),
            "sources": Counter(str(row.get("file_name", "")).split("/")[0] for row in rows),
        }
        result["files"][str(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/storage22t/lht/datasets"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/phase5_data_manifest.json"))
    parser.add_argument(
        "actions",
        nargs="+",
        choices=("download-hf", "extract-reefknot", "download-coco", "extract-coco", "verify"),
    )
    args = parser.parse_args()
    for action in args.actions:
        if action == "download-hf":
            for repo_id, dirname in HF_REPOS.values():
                download_hf(repo_id, args.root / dirname)
        elif action == "extract-reefknot":
            extract_reefknot(args.root)
        elif action == "download-coco":
            download_coco(args.root)
        elif action == "extract-coco":
            extract_coco(args.root)
        elif action == "verify":
            verify(args.root, args.manifest)


if __name__ == "__main__":
    sys.exit(main())
