#!/usr/bin/env python3
"""Replace selected datasets in a mixed cache without duplicating tensors."""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
from pathlib import Path
from typing import Any

from common import atomic_json, sha256_file


ARRAY_FIELDS = ("input", "clean", "teacher")


def load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("records"), list):
        raise ValueError(f"Invalid cache manifest: {path}")
    return document


def safe_relative(value: object, manifest: Path) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe array path in {manifest}: {relative}")
    return relative


def link(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def parse_overlay(value: str) -> tuple[str, Path]:
    dataset, separator, root = value.partition("=")
    if not separator or not dataset or not root:
        raise argparse.ArgumentTypeError("overlay must use DATASET=CACHE_ROOT")
    return dataset, Path(root).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument(
        "--overlay",
        type=parse_overlay,
        action="append",
        required=True,
        metavar="DATASET=CACHE_ROOT",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preprocessing", required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    base_root = args.base_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    build_root = output_root.with_name(f".{output_root.name}.building")
    backup_root = output_root.with_name(f".{output_root.name}.previous")
    overlays = dict(args.overlay)
    if len(overlays) != len(args.overlay):
        raise ValueError("Each overlay dataset may be specified only once")

    base = load_manifest(base_root)
    teacher_hash = str(base.get("teacher_checkpoint_sha256", ""))
    if len(teacher_hash) != 64:
        raise ValueError("Base manifest has no teacher checkpoint identity")
    overlay_documents = {dataset: load_manifest(root) for dataset, root in overlays.items()}
    for dataset, document in overlay_documents.items():
        if document.get("teacher_checkpoint_sha256") != teacher_hash:
            raise ValueError(f"Teacher checkpoint mismatch for {dataset}")
        present = {str(row.get("dataset")) for row in document["records"]}
        if present != {dataset}:
            raise ValueError(
                f"Overlay cache for {dataset} contains unexpected datasets: {sorted(present)}"
            )

    if output_root.exists() and not args.replace:
        raise FileExistsError(output_root)
    if build_root.exists():
        if not args.replace:
            raise FileExistsError(build_root)
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    materialization: collections.Counter[str] = collections.Counter()
    sources = [
        (
            "base",
            base_root,
            [
                row
                for row in base["records"]
                if not (
                    str(row.get("dataset")) in overlays
                    and str(row.get("split")) == "train"
                )
            ],
        ),
        *[
            (
                f"overlay_{dataset}",
                root,
                [
                    row
                    for row in overlay_documents[dataset]["records"]
                    if str(row.get("split")) == "train"
                ],
            )
            for dataset, root in sorted(overlays.items())
        ],
    ]
    try:
        for namespace, source_root, source_records in sources:
            for source_record in source_records:
                record = dict(source_record)
                if "gt_weight" not in record or "kd_weight" not in record:
                    if record.get("supervision") != "teacher_only":
                        raise ValueError(
                            "Only teacher-only overlay records may omit supervision weights: "
                            f"{record.get('dataset')}/{record.get('id')}"
                        )
                    record["gt_weight"] = 0.0
                    record["kd_weight"] = 1.0
                for field in ARRAY_FIELDS:
                    relative = safe_relative(
                        source_record.get(field), source_root / "manifest.json"
                    )
                    source = source_root / relative
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    # Preserve the accepted synthetic namespace exactly.
                    destination_relative = (
                        relative if namespace == "base" else Path(namespace) / relative
                    )
                    materialization[link(source, build_root / destination_relative)] += 1
                    record[field] = str(destination_relative)
                if namespace != "base":
                    record["mixed_source"] = namespace
                    record["mixed_source_index"] = len(records)
                records.append(record)

        identifiers = [str(row.get("id", "")) for row in records]
        if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError("Combined cache contains missing or duplicate record IDs")
        scene_splits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
        for row in records:
            scene_splits[(str(row["dataset"]), str(row["scene"]))].add(str(row["split"]))
        leakage = [key for key, splits in scene_splits.items() if len(splits) != 1]
        if leakage:
            raise ValueError(f"Scene leakage after overlay: {leakage[0]}")

        records.sort(
            key=lambda row: (
                str(row["split"]),
                str(row["dataset"]),
                str(row["scene"]),
                str(row["input"]),
            )
        )
        counts = collections.Counter((str(row["split"]), str(row["dataset"])) for row in records)
        document = {
            **{key: value for key, value in base.items() if key != "records"},
            "preprocessing": args.preprocessing,
            "overlay_sources": {
                dataset: {
                    "root": str(root),
                    "manifest_sha256": sha256_file(root / "manifest.json"),
                    "preprocessing": overlay_documents[dataset].get("preprocessing"),
                    "records": len(overlay_documents[dataset]["records"]),
                }
                for dataset, root in sorted(overlays.items())
            },
            "records": records,
        }
        atomic_json(build_root / "manifest.json", document)
        report = {
            "manifest": str(output_root / "manifest.json"),
            "records": len(records),
            "counts": {
                f"{split}/{dataset}": count
                for (split, dataset), count in sorted(counts.items())
            },
            "scene_groups": len(scene_splits),
            "scene_leakage": 0,
            "array_materialization": dict(sorted(materialization.items())),
            "replaced_datasets": sorted(overlays),
        }
        atomic_json(build_root / "manifest.report.json", report)

        if backup_root.exists():
            shutil.rmtree(backup_root)
        if output_root.exists():
            output_root.rename(backup_root)
        build_root.rename(output_root)
        if backup_root.exists():
            shutil.rmtree(backup_root)
        print(json.dumps(report, indent=2))
    except BaseException:
        if build_root.exists():
            shutil.rmtree(build_root)
        raise


if __name__ == "__main__":
    main()
