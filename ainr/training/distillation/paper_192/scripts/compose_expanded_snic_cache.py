#!/usr/bin/env python3
"""Compose immutable validation tensors with expanded UHD/SNIC training caches."""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
from pathlib import Path

from common import atomic_json, resolve_paper_path, sha256_file


ARRAY_FIELDS = ("input", "clean", "teacher")


def load_manifest(root: Path) -> tuple[dict, list[dict]]:
    path = root / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Manifest has no records: {path}")
    return document, records


def materialize(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-root",
        type=Path,
        default=resolve_paper_path("data/uhd_snic_gate/cache"),
    )
    parser.add_argument(
        "--expanded-snic-root",
        type=Path,
        default=resolve_paper_path("data/snic_gate_v2_snic16/cache"),
    )
    parser.add_argument(
        "--expanded-uhd-root",
        type=Path,
        help="Replace base UHD training records with this UHD-only cache.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=resolve_paper_path("data/uhd_snic_gate_v2_snic16/cache"),
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    base_root = args.base_root.resolve()
    expanded_snic_root = args.expanded_snic_root.resolve()
    expanded_uhd_root = (
        args.expanded_uhd_root.resolve()
        if args.expanded_uhd_root is not None
        else None
    )
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.replace:
            raise FileExistsError(f"Output cache exists: {output_root}; pass --replace")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    base_document, base_records = load_manifest(base_root)
    expanded_snic_document, expanded_snic_records = load_manifest(expanded_snic_root)
    preprocessing = str(base_document.get("preprocessing", ""))
    if (
        not preprocessing
        or expanded_snic_document.get("preprocessing") != preprocessing
    ):
        raise RuntimeError("Base and expanded SNIC preprocessing versions differ")
    teacher_hash = str(base_document.get("teacher_checkpoint_sha256", ""))
    if (
        not teacher_hash
        or expanded_snic_document.get("teacher_checkpoint_sha256") != teacher_hash
    ):
        raise RuntimeError("Base and expanded SNIC teacher checkpoints differ")
    invalid = [
        record
        for record in expanded_snic_records
        if record.get("dataset") != "snic_sony"
    ]
    if invalid:
        raise ValueError("Expanded cache contains a non-SNIC record")
    expanded_uhd_document: dict | None = None
    expanded_uhd_records: list[dict] = []
    if expanded_uhd_root is not None:
        expanded_uhd_document, expanded_uhd_records = load_manifest(expanded_uhd_root)
        if expanded_uhd_document.get("preprocessing") != preprocessing:
            raise RuntimeError("Base and expanded UHD preprocessing versions differ")
        if expanded_uhd_document.get("teacher_checkpoint_sha256") != teacher_hash:
            raise RuntimeError("Base and expanded UHD teacher checkpoints differ")
        invalid = [
            record
            for record in expanded_uhd_records
            if record.get("dataset") != "uhd_ll"
        ]
        if invalid:
            raise ValueError("Expanded cache contains a non-UHD record")

    selected = [
        ("base", base_root, record)
        for record in base_records
        if not (
            record.get("split") == "train"
            and (
                record.get("dataset") == "snic_sony"
                or (
                    expanded_uhd_root is not None
                    and record.get("dataset") == "uhd_ll"
                )
            )
        )
    ]
    selected.extend(
        ("expanded_snic", expanded_snic_root, record)
        for record in expanded_snic_records
        if record.get("split") == "train"
    )
    if expanded_uhd_root is not None:
        selected.extend(
            ("expanded_uhd", expanded_uhd_root, record)
            for record in expanded_uhd_records
            if record.get("split") == "train"
        )
    output: list[dict] = []
    modes: collections.Counter[str] = collections.Counter()
    for namespace, source_root, source_record in selected:
        record = dict(source_record)
        record["composed_source"] = namespace
        for field in ARRAY_FIELDS:
            source = source_root / source_record[field]
            if not source.is_file():
                raise FileNotFoundError(source)
            relative = Path(namespace) / source_record[field]
            modes[materialize(source, output_root / relative)] += 1
            record[field] = str(relative)
        output.append(record)
    output.sort(key=lambda row: (row["split"], row["dataset"], row["scene"], row["input"]))

    scene_splits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    for record in output:
        scene_splits[(str(record["dataset"]), str(record["scene"]))].add(
            str(record["split"])
        )
        counts[(str(record["split"]), str(record["dataset"]))] += 1
    leakage = [key for key, splits in scene_splits.items() if len(splits) != 1]
    if leakage:
        raise RuntimeError(f"Scene leakage in composed cache: {leakage[0]}")

    manifest = output_root / "manifest.json"
    sources = {
        "base": {
            "root": str(base_root),
            "manifest": str(base_root / "manifest.json"),
            "manifest_sha256": sha256_file(base_root / "manifest.json"),
            "records": len(base_records),
        },
        "expanded_snic": {
            "root": str(expanded_snic_root),
            "manifest": str(expanded_snic_root / "manifest.json"),
            "manifest_sha256": sha256_file(expanded_snic_root / "manifest.json"),
            "records": len(expanded_snic_records),
        },
    }
    if expanded_uhd_root is not None:
        sources["expanded_uhd"] = {
            "root": str(expanded_uhd_root),
            "manifest": str(expanded_uhd_root / "manifest.json"),
            "manifest_sha256": sha256_file(expanded_uhd_root / "manifest.json"),
            "records": len(expanded_uhd_records),
        }
    atomic_json(
        manifest,
        {
            "schema_version": 3,
            "purpose": "Expanded UHD/SNIC training coverage with immutable validation tensors",
            "preprocessing": preprocessing,
            "teacher_checkpoint_sha256": teacher_hash,
            "sources": sources,
            "records": output,
        },
    )
    report = {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "records": len(output),
        "counts": {
            f"{split}/{dataset}": count
            for (split, dataset), count in sorted(counts.items())
        },
        "scene_groups": len(scene_splits),
        "scene_leakage": 0,
        "array_materialization": dict(sorted(modes.items())),
        "sources": sources,
    }
    atomic_json(output_root / "manifest.report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
