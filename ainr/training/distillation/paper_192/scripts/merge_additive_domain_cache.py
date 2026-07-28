#!/usr/bin/env python3
"""Merge train rows from compatible domain caches and keep base validation fixed."""

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


def materialize(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def parse_source(value: str) -> tuple[str, Path]:
    label, separator, raw_root = value.partition("=")
    if not separator or not label or not raw_root:
        raise argparse.ArgumentTypeError("source must use LABEL=CACHE_ROOT")
    return label, Path(raw_root).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=parse_source,
        action="append",
        required=True,
        metavar="LABEL=CACHE_ROOT",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    base_root = args.base_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sources = [("base", base_root), *args.source]
    labels = [label for label, _ in sources]
    if len(labels) != len(set(labels)):
        raise ValueError("Source labels must be unique")

    documents = {label: load_manifest(root) for label, root in sources}
    teacher_hash = str(documents["base"].get("teacher_checkpoint_sha256", ""))
    preprocessing = str(documents["base"].get("preprocessing", ""))
    for label, document in documents.items():
        if document.get("teacher_checkpoint_sha256") != teacher_hash:
            raise ValueError(f"Teacher checkpoint mismatch for {label}")
        if document.get("preprocessing") != preprocessing:
            raise ValueError(f"Preprocessing mismatch for {label}")
        present = {str(row.get("dataset")) for row in document["records"]}
        if present != {args.dataset}:
            raise ValueError(f"{label} contains unexpected datasets: {sorted(present)}")

    build_root = output_root.with_name(f".{output_root.name}.building")
    if output_root.exists() and not args.replace:
        raise FileExistsError(output_root)
    if build_root.exists():
        if not args.replace:
            raise FileExistsError(build_root)
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    modes: collections.Counter[str] = collections.Counter()
    selected: list[tuple[str, Path, dict[str, Any]]] = []
    for label, root in sources:
        selected.extend(
            (label, root, row)
            for row in documents[label]["records"]
            if str(row.get("split")) == "train" or label == "base"
        )
    try:
        for label, root, source_record in selected:
            record = dict(source_record)
            original_id = str(record.get("id", ""))
            if not original_id:
                raise ValueError(f"Missing record ID in {label}")
            record["id"] = f"{label}_{original_id}"
            record["additive_source"] = label
            for field in ARRAY_FIELDS:
                relative = safe_relative(source_record.get(field), root / "manifest.json")
                source = root / relative
                if not source.is_file():
                    raise FileNotFoundError(source)
                destination_relative = Path(label) / relative
                modes[materialize(source, build_root / destination_relative)] += 1
                record[field] = str(destination_relative)
            records.append(record)

        identifiers = [str(row["id"]) for row in records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Merged cache contains duplicate record IDs")
        scene_splits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
        for row in records:
            scene_splits[(str(row["dataset"]), str(row["scene"]))].add(str(row["split"]))
        leakage = [key for key, splits in scene_splits.items() if len(splits) != 1]
        if leakage:
            raise ValueError(f"Scene leakage after merge: {leakage[0]}")

        records.sort(key=lambda row: (str(row["split"]), str(row["scene"]), str(row["id"])))
        manifest = {
            **{key: value for key, value in documents["base"].items() if key != "records"},
            "additive_sources": {
                label: {
                    "root": str(root),
                    "manifest_sha256": sha256_file(root / "manifest.json"),
                }
                for label, root in sources
            },
            "records": records,
        }
        counts = collections.Counter(
            (str(row["split"]), str(row["dataset"]), str(row["additive_source"]))
            for row in records
        )
        atomic_json(build_root / "manifest.json", manifest)
        atomic_json(
            build_root / "manifest.report.json",
            {
                "records": len(records),
                "counts": {
                    f"{split}/{dataset}/{source}": count
                    for (split, dataset, source), count in sorted(counts.items())
                },
                "scene_groups": len(scene_splits),
                "scene_leakage": 0,
                "array_materialization": dict(sorted(modes.items())),
            },
        )
        if output_root.exists():
            shutil.rmtree(output_root)
        build_root.rename(output_root)
    except BaseException:
        if build_root.exists():
            shutil.rmtree(build_root)
        raise


if __name__ == "__main__":
    main()
