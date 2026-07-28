#!/usr/bin/env python3
"""Hard-link the public and high-ISO caches behind one stable manifest."""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
from pathlib import Path

from common import atomic_json, resolve_paper_path, sha256_file


ARRAY_FIELDS = ("input", "clean", "teacher")


def read_manifest(path: Path) -> tuple[dict, list[dict]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    records = document.get("records") if isinstance(document, dict) else document
    if not isinstance(records, list):
        raise ValueError(f"Manifest must be a list or contain records: {path}")
    return document if isinstance(document, dict) else {}, records


def hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-root", type=Path, default=resolve_paper_path("data/cache"))
    parser.add_argument(
        "--target-root", type=Path, default=resolve_paper_path("data/high_iso_gate/cache")
    )
    parser.add_argument(
        "--expanded-nind-root",
        type=Path,
        help="Replace target-cache NIND records with this expanded NIND-only cache.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=resolve_paper_path("data/high_iso_ablation/cache")
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    public_root = args.public_root.resolve()
    target_root = args.target_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.replace:
            raise FileExistsError(f"Output cache exists: {output_root}; pass --replace")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    sources = [
        ("public", public_root, public_root / "manifest.json"),
        ("target", target_root, target_root / "manifest.json"),
    ]
    if args.expanded_nind_root is not None:
        expanded_root = args.expanded_nind_root.resolve()
        sources.append(("expanded_nind", expanded_root, expanded_root / "manifest.json"))
    output = []
    link_modes: collections.Counter[str] = collections.Counter()
    source_provenance = {}
    for namespace, source_root, manifest_path in sources:
        document, records = read_manifest(manifest_path)
        source_provenance[namespace] = {
            "root": str(source_root),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "teacher_checkpoint_sha256": document.get("teacher_checkpoint_sha256"),
            "records": len(records),
        }
        for record in records:
            if args.expanded_nind_root is not None:
                if (
                    namespace == "target"
                    and record.get("dataset") == "nind"
                    and record.get("split") == "train"
                ):
                    continue
                if namespace == "expanded_nind" and record.get("dataset") != "nind":
                    raise ValueError(
                        "Expanded NIND cache contains a non-NIND record: "
                        f"{record.get('dataset')!r}"
                    )
                if namespace == "expanded_nind" and record.get("split") != "train":
                    continue
            combined = {**record, "source_cache": namespace}
            combined.setdefault("supervision", "paired")
            for field in ARRAY_FIELDS:
                source = source_root / record[field]
                if not source.is_file():
                    raise FileNotFoundError(source)
                relative = Path(namespace) / record[field]
                link_modes[hardlink_or_copy(source, output_root / relative)] += 1
                combined[field] = str(relative)
            output.append(combined)

    teacher_hashes = {
        value["teacher_checkpoint_sha256"]
        for value in source_provenance.values()
        if value["teacher_checkpoint_sha256"]
    }
    if len(teacher_hashes) != 1:
        raise RuntimeError(f"Source caches use different teacher checkpoints: {teacher_hashes}")
    counts: collections.Counter[tuple[str, str, str]] = collections.Counter()
    scene_splits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for record in output:
        counts[(record["split"], record["dataset"], record["supervision"])] += 1
        scene_splits[(record["dataset"], record["scene"])].add(record["split"])
    leakage = [key for key, splits in scene_splits.items() if len(splits) != 1]
    if leakage:
        raise RuntimeError(f"Scene leakage in combined cache: {leakage[0]}")

    manifest = output_root / "manifest.json"
    payload = {
        "schema_version": 1,
        "purpose": "Controlled NIND teacher-only versus full-reference ablation",
        "preprocessing": "exact_hwc_rgb_float_0_1_v1",
        "teacher_checkpoint_sha256": next(iter(teacher_hashes)),
        "sources": source_provenance,
        "records": output,
    }
    atomic_json(manifest, payload)
    report = {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "records": len(output),
        "scene_groups": len(scene_splits),
        "scene_leakage": 0,
        "counts": {
            f"{split}/{dataset}/{supervision}": count
            for (split, dataset, supervision), count in sorted(counts.items())
        },
        "array_materialization": dict(sorted(link_modes.items())),
        "sources": source_provenance,
    }
    atomic_json(output_root / "manifest.report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
