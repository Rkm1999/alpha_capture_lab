#!/usr/bin/env python3
"""Merge independently downloaded SNIC archive manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import atomic_json, resolve_paper_path
DEFAULT_ARCHIVES = {
    13237248: "sony_a7r_iii_indoor1.zip",
    13237336: "sony_a7r_iii_indoor2.zip",
    13242871: "sony_a7r_iii_outdoor1.zip",
    13243019: "sony_a7r_iii_outdoor2.zip",
    13237440: "sony_a7r_iii_outdoor3.zip",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=resolve_paper_path("../data/sources/snic/sony_a7r_iii"),
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    documents = []
    for file_id in DEFAULT_ARCHIVES:
        path = root / f"download_manifest.{file_id}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("complete") is not True:
            raise RuntimeError(f"SNIC archive manifest is incomplete: {path}")
        archives = document.get("archives")
        if not isinstance(archives, list) or len(archives) != 1:
            raise RuntimeError(f"Expected one archive in {path}")
        if int(archives[0]["datafile_id"]) != file_id:
            raise RuntimeError(f"SNIC archive ID mismatch in {path}")
        documents.append(document)
    template = documents[0]
    invariant_keys = (
        "schema_version",
        "dataset",
        "doi",
        "license",
        "camera",
        "integrity",
        "noisy_isos",
    )
    for document in documents[1:]:
        for key in invariant_keys:
            if document.get(key) != template.get(key):
                raise RuntimeError(f"SNIC manifest mismatch for {key}")
    merged = {
        key: template[key] for key in invariant_keys
    }
    merged["complete"] = True
    merged["archives"] = sorted(
        (document["archives"][0] for document in documents),
        key=lambda row: int(row["datafile_id"]),
    )
    atomic_json(root / "download_manifest.json", merged)
    print(
        json.dumps(
            {
                "manifest": str(root / "download_manifest.json"),
                "archives": len(merged["archives"]),
                "entries": sum(int(row["entries"]) for row in merged["archives"]),
                "uncompressed_bytes": sum(
                    int(row["uncompressed_bytes"]) for row in merged["archives"]
                ),
                "compressed_bytes": sum(
                    int(row["compressed_bytes"]) for row in merged["archives"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
