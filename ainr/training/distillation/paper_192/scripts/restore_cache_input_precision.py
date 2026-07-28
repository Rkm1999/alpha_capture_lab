#!/usr/bin/env python3
"""Restore FP16 cache inputs from source images as exact decoded FP32 crops."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    cache_root = args.cache_root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    manifest_path = cache_root / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))

    grouped: dict[str, list[dict]] = defaultdict(list)
    already_fp32 = 0
    for record in document["records"]:
        path = cache_root / str(record["input"])
        value = np.load(path, mmap_mode="r")
        if value.dtype == np.float32:
            already_fp32 += 1
            continue
        if value.dtype != np.float16:
            raise ValueError(f"Unexpected input dtype {value.dtype}: {path}")
        grouped[str(record["source_input"])].append(record)

    restored = 0
    for relative, records in tqdm(grouped.items(), desc="Restoring FP32 inputs"):
        with Image.open(source_root / relative) as image:
            decoded = ImageOps.exif_transpose(image).convert("RGB")
            for record in records:
                left, top, width, height = map(int, record["crop"])
                if (width, height) != (192, 192):
                    raise ValueError(f"Unexpected crop size: {record['crop']}")
                value = (
                    np.asarray(
                        decoded.crop((left, top, left + width, top + height)),
                        dtype=np.float32,
                    )
                    / 255.0
                )
                destination = cache_root / str(record["input"])
                temporary = destination.with_suffix(".tmp.npy")
                np.save(temporary, value.astype(np.float32))
                os.replace(temporary, destination)
                restored += 1

    document["input_cache_dtype"] = "float32"
    temporary_manifest = manifest_path.with_suffix(".tmp.json")
    temporary_manifest.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    print(
        json.dumps(
            {
                "already_fp32": already_fp32,
                "restored_fp32": restored,
                "source_images": len(grouped),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
