#!/usr/bin/env python3
"""Build the Sony-adjacent JPEG and high-noise NIND source manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from PIL import ExifTags, Image, ImageOps

from common import atomic_json, resolve_paper_path


HIGH_NIND_LEVEL = re.compile(r"_ISO(6400(?:-\d+)?|H[1-4])\.", re.IGNORECASE)
POLYU_ISO = {
    "book": 1600,
    "class": 1600,
    "compu": 3200,
    "door": 3200,
    "plant": 3200,
    "stair": 1600,
    "toy": 1600,
    "water": 6400,
}
POLYU_VALIDATION_SCENES = {"compu", "water"}
NIND_GATE_CROPS = {
    "nind_books": [960, 1344],
    "nind_chapel": [1152, 384],
    "nind_claytools": [3072, 1728],
    "nind_parking-keyboard": [2880, 960],
    "nind_shells": [576, 768],
    "nind_whistle": [3264, 384],
}
CAMERA_MODEL_TAG = next(key for key, value in ExifTags.TAGS.items() if value == "Model")


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).size


def camera_model(path: Path) -> str | None:
    with Image.open(path) as image:
        value = image.getexif().get(CAMERA_MODEL_TAG)
    return str(value).strip() if value else None


def relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def polyu_sony_pairs(source_root: Path) -> list[dict]:
    root = source_root / "polyu" / "OriginalImages"
    records = []
    for noisy in sorted(root.glob("SonyA7II_*_Real.JPG")):
        scene_name = noisy.stem.removeprefix("SonyA7II_").removesuffix("_Real")
        clean = noisy.with_name(noisy.name.replace("_Real.JPG", "_mean.JPG"))
        if scene_name not in POLYU_ISO or not clean.is_file():
            continue
        if image_size(noisy) != image_size(clean):
            raise ValueError(f"PolyU geometry mismatch: {noisy} and {clean}")
        iso = POLYU_ISO[scene_name]
        records.append(
            {
                "dataset": "polyu_sony",
                "scene": f"polyu_sony_{scene_name}",
                "input": relative(noisy, source_root),
                "clean": relative(clean, source_root),
                "camera": "Sony A7 II",
                "iso": iso,
                "noise_level": str(iso),
                "clean_level": "burst mean",
                "domain": "camera-processed JPEG",
                "supervision": "paired",
                "license_status": "separate permission reported by project owner",
                "split": "validation" if scene_name in POLYU_VALIDATION_SCENES else "train",
            }
        )
    if len(records) != len(POLYU_ISO):
        raise RuntimeError(f"Expected {len(POLYU_ISO)} PolyU Sony pairs, found {len(records)}")
    return records


def nind_high_pairs(source_root: Path, extended_manifest: Path) -> list[dict]:
    document = json.loads(extended_manifest.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise ValueError("The extended source manifest must be a JSON list")
    attribution_path = source_root / "nind" / "ATTRIBUTION.json"
    attribution = {
        str(row["file"]): row
        for row in json.loads(attribution_path.read_text(encoding="utf-8"))
    }
    records = []
    for source in document:
        if source.get("dataset") != "nind":
            continue
        if "mvb-sainte-anne" in str(source.get("scene", "")).lower():
            # These are the only local CC BY-SA noisy variants. Keep the gate's
            # redistribution terms simple by using the CC BY/public-domain subset.
            continue
        match = HIGH_NIND_LEVEL.search(str(source.get("input", "")))
        if not match:
            continue
        noisy = source_root / source["input"]
        clean = source_root / source["clean"]
        plain_base = clean.with_name(re.sub(r"_ISO\d+(?:-\d+)?\.", "_ISO200.", clean.name))
        if plain_base.is_file():
            clean = plain_base
        if image_size(noisy) != image_size(clean):
            continue
        noise_level = match.group(1).upper()
        model = camera_model(noisy) or camera_model(clean) or "unknown"
        noisy_attribution = attribution.get(relative(noisy, source_root / "nind"))
        clean_attribution = attribution.get(relative(clean, source_root / "nind"))
        if noisy_attribution is None or clean_attribution is None:
            raise RuntimeError(f"NIND attribution is missing for {noisy} or {clean}")
        records.append(
            {
                **source,
                "clean": relative(clean, source_root),
                "camera": model,
                "iso": 6400 if noise_level.startswith("6400") else None,
                "noise_level": noise_level,
                "clean_level": re.search(
                    r"_ISO([^.]*)\.", str(source["clean"]), re.IGNORECASE
                ).group(1).upper(),
                "domain": "raw-developed aligned sRGB",
                "supervision": "teacher_only",
                "license_status": (
                    f"input {noisy_attribution['license']}; "
                    f"reference {clean_attribution['license']}"
                ),
                "source_url": noisy_attribution["source"],
                "reference_url": clean_attribution["source"],
                "fixed_crops": (
                    [NIND_GATE_CROPS[source["scene"]]]
                    if source["scene"] in NIND_GATE_CROPS
                    else []
                ),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=resolve_paper_path("../data/sources"),
    )
    parser.add_argument(
        "--extended-manifest",
        type=Path,
        default=resolve_paper_path("../data/extended_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "data/high_iso_source_manifest.json",
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    records = polyu_sony_pairs(source_root) + nind_high_pairs(
        source_root, args.extended_manifest.resolve()
    )
    records.sort(key=lambda row: (row["dataset"], row["scene"], row["noise_level"]))
    payload = {
        "schema_version": 1,
        "purpose": "Sony-adjacent JPEG and high-noise exact-192 data gate",
        "polyu_permission": "Separate permission reported by project owner; archive its terms externally.",
        "nind_h_semantics": (
            "H1-H4 are exposure-corrected underexposures captured at the camera's "
            "maximum ISO, not literal ISO 12800-51200 labels."
        ),
        "records": records,
    }
    atomic_json(args.output.resolve(), payload)
    counts = Counter((row["dataset"], row["noise_level"]) for row in records)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "records": len(records),
                "scenes": len({(row["dataset"], row["scene"]) for row in records}),
                "counts": {f"{dataset}/{level}": count for (dataset, level), count in sorted(counts.items())},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
