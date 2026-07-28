#!/usr/bin/env python3
"""Build source records for full UHD-LL and the selected Sony SNIC subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from common import atomic_json, resolve_paper_path


SNIC_NOISY = re.compile(
    r"^(?P<prefix>.+)_(?P<iso>\d{5})_noisy_01\.tiff$",
    re.IGNORECASE,
)
EXPECTED_SNIC_ARCHIVES = {
    13237248: "sony_a7r_iii_indoor1",
    13237336: "sony_a7r_iii_indoor2",
    13242871: "sony_a7r_iii_outdoor1",
    13243019: "sony_a7r_iii_outdoor2",
    13237440: "sony_a7r_iii_outdoor3",
}
EXPECTED_SNIC_ISOS = {1600, 3200, 6400, 12800}


def relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def deterministic_split(dataset: str, scene: str, validation_fraction: float) -> str:
    value = int(hashlib.sha256(f"{dataset}:{scene}".encode()).hexdigest()[:8], 16)
    return "validation" if value / 2**32 < validation_fraction else "train"


def complete_jpeg(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 4:
        return False
    with path.open("rb") as source:
        if source.read(2) != b"\xff\xd8":
            return False
        source.seek(-2, 2)
        return source.read(2) == b"\xff\xd9"


def complete_tiff_header(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 8:
        return False
    with path.open("rb") as source:
        return source.read(4) in {b"II*\x00", b"MM\x00*"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_uhd_names(split_root: Path) -> set[str]:
    list_path = split_root / "data_list.txt"
    if not list_path.is_file():
        raise FileNotFoundError(f"UHD-LL official split list is missing: {list_path}")
    names = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise RuntimeError(f"UHD-LL split list contains duplicate names: {list_path}")
    for name in names:
        if Path(name).name != name or not name.endswith("_UHD_LL.JPG"):
            raise RuntimeError(f"Invalid UHD-LL split-list entry {name!r}: {list_path}")
    return set(names)


def completed_snic_inputs(root: Path) -> list[Path]:
    manifest_path = root / "download_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"SNIC download manifest is missing; finish download_snic_subset.py first: {manifest_path}"
        )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("complete") is not True:
        raise RuntimeError(f"SNIC download is not marked complete: {manifest_path}")
    if document.get("integrity") != "size_and_zip_crc32_verified":
        raise RuntimeError(f"SNIC download has not completed strict CRC verification: {manifest_path}")
    if set(map(int, document.get("noisy_isos", []))) != EXPECTED_SNIC_ISOS:
        raise RuntimeError(
            f"SNIC manifest must contain ISO {sorted(EXPECTED_SNIC_ISOS)}, "
            f"found {document.get('noisy_isos')}"
        )
    archives = document.get("archives")
    if not isinstance(archives, list):
        raise RuntimeError(f"SNIC manifest has no archive list: {manifest_path}")
    by_id = {int(row["datafile_id"]): row for row in archives}
    if set(by_id) != set(EXPECTED_SNIC_ARCHIVES) or len(by_id) != len(archives):
        raise RuntimeError(
            "SNIC manifest must contain each selected archive exactly once; "
            f"expected {sorted(EXPECTED_SNIC_ARCHIVES)}, found {sorted(by_id)}"
        )
    partials = sorted(root.rglob("*.part"))
    if partials:
        raise RuntimeError(f"SNIC download still contains a partial file: {partials[0]}")

    noisy_paths: list[Path] = []
    seen_destinations: set[Path] = set()
    for file_id, expected_stem in EXPECTED_SNIC_ARCHIVES.items():
        archive = by_id[file_id]
        if Path(str(archive.get("name", ""))).stem != expected_stem:
            raise RuntimeError(f"Unexpected SNIC archive name for {file_id}: {archive.get('name')}")
        entries = archive.get("zip_entries")
        if not isinstance(entries, list) or int(archive.get("entries", -1)) != len(entries):
            raise RuntimeError(f"Incomplete SNIC entry report for archive {file_id}")
        states = archive.get("states", {})
        if sum(int(states.get(key, 0)) for key in ("downloaded", "existing")) != len(entries):
            raise RuntimeError(f"Incomplete SNIC extraction state for archive {file_id}")
        archive_root = root / expected_stem
        iso_counts: Counter[int] = Counter()
        clean_count = 0
        for entry in entries:
            member = Path(str(entry["name"]))
            if member.is_absolute() or any(part == ".." for part in member.parts):
                raise RuntimeError(f"Unsafe SNIC archive entry: {member}")
            destination = archive_root / member
            if destination in seen_destinations:
                raise RuntimeError(f"Duplicate SNIC extracted path in manifest: {destination}")
            seen_destinations.add(destination)
            expected_size = int(entry["size"])
            if not destination.is_file() or destination.stat().st_size != expected_size:
                actual = destination.stat().st_size if destination.is_file() else None
                raise RuntimeError(
                    f"Incomplete SNIC file {destination}: expected {expected_size}, found {actual}"
                )
            if not complete_tiff_header(destination):
                raise RuntimeError(f"Invalid SNIC TIFF header: {destination}")
            match = SNIC_NOISY.match(member.as_posix())
            if match and int(match.group("iso")) in EXPECTED_SNIC_ISOS:
                noisy_paths.append(destination)
                iso_counts[int(match.group("iso"))] += 1
            elif member.name.lower().endswith("_00100_clean_01.tiff"):
                clean_count += 1
        if set(iso_counts) != EXPECTED_SNIC_ISOS or len(set(iso_counts.values())) != 1:
            raise RuntimeError(
                f"SNIC archive {file_id} has inconsistent noisy counts by ISO: {dict(iso_counts)}"
            )
        if clean_count != next(iter(iso_counts.values())):
            raise RuntimeError(
                f"SNIC archive {file_id} has {clean_count} clean files for {dict(iso_counts)}"
            )
    if not noisy_paths:
        raise RuntimeError("Completed SNIC manifest contains no selected noisy TIFF inputs")
    return sorted(noisy_paths)


def uhd_records(source_root: Path) -> list[dict]:
    root = source_root / "uhd-ll" / "UHD-LL"
    partials = sorted(root.rglob("*.part"))
    if partials:
        raise RuntimeError(f"UHD-LL download still contains a partial file: {partials[0]}")
    records = []
    for source_split, split in (("training_set", "train"), ("testing_set", "validation")):
        split_root = root / source_split
        input_root = root / source_split / "input"
        gt_root = root / source_split / "gt"
        official_names = official_uhd_names(split_root)
        input_names = {path.name for path in input_root.glob("*_UHD_LL.JPG")}
        gt_names = {path.name for path in gt_root.glob("*_UHD_LL.JPG")}
        if input_names != official_names or gt_names != official_names:
            raise RuntimeError(
                f"UHD-LL {source_split} does not match data_list.txt: "
                f"input missing={len(official_names - input_names)}, input extra={len(input_names - official_names)}, "
                f"GT missing={len(official_names - gt_names)}, GT extra={len(gt_names - official_names)}"
            )
        for name in sorted(official_names):
            noisy = input_root / name
            clean = gt_root / noisy.name
            if not complete_jpeg(noisy) or not complete_jpeg(clean):
                raise RuntimeError(f"UHD-LL pair is not a complete JPEG: {noisy.name}")
            identifier = noisy.stem.removesuffix("_UHD_LL")
            records.append(
                {
                    "dataset": "uhd_ll",
                    "scene": f"uhd_ll_{identifier}",
                    "input": relative(noisy, source_root),
                    "clean": relative(clean, source_root),
                    "split": split,
                    "camera": "Sony a6300 or Sony a7 (public EXIF stripped)",
                    "iso": None,
                    "noise_level": "low-light capture; ISO metadata unavailable",
                    "clean_level": "normal-exposure reference",
                    "domain": "camera-processed 8-bit sRGB JPEG",
                    "supervision": "uhd_hybrid_pair_candidate",
                    "clean_transform": "linear_rgb_local_ratio128_teacher_lowpass8_v2",
                    "license_status": "separate training permission reported by project owner",
                    "source_url": (
                        "https://drive.google.com/drive/folders/"
                        "1IneTwBsSiSSVXGoXQ9_hE1cO2d4Fd4DN"
                    ),
                }
            )
    expected = {"train": 2000, "validation": 150}
    counts = Counter(record["split"] for record in records)
    if dict(counts) != expected:
        raise RuntimeError(f"Expected complete UHD-LL split {expected}, found {dict(counts)}")
    for field in ("input", "clean"):
        hashes: dict[str, set[str]] = {"train": set(), "validation": set()}
        for record in records:
            hashes[record["split"]].add(sha256(source_root / record[field]))
        overlap = hashes["train"] & hashes["validation"]
        if overlap:
            raise RuntimeError(f"UHD-LL has exact {field} payload leakage across train/validation")
    return records


def snic_records(source_root: Path, validation_fraction: float) -> list[dict]:
    root = source_root / "snic" / "sony_a7r_iii"
    records = []
    for noisy in completed_snic_inputs(root):
        match = SNIC_NOISY.match(noisy.name)
        if match is None:
            continue
        clean = noisy.with_name(f"{match.group('prefix')}_00100_clean_01.tiff")
        if not clean.is_file():
            raise FileNotFoundError(f"SNIC clean counterpart is missing: {clean}")
        iso = int(match.group("iso"))
        scene_folder = noisy.parent.relative_to(root).as_posix()
        scene = f"snic_sony_a7r_iii:{scene_folder}"
        records.append(
            {
                "dataset": "snic_sony",
                "scene": scene,
                "input": relative(noisy, source_root),
                "clean": relative(clean, source_root),
                "split": deterministic_split(
                    "snic_sony", scene_folder, validation_fraction
                ),
                "camera": "Sony A7R III",
                "iso": iso,
                "noise_level": str(iso),
                "clean_level": "ISO 100 clean source",
                "domain": "calibrated-noise 16-bit sRGB TIFF",
                "supervision": "paired",
                "clean_transform": "identity",
                "license_status": "MIT",
                "source_url": "https://doi.org/10.7910/DVN/SGHDCP",
            }
        )
    if not records:
        raise RuntimeError(f"No SNIC Sony TIFF pairs found under {root}")
    split_counts = Counter(record["split"] for record in records)
    if set(split_counts) != {"train", "validation"}:
        raise RuntimeError(f"SNIC deterministic split must contain train and validation: {split_counts}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=resolve_paper_path("../data/sources"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=resolve_paper_path("data/uhd_snic_source_manifest.json"),
    )
    parser.add_argument("--snic-validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.0 < args.snic_validation_fraction < 1.0:
        parser.error("--snic-validation-fraction must be between zero and one")
    source_root = args.source_root.resolve()
    records = uhd_records(source_root) + snic_records(
        source_root, args.snic_validation_fraction
    )
    records.sort(key=lambda row: (row["dataset"], row["split"], row["scene"], row["input"]))
    counts = Counter((row["split"], row["dataset"]) for row in records)
    payload = {
        "schema_version": 1,
        "purpose": (
            "Full UHD-LL local-exposure teacher/clean hybrid targets plus selected "
            "Sony SNIC exact-192 paired-reference domain expansion"
        ),
        "uhd_permission": (
            "Separate model-training permission reported by project owner; "
            "archive the written terms outside the repository."
        ),
        "snic_license": "MIT",
        "records": records,
    }
    atomic_json(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "records": len(records),
                "scenes": len({(row["dataset"], row["scene"]) for row in records}),
                "counts": {
                    f"{split}/{dataset}": count
                    for (split, dataset), count in sorted(counts.items())
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
