#!/usr/bin/env python3
"""Prepare audited exact-192 crops for the general-camera paired RGB run."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from common import atomic_json, load_config, resolve_paper_path, seed_everything, sha256_file
from prepare_domain_dataset import alignment_gate, luminance
from src.scunet_teacher import load_scunet_teacher


TILE = 192
ARRAY_FIELDS = ("input", "clean", "teacher")


def contained_source(root: Path, value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Source path must stay inside the source root: {relative}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Source path escapes the source root: {relative}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def decode_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def decode_mask(path: Path, expected_size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        mask = ImageOps.exif_transpose(image).convert("L").copy()
    if mask.size != expected_size:
        raise ValueError(
            f"Mask geometry {mask.size} does not match paired images {expected_size}: {path}"
        )
    return mask


def crop_seed(seed: int, source: dict[str, Any]) -> int:
    identity = ":".join(
        str(source.get(key, "")) for key in ("dataset", "camera", "scene", "input", "clean")
    )
    return seed ^ int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)


def stratified_positions(
    image: Image.Image,
    count: int,
    candidates: int,
    seed: int,
    mask: Image.Image | None,
    minimum_mask_valid_fraction: float,
) -> list[tuple[int, int, float, float]]:
    width, height = image.size
    if width < TILE or height < TILE:
        raise ValueError(f"Source geometry {width}x{height} is smaller than {TILE}")
    thumbnail_width = min(512, width)
    thumbnail_height = max(1, round(height * thumbnail_width / width))
    thumbnail = np.asarray(
        image.resize((thumbnail_width, thumbnail_height), Image.Resampling.BOX),
        dtype=np.float32,
    ) / 255.0
    generator = random.Random(seed)
    options: list[tuple[int, int, float, float]] = []
    attempts = max(candidates, count) * (8 if mask is not None else 1)
    for _ in range(attempts):
        left = generator.randint(0, width - TILE)
        top = generator.randint(0, height - TILE)
        valid_fraction = 1.0
        if mask is not None:
            mask_crop = np.asarray(
                mask.crop((left, top, left + TILE, top + TILE)), dtype=np.uint8
            )
            valid_fraction = float(np.count_nonzero(mask_crop >= 128) / mask_crop.size)
            if valid_fraction < minimum_mask_valid_fraction:
                continue
        x0 = min(thumbnail_width - 1, int(left * thumbnail_width / width))
        y0 = min(thumbnail_height - 1, int(top * thumbnail_height / height))
        x1 = max(
            x0 + 1,
            min(thumbnail_width, math.ceil((left + TILE) * thumbnail_width / width)),
        )
        y1 = max(
            y0 + 1,
            min(thumbnail_height, math.ceil((top + TILE) * thumbnail_height / height)),
        )
        mean = float(luminance(thumbnail[y0:y1, x0:x1]).mean())
        options.append((left, top, mean, valid_fraction))
        if len(options) >= max(candidates, count):
            break
    if len(options) < count:
        raise RuntimeError(
            f"Only {len(options)} valid crop positions remain after mask filtering; "
            f"need {count} at minimum fraction {minimum_mask_valid_fraction}"
        )
    options.sort(key=lambda value: value[2])
    chosen: list[tuple[int, int, float, float]] = []
    used: set[tuple[int, int]] = set()
    for quantile in np.linspace(0.03, 0.94, count):
        target = round(float(quantile) * (len(options) - 1))
        order = sorted(range(len(options)), key=lambda index: abs(index - target))
        selected = next(options[index] for index in order if options[index][:2] not in used)
        chosen.append(selected)
        used.add(selected[:2])
    return chosen


def validate_source_manifest(
    document: Any,
    source_root: Path,
    datasets: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("General source manifest must be a schema-version 1 object")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("General source manifest has no records")
    required = {"dataset", "camera", "scene", "split", "input", "clean"}
    identities: set[tuple[str, str, str]] = set()
    scene_splits: dict[tuple[str, str, str], set[str]] = collections.defaultdict(set)
    found: collections.Counter[tuple[str, str]] = collections.Counter()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Source record {index} is not an object")
        missing = required - set(record)
        if missing:
            raise ValueError(f"Source record {index} is missing {sorted(missing)}")
        dataset = str(record["dataset"])
        if dataset not in datasets:
            raise ValueError(f"Source record {index} has unconfigured dataset {dataset!r}")
        split = str(record["split"])
        if split not in {"train", "validation"}:
            raise ValueError(f"Source record {index} has invalid split {split!r}")
        noisy = contained_source(source_root, record["input"])
        clean = contained_source(source_root, record["clean"])
        if record.get("mask") is not None:
            contained_source(source_root, record["mask"])
        identity = (dataset, str(noisy), str(clean))
        if identity in identities:
            raise ValueError(f"Duplicate source pair: {identity}")
        identities.add(identity)
        key = (dataset, str(record["camera"]), str(record["scene"]))
        scene_splits[key].add(split)
        found[(split, dataset)] += 1
    leakage = [key for key, splits in scene_splits.items() if len(splits) != 1]
    if leakage:
        raise RuntimeError(f"Scene leakage across source splits: {leakage[0]}")
    for dataset in datasets:
        for split in ("train", "validation"):
            if found[(split, dataset)] == 0:
                raise RuntimeError(f"Source manifest has no {split} records for {dataset}")
    return records


def camera_scene_weights(records: list[dict[str, Any]]) -> None:
    """Give each camera and then each scene equal influence within a dataset."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        if record["split"] == "train":
            grouped[(record["dataset"], record["camera"])].append(record)
        else:
            record["sample_weight"] = 1.0
    cameras_per_dataset = collections.Counter(dataset for dataset, _ in grouped)
    for (dataset, _camera), camera_records in grouped.items():
        by_scene: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for record in camera_records:
            by_scene[record["scene"]].append(record)
        camera_factor = 1.0 / cameras_per_dataset[dataset]
        for scene_records in by_scene.values():
            weight = camera_factor / len(by_scene) / len(scene_records)
            for record in scene_records:
                record["sample_weight"] = weight


def prepare_source(
    source: dict[str, Any],
    *,
    source_root: Path,
    patch_counts: dict[str, dict[str, int]],
    candidates: int,
    seed: int,
    minimum_mask_valid_fraction: float,
    gate_config: dict[str, Any],
    alpha: float,
) -> tuple[list[tuple[dict[str, Any], np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    """Decode and crop one pair without touching CUDA or shared output state."""

    dataset = str(source["dataset"])
    noisy_path = contained_source(source_root, source["input"])
    clean_path = contained_source(source_root, source["clean"])
    failures: list[dict[str, Any]] = []
    failure_identity = {
        "dataset": dataset,
        "split": str(source["split"]),
        "camera": str(source["camera"]),
        "scene": str(source["scene"]),
    }
    try:
        noisy_image = decode_rgb(noisy_path)
        clean_image = decode_rgb(clean_path)
    except Exception as error:
        return [], [
            {
                **failure_identity,
                "input": str(source["input"]),
                "clean": str(source["clean"]),
                "reason": f"{type(error).__name__}: {error}",
            }
        ]
    if noisy_image.size != clean_image.size:
        return [], [
            {
                **failure_identity,
                "input": str(source["input"]),
                "clean": str(source["clean"]),
                "reason": f"geometry_mismatch: {noisy_image.size} vs {clean_image.size}",
            }
        ]
    mask_path = (
        contained_source(source_root, source["mask"])
        if source.get("mask") is not None
        else None
    )
    try:
        mask_image = decode_mask(mask_path, noisy_image.size) if mask_path is not None else None
    except Exception as error:
        failures.append(
            {
                **failure_identity,
                "mask": str(source.get("mask")),
                "reason": f"mask_{type(error).__name__}: {error}",
            }
        )
        mask_image = None
    positions = stratified_positions(
        noisy_image,
        patch_counts[str(source["split"])][dataset],
        candidates,
        crop_seed(seed, source),
        mask_image,
        minimum_mask_valid_fraction,
    )
    pair_hash = hashlib.sha256(
        f"{dataset}:{source['camera']}:{source['scene']}:{source['input']}:{source['clean']}".encode()
    ).hexdigest()
    prepared: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    for index, (left, top, mean_luminance, mask_valid_fraction) in enumerate(positions):
        box = (left, top, left + TILE, top + TILE)
        noisy = np.asarray(noisy_image.crop(box), dtype=np.float32) / 255.0
        clean = np.asarray(clean_image.crop(box), dtype=np.float32) / 255.0
        gate = alignment_gate(noisy, clean, gate_config)
        paired = bool(gate["passed"])
        metadata = {
            "id": f"{pair_hash[:20]}_{index:02d}",
            "dataset": dataset,
            "camera": str(source["camera"]),
            "scene": str(source["scene"]),
            "split": str(source["split"]),
            "source_input": str(source["input"]),
            "source_clean": str(source["clean"]),
            "crop": [left, top, TILE, TILE],
            "crop_mean_luminance": mean_luminance,
            "mask_valid_fraction": mask_valid_fraction,
            "supervision": "paired" if paired else "teacher_only_alignment_rejected",
            "gt_weight": 1.0 if paired else 0.0,
            "kd_weight": alpha if paired else 1.0,
            "alignment_gate": gate,
        }
        for field in ("iso", "noise_level", "domain", "license_status", "source_url"):
            if field in source and source[field] is not None:
                metadata[field] = source[field]
        if mask_path is not None:
            metadata["source_mask"] = str(source["mask"])
        prepared.append((metadata, noisy, clean))
    return prepared, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs/general_camera_v1.yaml",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--limit-pairs", type=int, help="Smoke only; selects pairs per dataset.")
    parser.add_argument("--decode-workers", type=int, default=8)
    args = parser.parse_args()
    if args.decode_workers < 1:
        parser.error("--decode-workers must be positive")

    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    source_root = resolve_paper_path(config["data"]["source_root"])
    source_manifest = resolve_paper_path(config["data"]["source_manifest"])
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    output_manifest = resolve_paper_path(config["data"]["manifest"])
    datasets = set(map(str, config["data"]["datasets"]))
    if int(config["data"]["tile_size"]) != TILE:
        raise ValueError(f"This pipeline requires tile_size={TILE}")
    source_document = json.loads(source_manifest.read_text(encoding="utf-8"))
    sources = validate_source_manifest(source_document, source_root, datasets)
    if args.limit_pairs is not None:
        if args.limit_pairs < 1:
            parser.error("--limit-pairs must be positive")
        limited: list[dict[str, Any]] = []
        counts: collections.Counter[tuple[str, str]] = collections.Counter()
        for source in sources:
            dataset = str(source["dataset"])
            split = str(source["split"])
            key = (dataset, split)
            if counts[key] < args.limit_pairs:
                limited.append(source)
                counts[key] += 1
        sources = limited

    if cache_root.exists():
        if not args.replace:
            raise FileExistsError(f"Cache already exists: {cache_root}; pass --replace")
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required to generate the frozen-teacher cache")
    teacher_checkpoint = resolve_paper_path(config["teacher"]["checkpoint"])
    teacher = load_scunet_teacher(
        resolve_paper_path(config["teacher"]["repository"]), teacher_checkpoint, device
    )
    batch_size = int(config["teacher"]["cache_batch_size"])
    train_patch_counts = {
        key: int(value) for key, value in config["data"]["patches_per_pair"].items()
    }
    validation_patch_counts = {
        key: int(value)
        for key, value in config["data"].get(
            "validation_patches_per_pair", train_patch_counts
        ).items()
    }
    patch_counts = {
        "train": train_patch_counts,
        "validation": validation_patch_counts,
    }
    for split, counts_for_split in patch_counts.items():
        if set(counts_for_split) != datasets:
            raise ValueError(
                f"{split} patches_per_pair must exactly match data.datasets: "
                f"patches={sorted(counts_for_split)}, datasets={sorted(datasets)}"
            )
        if any(count < 1 for count in counts_for_split.values()):
            raise ValueError(f"Every {split} patches_per_pair value must be positive")
    candidates = int(config["data"]["crop_candidates"])
    if candidates < max(
        count for counts_for_split in patch_counts.values() for count in counts_for_split.values()
    ):
        raise ValueError("crop_candidates must be at least the largest patches_per_pair value")
    minimum_mask_valid_fraction = float(config["data"]["minimum_mask_valid_fraction"])
    if not 0.0 < minimum_mask_valid_fraction <= 1.0:
        raise ValueError("minimum_mask_valid_fraction must be in (0, 1]")
    alpha = float(config["training"]["alpha"])
    gate_config = config["data"]["alignment_gate"]
    output: list[dict[str, Any]] = []
    skipped_sources: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []

    def flush() -> None:
        if not pending:
            return
        noisy_batch = np.stack([item[1] for item in pending])
        value = torch.from_numpy(noisy_batch).permute(0, 3, 1, 2).to(device)
        with torch.inference_mode():
            prediction = teacher(value).clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy()
        for (metadata, noisy, clean), target in zip(pending, prediction, strict=True):
            base = Path(metadata["split"]) / metadata["dataset"] / metadata["id"]
            paths = {
                "input": str(base.with_name(base.name + "_input.npy")),
                "clean": str(base.with_name(base.name + "_clean.npy")),
                "teacher": str(base.with_name(base.name + "_teacher.npy")),
            }
            arrays = {
                "input": noisy.astype(np.float32),
                "clean": clean.astype(np.float32),
                "teacher": target.astype(np.float16),
            }
            hashes: dict[str, str] = {}
            for field in ARRAY_FIELDS:
                destination = cache_root / paths[field]
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.save(destination, arrays[field])
                hashes[field] = sha256_file(destination)
            output.append({**metadata, **paths, "array_sha256": hashes})
        pending.clear()

    worker_args = {
        "source_root": source_root,
        "patch_counts": patch_counts,
        "candidates": candidates,
        "seed": seed,
        "minimum_mask_valid_fraction": minimum_mask_valid_fraction,
        "gate_config": gate_config,
        "alpha": alpha,
    }
    source_iterator = iter(sources)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.decode_workers) as executor:
        futures: collections.deque[concurrent.futures.Future] = collections.deque()
        for _ in range(min(len(sources), args.decode_workers * 2)):
            source = next(source_iterator, None)
            if source is not None:
                futures.append(executor.submit(prepare_source, source, **worker_args))
        with tqdm(total=len(sources), desc="Preparing general-camera pairs") as progress:
            while futures:
                prepared, failures = futures.popleft().result()
                skipped_sources.extend(failures)
                for item in prepared:
                    pending.append(item)
                    if len(pending) >= batch_size:
                        flush()
                progress.update(1)
                source = next(source_iterator, None)
                if source is not None:
                    futures.append(executor.submit(prepare_source, source, **worker_args))
    flush()
    camera_scene_weights(output)
    output.sort(key=lambda row: (row["split"], row["dataset"], row["camera"], row["id"]))

    counts = collections.Counter((row["split"], row["dataset"]) for row in output)
    paired_counts = collections.Counter(
        (row["split"], row["dataset"], row["supervision"]) for row in output
    )
    payload = {
        "schema_version": 2,
        "preprocessing": config["project"]["preprocessing_version"],
        "source_preprocessing": config["data"]["source_preprocessing"],
        "sources": {
            "general": {
                "preprocessing": config["data"]["source_preprocessing"]["general"],
                "manifest": str(source_manifest),
                "manifest_sha256": sha256_file(source_manifest),
                "records": len(sources),
            }
        },
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "teacher_inference_dtype": "float32",
        "records": output,
    }
    atomic_json(output_manifest, payload)
    report = {
        "manifest": str(output_manifest),
        "source_pairs": len(sources),
        "records": len(output),
        "source_failures": len(skipped_sources),
        "skipped_sources": skipped_sources[:200],
        "counts": {
            f"{split}/{dataset}": count for (split, dataset), count in sorted(counts.items())
        },
        "supervision_counts": {
            f"{split}/{dataset}/{supervision}": count
            for (split, dataset, supervision), count in sorted(paired_counts.items())
        },
        "cameras": sorted({row["camera"] for row in output}),
        "array_dtypes": {"input": "float32", "clean": "float32", "teacher": "float16"},
        "smoke": args.limit_pairs is not None,
    }
    atomic_json(output_manifest.with_suffix(".report.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
