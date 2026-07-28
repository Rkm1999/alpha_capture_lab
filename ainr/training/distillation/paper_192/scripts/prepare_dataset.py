#!/usr/bin/env python3
"""Build fresh exact-192 paired patches and full-precision SCUNet labels."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from common import atomic_json, load_config, resolve_paper_path, seed_everything, sha256_file
from src.scunet_teacher import load_scunet_teacher


def split_for(dataset: str, scene: str, fraction: float) -> str:
    key = f"{dataset}:{scene}".encode("utf-8")
    value = int(hashlib.sha256(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "validation" if value < fraction else "train"


def crop_positions(width: int, height: int, count: int, seed: int) -> list[tuple[int, int]]:
    if width < 192 or height < 192:
        return []
    generator = random.Random(seed)
    return [
        (generator.randint(0, width - 192), generator.randint(0, height - 192))
        for _ in range(count)
    ]


def decoded_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parents[1] / "configs/default.yaml"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete and rebuild an existing paper_192 cache.",
    )
    parser.add_argument("--limit-pairs", type=int, help="Smoke-test only; not for final training.")
    parser.add_argument(
        "--reuse-cache",
        type=Path,
        help="Hard-link matching records from an existing immutable cache.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    source_manifest = resolve_paper_path(config["data"]["source_manifest"])
    source_root = resolve_paper_path(config["data"]["source_root"])
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    output_manifest = resolve_paper_path(config["data"]["manifest"])
    teacher_repo = resolve_paper_path(config["teacher"]["repository"])
    teacher_checkpoint = resolve_paper_path(config["teacher"]["checkpoint"])
    cache_dtype_name = str(config["data"].get("cache_dtype", "float32"))
    if cache_dtype_name not in {"float16", "float32"}:
        raise ValueError("data.cache_dtype must be float16 or float32")
    cache_dtype = np.float16 if cache_dtype_name == "float16" else np.float32
    allowed = set(config["data"]["datasets"])
    seed = int(config["project"]["seed"])
    seed_everything(seed)

    if cache_root.exists():
        if not args.replace:
            raise FileExistsError(f"Cache already exists: {cache_root}; pass --replace to rebuild")
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True)

    source_document = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_records = (
        source_document.get("records")
        if isinstance(source_document, dict)
        else source_document
    )
    if not isinstance(source_records, list):
        raise ValueError("Source manifest must be a JSON list or contain a 'records' list")
    records = [item for item in source_records if item.get("dataset") in allowed]
    if args.limit_pairs is not None:
        records = records[: args.limit_pairs]
    if not records:
        raise RuntimeError("No paired source records matched the configured datasets")

    reused_records: dict[str, tuple[dict, Path]] = {}
    if args.reuse_cache is not None:
        reuse_root = args.reuse_cache.expanduser().resolve()
        reuse_manifest_path = reuse_root / "manifest.json"
        reuse_document = json.loads(reuse_manifest_path.read_text(encoding="utf-8"))
        if reuse_document.get("preprocessing") != config["project"]["preprocessing_version"]:
            raise ValueError("Reuse cache preprocessing does not match")
        if reuse_document.get("source_manifest_sha256") != sha256_file(source_manifest):
            raise ValueError("Reuse cache source manifest does not match")
        if reuse_document.get("teacher_checkpoint_sha256") != sha256_file(
            teacher_checkpoint
        ):
            raise ValueError("Reuse cache teacher checkpoint does not match")
        for record in reuse_document.get("records", []):
            identifier = str(record.get("id", ""))
            if not identifier or identifier in reused_records:
                raise ValueError("Reuse cache has missing or duplicate record IDs")
            reused_records[identifier] = (record, reuse_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required to build the full SCUNet cache")
    teacher = load_scunet_teacher(teacher_repo, teacher_checkpoint, device)
    batch_size = int(config["teacher"]["cache_batch_size"])
    validation_fraction = float(config["data"]["validation_fraction"])
    train_patch_counts = {
        str(dataset): int(count)
        for dataset, count in config["data"]["patches_per_pair"].items()
    }
    validation_patch_counts = {
        str(dataset): int(count)
        for dataset, count in config["data"].get(
            "validation_patches_per_pair", train_patch_counts
        ).items()
    }
    for split, counts in (
        ("train", train_patch_counts),
        ("validation", validation_patch_counts),
    ):
        if set(counts) != allowed:
            raise ValueError(
                f"{split} patches_per_pair must exactly match data.datasets: "
                f"{sorted(counts)} != {sorted(allowed)}"
            )
        if any(count < 1 for count in counts.values()):
            raise ValueError(f"Every {split} patches_per_pair value must be positive")

    pending: list[tuple[dict, np.ndarray, np.ndarray]] = []
    output: list[dict] = []
    reused_count = 0

    def flush() -> None:
        if not pending:
            return
        noisy_batch = np.stack([item[1] for item in pending])
        value = torch.from_numpy(noisy_batch).permute(0, 3, 1, 2).to(device)
        with torch.inference_mode():
            prediction = teacher(value).clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy()
        for (metadata, noisy, clean), target in zip(pending, prediction, strict=True):
            base = Path(metadata["split"]) / metadata["dataset"] / metadata["id"]
            relative_paths = {
                "input": str(base.with_name(base.name + "_input.npy")),
                "clean": str(base.with_name(base.name + "_clean.npy")),
                "teacher": str(base.with_name(base.name + "_teacher.npy")),
            }
            for path in relative_paths.values():
                (cache_root / path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_root / relative_paths["input"], noisy.astype(cache_dtype))
            np.save(cache_root / relative_paths["clean"], clean.astype(cache_dtype))
            np.save(cache_root / relative_paths["teacher"], target.astype(np.float16))
            output.append({**metadata, **relative_paths})
        pending.clear()

    for source in tqdm(records, desc="Preparing paired exact-192 samples"):
        dataset = str(source["dataset"])
        scene = str(source["scene"])
        noisy_path = source_root / source["input"]
        clean_path = source_root / source["clean"]
        noisy_image = decoded_rgb(noisy_path)
        clean_image = decoded_rgb(clean_path)
        if noisy_image.size != clean_image.size:
            raise ValueError(f"Geometry mismatch: {noisy_path} and {clean_path}")
        width, height = noisy_image.size
        source_key = hashlib.sha256(
            f"{dataset}:{scene}:{source['input']}:{source['clean']}".encode("utf-8")
        ).hexdigest()
        split = str(source.get("split") or split_for(dataset, scene, validation_fraction))
        if split not in {"train", "validation"}:
            raise ValueError(f"Invalid split {split!r} for {dataset}/{scene}")
        patch_counts = (
            train_patch_counts if split == "train" else validation_patch_counts
        )
        count = patch_counts[dataset]
        crop_identity = (
            f"{dataset}:{scene}:{source['clean']}"
            if config["data"].get("scene_stable_crops", False)
            else f"{dataset}:{scene}:{source['input']}:{source['clean']}"
        )
        crop_seed = int(hashlib.sha256(crop_identity.encode("utf-8")).hexdigest()[:8], 16)
        fixed_positions = [tuple(map(int, value)) for value in source.get("fixed_crops", [])]
        invalid_fixed = [
            value
            for value in fixed_positions
            if len(value) != 2
            or value[0] < 0
            or value[1] < 0
            or value[0] + 192 > width
            or value[1] + 192 > height
        ]
        if invalid_fixed:
            raise ValueError(f"Invalid fixed crop {invalid_fixed[0]} for {noisy_path}")
        positions = list(dict.fromkeys(fixed_positions))[:count]
        generated_positions = crop_positions(
            width,
            height,
            max(count * 4, count),
            seed ^ crop_seed,
        )
        positions.extend(value for value in generated_positions if value not in positions)
        positions = positions[:count]
        if len(positions) != count:
            raise ValueError(f"Source is smaller than 192 px: {noisy_path}")
        source_metadata = {
            key: source[key]
            for key in (
                "camera",
                "clean_level",
                "domain",
                "iso",
                "license_status",
                "noise_level",
                "reference_url",
                "source_url",
                "supervision",
            )
            if key in source
        }
        for index, (left, top) in enumerate(positions):
            box = (left, top, left + 192, top + 192)
            noisy = np.asarray(noisy_image.crop(box), dtype=np.float32) / 255.0
            clean = np.asarray(clean_image.crop(box), dtype=np.float32) / 255.0
            metadata = {
                "id": f"{source_key[:16]}_{index:02d}",
                "dataset": dataset,
                "scene": scene,
                "split": split,
                "source_input": str(source["input"]),
                "source_clean": str(source["clean"]),
                "crop": [left, top, 192, 192],
                **source_metadata,
            }
            reusable = reused_records.get(metadata["id"])
            if reusable is not None:
                reused, reuse_root = reusable
                identity_fields = (
                    "dataset",
                    "scene",
                    "split",
                    "source_input",
                    "source_clean",
                    "crop",
                )
                if any(reused.get(field) != metadata.get(field) for field in identity_fields):
                    raise ValueError(
                        f"Reuse record identity mismatch: {metadata['id']}"
                    )
                relative_paths = {
                    field: str(
                        Path(metadata["split"])
                        / metadata["dataset"]
                        / f"{metadata['id']}_{field}.npy"
                    )
                    for field in ("input", "clean", "teacher")
                }
                for field, relative in relative_paths.items():
                    source_path = reuse_root / str(reused[field])
                    destination = cache_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source_path, destination)
                output.append({**metadata, **relative_paths})
                reused_count += 1
                continue
            pending.append((metadata, noisy, clean))
            if len(pending) >= batch_size:
                flush()
    flush()

    output.sort(key=lambda item: (item["split"], item["dataset"], item["id"]))
    scene_splits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    for record in output:
        scene_splits[(record["dataset"], record["scene"])].add(record["split"])
        counts[(record["split"], record["dataset"])] += 1
    leakage = [key for key, splits in scene_splits.items() if len(splits) != 1]
    if leakage:
        raise RuntimeError(f"Scene leakage found after generation: {leakage[0]}")

    payload = {
        "schema_version": 1,
        "preprocessing": config["project"]["preprocessing_version"],
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "data_root": str(cache_root),
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "teacher_inference_dtype": "float32",
        "teacher_cache_dtype": "float16",
        "teacher_generation_environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "batch_size": batch_size,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "seed": seed,
        },
        "records": output,
    }
    atomic_json(output_manifest, payload)
    report = {
        "manifest": str(output_manifest),
        "records": len(output),
        "source_pairs": len(records),
        "counts": {f"{split}/{dataset}": count for (split, dataset), count in sorted(counts.items())},
        "scene_groups": len(scene_splits),
        "scene_leakage": 0,
        "reused_records": reused_count,
        "teacher_inference_dtype": "float32",
        "input_cache_dtype": (
            f"mixed(float32,{cache_dtype_name})" if reused_count else cache_dtype_name
        ),
        "clean_cache_dtype": (
            f"mixed(float32,{cache_dtype_name})" if reused_count else cache_dtype_name
        ),
        "teacher_cache_dtype": "float16",
    }
    atomic_json(output_manifest.with_suffix(".report.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
