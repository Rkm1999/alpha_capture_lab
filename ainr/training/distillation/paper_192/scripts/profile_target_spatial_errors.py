#!/usr/bin/env python3
"""Profile spatial sources of weak very-coarse chroma target capture."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(PAPER_ROOT / "scripts"))

from common import atomic_json, load_config, resolve_paper_path, sha256_file  # noqa: E402
from src.mixed_dataset import MixedDistillationDataset  # noqa: E402
from src.noise_conditioning import model_input_from_config  # noqa: E402
from src.student import student_from_checkpoint  # noqa: E402
from train_mixed import chroma_projection, gaussian_blur, target_record_selected  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def luma_gradient(value: torch.Tensor) -> torch.Tensor:
    weights = value.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    luma = (value * weights).sum(1, keepdim=True)
    horizontal = F.pad((luma[..., 1:] - luma[..., :-1]).abs(), (0, 1, 0, 0))
    vertical = F.pad((luma[..., 1:, :] - luma[..., :-1, :]).abs(), (0, 0, 0, 1))
    return 0.5 * (horizontal + vertical)


def border_region_masks(value: torch.Tensor) -> dict[str, torch.Tensor]:
    height, width = value.shape[-2:]
    y = torch.arange(height, device=value.device).view(1, 1, height, 1)
    x = torch.arange(width, device=value.device).view(1, 1, 1, width)
    distance = torch.minimum(
        torch.minimum(y, height - 1 - y),
        torch.minimum(x, width - 1 - x),
    )
    return {
        "border_0_7": distance < 8,
        "border_8_23": (distance >= 8) & (distance < 24),
        "border_24_47": (distance >= 24) & (distance < 48),
        "interior_48_plus": distance >= 48,
    }


def quantile_masks(value: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
    flattened = value.flatten(1)
    q50 = torch.quantile(flattened, 0.50, dim=1).view(-1, 1, 1, 1)
    q75 = torch.quantile(flattened, 0.75, dim=1).view(-1, 1, 1, 1)
    q90 = torch.quantile(flattened, 0.90, dim=1).view(-1, 1, 1, 1)
    return {
        f"{prefix}_bottom_50": value <= q50,
        f"{prefix}_50_75": (value > q50) & (value <= q75),
        f"{prefix}_75_90": (value > q75) & (value <= q90),
        f"{prefix}_top_10": value > q90,
    }


def capture_per_sample(
    error: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expanded = mask.expand(-1, target.shape[1], -1, -1)
    count = expanded.flatten(1).sum(1).clamp_min(1)
    magnitude = (target.abs() * expanded).flatten(1).sum(1) / count
    absolute_error = (error.abs() * expanded).flatten(1).sum(1) / count
    capture = 1.0 - absolute_error / magnitude.clamp_min(1e-6)
    return capture, magnitude, absolute_error


def equal_dataset_summary(
    values: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for region, by_dataset in sorted(values.items()):
        dataset_means = {
            dataset: float(np.mean(samples))
            for dataset, samples in sorted(by_dataset.items())
        }
        result[region] = {
            "equal_dataset_mean": float(np.mean(list(dataset_means.values()))),
            "by_dataset": dataset_means,
            "samples": {dataset: len(samples) for dataset, samples in by_dataset.items()},
        }
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = resolve_paper_path(config["data"]["manifest"])
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    dataset = MixedDistillationDataset(
        manifest, root=cache_root, split="validation", augment=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(config["training"]["workers"]),
        pin_memory=True,
        persistent_workers=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = student_from_checkpoint(checkpoint).to(device).eval()
    model_config = checkpoint["config"]["model"]
    target_config = config["target_validation"]
    coarse_sigma = float(target_config["coarse_sigma"])
    very_coarse_sigma = float(target_config["very_coarse_sigma"])

    captures: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    magnitudes: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    selected_count: dict[str, int] = defaultdict(int)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="spatial profile"):
            datasets = [str(value) for value in batch["dataset"]]
            isos = [int(value) for value in batch["iso"]]
            selected = [
                index
                for index, (dataset_name, iso) in enumerate(
                    zip(datasets, isos, strict=True)
                )
                if target_record_selected(dataset_name, iso, target_config)
            ]
            if not selected:
                continue
            indices = torch.tensor(selected, device=device)
            noisy = batch["noisy"].to(device).index_select(0, indices)
            teacher = batch["teacher"].to(device).index_select(0, indices)
            output = model(model_input_from_config(noisy, model_config))
            teacher_chroma = chroma_projection(teacher - noisy)
            student_chroma = chroma_projection(output - noisy)
            teacher_band = gaussian_blur(
                teacher_chroma, coarse_sigma
            ) - gaussian_blur(teacher_chroma, very_coarse_sigma)
            student_band = gaussian_blur(
                student_chroma, coarse_sigma
            ) - gaussian_blur(student_chroma, very_coarse_sigma)
            band_error = student_band - teacher_band
            selected_datasets = [datasets[index] for index in selected]

            masks = border_region_masks(teacher_band)
            masks.update(quantile_masks(luma_gradient(noisy), "input_gradient"))
            band_magnitude = teacher_band.abs().mean(1, keepdim=True)
            masks.update(quantile_masks(band_magnitude, "teacher_band_magnitude"))
            masks["all_pixels"] = torch.ones_like(
                band_magnitude, dtype=torch.bool
            )
            for region, mask in masks.items():
                capture, magnitude, absolute_error = capture_per_sample(
                    band_error, teacher_band, mask
                )
                for dataset_name, capture_value, magnitude_value, error_value in zip(
                    selected_datasets,
                    capture.cpu().tolist(),
                    magnitude.cpu().tolist(),
                    absolute_error.cpu().tolist(),
                    strict=True,
                ):
                    captures[region][dataset_name].append(float(capture_value))
                    magnitudes[region][dataset_name].append(float(magnitude_value))
                    errors[region][dataset_name].append(float(error_value))
            for dataset_name in selected_datasets:
                selected_count[dataset_name] += 1

    report = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
        },
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "selected_samples": dict(sorted(selected_count.items())),
        "capture": equal_dataset_summary(captures),
        "teacher_magnitude": equal_dataset_summary(magnitudes),
        "absolute_error": equal_dataset_summary(errors),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
