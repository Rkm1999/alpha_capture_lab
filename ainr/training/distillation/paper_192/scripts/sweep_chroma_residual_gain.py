#!/usr/bin/env python3
"""Sweep a deployable chroma-correction gain on the pinned validation set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(PAPER_ROOT / "scripts"))

from common import atomic_json, load_config, resolve_paper_path, sha256_file  # noqa: E402
from src.metrics import psnr_per_image  # noqa: E402
from src.mixed_dataset import MixedDistillationDataset  # noqa: E402
from src.noise_conditioning import model_input_from_config  # noqa: E402
from src.student import student_from_checkpoint  # noqa: E402
from train_mixed import (  # noqa: E402
    chroma_projection,
    target_correction_tensors,
    target_record_selected,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gains",
        type=float,
        nargs="+",
        default=(0.8, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.2),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def mean_by_dataset(
    values: dict[str, dict[str, list[float]]],
) -> dict[str, float]:
    return {
        name: float(np.mean([np.mean(rows) for rows in by_dataset.values()]))
        for name, by_dataset in values.items()
    }


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
    score_names = tuple(target_config["score_metrics"])
    selection_datasets = set(config["training"]["selection_datasets"])
    target: dict[float, dict[str, dict[str, list[float]]]] = {
        gain: defaultdict(lambda: defaultdict(list)) for gain in args.gains
    }
    general_psnr: dict[float, dict[str, list[float]]] = {
        gain: defaultdict(list) for gain in args.gains
    }

    with torch.inference_mode():
        for batch in tqdm(loader, desc="gain sweep"):
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            teacher = batch["teacher"].to(device)
            output = model(model_input_from_config(noisy, model_config))
            student_chroma = chroma_projection(output - noisy)
            datasets = [str(value) for value in batch["dataset"]]
            isos = [int(value) for value in batch["iso"]]
            gt_valid = [float(value) > 0.0 for value in batch["gt_weight"]]
            target_indices = [
                index
                for index, (dataset_name, iso) in enumerate(
                    zip(datasets, isos, strict=True)
                )
                if target_record_selected(dataset_name, iso, target_config)
            ]
            index_tensor = torch.tensor(target_indices, device=device)
            selected_datasets = [datasets[index] for index in target_indices]
            for gain in args.gains:
                candidate = (output + (gain - 1.0) * student_chroma).clamp(0.0, 1.0)
                psnr = psnr_per_image(
                    candidate, clean, border=int(config["metrics"]["border_crop"])
                ).cpu().tolist()
                for dataset_name, valid, value in zip(
                    datasets, gt_valid, psnr, strict=True
                ):
                    if valid and dataset_name in selection_datasets:
                        general_psnr[gain][dataset_name].append(float(value))
                if not target_indices:
                    continue
                tensors = target_correction_tensors(
                    candidate.index_select(0, index_tensor),
                    noisy.index_select(0, index_tensor),
                    teacher.index_select(0, index_tensor),
                    shadow_luminance_threshold=float(
                        target_config["shadow_luminance_threshold"]
                    ),
                    fine_sigma=float(target_config["fine_sigma"]),
                    medium_sigma=float(target_config["medium_sigma"]),
                    coarse_sigma=float(target_config["coarse_sigma"]),
                    very_coarse_sigma=float(target_config["very_coarse_sigma"]),
                )
                shadow_valid = tensors.pop("shadow_sample_valid").cpu().tolist()
                shadow_names = {
                    "shadow_teacher_correction_capture",
                    "shadow_chroma_teacher_correction_capture",
                }
                for name in score_names:
                    values = tensors[name].cpu().tolist()
                    for dataset_name, valid, value in zip(
                        selected_datasets, shadow_valid, values, strict=True
                    ):
                        if name in shadow_names and not valid:
                            continue
                        target[gain][name][dataset_name].append(float(value))

    results: dict[str, Any] = {}
    minimum_weight = float(target_config.get("minimum_metric_weight", 0.5))
    for gain in args.gains:
        metric_means = mean_by_dataset(target[gain])
        score_values = np.asarray(list(metric_means.values()))
        selection_psnr = float(
            np.mean(
                [
                    np.mean(general_psnr[gain][dataset_name])
                    for dataset_name in sorted(selection_datasets)
                ]
            )
        )
        results[str(gain)] = {
            "target_score": float(
                (1.0 - minimum_weight) * score_values.mean()
                + minimum_weight * score_values.min()
            ),
            "target_minimum": float(score_values.min()),
            "target_metrics": metric_means,
            "selection_psnr": selection_psnr,
        }
    report = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
