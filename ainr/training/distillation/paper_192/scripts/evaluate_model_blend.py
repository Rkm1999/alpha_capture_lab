#!/usr/bin/env python3
"""Evaluate output blends of two students on pinned mixed validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))

from common import atomic_json, load_config, resolve_paper_path, sha256_file  # noqa: E402
from src.mixed_dataset import MixedDistillationDataset  # noqa: E402
from src.student import student_from_checkpoint  # noqa: E402
from train_mixed import validate  # noqa: E402


class ModelBlend(nn.Module):
    def __init__(self, first: nn.Module, second: nn.Module, second_weight: float) -> None:
        super().__init__()
        self.first = first
        self.second = second
        self.second_weight = float(second_weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.first(value)
        second = self.second(value)
        return torch.lerp(first, second, self.second_weight).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--second-weight", type=float, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if any(not 0.0 <= weight <= 1.0 for weight in args.second_weight):
        raise ValueError("blend weights must be in [0,1]")

    config = load_config(args.config)
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
    first_path = args.first.expanduser().resolve()
    second_path = args.second.expanduser().resolve()
    first_checkpoint = torch.load(first_path, map_location="cpu", weights_only=False)
    second_checkpoint = torch.load(second_path, map_location="cpu", weights_only=False)
    first_config = first_checkpoint["config"]["model"].get("noise_conditioning")
    second_config = second_checkpoint["config"]["model"].get("noise_conditioning")
    if first_config != second_config:
        raise ValueError("students use different noise conditioning")

    metrics_config = config["metrics"]
    selection_datasets = set(map(str, config["training"]["selection_datasets"]))
    results = {}
    for weight in args.second_weight:
        model = ModelBlend(
            student_from_checkpoint(first_checkpoint),
            student_from_checkpoint(second_checkpoint),
            weight,
        ).to(device).eval()
        results[f"{weight:.4f}"] = validate(
            model,
            loader,
            device,
            int(metrics_config["border_crop"]),
            int(metrics_config["ssim_window_size"]),
            float(metrics_config["ssim_sigma"]),
            selection_datasets,
            config.get("target_validation"),
            None,
            first_config,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "first": {"path": str(first_path), "sha256": sha256_file(first_path)},
        "second": {"path": str(second_path), "sha256": sha256_file(second_path)},
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "validation_rows": len(dataset),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                weight: {
                    "target": metrics["target_validation"]["score"],
                    "weakest": metrics["target_validation"]["score_metric_minimum"],
                    "selection_psnr": metrics["selection_student_psnr"],
                    "selection_ssim": metrics["selection_student_ssim"],
                }
                for weight, metrics in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
