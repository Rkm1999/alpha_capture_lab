#!/usr/bin/env python3
"""Evaluate calibrated student-correction strength on pinned mixed validation."""

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


class ResidualScale(nn.Module):
    def __init__(self, model: nn.Module, scale: float) -> None:
        super().__init__()
        self.model = model
        self.scale = float(scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        noisy = value[:, :3]
        restored = self.model(value)
        return (noisy + self.scale * (restored - noisy)).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scale", type=float, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if any(scale <= 0.0 for scale in args.scale):
        raise ValueError("residual scales must be positive")
    if len(set(args.scale)) != len(args.scale):
        raise ValueError("residual scales must be unique")

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
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    conditioning_config = checkpoint.get("config", {}).get("model", {}).get(
        "noise_conditioning"
    )
    metrics_config = config["metrics"]
    selection_datasets = set(map(str, config["training"]["selection_datasets"]))
    results = {}
    for scale in args.scale:
        base = student_from_checkpoint(checkpoint).to(device).eval()
        model = ResidualScale(base, scale).to(device).eval()
        results[f"{scale:.4f}"] = validate(
            model,
            loader,
            device,
            int(metrics_config["border_crop"]),
            int(metrics_config["ssim_window_size"]),
            float(metrics_config["ssim_sigma"]),
            selection_datasets,
            config.get("target_validation"),
            None,
            conditioning_config,
        )
        del model, base
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
        },
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "validation_rows": len(dataset),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                scale: {
                    "target": metrics["target_validation"]["score"],
                    "selection_psnr": metrics["selection_student_psnr"],
                    "selection_ssim": metrics["selection_student_ssim"],
                    "components": metrics["target_validation"][
                        "metric_dataset_means"
                    ],
                }
                for scale, metrics in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
