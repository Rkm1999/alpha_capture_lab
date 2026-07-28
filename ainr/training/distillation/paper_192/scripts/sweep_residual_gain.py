#!/usr/bin/env python3
"""Measure whether scalar residual calibration improves held-out target capture."""

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


class ResidualGainModel(nn.Module):
    def __init__(self, model: nn.Module, gain: float) -> None:
        super().__init__()
        self.model = model
        self.gain = float(gain)

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        noisy = model_input[:, :3]
        restored = self.model(model_input)
        return (noisy + self.gain * (restored - noisy)).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gains",
        type=float,
        nargs="+",
        default=(0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if (
        not args.gains
        or len(args.gains) != len(set(args.gains))
        or any(not 0.0 < gain <= 4.0 for gain in args.gains)
    ):
        raise ValueError("gains must be unique values in (0,4]")

    config = load_config(args.config)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
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
    base_model = student_from_checkpoint(checkpoint).to(device).eval()
    conditioning_config = (
        checkpoint.get("config", {}).get("model", {}).get("noise_conditioning")
    )
    metrics_config = config["metrics"]
    selection_datasets = set(
        map(str, config["training"]["selection_datasets"])
    )
    results: dict[str, dict] = {}
    for gain in args.gains:
        label = f"{gain:g}"
        results[label] = validate(
            ResidualGainModel(base_model, gain).to(device).eval(),
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

    best_gain = max(
        results,
        key=lambda label: results[label]["target_validation"]["score"],
    )
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
        "best": {
            "gain": float(best_gain),
            "target_score": results[best_gain]["target_validation"]["score"],
            "selection_psnr": results[best_gain]["selection_student_psnr"],
            "selection_ssim": results[best_gain]["selection_student_ssim"],
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(json.dumps(report["best"], indent=2))


if __name__ == "__main__":
    main()
