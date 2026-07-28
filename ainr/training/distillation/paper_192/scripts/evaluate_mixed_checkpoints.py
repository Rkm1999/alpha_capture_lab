#!/usr/bin/env python3
"""Evaluate multiple mixed-training checkpoints on one pinned validation set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))

from common import atomic_json, load_config, resolve_paper_path, sha256_file  # noqa: E402
from src.mixed_dataset import MixedDistillationDataset  # noqa: E402
from src.student import student_from_checkpoint  # noqa: E402
from train_mixed import target_record_selected, validate  # noqa: E402


def checkpoint_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint label and path must be non-empty")
    return label, Path(raw_path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=checkpoint_spec, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
        help="Evaluate the deterministic, non-augmented form of this split.",
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Evaluate only rows selected by target_validation.",
    )
    args = parser.parse_args()
    labels = [label for label, _ in args.checkpoint]
    if len(labels) != len(set(labels)):
        raise ValueError("checkpoint labels must be unique")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    config = load_config(args.config)
    manifest = resolve_paper_path(config["data"]["manifest"])
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    dataset = MixedDistillationDataset(
        manifest, root=cache_root, split=args.split, augment=False
    )
    evaluation_dataset = dataset
    if args.target_only:
        target_config = config.get("target_validation")
        if not target_config or not bool(target_config.get("enabled", False)):
            raise ValueError("--target-only requires enabled target_validation")
        indices = [
            index
            for index, record in enumerate(dataset.records)
            if target_record_selected(
                str(record.dataset), int(record.iso or -1), target_config
            )
        ]
        if not indices:
            raise ValueError("Target selector matched no rows")
        evaluation_dataset = Subset(dataset, indices)
    loader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(config["training"]["workers"]),
        pin_memory=True,
        persistent_workers=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_config = config["metrics"]
    selection_datasets = set(map(str, config["training"]["selection_datasets"]))
    if args.split == "train":
        clean_valid_datasets = {
            str(record.dataset)
            for record in dataset.records
            if float(record.gt_weight) > 0.0
        }
        selection_datasets &= clean_valid_datasets
        if not selection_datasets:
            raise ValueError("Training split has no clean-valid selection dataset")
    if args.target_only:
        selection_datasets &= {
            str(dataset.records[index].dataset) for index in indices
        }
        if not selection_datasets:
            raise ValueError("Target rows contain no clean-valid selection dataset")
    results: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    for label, path in args.checkpoint:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = student_from_checkpoint(checkpoint).to(device).eval()
        checkpoint_model_config = checkpoint.get("config", {}).get("model", {})
        conditioning_config = checkpoint_model_config.get("noise_conditioning")
        results[label] = validate(
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
        checkpoints[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "epoch": int(checkpoint["epoch"]),
            "run_fingerprint": str(checkpoint["run_fingerprint"]),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison: dict[str, Any] = {}
    if len(labels) == 2:
        baseline, candidate = labels
        comparison = {
            "baseline": baseline,
            "candidate": candidate,
            "selection_psnr_delta": (
                results[candidate]["selection_student_psnr"]
                - results[baseline]["selection_student_psnr"]
            ),
            "selection_ssim_delta": (
                results[candidate]["selection_student_ssim"]
                - results[baseline]["selection_student_ssim"]
            ),
            "target_score_delta": (
                results[candidate]["target_validation"]["score"]
                - results[baseline]["target_validation"]["score"]
            ),
        }
    report = {
        "schema_version": 1,
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "split": args.split,
        "target_only": bool(args.target_only),
        "validation_rows": len(evaluation_dataset),
        "checkpoints": checkpoints,
        "results": results,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
