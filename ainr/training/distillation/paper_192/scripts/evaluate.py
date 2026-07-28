#!/usr/bin/env python3
"""Evaluate the controlled matrix with identical per-image RGB metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import atomic_json, load_config, resolve_paper_path, seed_everything, sha256_file
from src.dataset import DistillationDataset
from src.metrics import gaussian_ssim_per_image, psnr_per_image
from src.student import LiteDenoiseNet


CHECKPOINT_LABEL_ALPHAS = {
    "alpha_0p0": 0.0,
    "alpha_0p7": 0.7,
    "alpha_0p9": 0.9,
}
RESERVED_LABELS = {"all", "clean", "noisy", "scunet", "teacher"}


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("checkpoint label cannot be empty")
    if label.casefold() in RESERVED_LABELS:
        raise argparse.ArgumentTypeError(f"checkpoint label is reserved: {label}")
    if label not in CHECKPOINT_LABEL_ALPHAS:
        supported = ", ".join(CHECKPOINT_LABEL_ALPHAS)
        raise argparse.ArgumentTypeError(
            f"checkpoint label must encode a supported alpha ({supported}), got {label!r}"
        )
    if not path:
        raise argparse.ArgumentTypeError("checkpoint path cannot be empty")
    return label, Path(path).expanduser().resolve()


def ordered_checkpoint_specs(
    values: list[tuple[str, Path]], *, require_paper_matrix: bool = False
) -> list[tuple[str, Path]]:
    labels = [label for label, _ in values]
    duplicates = sorted(label for label in set(labels) if labels.count(label) > 1)
    if duplicates:
        raise ValueError(f"Duplicate checkpoint labels: {duplicates}")
    if any(label.casefold() in RESERVED_LABELS for label in labels):
        raise ValueError("Checkpoint labels cannot use reserved metric or panel names")
    unsupported = sorted(set(labels) - set(CHECKPOINT_LABEL_ALPHAS))
    if unsupported:
        raise ValueError(f"Unsupported checkpoint labels: {unsupported}")
    if require_paper_matrix and set(labels) != set(CHECKPOINT_LABEL_ALPHAS):
        raise ValueError(
            "The paper matrix requires exactly alpha_0p0, alpha_0p7, and alpha_0p9"
        )
    return sorted(values, key=lambda item: CHECKPOINT_LABEL_ALPHAS[item[0]])


def checkpoint_identity(label: str, path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    expected_alpha = CHECKPOINT_LABEL_ALPHAS[label]
    actual_alpha = float(checkpoint["alpha"])
    if not math.isclose(actual_alpha, expected_alpha, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Checkpoint label {label} encodes alpha {expected_alpha}, "
            f"but {path} contains alpha {actual_alpha}"
        )
    run_fingerprint = checkpoint.get("run_fingerprint")
    if not isinstance(run_fingerprint, str) or not run_fingerprint:
        raise ValueError(f"Checkpoint is missing its run fingerprint: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": int(checkpoint["epoch"]),
        "alpha": actual_alpha,
        "run_fingerprint": run_fingerprint,
        "best_psnr_at_save": float(checkpoint["best_psnr"]),
    }


def validate_checkpoint_run(
    path: Path,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    manifest: Path,
) -> None:
    from train import run_fingerprint

    if path.name != "best.pt":
        raise ValueError(f"Final evaluation requires a best.pt checkpoint: {path}")
    alpha = float(checkpoint["alpha"])
    epochs = int(config["training"]["epochs"])
    expected_fingerprint, _ = run_fingerprint(
        config,
        manifest,
        alpha,
        epochs,
        max_train_batches=None,
        max_val_batches=None,
    )
    if checkpoint.get("run_fingerprint") != expected_fingerprint:
        raise ValueError(
            f"Checkpoint does not match the current full-run configuration and sources: {path}"
        )

    status_path = path.parent / "status.json"
    if not status_path.is_file():
        raise ValueError(f"Checkpoint run status is missing: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("state") != "complete"
        or int(status.get("epoch", -1)) != epochs
        or int(status.get("epochs", -1)) != epochs
        or not math.isclose(
            float(status.get("alpha", math.nan)), alpha, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError(f"Checkpoint run is not a completed {epochs}-epoch run: {path}")


def validation_content_identity(dataset: DistillationDataset) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    for record in dataset.records:
        for field in ("input", "clean", "teacher"):
            relative_path = str(getattr(record, field))
            path = (dataset.root / relative_path).resolve()
            entry = [field, relative_path, sha256_file(path)]
            digest.update(json.dumps(entry, separators=(",", ":")).encode("utf-8"))
            file_count += 1
    return {
        "cache_root": str(dataset.root),
        "files": file_count,
        "sha256": digest.hexdigest(),
    }


def evaluation_identifier(provenance: dict[str, Any]) -> str:
    encoded = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metric_pair(
    prediction: torch.Tensor,
    clean: torch.Tensor,
    border: int,
    window_size: int,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        psnr_per_image(prediction, clean, border=border),
        gaussian_ssim_per_image(
            prediction,
            clean,
            border=border,
            window_size=window_size,
            sigma=sigma,
        ),
    )


def summarize(rows: list[dict], labels: list[str]) -> dict:
    result = {}
    datasets = sorted({row["dataset"] for row in rows}) + ["all"]
    methods = ["noisy", "teacher", *labels]
    for dataset in datasets:
        selected = rows if dataset == "all" else [row for row in rows if row["dataset"] == dataset]
        result[dataset] = {"images": len(selected)}
        for method in methods:
            psnr_values = np.asarray([row["metrics"][method]["psnr"] for row in selected])
            ssim_values = np.asarray([row["metrics"][method]["ssim"] for row in selected])
            result[dataset][method] = {
                "psnr_mean": float(psnr_values.mean()),
                "psnr_median": float(np.median(psnr_values)),
                "psnr_minimum": float(psnr_values.min()),
                "ssim_mean": float(ssim_values.mean()),
                "ssim_median": float(np.median(ssim_values)),
                "ssim_minimum": float(ssim_values.min()),
            }
    for label in labels:
        result["all"][label]["teacher_psnr_gap"] = (
            result["all"]["teacher"]["psnr_mean"] - result["all"][label]["psnr_mean"]
        )
    return result


def to_image(value: torch.Tensor) -> Image.Image:
    array = value.detach().float().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.rint(array * 255).clip(0, 255).astype(np.uint8))


def comparison_sheet(
    dataset: DistillationDataset,
    models: dict[str, LiteDenoiseNet],
    destination: Path,
    device: torch.device,
    seed: int,
    count: int,
) -> None:
    indices = random.Random(seed).sample(range(len(dataset)), min(count, len(dataset)))
    headings = ["noisy", *models.keys(), "SCUNet", "clean"]
    width = 192 * len(headings)
    rows = []
    with torch.inference_mode():
        for index in indices:
            sample = dataset[index]
            noisy = sample["noisy"].unsqueeze(0).to(device)
            images = [to_image(sample["noisy"])]
            for model in models.values():
                images.append(to_image(model(noisy)[0]))
            images.extend((to_image(sample["teacher"]), to_image(sample["clean"])))
            row = Image.new("RGB", (width, 220), "white")
            draw = ImageDraw.Draw(row)
            draw.text((4, 4), f"{sample['dataset']} / {sample['scene']}", fill="black")
            for column, (heading, image) in enumerate(zip(headings, images, strict=True)):
                left = column * 192
                row.paste(image, (left, 28))
                draw.text((left + 4, 204), heading, fill="black")
            rows.append(row)
    sheet = Image.new("RGB", (width, 220 * len(rows)), "white")
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * 220))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parents[1] / "configs/default.yaml"
    )
    parser.add_argument("--checkpoint", action="append", required=True, type=parse_checkpoint)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--contact-count", type=int, default=12)
    parser.add_argument(
        "--require-paper-matrix",
        action="store_true",
        help="Require exactly the configured alpha 0.0, 0.7, and 0.9 checkpoints.",
    )
    args = parser.parse_args()
    if args.contact_count <= 0:
        raise ValueError("contact-count must be positive")
    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    manifest = resolve_paper_path(config["data"]["manifest"])
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve_paper_path(config["outputs"]["evaluation"]) / "matrix"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = DistillationDataset(manifest, root=cache_root, split="validation")
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["workers"]),
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    checkpoint_metadata = {}
    checkpoint_specs = ordered_checkpoint_specs(
        args.checkpoint, require_paper_matrix=args.require_paper_matrix
    )
    for label, checkpoint_path in checkpoint_specs:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_checkpoint_run(checkpoint_path, checkpoint, config, manifest)
        checkpoint_metadata[label] = checkpoint_identity(label, checkpoint_path, checkpoint)
        model = LiteDenoiseNet().eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        models[label] = model.to(device)

    border = int(config["metrics"]["border_crop"])
    window_size = int(config["metrics"]["ssim_window_size"])
    sigma = float(config["metrics"]["ssim_sigma"])
    provenance = {
        "schema_version": 1,
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "validation_content": validation_content_identity(dataset),
        "preprocessing_version": str(config["project"]["preprocessing_version"]),
        "validation": {"split": "validation", "samples": len(dataset)},
        "metrics": {
            "target": "clean",
            "color_space": "RGB",
            "border_crop": border,
            "ssim_window_size": window_size,
            "ssim_sigma": sigma,
        },
        "checkpoints": checkpoint_metadata,
    }
    evaluation_id = evaluation_identifier(provenance)
    rows = []
    images_root = output_dir / "images"
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating matrix"):
            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            predictions = {label: model(noisy) for label, model in models.items()}
            all_outputs = {"noisy": noisy, "teacher": teacher, **predictions}
            metrics = {
                label: metric_pair(output, clean, border, window_size, sigma)
                for label, output in all_outputs.items()
            }
            for row_index, selection_index in enumerate(batch["index"].tolist()):
                record = dataset.records[int(selection_index)]
                row = {
                    "evaluation_id": evaluation_id,
                    "index": int(selection_index),
                    "dataset": record.dataset,
                    "scene": record.scene,
                    "input": record.input,
                    "clean": record.clean,
                    "teacher": record.teacher,
                    "metrics": {
                        label: {
                            "psnr": float(pair[0][row_index].cpu()),
                            "ssim": float(pair[1][row_index].cpu()),
                        }
                        for label, pair in metrics.items()
                    },
                }
                rows.append(row)
                if args.save_images:
                    for label, output in predictions.items():
                        destination = images_root / label / record.dataset / f"{selection_index:05d}.png"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        to_image(output[row_index]).save(destination)

    per_image_path = output_dir / "per_image.jsonl"
    with per_image_path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True) + "\n")
    per_image_metadata = {
        "path": str(per_image_path.resolve()),
        "sha256": sha256_file(per_image_path),
        "rows": len(rows),
        "evaluation_id": evaluation_id,
    }
    labels = list(models)
    report = {
        "schema_version": 2,
        "evaluation_id": evaluation_id,
        "provenance": provenance,
        "checkpoints": checkpoint_metadata,
        "manifest": str(manifest),
        "manifest_sha256": provenance["manifest"]["sha256"],
        "border_crop": border,
        "ssim": {"window_size": window_size, "sigma": sigma},
        "summary": summarize(rows, labels),
        "per_image_metrics": str(per_image_path),
        "per_image": per_image_metadata,
    }
    atomic_json(output_dir / "summary.json", report)
    comparison_sheet(
        dataset,
        models,
        output_dir / "comparison_contact_sheet.jpg",
        device,
        seed,
        args.contact_count,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
