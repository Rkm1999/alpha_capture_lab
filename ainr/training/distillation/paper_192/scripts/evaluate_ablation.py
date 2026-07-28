#!/usr/bin/env python3
"""Compare the two controlled NIND supervision ablation checkpoints.

The NIND reference is an exposure-corrected low-ISO image rather than a
strict clean target. Consequently this evaluator reports both reference
metrics and agreement with the cached SCUNet teacher. Shadow metrics use a
mask derived from noisy-image Rec.709 luma below 0.25.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import atomic_json, load_config, resolve_paper_path, seed_everything, sha256_file
from src.dataset import DistillationDataset
from src.metrics import gaussian_ssim_per_image, psnr_per_image
from src.student import LiteDenoiseNet


PAPER_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("noisy", "teacher", "nind_teacher_only", "nind_full_reference")
STUDENT_METHODS = ("nind_teacher_only", "nind_full_reference")
EXPECTED_CHECKPOINTS = {
    "nind_teacher_only": ("nind_teacher_only", 0.0),
    "nind_full_reference": ("nind_full_reference", 1.0),
}
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
CORRECTION_MSE_FLOOR = 1e-6
CONTROLLED_NORMALIZATION = {
    "mode": "<controlled-supervision-mode>",
    "nind_gt_weight": "<controlled-nind-gt-weight>",
    "_config_path": "<controlled-config-path>",
    "outputs": "<run-output-paths>",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PAPER_ROOT / "configs/high_iso_ablation_teacher_only.yaml",
    )
    parser.add_argument(
        "--teacher-only-checkpoint",
        type=Path,
        default=PAPER_ROOT / "runs/high_iso_ablation/nind_teacher_only/best.pt",
    )
    parser.add_argument(
        "--full-reference-checkpoint",
        type=Path,
        default=PAPER_ROOT / "runs/high_iso_ablation/nind_full_reference/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_ROOT / "evaluation/high_iso_ablation/comparison",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--contact-count", type=int, default=12)
    parser.add_argument("--no-contact-sheet", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow diagnostic checkpoints whose run status is not complete.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        help="Diagnostic-only limit; requires --allow-incomplete.",
    )
    args = parser.parse_args()
    if args.workers is not None and args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.contact_count <= 0:
        parser.error("--contact-count must be positive")
    if args.limit_batches is not None and args.limit_batches <= 0:
        parser.error("--limit-batches must be positive")
    if args.limit_batches is not None and not args.allow_incomplete:
        parser.error("--limit-batches requires --allow-incomplete")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_controlled_payload(value: Any) -> Any:
    """Normalize only the expected differences between the two ablation runs."""

    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            if key in CONTROLLED_NORMALIZATION:
                normalized[key] = CONTROLLED_NORMALIZATION[key]
            else:
                normalized[key] = normalize_controlled_payload(value[key])
        return normalized
    if isinstance(value, list):
        return [normalize_controlled_payload(item) for item in value]
    return value


def first_difference(left: Any, right: Any, path: str = "root") -> str | None:
    if type(left) is not type(right):
        return f"{path}: types differ ({type(left).__name__} != {type(right).__name__})"
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            return f"{path}: keys differ ({sorted(left_keys ^ right_keys)})"
        for key in sorted(left):
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: lengths differ ({len(left)} != {len(right)})"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = first_difference(left_item, right_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    return None if left == right else f"{path}: values differ ({left!r} != {right!r})"


def controlled_run_identity(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    normalized = {
        label: normalize_controlled_payload(payload) for label, payload in payloads.items()
    }
    labels = list(normalized)
    if len(labels) != 2:
        raise ValueError(f"Expected exactly two controlled runs, got {labels}")
    difference = first_difference(normalized[labels[0]], normalized[labels[1]])
    if difference is not None:
        raise ValueError(
            "Ablation runs differ outside supervision mode/NIND GT weight/output paths: "
            f"{difference}"
        )
    canonical = normalized[labels[0]]
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "normalized_fields": dict(CONTROLLED_NORMALIZATION),
        "payload": canonical,
    }


def validation_content_identity(dataset: DistillationDataset) -> dict[str, Any]:
    """Hash every selected validation tensor, not only its manifest path."""

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


def load_checkpoint_model(
    label: str,
    path: Path,
    device: torch.device,
    manifest_path: Path,
    *,
    allow_incomplete: bool,
) -> tuple[LiteDenoiseNet, dict[str, Any], dict[str, Any]]:
    checkpoint_path = path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing {label} checkpoint: {checkpoint_path}")
    if not allow_incomplete and checkpoint_path.name != "best.pt":
        raise ValueError(f"Final evaluation requires a best.pt checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_mode, expected_weight = EXPECTED_CHECKPOINTS[label]
    actual_mode = str(checkpoint.get("mode", ""))
    actual_weight = float(checkpoint.get("nind_gt_weight", math.nan))
    if actual_mode != expected_mode or not math.isclose(
        actual_weight, expected_weight, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"Checkpoint {checkpoint_path} is {actual_mode!r} with NIND GT weight "
            f"{actual_weight}, expected {expected_mode!r} with {expected_weight}"
        )
    if not math.isclose(float(checkpoint.get("alpha", math.nan)), 0.7, abs_tol=1e-12):
        raise ValueError(f"Checkpoint does not use the controlled alpha=0.7: {checkpoint_path}")

    status_path = checkpoint_path.parent / "status.json"
    run_path = checkpoint_path.parent / "run.json"
    if not status_path.is_file() or not run_path.is_file():
        raise ValueError(f"Run metadata is missing beside checkpoint: {checkpoint_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run_config = run.get("config")
    if not isinstance(run_config, dict):
        raise ValueError(f"Run configuration is missing or invalid: {run_path}")
    if checkpoint.get("config") != run_config:
        raise ValueError(f"Checkpoint/run configuration mismatch: {checkpoint_path}")

    run_mode = str(run.get("mode", ""))
    run_weight = float(run.get("nind_gt_weight", math.nan))
    configured_weight = float(
        run_config.get("training", {}).get("nind_gt_weight", math.nan)
    )
    if run_mode != expected_mode or any(
        not math.isclose(value, expected_weight, rel_tol=0.0, abs_tol=1e-12)
        for value in (run_weight, configured_weight)
    ):
        raise ValueError(
            f"Run metadata does not match controlled mode {expected_mode}: {run_path}"
        )

    expected_epochs = int(run["epochs_effective"])
    diagnostic_limits = run.get("diagnostic_limits")
    if not isinstance(diagnostic_limits, dict):
        raise ValueError(f"Run diagnostic limits are missing: {run_path}")
    from train_ablation import run_fingerprint

    expected_fingerprint, expected_fingerprint_payload = run_fingerprint(
        run_config,
        manifest_path,
        expected_epochs,
        diagnostic_limits.get("max_train_batches"),
        diagnostic_limits.get("max_val_batches"),
    )
    if run.get("run_fingerprint_payload") != expected_fingerprint_payload:
        raise ValueError(f"Stored run fingerprint payload is stale or altered: {run_path}")
    if checkpoint.get("run_fingerprint") != expected_fingerprint or run.get(
        "run_fingerprint"
    ) != expected_fingerprint:
        raise ValueError(
            f"Checkpoint does not match its configuration, manifest, and current sources: "
            f"{checkpoint_path}"
        )

    if status.get("mode") != expected_mode or int(status.get("epochs", -1)) != expected_epochs:
        raise ValueError(
            f"Run status does not identify the expected controlled run: {status_path}"
        )
    status_epoch = int(status.get("epoch", -1))
    checkpoint_epoch = int(checkpoint["epoch"])
    if (
        status_epoch < checkpoint_epoch
        or checkpoint_epoch < 1
        or checkpoint_epoch > expected_epochs
    ):
        raise ValueError(f"Checkpoint/status epoch mismatch: {checkpoint_path}")
    if not allow_incomplete:
        if status.get("state") != "complete" or status_epoch != expected_epochs:
            raise ValueError(
                f"Run has not completed all {expected_epochs} epochs: {checkpoint_path}"
            )
        if run.get("diagnostic_limits") != {
            "max_train_batches": None,
            "max_val_batches": None,
        }:
            raise ValueError(f"Final evaluation rejects diagnostic run: {checkpoint_path}")
    if checkpoint_path.name == "best.pt":
        saved_metrics = checkpoint.get("metrics", {})
        saved_selection_psnr = float(
            saved_metrics.get("validation", {}).get("selection_student_psnr", math.nan)
        )
        best_psnr = float(checkpoint.get("best_psnr", math.nan))
        status_best_psnr = float(status.get("best_selection_psnr", math.nan))
        if saved_metrics.get("improved") is not True or any(
            not math.isclose(value, best_psnr, rel_tol=0.0, abs_tol=1e-12)
            for value in (saved_selection_psnr, status_best_psnr)
        ):
            raise ValueError(
                f"best.pt is inconsistent with recorded best metrics: {checkpoint_path}"
            )

    model = LiteDenoiseNet().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    metadata = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "mode": actual_mode,
        "alpha": float(checkpoint["alpha"]),
        "nind_gt_weight": actual_weight,
        "best_selection_psnr_at_save": float(checkpoint["best_psnr"]),
        "run_fingerprint": str(checkpoint["run_fingerprint"]),
        "initial_model_sha256": str(run.get("initial_model_sha256", "")),
        "manifest_sha256": str(
            run.get("environment", {}).get("dataset_manifest_sha256", "")
        ),
        "run_status": str(status.get("state", "unknown")),
        "expected_epochs": expected_epochs,
        "status_epoch": status_epoch,
        "source_sha256": expected_fingerprint_payload["source_sha256"],
    }
    stable_environment_keys = (
        "python",
        "torch",
        "cuda_runtime",
        "cuda_available",
        "cuda_device",
        "platform",
        "project_commit",
        "teacher_repo_commit",
        "teacher_checkpoint_sha256",
        "preprocessing_version",
    )
    environment = run.get("environment", {})
    control_payload = {
        "mode": run_mode,
        "nind_gt_weight": run_weight,
        "config": run_config,
        "run_fingerprint_payload": expected_fingerprint_payload,
        "epochs_effective": expected_epochs,
        "batch_size": run.get("batch_size"),
        "dataset_counts": run.get("dataset_counts"),
        "diagnostic_limits": diagnostic_limits,
        "initial_model_sha256": run.get("initial_model_sha256"),
        "loss_coefficients": run.get("loss_coefficients"),
        "selection_datasets": run.get("selection_datasets"),
        "train_samples": run.get("train_samples"),
        "validation_samples": run.get("validation_samples"),
        "environment": {key: environment.get(key) for key in stable_environment_keys},
    }
    return model, metadata, control_payload


def load_manifest_metadata(
    manifest_path: Path, dataset: DistillationDataset
) -> list[dict[str, Any]]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_records = document["records"] if isinstance(document, dict) else document
    validation = [row for row in raw_records if str(row["split"]) == "validation"]
    if len(validation) != len(dataset):
        raise ValueError(
            f"Validation manifest mismatch: raw={len(validation)}, dataset={len(dataset)}"
        )
    for index, (raw, record) in enumerate(zip(validation, dataset.records, strict=True)):
        identity = (str(raw["input"]), str(raw["clean"]), str(raw["teacher"]))
        expected = (record.input, record.clean, record.teacher)
        if identity != expected:
            raise ValueError(f"Validation manifest order mismatch at record {index}")
    return validation


def metric_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    border: int,
    window_size: int,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        psnr_per_image(prediction, target, border=border),
        gaussian_ssim_per_image(
            prediction,
            target,
            border=border,
            window_size=window_size,
            sigma=sigma,
        ),
    )


def finite_float(value: torch.Tensor | float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def crop_spatial(value: torch.Tensor, border: int) -> torch.Tensor:
    return value if border == 0 else value[..., border:-border, border:-border]


def shadow_metrics(
    outputs: dict[str, torch.Tensor],
    noisy: torch.Tensor,
    teacher: torch.Tensor,
    border: int,
) -> list[dict[str, Any]]:
    noisy_cropped = crop_spatial(noisy.float(), border)
    teacher_cropped = crop_spatial(teacher.float(), border)
    luma = sum(
        weight * noisy_cropped[:, channel]
        for channel, weight in enumerate(LUMA_WEIGHTS)
    )
    masks = luma < 0.25
    teacher_correction = teacher_cropped - noisy_cropped
    results: list[dict[str, Any]] = []
    for image_index in range(noisy.shape[0]):
        mask = masks[image_index]
        pixel_count = int(mask.sum().item())
        total_pixels = int(mask.numel())
        row: dict[str, Any] = {
            "pixels": pixel_count,
            "fraction": pixel_count / total_pixels,
            "teacher_correction_mse": None,
            "correction_capture_eligible": False,
            "methods": {},
        }
        if pixel_count == 0:
            for label in outputs:
                row["methods"][label] = None
            results.append(row)
            continue

        channel_mask = mask.unsqueeze(0).expand(3, -1, -1)
        target_vector = teacher_correction[image_index][channel_mask]
        target_energy = float(torch.dot(target_vector, target_vector))
        target_mean_energy = target_energy / int(target_vector.numel())
        stable_teacher_correction = target_mean_energy >= CORRECTION_MSE_FLOOR
        row["teacher_correction_mse"] = target_mean_energy
        row["correction_capture_eligible"] = stable_teacher_correction
        noisy_teacher_mae = float(target_vector.abs().mean())
        for label, output in outputs.items():
            prediction = crop_spatial(output.float(), border)[image_index]
            error_vector = (prediction - teacher_cropped[image_index])[channel_mask]
            mae = float(error_vector.abs().mean())
            mse = float(error_vector.square().mean())
            correction = (prediction - noisy_cropped[image_index])[channel_mask]
            correction_energy = float(torch.dot(correction, correction))
            projection = (
                float(torch.dot(correction, target_vector)) / target_energy
                if stable_teacher_correction
                else None
            )
            alignment = (
                float(torch.dot(correction, target_vector))
                / math.sqrt(correction_energy * target_energy)
                if correction_energy > 1e-12 and stable_teacher_correction
                else None
            )
            gap_closed = (
                1.0 - mae / noisy_teacher_mae
                if stable_teacher_correction and noisy_teacher_mae > 1e-12
                else None
            )
            row["methods"][label] = {
                "teacher_mae": mae,
                "teacher_rmse": math.sqrt(mse),
                "teacher_psnr": -10.0 * math.log10(mse) if mse > 0.0 else None,
                "correction_capture": projection,
                "correction_alignment": alignment,
                "teacher_gap_closed": gap_closed,
            }
        results.append(row)
    return results


def stats(values: Iterable[float | None]) -> dict[str, Any]:
    valid = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if valid.size == 0:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": int(valid.size),
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "minimum": float(valid.min()),
        "maximum": float(valid.max()),
    }


def summarize_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    reference = [row["metrics"][method]["reference"] for row in rows]
    teacher = [row["metrics"][method]["teacher"] for row in rows]
    shadows = [row["shadow"]["methods"].get(method) for row in rows]

    def shadow_values(name: str) -> list[float | None]:
        return [value.get(name) if value is not None else None for value in shadows]

    return {
        "reference": {
            "psnr": stats(value["psnr"] for value in reference),
            "ssim": stats(value["ssim"] for value in reference),
        },
        "teacher_agreement": {
            "psnr": stats(value["psnr"] for value in teacher),
            "ssim": stats(value["ssim"] for value in teacher),
        },
        "shadow_teacher_agreement": {
            "images_with_shadow": sum(value is not None for value in shadows),
            "teacher_mae": stats(shadow_values("teacher_mae")),
            "teacher_rmse": stats(shadow_values("teacher_rmse")),
            "teacher_psnr": stats(shadow_values("teacher_psnr")),
            "correction_capture": stats(shadow_values("correction_capture")),
            "correction_alignment": stats(shadow_values("correction_alignment")),
            "teacher_gap_closed": stats(shadow_values("teacher_gap_closed")),
        },
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    datasets = sorted({str(row["dataset"]) for row in rows})
    for dataset in [*datasets, "all"]:
        selected = rows if dataset == "all" else [row for row in rows if row["dataset"] == dataset]
        result[dataset] = {
            "images": len(selected),
            "scenes": len({str(row["scene"]) for row in selected}),
            "shadow_fraction": stats(row["shadow"]["fraction"] for row in selected),
            "shadow_pixels": sum(int(row["shadow"]["pixels"]) for row in selected),
            "teacher_correction_mse": stats(
                row["shadow"].get("teacher_correction_mse") for row in selected
            ),
            "correction_capture_eligible_images": sum(
                bool(row["shadow"].get("correction_capture_eligible")) for row in selected
            ),
            "methods": {method: summarize_method(selected, method) for method in METHODS},
        }
    return result


def noise_level_sort_key(value: str) -> tuple[int, int, str]:
    if value.isdigit():
        return (0, int(value), "")
    if len(value) > 1 and value[0].upper() == "H" and value[1:].isdigit():
        return (1, int(value[1:]), "")
    return (2, 0, value)


def build_nind_noise_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nind_rows = [row for row in rows if row["dataset"] == "nind"]
    levels = sorted(
        {str(row["noise_level"]) for row in nind_rows}, key=noise_level_sort_key
    )
    result: dict[str, Any] = {}
    for level in levels:
        selected = [row for row in nind_rows if str(row["noise_level"]) == level]
        result[level] = {
            "images": len(selected),
            "scenes": len({str(row["scene"]) for row in selected}),
            "shadow_fraction": stats(row["shadow"]["fraction"] for row in selected),
            "shadow_pixels": sum(int(row["shadow"]["pixels"]) for row in selected),
            "teacher_correction_mse": stats(
                row["shadow"].get("teacher_correction_mse") for row in selected
            ),
            "correction_capture_eligible_images": sum(
                bool(row["shadow"].get("correction_capture_eligible")) for row in selected
            ),
            "methods": {method: summarize_method(selected, method) for method in METHODS},
        }
    return result


def mean_at(summary: dict[str, Any], dataset: str, method: str, *path: str) -> float | None:
    value: Any = summary[dataset]["methods"][method]
    for key in path:
        value = value[key]
    return value.get("mean") if isinstance(value, dict) else None


def delta(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def build_deltas(summary: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for dataset in summary:
        full = "nind_full_reference"
        teacher_only = "nind_teacher_only"
        result[dataset] = {
            "definition": "nind_full_reference minus nind_teacher_only",
            "reference_psnr": delta(
                mean_at(summary, dataset, full, "reference", "psnr"),
                mean_at(summary, dataset, teacher_only, "reference", "psnr"),
            ),
            "reference_ssim": delta(
                mean_at(summary, dataset, full, "reference", "ssim"),
                mean_at(summary, dataset, teacher_only, "reference", "ssim"),
            ),
            "teacher_psnr": delta(
                mean_at(summary, dataset, full, "teacher_agreement", "psnr"),
                mean_at(summary, dataset, teacher_only, "teacher_agreement", "psnr"),
            ),
            "shadow_teacher_mae": delta(
                mean_at(summary, dataset, full, "shadow_teacher_agreement", "teacher_mae"),
                mean_at(
                    summary,
                    dataset,
                    teacher_only,
                    "shadow_teacher_agreement",
                    "teacher_mae",
                ),
            ),
            "shadow_correction_capture": delta(
                mean_at(
                    summary,
                    dataset,
                    full,
                    "shadow_teacher_agreement",
                    "correction_capture",
                ),
                mean_at(
                    summary,
                    dataset,
                    teacher_only,
                    "shadow_teacher_agreement",
                    "correction_capture",
                ),
            ),
            "shadow_teacher_gap_closed": delta(
                mean_at(
                    summary,
                    dataset,
                    full,
                    "shadow_teacher_agreement",
                    "teacher_gap_closed",
                ),
                mean_at(
                    summary,
                    dataset,
                    teacher_only,
                    "shadow_teacher_agreement",
                    "teacher_gap_closed",
                ),
            ),
        }
    return result


def csv_value(value: Any) -> Any:
    return "" if value is None else value


def write_per_image_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "index",
        "dataset",
        "scene",
        "noise_level",
        "iso",
        "supervision",
        "method",
        "reference_psnr",
        "reference_ssim",
        "teacher_psnr",
        "teacher_ssim",
        "shadow_pixels",
        "shadow_fraction",
        "shadow_teacher_correction_mse",
        "shadow_correction_capture_eligible",
        "shadow_teacher_mae",
        "shadow_teacher_rmse",
        "shadow_teacher_psnr",
        "shadow_correction_capture",
        "shadow_correction_alignment",
        "shadow_teacher_gap_closed",
    )
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method in METHODS:
                metric = row["metrics"][method]
                shadow = row["shadow"]["methods"].get(method) or {}
                writer.writerow(
                    {
                        "index": row["index"],
                        "dataset": row["dataset"],
                        "scene": row["scene"],
                        "noise_level": row["noise_level"],
                        "iso": csv_value(row["iso"]),
                        "supervision": row["supervision"],
                        "method": method,
                        "reference_psnr": csv_value(metric["reference"]["psnr"]),
                        "reference_ssim": csv_value(metric["reference"]["ssim"]),
                        "teacher_psnr": csv_value(metric["teacher"]["psnr"]),
                        "teacher_ssim": csv_value(metric["teacher"]["ssim"]),
                        "shadow_pixels": row["shadow"]["pixels"],
                        "shadow_fraction": row["shadow"]["fraction"],
                        "shadow_teacher_correction_mse": csv_value(
                            row["shadow"].get("teacher_correction_mse")
                        ),
                        "shadow_correction_capture_eligible": bool(
                            row["shadow"].get("correction_capture_eligible")
                        ),
                        "shadow_teacher_mae": csv_value(shadow.get("teacher_mae")),
                        "shadow_teacher_rmse": csv_value(shadow.get("teacher_rmse")),
                        "shadow_teacher_psnr": csv_value(shadow.get("teacher_psnr")),
                        "shadow_correction_capture": csv_value(
                            shadow.get("correction_capture")
                        ),
                        "shadow_correction_alignment": csv_value(
                            shadow.get("correction_alignment")
                        ),
                        "shadow_teacher_gap_closed": csv_value(
                            shadow.get("teacher_gap_closed")
                        ),
                    }
                )


def write_summary_csv(
    path: Path, summary: dict[str, Any], *, group_field: str = "dataset"
) -> None:
    fields = (
        group_field,
        "images",
        "scenes",
        "correction_capture_eligible_images",
        "method",
        "reference_psnr_mean",
        "reference_ssim_mean",
        "teacher_psnr_mean",
        "teacher_ssim_mean",
        "shadow_teacher_correction_mse_mean",
        "shadow_teacher_mae_mean",
        "shadow_teacher_psnr_mean",
        "shadow_correction_capture_mean",
        "shadow_correction_alignment_mean",
        "shadow_teacher_gap_closed_mean",
    )
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for group, group_summary in summary.items():
            for method, metric in group_summary["methods"].items():
                shadow = metric["shadow_teacher_agreement"]
                writer.writerow(
                    {
                        group_field: group,
                        "images": group_summary["images"],
                        "scenes": group_summary["scenes"],
                        "correction_capture_eligible_images": group_summary[
                            "correction_capture_eligible_images"
                        ],
                        "method": method,
                        "reference_psnr_mean": csv_value(metric["reference"]["psnr"]["mean"]),
                        "reference_ssim_mean": csv_value(metric["reference"]["ssim"]["mean"]),
                        "teacher_psnr_mean": csv_value(
                            metric["teacher_agreement"]["psnr"]["mean"]
                        ),
                        "teacher_ssim_mean": csv_value(
                            metric["teacher_agreement"]["ssim"]["mean"]
                        ),
                        "shadow_teacher_correction_mse_mean": csv_value(
                            group_summary["teacher_correction_mse"]["mean"]
                        ),
                        "shadow_teacher_mae_mean": csv_value(shadow["teacher_mae"]["mean"]),
                        "shadow_teacher_psnr_mean": csv_value(shadow["teacher_psnr"]["mean"]),
                        "shadow_correction_capture_mean": csv_value(
                            shadow["correction_capture"]["mean"]
                        ),
                        "shadow_correction_alignment_mean": csv_value(
                            shadow["correction_alignment"]["mean"]
                        ),
                        "shadow_teacher_gap_closed_mean": csv_value(
                            shadow["teacher_gap_closed"]["mean"]
                        ),
                    }
                )


def tensor_image(value: torch.Tensor) -> Image.Image:
    pixels = value.detach().float().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8))


def contact_font() -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(candidate, 13)
    return ImageFont.load_default()


def contact_selection(rows: list[dict[str, Any]], count: int) -> dict[str, Any]:
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["dataset"]), str(row["noise_level"]))
        by_stratum[key].append(row)
    for candidates in by_stratum.values():
        candidates.sort(
            key=lambda row: (
                -float(row["shadow"]["fraction"]),
                str(row["scene"]),
                int(row["index"]),
            )
        )

    strata = sorted(
        by_stratum,
        key=lambda key: (
            0 if key[0] == "nind" else 1,
            "" if key[0] == "nind" else key[0],
            noise_level_sort_key(key[1]),
        ),
    )
    limit = min(count, len(rows))
    selected_rows: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    used_scenes: dict[str, set[str]] = defaultdict(set)

    def select_one(stratum: tuple[str, str], *, require_unique_scene: bool) -> bool:
        dataset_name = stratum[0]
        for candidate in by_stratum[stratum]:
            index = int(candidate["index"])
            scene = str(candidate["scene"])
            if index in selected_indices:
                continue
            if require_unique_scene and scene in used_scenes[dataset_name]:
                continue
            selected_indices.add(index)
            used_scenes[dataset_name].add(scene)
            selected_rows.append(candidate)
            return True
        return False

    # Cover every NIND noise level first, then every other dataset/noise stratum.
    for stratum in strata:
        if len(selected_rows) >= limit:
            break
        select_one(stratum, require_unique_scene=True)

    # Keep cycling through strata while a new scene is available.
    progress = True
    while len(selected_rows) < limit and progress:
        progress = False
        for stratum in strata:
            if len(selected_rows) >= limit:
                break
            progress = select_one(stratum, require_unique_scene=True) or progress

    # Only repeat scenes after exhausting scene diversity, while retaining strata balance.
    progress = True
    while len(selected_rows) < limit and progress:
        progress = False
        for stratum in strata:
            if len(selected_rows) >= limit:
                break
            progress = select_one(stratum, require_unique_scene=False) or progress

    return {
        "policy": (
            "NIND-noise-level coverage first; then remaining dataset/noise strata; "
            "unique scenes before repeated scenes; candidates within each stratum sorted "
            "by descending shadow fraction and stable index"
        ),
        "requested_count": count,
        "selected_count": len(selected_rows),
        "indices": [int(row["index"]) for row in selected_rows],
        "samples": [
            {
                "index": int(row["index"]),
                "dataset": str(row["dataset"]),
                "scene": str(row["scene"]),
                "noise_level": str(row["noise_level"]),
                "shadow_fraction": float(row["shadow"]["fraction"]),
            }
            for row in selected_rows
        ],
    }


def write_contact_sheet(
    path: Path,
    dataset: DistillationDataset,
    metadata: list[dict[str, Any]],
    models: dict[str, LiteDenoiseNet],
    device: torch.device,
    indices: list[int],
) -> None:
    headings = ("Noisy", "Teacher only", "Full reference", "SCUNet", "Reference")
    model_order = STUDENT_METHODS
    tile = 192
    header = 32
    footer = 20
    row_height = header + tile + footer
    font = contact_font()
    sheet = Image.new("RGB", (tile * len(headings), row_height * len(indices)), "white")
    with torch.inference_mode():
        for row_index, index in enumerate(indices):
            sample = dataset[index]
            noisy = sample["noisy"].unsqueeze(0).to(device)
            images = [tensor_image(sample["noisy"])]
            images.extend(tensor_image(models[label](noisy)[0]) for label in model_order)
            images.extend((tensor_image(sample["teacher"]), tensor_image(sample["clean"])))
            top = row_index * row_height
            draw = ImageDraw.Draw(sheet)
            raw = metadata[index]
            descriptor = (
                f"{sample['dataset']} / {sample['scene']} / "
                f"noise={raw.get('noise_level', 'unknown')}"
            )
            draw.text((4, top + 4), descriptor, fill="black", font=font)
            for column, (heading, image) in enumerate(zip(headings, images, strict=True)):
                left = column * tile
                sheet.paste(image, (left, top + header))
                draw.text((left + 4, top + header + tile + 2), heading, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, compress_level=6)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    device = choose_device(args.device)
    manifest_path = resolve_paper_path(config["data"]["manifest"])
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DistillationDataset(
        manifest_path,
        root=cache_root,
        split="validation",
        augment=False,
    )
    metadata = load_manifest_metadata(manifest_path, dataset)
    validation_content = validation_content_identity(dataset)
    workers = int(config["training"]["workers"] if args.workers is None else args.workers)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    models: dict[str, LiteDenoiseNet] = {}
    checkpoint_metadata: dict[str, Any] = {}
    control_payloads: dict[str, dict[str, Any]] = {}
    checkpoint_paths = {
        "nind_teacher_only": args.teacher_only_checkpoint,
        "nind_full_reference": args.full_reference_checkpoint,
    }
    for label, checkpoint_path in checkpoint_paths.items():
        models[label], checkpoint_metadata[label], control_payloads[label] = (
            load_checkpoint_model(
                label,
                checkpoint_path,
                device,
                manifest_path,
                allow_incomplete=args.allow_incomplete,
            )
        )
    controlled_identity = controlled_run_identity(control_payloads)
    initialization_hashes = {
        value["initial_model_sha256"] for value in checkpoint_metadata.values()
    }
    if len(initialization_hashes) != 1 or "" in initialization_hashes:
        raise ValueError("Ablation runs do not report the same deterministic initialization")
    expected_manifest_sha256 = sha256_file(manifest_path)
    checkpoint_manifest_hashes = {
        value["manifest_sha256"] for value in checkpoint_metadata.values()
    }
    if checkpoint_manifest_hashes != {expected_manifest_sha256}:
        raise ValueError(
            "Ablation checkpoints were not trained from the selected validation manifest"
        )

    border = int(config["metrics"]["border_crop"])
    window_size = int(config["metrics"]["ssim_window_size"])
    sigma = float(config["metrics"]["ssim_sigma"])
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="Evaluating NIND ablation")):
            if args.limit_batches is not None and batch_index >= args.limit_batches:
                break
            noisy = batch["noisy"].to(device, non_blocking=True)
            reference = batch["clean"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            predictions = {label: model(noisy) for label, model in models.items()}
            outputs = {"noisy": noisy, "teacher": teacher, **predictions}
            reference_metrics = {
                label: metric_pair(
                    output,
                    reference,
                    border=border,
                    window_size=window_size,
                    sigma=sigma,
                )
                for label, output in outputs.items()
            }
            teacher_metrics = {
                label: metric_pair(
                    output,
                    teacher,
                    border=border,
                    window_size=window_size,
                    sigma=sigma,
                )
                for label, output in outputs.items()
                if label != "teacher"
            }
            shadows = shadow_metrics(outputs, noisy, teacher, border)
            for local_index, selection_index in enumerate(batch["index"].tolist()):
                selection_index = int(selection_index)
                record = dataset.records[selection_index]
                raw = metadata[selection_index]
                row_metrics: dict[str, Any] = {}
                for label in METHODS:
                    reference_pair = reference_metrics[label]
                    if label == "teacher":
                        teacher_pair = None
                    else:
                        teacher_pair = teacher_metrics[label]
                    row_metrics[label] = {
                        "reference": {
                            "psnr": finite_float(reference_pair[0][local_index]),
                            "ssim": finite_float(reference_pair[1][local_index]),
                        },
                        "teacher": {
                            "psnr": (
                                finite_float(teacher_pair[0][local_index])
                                if teacher_pair is not None
                                else None
                            ),
                            "ssim": (
                                finite_float(teacher_pair[1][local_index])
                                if teacher_pair is not None
                                else 1.0
                            ),
                        },
                    }
                rows.append(
                    {
                        "index": selection_index,
                        "id": str(raw.get("id", selection_index)),
                        "dataset": record.dataset,
                        "scene": record.scene,
                        "noise_level": str(raw.get("noise_level", "unknown")),
                        "iso": raw.get("iso"),
                        "supervision": str(raw.get("supervision", "unknown")),
                        "input": record.input,
                        "reference": record.clean,
                        "teacher": record.teacher,
                        "metrics": row_metrics,
                        "shadow": shadows[local_index],
                    }
                )
    if not rows:
        raise RuntimeError("Evaluation processed no validation samples")
    if args.limit_batches is None and len(rows) != len(dataset):
        raise RuntimeError(
            f"Complete evaluation processed {len(rows)} of {len(dataset)} validation samples"
        )

    per_image_jsonl = output_dir / "per_image.jsonl"
    with per_image_jsonl.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    per_image_csv = output_dir / "per_image.csv"
    write_per_image_csv(per_image_csv, rows)
    summary = build_summary(rows)
    deltas = build_deltas(summary)
    nind_by_noise_level = build_nind_noise_summary(rows)
    nind_noise_level_deltas = build_deltas(nind_by_noise_level)
    nind_noise_csv = output_dir / "nind_by_noise_level.csv"
    write_summary_csv(nind_noise_csv, nind_by_noise_level, group_field="noise_level")
    evaluator_sources = {
        relative: sha256_file(PAPER_ROOT / relative)
        for relative in (
            "scripts/evaluate_ablation.py",
            "src/dataset.py",
            "src/metrics.py",
            "src/student.py",
        )
    }
    contact_metadata: dict[str, Any] = {"enabled": not args.no_contact_sheet}
    if not args.no_contact_sheet:
        contact_metadata.update(contact_selection(rows, args.contact_count))
    provenance = {
        "schema_version": 2,
        "config": {
            "path": str(args.config.expanduser().resolve()),
            "sha256": sha256_file(args.config.expanduser().resolve()),
        },
        "manifest": {"path": str(manifest_path), "sha256": expected_manifest_sha256},
        "validation_content": validation_content,
        "preprocessing_version": str(config["project"]["preprocessing_version"]),
        "samples": len(rows),
        "complete_validation": args.limit_batches is None,
        "device": str(device),
        "checkpoints": checkpoint_metadata,
        "controlled_run_identity": controlled_identity,
        "evaluator_sources": evaluator_sources,
        "metrics": {
            "color_space": "RGB [0,1]",
            "border_crop": border,
            "ssim_window_size": window_size,
            "ssim_sigma": sigma,
            "shadow_mask": "Rec.709 luma(noisy RGB) < 0.25 after border crop",
            "correction_capture": (
                "dot(method-noisy, teacher-noisy) / "
                "dot(teacher-noisy, teacher-noisy), within the shadow mask; reported "
                f"only when mean((teacher-noisy)^2) >= {CORRECTION_MSE_FLOOR:g}"
            ),
            "correction_capture_teacher_mse_floor_per_rgb_element": CORRECTION_MSE_FLOOR,
            "correction_alignment": (
                "cosine similarity(method-noisy, teacher-noisy), within the shadow mask; "
                "uses the same teacher-correction MSE eligibility floor"
            ),
            "teacher_gap_closed": (
                "1 - MAE(method, teacher) / MAE(noisy, teacher), within the shadow mask; "
                "uses the same teacher-correction MSE eligibility floor"
            ),
        },
        "contact_sheet_selection": contact_metadata,
        "reference_semantics": {
            "midd": "paired clean target",
            "sidd": "paired clean target",
            "polyu_sony": "burst-mean reference",
            "nind": (
                "exposure-corrected lower-ISO reference; report separately and do not "
                "interpret as strict clean ground truth"
            ),
        },
    }
    evaluation_payload = {
        "manifest_sha256": provenance["manifest"]["sha256"],
        "validation_content_sha256": validation_content["sha256"],
        "checkpoints": {
            label: value["sha256"] for label, value in checkpoint_metadata.items()
        },
        "controlled_run_identity_sha256": controlled_identity["sha256"],
        "evaluator_sources": evaluator_sources,
        "metrics": provenance["metrics"],
        "samples": len(rows),
        "device": str(device),
    }
    evaluation_id = hashlib.sha256(
        json.dumps(evaluation_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": 2,
        "evaluation_id": evaluation_id,
        "provenance": provenance,
        "summary": summary,
        "deltas": deltas,
        "nind_by_noise_level": nind_by_noise_level,
        "nind_noise_level_deltas": nind_noise_level_deltas,
        "artifacts": {
            "per_image_jsonl": str(per_image_jsonl),
            "per_image_csv": str(per_image_csv),
            "summary_csv": str(output_dir / "summary.csv"),
            "nind_by_noise_level_csv": str(nind_noise_csv),
        },
    }
    atomic_json(output_dir / "summary.json", report)
    write_summary_csv(output_dir / "summary.csv", summary)
    if not args.no_contact_sheet:
        contact_path = output_dir / "comparison_contact_sheet.png"
        write_contact_sheet(
            contact_path,
            dataset,
            metadata,
            models,
            device,
            contact_metadata["indices"],
        )
        report["artifacts"]["contact_sheet"] = str(contact_path)
        atomic_json(output_dir / "summary.json", report)

    concise = {
        "evaluation_id": evaluation_id,
        "samples": len(rows),
        "device": str(device),
        "nind": summary.get("nind"),
        "nind_delta_full_minus_teacher_only": deltas.get("nind"),
        "nind_noise_level_deltas": nind_noise_level_deltas,
        "output_dir": str(output_dir),
    }
    print(json.dumps(concise, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
