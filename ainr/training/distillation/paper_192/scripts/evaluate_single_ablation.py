#!/usr/bin/env python3
"""Evaluate one completed NIND ablation checkpoint before the next run starts."""

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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import atomic_json, load_config, resolve_paper_path, seed_everything, sha256_file
from evaluate_ablation import (
    choose_device,
    contact_font,
    contact_selection,
    finite_float,
    load_checkpoint_model,
    load_manifest_metadata,
    metric_pair,
    shadow_metrics,
    tensor_image,
    validation_content_identity,
)
from src.dataset import DistillationDataset


PAPER_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("noisy", "student", "teacher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PAPER_ROOT / "configs/high_iso_ablation_teacher_only.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PAPER_ROOT / "runs/high_iso_ablation/nind_teacher_only/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PAPER_ROOT
            / "evaluation/high_iso_ablation/nind_teacher_only_validation"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--contact-count", type=int, default=12)
    args = parser.parse_args()
    if args.workers is not None and args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.contact_count <= 0:
        parser.error("--contact-count must be positive")
    return args


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
    agreement = [row["metrics"][method]["teacher"] for row in rows]
    shadows = [row["shadow"]["methods"].get(method) for row in rows]

    def shadow_values(name: str) -> list[float | None]:
        return [value.get(name) if value is not None else None for value in shadows]

    return {
        "reference": {
            "psnr": stats(value["psnr"] for value in reference),
            "ssim": stats(value["ssim"] for value in reference),
        },
        "teacher_agreement": {
            "psnr": stats(value["psnr"] for value in agreement),
            "ssim": stats(value["ssim"] for value in agreement),
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


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": len(rows),
        "scenes": len({str(row["scene"]) for row in rows}),
        "shadow_fraction": stats(row["shadow"]["fraction"] for row in rows),
        "shadow_pixels": sum(int(row["shadow"]["pixels"]) for row in rows),
        "correction_capture_eligible_images": sum(
            bool(row["shadow"].get("correction_capture_eligible")) for row in rows
        ),
        "methods": {method: summarize_method(rows, method) for method in METHODS},
    }


def build_dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    datasets = sorted({str(row["dataset"]) for row in rows})
    result = {
        dataset: summarize_group([row for row in rows if row["dataset"] == dataset])
        for dataset in datasets
    }
    result["all"] = summarize_group(rows)
    return result


def noise_level_sort_key(value: str) -> tuple[int, int, str]:
    if value.isdigit():
        return (0, int(value), "")
    if len(value) > 1 and value[0].upper() == "H" and value[1:].isdigit():
        return (1, int(value[1:]), "")
    return (2, 0, value)


def build_nind_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nind = [row for row in rows if row["dataset"] == "nind"]
    levels = sorted({str(row["noise_level"]) for row in nind}, key=noise_level_sort_key)
    return {
        level: summarize_group([row for row in nind if str(row["noise_level"]) == level])
        for level in levels
    }


def csv_value(value: Any) -> Any:
    return "" if value is None else value


def write_per_image_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "index", "dataset", "scene", "noise_level", "iso", "method",
        "reference_psnr", "reference_ssim", "teacher_psnr", "teacher_ssim",
        "shadow_pixels", "shadow_fraction", "shadow_teacher_mae",
        "shadow_teacher_psnr", "shadow_correction_capture",
        "shadow_correction_alignment", "shadow_teacher_gap_closed",
    )
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method in METHODS:
                metric = row["metrics"][method]
                shadow = row["shadow"]["methods"].get(method) or {}
                writer.writerow({
                    "index": row["index"],
                    "dataset": row["dataset"],
                    "scene": row["scene"],
                    "noise_level": row["noise_level"],
                    "iso": csv_value(row["iso"]),
                    "method": method,
                    "reference_psnr": csv_value(metric["reference"]["psnr"]),
                    "reference_ssim": csv_value(metric["reference"]["ssim"]),
                    "teacher_psnr": csv_value(metric["teacher"]["psnr"]),
                    "teacher_ssim": csv_value(metric["teacher"]["ssim"]),
                    "shadow_pixels": row["shadow"]["pixels"],
                    "shadow_fraction": row["shadow"]["fraction"],
                    "shadow_teacher_mae": csv_value(shadow.get("teacher_mae")),
                    "shadow_teacher_psnr": csv_value(shadow.get("teacher_psnr")),
                    "shadow_correction_capture": csv_value(shadow.get("correction_capture")),
                    "shadow_correction_alignment": csv_value(shadow.get("correction_alignment")),
                    "shadow_teacher_gap_closed": csv_value(shadow.get("teacher_gap_closed")),
                })


def write_summary_csv(path: Path, summary: dict[str, Any], group_name: str) -> None:
    fields = (
        group_name, "images", "scenes", "method", "reference_psnr",
        "reference_ssim", "teacher_psnr", "teacher_ssim", "shadow_teacher_mae",
        "shadow_teacher_psnr", "shadow_correction_capture",
        "shadow_correction_alignment", "shadow_teacher_gap_closed",
    )
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for group, values in summary.items():
            for method, metric in values["methods"].items():
                shadow = metric["shadow_teacher_agreement"]
                writer.writerow({
                    group_name: group,
                    "images": values["images"],
                    "scenes": values["scenes"],
                    "method": method,
                    "reference_psnr": csv_value(metric["reference"]["psnr"]["mean"]),
                    "reference_ssim": csv_value(metric["reference"]["ssim"]["mean"]),
                    "teacher_psnr": csv_value(metric["teacher_agreement"]["psnr"]["mean"]),
                    "teacher_ssim": csv_value(metric["teacher_agreement"]["ssim"]["mean"]),
                    "shadow_teacher_mae": csv_value(shadow["teacher_mae"]["mean"]),
                    "shadow_teacher_psnr": csv_value(shadow["teacher_psnr"]["mean"]),
                    "shadow_correction_capture": csv_value(shadow["correction_capture"]["mean"]),
                    "shadow_correction_alignment": csv_value(shadow["correction_alignment"]["mean"]),
                    "shadow_teacher_gap_closed": csv_value(shadow["teacher_gap_closed"]["mean"]),
                })


def write_contact_sheet(
    path: Path,
    dataset: DistillationDataset,
    metadata: list[dict[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    indices: list[int],
) -> None:
    headings = ("Noisy", "Teacher-only student", "SCUNet", "Reference")
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
            images = (
                tensor_image(sample["noisy"]),
                tensor_image(model(noisy)[0]),
                tensor_image(sample["teacher"]),
                tensor_image(sample["clean"]),
            )
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
    sheet.save(path, compress_level=6)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    seed_everything(int(config["project"]["seed"]))
    device = choose_device(args.device)
    manifest_path = resolve_paper_path(config["data"]["manifest"])
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DistillationDataset(
        manifest_path, root=cache_root, split="validation", augment=False
    )
    if len(dataset) != 850:
        raise ValueError(f"Expected the complete 850-sample validation set, got {len(dataset)}")
    metadata = load_manifest_metadata(manifest_path, dataset)
    content_identity = validation_content_identity(dataset)
    model, checkpoint_metadata, _ = load_checkpoint_model(
        "nind_teacher_only",
        args.checkpoint,
        device,
        manifest_path,
        allow_incomplete=False,
    )
    manifest_sha256 = sha256_file(manifest_path)
    if checkpoint_metadata["manifest_sha256"] != manifest_sha256:
        raise ValueError("Checkpoint and selected manifest hashes differ")

    workers = int(config["training"]["workers"] if args.workers is None else args.workers)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    border = int(config["metrics"]["border_crop"])
    window_size = int(config["metrics"]["ssim_window_size"])
    sigma = float(config["metrics"]["ssim_sigma"])
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Validating teacher-only checkpoint"):
            noisy = batch["noisy"].to(device, non_blocking=True)
            reference = batch["clean"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            student = model(noisy)
            outputs = {"noisy": noisy, "student": student, "teacher": teacher}
            reference_metrics = {
                label: metric_pair(
                    output, reference, border=border, window_size=window_size, sigma=sigma
                )
                for label, output in outputs.items()
            }
            teacher_metrics = {
                label: metric_pair(
                    output, teacher, border=border, window_size=window_size, sigma=sigma
                )
                for label, output in outputs.items()
                if label != "teacher"
            }
            shadows = shadow_metrics(outputs, noisy, teacher, border)
            for local_index, selection_index in enumerate(batch["index"].tolist()):
                selection_index = int(selection_index)
                raw = metadata[selection_index]
                record = dataset.records[selection_index]
                row_metrics: dict[str, Any] = {}
                for label in METHODS:
                    reference_pair = reference_metrics[label]
                    teacher_pair = teacher_metrics.get(label)
                    row_metrics[label] = {
                        "reference": {
                            "psnr": finite_float(reference_pair[0][local_index]),
                            "ssim": finite_float(reference_pair[1][local_index]),
                        },
                        "teacher": {
                            "psnr": finite_float(teacher_pair[0][local_index]) if teacher_pair else None,
                            "ssim": finite_float(teacher_pair[1][local_index]) if teacher_pair else 1.0,
                        },
                    }
                rows.append({
                    "index": selection_index,
                    "id": str(raw.get("id", selection_index)),
                    "dataset": record.dataset,
                    "scene": record.scene,
                    "noise_level": str(raw.get("noise_level", "unknown")),
                    "iso": raw.get("iso"),
                    "supervision": str(raw.get("supervision", "unknown")),
                    "metrics": row_metrics,
                    "shadow": shadows[local_index],
                })
    if len(rows) != len(dataset):
        raise RuntimeError(f"Processed {len(rows)} of {len(dataset)} validation samples")

    per_image_jsonl = output_dir / "per_image.jsonl"
    with per_image_jsonl.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_per_image_csv(output_dir / "per_image.csv", rows)
    summary = build_dataset_summary(rows)
    nind_summary = build_nind_summary(rows)
    write_summary_csv(output_dir / "summary.csv", summary, "dataset")
    write_summary_csv(output_dir / "nind_by_noise_level.csv", nind_summary, "noise_level")

    selected = contact_selection(rows, args.contact_count)
    contact_path = output_dir / "contact_sheet.png"
    write_contact_sheet(
        contact_path, dataset, metadata, model, device, selected["indices"]
    )
    evaluator_path = Path(__file__).resolve()
    provenance_payload = {
        "checkpoint_sha256": checkpoint_metadata["sha256"],
        "manifest_sha256": manifest_sha256,
        "validation_content_sha256": content_identity["sha256"],
        "evaluator_sha256": sha256_file(evaluator_path),
        "samples": len(rows),
    }
    evaluation_id = hashlib.sha256(
        json.dumps(provenance_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "provenance": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
            "validation_content": content_identity,
            "checkpoint": checkpoint_metadata,
            "evaluator": {"path": str(evaluator_path), "sha256": sha256_file(evaluator_path)},
            "device": str(device),
            "samples": len(rows),
            "dataset_counts": {
                name: sum(row["dataset"] == name for row in rows)
                for name in sorted({row["dataset"] for row in rows})
            },
            "metrics": {
                "color_space": "RGB [0,1]",
                "border_crop": border,
                "ssim_window_size": window_size,
                "ssim_sigma": sigma,
                "shadow_mask": "Rec.709 luma(noisy RGB) < 0.25 after border crop",
                "shadow_agreement_target": "cached SCUNet teacher",
            },
            "contact_sheet_selection": selected,
        },
        "summary": summary,
        "nind_by_noise_level": nind_summary,
        "artifacts": {
            "per_image_jsonl": str(per_image_jsonl),
            "per_image_csv": str(output_dir / "per_image.csv"),
            "summary_csv": str(output_dir / "summary.csv"),
            "nind_by_noise_level_csv": str(output_dir / "nind_by_noise_level.csv"),
            "contact_sheet": str(contact_path),
        },
    }
    atomic_json(output_dir / "summary.json", report)
    print(json.dumps({
        "evaluation_id": evaluation_id,
        "samples": len(rows),
        "device": str(device),
        "student": summary["all"]["methods"]["student"],
        "nind_student": summary["nind"]["methods"]["student"],
        "output_dir": str(output_dir),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
