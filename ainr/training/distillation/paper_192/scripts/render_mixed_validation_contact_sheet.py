#!/usr/bin/env python3
"""Render an auditable contact sheet for the final mixed validation model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from common import atomic_json, resolve_paper_path, seed_everything, sha256_file
from src.metrics import gaussian_ssim_per_image, psnr_per_image
from src.mixed_dataset import MixedDistillationDataset
from src.noise_conditioning import model_input_from_config
from src.student import student_from_checkpoint


PAPER_ROOT = Path(__file__).resolve().parents[1]
TILE = 192
TITLE_HEIGHT = 72
DESCRIPTOR_HEIGHT = 44
FOOTER_HEIGHT = 40
ROW_HEIGHT = DESCRIPTOR_HEIGHT + TILE + FOOTER_HEIGHT
PANEL_LABELS = ("Noisy input", "Student", "SCUNet teacher", "Reference target")

# Fixed independently of model output so future checkpoints can use the same visual set.
SELECTION = (
    ("4d01f037a82cbb12_00", "dark/noisy"),
    ("831fe17530efd57a_01", "midtone detail"),
    ("f62de5d9b9bbfcd0_00", "ISO 6400 shadow/detail"),
    ("d655dba4caac75da_00", "H2 detail"),
    ("8ae0ea808065154e_01", "H3 severe noise"),
    ("9fb230f967eb55bc_02", "ISO 3200 detail/shadow"),
    ("15e4ddaea1663b57_14", "ISO 6400 dark"),
    ("82196c65397e9f26_06", "dark/noisy detail"),
    ("a64af0d9e46e7fbc_04", "midtone severe noise"),
    ("1c0437155792b9cb_01", "ISO 1600 detail"),
    ("03bed52ff44ce449_00", "ISO 6400 deep shadow"),
    ("88b13447cf5f50ad_05", "ISO 12800 detail"),
    ("b17014f4d6b26134_00", "deep-shadow detail"),
    ("e88f4875587d9256_02", "strong paired residual"),
    ("499802667ae78723_03", "bright/detail coverage"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PAPER_ROOT
        / "evaluation/domain_expansion/uhd_snic_alpha_0p7_final/best_snapshot.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_ROOT / "evaluation/domain_expansion/uhd_snic_alpha_0p7_final",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    return args


def choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def uint8_tensor(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().permute(1, 2, 0).cpu().numpy()
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def panel_hash(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def fit_text(
    draw: ImageDraw.ImageDraw,
    value: str,
    text_font: ImageFont.ImageFont,
    width: int,
) -> str:
    if draw.textlength(value, font=text_font) <= width:
        return value
    suffix = "..."
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = value[:middle] + suffix
        if draw.textlength(candidate, font=text_font) <= width:
            low = middle
        else:
            high = middle - 1
    return value[:low] + suffix


def metric_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    border: int,
    window_size: int,
    sigma: float,
) -> tuple[list[float], list[float]]:
    psnr = psnr_per_image(prediction, target, border=border)
    ssim = gaussian_ssim_per_image(
        prediction,
        target,
        border=border,
        window_size=window_size,
        sigma=sigma,
    )
    return ([float(value) for value in psnr], [float(value) for value in ssim])


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    records = document.get("records") if isinstance(document, dict) else document
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Invalid mixed manifest: {path}")
    return [row for row in records if row.get("split") == "validation"]


def display_noise(record: dict[str, Any]) -> str:
    value = record.get("iso", record.get("noise_level", "unspecified"))
    text = str(value)
    if text.startswith("low-light capture"):
        return "low-light"
    return text


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "validation_contact_sheet.png"
    jpg_path = output_dir / "validation_contact_sheet.jpg"
    report_path = output_dir / "validation_contact_sheet.json"
    for path in (png_path, jpg_path, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed_everything(int(config["project"]["seed"]))
    manifest_path = resolve_paper_path(config["data"]["manifest"])
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    validation_records = load_manifest_records(manifest_path)
    dataset = MixedDistillationDataset(
        manifest_path,
        root=cache_root,
        split="validation",
        augment=False,
    )
    if len(dataset) != len(validation_records):
        raise ValueError("Mixed validation dataset and manifest row counts differ")

    id_to_index: dict[str, int] = {}
    for index, record in enumerate(validation_records):
        record_id = str(record.get("id", ""))
        if not record_id or record_id in id_to_index:
            raise ValueError(f"Missing or duplicate validation record id: {record_id!r}")
        id_to_index[record_id] = index

    selected: list[tuple[int, dict[str, Any], str, dict[str, Any]]] = []
    for record_id, reason in SELECTION:
        if record_id not in id_to_index:
            raise ValueError(f"Selected record is absent from the validation manifest: {record_id}")
        index = id_to_index[record_id]
        raw = validation_records[index]
        if float(raw["gt_weight"]) <= 0.0:
            raise ValueError(f"Selected record lacks clean/reference supervision: {record_id}")
        sample = dataset[index]
        for key in ("dataset", "scene", "split", "supervision"):
            if str(sample[key]) != str(raw[key]):
                raise ValueError(f"Dataset and manifest disagree for {record_id}: {key}")
        selected.append((index, raw, reason, sample))
    if len(selected) != len(SELECTION) or len({row[0] for row in selected}) != len(SELECTION):
        raise ValueError("Contact-sheet selection is incomplete or contains duplicates")
    selection_datasets = set(config["training"]["selection_datasets"])
    if {str(row[1]["dataset"]) for row in selected} != selection_datasets:
        raise ValueError(
            "Contact-sheet selection does not cover every real checkpoint-selection dataset"
        )
    panel_labels = (
        PANEL_LABELS[0],
        f"Student e{int(checkpoint['epoch'])}",
        *PANEL_LABELS[2:],
    )

    device = choose_device(args.device)
    model = student_from_checkpoint(checkpoint).to(device).eval()
    model_config = checkpoint.get("config", {}).get("model", {})
    noisy = torch.stack([row[3]["noisy"] for row in selected])
    clean = torch.stack([row[3]["clean"] for row in selected])
    teacher = torch.stack([row[3]["teacher"] for row in selected])
    started = time.perf_counter()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(selected), args.batch_size):
            batch = noisy[start : start + args.batch_size].to(device)
            model_input = model_input_from_config(batch, model_config)
            predictions.append(model(model_input).clamp(0.0, 1.0).cpu())
    prediction = torch.cat(predictions)
    inference_seconds = time.perf_counter() - started

    metrics_config = config["metrics"]
    metric_args = {
        "border": int(metrics_config["border_crop"]),
        "window_size": int(metrics_config["ssim_window_size"]),
        "sigma": float(metrics_config["ssim_sigma"]),
    }
    noisy_reference = metric_values(noisy, clean, **metric_args)
    student_reference = metric_values(prediction, clean, **metric_args)
    teacher_reference = metric_values(teacher, clean, **metric_args)
    student_teacher = metric_values(prediction, teacher, **metric_args)

    width = TILE * len(panel_labels)
    height = TITLE_HEIGHT + ROW_HEIGHT * len(selected)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font, heading_font, body_font, small_font = font(18), font(12), font(10), font(9)
    draw.text(
        (7, 5),
        f"Final mixed validation contact sheet - epoch {int(checkpoint['epoch'])}",
        fill="black",
        font=title_font,
    )
    draw.text(
        (7, 29),
        "Fixed clean-supervised validation samples; native 192 x 192 pixels",
        fill=(55, 55, 55),
        font=body_font,
    )
    for column, label in enumerate(panel_labels):
        left = column * TILE
        text_width = draw.textlength(label, font=heading_font)
        draw.text(
            (left + max(4.0, (TILE - text_width) / 2.0), 49),
            label,
            fill="black",
            font=heading_font,
        )

    rows: list[dict[str, Any]] = []
    panel_arrays: list[tuple[np.ndarray, ...]] = []
    luma_weights = torch.tensor((0.2126, 0.7152, 0.0722)).view(3, 1, 1)
    for row_number, (index, raw, reason, sample) in enumerate(selected):
        top = TITLE_HEIGHT + row_number * ROW_HEIGHT
        if row_number % 2:
            draw.rectangle((0, top, width - 1, top + DESCRIPTOR_HEIGHT - 1), fill=(245, 245, 245))
            draw.rectangle(
                (0, top + DESCRIPTOR_HEIGHT + TILE, width - 1, top + ROW_HEIGHT - 1),
                fill=(245, 245, 245),
            )
        descriptor = (
            f"{str(raw['dataset']).upper()} | {reason} | scene {raw['scene']}"
        )
        metadata = (
            f"id {raw['id']} | noise {display_noise(raw)} | {raw['supervision']} | "
            f"GT {float(raw['gt_weight']):.1f} / KD {float(raw['kd_weight']):.1f}"
        )
        draw.text(
            (5, top + 3),
            fit_text(draw, descriptor, body_font, width - 10),
            fill="black",
            font=body_font,
        )
        draw.text(
            (5, top + 21),
            fit_text(draw, metadata, small_font, width - 10),
            fill=(55, 55, 55),
            font=small_font,
        )
        arrays = (
            uint8_tensor(sample["noisy"]),
            uint8_tensor(prediction[row_number]),
            uint8_tensor(sample["teacher"]),
            uint8_tensor(sample["clean"]),
        )
        panel_arrays.append(arrays)
        image_top = top + DESCRIPTOR_HEIGHT
        for column, array in enumerate(arrays):
            sheet.paste(Image.fromarray(array, mode="RGB"), (column * TILE, image_top))
        footer_top = image_top + TILE
        metric_text = (
            f"Reference PSNR/SSIM: noisy {noisy_reference[0][row_number]:.2f}/"
            f"{noisy_reference[1][row_number]:.3f} | student "
            f"{student_reference[0][row_number]:.2f}/{student_reference[1][row_number]:.3f} | "
            f"SCUNet {teacher_reference[0][row_number]:.2f}/{teacher_reference[1][row_number]:.3f}"
        )
        draw.text(
            (5, footer_top + 4),
            fit_text(draw, metric_text, small_font, width - 10),
            fill="black",
            font=small_font,
        )
        luma = (sample["noisy"] * luma_weights).sum(dim=0)
        detail_text = (
            f"Student-SCUNet {student_teacher[0][row_number]:.2f} dB/"
            f"{student_teacher[1][row_number]:.3f} | input luma {float(luma.mean()):.3f} | "
            f"shadow pixels {float((luma < 0.25).float().mean()):.0%}"
        )
        draw.text(
            (5, footer_top + 21),
            fit_text(draw, detail_text, small_font, width - 10),
            fill=(55, 55, 55),
            font=small_font,
        )
        rows.append(
            {
                "row": row_number,
                "manifest_validation_index": index,
                "id": str(raw["id"]),
                "dataset": str(raw["dataset"]),
                "scene": str(raw["scene"]),
                "noise": display_noise(raw),
                "reason": reason,
                "supervision": str(raw["supervision"]),
                "gt_weight": float(raw["gt_weight"]),
                "kd_weight": float(raw["kd_weight"]),
                "input_mean_luma": float(luma.mean()),
                "input_shadow_fraction": float((luma < 0.25).float().mean()),
                "metrics": {
                    "noisy_reference": {
                        "psnr": noisy_reference[0][row_number],
                        "ssim": noisy_reference[1][row_number],
                    },
                    "student_reference": {
                        "psnr": student_reference[0][row_number],
                        "ssim": student_reference[1][row_number],
                    },
                    "teacher_reference": {
                        "psnr": teacher_reference[0][row_number],
                        "ssim": teacher_reference[1][row_number],
                    },
                    "student_teacher": {
                        "psnr": student_teacher[0][row_number],
                        "ssim": student_teacher[1][row_number],
                    },
                },
                "panel_sha256": {
                    label: panel_hash(array)
                    for label, array in zip(panel_labels, arrays, strict=True)
                },
            }
        )

    png_tmp = png_path.with_suffix(".png.tmp")
    jpg_tmp = jpg_path.with_suffix(".jpg.tmp")
    sheet.save(png_tmp, format="PNG", optimize=True)
    png_tmp.replace(png_path)
    sheet.save(jpg_tmp, format="JPEG", quality=97, subsampling=0, optimize=True)
    jpg_tmp.replace(jpg_path)

    with Image.open(png_path) as rendered:
        if rendered.mode != "RGB" or rendered.size != (width, height):
            raise ValueError("Rendered contact sheet has unexpected mode or dimensions")
        rendered_array = np.asarray(rendered)
    for row_number, arrays in enumerate(panel_arrays):
        image_top = TITLE_HEIGHT + row_number * ROW_HEIGHT + DESCRIPTOR_HEIGHT
        for column, expected in enumerate(arrays):
            actual = rendered_array[
                image_top : image_top + TILE,
                column * TILE : (column + 1) * TILE,
            ]
            if not np.array_equal(actual, expected):
                raise ValueError(f"Rendered panel pixels changed at row {row_number}, column {column}")
    if all(np.array_equal(row[0], row[1]) for row in panel_arrays):
        raise ValueError("Every student panel unexpectedly equals its noisy input")
    if all(np.array_equal(row[1], row[2]) for row in panel_arrays):
        raise ValueError("Every student panel unexpectedly equals the SCUNet teacher")

    snapshot_manifest_path = checkpoint_path.parent / "snapshot_manifest.json"
    snapshot_manifest = (
        json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
        if snapshot_manifest_path.is_file()
        else None
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if snapshot_manifest is not None and snapshot_manifest["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Checkpoint differs from its snapshot manifest")
    report = {
        "schema_version": 1,
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "epoch": int(checkpoint["epoch"]),
            "run_fingerprint": str(checkpoint["run_fingerprint"]),
        },
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "preprocessing_version": str(config["project"]["preprocessing_version"]),
        "validation": {
            "total_rows": len(dataset),
            "clean_supervised_rows": sum(record.gt_weight > 0 for record in dataset.records),
            "selected_rows": len(rows),
            "selection_policy": "fixed model-independent allow-list covering all six domains",
        },
        "render": {
            "png": str(png_path),
            "png_sha256": sha256_file(png_path),
            "jpg_preview": str(jpg_path),
            "jpg_sha256": sha256_file(jpg_path),
            "width": width,
            "height": height,
            "mode": "RGB",
            "tile_size": TILE,
            "title_height": TITLE_HEIGHT,
            "row_height": ROW_HEIGHT,
            "descriptor_height": DESCRIPTOR_HEIGHT,
            "footer_height": FOOTER_HEIGHT,
            "panel_labels": list(panel_labels),
            "panel_pixels_verified_exact": True,
        },
        "inference": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "batch_size": args.batch_size,
            "seconds": inference_seconds,
        },
        "metric_configuration": metric_args,
        "rows": rows,
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "png": str(png_path),
                "jpg": str(jpg_path),
                "report": str(report_path),
                "rows": len(rows),
                "dimensions": [width, height],
                "inference_seconds": inference_seconds,
                "png_sha256": report["render"]["png_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
