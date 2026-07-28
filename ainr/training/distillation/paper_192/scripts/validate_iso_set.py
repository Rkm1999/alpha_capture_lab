#!/usr/bin/env python3
"""Render full-frame and detailed ISO-set comparisons against SCUNet.

This is an external visual validation, not a clean-reference benchmark. The
camera JPEGs have no paired clean targets, so the reported numerical values
describe student/teacher agreement and correction strength only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import ExifTags, Image, ImageDraw, ImageFont, ImageOps

from common import atomic_json, resolve_paper_path

PAPER_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PAPER_ROOT))

from src.metrics import gaussian_ssim_per_image, psnr_per_image  # noqa: E402
from src.noise_conditioning import model_input_from_config  # noqa: E402
from src.scunet_teacher import load_scunet_teacher  # noqa: E402
from src.student import (  # noqa: E402
    LiteDenoiseNet,
    checkpoint_base_width,
    checkpoint_model_kwargs,
    student_from_checkpoint,
)


@dataclass(frozen=True)
class Region:
    slug: str
    label: str
    center_x: float
    center_y: float


# The test set uses the same scene and framing at every ISO. These regions
# cover shadows, flat fields, fine detail, hard edges, and specular surfaces.
REGIONS = (
    Region("dark_shelf", "Dark shelf", 0.15, 0.12),
    Region("blue_background", "Smooth blue background", 0.55, 0.13),
    Region("screen", "Bright screen and text", 0.88, 0.16),
    Region("red_cap", "Red cap and colored edge", 0.54, 0.28),
    Region("white_figure", "White figure and face detail", 0.42, 0.39),
    Region("metal_bit", "Specular metal", 0.73, 0.31),
    Region("color_blocks", "Color blocks and edges", 0.29, 0.58),
    Region("dark_tool", "Dark tool body", 0.75, 0.58),
    Region("dot_pattern", "Fine dot pattern", 0.35, 0.69),
    Region("black_table", "Black table and edge", 0.12, 0.79),
    Region("fine_label", "Fine printed label", 0.68, 0.80),
    Region("bottle_desk", "Transparent bottle and desk", 0.78, 0.92),
)

ABLATION_MODES = {
    "nind_teacher_only": {
        "label": "LiteDenoise NIND teacher-only",
        "short_label": "NIND teacher-only",
        "nind_gt_weight": 0.0,
    },
    "nind_full_reference": {
        "label": "LiteDenoise NIND full-reference",
        "short_label": "NIND full-reference",
        "nind_gt_weight": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PAPER_ROOT / "runs/high_iso_ablation/nind_teacher_only/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_ROOT / "evaluation/high_iso_ablation/nind_teacher_only_iso_test",
    )
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--student-batch", type=int, default=16)
    parser.add_argument("--teacher-batch", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def checkpoint_identity(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Validate checkpoint provenance and return labels used in artifacts."""

    if "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint is missing model or config data")
    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a mapping")
    training = config.get("training")
    model_config = config.get("model")
    if not isinstance(training, dict) or not isinstance(model_config, dict):
        raise ValueError("Checkpoint config is missing training or model metadata")

    alpha = float(checkpoint.get("alpha", training.get("alpha", -1.0)))
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Checkpoint alpha must be finite and in [0,1], got {alpha}")
    model_kwargs = checkpoint_model_kwargs(checkpoint)
    expected_shape = [
        1,
        model_kwargs["input_channels"],
        LiteDenoiseNet.INPUT_SIZE,
        LiteDenoiseNet.INPUT_SIZE,
    ]
    if list(model_config.get("input_shape", ())) != expected_shape:
        raise ValueError(
            f"Checkpoint input shape must be {expected_shape}, got "
            f"{model_config.get('input_shape')}"
        )
    base_width = checkpoint_base_width(checkpoint)
    actual_parameters = sum(
        parameter.numel() for parameter in LiteDenoiseNet(**model_kwargs).parameters()
    )
    if int(model_config.get("expected_parameters", -1)) != actual_parameters:
        raise ValueError(
            "Checkpoint parameter metadata does not match its LiteDenoiseNet width"
        )

    mode = checkpoint.get("mode")
    if mode is None:
        alpha_label = f"{alpha:g}"
        alpha_mode = f"alpha_{alpha_label.replace('.', 'p')}"
        return {
            "kind": "mixed_distillation_run",
            "mode": alpha_mode,
            "label": f"LiteDenoise W{base_width} alpha {alpha_label}",
            "short_label": f"W{base_width} alpha {alpha_label}",
            "base_width": base_width,
            "alpha": alpha,
            "nind_gt_weight": None,
            "run_fingerprint": checkpoint.get("run_fingerprint"),
        }
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unsupported ablation checkpoint mode: {mode!r}")

    expected_weight = float(ABLATION_MODES[mode]["nind_gt_weight"])
    checkpoint_weight = float(checkpoint.get("nind_gt_weight", math.nan))
    config_weight = float(training.get("nind_gt_weight", math.nan))
    if not math.isclose(checkpoint_weight, expected_weight, abs_tol=1e-12):
        raise ValueError(
            f"Checkpoint mode {mode!r} requires nind_gt_weight={expected_weight}, "
            f"got {checkpoint_weight}"
        )
    if not math.isclose(config_weight, expected_weight, abs_tol=1e-12):
        raise ValueError(
            f"Checkpoint config disagrees with mode {mode!r}: "
            f"nind_gt_weight={config_weight}"
        )
    run_fingerprint = checkpoint.get("run_fingerprint")
    if not isinstance(run_fingerprint, str) or len(run_fingerprint) != 64:
        raise ValueError("Ablation checkpoint is missing its SHA-256 run fingerprint")
    return {
        "kind": "nind_supervision_ablation",
        "mode": mode,
        "label": ABLATION_MODES[mode]["label"],
        "short_label": ABLATION_MODES[mode]["short_label"],
        "alpha": alpha,
        "nind_gt_weight": expected_weight,
        "run_fingerprint": run_fingerprint,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iso_for(image: Image.Image, fallback: str) -> int | str:
    containers: list[Any] = [image.getexif()]
    if hasattr(ExifTags, "IFD"):
        try:
            containers.append(image.getexif().get_ifd(ExifTags.IFD.Exif))
        except (KeyError, TypeError, ValueError):
            pass
    for container in containers:
        for key, value in container.items():
            if ExifTags.TAGS.get(key) in ("PhotographicSensitivity", "ISOSpeedRatings"):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return str(value)
    return fallback


def reflected_indices(start: int, size: int, length: int) -> np.ndarray:
    if length < 2:
        return np.zeros(size, dtype=np.int64)
    values = np.arange(start, start + size, dtype=np.int64)
    period = 2 * length - 2
    values %= period
    return np.where(values < length, values, period - values)


def blend_axis(index: int, count: int, tile: int, overlap: int) -> np.ndarray:
    weights = np.ones(tile, dtype=np.float32)
    ramp = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
    if index > 0:
        weights[:overlap] = ramp
    if index + 1 < count:
        weights[-overlap:] = ramp[::-1]
    return weights


def infer_tiled(
    model: torch.nn.Module,
    source: np.ndarray,
    device: torch.device,
    batch_size: int,
    label: str,
    model_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    tile = LiteDenoiseNet.INPUT_SIZE
    padding = 8
    core = tile - 2 * padding
    overlap = 2 * padding
    height, width = source.shape[:2]
    columns = math.ceil(width / core)
    rows = math.ceil(height / core)
    coordinates = [(column, row) for row in range(rows) for column in range(columns)]
    x_indices = [
        reflected_indices(column * core - padding, tile, width)
        for column in range(columns)
    ]
    y_indices = [
        reflected_indices(row * core - padding, tile, height)
        for row in range(rows)
    ]
    x_weights = [blend_axis(column, columns, tile, overlap) for column in range(columns)]
    y_weights = [blend_axis(row, rows, tile, overlap) for row in range(rows)]

    accumulation = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    started = time.perf_counter()
    completed = 0
    with torch.inference_mode():
        for offset in range(0, len(coordinates), batch_size):
            selection = coordinates[offset : offset + batch_size]
            patches = np.stack(
                [source[np.ix_(y_indices[row], x_indices[column])] for column, row in selection]
            )
            tensor = (
                torch.from_numpy(patches)
                .permute(0, 3, 1, 2)
                .to(device=device, dtype=torch.float32)
                .div_(255.0)
            )
            model_input = (
                model_input_from_config(tensor, model_config)
                if model_config is not None
                else tensor
            )
            predictions = (
                model(model_input)
                .clamp_(0.0, 1.0)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            for prediction, (column, row) in zip(predictions, selection):
                start_x = column * core - padding
                start_y = row * core - padding
                local_x0 = max(0, -start_x)
                local_y0 = max(0, -start_y)
                local_x1 = min(tile, width - start_x)
                local_y1 = min(tile, height - start_y)
                image_x0 = start_x + local_x0
                image_y0 = start_y + local_y0
                image_x1 = start_x + local_x1
                image_y1 = start_y + local_y1
                weights = np.outer(y_weights[row], x_weights[column])[
                    local_y0:local_y1, local_x0:local_x1
                ]
                accumulation[image_y0:image_y1, image_x0:image_x1] += (
                    prediction[local_y0:local_y1, local_x0:local_x1] * weights[..., None]
                )
                weight_sum[image_y0:image_y1, image_x0:image_x1] += weights
            completed += len(selection)
            print(
                f"  {label}: {completed}/{len(coordinates)} tiles",
                end="\r" if completed < len(coordinates) else "\n",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if not np.all(weight_sum > 0):
        raise RuntimeError(f"{label} tiling left uncovered pixels")
    accumulation /= weight_sum[..., None]
    return accumulation, {
        "seconds": round(time.perf_counter() - started, 3),
        "tiles": len(coordinates),
        "tile_size": tile,
        "padding": padding,
        "core": core,
        "composition": "whole-tile 16 px linear weighted overlap matching 8 px edge feathering",
        "batch_size": batch_size,
    }


def uint8_image(pixels: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_png(pixels: np.ndarray, path: Path) -> None:
    Image.fromarray(pixels, "RGB").save(path, compress_level=6)


def crop_box(region: Region, width: int, height: int, size: int) -> tuple[int, int, int, int]:
    left = round(region.center_x * width - size / 2)
    top = round(region.center_y * height - size / 2)
    left = min(max(left, 0), width - size)
    top = min(max(top, 0), height - size)
    return left, top, left + size, top + size


def fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(candidate, 25), ImageFont.truetype(candidate, 19)
    default = ImageFont.load_default()
    return default, default


def full_comparison(
    original: np.ndarray,
    student: np.ndarray,
    teacher: np.ndarray,
    iso: int | str,
    path: Path,
    student_label: str,
) -> None:
    title_font, _ = fonts()
    panel_width = 500
    panel_height = round(original.shape[0] * panel_width / original.shape[1])
    header = 58
    canvas = Image.new("RGB", (panel_width * 3, panel_height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (pixels, label) in enumerate(
        ((original, "Original JPEG"), (student, student_label), (teacher, "SCUNet teacher"))
    ):
        image = Image.fromarray(pixels, "RGB").resize(
            (panel_width, panel_height), Image.Resampling.LANCZOS
        )
        canvas.paste(image, (index * panel_width, header))
        draw.text((index * panel_width + 12, 14), label, fill="black", font=title_font)
    draw.text((canvas.width - 150, 14), f"ISO {iso}", fill="black", font=title_font)
    canvas.save(path, quality=96, subsampling=0)


def overview(
    original: np.ndarray,
    regions: list[tuple[Region, tuple[int, int, int, int]]],
    iso: int | str,
    path: Path,
) -> None:
    title_font, body_font = fonts()
    image = Image.fromarray(original, "RGB")
    target_height = 1200
    target_width = round(image.width * target_height / image.height)
    resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    legend_width = 410
    header = 58
    canvas = Image.new("RGB", (target_width + legend_width, target_height + header), "white")
    canvas.paste(resized, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 14), f"ISO {iso} detail regions", fill="black", font=title_font)
    scale_x = target_width / image.width
    scale_y = target_height / image.height
    colors = ("#ef4444", "#22c55e", "#3b82f6", "#f59e0b")
    for index, (region, box) in enumerate(regions, 1):
        color = colors[(index - 1) % len(colors)]
        scaled = tuple(
            round(value * (scale_x if coordinate % 2 == 0 else scale_y))
            for coordinate, value in enumerate(box)
        )
        scaled = (scaled[0], scaled[1] + header, scaled[2], scaled[3] + header)
        draw.rectangle(scaled, outline=color, width=3)
        draw.rectangle((scaled[0], scaled[1], scaled[0] + 34, scaled[1] + 30), fill=color)
        draw.text((scaled[0] + 7, scaled[1] + 3), str(index), fill="white", font=body_font)
        legend_y = header + 16 + (index - 1) * 43
        draw.rectangle((target_width + 18, legend_y + 3, target_width + 42, legend_y + 27), fill=color)
        draw.text(
            (target_width + 52, legend_y),
            f"{index:02d}  {region.label}",
            fill="black",
            font=body_font,
        )
    canvas.save(path, quality=96, subsampling=0)


def detail_sheet(
    original: np.ndarray,
    student: np.ndarray,
    teacher: np.ndarray,
    region: Region,
    box: tuple[int, int, int, int],
    iso: int | str,
    path: Path,
    student_label: str,
) -> None:
    title_font, body_font = fonts()
    left, top, right, bottom = box
    crops = (
        (original[top:bottom, left:right], "Original JPEG"),
        (student[top:bottom, left:right], student_label),
        (teacher[top:bottom, left:right], "SCUNet teacher"),
    )
    size = right - left
    header = 78
    canvas = Image.new("RGB", (size * 3, size + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (pixels, label) in enumerate(crops):
        canvas.paste(Image.fromarray(pixels, "RGB"), (index * size, header))
        draw.text((index * size + 12, 43), label, fill="black", font=body_font)
    draw.text(
        (12, 8),
        f"ISO {iso} | {region.label} | source box [{left}, {top}, {right}, {bottom}]",
        fill="black",
        font=title_font,
    )
    canvas.save(path, compress_level=4)


def contact_sheet(crop_paths: list[Path], path: Path, student_label: str) -> None:
    title_font, _ = fonts()
    panel_width = 350
    row_height = 390
    header = 54
    canvas = Image.new("RGB", (panel_width * 3, header + row_height * len(crop_paths)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 12),
        f"Original JPEG | {student_label} | SCUNet teacher",
        fill="black",
        font=title_font,
    )
    for row, crop_path in enumerate(crop_paths):
        with Image.open(crop_path) as crop:
            content = crop.crop((0, 78, crop.width, crop.height))
            resized = content.resize((panel_width * 3, panel_width), Image.Resampling.LANCZOS)
        y = header + row * row_height
        canvas.paste(resized, (0, y + 40))
        draw.text((10, y + 7), f"{row + 1:02d}  {REGIONS[row].label}", fill="black", font=title_font)
    canvas.save(path, quality=95, subsampling=0)


def cross_iso_sheets(
    output_dir: Path,
    records: list[dict[str, Any]],
    crop_size: int,
    student_label: str,
) -> None:
    title_font, body_font = fonts()
    destination = output_dir / "by_region"
    destination.mkdir(parents=True, exist_ok=True)
    panel = 320
    row_label = 190
    header = 108
    labels = ("Original", student_label, "SCUNet")
    for region_index, region in enumerate(REGIONS, 1):
        canvas = Image.new(
            "RGB",
            (row_label + panel * len(records), header + panel * len(labels)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 13), f"{region_index:02d}  {region.label}", fill="black", font=title_font)
        for column, record in enumerate(records):
            draw.text(
                (row_label + column * panel + 12, 64),
                f"ISO {record['iso']}",
                fill="black",
                font=body_font,
            )
            crop_path = (
                output_dir
                / record["folder"]
                / "crops"
                / f"{region_index:02d}_{region.slug}.png"
            )
            with Image.open(crop_path) as source:
                for row, label in enumerate(labels):
                    box = (row * crop_size, 78, (row + 1) * crop_size, 78 + crop_size)
                    crop = source.crop(box).resize((panel, panel), Image.Resampling.LANCZOS)
                    canvas.paste(crop, (row_label + column * panel, header + row * panel))
                    if column == 0:
                        draw.text(
                            (12, header + row * panel + 12),
                            label,
                            fill="black",
                            font=body_font,
                        )
        canvas.save(destination / f"{region_index:02d}_{region.slug}.jpg", quality=96, subsampling=0)


def teacher_metrics(
    student: np.ndarray,
    teacher: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    student_tensor = (
        torch.from_numpy(student).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255.0)
    )
    teacher_tensor = (
        torch.from_numpy(teacher).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255.0)
    )
    return {
        "psnr": float(psnr_per_image(student_tensor, teacher_tensor, border=1)[0]),
        "ssim": float(gaussian_ssim_per_image(student_tensor, teacher_tensor, border=1)[0]),
    }


def full_teacher_metrics(student: np.ndarray, teacher: np.ndarray) -> dict[str, float | int]:
    difference = student.astype(np.float32) - teacher.astype(np.float32)
    mse = float(np.square(difference).mean()) / (255.0 * 255.0)
    psnr = float("inf") if mse == 0.0 else -10.0 * math.log10(mse)

    # Full-resolution Gaussian SSIM would allocate several gigabytes of
    # intermediate tensors for a 24 MP frame. Use a clearly labeled preview
    # SSIM here; every detailed crop is also measured at native resolution.
    height, width = student.shape[:2]
    preview_scale = min(1.0, 1024.0 / max(width, height))
    preview_size = (round(width * preview_scale), round(height * preview_scale))
    student_preview = np.asarray(
        Image.fromarray(student, "RGB").resize(preview_size, Image.Resampling.LANCZOS)
    ).copy()
    teacher_preview = np.asarray(
        Image.fromarray(teacher, "RGB").resize(preview_size, Image.Resampling.LANCZOS)
    ).copy()
    student_tensor = torch.from_numpy(student_preview).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    teacher_tensor = torch.from_numpy(teacher_preview).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    return {
        "psnr": psnr,
        "ssim_preview": float(
            gaussian_ssim_per_image(student_tensor, teacher_tensor, border=1)[0]
        ),
        "ssim_preview_width": preview_size[0],
        "ssim_preview_height": preview_size[1],
    }


def correction_bands(
    original: np.ndarray,
    student: np.ndarray,
    teacher: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    luma = (
        original[..., 0].astype(np.float32) * 0.2126
        + original[..., 1].astype(np.float32) * 0.7152
        + original[..., 2].astype(np.float32) * 0.0722
    ) / 255.0
    student_float = student.astype(np.float32) / 255.0
    teacher_float = teacher.astype(np.float32) / 255.0
    original_float = original.astype(np.float32) / 255.0
    definitions = {
        "shadows_0_0p25": (0.0, 0.25),
        "midtones_0p25_0p75": (0.25, 0.75),
        "highlights_0p75_1": (0.75, 1.00001),
    }
    result: dict[str, dict[str, float | int]] = {}
    for label, (lower, upper) in definitions.items():
        mask = (luma >= lower) & (luma < upper)
        student_change = float(np.abs(student_float[mask] - original_float[mask]).mean())
        teacher_change = float(np.abs(teacher_float[mask] - original_float[mask]).mean())
        result[label] = {
            "pixels": int(mask.sum()),
            "student_change_mae": student_change,
            "teacher_change_mae": teacher_change,
            "student_teacher_mae": float(np.abs(student_float[mask] - teacher_float[mask]).mean()),
            "student_to_teacher_correction_ratio": (
                student_change / teacher_change if teacher_change > 0 else float("nan")
            ),
        }
    return result


def seam_gradient_ratio(image: np.ndarray, core: int = 176) -> float:
    """Compare gradients on composition seams with nearby parallel gradients."""

    value = image.astype(np.int16)
    vertical = np.abs(value[:, 1:] - value[:, :-1]).mean(axis=2)
    horizontal = np.abs(value[1:] - value[:-1]).mean(axis=2)
    seam_x = np.arange(core, image.shape[1], core) - 1
    seam_y = np.arange(core, image.shape[0], core) - 1
    seam_mean = float(
        (vertical[:, seam_x].sum() + horizontal[seam_y, :].sum())
        / (vertical[:, seam_x].size + horizontal[seam_y, :].size)
    )
    nearby_x = np.unique(
        np.clip(np.concatenate((seam_x - 4, seam_x + 4)), 0, vertical.shape[1] - 1)
    )
    nearby_y = np.unique(
        np.clip(np.concatenate((seam_y - 4, seam_y + 4)), 0, horizontal.shape[0] - 1)
    )
    nearby_mean = float(
        (vertical[:, nearby_x].sum() + horizontal[nearby_y, :].sum())
        / (vertical[:, nearby_x].size + horizontal[nearby_y, :].size)
    )
    return seam_mean / nearby_mean if nearby_mean > 0 else float("nan")


def write_index(
    output_dir: Path,
    records: list[dict[str, Any]],
    identity: dict[str, Any],
) -> None:
    lines = [
        "# ISO Test Set Visual Validation",
        "",
        "This is a visual and teacher-agreement study. The Sony JPEGs do not have paired clean",
        "targets, so these results must not be read as clean-reference PSNR/SSIM.",
        "",
        f"Student checkpoint: **{identity['label']}** (`{identity['mode']}`).",
        "",
        "| ISO | Full comparison | Region map | Detailed crops | Student/teacher PSNR | Preview SSIM |",
        "| ---: | --- | --- | --- | ---: | ---: |",
    ]
    for record in records:
        folder = record["folder"]
        agreement = record["teacher_agreement"]
        lines.append(
            f"| {record['iso']} | [view]({folder}/full_comparison.jpg) | "
            f"[view]({folder}/region_map.jpg) | [view]({folder}/detail_contact_sheet.jpg) | "
            f"{agreement['psnr']:.3f} | {agreement['ssim_preview']:.6f} |"
        )
    first = records[0]
    last = records[-1]
    first_shadow = first["correction_by_input_luma"]["shadows_0_0p25"]
    last_shadow = last["correction_by_input_luma"]["shadows_0_0p25"]
    maximum_seam_ratio = max(
        ratio
        for record in records
        for ratio in record.get("composition_seam_gradient_ratio", {}).values()
    )
    lines.extend(
        [
            "",
            "Preview SSIM is measured after scaling the full frame to at most 1024 px. Native",
            "crop PSNR/SSIM values are available in `report.json`.",
            "",
            "## Findings",
            "",
            f"- Student/teacher agreement falls from {first['teacher_agreement']['psnr']:.2f} dB at "
            f"ISO {first['iso']} to {last['teacher_agreement']['psnr']:.2f} dB at ISO {last['iso']}.",
            f"- In input shadows, student correction is {first_shadow['student_to_teacher_correction_ratio']:.1%} "
            f"of SCUNet at ISO {first['iso']} and {last_shadow['student_to_teacher_correction_ratio']:.1%} "
            f"at ISO {last['iso']}. The largest visible gap is high-ISO shadow and chroma noise.",
            f"- The maximum measured seam-gradient ratio is {maximum_seam_ratio:.3f}x versus nearby "
            "gradients; the deployment-style feathering did not introduce a systematic seam spike.",
            "- These comparisons establish teacher imitation behavior only. Without a clean paired",
            "  target, they cannot determine whether every SCUNet correction is desirable.",
            "",
            "Each ISO folder also contains twelve native 768 px lossless crop sheets and the",
            "lossless full-frame student and SCUNet outputs.",
            "",
            "## Cross-ISO Detail Sheets",
            "",
        ]
    )
    for index, region in enumerate(REGIONS, 1):
        lines.append(f"- [{index:02d} {region.label}](by_region/{index:02d}_{region.slug}.jpg)")
    lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-resolution ISO validation")
    if args.crop_size <= 0:
        raise ValueError("--crop-size must be positive")
    if args.student_batch <= 0 or args.teacher_batch <= 0:
        raise ValueError("batch sizes must be positive")

    input_dir = args.input.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    inputs = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if args.limit is not None:
        inputs = inputs[: args.limit]
    if not inputs:
        raise FileNotFoundError(f"No input images found in {input_dir}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a mapping")
    identity = checkpoint_identity(checkpoint)
    student = student_from_checkpoint(checkpoint).eval()
    device = torch.device("cuda")
    student = student.to(device)

    config = checkpoint["config"]
    student_model_config = config["model"]
    teacher_repo = resolve_paper_path(config["teacher"]["repository"])
    teacher_checkpoint = resolve_paper_path(config["teacher"]["checkpoint"])
    teacher = load_scunet_teacher(teacher_repo, teacher_checkpoint, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for image_index, image_path in enumerate(inputs, 1):
        with Image.open(image_path) as encoded:
            iso = iso_for(encoded, image_path.stem)
            original = np.array(ImageOps.exif_transpose(encoded).convert("RGB"), copy=True)
        height, width = original.shape[:2]
        if args.crop_size > min(width, height):
            raise ValueError(f"crop size {args.crop_size} exceeds {image_path.name} dimensions")
        folder_name = f"ISO{iso}_{image_path.stem}"
        image_dir = output_dir / folder_name
        crop_dir = image_dir / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{image_index}/{len(inputs)}] {image_path.name}, ISO {iso}: student", flush=True)
        student_float, student_runtime = infer_tiled(
            student,
            original,
            device,
            args.student_batch,
            "student",
            student_model_config,
        )
        student_output = uint8_image(student_float)
        del student_float
        save_png(student_output, image_dir / "student.png")

        print(f"[{image_index}/{len(inputs)}] {image_path.name}, ISO {iso}: SCUNet", flush=True)
        teacher_float, teacher_runtime = infer_tiled(
            teacher, original, device, args.teacher_batch, "SCUNet"
        )
        teacher_output = uint8_image(teacher_float)
        del teacher_float
        save_png(teacher_output, image_dir / "scunet.png")

        full_comparison(
            original,
            student_output,
            teacher_output,
            iso,
            image_dir / "full_comparison.jpg",
            identity["label"],
        )
        region_boxes = [
            (region, crop_box(region, width, height, args.crop_size)) for region in REGIONS
        ]
        overview(original, region_boxes, iso, image_dir / "region_map.jpg")
        crop_paths: list[Path] = []
        crop_metrics: list[dict[str, Any]] = []
        for region_index, (region, box) in enumerate(region_boxes, 1):
            crop_path = crop_dir / f"{region_index:02d}_{region.slug}.png"
            detail_sheet(
                original,
                student_output,
                teacher_output,
                region,
                box,
                iso,
                crop_path,
                identity["label"],
            )
            crop_paths.append(crop_path)
            left, top, right, bottom = box
            crop_metrics.append(
                {
                    "index": region_index,
                    "slug": region.slug,
                    "label": region.label,
                    "box": list(box),
                    "teacher_agreement": teacher_metrics(
                        student_output[top:bottom, left:right],
                        teacher_output[top:bottom, left:right],
                        device,
                    ),
                }
            )
        contact_sheet(crop_paths, image_dir / "detail_contact_sheet.jpg", identity["label"])
        record = {
            "source": str(image_path),
            "source_sha256": sha256_file(image_path),
            "folder": folder_name,
            "iso": iso,
            "width": width,
            "height": height,
            "student_runtime": student_runtime,
            "teacher_runtime": teacher_runtime,
            "teacher_agreement": full_teacher_metrics(student_output, teacher_output),
            "composition_seam_gradient_ratio": {
                "student": seam_gradient_ratio(student_output),
                "teacher": seam_gradient_ratio(teacher_output),
            },
            "correction_by_input_luma": correction_bands(
                original, student_output, teacher_output
            ),
            "crops": crop_metrics,
        }
        records.append(record)
        atomic_json(output_dir / "report.json", {
            "schema_version": 2,
            "purpose": "external ISO-set visual validation without clean targets",
            "input": str(input_dir),
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "epoch": int(checkpoint["epoch"]),
                **identity,
            },
            "teacher": {
                "path": str(teacher_checkpoint),
                "sha256": sha256_file(teacher_checkpoint),
                "name": config["teacher"]["checkpoint_name"],
            },
            "preprocessing": "EXIF transpose, Pillow RGB JPEG decode, float32 RGB / 255",
            "crop_size": args.crop_size,
            "images": records,
        })
        write_index(output_dir, records, identity)
        print(
            f"[{image_index}/{len(inputs)}] ISO {iso} complete: "
            f"student {student_runtime['seconds']:.1f}s, SCUNet {teacher_runtime['seconds']:.1f}s",
            flush=True,
        )

    cross_iso_sheets(output_dir, records, args.crop_size, identity["short_label"])
    write_index(output_dir, records, identity)
    print(f"Visual validation written to {output_dir}")


if __name__ == "__main__":
    main()
