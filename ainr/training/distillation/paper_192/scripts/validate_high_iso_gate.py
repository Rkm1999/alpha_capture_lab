#!/usr/bin/env python3
"""Qualify and render the Sony-adjacent/high-noise exact-192 candidate data."""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm

from common import atomic_json, load_config, resolve_paper_path, sha256_file
from src.scunet_teacher import load_scunet_teacher


TILE = 192
PANEL_KEYS = ("input", "teacher", "clean", "correction")
PANEL_LABELS = ("Noisy", "SCUNet", "Reference", "Correction x4")
NIND_REPRESENTATIVES = (
    ("nind_books", "6400"),
    ("nind_parking-keyboard", "H1"),
    ("nind_books", "H2"),
    ("nind_parking-keyboard", "H3"),
    ("nind_claytools", "H3"),
    ("nind_chapel", "H3"),
    ("nind_shells", "H3"),
    ("nind_whistle", "H4"),
)
NIND_PROGRESSION_SCENES = ("nind_parking-keyboard", "nind_chapel")
NIND_PROGRESSION_LEVELS = ("6400", "H1", "H2", "H3")


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(path, size=size) if path.is_file() else ImageFont.load_default()


def load_array(cache_root: Path, relative_path: str) -> np.ndarray:
    path = cache_root / relative_path
    value = np.load(path, allow_pickle=False)
    if value.shape != (TILE, TILE, 3):
        raise ValueError(f"Unexpected array shape {value.shape}: {path}")
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"Array is not floating point: {path}")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all() or float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError(f"Array is non-finite or outside [0, 1]: {path}")
    return value


def luminance(value: np.ndarray) -> np.ndarray:
    return value[..., 0] * 0.2126 + value[..., 1] * 0.7152 + value[..., 2] * 0.0722


def phase_shift(first: np.ndarray, second: np.ndarray) -> tuple[int, int] | None:
    def prepared(value: np.ndarray) -> np.ndarray:
        gray = np.rint(luminance(value) * 255.0).clip(0, 255).astype(np.uint8)
        blurred = Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=2.0))
        result = np.asarray(blurred, dtype=np.float64) / 255.0
        result -= result.mean()
        return result

    first_gray = prepared(first)
    second_gray = prepared(second)
    if first_gray.std() < 0.012 or second_gray.std() < 0.012:
        return None
    cross = np.fft.fft2(first_gray) * np.conj(np.fft.fft2(second_gray))
    cross /= np.maximum(np.abs(cross), 1e-12)
    correlation = np.abs(np.fft.ifft2(cross))
    row, column = np.unravel_index(np.argmax(correlation), correlation.shape)
    if row > first_gray.shape[0] // 2:
        row -= first_gray.shape[0]
    if column > first_gray.shape[1] // 2:
        column -= first_gray.shape[1]
    return int(row), int(column)


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    error = float(np.mean(np.square(first.astype(np.float64) - second.astype(np.float64))))
    return math.inf if error == 0.0 else -10.0 * math.log10(error)


def texture_score(value: np.ndarray) -> float:
    gray = luminance(value)
    horizontal = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    vertical = np.abs(gray[1:, :] - gray[:-1, :]).mean()
    return float(horizontal + vertical)


def measurements(record: dict[str, Any], cache_root: Path) -> dict[str, Any]:
    noisy = load_array(cache_root, record["input"])
    clean = load_array(cache_root, record["clean"])
    teacher = load_array(cache_root, record["teacher"])
    gray = luminance(noisy)
    shadow = gray < 0.25
    correction = np.abs(teacher - noisy).mean(axis=2)
    shift = phase_shift(noisy, clean)
    noisy_psnr = psnr(noisy, clean)
    teacher_psnr = psnr(teacher, clean)
    return {
        "record": record,
        "arrays": {"input": noisy, "clean": clean, "teacher": teacher},
        "metrics": {
            "mean_luminance": float(gray.mean()),
            "shadow_fraction": float(shadow.mean()),
            "texture": texture_score(clean),
            "teacher_correction_mae": float(correction.mean()),
            "shadow_teacher_correction_mae": (
                float(correction[shadow].mean()) if bool(shadow.any()) else 0.0
            ),
            "noisy_reference_psnr": noisy_psnr,
            "teacher_reference_psnr": teacher_psnr,
            "teacher_psnr_gain": teacher_psnr - noisy_psnr,
            "phase_shift": list(shift) if shift is not None else None,
        },
    }


def selection_score(row: dict[str, Any]) -> float:
    metric = row["metrics"]
    return (
        metric["shadow_fraction"]
        + 8.0 * metric["shadow_teacher_correction_mae"]
        + 4.0 * metric["texture"]
    )


def aligned(row: dict[str, Any]) -> bool:
    return row["metrics"]["phase_shift"] in (None, [0, 0])


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aligned_rows = [row for row in rows if aligned(row)]
    candidates = aligned_rows or rows
    if not candidates:
        raise RuntimeError("Cannot choose a contact-sheet row from an empty group")
    fixed = [row for row in candidates if row["record"].get("crop") in (
        [960, 1344, 192, 192],
        [1152, 384, 192, 192],
        [3072, 1728, 192, 192],
        [2880, 960, 192, 192],
        [576, 768, 192, 192],
        [3264, 384, 192, 192],
    )]
    return max(fixed or candidates, key=selection_score)


def representative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    polyu = [row for row in rows if row["record"]["dataset"] == "polyu_sony"]
    for scene in sorted({row["record"]["scene"] for row in polyu}):
        selected.append(choose_best([row for row in polyu if row["record"]["scene"] == scene]))
    nind = [row for row in rows if row["record"]["dataset"] == "nind"]
    for scene, level in NIND_REPRESENTATIVES:
        group = [
            row
            for row in nind
            if row["record"]["scene"] == scene
            and row["record"].get("noise_level") == level
        ]
        if group:
            selected.append(choose_best(group))
    return selected


def progression_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for scene in NIND_PROGRESSION_SCENES:
        for level in NIND_PROGRESSION_LEVELS:
            group = [
                row
                for row in rows
                if row["record"]["dataset"] == "nind"
                and row["record"]["scene"] == scene
                and row["record"].get("noise_level") == level
            ]
            if group:
                selected.append(choose_best(group))
    return selected


def uint8_image(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(value * 255.0).clip(0, 255).astype(np.uint8), "RGB")


def render_sheet(rows: list[dict[str, Any]], destination: Path, title: str) -> None:
    title_height = 44
    row_height = 262
    sheet = Image.new("RGB", (TILE * len(PANEL_KEYS), title_height + row_height * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = font(18)
    label_font = font(13)
    small_font = font(11)
    draw.text((8, 10), title, fill="black", font=title_font)
    for row_number, row in enumerate(rows):
        record = row["record"]
        metric = row["metrics"]
        top = title_height + row_number * row_height
        arrays = row["arrays"]
        correction = np.clip(0.5 + 4.0 * (arrays["teacher"] - arrays["input"]), 0.0, 1.0)
        panels = (arrays["input"], arrays["teacher"], arrays["clean"], correction)
        descriptor = (
            f"{record['dataset']} | {record['scene']} | "
            f"{record.get('camera', 'unknown')} | {record.get('noise_level', 'unknown')} | "
            f"crop {record.get('crop')} | {record['split']} | "
            f"{record.get('supervision', 'unknown')}"
        )
        draw.rectangle((0, top, sheet.width, top + row_height - 1), outline=(210, 210, 210))
        draw.text((6, top + 4), descriptor, fill="black", font=small_font)
        for column, (panel, label) in enumerate(zip(panels, PANEL_LABELS, strict=True)):
            left = column * TILE
            sheet.paste(uint8_image(panel), (left, top + 26))
            draw.text((left + 5, top + 220), label, fill="black", font=label_font)
        metrics_text = (
            f"gain {metric['teacher_psnr_gain']:+.2f} dB | "
            f"shadow {metric['shadow_fraction']:.0%} | shift {metric['phase_shift']}"
        )
        draw.text((6, top + 242), metrics_text, fill="black", font=small_font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, optimize=True)
    sheet.convert("RGB").save(
        destination.with_suffix(".jpg"), quality=98, subsampling=0, optimize=True
    )


def finite_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def report_row(row: dict[str, Any]) -> dict[str, Any]:
    record = row["record"]
    return {
        "dataset": record["dataset"],
        "scene": record["scene"],
        "camera": record.get("camera"),
        "noise_level": record.get("noise_level"),
        "crop": record.get("crop"),
        "split": record["split"],
        **row["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs/high_iso_data_gate.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    manifest_path = resolve_paper_path(config["data"]["manifest"])
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve_paper_path(config["outputs"]["evaluation"])
    )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = document.get("records") if isinstance(document, dict) else document
    if not isinstance(records, list) or not records:
        raise ValueError("Prepared manifest contains no records")

    rows = [measurements(record, cache_root) for record in tqdm(records, desc="High-ISO gate")]
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    level_counts: collections.Counter[str] = collections.Counter()
    supervision_counts: collections.Counter[str] = collections.Counter()
    license_counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        record = row["record"]
        grouped[record["dataset"]].append(row)
        level_counts[f"{record['dataset']}/{record.get('noise_level', 'unknown')}"] += 1
        supervision_counts[str(record.get("supervision", "unknown"))] += 1
        license_counts[str(record.get("license_status", "unknown"))] += 1

    summaries = {}
    for dataset, values in sorted(grouped.items()):
        gains = [row["metrics"]["teacher_psnr_gain"] for row in values]
        summaries[dataset] = {
            "records": len(values),
            "teacher_psnr_gain_db": finite_summary(gains),
            "teacher_better_fraction": float(np.mean(np.asarray(gains) > 0.0)),
            "mean_shadow_fraction": float(
                np.mean([row["metrics"]["shadow_fraction"] for row in values])
            ),
        }

    selected = representative_rows(rows)
    progressions = progression_rows(rows)
    contact_png = output_dir / "data_gate_contact_sheet.png"
    progression_png = output_dir / "nind_progression_contact_sheet.png"
    render_sheet(
        selected,
        contact_png,
        "High-ISO candidate data gate: native 192 px, exact teacher context",
    )
    render_sheet(
        progressions,
        progression_png,
        "NIND severity progression: identical scene-stable native 192 px crops",
    )
    alignment_issues = [
        report_row(row)
        for row in rows
        if row["metrics"]["phase_shift"] not in (None, [0, 0])
    ]
    alignment_rows = [
        row
        for row in rows
        if row["metrics"]["phase_shift"] not in (None, [0, 0])
    ]
    alignment_png = output_dir / "alignment_review_contact_sheet.png"
    if alignment_rows:
        render_sheet(
            alignment_rows,
            alignment_png,
            "Alignment exceptions: visually accept, register, or quarantine before training",
        )
    parity: dict[str, Any] = {"skipped": True}
    if not args.skip_parity:
        parity_count = min(int(config["teacher"]["parity_samples"]), len(rows))
        indices = random.Random(int(config["project"]["seed"])).sample(
            range(len(rows)), parity_count
        )
        values = torch.from_numpy(
            np.stack([rows[index]["arrays"]["input"] for index in indices])
        ).permute(0, 3, 1, 2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher_model = load_scunet_teacher(
            resolve_paper_path(config["teacher"]["repository"]),
            resolve_paper_path(config["teacher"]["checkpoint"]),
            device,
        )
        fresh_batches = []
        batch_size = int(config["teacher"]["cache_batch_size"])
        with torch.inference_mode():
            for offset in range(0, parity_count, batch_size):
                fresh_batches.append(
                    teacher_model(values[offset : offset + batch_size].to(device))
                    .clamp(0.0, 1.0)
                    .cpu()
                )
        fresh = torch.cat(fresh_batches).permute(0, 2, 3, 1).numpy()
        cached = np.stack([rows[index]["arrays"]["teacher"] for index in indices])
        difference = np.abs(fresh - cached)
        parity = {
            "skipped": False,
            "samples": parity_count,
            "maximum_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
            "tolerance": float(config["teacher"]["parity_maximum_absolute_error"]),
        }
        if parity["maximum_absolute_error"] > parity["tolerance"]:
            raise RuntimeError(
                "Teacher cache parity exceeded tolerance: "
                f"{parity['maximum_absolute_error']} > {parity['tolerance']}"
            )
    report = {
        "status": "pending_visual_review",
        "preprocessing": config["project"]["preprocessing_version"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "records": len(rows),
        "source_pairs": len({row["record"]["source_input"] for row in rows}),
        "counts": dict(sorted(level_counts.items())),
        "supervision": dict(sorted(supervision_counts.items())),
        "licenses": dict(sorted(license_counts.items())),
        "dataset_summaries": summaries,
        "teacher_cache_parity": parity,
        "alignment": {
            "nonzero_phase_shift_records": len(alignment_issues),
            "low_texture_unmeasured_records": sum(
                row["metrics"]["phase_shift"] is None for row in rows
            ),
            "issues": alignment_issues,
        },
        "contact_sheet": {
            "png": str(contact_png),
            "png_sha256": sha256_file(contact_png),
            "jpg": str(contact_png.with_suffix('.jpg')),
            "rows": [report_row(row) for row in selected],
        },
        "progression_sheet": {
            "png": str(progression_png),
            "png_sha256": sha256_file(progression_png),
            "jpg": str(progression_png.with_suffix('.jpg')),
            "rows": [report_row(row) for row in progressions],
        },
        "alignment_review_sheet": (
            {
                "png": str(alignment_png),
                "png_sha256": sha256_file(alignment_png),
                "jpg": str(alignment_png.with_suffix('.jpg')),
                "rows": len(alignment_rows),
            }
            if alignment_rows
            else None
        ),
        "decision": (
            "Do not start training until the PolyU reference quality, SCUNet targets, "
            "NIND H-series behavior, and reported alignment exceptions are visually accepted."
        ),
    }
    destination = output_dir / "data_gate_report.json"
    atomic_json(destination, report)
    print(json.dumps({
        "report": str(destination),
        "contact_sheet": str(contact_png),
        "progression_sheet": str(progression_png),
        "alignment_review_sheet": str(alignment_png) if alignment_rows else None,
        "records": len(rows),
        "alignment_issues": len(alignment_issues),
        "dataset_summaries": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
