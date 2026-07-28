#!/usr/bin/env python3
"""Compare existing baseline, teacher-only, and full-reference high-ISO outputs.

This script does not run any denoiser. It validates the three completed ISO
evaluation trees, then builds lossless, native-resolution crop sheets and
exports deltas for metrics already recorded in each run's report.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps, __version__ as PILLOW_VERSION


PAPER_ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = PAPER_ROOT / "evaluation/high_iso_ablation"
DEFAULT_TEACHER_ONLY_REPORT = ABLATION_ROOT / "nind_teacher_only_iso_test/report.json"
DEFAULT_FULL_REFERENCE_REPORT = ABLATION_ROOT / "nind_full_reference_iso_test/report.json"
DEFAULT_BASELINE_REPORT = PAPER_ROOT / "evaluation/iso_test_alpha_0p7/report.json"
DEFAULT_OUTPUT_DIR = ABLATION_ROOT / "iso_arm_comparison"

ARM_KEYS = ("baseline", "teacher_only", "full_reference")
ARM_LABELS = {
    "baseline": "Alpha 0.7 baseline",
    "teacher_only": "Teacher-only",
    "full_reference": "Full-reference",
}
COLUMN_LABELS = (
    "Original",
    "Alpha 0.7 baseline",
    "Teacher-only",
    "Full-reference",
    "SCUNet",
)
VALUE_KEYS = (
    "baseline",
    "teacher_only",
    "full_reference",
    "delta_teacher_only_minus_baseline",
    "delta_full_reference_minus_baseline",
    "delta_full_reference_minus_teacher_only",
)
LUMA_METRICS = (
    "student_change_mae",
    "student_teacher_mae",
    "student_to_teacher_correction_ratio",
)


@dataclass(frozen=True)
class IsoInput:
    iso: int
    folder: str
    width: int
    height: int
    crop_size: int
    source: Path
    baseline_student: Path
    teacher_only_student: Path
    full_reference_student: Path
    scunet: Path
    crops: tuple[dict[str, Any], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument(
        "--teacher-only-report",
        type=Path,
        default=DEFAULT_TEACHER_ONLY_REPORT,
    )
    parser.add_argument(
        "--full-reference-report",
        type=Path,
        default=DEFAULT_FULL_REFERENCE_REPORT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Optional relocated directory containing the original images",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        choices=range(0, 10),
        default=6,
        metavar="0..9",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    atomic_bytes(path, content.encode("utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_png(image: Image.Image, path: Path, compression: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG", compress_level=compression, optimize=False)
    os.replace(temporary, path)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/usr/share/fonts/TTF") / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolved_existing(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def validate_identity(
    report: dict[str, Any],
    report_path: Path,
    expected_mode: str | None,
) -> None:
    expected_schema = 1 if expected_mode is None else 2
    require(
        report.get("schema_version") == expected_schema,
        f"Expected schema {expected_schema} in {report_path}, got {report.get('schema_version')!r}",
    )
    checkpoint = report.get("checkpoint")
    require(isinstance(checkpoint, dict), f"Missing checkpoint metadata in {report_path}")
    if expected_mode is None:
        require(
            math.isclose(float(checkpoint.get("alpha", math.nan)), 0.7, abs_tol=1e-12),
            f"Expected legacy alpha 0.7 checkpoint metadata in {report_path}",
        )
        require(
            checkpoint.get("mode") in (None, "alpha_0p7"),
            f"Unexpected baseline mode in {report_path}: {checkpoint.get('mode')!r}",
        )
    else:
        require(
            checkpoint.get("mode") == expected_mode,
            f"Expected mode {expected_mode!r} in {report_path}, got {checkpoint.get('mode')!r}",
        )
    images = report.get("images")
    require(isinstance(images, list) and images, f"No image records in {report_path}")
    crop_size = report.get("crop_size")
    require(isinstance(crop_size, int) and crop_size > 0, f"Invalid crop size in {report_path}")


def validate_referenced_hash(metadata: dict[str, Any], label: str) -> dict[str, Any]:
    path_value = metadata.get("path")
    expected = metadata.get("sha256")
    require(isinstance(path_value, str), f"Missing {label} path")
    require(isinstance(expected, str) and len(expected) == 64, f"Missing {label} SHA-256")
    path = resolved_existing(Path(path_value), label)
    actual = sha256_file(path)
    require(actual == expected, f"{label} hash mismatch: {path}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def report_records(report: dict[str, Any], label: str) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for record in report["images"]:
        require(isinstance(record, dict), f"Invalid image record in {label}")
        iso = record.get("iso")
        require(isinstance(iso, int), f"Non-integer ISO in {label}: {iso!r}")
        require(iso not in records, f"Duplicate ISO {iso} in {label}")
        records[iso] = record
    return records


def validate_rgb_image(path: Path, size: tuple[int, int], image_format: str) -> None:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        require(image.format == image_format, f"Expected {image_format} input: {path}")
        require(image.mode == "RGB", f"Expected RGB image: {path}, got {image.mode}")
        require(image.size == size, f"Unexpected dimensions for {path}: {image.size} != {size}")


def resolve_source(record: dict[str, Any], source_dir: Path | None) -> Path:
    reported = Path(record["source"])
    candidate = source_dir / reported.name if source_dir is not None else reported
    return resolved_existing(candidate, f"ISO {record['iso']} source")


def validate_inputs(
    baseline_report: dict[str, Any],
    teacher_report: dict[str, Any],
    full_report: dict[str, Any],
    baseline_report_path: Path,
    teacher_report_path: Path,
    full_report_path: Path,
    source_dir: Path | None,
) -> tuple[list[IsoInput], dict[str, Any]]:
    validate_identity(baseline_report, baseline_report_path, None)
    validate_identity(teacher_report, teacher_report_path, "nind_teacher_only")
    validate_identity(full_report, full_report_path, "nind_full_reference")
    require(
        baseline_report["crop_size"] == teacher_report["crop_size"] == full_report["crop_size"],
        "The report crop sizes differ",
    )
    require(
        baseline_report.get("teacher") == teacher_report.get("teacher") == full_report.get("teacher"),
        "Teacher metadata differs",
    )

    baseline_records = report_records(baseline_report, "baseline report")
    teacher_records = report_records(teacher_report, "teacher-only report")
    full_records = report_records(full_report, "full-reference report")
    require(
        baseline_records.keys() == teacher_records.keys() == full_records.keys(),
        "The ISO sets differ between reports",
    )

    checkpoints = {
        "baseline": validate_referenced_hash(baseline_report["checkpoint"], "alpha 0.7 baseline checkpoint"),
        "teacher_only": validate_referenced_hash(teacher_report["checkpoint"], "teacher-only checkpoint"),
        "full_reference": validate_referenced_hash(full_report["checkpoint"], "full-reference checkpoint"),
        "scunet_teacher": validate_referenced_hash(teacher_report["teacher"], "SCUNet checkpoint"),
    }
    report_hashes = {
        "baseline": {
            "path": str(baseline_report_path),
            "sha256": sha256_file(baseline_report_path),
            "bytes": baseline_report_path.stat().st_size,
        },
        "teacher_only": {
            "path": str(teacher_report_path),
            "sha256": sha256_file(teacher_report_path),
            "bytes": teacher_report_path.stat().st_size,
        },
        "full_reference": {
            "path": str(full_report_path),
            "sha256": sha256_file(full_report_path),
            "bytes": full_report_path.stat().st_size,
        },
    }

    crop_size = teacher_report["crop_size"]
    inputs: list[IsoInput] = []
    manifest_images: list[dict[str, Any]] = []
    for iso in sorted(teacher_records):
        arm_records = {
            "baseline": baseline_records[iso],
            "teacher_only": teacher_records[iso],
            "full_reference": full_records[iso],
        }
        reference_record = arm_records["baseline"]
        for key in ("folder", "width", "height", "source_sha256"):
            require(
                len({json.dumps(record.get(key), sort_keys=True) for record in arm_records.values()}) == 1,
                f"ISO {iso} report mismatch for {key}",
            )
        folder = reference_record["folder"]
        width = reference_record["width"]
        height = reference_record["height"]
        require(isinstance(folder, str), f"Invalid folder for ISO {iso}")
        require(isinstance(width, int) and isinstance(height, int), f"Invalid size for ISO {iso}")

        source = resolve_source(reference_record, source_dir)
        source_hash = sha256_file(source)
        require(
            source_hash == reference_record["source_sha256"],
            f"ISO {iso} source hash mismatch: {source}",
        )
        with Image.open(source) as encoded:
            original = ImageOps.exif_transpose(encoded)
            require(original.size == (width, height), f"ISO {iso} source dimensions differ after EXIF transpose")

        roots = {
            "baseline": baseline_report_path.parent / folder,
            "teacher_only": teacher_report_path.parent / folder,
            "full_reference": full_report_path.parent / folder,
        }
        students = {
            arm: resolved_existing(root / "student.png", f"ISO {iso} {ARM_LABELS[arm]} output")
            for arm, root in roots.items()
        }
        scunets = {
            arm: resolved_existing(root / "scunet.png", f"ISO {iso} {ARM_LABELS[arm]} SCUNet")
            for arm, root in roots.items()
        }
        expected_size = (width, height)
        for path in (*students.values(), *scunets.values()):
            validate_rgb_image(path, expected_size, "PNG")

        hashes = {
            "source": source_hash,
            "students": {arm: sha256_file(path) for arm, path in students.items()},
            "scunets": {arm: sha256_file(path) for arm, path in scunets.items()},
        }
        require(
            len(set(hashes["scunets"].values())) == 1,
            f"ISO {iso} SCUNet outputs are not byte-identical across all three reports",
        )
        require(
            len(set(hashes["students"].values())) == len(ARM_KEYS),
            f"ISO {iso} student outputs are not distinct across all three arms",
        )

        crop_records = reference_record["crops"]
        require(isinstance(crop_records, list) and crop_records, f"ISO {iso} has no crops")
        crop_definitions = []
        for arm, record in arm_records.items():
            arm_crops = record.get("crops")
            require(
                isinstance(arm_crops, list) and len(arm_crops) == len(crop_records),
                f"ISO {iso} {arm} crop count differs",
            )
            crop_definitions.append(
                [(crop.get("index"), crop.get("slug"), crop.get("label"), crop.get("box")) for crop in arm_crops]
            )
        require(
            all(definition == crop_definitions[0] for definition in crop_definitions),
            f"ISO {iso} crop definitions differ across reports",
        )
        seen_indices: set[int] = set()
        for crop in crop_records:
            require(isinstance(crop, dict), f"ISO {iso} has an invalid crop record")
            index = crop.get("index")
            box = crop.get("box")
            require(isinstance(index, int) and index not in seen_indices, f"ISO {iso} duplicate crop index")
            require(
                isinstance(box, list)
                and len(box) == 4
                and all(isinstance(value, int) for value in box),
                f"ISO {iso} crop {index} has an invalid box",
            )
            left, top, right, bottom = box
            require(
                right - left == crop_size and bottom - top == crop_size,
                f"ISO {iso} crop {index} is not {crop_size}x{crop_size}",
            )
            require(0 <= left < right <= width and 0 <= top < bottom <= height, f"ISO {iso} crop {index} is out of bounds")
            seen_indices.add(index)

        inputs.append(
            IsoInput(
                iso=iso,
                folder=folder,
                width=width,
                height=height,
                crop_size=crop_size,
                source=source,
                baseline_student=students["baseline"],
                teacher_only_student=students["teacher_only"],
                full_reference_student=students["full_reference"],
                scunet=scunets["baseline"],
                crops=tuple(sorted(crop_records, key=lambda item: item["index"])),
            )
        )
        manifest_images.append(
            {
                "iso": iso,
                "folder": folder,
                "dimensions": [width, height],
                "crop_count": len(crop_records),
                "files": {
                    "source": {"path": str(source), "sha256": hashes["source"], "bytes": source.stat().st_size},
                    "students": {
                        arm: {"path": str(path), "sha256": hashes["students"][arm], "bytes": path.stat().st_size}
                        for arm, path in students.items()
                    },
                    "scunet_copies": {
                        arm: {"path": str(path), "sha256": hashes["scunets"][arm], "bytes": path.stat().st_size}
                        for arm, path in scunets.items()
                    },
                },
            }
        )

    crop_signatures = [
        [(crop["index"], crop["slug"], crop["label"], crop["box"]) for crop in item.crops]
        for item in inputs
    ]
    require(all(signature == crop_signatures[0] for signature in crop_signatures), "Crop definitions vary across ISO")
    provenance = {
        "reports": report_hashes,
        "checkpoints": checkpoints,
        "images": manifest_images,
        "validated": {
            "report_schema_ablation_modes_and_legacy_baseline_alpha": True,
            "checkpoint_and_teacher_hashes": True,
            "source_hashes": True,
            "source_and_output_dimensions": True,
            "rgb_png_student_and_scunet_outputs": True,
            "matching_crop_definitions": True,
            "byte_identical_scunet_across_arms": True,
            "distinct_student_outputs_across_arms": True,
        },
    }
    return inputs, provenance


def open_comparison_images(
    item: IsoInput,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
    with Image.open(item.source) as encoded:
        original = ImageOps.exif_transpose(encoded).convert("RGB").copy()
    with Image.open(item.baseline_student) as encoded:
        baseline = encoded.convert("RGB").copy()
    with Image.open(item.teacher_only_student) as encoded:
        teacher_only = encoded.convert("RGB").copy()
    with Image.open(item.full_reference_student) as encoded:
        full_reference = encoded.convert("RGB").copy()
    with Image.open(item.scunet) as encoded:
        scunet = encoded.convert("RGB").copy()
    return original, baseline, teacher_only, full_reference, scunet


def draw_column_headers(draw: ImageDraw.ImageDraw, width: int, header_height: int) -> None:
    heading = font(28, bold=True)
    subheading = font(18)
    for column, label in enumerate(COLUMN_LABELS):
        x = column * width + 16
        draw.text((x, 16), label, fill="#111111", font=heading)
    draw.text(
        (16, header_height - 29),
        "All image panels are unscaled native crops; PNG encoding is lossless.",
        fill="#444444",
        font=subheading,
    )


def render_per_iso_sheet(item: IsoInput, destination: Path, compression: int) -> None:
    header_height = 94
    row_label_height = 44
    panel = item.crop_size
    row_height = row_label_height + panel
    canvas = Image.new(
        "RGB",
        (panel * len(COLUMN_LABELS), header_height + row_height * len(item.crops)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw_column_headers(draw, panel, header_height)
    row_font = font(21, bold=True)
    images = open_comparison_images(item)
    try:
        for row, crop in enumerate(item.crops):
            left, top, right, bottom = crop["box"]
            y = header_height + row * row_height
            draw.rectangle((0, y, canvas.width, y + row_label_height - 1), fill="#f0f1f2")
            draw.text(
                (12, y + 9),
                f"ISO {item.iso} | {crop['index']:02d} {crop['label']} | box [{left}, {top}, {right}, {bottom}]",
                fill="#111111",
                font=row_font,
            )
            for column, source in enumerate(images):
                native_crop = source.crop((left, top, right, bottom))
                require(native_crop.size == (panel, panel), "Internal native crop size error")
                canvas.paste(native_crop, (column * panel, y + row_label_height))
    finally:
        for image in images:
            image.close()
    atomic_png(canvas, destination, compression)
    canvas.close()


def render_region_sheet(
    items: list[IsoInput],
    crop_position: int,
    destination: Path,
    compression: int,
) -> None:
    panel = items[0].crop_size
    header_height = 94
    row_label_height = 44
    row_height = row_label_height + panel
    canvas = Image.new(
        "RGB",
        (panel * len(COLUMN_LABELS), header_height + row_height * len(items)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw_column_headers(draw, panel, header_height)
    row_font = font(21, bold=True)
    for row, item in enumerate(items):
        crop = item.crops[crop_position]
        left, top, right, bottom = crop["box"]
        y = header_height + row * row_height
        draw.rectangle((0, y, canvas.width, y + row_label_height - 1), fill="#f0f1f2")
        draw.text(
            (12, y + 9),
            f"ISO {item.iso} | {crop['index']:02d} {crop['label']} | box [{left}, {top}, {right}, {bottom}]",
            fill="#111111",
            font=row_font,
        )
        images = open_comparison_images(item)
        try:
            for column, source in enumerate(images):
                native_crop = source.crop((left, top, right, bottom))
                require(native_crop.size == (panel, panel), "Internal native crop size error")
                canvas.paste(native_crop, (column * panel, y + row_label_height))
        finally:
            for image in images:
                image.close()
    atomic_png(canvas, destination, compression)
    canvas.close()


def metric_values(
    baseline: float,
    teacher_only: float,
    full_reference: float,
) -> dict[str, float]:
    baseline_value = float(baseline)
    teacher_value = float(teacher_only)
    full_value = float(full_reference)
    require(
        all(math.isfinite(value) for value in (baseline_value, teacher_value, full_value)),
        "Non-finite report metric",
    )
    return {
        "baseline": baseline_value,
        "teacher_only": teacher_value,
        "full_reference": full_value,
        "delta_teacher_only_minus_baseline": teacher_value - baseline_value,
        "delta_full_reference_minus_baseline": full_value - baseline_value,
        "delta_full_reference_minus_teacher_only": full_value - teacher_value,
    }


def add_csv_row(
    rows: list[dict[str, Any]],
    scope: str,
    metric: str,
    values: dict[str, float],
    *,
    iso: int | str = "",
    region_index: int | str = "",
    region_slug: str = "",
    luma_band: str = "",
) -> None:
    rows.append(
        {
            "scope": scope,
            "iso": iso,
            "region_index": region_index,
            "region_slug": region_slug,
            "luma_band": luma_band,
            "metric": metric,
            **values,
        }
    )


def mean_metric_values(values: Iterable[dict[str, float]]) -> dict[str, float]:
    values = list(values)
    require(bool(values), "Cannot average an empty metric collection")
    return {
        key: sum(value[key] for value in values) / len(values)
        for key in VALUE_KEYS
    }


def build_metrics(
    baseline_report: dict[str, Any],
    teacher_report: dict[str, Any],
    full_report: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arm_reports = {
        "baseline": baseline_report,
        "teacher_only": teacher_report,
        "full_reference": full_report,
    }
    records = {
        arm: report_records(report, f"{arm} report")
        for arm, report in arm_reports.items()
    }
    csv_rows: list[dict[str, Any]] = []
    per_iso: list[dict[str, Any]] = []

    for iso in sorted(records["baseline"]):
        arm_records = {arm: records[arm][iso] for arm in ARM_KEYS}
        teacher_agreement = {
            metric: metric_values(
                *(arm_records[arm]["teacher_agreement"][metric] for arm in ARM_KEYS)
            )
            for metric in ("psnr", "ssim_preview")
        }
        for metric, values in teacher_agreement.items():
            add_csv_row(csv_rows, "full_frame_teacher_agreement", metric, values, iso=iso)

        seam = metric_values(
            *(arm_records[arm]["composition_seam_gradient_ratio"]["student"] for arm in ARM_KEYS)
        )
        add_csv_row(csv_rows, "composition", "student_seam_gradient_ratio", seam, iso=iso)

        correction: dict[str, Any] = {}
        require(
            len(
                {
                    tuple(sorted(arm_records[arm]["correction_by_input_luma"]))
                    for arm in ARM_KEYS
                }
            )
            == 1,
            f"ISO {iso} luma bands differ",
        )
        for band in arm_records["baseline"]["correction_by_input_luma"]:
            correction[band] = {}
            for metric in LUMA_METRICS:
                values = metric_values(
                    *(
                        arm_records[arm]["correction_by_input_luma"][band][metric]
                        for arm in ARM_KEYS
                    )
                )
                correction[band][metric] = values
                add_csv_row(csv_rows, "correction_by_input_luma", metric, values, iso=iso, luma_band=band)

        arm_crops = {
            arm: {crop["index"]: crop for crop in arm_records[arm]["crops"]}
            for arm in ARM_KEYS
        }
        require(
            arm_crops["baseline"].keys()
            == arm_crops["teacher_only"].keys()
            == arm_crops["full_reference"].keys(),
            f"ISO {iso} crop sets differ",
        )
        crops: list[dict[str, Any]] = []
        for index in sorted(arm_crops["baseline"]):
            reference_crop = arm_crops["baseline"][index]
            agreement = {
                metric: metric_values(
                    *(
                        arm_crops[arm][index]["teacher_agreement"][metric]
                        for arm in ARM_KEYS
                    )
                )
                for metric in ("psnr", "ssim")
            }
            for metric, values in agreement.items():
                add_csv_row(
                    csv_rows,
                    "native_crop_teacher_agreement",
                    metric,
                    values,
                    iso=iso,
                    region_index=index,
                    region_slug=reference_crop["slug"],
                )
            crops.append(
                {
                    "index": index,
                    "slug": reference_crop["slug"],
                    "label": reference_crop["label"],
                    "box": reference_crop["box"],
                    "teacher_agreement": agreement,
                }
            )

        per_iso.append(
            {
                "iso": iso,
                "folder": arm_records["baseline"]["folder"],
                "full_frame_teacher_agreement": teacher_agreement,
                "student_seam_gradient_ratio": seam,
                "correction_by_input_luma": correction,
                "native_crops": crops,
            }
        )

    aggregate_full = {
        metric: mean_metric_values(
            record["full_frame_teacher_agreement"][metric] for record in per_iso
        )
        for metric in ("psnr", "ssim_preview")
    }
    for metric, values in aggregate_full.items():
        add_csv_row(csv_rows, "aggregate_mean_full_frame_teacher_agreement", metric, values)

    aggregate_crops = {
        metric: mean_metric_values(
            crop["teacher_agreement"][metric]
            for record in per_iso
            for crop in record["native_crops"]
        )
        for metric in ("psnr", "ssim")
    }
    for metric, values in aggregate_crops.items():
        add_csv_row(csv_rows, "aggregate_mean_native_crop_teacher_agreement", metric, values)

    by_region: list[dict[str, Any]] = []
    for crop_position, reference_crop in enumerate(per_iso[0]["native_crops"]):
        agreement = {
            metric: mean_metric_values(
                record["native_crops"][crop_position]["teacher_agreement"][metric]
                for record in per_iso
            )
            for metric in ("psnr", "ssim")
        }
        for metric, values in agreement.items():
            add_csv_row(
                csv_rows,
                "aggregate_mean_region_teacher_agreement",
                metric,
                values,
                region_index=reference_crop["index"],
                region_slug=reference_crop["slug"],
            )
        by_region.append(
            {
                "index": reference_crop["index"],
                "slug": reference_crop["slug"],
                "label": reference_crop["label"],
                "teacher_agreement": agreement,
            }
        )

    metrics = {
        "schema_version": 1,
        "purpose": "existing-output comparison; no inference and no metric recomputation from pixels",
        "delta_definitions": {
            "delta_teacher_only_minus_baseline": "teacher_only - baseline",
            "delta_full_reference_minus_baseline": "full_reference - baseline",
            "delta_full_reference_minus_teacher_only": "full_reference - teacher_only",
        },
        "interpretation": {
            "teacher_agreement": "Higher PSNR/SSIM means closer to SCUNet, not necessarily closer to a clean target.",
            "correction_ratio": "A value nearer 1 means the student applies a correction magnitude nearer SCUNet.",
            "seam_gradient_ratio": "A value nearer 1 is neutral; raw deltas are not intrinsically better or worse.",
        },
        "arms": {
            key: {
                "label": ARM_LABELS[key],
                "checkpoint": arm_reports[key]["checkpoint"],
            }
            for key in ARM_KEYS
        },
        "aggregate_mean": {
            "full_frame_teacher_agreement": aggregate_full,
            "native_crop_teacher_agreement": aggregate_crops,
            "by_region_teacher_agreement": by_region,
        },
        "per_iso": per_iso,
    }
    return metrics, csv_rows


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = (
        "scope",
        "iso",
        "region_index",
        "region_slug",
        "luma_band",
        "metric",
        *VALUE_KEYS,
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def relative_link(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def write_readme(output_dir: Path, items: list[IsoInput], metrics: dict[str, Any]) -> None:
    lines = [
        "# Baseline vs NIND ISO Comparison",
        "",
        "This directory compares already-rendered outputs. No student or SCUNet inference is",
        "performed by the comparison script. Every image panel is an unscaled 768 x 768 crop,",
        "and every comparison sheet is lossless PNG.",
        "",
        "The source Sony JPEGs have no paired clean targets. PSNR and SSIM below measure",
        "agreement with SCUNet only; they are not clean-reference quality scores.",
        "",
        "The alpha 0.7 report predates the ablation identity fields. Its baseline identity is",
        "validated from alpha=0.7 plus the report and actual checkpoint SHA-256 values.",
        "",
        "## Full-frame PSNR to SCUNet",
        "",
        "| ISO | Baseline | Teacher-only | Full-reference | T - B | F - B | F - T |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in metrics["per_iso"]:
        psnr = record["full_frame_teacher_agreement"]["psnr"]
        lines.append(
            f"| {record['iso']} | {psnr['baseline']:.3f} | {psnr['teacher_only']:.3f} | "
            f"{psnr['full_reference']:.3f} | {psnr['delta_teacher_only_minus_baseline']:+.3f} | "
            f"{psnr['delta_full_reference_minus_baseline']:+.3f} | "
            f"{psnr['delta_full_reference_minus_teacher_only']:+.3f} |"
        )
    aggregate = metrics["aggregate_mean"]["full_frame_teacher_agreement"]
    lines.extend(
        [
            f"| **Mean** | **{aggregate['psnr']['baseline']:.3f}** | "
            f"**{aggregate['psnr']['teacher_only']:.3f}** | "
            f"**{aggregate['psnr']['full_reference']:.3f}** | "
            f"**{aggregate['psnr']['delta_teacher_only_minus_baseline']:+.3f}** | "
            f"**{aggregate['psnr']['delta_full_reference_minus_baseline']:+.3f}** | "
            f"**{aggregate['psnr']['delta_full_reference_minus_teacher_only']:+.3f}** |",
            "",
            "## Full-frame preview SSIM to SCUNet",
            "",
            "| ISO | Baseline | Teacher-only | Full-reference | T - B | F - B | F - T |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in metrics["per_iso"]:
        ssim = record["full_frame_teacher_agreement"]["ssim_preview"]
        lines.append(
            f"| {record['iso']} | {ssim['baseline']:.6f} | {ssim['teacher_only']:.6f} | "
            f"{ssim['full_reference']:.6f} | {ssim['delta_teacher_only_minus_baseline']:+.6f} | "
            f"{ssim['delta_full_reference_minus_baseline']:+.6f} | "
            f"{ssim['delta_full_reference_minus_teacher_only']:+.6f} |"
        )
    lines.extend(
        [
            f"| **Mean** | **{aggregate['ssim_preview']['baseline']:.6f}** | "
            f"**{aggregate['ssim_preview']['teacher_only']:.6f}** | "
            f"**{aggregate['ssim_preview']['full_reference']:.6f}** | "
            f"**{aggregate['ssim_preview']['delta_teacher_only_minus_baseline']:+.6f}** | "
            f"**{aggregate['ssim_preview']['delta_full_reference_minus_baseline']:+.6f}** | "
            f"**{aggregate['ssim_preview']['delta_full_reference_minus_teacher_only']:+.6f}** |",
            "",
            "`B` is baseline, `T` is teacher-only, and `F` is full-reference. See",
            "[metrics.json](metrics.json) for all full-frame, luma-band, native-crop, and",
            "pairwise delta values, or [metrics.csv](metrics.csv) for a flat export.",
            "",
            "## Per-ISO native sheets",
            "",
        ]
    )
    for item in items:
        path = output_dir / "per_iso" / f"{item.folder}.png"
        lines.append(f"- [ISO {item.iso}]({relative_link(path, output_dir)})")
    lines.extend(["", "## Cross-ISO native region sheets", ""])
    for crop in items[0].crops:
        path = output_dir / "by_region" / f"{crop['index']:02d}_{crop['slug']}.png"
        lines.append(f"- [{crop['index']:02d} {crop['label']}]({relative_link(path, output_dir)})")
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "From `paper_192`:",
            "",
            "```bash",
            "$PYTHON scripts/compare_iso_arms.py",
            "```",
            "",
            "Input and output hashes, dimensions, and validation assertions are recorded in",
            "[manifest.json](manifest.json).",
            "",
        ]
    )
    atomic_text(output_dir / "README.md", "\n".join(lines))


def artifact_metadata(path: Path, output_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": relative_link(path, output_dir),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix.lower() == ".png":
        with Image.open(path) as image:
            result["dimensions"] = list(image.size)
            result["mode"] = image.mode
            result["format"] = image.format
    return result


def main() -> None:
    args = parse_args()
    baseline_report_path = resolved_existing(args.baseline_report, "alpha 0.7 baseline report")
    teacher_report_path = resolved_existing(args.teacher_only_report, "teacher-only report")
    full_report_path = resolved_existing(args.full_reference_report, "full-reference report")
    output_dir = args.output_dir.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve() if args.source_dir else None
    if source_dir is not None and not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    baseline_report = load_json(baseline_report_path)
    teacher_report = load_json(teacher_report_path)
    full_report = load_json(full_report_path)
    items, provenance = validate_inputs(
        baseline_report,
        teacher_report,
        full_report,
        baseline_report_path,
        teacher_report_path,
        full_report_path,
        source_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, csv_rows = build_metrics(baseline_report, teacher_report, full_report)
    metrics_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    atomic_json(metrics_path, metrics)
    write_metrics_csv(csv_path, csv_rows)

    generated: list[Path] = [metrics_path, csv_path]
    for position, item in enumerate(items, 1):
        destination = output_dir / "per_iso" / f"{item.folder}.png"
        print(f"[{position}/{len(items)}] Rendering per-ISO sheet for ISO {item.iso}", flush=True)
        render_per_iso_sheet(item, destination, args.png_compression)
        generated.append(destination)

    for crop_position, crop in enumerate(items[0].crops):
        destination = output_dir / "by_region" / f"{crop['index']:02d}_{crop['slug']}.png"
        print(
            f"[{crop_position + 1}/{len(items[0].crops)}] Rendering cross-ISO region {crop['label']}",
            flush=True,
        )
        render_region_sheet(items, crop_position, destination, args.png_compression)
        generated.append(destination)

    write_readme(output_dir, items, metrics)
    readme_path = output_dir / "README.md"
    generated.append(readme_path)

    manifest = {
        "schema_version": 1,
        "purpose": "reproducible comparison of existing alpha 0.7 baseline, teacher-only, and full-reference ISO outputs",
        "inference_performed": False,
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "python_executable": sys.executable,
            "arguments": {
                "baseline_report": str(baseline_report_path),
                "teacher_only_report": str(teacher_report_path),
                "full_reference_report": str(full_report_path),
                "output_dir": str(output_dir),
                "source_dir": str(source_dir) if source_dir is not None else None,
                "png_compression": args.png_compression,
            },
        },
        "crop_encoding": {
            "source_panels": "EXIF-transposed Pillow RGB decode of original JPEG",
            "output_panels": "direct crops of existing 8-bit RGB PNG outputs",
            "resampling": "none",
            "sheet_format": "lossless RGB PNG",
            "native_crop_size": [items[0].crop_size, items[0].crop_size],
        },
        "software": {
            "python": platform.python_version(),
            "pillow": PILLOW_VERSION,
            "platform": platform.platform(),
        },
        "inputs": provenance,
        "artifacts": [artifact_metadata(path, output_dir) for path in sorted(generated)],
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(f"Comparison written to {output_dir}")


if __name__ == "__main__":
    main()
