#!/usr/bin/env python3
"""Measure student fidelity using only changes made by the SCUNet teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image, ImageOps
from scipy.ndimage import gaussian_filter


def load(path: Path, transpose: bool = False) -> np.ndarray:
    image = Image.open(path)
    if transpose:
        image = ImageOps.exif_transpose(image)
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def source_iso(path: Path) -> int:
    image = Image.open(path)
    exif = image.getexif()
    containers = [exif]
    if hasattr(ExifTags, "IFD"):
        try:
            containers.append(exif.get_ifd(ExifTags.IFD.Exif))
        except KeyError:
            pass
    iso_keys = [key for key, name in ExifTags.TAGS.items()
                if name in ("PhotographicSensitivity", "ISOSpeedRatings")]
    for container in containers:
        for key in iso_keys:
            if key in container:
                return int(container[key])
    digits = "".join(character for character in path.stem if character.isdigit())
    if digits:
        return int(digits)
    raise ValueError(f"Cannot determine ISO for {path}")


def highpass(value: np.ndarray) -> np.ndarray:
    padded = np.pad(value, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    blurred = (
        padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:]
        + 2 * padded[1:-1, :-2] + 4 * padded[1:-1, 1:-1] + 2 * padded[1:-1, 2:]
        + padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
    ) / 16.0
    return value - blurred


def vector_metrics(student: np.ndarray, teacher: np.ndarray) -> dict[str, float]:
    student, teacher = student.reshape(-1), teacher.reshape(-1)
    student_energy = float(np.dot(student, student))
    teacher_energy = float(np.dot(teacher, teacher))
    student_norm, teacher_norm = np.sqrt(student_energy), np.sqrt(teacher_energy)
    cosine = float(np.dot(student, teacher) / max(student_norm * teacher_norm, 1e-12))
    magnitude_ratio = float(student_norm / max(teacher_norm, 1e-12))
    magnitude_agreement = min(magnitude_ratio, 1.0 / max(magnitude_ratio, 1e-12))
    difference = student - teacher
    relative_error = float(np.sqrt(np.dot(difference, difference)) / max(teacher_norm, 1e-12))
    return {
        "cosine": cosine,
        "magnitude_ratio": magnitude_ratio,
        "magnitude_agreement": magnitude_agreement,
        "combined_similarity": cosine * magnitude_agreement,
        "relative_l2_error": relative_error,
        "error_accuracy": max(0.0, 1.0 - relative_error),
    }


def frequency_metrics(student: np.ndarray, teacher: np.ndarray) -> dict[str, dict[str, float]]:
    result = {}
    previous_student, previous_teacher = student, teacher
    for name, sigma in (("fine", 1.0), ("medium", 4.0), ("coarse", 16.0)):
        blurred_student = gaussian_filter(student, sigma=(sigma, sigma, 0), mode="reflect")
        blurred_teacher = gaussian_filter(teacher, sigma=(sigma, sigma, 0), mode="reflect")
        result[name] = vector_metrics(
            previous_student - blurred_student,
            previous_teacher - blurred_teacher,
        )
        previous_student, previous_teacher = blurred_student, blurred_teacher
    result["very_coarse"] = vector_metrics(previous_student, previous_teacher)
    return result


def weakness_aware_score(bands: dict[str, dict[str, float]]) -> dict[str, float]:
    similarities = {name: values["combined_similarity"] for name, values in bands.items()}
    weights = {"fine": 1.0, "medium": 2.0, "coarse": 3.0, "very_coarse": 1.0}
    harmonic = sum(weights.values()) / sum(
        weights[name] / max(min(similarities[name], 1.0), 1e-4)
        for name in weights
    )
    critical = min(similarities["medium"], similarities["coarse"])
    return {
        "weighted_harmonic": harmonic,
        "critical_band": critical,
        "score": 0.5 * harmonic + 0.5 * critical,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--teacher-outputs", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    iso_values = (1600, 3200, 6400, 12800, 25600, 51200)
    sources = [path for path in args.input.iterdir() if path.suffix.lower() in (".jpg", ".jpeg")]
    sources_by_iso = {source_iso(path): path for path in sources}
    if len(sources_by_iso) != len(sources):
        raise ValueError("Source set contains duplicate ISO values")
    missing = set(iso_values) - sources_by_iso.keys()
    if missing or len(sources) != len(iso_values):
        raise ValueError(f"Expected ISO values {iso_values}; missing={sorted(missing)} sources={len(sources)}")
    for iso in iso_values:
        source_path = sources_by_iso[iso]
        source = load(source_path, transpose=True)
        student = load(args.outputs / f"ISO{iso}_distilled.jpg") - source
        teacher_root = args.teacher_outputs or args.outputs
        teacher = load(teacher_root / f"ISO{iso}_scunet.jpg") - source
        correction_magnitude = np.sqrt(np.mean(teacher * teacher, axis=2))
        threshold = float(np.quantile(correction_magnitude, 0.5))
        mask = correction_magnitude >= threshold
        bands = frequency_metrics(student, teacher)
        row = {
            "iso": iso,
            "global": vector_metrics(student, teacher),
            "changed_pixels": vector_metrics(student[mask], teacher[mask]),
            "high_frequency": vector_metrics(highpass(student), highpass(teacher)),
            "frequency_bands": bands,
            "weakness_aware": weakness_aware_score(bands),
            "teacher_change_rms": float(np.sqrt(np.mean(teacher * teacher))),
        }
        rows.append(row)
        print(
            f"ISO {iso}: changed={row['changed_pixels']['combined_similarity'] * 100:.2f}% "
            f"cos={row['changed_pixels']['cosine'] * 100:.2f}% "
            f"mag={row['changed_pixels']['magnitude_agreement'] * 100:.2f}% "
            f"error={row['changed_pixels']['relative_l2_error'] * 100:.2f}% "
            f"highfreq={row['high_frequency']['combined_similarity'] * 100:.2f}% "
            f"weak={row['weakness_aware']['score'] * 100:.2f}%"
        )
    summary = {}
    for section in ("global", "changed_pixels", "high_frequency"):
        summary[section] = {
            key: float(np.mean([row[section][key] for row in rows]))
            for key in rows[0][section]
        }
    summary["frequency_bands"] = {
        band: {
            key: float(np.mean([row["frequency_bands"][band][key] for row in rows]))
            for key in rows[0]["frequency_bands"][band]
        }
        for band in rows[0]["frequency_bands"]
    }
    summary["weakness_aware"] = weakness_aware_score(summary["frequency_bands"])
    report = {"primary_metric": "weakness_aware.score", "summary": summary, "rows": rows}
    print(json.dumps(report["summary"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
