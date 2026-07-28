#!/usr/bin/env python3
"""Audit and visualize the prepared UHD-LL/SNIC exact-192 domain cache."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from common import atomic_json, load_config, resolve_paper_path, sha256_file
from prepare_domain_dataset import (
    ImageSource,
    UHD_CLEAN_TRANSFORM,
    UHD_HYBRID_SUPERVISION,
    UHD_HYBRID_TARGET,
    alignment_gate,
    apply_local_gain,
    build_local_gain_field,
    build_uhd_hybrid_target,
    crop_seed,
    jpeg_roundtrip,
    sample_thumbnail_field,
    stratified_positions,
)
from src.scunet_teacher import load_scunet_teacher


TILE = 192
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
EXPECTED_SPLITS = {"train", "validation"}
EXPECTED_UHD_SOURCE_PAIRS = {"train": 2000, "validation": 150}
ARRAY_KEYS = ("input", "teacher", "clean")
PANEL_LABELS = ("Noisy input", "SCUNet teacher", "Clean target", "Teacher correction x4")


class Findings:
    """Collect bounded diagnostics so one run reports more than the first failure."""

    def __init__(self, examples_per_code: int = 5) -> None:
        self.examples_per_code = examples_per_code
        self.counts: dict[str, collections.Counter[str]] = {
            "errors": collections.Counter(),
            "warnings": collections.Counter(),
        }
        self.examples: dict[str, dict[str, list[str]]] = {
            "errors": collections.defaultdict(list),
            "warnings": collections.defaultdict(list),
        }

    def add(self, severity: str, code: str, message: str) -> None:
        self.counts[severity][code] += 1
        bucket = self.examples[severity][code]
        if len(bucket) < self.examples_per_code:
            bucket.append(message)

    def error(self, code: str, message: str) -> None:
        self.add("errors", code, message)

    def warn(self, code: str, message: str) -> None:
        self.add("warnings", code, message)

    @property
    def failed(self) -> bool:
        return bool(self.counts["errors"])

    def report(self) -> dict[str, Any]:
        return {
            severity: {
                "total": int(sum(self.counts[severity].values())),
                "by_code": dict(sorted(self.counts[severity].items())),
                "examples": dict(sorted(self.examples[severity].items())),
            }
            for severity in ("errors", "warnings")
        }


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(path, size=size) if path.is_file() else ImageFont.load_default()


def contained_path(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def approximately(value: Any, expected: float, tolerance: float = 1e-7) -> bool:
    return finite_number(value) and abs(float(value) - expected) <= tolerance


def load_checked_array(
    cache_root: Path,
    record: dict[str, Any],
    key: str,
    expected_dtype: np.dtype[Any],
    findings: Findings,
    teacher_digest: hashlib._Hash | None = None,
) -> np.ndarray | None:
    record_id = str(record.get("id", "<missing-id>"))
    path = contained_path(cache_root, record.get(key))
    if path is None:
        findings.error("unsafe_array_path", f"{record_id}/{key}: {record.get(key)!r}")
        return None
    if not path.is_file():
        findings.error("missing_array", f"{record_id}/{key}: {path}")
        return None
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:  # noqa: BLE001 - report corrupt cache entries together
        findings.error("unreadable_array", f"{record_id}/{key}: {error}")
        return None
    if value.shape != (TILE, TILE, 3):
        findings.error("array_shape", f"{record_id}/{key}: {value.shape}")
        return None
    if value.dtype != expected_dtype:
        findings.error(
            "array_dtype",
            f"{record_id}/{key}: {value.dtype}, expected {expected_dtype}",
        )
    if not np.issubdtype(value.dtype, np.floating):
        findings.error("array_not_float", f"{record_id}/{key}: {value.dtype}")
        return None
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        findings.error("array_nonfinite", f"{record_id}/{key}")
        return None
    minimum = float(result.min())
    maximum = float(result.max())
    if minimum < 0.0 or maximum > 1.0:
        findings.error(
            "array_range",
            f"{record_id}/{key}: [{minimum:.8g}, {maximum:.8g}]",
        )
        return None
    if key == "teacher" and teacher_digest is not None:
        relative = str(record[key]).encode("utf-8")
        teacher_digest.update(len(relative).to_bytes(4, "little"))
        teacher_digest.update(relative)
        teacher_digest.update(value.dtype.str.encode("ascii"))
        teacher_digest.update(np.asarray(value).tobytes(order="C"))
    return result


def luminance(value: np.ndarray) -> np.ndarray:
    return value @ LUMA


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    error = float(np.mean(np.square(first.astype(np.float64) - second.astype(np.float64))))
    return math.inf if error == 0.0 else -10.0 * math.log10(error)


def texture_score(value: np.ndarray) -> float:
    gray = luminance(value)
    horizontal = float(np.abs(gray[:, 1:] - gray[:, :-1]).mean())
    vertical = float(np.abs(gray[1:, :] - gray[:-1, :]).mean())
    return horizontal + vertical


def array_metrics(record: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    noisy = arrays["input"]
    teacher = arrays["teacher"]
    clean = arrays["clean"]
    gray = luminance(noisy)
    correction = np.abs(teacher - noisy).mean(axis=2)
    noisy_psnr = psnr(noisy, clean)
    teacher_psnr = psnr(teacher, clean)
    return {
        "id": record["id"],
        "dataset": record["dataset"],
        "split": record["split"],
        "scene": record["scene"],
        "supervision": record["supervision"],
        "noise_level": record.get("noise_level"),
        "iso": record.get("iso"),
        "crop": record.get("crop"),
        "mean_luminance": float(gray.mean()),
        "shadow_fraction": float((gray < 0.25).mean()),
        "texture": texture_score(clean),
        "teacher_correction_mae": float(correction.mean()),
        "noisy_reference_psnr": noisy_psnr,
        "teacher_reference_psnr": teacher_psnr,
        "teacher_psnr_gain": teacher_psnr - noisy_psnr,
        "uhd_hybrid_gate": record.get("uhd_hybrid_gate"),
        "record": record,
    }


def expected_gate_failures(
    gate: dict[str, Any],
    target_config: dict[str, Any],
) -> list[str] | None:
    keys = (
        "texture",
        "zero_edge_correlation",
        "best_edge_correlation",
        "alignment_gain",
        "local_gain_minimum",
        "local_gain_maximum",
    )
    if not all(finite_number(gate.get(key)) for key in keys):
        return None
    shift = gate.get("best_shift")
    gain_mean = gate.get("local_gain_mean")
    if (
        not isinstance(shift, list)
        or len(shift) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in shift)
        or not isinstance(gain_mean, list)
        or len(gain_mean) != 3
        or not all(finite_number(value) for value in gain_mean)
    ):
        return None
    radius = int(target_config["alignment_search_radius"])
    if any(abs(value) > radius for value in shift):
        return None
    if float(gate["best_edge_correlation"]) + 1e-9 < float(
        gate["zero_edge_correlation"]
    ) or not approximately(
        gate["alignment_gain"],
        float(gate["best_edge_correlation"]) - float(gate["zero_edge_correlation"]),
        tolerance=1e-6,
    ):
        return None
    failures = []
    if float(gate["texture"]) < float(target_config["minimum_texture"]):
        failures.append("insufficient_texture")
    if float(gate["zero_edge_correlation"]) < float(
        target_config["minimum_zero_shift_correlation"]
    ):
        failures.append("zero_shift_correlation")
    if shift != [0, 0] and float(gate["alignment_gain"]) > float(
        target_config["maximum_nonzero_shift_gain"]
    ):
        failures.append("nonzero_shift_improvement")
    return failures


def compare_reconstructed_array(
    record_id: str,
    label: str,
    expected: np.ndarray,
    actual: np.ndarray,
    tolerance: float,
    findings: Findings,
) -> tuple[float, float]:
    if expected.shape != actual.shape:
        findings.error(
            "reconstructed_array_shape",
            f"{record_id}/{label}: expected={expected.shape}, cached={actual.shape}",
        )
        return math.inf, math.inf
    difference = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
    maximum = float(difference.max())
    mean = float(difference.mean())
    if maximum > tolerance:
        findings.error(
            f"reconstructed_{label}_mismatch",
            f"{record_id}: max={maximum:.8g}, mean={mean:.8g}, tolerance={tolerance:.8g}",
        )
    return maximum, mean


def compare_reconstructed_gate(
    record_id: str,
    expected: dict[str, Any],
    actual: Any,
    findings: Findings,
) -> None:
    if not isinstance(actual, dict):
        findings.error("reconstructed_gate_missing", record_id)
        return
    for key in (
        "texture",
        "zero_edge_correlation",
        "best_edge_correlation",
        "alignment_gain",
        "local_gain_minimum",
        "local_gain_maximum",
    ):
        if not finite_number(actual.get(key)) or not math.isclose(
            float(actual[key]), float(expected[key]), rel_tol=0.0, abs_tol=1e-8
        ):
            findings.error(
                "reconstructed_gate_metric",
                f"{record_id}/{key}: source={expected[key]!r}, cached={actual.get(key)!r}",
            )
    actual_mean = actual.get("local_gain_mean")
    if (
        not isinstance(actual_mean, list)
        or len(actual_mean) != 3
        or not all(finite_number(value) for value in actual_mean)
        or not np.allclose(
            np.asarray(actual_mean, dtype=np.float64),
            np.asarray(expected["local_gain_mean"], dtype=np.float64),
            rtol=0.0,
            atol=1e-8,
        )
    ):
        findings.error(
            "reconstructed_gate_metric",
            f"{record_id}/local_gain_mean: source={expected['local_gain_mean']!r}, "
            f"cached={actual_mean!r}",
        )
    for key in ("best_shift", "failure_reasons", "passed"):
        if actual.get(key) != expected[key]:
            findings.error(
                "reconstructed_gate_decision",
                f"{record_id}/{key}: source={expected[key]!r}, cached={actual.get(key)!r}",
            )


def cached_reconstruction_array(cache_root: Path, record: dict[str, Any], key: str) -> np.ndarray:
    path = contained_path(cache_root, record.get(key))
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Cannot reconstruct {record.get('id')}/{key}: {path}")
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)


def reconstruct_uhd_cache(
    records: list[dict[str, Any]],
    source_by_key: dict[tuple[str, str, str], dict[str, Any]],
    source_root: Path,
    cache_root: Path,
    config: dict[str, Any],
    tolerance: float,
    findings: Findings,
) -> dict[str, Any]:
    """Rebuild UHD inputs, gates, and hybrid targets from the immutable sources."""

    selected = [
        record
        for record in records
        if isinstance(record, dict) and record.get("dataset") == "uhd_ll"
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in selected:
        key = (
            "uhd_ll",
            str(record.get("source_input")),
            str(record.get("source_clean")),
        )
        grouped[key].append(record)

    target_config = config["uhd_hybrid_target"]
    patch_count = int(config["data"]["patches_per_pair"]["uhd_ll"])
    candidate_count = int(config["data"]["crop_candidates"])
    seed = int(config["project"]["seed"])
    quality_value = config["data"].get("jpeg_quality", {}).get("uhd_ll")
    quality = int(quality_value) if quality_value is not None else None
    maximum_input_error = 0.0
    maximum_clean_error = 0.0
    input_mean_errors: list[float] = []
    clean_mean_errors: list[float] = []
    reconstructed = 0

    for source_key in tqdm(sorted(grouped), desc="Reconstructing UHD hybrid targets"):
        source = source_by_key.get(source_key)
        if source is None:
            findings.error("reconstruction_source_missing", repr(source_key))
            continue
        noisy_path = contained_path(source_root, source.get("input"))
        clean_path = contained_path(source_root, source.get("clean"))
        if noisy_path is None or clean_path is None or not noisy_path.is_file() or not clean_path.is_file():
            findings.error(
                "reconstruction_source_missing",
                f"{source_key}: input={noisy_path}, clean={clean_path}",
            )
            continue
        with ImageSource(noisy_path) as noisy_source, ImageSource(clean_path) as clean_source:
            if (noisy_source.width, noisy_source.height) != (
                clean_source.width,
                clean_source.height,
            ):
                findings.error("reconstruction_geometry", repr(source_key))
                continue
            thumbnail_width = int(target_config["thumbnail_width"])
            noisy_thumbnail = noisy_source.thumbnail(thumbnail_width)
            clean_thumbnail = clean_source.thumbnail(thumbnail_width)
            gain_field = build_local_gain_field(
                noisy_thumbnail,
                clean_thumbnail,
                noisy_source.width,
                target_config,
            )
            positions = stratified_positions(
                noisy_thumbnail,
                noisy_source.width,
                noisy_source.height,
                patch_count,
                candidate_count,
                crop_seed(source, seed),
            )
            source_hash = hashlib.sha256(
                f"uhd_ll:{source['input']}:{source['clean']}".encode()
            ).hexdigest()
            expected_by_id = {
                f"{source_hash[:16]}_{index:02d}": (left, top, crop_luma)
                for index, (left, top, crop_luma) in enumerate(positions)
            }
            for record in grouped[source_key]:
                record_id = str(record.get("id"))
                expected_position = expected_by_id.get(record_id)
                if expected_position is None:
                    findings.error(
                        "reconstructed_crop_id",
                        f"{record_id}: not one of {sorted(expected_by_id)}",
                    )
                    continue
                left, top, crop_luma = expected_position
                expected_crop = [left, top, TILE, TILE]
                if record.get("crop") != expected_crop:
                    findings.error(
                        "reconstructed_crop_position",
                        f"{record_id}: source={expected_crop}, cached={record.get('crop')!r}",
                    )
                    continue
                if not finite_number(record.get("crop_mean_luminance")) or not math.isclose(
                    float(record["crop_mean_luminance"]),
                    crop_luma,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    findings.error(
                        "reconstructed_crop_luminance",
                        f"{record_id}: source={crop_luma!r}, "
                        f"cached={record.get('crop_mean_luminance')!r}",
                    )

                noisy = noisy_source.crop(left, top)
                raw_clean = clean_source.crop(left, top)
                gain = sample_thumbnail_field(
                    gain_field,
                    left,
                    top,
                    noisy_source.width,
                    noisy_source.height,
                )
                mapped_clean = apply_local_gain(raw_clean, gain)
                expected_gate = alignment_gate(noisy, mapped_clean, target_config)
                expected_gate.update(
                    {
                        "local_gain_minimum": float(gain.min()),
                        "local_gain_mean": gain.mean(axis=(0, 1)).tolist(),
                        "local_gain_maximum": float(gain.max()),
                    }
                )
                compare_reconstructed_gate(
                    record_id,
                    expected_gate,
                    record.get("uhd_hybrid_gate"),
                    findings,
                )

                expected_input = noisy
                if quality is not None:
                    expected_input = jpeg_roundtrip(expected_input, quality)
                    mapped_clean = jpeg_roundtrip(mapped_clean, quality)
                try:
                    actual_input = cached_reconstruction_array(cache_root, record, "input")
                    actual_clean = cached_reconstruction_array(cache_root, record, "clean")
                    cached_teacher = cached_reconstruction_array(cache_root, record, "teacher")
                except (OSError, ValueError) as error:
                    findings.error("reconstruction_array_unreadable", f"{record_id}: {error}")
                    continue
                expected_clean = (
                    build_uhd_hybrid_target(cached_teacher, mapped_clean, target_config)
                    if expected_gate["passed"]
                    else mapped_clean
                )
                input_maximum, input_mean = compare_reconstructed_array(
                    record_id,
                    "input",
                    expected_input,
                    actual_input,
                    1e-7,
                    findings,
                )
                clean_maximum, clean_mean = compare_reconstructed_array(
                    record_id,
                    "clean_target",
                    expected_clean,
                    actual_clean,
                    tolerance if expected_gate["passed"] else 1e-7,
                    findings,
                )
                maximum_input_error = max(maximum_input_error, input_maximum)
                maximum_clean_error = max(maximum_clean_error, clean_maximum)
                input_mean_errors.append(input_mean)
                clean_mean_errors.append(clean_mean)
                reconstructed += 1

    return {
        "records_requested": len(selected),
        "records_reconstructed": reconstructed,
        "source_pairs": len(grouped),
        "maximum_input_absolute_error": maximum_input_error,
        "mean_input_absolute_error": (
            float(np.mean(input_mean_errors)) if input_mean_errors else None
        ),
        "maximum_clean_target_absolute_error": maximum_clean_error,
        "mean_clean_target_absolute_error": (
            float(np.mean(clean_mean_errors)) if clean_mean_errors else None
        ),
        "hybrid_target_tolerance": tolerance,
    }


def validate_record_semantics(
    record: dict[str, Any],
    allowed_datasets: set[str],
    target_config: dict[str, Any],
    jpeg_quality: dict[str, Any],
    findings: Findings,
) -> None:
    record_id = str(record.get("id", "<missing-id>"))
    dataset = record.get("dataset")
    split = record.get("split")
    if dataset not in allowed_datasets:
        findings.error("unexpected_dataset", f"{record_id}: {dataset!r}")
        return
    if split not in EXPECTED_SPLITS:
        findings.error("unexpected_split", f"{record_id}: {split!r}")
    if not isinstance(record.get("scene"), str) or not record["scene"]:
        findings.error("missing_scene", record_id)
    crop = record.get("crop")
    if (
        not isinstance(crop, list)
        or len(crop) != 4
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in crop)
        or crop[0] < 0
        or crop[1] < 0
        or crop[2:] != [TILE, TILE]
    ):
        findings.error("invalid_crop", f"{record_id}: {crop!r}")
    if not finite_number(record.get("crop_mean_luminance")) or not (
        0.0 <= float(record.get("crop_mean_luminance", -1.0)) <= 1.0
    ):
        findings.error(
            "invalid_crop_luminance",
            f"{record_id}: {record.get('crop_mean_luminance')!r}",
        )

    supervision = record.get("supervision")
    gate = record.get("uhd_hybrid_gate")
    if dataset == "uhd_ll":
        if not isinstance(gate, dict):
            findings.error("missing_uhd_hybrid_gate", record_id)
            return
        failures = expected_gate_failures(gate, target_config)
        if failures is None:
            findings.error("invalid_uhd_hybrid_metrics", record_id)
            return
        passed = not failures
        if gate.get("passed") is not passed:
            findings.error(
                "gate_decision_mismatch",
                f"{record_id}: stored={gate.get('passed')!r}, recomputed={passed}",
            )
        if gate.get("failure_reasons") != failures:
            findings.error(
                "gate_failure_reasons",
                f"{record_id}: stored={gate.get('failure_reasons')!r}, recomputed={failures}",
            )
        gain_minimum = float(gate["local_gain_minimum"])
        gain_maximum = float(gate["local_gain_maximum"])
        gain_tolerance = 4.0 * float(np.finfo(np.float32).eps)
        if not (
            float(target_config["minimum_gain"]) - gain_tolerance
            <= gain_minimum
            <= gain_maximum
            <= float(target_config["maximum_gain"]) + gain_tolerance
        ):
            findings.error("invalid_local_gain_range", record_id)
        expected_supervision = UHD_HYBRID_SUPERVISION if passed else "teacher_only"
        expected_gt = 1.0 if passed else 0.0
        expected_kd = 0.7 if passed else 1.0
        expected_target = UHD_HYBRID_TARGET if passed else "local_ratio_candidate_ignored"
    elif dataset == "snic_sony":
        if gate not in (None, {}):
            findings.error("unexpected_uhd_hybrid_gate", record_id)
        expected_supervision = "paired"
        expected_gt = 1.0
        expected_kd = 0.7
        expected_target = "native_paired_reference"
        iso = record.get("iso")
        if not isinstance(iso, int) or isinstance(iso, bool) or iso <= 100:
            findings.error("invalid_snic_iso", f"{record_id}: {iso!r}")
        if "snic_sony" in jpeg_quality and record.get("jpeg_quality") != int(
            jpeg_quality["snic_sony"]
        ):
            findings.error(
                "snic_jpeg_quality",
                f"{record_id}: {record.get('jpeg_quality')!r}",
            )
    else:
        return

    if supervision != expected_supervision:
        findings.error(
            "supervision_mismatch",
            f"{record_id}: {supervision!r}, expected {expected_supervision!r}",
        )
    if not approximately(record.get("gt_weight"), expected_gt):
        findings.error(
            "gt_weight_mismatch",
            f"{record_id}: {record.get('gt_weight')!r}, expected {expected_gt}",
        )
    if not approximately(record.get("kd_weight"), expected_kd):
        findings.error(
            "kd_weight_mismatch",
            f"{record_id}: {record.get('kd_weight')!r}, expected {expected_kd}",
        )
    if record.get("clean_target") != expected_target:
        findings.error(
            "clean_target_mismatch",
            f"{record_id}: {record.get('clean_target')!r}, expected {expected_target!r}",
        )


def stratified_subset(records: list[dict[str, Any]], maximum: int, seed: int) -> list[dict[str, Any]]:
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if maximum >= len(records):
        return records

    generator = random.Random(seed)
    detailed_groups: dict[tuple[Any, ...], list[int]] = collections.defaultdict(list)
    detailed_by_coarse: dict[tuple[Any, ...], list[tuple[Any, ...]]] = (
        collections.defaultdict(list)
    )
    for index, record in enumerate(records):
        coarse = (record.get("dataset"), record.get("split"))
        detailed = (
            *coarse,
            record.get("supervision"),
            record.get("noise_level") if record.get("dataset") == "snic_sony" else None,
        )
        detailed_groups[detailed].append(index)

    def stable_key(value: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(str(item) for item in value)

    for detailed in sorted(detailed_groups, key=stable_key):
        indices = detailed_groups[detailed]
        indices.sort(key=lambda index: str(records[index].get("id")))
        generator.shuffle(indices)
        detailed_by_coarse[detailed[:2]].append(detailed)
    for detailed_keys in detailed_by_coarse.values():
        detailed_keys.sort(key=stable_key)
        generator.shuffle(detailed_keys)

    selected_indices: list[int] = []
    represented_details: set[tuple[Any, ...]] = set()
    remaining_coarse = set(detailed_by_coarse)
    seen_datasets: set[Any] = set()
    seen_splits: set[Any] = set()

    def take(detailed: tuple[Any, ...]) -> bool:
        indices = detailed_groups[detailed]
        if not indices:
            return False
        selected_indices.append(indices.pop())
        represented_details.add(detailed)
        return True

    # Cover dataset/split combinations first. When the cap is smaller than the
    # number of combinations, greedily prefer a row that adds an unseen dataset
    # and split so a two-row cap can still cover both domains and both splits.
    while remaining_coarse and len(selected_indices) < maximum:
        scores = {
            coarse: int(coarse[0] not in seen_datasets) + int(coarse[1] not in seen_splits)
            for coarse in remaining_coarse
        }
        best_score = max(scores.values())
        candidates = sorted(
            (coarse for coarse, score in scores.items() if score == best_score),
            key=stable_key,
        )
        coarse = candidates[generator.randrange(len(candidates))]
        detailed = detailed_by_coarse[coarse][0]
        if not take(detailed):
            raise RuntimeError(f"Empty detailed validation stratum: {detailed}")
        seen_datasets.add(coarse[0])
        seen_splits.add(coarse[1])
        remaining_coarse.remove(coarse)

    # Give every coarse combination one new detailed supervision/noise stratum
    # per round before taking a third stratum from any combination.
    coarse_order = sorted(detailed_by_coarse, key=stable_key)
    generator.shuffle(coarse_order)
    while len(selected_indices) < maximum:
        progressed = False
        for coarse in coarse_order:
            candidates = [
                detailed
                for detailed in detailed_by_coarse[coarse]
                if detailed not in represented_details and detailed_groups[detailed]
            ]
            if candidates and len(selected_indices) < maximum:
                take(candidates[0])
                progressed = True
        if not progressed:
            break

    # All detailed strata are represented. Fill any remaining budget in
    # deterministic round-robin order instead of exhausting one stratum first.
    detailed_order = sorted(detailed_groups, key=stable_key)
    generator.shuffle(detailed_order)
    while len(selected_indices) < maximum:
        progressed = False
        for detailed in detailed_order:
            if len(selected_indices) >= maximum:
                break
            progressed = take(detailed) or progressed
        if not progressed:
            break
    return [records[index] for index in selected_indices]


def finite_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def row_score(row: dict[str, Any]) -> float:
    return (
        4.0 * float(row["texture"])
        + 8.0 * float(row["teacher_correction_mae"])
        + float(row["shadow_fraction"])
    )


def representative_rows(rows: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    coarse: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    detailed: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        coarse[(row["dataset"], row["split"])].append(row)
        luma_band = "shadow" if row["mean_luminance"] < 0.22 else "mid-bright"
        domain_level = (
            row.get("noise_level") if row["dataset"] == "snic_sony" else row["supervision"]
        )
        detailed[(row["dataset"], row["split"], domain_level, luma_band)].append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_best(group: list[dict[str, Any]]) -> None:
        candidate = max(group, key=row_score)
        if candidate["id"] not in selected_ids:
            selected.append(candidate)
            selected_ids.add(candidate["id"])

    for key in sorted(coarse):
        add_best(coarse[key])
    for key in sorted(detailed, key=lambda value: tuple(str(item) for item in value)):
        if len(selected) >= maximum:
            break
        add_best(detailed[key])
    if len(selected) < maximum:
        for candidate in sorted(rows, key=row_score, reverse=True):
            if len(selected) >= maximum:
                break
            if candidate["id"] not in selected_ids:
                selected.append(candidate)
                selected_ids.add(candidate["id"])
    return sorted(
        selected,
        key=lambda row: (
            row["dataset"],
            row["split"],
            str(row.get("noise_level")),
            row["mean_luminance"],
        ),
    )


def uint8_image(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(value * 255.0).clip(0, 255).astype(np.uint8), "RGB")


def gate_text(row: dict[str, Any]) -> str:
    gate = row.get("uhd_hybrid_gate")
    if not isinstance(gate, dict):
        return f"ISO {row.get('iso', 'unknown')} | native paired reference"
    return (
        f"hybrid gate {'PASS' if gate.get('passed') else 'FAIL'} | shift "
        f"{gate.get('best_shift')} | edge zero/best "
        f"{float(gate.get('zero_edge_correlation', math.nan)):.3f}/"
        f"{float(gate.get('best_edge_correlation', math.nan)):.3f} | texture "
        f"{float(gate.get('texture', math.nan)):.4f}"
    )


def render_contact_sheet(
    rows: list[dict[str, Any]],
    cache_root: Path,
    destination: Path,
) -> None:
    title_height = 52
    descriptor_height = 42
    footer_height = 48
    row_height = descriptor_height + TILE + footer_height
    width = TILE * 4
    height = title_height + max(1, len(rows)) * row_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 9),
        "UHD hybrid + SNIC gate: native 192 x 192 crops",
        fill="black",
        font=font(19),
    )
    label_font = font(12)
    small_font = font(10)
    for row_index, row in enumerate(rows):
        record = row["record"]
        arrays: dict[str, np.ndarray] = {}
        for key in ARRAY_KEYS:
            path = contained_path(cache_root, record[key])
            if path is None:
                raise ValueError(f"Unsafe contact-sheet path: {record[key]}")
            arrays[key] = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        correction = np.clip(0.5 + 4.0 * (arrays["teacher"] - arrays["input"]), 0.0, 1.0)
        panels = (arrays["input"], arrays["teacher"], arrays["clean"], correction)
        top = title_height + row_index * row_height
        draw.rectangle((0, top, width - 1, top + row_height - 1), outline=(205, 205, 205))
        descriptor = (
            f"{record['dataset']} | {record['split']} | {record['scene']} | "
            f"noise {record.get('noise_level', 'unknown')} | {record['supervision']} | "
            f"GT {float(record['gt_weight']):.1f} / KD {float(record['kd_weight']):.1f}"
        )
        draw.text((6, top + 4), descriptor, fill="black", font=small_font)
        draw.text((6, top + 21), gate_text(row), fill=(45, 45, 45), font=small_font)
        image_top = top + descriptor_height
        for column, (panel, label) in enumerate(zip(panels, PANEL_LABELS, strict=True)):
            left = column * TILE
            sheet.paste(uint8_image(panel), (left, image_top))
            if column == 2 and record["supervision"] == "teacher_only":
                label = "Local clean candidate (ignored)"
            elif column == 2 and record["supervision"] == UHD_HYBRID_SUPERVISION:
                label = "Teacher-LP + clean-HP target"
            draw.rectangle((left, image_top + TILE - 19, left + TILE, image_top + TILE), fill="white")
            draw.text((left + 4, image_top + TILE - 17), label, fill="black", font=label_font)
        metrics = (
            f"input/reference {row['noisy_reference_psnr']:.2f} dB | "
            f"teacher/reference {row['teacher_reference_psnr']:.2f} dB | "
            f"gain {row['teacher_psnr_gain']:+.2f} dB | shadow {row['shadow_fraction']:.0%} | "
            f"teacher change {row['teacher_correction_mae']:.4f}"
        )
        draw.text((6, image_top + TILE + 6), metrics, fill="black", font=small_font)
        draw.text(
            (6, image_top + TILE + 23),
            f"crop {record.get('crop')} | id {record['id']}",
            fill=(65, 65, 65),
            font=small_font,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, optimize=True)
    sheet.save(destination.with_suffix(".jpg"), quality=97, subsampling=0, optimize=True)


def teacher_parity(
    records: list[dict[str, Any]],
    cache_root: Path,
    config: dict[str, Any],
    samples: int,
    tolerance: float,
    seed: int,
    findings: Findings,
) -> dict[str, Any]:
    if samples <= 0:
        return {"skipped": True, "reason": "zero samples requested"}
    if not torch.cuda.is_available():
        findings.error("teacher_parity_requires_cuda", "CUDA is not available")
        return {"skipped": True, "reason": "CUDA unavailable"}
    selected = stratified_subset(records, min(samples, len(records)), seed)
    valid: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    parity_findings = Findings()
    for record in selected:
        noisy = load_checked_array(
            cache_root, record, "input", np.dtype(np.float32), parity_findings
        )
        cached = load_checked_array(
            cache_root,
            record,
            "teacher",
            np.dtype(config["teacher"]["cache_dtype"]),
            parity_findings,
        )
        if noisy is not None and cached is not None:
            valid.append((record, noisy, cached))
    if parity_findings.failed or len(valid) != len(selected):
        findings.error("teacher_parity_input", "Parity sample arrays failed validation")
        return {"skipped": True, "reason": "invalid parity sample arrays"}
    device = torch.device("cuda")
    model = load_scunet_teacher(
        resolve_paper_path(config["teacher"]["repository"]),
        resolve_paper_path(config["teacher"]["checkpoint"]),
        device,
    )
    batch_size = int(config["teacher"].get("cache_batch_size", 4))
    fresh: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(valid), batch_size):
            batch = np.stack([row[1] for row in valid[offset : offset + batch_size]])
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
            output = model(tensor).clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy()
            fresh.extend(output)
    cached = np.stack([row[2] for row in valid])
    difference = np.abs(np.stack(fresh) - cached)
    maximum = float(difference.max())
    mean = float(difference.mean())
    if maximum > tolerance:
        findings.error(
            "teacher_parity",
            f"maximum absolute error {maximum:.8g} exceeds {tolerance:.8g}",
        )
    return {
        "skipped": False,
        "samples": len(valid),
        "sample_ids": [row[0]["id"] for row in valid],
        "maximum_absolute_error": maximum,
        "mean_absolute_error": mean,
        "tolerance": tolerance,
        "passed": maximum <= tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs/uhd_snic_data_gate.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "evaluation/uhd_snic_data_gate",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Check all metadata but sample array I/O; not a release-quality gate.",
    )
    parser.add_argument("--smoke-array-records", type=int, default=24)
    parser.add_argument("--max-sheet-rows", type=int, default=16)
    parser.add_argument("--teacher-parity-samples", type=int)
    parser.add_argument("--teacher-parity-tolerance", type=float, default=5e-4)
    parser.add_argument("--hybrid-reconstruction-tolerance", type=float, default=1e-4)
    parser.add_argument("--skip-teacher-parity", action="store_true")
    args = parser.parse_args()
    if args.hybrid_reconstruction_tolerance <= 0.0:
        parser.error("--hybrid-reconstruction-tolerance must be positive")

    config = load_config(args.config)
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    manifest_path = resolve_paper_path(config["data"]["manifest"])
    source_manifest_path = resolve_paper_path(config["data"]["source_manifest"])
    source_root = resolve_paper_path(config["data"]["source_root"])
    checkpoint = resolve_paper_path(config["teacher"]["checkpoint"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    findings = Findings()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared cache manifest does not exist: {manifest_path}")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest does not exist: {source_manifest_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {checkpoint}")

    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_document = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    records = document.get("records") if isinstance(document, dict) else None
    source_records = source_document.get("records") if isinstance(source_document, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("Prepared cache manifest has no records")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("Domain source manifest has no records")

    allowed_datasets = set(config["data"]["datasets"])
    patch_counts = {
        str(dataset): int(count)
        for dataset, count in config["data"]["patches_per_pair"].items()
    }
    target_config = config["uhd_hybrid_target"]
    jpeg_quality = config["data"].get("jpeg_quality", {})
    seed = int(config["project"]["seed"])

    actual_source_hash = sha256_file(source_manifest_path)
    actual_checkpoint_hash = sha256_file(checkpoint)
    if document.get("preprocessing") != config["project"]["preprocessing_version"]:
        findings.error(
            "preprocessing_mismatch",
            f"manifest={document.get('preprocessing')!r}, config={config['project']['preprocessing_version']!r}",
        )
    if document.get("source_manifest_sha256") != actual_source_hash:
        findings.error(
            "source_manifest_hash",
            f"stored={document.get('source_manifest_sha256')!r}, actual={actual_source_hash}",
        )
    if document.get("teacher_checkpoint_sha256") != actual_checkpoint_hash:
        findings.error(
            "teacher_checkpoint_hash",
            f"stored={document.get('teacher_checkpoint_sha256')!r}, actual={actual_checkpoint_hash}",
        )
    if document.get("uhd_hybrid_target") != target_config:
        findings.error(
            "uhd_hybrid_target_metadata",
            "Manifest target parameters do not exactly match the active configuration",
        )

    source_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    expected_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    source_scene_splits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for source in source_records:
        if source.get("dataset") not in allowed_datasets:
            continue
        key = (str(source.get("dataset")), str(source.get("input")), str(source.get("clean")))
        if key in source_by_key:
            findings.error("duplicate_source_pair", repr(key))
        source_by_key[key] = source
        dataset = str(source.get("dataset"))
        split = str(source.get("split"))
        expected_transform = UHD_CLEAN_TRANSFORM if dataset == "uhd_ll" else "identity"
        if source.get("clean_transform") != expected_transform:
            findings.error(
                "source_clean_transform",
                f"{dataset}/{source.get('scene')}: {source.get('clean_transform')!r}, "
                f"expected {expected_transform!r}",
            )
        if dataset not in patch_counts:
            findings.error("missing_patch_count", dataset)
            continue
        expected_counts[(split, dataset)] += patch_counts[dataset]
        source_scene_splits[(dataset, str(source.get("scene")))].add(split)
        for key_name in ("input", "clean"):
            path = contained_path(source_root, source.get(key_name))
            if path is None:
                findings.error(
                    "unsafe_source_path",
                    f"{dataset}/{source.get('scene')}/{key_name}: {source.get(key_name)!r}",
                )
            elif not path.is_file() or path.stat().st_size <= 0:
                findings.error("missing_source_file", str(path))
    for key, splits in source_scene_splits.items():
        if len(splits) > 1:
            findings.error("source_scene_leakage", f"{key}: {sorted(splits)}")
    uhd_source_counts = collections.Counter(
        str(source.get("split"))
        for source in source_by_key.values()
        if source.get("dataset") == "uhd_ll"
    )
    for split, expected in EXPECTED_UHD_SOURCE_PAIRS.items():
        if uhd_source_counts[split] != expected:
            findings.error(
                "uhd_source_count",
                f"{split}: source manifest={uhd_source_counts[split]}, expected={expected}",
            )

    ids: set[str] = set()
    array_paths: dict[str, set[str]] = {key: set() for key in ARRAY_KEYS}
    scene_splits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    cache_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    supervision_counts: collections.Counter[tuple[str, str, str]] = collections.Counter()
    source_patch_counts: collections.Counter[tuple[str, str, str]] = collections.Counter()
    source_crop_sets: dict[tuple[str, str, str], set[tuple[int, ...]]] = collections.defaultdict(set)
    uhd_gate_records: list[dict[str, Any]] = []
    records_by_source: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)

    for record in records:
        if not isinstance(record, dict):
            findings.error("record_type", repr(type(record)))
            continue
        validate_record_semantics(record, allowed_datasets, target_config, jpeg_quality, findings)
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            findings.error("missing_id", repr(record_id))
        elif record_id in ids:
            findings.error("duplicate_id", record_id)
        else:
            ids.add(record_id)
        dataset = str(record.get("dataset"))
        split = str(record.get("split"))
        scene = str(record.get("scene"))
        cache_counts[(split, dataset)] += 1
        supervision_counts[(split, dataset, str(record.get("supervision")))] += 1
        scene_splits[(dataset, scene)].add(split)
        source_key = (dataset, str(record.get("source_input")), str(record.get("source_clean")))
        source_patch_counts[source_key] += 1
        records_by_source[source_key].append(record)
        crop = record.get("crop")
        if isinstance(crop, list) and all(isinstance(value, int) for value in crop):
            crop_key = tuple(crop)
            if crop_key in source_crop_sets[source_key]:
                findings.error("duplicate_source_crop", f"{source_key}: {crop_key}")
            source_crop_sets[source_key].add(crop_key)
        gate = record.get("uhd_hybrid_gate")
        if isinstance(gate, dict):
            try:
                canonical_json(gate)
            except (TypeError, ValueError) as error:
                findings.error(
                    "unserializable_uhd_hybrid_gate",
                    f"{record_id}: {error}",
                )
            if dataset == "uhd_ll":
                uhd_gate_records.append(record)
        source = source_by_key.get(source_key)
        if source is None:
            findings.error("orphan_cache_source", repr(source_key))
        elif source.get("split") != record.get("split") or source.get("scene") != record.get("scene"):
            findings.error(
                "cache_source_metadata",
                f"{record_id}: cache {record.get('split')}/{record.get('scene')}, "
                f"source {source.get('split')}/{source.get('scene')}",
            )
        for key_name in ARRAY_KEYS:
            relative = record.get(key_name)
            if isinstance(relative, str):
                if relative in array_paths[key_name]:
                    findings.error("duplicate_array_path", f"{key_name}: {relative}")
                array_paths[key_name].add(relative)
                parts = Path(relative).parts
                if len(parts) < 3 or parts[0] != split or parts[1] != dataset:
                    findings.error(
                        "array_layout",
                        f"{record_id}/{key_name}: {relative}, expected {split}/{dataset}/...",
                    )

    leakage = {
        f"{dataset}/{scene}": sorted(splits)
        for (dataset, scene), splits in scene_splits.items()
        if len(splits) > 1
    }
    for scene, splits in leakage.items():
        findings.error("cache_scene_leakage", f"{scene}: {splits}")

    missing_sources = sorted(set(source_by_key) - set(source_patch_counts))
    if args.smoke:
        if missing_sources:
            findings.warn(
                "partial_cache_sources",
                f"Smoke mode ignores {len(missing_sources)} source pairs absent from the cache",
            )
    else:
        for source_key in missing_sources:
            findings.error("missing_cache_source", repr(source_key))
    for source_key, count in source_patch_counts.items():
        expected = patch_counts.get(source_key[0])
        if expected is not None and count != expected:
            message = f"{source_key}: {count} patches, expected {expected}"
            if args.smoke:
                findings.warn("partial_source_patch_count", message)
            else:
                findings.error("source_patch_count", message)
        source_rows = records_by_source[source_key]
        semantic_values = {
            (
                row.get("supervision"),
                row.get("gt_weight"),
                row.get("kd_weight"),
                row.get("jpeg_quality"),
            )
            for row in source_rows
        }
        if source_key[0] == "snic_sony" and len(semantic_values) > 1:
            findings.error("inconsistent_source_supervision", repr(source_key))
    if not args.smoke:
        for key in set(expected_counts) | set(cache_counts):
            if expected_counts[key] != cache_counts[key]:
                findings.error(
                    "dataset_count",
                    f"{key[0]}/{key[1]}: cache={cache_counts[key]}, expected={expected_counts[key]}",
                )

    io_records = (
        stratified_subset(records, max(1, args.smoke_array_records), seed)
        if args.smoke
        else records
    )
    teacher_digest = hashlib.sha256()
    metric_rows: list[dict[str, Any]] = []
    expected_dtypes = {
        "input": np.dtype(np.float32),
        "clean": np.dtype(np.float32),
        "teacher": np.dtype(config["teacher"]["cache_dtype"]),
    }
    for record in tqdm(io_records, desc="Validating UHD-LL/SNIC cache arrays"):
        arrays = {
            key: load_checked_array(
                cache_root,
                record,
                key,
                expected_dtypes[key],
                findings,
                teacher_digest if key == "teacher" else None,
            )
            for key in ARRAY_KEYS
        }
        if all(value is not None for value in arrays.values()):
            typed_arrays = {key: value for key, value in arrays.items() if value is not None}
            metric_rows.append(array_metrics(record, typed_arrays))

    reconstruction = reconstruct_uhd_cache(
        io_records if args.smoke else records,
        source_by_key,
        source_root,
        cache_root,
        config,
        args.hybrid_reconstruction_tolerance,
        findings,
    )
    if reconstruction["records_reconstructed"] != reconstruction["records_requested"]:
        findings.error(
            "reconstruction_coverage",
            f"requested={reconstruction['records_requested']}, "
            f"reconstructed={reconstruction['records_reconstructed']}",
        )

    selected_rows = representative_rows(metric_rows, max(1, args.max_sheet_rows))
    contact_png = output_dir / "data_gate_contact_sheet.png"
    if selected_rows:
        render_contact_sheet(selected_rows, cache_root, contact_png)
    else:
        findings.error("contact_sheet_empty", "No valid arrays were available to render")

    parity_samples = args.teacher_parity_samples
    if parity_samples is None:
        parity_samples = 0 if args.smoke else 8
    if args.skip_teacher_parity:
        parity_samples = 0
    parity = teacher_parity(
        io_records,
        cache_root,
        config,
        parity_samples,
        args.teacher_parity_tolerance,
        seed ^ 0x5C11,
        findings,
    )

    dataset_summaries = {}
    for dataset in sorted(allowed_datasets):
        dataset_rows = [row for row in metric_rows if row["dataset"] == dataset]
        dataset_summaries[dataset] = {
            "array_records_checked": len(dataset_rows),
            "mean_luminance": finite_summary(row["mean_luminance"] for row in dataset_rows),
            "shadow_fraction": finite_summary(row["shadow_fraction"] for row in dataset_rows),
            "teacher_correction_mae": finite_summary(
                row["teacher_correction_mae"] for row in dataset_rows
            ),
            "teacher_psnr_gain_db": finite_summary(
                row["teacher_psnr_gain"] for row in dataset_rows
            ),
            "teacher_better_fraction": (
                float(np.mean([row["teacher_psnr_gain"] > 0.0 for row in dataset_rows]))
                if dataset_rows
                else None
            ),
        }

    gate_report_path = manifest_path.parent / "uhd_hybrid_gate.json"
    stored_gate_summary: dict[str, Any] | None = None
    if gate_report_path.is_file():
        stored_gate_summary = json.loads(gate_report_path.read_text(encoding="utf-8"))
        if stored_gate_summary.get("target_parameters") != target_config:
            findings.error(
                "uhd_hybrid_gate_parameters",
                "Stored gate parameters do not exactly match the active configuration",
            )
        actual_sources = {
            (record["split"], record["source_input"])
            for record in uhd_gate_records
        }
        actual_passed = sum(bool(record["uhd_hybrid_gate"]["passed"]) for record in uhd_gate_records)
        expected_summary = {
            "source_pairs": len(actual_sources),
            "crops": len(uhd_gate_records),
            "passed": actual_passed,
            "failed": len(uhd_gate_records) - actual_passed,
        }
        for key, value in expected_summary.items():
            if stored_gate_summary.get(key) != value:
                findings.error(
                    "uhd_hybrid_gate_report",
                    f"{key}: stored={stored_gate_summary.get(key)!r}, actual={value}",
                )
    else:
        findings.error("missing_uhd_hybrid_gate_report", str(gate_report_path))

    mode = "smoke" if args.smoke else "full"
    status = f"{mode}_{'failed' if findings.failed else 'passed'}"
    report = {
        "schema_version": 1,
        "status": status,
        "mode": mode,
        "release_gate_passed": bool(not args.smoke and not findings.failed),
        "preprocessing": config["project"]["preprocessing_version"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": actual_source_hash,
        "cache_root": str(cache_root),
        "records": len(records),
        "array_records_checked": len(io_records),
        "array_records_valid": len(metric_rows),
        "counts": {
            "expected": {
                f"{split}/{dataset}": count
                for (split, dataset), count in sorted(expected_counts.items())
            },
            "cache": {
                f"{split}/{dataset}": count
                for (split, dataset), count in sorted(cache_counts.items())
            },
            "supervision": {
                f"{split}/{dataset}/{supervision}": count
                for (split, dataset, supervision), count in sorted(supervision_counts.items())
            },
        },
        "scenes": {
            "groups": len(scene_splits),
            "leakage": leakage,
        },
        "teacher": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_checkpoint_hash,
            "manifest_checkpoint_sha256": document.get("teacher_checkpoint_sha256"),
            "array_content_sha256": teacher_digest.hexdigest(),
            "array_hash_coverage": "sampled" if args.smoke else "complete",
            "array_files_hashed": len(metric_rows),
            "cache_dtype": str(expected_dtypes["teacher"]),
            "parity": parity,
        },
        "uhd_hybrid_gate": {
            "report": str(gate_report_path),
            "source_pairs": len(
                {(record["split"], record["source_input"]) for record in uhd_gate_records}
            ),
            "crops": len(uhd_gate_records),
            "passed": sum(
                bool(record["uhd_hybrid_gate"]["passed"]) for record in uhd_gate_records
            ),
            "failed": sum(
                not bool(record["uhd_hybrid_gate"]["passed"])
                for record in uhd_gate_records
            ),
            "target_parameters": target_config,
            "source_reconstruction": {
                **reconstruction,
                "coverage": "sampled_non_release" if args.smoke else "complete",
            },
        },
        "dataset_summaries": dataset_summaries,
        "contact_sheet": (
            {
                "png": str(contact_png),
                "png_sha256": sha256_file(contact_png),
                "jpg": str(contact_png.with_suffix(".jpg")),
                "jpg_sha256": sha256_file(contact_png.with_suffix(".jpg")),
                "rows": [
                    {
                        key: row.get(key)
                        for key in (
                            "id",
                            "dataset",
                            "split",
                            "scene",
                            "supervision",
                            "noise_level",
                            "crop",
                            "mean_luminance",
                            "shadow_fraction",
                            "teacher_correction_mae",
                            "noisy_reference_psnr",
                            "teacher_reference_psnr",
                            "teacher_psnr_gain",
                            "uhd_hybrid_gate",
                        )
                    }
                    for row in selected_rows
                ],
            }
            if selected_rows
            else None
        ),
        "findings": findings.report(),
        "decision": (
            "This is a release-quality data gate; visually inspect the contact sheet before "
            "starting training."
            if not args.smoke and not findings.failed
            else "Smoke mode does not approve a cache for training. Run the full gate next."
            if args.smoke and not findings.failed
            else "Do not train from this cache until all reported errors are resolved."
        ),
    }
    destination = output_dir / "data_gate_report.json"
    atomic_json(destination, report)
    print(
        json.dumps(
            {
                "status": status,
                "report": str(destination),
                "contact_sheet": str(contact_png) if selected_rows else None,
                "records": len(records),
                "array_records_checked": len(io_records),
                "errors": report["findings"]["errors"]["total"],
                "warnings": report["findings"]["warnings"]["total"],
                "teacher_array_sha256": teacher_digest.hexdigest(),
            },
            indent=2,
        )
    )
    if findings.failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
