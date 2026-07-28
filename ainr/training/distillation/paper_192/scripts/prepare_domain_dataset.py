#!/usr/bin/env python3
"""Prepare exact-192 UHD-LL/SNIC crops with audited supervision weights."""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, sobel
from tqdm import tqdm

from common import atomic_json, load_config, resolve_paper_path, seed_everything, sha256_file
from src.scunet_teacher import load_scunet_teacher


TILE = 192
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
UHD_CLEAN_TRANSFORM = "linear_rgb_local_ratio128_teacher_lowpass8_v2"
UHD_HYBRID_SUPERVISION = "uhd_hybrid_paired"
UHD_HYBRID_TARGET = "teacher_lowpass_plus_local_clean_highpass_linear_rgb"


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * value ** (1.0 / 2.4) - 0.055)


def luminance(value: np.ndarray) -> np.ndarray:
    return value[..., :3] @ LUMA


class ImageSource:
    """Decode JPEG or memory-map a 16-bit RGB TIFF without a full float copy."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._pil: Image.Image | None = None
        self._array: np.ndarray | None = None
        if path.suffix.lower() in {".tif", ".tiff"}:
            value = tifffile.memmap(path)
            if value.ndim != 3 or value.shape[2] < 3:
                raise ValueError(f"SNIC TIFF must be HWC RGB, got {value.shape}: {path}")
            self._array = value[..., :3]
            self.height, self.width = map(int, self._array.shape[:2])
        else:
            self._pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            self.width, self.height = self._pil.size

    def close(self) -> None:
        if self._pil is not None:
            self._pil.close()

    def __enter__(self) -> "ImageSource":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        self.close()

    @staticmethod
    def _float_rgb(value: np.ndarray) -> np.ndarray:
        if np.issubdtype(value.dtype, np.integer):
            scale = float(np.iinfo(value.dtype).max)
            result = value.astype(np.float32) / scale
        elif np.issubdtype(value.dtype, np.floating):
            result = value.astype(np.float32)
        else:
            raise ValueError(f"Unsupported image dtype: {value.dtype}")
        if not np.isfinite(result).all() or float(result.min()) < 0.0 or float(result.max()) > 1.0:
            raise ValueError("Decoded image is non-finite or outside [0, 1]")
        return result

    def crop(self, left: int, top: int, size: int = TILE) -> np.ndarray:
        if self._pil is not None:
            return np.asarray(
                self._pil.crop((left, top, left + size, top + size)), dtype=np.float32
            ) / 255.0
        assert self._array is not None
        return self._float_rgb(self._array[top : top + size, left : left + size])

    def thumbnail(self, target_width: int) -> np.ndarray:
        target_height = max(1, round(self.height * target_width / self.width))
        if self._pil is not None:
            image = self._pil.resize((target_width, target_height), Image.Resampling.BOX)
            return np.asarray(image, dtype=np.float32) / 255.0
        assert self._array is not None
        stride = max(1, self.width // target_width)
        sampled = self._float_rgb(self._array[::stride, ::stride])
        image = Image.fromarray(np.rint(sampled * 255.0).astype(np.uint8), "RGB")
        image = image.resize((target_width, target_height), Image.Resampling.BOX)
        return np.asarray(image, dtype=np.float32) / 255.0


def gaussian_rgb(value: np.ndarray, sigma: float) -> np.ndarray:
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB data, got {value.shape}")
    if sigma <= 0.0:
        raise ValueError(f"Gaussian sigma must be positive, got {sigma}")
    return gaussian_filter(value, sigma=(sigma, sigma, 0.0), mode="reflect")


def build_local_gain_field(
    noisy_thumbnail: np.ndarray,
    clean_thumbnail: np.ndarray,
    source_width: int,
    config: dict[str, Any],
) -> np.ndarray:
    """Estimate a stable low-frequency clean-to-noisy gain in linear RGB."""
    if noisy_thumbnail.shape != clean_thumbnail.shape:
        raise ValueError(
            f"UHD thumbnail geometry differs: {noisy_thumbnail.shape} and {clean_thumbnail.shape}"
        )
    if source_width < noisy_thumbnail.shape[1]:
        raise ValueError("UHD target thumbnail cannot be wider than its source")
    sigma = (
        float(config["illumination_sigma_full_resolution"])
        * noisy_thumbnail.shape[1]
        / source_width
    )
    noisy_low = gaussian_rgb(srgb_to_linear(noisy_thumbnail), sigma)
    clean_low = gaussian_rgb(srgb_to_linear(clean_thumbnail), sigma)
    denominator_floor = float(config["channel_denominator_floor"])
    confidence_scale = float(config["channel_confidence_scale"])
    luminance_gain = luminance(noisy_low) / np.maximum(
        luminance(clean_low), denominator_floor
    )
    channel_gain = noisy_low / np.maximum(clean_low, denominator_floor)
    confidence = np.clip(clean_low / confidence_scale, 0.0, 1.0)
    gain = confidence * channel_gain + (1.0 - confidence) * luminance_gain[..., None]
    gain = np.clip(
        gain,
        float(config["minimum_gain"]),
        float(config["maximum_gain"]),
    )
    gain = gaussian_rgb(gain, float(config["gain_smoothing_sigma_thumbnail"]))
    return gain.astype(np.float32)


def sample_thumbnail_field(
    field: np.ndarray,
    left: int,
    top: int,
    source_width: int,
    source_height: int,
    size: int = TILE,
) -> np.ndarray:
    """Bilinearly sample a thumbnail field at full-resolution crop centers."""
    field_height, field_width = field.shape[:2]
    x = (left + np.arange(size, dtype=np.float32) + 0.5) * field_width / source_width - 0.5
    y = (top + np.arange(size, dtype=np.float32) + 0.5) * field_height / source_height - 0.5
    x = np.clip(x, 0.0, field_width - 1.0)
    y = np.clip(y, 0.0, field_height - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, field_width - 1)
    y1 = np.minimum(y0 + 1, field_height - 1)
    wx = (x - x0).reshape(1, size, 1)
    wy = (y - y0).reshape(size, 1, 1)
    top_row = field[y0[:, None], x0[None, :]] * (1.0 - wx) + field[
        y0[:, None], x1[None, :]
    ] * wx
    bottom_row = field[y1[:, None], x0[None, :]] * (1.0 - wx) + field[
        y1[:, None], x1[None, :]
    ] * wx
    return (top_row * (1.0 - wy) + bottom_row * wy).astype(np.float32)


def apply_local_gain(clean: np.ndarray, gain: np.ndarray) -> np.ndarray:
    if clean.shape != gain.shape:
        raise ValueError(f"Clean crop and gain field differ: {clean.shape} and {gain.shape}")
    mapped = srgb_to_linear(clean) * gain
    return linear_to_srgb(np.clip(mapped, 0.0, 1.0)).astype(np.float32)


def signed_gradients(value: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    gray = gaussian_filter(luminance(value), sigma=sigma, mode="reflect")
    return (
        sobel(gray, axis=1, mode="reflect"),
        sobel(gray, axis=0, mode="reflect"),
    )


def gradient_component_correlation(
    first_x: np.ndarray,
    first_y: np.ndarray,
    second_x: np.ndarray,
    second_y: np.ndarray,
    first_threshold: float | None = None,
    second_threshold: float | None = None,
) -> float:
    if first_threshold is None:
        first_threshold = gradient_threshold(first_x, first_y)
    if second_threshold is None:
        second_threshold = gradient_threshold(second_x, second_y)
    count = 0
    sum_first = 0.0
    sum_second = 0.0
    sum_first_square = 0.0
    sum_second_square = 0.0
    sum_product = 0.0
    for first, second in ((first_x, second_x), (first_y, second_y)):
        mask = (np.abs(first) > first_threshold) | (np.abs(second) > second_threshold)
        selected_first = first[mask].astype(np.float64)
        selected_second = second[mask].astype(np.float64)
        count += int(selected_first.size)
        sum_first += float(selected_first.sum())
        sum_second += float(selected_second.sum())
        # BLAS thread startup dominates these small reductions. Direct NumPy
        # sums are deterministic and materially faster for 192 px crops.
        sum_first_square += float(np.square(selected_first).sum())
        sum_second_square += float(np.square(selected_second).sum())
        sum_product += float((selected_first * selected_second).sum())
    if count < 256:
        return 0.0
    first_variance = sum_first_square - sum_first * sum_first / count
    second_variance = sum_second_square - sum_second * sum_second / count
    if first_variance <= 1e-14 or second_variance <= 1e-14:
        return 0.0
    covariance = sum_product - sum_first * sum_second / count
    correlation = covariance / math.sqrt(first_variance * second_variance)
    return correlation if math.isfinite(correlation) else 0.0


def gradient_threshold(horizontal: np.ndarray, vertical: np.ndarray) -> float:
    magnitudes = np.concatenate((np.abs(horizontal).ravel(), np.abs(vertical).ravel()))
    return float(np.quantile(magnitudes, 0.55))


def signed_gradient_correlation(first: np.ndarray, second: np.ndarray, sigma: float) -> float:
    first_x, first_y = signed_gradients(first, sigma)
    second_x, second_y = signed_gradients(second, sigma)
    return gradient_component_correlation(first_x, first_y, second_x, second_y)


def shifted_views(
    first: np.ndarray,
    second: np.ndarray,
    vertical: int,
    horizontal: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = first.shape[:2]
    first_y = slice(max(0, vertical), min(height, height + vertical))
    first_x = slice(max(0, horizontal), min(width, width + horizontal))
    second_y = slice(max(0, -vertical), min(height, height - vertical))
    second_x = slice(max(0, -horizontal), min(width, width - horizontal))
    return first[first_y, first_x], second[second_y, second_x]


def alignment_gate(
    noisy: np.ndarray,
    mapped_clean: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    sigma = float(config["alignment_prefilter_sigma"])
    noisy_x, noisy_y = signed_gradients(noisy, sigma)
    clean_x, clean_y = signed_gradients(mapped_clean, sigma)
    noisy_threshold = gradient_threshold(noisy_x, noisy_y)
    clean_threshold = gradient_threshold(clean_x, clean_y)
    texture = float(np.mean(np.hypot(clean_x, clean_y)))
    radius = int(config["alignment_search_radius"])
    scores: dict[tuple[int, int], float] = {}
    for vertical in range(-radius, radius + 1):
        for horizontal in range(-radius, radius + 1):
            noisy_x_view, clean_x_view = shifted_views(
                noisy_x, clean_x, vertical, horizontal
            )
            noisy_y_view, clean_y_view = shifted_views(
                noisy_y, clean_y, vertical, horizontal
            )
            scores[(vertical, horizontal)] = gradient_component_correlation(
                noisy_x_view,
                noisy_y_view,
                clean_x_view,
                clean_y_view,
                noisy_threshold,
                clean_threshold,
            )
    best_shift, best_correlation = max(scores.items(), key=lambda item: item[1])
    zero_correlation = scores[(0, 0)]
    alignment_gain = best_correlation - zero_correlation
    failures = []
    if texture < float(config["minimum_texture"]):
        failures.append("insufficient_texture")
    if zero_correlation < float(config["minimum_zero_shift_correlation"]):
        failures.append("zero_shift_correlation")
    if best_shift != (0, 0) and alignment_gain > float(config["maximum_nonzero_shift_gain"]):
        failures.append("nonzero_shift_improvement")
    return {
        "texture": texture,
        "zero_edge_correlation": zero_correlation,
        "best_edge_correlation": best_correlation,
        "best_shift": list(best_shift),
        "alignment_gain": alignment_gain,
        "failure_reasons": failures,
        "passed": not failures,
    }


def build_uhd_hybrid_target(
    teacher: np.ndarray,
    mapped_clean: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    if teacher.shape != mapped_clean.shape:
        raise ValueError(
            f"Teacher and mapped-clean crops differ: {teacher.shape} and {mapped_clean.shape}"
        )
    sigma = float(config["teacher_lowpass_sigma"])
    detail_gain = float(config["clean_detail_gain"])
    teacher_linear = srgb_to_linear(teacher)
    clean_linear = srgb_to_linear(mapped_clean)
    target = gaussian_rgb(teacher_linear, sigma) + detail_gain * (
        clean_linear - gaussian_rgb(clean_linear, sigma)
    )
    return linear_to_srgb(np.clip(target, 0.0, 1.0)).astype(np.float32)


def edge_map(value: np.ndarray) -> np.ndarray:
    gray = luminance(value)
    horizontal = np.zeros_like(gray)
    vertical = np.zeros_like(gray)
    horizontal[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    vertical[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.sqrt(horizontal * horizontal + vertical * vertical)


def edge_correlation(first: np.ndarray, second: np.ndarray) -> float:
    one = edge_map(first).ravel()
    two = edge_map(second).ravel()
    threshold_one = np.quantile(one, 0.5)
    threshold_two = np.quantile(two, 0.5)
    mask = (one > threshold_one) | (two > threshold_two)
    if int(mask.sum()) < 128 or float(one[mask].std()) < 1e-8 or float(two[mask].std()) < 1e-8:
        return 0.0
    correlation = float(np.corrcoef(one[mask], two[mask])[0, 1])
    return correlation if math.isfinite(correlation) else 0.0


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    mse = float(np.mean(np.square(first.astype(np.float64) - second.astype(np.float64))))
    return math.inf if mse == 0.0 else -10.0 * math.log10(mse)


def stratified_positions(
    thumbnail: np.ndarray,
    width: int,
    height: int,
    count: int,
    candidates: int,
    seed: int,
) -> list[tuple[int, int, float]]:
    if width < TILE or height < TILE:
        raise ValueError(f"Source geometry {width}x{height} is smaller than {TILE}")
    generator = random.Random(seed)
    thumb_height, thumb_width = thumbnail.shape[:2]
    options = []
    for _ in range(max(candidates, count)):
        left = generator.randint(0, width - TILE)
        top = generator.randint(0, height - TILE)
        x0 = min(thumb_width - 1, int(left * thumb_width / width))
        y0 = min(thumb_height - 1, int(top * thumb_height / height))
        x1 = max(x0 + 1, min(thumb_width, math.ceil((left + TILE) * thumb_width / width)))
        y1 = max(y0 + 1, min(thumb_height, math.ceil((top + TILE) * thumb_height / height)))
        mean = float(luminance(thumbnail[y0:y1, x0:x1]).mean())
        options.append((left, top, mean))
    options.sort(key=lambda value: value[2])
    quantiles = np.linspace(0.03, 0.90, count)
    chosen = []
    used = set()
    for quantile in quantiles:
        index = round(float(quantile) * (len(options) - 1))
        order = sorted(range(len(options)), key=lambda candidate: abs(candidate - index))
        selected = next(options[candidate] for candidate in order if options[candidate][:2] not in used)
        chosen.append(selected)
        used.add(selected[:2])
    return chosen


def jpeg_roundtrip(value: np.ndarray, quality: int | None) -> np.ndarray:
    quantized = np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
    if quality is None:
        return quantized.astype(np.float32) / 255.0
    buffer = io.BytesIO()
    Image.fromarray(quantized, "RGB").save(
        buffer, format="JPEG", quality=quality, subsampling=0, optimize=False
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255.0


def source_path(source_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ValueError(f"Source manifest path must be relative and contained: {value!r}")
    resolved = (source_root / relative).resolve()
    if not resolved.is_relative_to(source_root.resolve()):
        raise ValueError(f"Source manifest path escapes source root: {value!r}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Source image is missing: {resolved}")
    return resolved


def validate_settings(config: dict[str, Any]) -> None:
    data = config["data"]
    if int(data["tile_size"]) != TILE:
        raise ValueError(f"This pipeline requires tile_size={TILE}, got {data['tile_size']}")
    if int(data["crop_candidates"]) < 1:
        raise ValueError("crop_candidates must be positive")
    datasets = list(data["datasets"])
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("data.datasets must be non-empty and unique")
    for dataset in datasets:
        if int(data["patches_per_pair"].get(dataset, 0)) < 1:
            raise ValueError(f"patches_per_pair must be positive for {dataset}")
    validation_patch_counts = data.get(
        "validation_patches_per_pair", data["patches_per_pair"]
    )
    if set(validation_patch_counts) != set(datasets):
        raise ValueError(
            "validation_patches_per_pair must exactly match data.datasets"
        )
    for dataset in datasets:
        if int(validation_patch_counts.get(dataset, 0)) < 1:
            raise ValueError(
                f"validation_patches_per_pair must be positive for {dataset}"
            )
    for dataset, quality in data.get("jpeg_quality", {}).items():
        if dataset not in datasets or not 1 <= int(quality) <= 100:
            raise ValueError(f"Invalid JPEG quality for {dataset}: {quality}")
    target = config["uhd_hybrid_target"]
    if int(target["thumbnail_width"]) < 64:
        raise ValueError("uhd_hybrid_target.thumbnail_width must be at least 64")
    positive_keys = (
        "illumination_sigma_full_resolution",
        "channel_denominator_floor",
        "channel_confidence_scale",
        "minimum_gain",
        "maximum_gain",
        "gain_smoothing_sigma_thumbnail",
        "teacher_lowpass_sigma",
        "clean_detail_gain",
        "alignment_prefilter_sigma",
        "minimum_texture",
        "minimum_zero_shift_correlation",
        "maximum_nonzero_shift_gain",
    )
    for key in positive_keys:
        if float(target[key]) <= 0.0:
            raise ValueError(f"uhd_hybrid_target.{key} must be positive")
    if float(target["minimum_gain"]) >= float(target["maximum_gain"]):
        raise ValueError("uhd_hybrid_target gain limits are inverted")
    if int(target["alignment_search_radius"]) < 1:
        raise ValueError("uhd_hybrid_target.alignment_search_radius must be positive")
    if int(config["teacher"]["cache_batch_size"]) < 1:
        raise ValueError("teacher.cache_batch_size must be positive")


def validate_source_records(sources: list[dict[str, Any]], source_root: Path) -> None:
    identities: set[tuple[str, str, str]] = set()
    scene_splits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for index, source in enumerate(sources):
        missing = {key for key in ("dataset", "scene", "split", "input", "clean") if key not in source}
        if missing:
            raise ValueError(f"Source record {index} is missing fields: {sorted(missing)}")
        if source["split"] not in {"train", "validation"}:
            raise ValueError(f"Invalid source split in record {index}: {source['split']}")
        dataset = str(source["dataset"])
        transform = source.get("clean_transform")
        if dataset == "uhd_ll" and transform != UHD_CLEAN_TRANSFORM:
            raise ValueError(
                f"UHD source record {index} uses stale clean transform {transform!r}; "
                "rebuild the source manifest"
            )
        if dataset == "snic_sony" and transform != "identity":
            raise ValueError(f"SNIC source record {index} must use identity clean transform")
        noisy_path = source_path(source_root, str(source["input"]))
        clean_path = source_path(source_root, str(source["clean"]))
        identity = (str(source["dataset"]), str(noisy_path), str(clean_path))
        if identity in identities:
            raise ValueError(f"Duplicate source pair in manifest: {source['input']}")
        identities.add(identity)
        scene_splits[(str(source["dataset"]), str(source["scene"]))].add(str(source["split"]))
    leakage = [key for key, splits in scene_splits.items() if len(splits) != 1]
    if leakage:
        raise RuntimeError(f"Source-manifest scene leakage detected: {leakage[0]}")


def crop_seed(source: dict[str, Any], seed: int) -> int:
    identity = f"{source['dataset']}:{source['scene']}:{source['clean']}"
    return seed ^ int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)


def audit_sources(
    sources: list[dict[str, Any]],
    source_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    target_config = config["uhd_hybrid_target"]
    train_patch_counts = config["data"]["patches_per_pair"]
    validation_patch_counts = config["data"].get(
        "validation_patches_per_pair", train_patch_counts
    )
    candidates = int(config["data"]["crop_candidates"])
    seed = int(config["project"]["seed"])
    for source in tqdm(sources, desc="Auditing UHD-LL/SNIC source pairs"):
        noisy_path = source_path(source_root, str(source["input"]))
        clean_path = source_path(source_root, str(source["clean"]))
        with ImageSource(noisy_path) as noisy_source, ImageSource(clean_path) as clean_source:
            geometry = [noisy_source.width, noisy_source.height]
            if geometry != [clean_source.width, clean_source.height]:
                raise ValueError(f"Geometry mismatch: {noisy_path} and {clean_path}")
            if min(geometry) < TILE:
                raise ValueError(f"Source geometry is smaller than {TILE}: {noisy_path}")
            thumbnail_width = int(target_config["thumbnail_width"])
            noisy_thumbnail = noisy_source.thumbnail(thumbnail_width)
            clean_thumbnail = clean_source.thumbnail(thumbnail_width)
            row = {
                "dataset": str(source["dataset"]),
                "scene": str(source["scene"]),
                "split": str(source["split"]),
                "source_input": str(source["input"]),
                "source_clean": str(source["clean"]),
                "geometry": geometry,
                "input_mean_luminance": float(luminance(noisy_thumbnail).mean()),
                "clean_mean_luminance": float(luminance(clean_thumbnail).mean()),
                "direct_psnr": psnr(noisy_thumbnail, clean_thumbnail),
                "direct_edge_correlation": edge_correlation(noisy_thumbnail, clean_thumbnail),
            }
            if source.get("clean_transform") == UHD_CLEAN_TRANSFORM:
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
                    int(
                        (
                            train_patch_counts
                            if source["split"] == "train"
                            else validation_patch_counts
                        )["uhd_ll"]
                    ),
                    candidates,
                    crop_seed(source, seed),
                )
                source_gates = []
                for crop_index, (left, top, crop_luma) in enumerate(positions):
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
                    gate = alignment_gate(noisy, mapped_clean, target_config)
                    gate.update(
                        {
                            "crop_index": crop_index,
                            "crop": [left, top, TILE, TILE],
                            "crop_mean_luminance": crop_luma,
                            "local_gain_minimum": float(gain.min()),
                            "local_gain_mean": gain.mean(axis=(0, 1)).tolist(),
                            "local_gain_maximum": float(gain.max()),
                        }
                    )
                    gate_row = {
                        "dataset": source["dataset"],
                        "scene": source["scene"],
                        "split": source["split"],
                        "source_input": source["input"],
                        **gate,
                    }
                    source_gates.append(gate_row)
                    gate_rows.append(gate_row)
                row["uhd_hybrid_gate"] = {
                    "crops": len(source_gates),
                    "passed": sum(bool(gate["passed"]) for gate in source_gates),
                    "failed": sum(not bool(gate["passed"]) for gate in source_gates),
                }
            rows.append(row)

    gate_counts = collections.Counter(
        (row["split"], "passed" if row["passed"] else "failed") for row in gate_rows
    )
    failure_counts = collections.Counter(
        reason for row in gate_rows for reason in row["failure_reasons"]
    )
    metric_summary = {}
    for key in (
        "texture",
        "zero_edge_correlation",
        "best_edge_correlation",
        "alignment_gain",
    ):
        values = np.asarray([float(row[key]) for row in gate_rows], dtype=np.float64)
        if values.size:
            metric_summary[key] = {
                "min": float(values.min()),
                "p05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
            }
    return {
        "schema_version": 2,
        "source_pairs": len(rows),
        "dataset_counts": dict(sorted(collections.Counter(row["dataset"] for row in rows).items())),
        "uhd_hybrid_gate": {
            "source_pairs": len(
                {(row["split"], row["source_input"]) for row in gate_rows}
            ),
            "crops": len(gate_rows),
            "passed": sum(bool(row["passed"]) for row in gate_rows),
            "failed": sum(not bool(row["passed"]) for row in gate_rows),
            "counts_by_split": {
                f"{split}/{state}": count
                for (split, state), count in sorted(gate_counts.items())
            },
            "failure_reasons": dict(sorted(failure_counts.items())),
            "metrics": metric_summary,
            "target_parameters": target_config,
            "rows": gate_rows,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs/uhd_snic_data_gate.yaml",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument(
        "--limit-pairs-per-dataset",
        type=int,
        help="Smoke-test limit applied independently to every configured dataset.",
    )
    parser.add_argument("--cache-root", type=Path, help="Override the configured cache root.")
    parser.add_argument(
        "--output-manifest",
        type=Path,
        help="Override the configured output manifest.",
    )
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="Audit every source pair and write gate statistics without loading SCUNet.",
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        help="Gate-only report path (defaults beside the configured cache directory).",
    )
    args = parser.parse_args()
    if args.limit_pairs is not None and args.limit_pairs_per_dataset is not None:
        parser.error("--limit-pairs and --limit-pairs-per-dataset are mutually exclusive")
    if args.limit_pairs is not None and args.limit_pairs < 1:
        parser.error("--limit-pairs must be positive")
    if args.limit_pairs_per_dataset is not None and args.limit_pairs_per_dataset < 1:
        parser.error("--limit-pairs-per-dataset must be positive")
    config = load_config(args.config)
    validate_settings(config)
    source_manifest = resolve_paper_path(config["data"]["source_manifest"])
    source_root = resolve_paper_path(config["data"]["source_root"])
    cache_root = (
        args.cache_root.resolve()
        if args.cache_root is not None
        else resolve_paper_path(config["data"]["cache_root"])
    )
    output_manifest = (
        args.output_manifest.resolve()
        if args.output_manifest is not None
        else resolve_paper_path(config["data"]["manifest"])
    )
    teacher_repo = resolve_paper_path(config["teacher"]["repository"])
    teacher_checkpoint = resolve_paper_path(config["teacher"]["checkpoint"])
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    document = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("records"), list):
        raise ValueError(f"Invalid source manifest: {source_manifest}")
    allowed = set(config["data"]["datasets"])
    if any(not isinstance(row, dict) or "dataset" not in row for row in document["records"]):
        raise ValueError(f"Source manifest records must be objects with a dataset: {source_manifest}")
    sources = [row for row in document["records"] if row["dataset"] in allowed]
    if args.limit_pairs is not None:
        sources = sources[: args.limit_pairs]
    elif args.limit_pairs_per_dataset is not None:
        limited = []
        seen: collections.Counter[str] = collections.Counter()
        for row in sources:
            dataset = str(row["dataset"])
            if seen[dataset] >= args.limit_pairs_per_dataset:
                continue
            limited.append(row)
            seen[dataset] += 1
        sources = limited
    if not sources:
        raise RuntimeError("Domain source manifest selection is empty")
    validate_source_records(sources, source_root)

    target_config = config["uhd_hybrid_target"]
    if args.gate_only:
        preflight_path = (
            args.preflight_report.resolve()
            if args.preflight_report is not None
            else cache_root.parent / "source_preflight.json"
        )
        preflight = audit_sources(sources, source_root, config)
        preflight.update(
            {
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": sha256_file(source_manifest),
                "preprocessing": config["project"]["preprocessing_version"],
            }
        )
        atomic_json(preflight_path, preflight)
        summary = {
            "preflight_report": str(preflight_path),
            "source_pairs": preflight["source_pairs"],
            "dataset_counts": preflight["dataset_counts"],
            "uhd_hybrid_gate": {
                key: value
                for key, value in preflight["uhd_hybrid_gate"].items()
                if key != "rows"
            },
        }
        print(json.dumps(summary, indent=2))
        return

    build_root = cache_root.with_name(f".{cache_root.name}.building")
    backup_root = cache_root.with_name(f".{cache_root.name}.previous")
    if backup_root.exists() and not cache_root.exists():
        backup_root.rename(cache_root)
    if cache_root.exists() and not args.replace:
        raise FileExistsError(f"Cache exists: {cache_root}; pass --replace")
    if build_root.exists():
        if not args.replace:
            raise FileExistsError(f"Incomplete cache build exists: {build_root}; pass --replace")
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required to cache full-precision SCUNet targets")
    teacher = load_scunet_teacher(teacher_repo, teacher_checkpoint, device)
    batch_size = int(config["teacher"]["cache_batch_size"])
    train_patch_counts = config["data"]["patches_per_pair"]
    validation_patch_counts = config["data"].get(
        "validation_patches_per_pair", train_patch_counts
    )
    candidate_count = int(config["data"]["crop_candidates"])
    jpeg_quality = config["data"].get("jpeg_quality", {})

    pending: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    output: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        noisy_batch = np.stack([row[1] for row in pending])
        tensor = torch.from_numpy(noisy_batch).permute(0, 3, 1, 2).to(device)
        with torch.inference_mode():
            prediction = teacher(tensor).clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy()
        for (metadata, noisy, clean), target in zip(pending, prediction, strict=True):
            cached_target = target.astype(np.float16)
            if metadata["supervision"] == UHD_HYBRID_SUPERVISION:
                # Build against the exact teacher tensor consumed during training,
                # so the hybrid target remains reproducible from the cache alone.
                clean = build_uhd_hybrid_target(
                    cached_target.astype(np.float32), clean, target_config
                )
            base = Path(metadata["split"]) / metadata["dataset"] / metadata["id"]
            paths = {
                name: str(base.with_name(base.name + f"_{name}.npy"))
                for name in ("input", "clean", "teacher")
            }
            for relative_path in paths.values():
                (build_root / relative_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(build_root / paths["input"], noisy.astype(np.float32))
            np.save(build_root / paths["clean"], clean.astype(np.float32))
            np.save(build_root / paths["teacher"], cached_target)
            output.append({**metadata, **paths})
        pending.clear()

    for source in tqdm(sources, desc="Preparing UHD-LL/SNIC exact-192 samples"):
        dataset = str(source["dataset"])
        noisy_path = source_path(source_root, str(source["input"]))
        clean_path = source_path(source_root, str(source["clean"]))
        with ImageSource(noisy_path) as noisy_source, ImageSource(clean_path) as clean_source:
            if (noisy_source.width, noisy_source.height) != (clean_source.width, clean_source.height):
                raise ValueError(f"Geometry mismatch: {noisy_path} and {clean_path}")
            thumbnail_width = int(target_config["thumbnail_width"])
            noisy_thumbnail = noisy_source.thumbnail(thumbnail_width)
            clean_thumbnail = clean_source.thumbnail(thumbnail_width)
            gain_field = None
            if source.get("clean_transform") == UHD_CLEAN_TRANSFORM:
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
                int(
                    (
                        train_patch_counts
                        if source["split"] == "train"
                        else validation_patch_counts
                    )[dataset]
                ),
                candidate_count,
                crop_seed(source, seed),
            )
            source_key = hashlib.sha256(
                f"{dataset}:{source['input']}:{source['clean']}".encode()
            ).hexdigest()
            quality = int(jpeg_quality[dataset]) if dataset in jpeg_quality else None
            for index, (left, top, crop_luma) in enumerate(positions):
                noisy = noisy_source.crop(left, top)
                clean = clean_source.crop(left, top)
                record_id = f"{source_key[:16]}_{index:02d}"
                if gain_field is not None:
                    gain = sample_thumbnail_field(
                        gain_field,
                        left,
                        top,
                        noisy_source.width,
                        noisy_source.height,
                    )
                    clean = apply_local_gain(clean, gain)
                    gate = alignment_gate(noisy, clean, target_config)
                    gate.update(
                        {
                            "local_gain_minimum": float(gain.min()),
                            "local_gain_mean": gain.mean(axis=(0, 1)).tolist(),
                            "local_gain_maximum": float(gain.max()),
                        }
                    )
                    passed = bool(gate["passed"])
                    supervision = UHD_HYBRID_SUPERVISION if passed else "teacher_only"
                    gt_weight = 1.0 if passed else 0.0
                    kd_weight = 0.7 if passed else 1.0
                    clean_target = (
                        UHD_HYBRID_TARGET if passed else "local_ratio_candidate_ignored"
                    )
                    gate_rows.append(
                        {
                            "id": record_id,
                            "dataset": dataset,
                            "scene": source["scene"],
                            "split": source["split"],
                            "source_input": source["input"],
                            "crop": [left, top, TILE, TILE],
                            **gate,
                        }
                    )
                else:
                    gate = None
                    supervision = "paired"
                    gt_weight = 1.0
                    kd_weight = 0.7
                    clean_target = "native_paired_reference"
                if quality is not None:
                    noisy = jpeg_roundtrip(noisy, quality)
                    clean = jpeg_roundtrip(clean, quality)
                metadata = {
                    "id": record_id,
                    "dataset": dataset,
                    "scene": str(source["scene"]),
                    "split": str(source["split"]),
                    "source_input": str(source["input"]),
                    "source_clean": str(source["clean"]),
                    "crop": [left, top, TILE, TILE],
                    "crop_mean_luminance": crop_luma,
                    "supervision": supervision,
                    "gt_weight": gt_weight,
                    "kd_weight": kd_weight,
                    "jpeg_quality": quality,
                    "clean_target": clean_target,
                    "uhd_hybrid_gate": gate,
                    **{
                        key: source[key]
                        for key in (
                            "camera",
                            "iso",
                            "noise_level",
                            "clean_level",
                            "domain",
                            "license_status",
                            "source_url",
                        )
                        if key in source
                    },
                }
                pending.append((metadata, noisy, clean))
                if len(pending) >= batch_size:
                    flush()
    flush()

    output.sort(key=lambda row: (row["split"], row["dataset"], row["id"]))
    counts = collections.Counter(
        (row["split"], row["dataset"], row["supervision"]) for row in output
    )
    scene_splits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for row in output:
        scene_splits[(row["dataset"], row["scene"])].add(row["split"])
    leakage = [key for key, splits in scene_splits.items() if len(splits) != 1]
    if leakage:
        raise RuntimeError(f"Scene leakage detected: {leakage[0]}")
    payload = {
        "schema_version": 2,
        "preprocessing": config["project"]["preprocessing_version"],
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "uhd_hybrid_target": target_config,
        "records": output,
    }
    gate_report_path = output_manifest.parent / "uhd_hybrid_gate.json"
    gate_counts = collections.Counter(
        (row["split"], "passed" if row["passed"] else "failed") for row in gate_rows
    )
    gate_failures = collections.Counter(
        reason for row in gate_rows for reason in row["failure_reasons"]
    )
    gate_summary = {
        "schema_version": 2,
        "rows": gate_rows,
        "source_pairs": len(
            {(row["split"], row["source_input"]) for row in gate_rows}
        ),
        "crops": len(gate_rows),
        "passed": sum(bool(row["passed"]) for row in gate_rows),
        "failed": sum(not bool(row["passed"]) for row in gate_rows),
        "counts_by_split": {
            f"{split}/{state}": count
            for (split, state), count in sorted(gate_counts.items())
        },
        "failure_reasons": dict(sorted(gate_failures.items())),
        "target_parameters": target_config,
    }
    report = {
        "manifest": str(output_manifest),
        "records": len(output),
        "source_pairs": len(sources),
        "counts": {
            f"{split}/{dataset}/{supervision}": count
            for (split, dataset, supervision), count in sorted(counts.items())
        },
        "scene_groups": len(scene_splits),
        "scene_leakage": 0,
        "uhd_hybrid_gate": {
            "path": str(gate_report_path),
            "source_pairs": gate_summary["source_pairs"],
            "crops": gate_summary["crops"],
            "passed": gate_summary["passed"],
            "failed": gate_summary["failed"],
            "counts_by_split": gate_summary["counts_by_split"],
            "failure_reasons": gate_summary["failure_reasons"],
        },
        "teacher_inference_dtype": "float32",
        "input_cache_dtype": "float32",
        "clean_cache_dtype": "float32",
        "teacher_cache_dtype": "float16",
    }
    documents = (
        (output_manifest, payload),
        (gate_report_path, gate_summary),
        (output_manifest.with_suffix(".report.json"), report),
    )
    external_documents = []
    for final_path, value in documents:
        try:
            relative_path = final_path.relative_to(cache_root)
        except ValueError:
            external_documents.append((final_path, value))
        else:
            atomic_json(build_root / relative_path, value)

    if backup_root.exists():
        shutil.rmtree(backup_root)
    had_previous = cache_root.exists()
    if had_previous:
        cache_root.rename(backup_root)
    try:
        build_root.rename(cache_root)
    except Exception:
        if had_previous and backup_root.exists() and not cache_root.exists():
            backup_root.rename(cache_root)
        raise
    if backup_root.exists():
        shutil.rmtree(backup_root)
    for final_path, value in external_documents:
        atomic_json(final_path, value)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
