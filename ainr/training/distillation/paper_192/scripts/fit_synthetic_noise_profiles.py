#!/usr/bin/env python3
"""Fit train-only SNIC linear-noise and post-ISP residual profiles."""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import lsq_linear
from tqdm import tqdm

from common import atomic_json, load_config, resolve_paper_path, sha256_file
from prepare_domain_dataset import (
    ImageSource,
    linear_to_srgb,
    luminance,
    source_path,
    srgb_to_linear,
    stratified_positions,
)


TILE = 192
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
LUMA_NORM_SQUARED = float(np.dot(LUMA, LUMA))
BAND_NAMES = ("fine", "medium", "coarse", "very_coarse")
POST_AMPLITUDE_FIELDS = tuple(
    f"{band}_{component}_rms"
    for band in BAND_NAMES
    for component in ("luma", "chroma")
) + (
    "row_luma_rms",
    "row_chroma_rms",
    "column_luma_rms",
    "column_chroma_rms",
)


def stable_seed(*parts: object) -> int:
    encoded = ":".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(encoded).hexdigest()[:8], 16)


def load_holdout_exclusions(config: dict[str, Any]) -> list[dict[str, str]]:
    """Validate the immutable path/SHA denylist used by fitting and generation."""

    raw = config["data"].get("holdout_exclusions", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError("data.holdout_exclusions must be a non-empty list")
    output = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError(f"Invalid holdout exclusion at index {index}")
        path = Path(str(row["path"])).expanduser().resolve()
        digest = str(row["sha256"]).lower()
        if len(digest) != 64:
            raise ValueError(f"Invalid holdout SHA-256 at index {index}: {digest!r}")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(f"Non-hex holdout SHA-256 at index {index}") from error
        if str(path) in seen_paths or digest in seen_hashes:
            raise ValueError(f"Duplicate holdout exclusion at index {index}")
        seen_paths.add(str(path))
        seen_hashes.add(digest)
        if path.is_file():
            actual = sha256_file(path)
            if actual.lower() != digest:
                raise RuntimeError(
                    f"Holdout file changed: {path}; configured={digest}, actual={actual}"
                )
        output.append({"path": str(path), "sha256": digest})
    return output


def reject_holdout(path: Path, digest: str, exclusions: list[dict[str, str]]) -> None:
    resolved = str(path.resolve())
    for row in exclusions:
        if resolved == row["path"] or digest.lower() == row["sha256"]:
            raise RuntimeError(f"Final ISO holdout entered synthetic preparation: {path}")


def source_document(config: dict[str, Any]) -> tuple[Path, Path, list[dict[str, Any]]]:
    manifest = resolve_paper_path(config["data"]["source_manifest"])
    root = resolve_paper_path(config["data"]["source_root"])
    document = json.loads(manifest.read_text(encoding="utf-8"))
    records = document.get("records") if isinstance(document, dict) else None
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Invalid source manifest: {manifest}")
    return manifest, root, records


def validate_settings(config: dict[str, Any]) -> None:
    data = config["data"]
    calibration = config["calibration"]
    if int(data["tile_size"]) != TILE:
        raise ValueError(f"Synthetic pipeline requires tile_size={TILE}")
    observed = [int(value) for value in calibration["observed_isos"]]
    targets = [int(value) for value in data["severity_isos"]]
    if observed != sorted(set(observed)) or not observed:
        raise ValueError("calibration.observed_isos must be sorted and unique")
    if targets != sorted(set(targets)) or min(targets) < max(observed):
        raise ValueError("data.severity_isos must be sorted, unique, and start at observed maximum")
    bins = [float(value) for value in calibration["luminance_bins"]]
    if bins != sorted(set(bins)) or bins[0] != 0.0 or bins[-1] != 1.0:
        raise ValueError("calibration.luminance_bins must be unique and span [0,1]")
    if not 0.0 < float(calibration["low_gradient_quantile"]) < 1.0:
        raise ValueError("calibration.low_gradient_quantile must be in (0,1)")
    if not 0.0 < float(calibration["darkest_bin_read_fraction"]) <= 1.0:
        raise ValueError("calibration.darkest_bin_read_fraction must be in (0,1]")
    if float(calibration["absolute_read_variance_floor"]) <= 0.0:
        raise ValueError("calibration.absolute_read_variance_floor must be positive")
    if int(calibration["robust_fit_iterations"]) < 1:
        raise ValueError("calibration.robust_fit_iterations must be positive")
    sigmas = [
        float(calibration["fine_sigma"]),
        float(calibration["medium_sigma"]),
        float(calibration["coarse_sigma"]),
    ]
    if sigmas != sorted(sigmas) or sigmas[0] <= 0.0:
        raise ValueError("calibration band sigmas must be positive and sorted")
    if float(calibration["post_isp_bias_removal_sigma"]) <= sigmas[-1]:
        raise ValueError("post_isp_bias_removal_sigma must exceed the coarsest band sigma")
    subsampling = [int(value) for value in calibration["post_isp_jpeg_subsampling"]]
    if (
        not subsampling
        or len(subsampling) != len(set(subsampling))
        or any(value not in {0, 1, 2} for value in subsampling)
    ):
        raise ValueError(
            "calibration.post_isp_jpeg_subsampling must contain unique Pillow modes"
        )
    if subsampling != [int(value) for value in config["synthesis"]["jpeg_subsampling"]]:
        raise ValueError(
            "calibration.post_isp_jpeg_subsampling must match synthesis.jpeg_subsampling"
        )
    for key in ("variance_log2_slope_bounds", "rms_log2_slope_bounds"):
        bounds = list(map(float, calibration[key]))
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError(f"Invalid calibration.{key}")


def low_gradient_mask(clean_linear: np.ndarray, quantile: float) -> np.ndarray:
    gray = luminance(clean_linear)
    vertical, horizontal = np.gradient(gray)
    gradient = np.hypot(vertical, horizontal)
    valid = (gray > 0.002) & (gray < 0.98)
    if int(valid.sum()) < 512:
        return valid
    threshold = float(np.quantile(gradient[valid], quantile))
    return valid & (gradient <= threshold)


def robust_variance(value: np.ndarray) -> float:
    if value.size == 0:
        return 0.0
    centered = value.astype(np.float64) - np.median(value)
    sigma = 1.4826 * float(np.median(np.abs(centered)))
    return sigma * sigma


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, cumulative[-1] * 0.5)])


def fit_constrained_variance(
    observations: list[tuple[float, float, int, int]],
    config: dict[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    """Robustly fit variance=shot*signal+read with a measured dark floor."""

    if not observations:
        raise RuntimeError("No valid variance observations")
    signal = np.asarray([row[0] for row in observations], dtype=np.float64)
    variance = np.asarray([row[1] for row in observations], dtype=np.float64)
    counts = np.asarray([row[2] for row in observations], dtype=np.float64)
    bin_indices = np.asarray([row[3] for row in observations], dtype=np.int64)
    darkest_bin = int(bin_indices.min())
    darkest = bin_indices == darkest_bin
    darkest_variance = weighted_median(variance[darkest], counts[darkest])
    read_floor = max(
        float(config["absolute_read_variance_floor"]),
        darkest_variance * float(config["darkest_bin_read_fraction"]),
    )
    design = np.stack((signal, np.ones_like(signal)), axis=1)
    robust_weights = np.ones_like(signal)
    base_weights = counts / max(float(counts.mean()), 1.0)
    solution = np.asarray((0.0, read_floor), dtype=np.float64)
    iterations = int(config["robust_fit_iterations"])
    huber_delta = float(config["robust_fit_huber_delta"])
    for _ in range(iterations):
        weights = np.sqrt(np.maximum(base_weights * robust_weights, 1e-12))
        result = lsq_linear(
            design * weights[:, None],
            variance * weights,
            bounds=(np.asarray((0.0, read_floor)), np.asarray((np.inf, np.inf))),
            method="trf",
            lsmr_tol="auto",
        )
        if not result.success:
            raise RuntimeError(f"Constrained heteroscedastic fit failed: {result.message}")
        solution = result.x
        residual = variance - design @ solution
        center = np.median(residual)
        scale = 1.4826 * float(np.median(np.abs(residual - center)))
        if scale <= 1e-15:
            break
        normalized = np.abs(residual - center) / (huber_delta * scale)
        robust_weights = np.ones_like(normalized)
        outliers = normalized > 1.0
        robust_weights[outliers] = 1.0 / normalized[outliers]
    predicted = design @ solution
    residual = variance - predicted
    weighted_rmse = math.sqrt(float(np.average(np.square(residual), weights=counts)))
    diagnostics = {
        "method": "iteratively_reweighted_bounded_least_squares_huber",
        "observations": len(observations),
        "darkest_bin_index": darkest_bin,
        "darkest_bin_observations": int(darkest.sum()),
        "darkest_bin_signal_median": weighted_median(signal[darkest], counts[darkest]),
        "darkest_bin_fine_variance_median": darkest_variance,
        "darkest_bin_read_fraction": float(config["darkest_bin_read_fraction"]),
        "read_variance_lower_bound": read_floor,
        "read_bound_active": bool(abs(float(solution[1]) - read_floor) <= read_floor * 1e-5),
        "shot_scale": float(solution[0]),
        "read_variance": float(solution[1]),
        "weighted_rmse": weighted_rmse,
        "median_absolute_residual": float(np.median(np.abs(residual))),
        "robust_weight_minimum": float(robust_weights.min()),
        "robust_weight_median": float(np.median(robust_weights)),
        "iterations": iterations,
    }
    return float(solution[0]), float(solution[1]), diagnostics


def split_luma_chroma(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split RGB vectors using the true LUMA projection."""

    luma_scalar = np.einsum("...c,c->...", value, LUMA, optimize=True)
    luma_vector = luma_scalar[..., None] * (LUMA / LUMA_NORM_SQUARED)
    chroma = value - luma_vector
    return luma_scalar, chroma


def residual_bands(value: np.ndarray, sigmas: tuple[float, float, float]) -> dict[str, np.ndarray]:
    low_fine = gaussian_filter(value, sigma=(sigmas[0], sigmas[0], 0.0), mode="reflect")
    low_medium = gaussian_filter(value, sigma=(sigmas[1], sigmas[1], 0.0), mode="reflect")
    low_coarse = gaussian_filter(value, sigma=(sigmas[2], sigmas[2], 0.0), mode="reflect")
    return {
        "fine": value - low_fine,
        "medium": low_fine - low_medium,
        "coarse": low_medium - low_coarse,
        "very_coarse": low_coarse,
    }


def post_isp_jpeg_roundtrip(
    value: np.ndarray,
    quality: int,
    subsampling: int,
) -> np.ndarray:
    quantized = np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(quantized, "RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
        optimize=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255.0


def draw_isp_profile(
    rng: np.random.Generator,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Draw the shared deterministic ISP distribution used by fit and synthesis."""

    wb_low, wb_high = map(float, config["white_balance_range"])
    wb = rng.uniform(wb_low, wb_high, 3)
    jitter = float(config["color_matrix_jitter"])
    matrix = np.eye(3) + rng.normal(0.0, jitter, (3, 3))
    matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1e-6)
    gamma_low, gamma_high = map(float, config["tone_gamma_range"])
    amount_low, amount_high = map(float, config["sharpen_amount_range"])
    sigma_low, sigma_high = map(float, config["sharpen_sigma_range"])
    quality_low, quality_high = map(int, config["jpeg_quality_range"])
    return {
        "white_balance": wb.tolist(),
        "color_matrix": matrix.tolist(),
        "tone_gamma": float(rng.uniform(gamma_low, gamma_high)),
        "sharpen_amount": float(rng.uniform(amount_low, amount_high)),
        "sharpen_sigma": float(rng.uniform(sigma_low, sigma_high)),
        "jpeg_quality": int(rng.integers(quality_low, quality_high + 1)),
        "jpeg_subsampling": int(rng.choice(list(map(int, config["jpeg_subsampling"])))),
    }


def apply_isp(value: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    white_balance = np.asarray(profile["white_balance"], dtype=np.float32)
    matrix = np.asarray(profile["color_matrix"], dtype=np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("ISP color_matrix must be a finite 3x3 matrix")
    linear = value * white_balance
    linear = np.einsum("...c,dc->...d", linear, matrix, optimize=True)
    encoded = linear_to_srgb(np.clip(linear, 0.0, 1.0)).astype(np.float32)
    encoded = np.power(np.clip(encoded, 0.0, 1.0), float(profile["tone_gamma"]))
    amount = float(profile["sharpen_amount"])
    if amount > 0.0:
        sigma = float(profile["sharpen_sigma"])
        smooth = gaussian_filter(encoded, sigma=(sigma, sigma, 0.0), mode="reflect")
        encoded = encoded + amount * (encoded - smooth)
    return post_isp_jpeg_roundtrip(
        encoded,
        int(profile["jpeg_quality"]),
        int(profile["jpeg_subsampling"]),
    )


def add_energy(
    accumulator: dict[tuple[int, str, str, int, str], list[float]],
    iso: int,
    band: str,
    component: str,
    bin_index: int,
    value: np.ndarray,
    mask: np.ndarray,
) -> None:
    if not bool(mask.any()):
        return
    selected = value[mask]
    accumulator[(iso, band, component, bin_index, "sum_square")][0] += float(
        np.square(selected).sum()
    )
    accumulator[(iso, band, component, bin_index, "count")][0] += int(selected.size)


def energy_rms(
    accumulator: dict[tuple[int, str, str, int, str], list[float]],
    iso: int,
    band: str,
    component: str,
    bin_index: int,
) -> float | None:
    count = int(accumulator[(iso, band, component, bin_index, "count")][0])
    if count == 0:
        return None
    square = accumulator[(iso, band, component, bin_index, "sum_square")][0]
    return math.sqrt(max(0.0, square / count))


def fill_missing(values: list[float | None]) -> list[float]:
    valid = [index for index, value in enumerate(values) if value is not None]
    if not valid:
        raise RuntimeError("No post-ISP residual observations for a luminance profile")
    output = []
    for index, value in enumerate(values):
        if value is not None:
            output.append(float(value))
            continue
        nearest = min(valid, key=lambda candidate: abs(candidate - index))
        output.append(float(values[nearest]))
    return output


def positive_correlation(covariance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    correlation = covariance / np.outer(scale, scale)
    correlation = np.clip((correlation + correlation.T) * 0.5, -0.95, 0.95)
    np.fill_diagonal(correlation, 1.0)
    values, vectors = np.linalg.eigh(correlation)
    correlation = (vectors * np.maximum(values, 1e-4)) @ vectors.T
    scale = np.sqrt(np.diag(correlation))
    correlation /= np.outer(scale, scale)
    np.fill_diagonal(correlation, 1.0)
    return correlation.astype(np.float64)


def global_rgb_correlation_from_moments(
    value_sum: np.ndarray,
    value_cross: np.ndarray,
    count: int,
    *,
    iso: int,
) -> np.ndarray:
    """Return the global post-ISP residual correlation for one observed ISO."""

    if count < 3:
        raise RuntimeError(f"Insufficient post-ISP RGB covariance samples for ISO {iso}")
    mean = np.asarray(value_sum, dtype=np.float64) / count
    global_covariance = (
        np.asarray(value_cross, dtype=np.float64) / count - np.outer(mean, mean)
    )
    return positive_correlation(global_covariance)


def positive_covariance(covariance: np.ndarray) -> np.ndarray:
    """Return a finite positive-definite covariance without changing its scale."""

    value = np.asarray(covariance, dtype=np.float64)
    value = np.nan_to_num((value + value.T) * 0.5, nan=0.0, posinf=0.0, neginf=0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    floor = max(float(np.max(eigenvalues)) * 1e-6, 1e-12)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def covariance_correlation(covariance: np.ndarray) -> np.ndarray:
    value = positive_covariance(covariance)
    scale = np.sqrt(np.maximum(np.diag(value), 1e-12))
    correlation = value / np.outer(scale, scale)
    correlation = np.clip((correlation + correlation.T) * 0.5, -0.999, 0.999)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def scale_covariance_luma_chroma(
    covariance: np.ndarray,
    luma_scale: float,
    chroma_scale: float,
) -> np.ndarray:
    direction = LUMA.astype(np.float64) / math.sqrt(LUMA_NORM_SQUARED)
    luma_projection = np.outer(direction, direction)
    transform = (
        luma_scale * luma_projection
        + chroma_scale * (np.eye(3, dtype=np.float64) - luma_projection)
    )
    return positive_covariance(transform @ covariance @ transform.T)


def rms_from_targets(profile: dict[str, Any], band: str, component: str) -> float:
    values = np.asarray(profile["post_isp_band_targets"][band][f"{component}_rms"], dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values))))


def fit_log_slope(
    observed: dict[str, dict[str, Any]],
    field: str,
    index: int | None,
    bounds: tuple[float, float],
) -> float:
    points = []
    for iso_text, profile in sorted(observed.items(), key=lambda row: int(row[0])):
        raw = profile[field] if index is None else profile[field][index]
        value = float(raw)
        if value > 0.0:
            points.append((math.log2(int(iso_text)), math.log2(value)))
    if len(points) < 2:
        return min(max(0.5, bounds[0]), bounds[1])
    design = np.stack(
        (np.asarray([row[0] for row in points]), np.ones(len(points))), axis=1
    )
    slope = float(np.linalg.lstsq(design, [row[1] for row in points], rcond=None)[0][0])
    return min(max(slope, bounds[0]), bounds[1])


def extrapolate_profiles(
    observed: dict[str, dict[str, Any]], targets: list[int], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    calibration = config["calibration"]
    base_iso = max(map(int, observed))
    base = observed[str(base_iso)]
    variance_bounds = tuple(map(float, calibration["variance_log2_slope_bounds"]))
    rms_bounds = tuple(map(float, calibration["rms_log2_slope_bounds"]))
    variance_slopes = {
        field: [fit_log_slope(observed, field, channel, variance_bounds) for channel in range(3)]
        for field in ("shot_scale", "read_variance")
    }
    amplitude_slopes = {
        field: fit_log_slope(observed, field, None, rms_bounds)
        for field in POST_AMPLITUDE_FIELDS
    }
    output = {}
    for target_iso in targets:
        if target_iso == base_iso:
            profile = json.loads(json.dumps(base, allow_nan=False))
            profile.update({"source": "observed", "base_iso": base_iso})
        else:
            stops = math.log2(target_iso / base_iso)
            profile = {
                "iso": target_iso,
                "source": "train_only_snic_log2_extrapolation",
                "base_iso": base_iso,
                "extrapolation_stops": stops,
                "shot_scale": [
                    float(base["shot_scale"][channel])
                    * 2.0 ** (variance_slopes["shot_scale"][channel] * stops)
                    for channel in range(3)
                ],
                "read_variance": [
                    float(base["read_variance"][channel])
                    * 2.0 ** (variance_slopes["read_variance"][channel] * stops)
                    for channel in range(3)
                ],
                "variance_fit_diagnostics": base["variance_fit_diagnostics"],
                "rgb_correlation": base["rgb_correlation"],
                "shadow_multiplier": base["shadow_multiplier"],
                "medium_field_sigma": base["medium_field_sigma"],
                "coarse_field_sigma": base["coarse_field_sigma"],
                "row_column_smoothing_sigma": base["row_column_smoothing_sigma"],
                "structured_model": base["structured_model"],
                **{
                    field: float(base[field]) * 2.0 ** (amplitude_slopes[field] * stops)
                    for field in POST_AMPLITUDE_FIELDS
                },
            }
            targets_by_band = {
                "luminance_bins": base["post_isp_band_targets"]["luminance_bins"],
            }
            for band in BAND_NAMES:
                targets_by_band[band] = {}
                for component in ("luma", "chroma"):
                    field = f"{band}_{component}_rms"
                    factor = 2.0 ** (amplitude_slopes[field] * stops)
                    targets_by_band[band][f"{component}_rms"] = [
                        float(value) * factor
                        for value in base["post_isp_band_targets"][band][f"{component}_rms"]
                    ]
                luma_factor = 2.0 ** (
                    amplitude_slopes[f"{band}_luma_rms"] * stops
                )
                chroma_factor = 2.0 ** (
                    amplitude_slopes[f"{band}_chroma_rms"] * stops
                )
                covariances = [
                    scale_covariance_luma_chroma(
                        np.asarray(value, dtype=np.float64),
                        luma_factor,
                        chroma_factor,
                    )
                    for value in base["post_isp_band_targets"][band]["rgb_covariance"]
                ]
                targets_by_band[band]["rgb_covariance"] = [
                    value.tolist() for value in covariances
                ]
                targets_by_band[band]["rgb_correlation"] = [
                    covariance_correlation(value).tolist() for value in covariances
                ]
            profile["post_isp_band_targets"] = targets_by_band
        profile["fit_slopes"] = {
            "shot_scale": variance_slopes["shot_scale"],
            "read_variance": variance_slopes["read_variance"],
            **amplitude_slopes,
        }
        output[str(target_iso)] = profile
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs/synthetic_camera_jpeg_gate.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--limit-pairs-per-iso",
        type=int,
        help="Smoke-test limit per observed ISO; never use for release profiles.",
    )
    parser.add_argument("--profiles", help="Comma-separated target ISO severities.")
    args = parser.parse_args()
    if args.limit_pairs_per_iso is not None and args.limit_pairs_per_iso < 1:
        parser.error("--limit-pairs-per-iso must be positive")

    config = load_config(args.config)
    validate_settings(config)
    exclusions = load_holdout_exclusions(config)
    manifest_path, source_root, raw_records = source_document(config)
    calibration = config["calibration"]
    observed_isos = {int(value) for value in calibration["observed_isos"]}
    selected = [
        row
        for row in raw_records
        if row.get("dataset") == calibration["dataset"]
        and row.get("split") == calibration["split"]
        and int(row.get("iso") or -1) in observed_isos
    ]
    selected.sort(key=lambda row: (int(row["iso"]), str(row["scene"]), str(row["input"])))
    if args.limit_pairs_per_iso is not None:
        limited = []
        counts: collections.Counter[int] = collections.Counter()
        for row in selected:
            iso = int(row["iso"])
            if counts[iso] < args.limit_pairs_per_iso:
                limited.append(row)
                counts[iso] += 1
        selected = limited
    found_isos = {int(row["iso"]) for row in selected}
    if found_isos != observed_isos:
        raise RuntimeError(f"Calibration records do not cover every observed ISO: {found_isos}")

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else resolve_paper_path(config["data"]["calibration_profile"])
    )
    if output.exists() and not args.replace:
        raise FileExistsError(f"Calibration profile exists: {output}; pass --replace")
    target_isos = (
        [int(value) for value in args.profiles.split(",")]
        if args.profiles
        else [int(value) for value in config["data"]["severity_isos"]]
    )
    if target_isos != sorted(set(target_isos)) or min(target_isos) < max(observed_isos):
        parser.error("--profiles must be sorted, unique, and at least the maximum observed ISO")

    hash_cache: dict[Path, str] = {}

    def checked_hash(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in hash_cache:
            hash_cache[resolved] = sha256_file(resolved)
        digest = hash_cache[resolved]
        reject_holdout(resolved, digest, exclusions)
        return digest

    variance_rows: dict[
        tuple[int, int], list[tuple[float, float, int, int]]
    ] = collections.defaultdict(list)
    post_correlation_sum = collections.defaultdict(lambda: np.zeros(3, dtype=np.float64))
    post_correlation_cross = collections.defaultdict(lambda: np.zeros((3, 3), dtype=np.float64))
    post_correlation_count: collections.Counter[int] = collections.Counter()
    post_energy = collections.defaultdict(lambda: [0.0])
    post_band_sum = collections.defaultdict(lambda: np.zeros(3, dtype=np.float64))
    post_band_cross = collections.defaultdict(lambda: np.zeros((3, 3), dtype=np.float64))
    post_band_count: collections.Counter[tuple[int, str, int]] = collections.Counter()
    line_energy = collections.defaultdict(lambda: [0.0])
    provenance = []
    seed = int(config["project"]["seed"])
    patches_per_pair = int(calibration["patches_per_pair"])
    crop_candidates = int(calibration["crop_candidates"])
    gradient_quantile = float(calibration["low_gradient_quantile"])
    bins = np.asarray(calibration["luminance_bins"], dtype=np.float32)
    minimum_pixels = int(calibration["minimum_pixels_per_bin"])
    sigmas = (
        float(calibration["fine_sigma"]),
        float(calibration["medium_sigma"]),
        float(calibration["coarse_sigma"]),
    )
    jpeg_subsampling = [int(value) for value in calibration["post_isp_jpeg_subsampling"]]
    bias_sigma = float(calibration["post_isp_bias_removal_sigma"])
    line_sigma = float(calibration["row_column_smoothing_sigma"])

    for row in tqdm(selected, desc="Fitting SNIC train-only noise profiles"):
        noisy_path = source_path(source_root, str(row["input"]))
        clean_path = source_path(source_root, str(row["clean"]))
        noisy_sha = checked_hash(noisy_path)
        clean_sha = checked_hash(clean_path)
        iso = int(row["iso"])
        with ImageSource(noisy_path) as noisy_source, ImageSource(clean_path) as clean_source:
            if (noisy_source.width, noisy_source.height) != (clean_source.width, clean_source.height):
                raise ValueError(f"Calibration geometry mismatch: {noisy_path}")
            clean_thumbnail = clean_source.thumbnail(480)
            positions = stratified_positions(
                clean_thumbnail,
                clean_source.width,
                clean_source.height,
                patches_per_pair,
                crop_candidates,
                stable_seed(seed, row["dataset"], row["scene"], row["clean"]),
            )
            for left, top, _ in positions:
                clean_srgb = clean_source.crop(left, top)
                noisy_srgb = noisy_source.crop(left, top)
                clean_linear = srgb_to_linear(clean_srgb).astype(np.float32)
                noisy_linear = srgb_to_linear(noisy_srgb).astype(np.float32)
                linear_residual = noisy_linear - clean_linear
                linear_residual -= np.median(linear_residual, axis=(0, 1), keepdims=True)
                mask = low_gradient_mask(clean_linear, gradient_quantile)
                if int(mask.sum()) < minimum_pixels:
                    continue
                linear_bands = residual_bands(linear_residual, sigmas)
                clean_linear_luma = luminance(clean_linear)
                for channel in range(3):
                    for bin_index, (low, high) in enumerate(
                        zip(bins[:-1], bins[1:], strict=True)
                    ):
                        selected_mask = mask & (clean_linear_luma >= low) & (clean_linear_luma < high)
                        count = int(selected_mask.sum())
                        if count < minimum_pixels:
                            continue
                        variance_rows[(iso, channel)].append(
                            (
                                float(clean_linear[..., channel][selected_mask].mean()),
                                robust_variance(linear_bands["fine"][..., channel][selected_mask]),
                                count,
                                bin_index,
                            )
                        )

                for subsampling in jpeg_subsampling:
                    calibration_isp_seed = stable_seed(
                        seed,
                        row["dataset"],
                        row["scene"],
                        row["clean"],
                        left,
                        top,
                        "post_isp",
                        subsampling,
                    )
                    calibration_isp = draw_isp_profile(
                        np.random.default_rng(calibration_isp_seed),
                        config["synthesis"],
                    )
                    calibration_isp["jpeg_subsampling"] = subsampling
                    clean_post = apply_isp(clean_linear, calibration_isp)
                    noisy_post = apply_isp(noisy_linear, calibration_isp)
                    post_residual = noisy_post - clean_post
                    post_residual -= gaussian_filter(
                        post_residual,
                        sigma=(bias_sigma, bias_sigma, 0.0),
                        mode="reflect",
                    )
                    post_bands = residual_bands(post_residual, sigmas)
                    post_luma = luminance(clean_post)
                    post_mask = low_gradient_mask(
                        srgb_to_linear(clean_post), gradient_quantile
                    )
                    post_values = post_residual[post_mask].astype(np.float64)
                    post_correlation_sum[iso] += post_values.sum(axis=0)
                    post_correlation_cross[iso] += post_values.T @ post_values
                    post_correlation_count[iso] += len(post_values)
                    for band_name, band in post_bands.items():
                        luma_scalar, chroma_vector = split_luma_chroma(band)
                        for bin_index, (low, high) in enumerate(
                            zip(bins[:-1], bins[1:], strict=True)
                        ):
                            selected_mask = (
                                post_mask & (post_luma >= low) & (post_luma < high)
                            )
                            if int(selected_mask.sum()) < minimum_pixels:
                                continue
                            add_energy(
                                post_energy,
                                iso,
                                band_name,
                                "luma",
                                bin_index,
                                luma_scalar,
                                selected_mask,
                            )
                            add_energy(
                                post_energy,
                                iso,
                                band_name,
                                "chroma",
                                bin_index,
                                chroma_vector,
                                selected_mask,
                            )
                            selected_band = band[selected_mask].astype(np.float64)
                            covariance_key = (iso, band_name, bin_index)
                            post_band_sum[covariance_key] += selected_band.sum(axis=0)
                            post_band_cross[covariance_key] += selected_band.T @ selected_band
                            post_band_count[covariance_key] += int(selected_band.shape[0])

                    low = post_bands["very_coarse"]
                    row_profile = gaussian_filter1d(
                        low.mean(axis=1), line_sigma, axis=0, mode="reflect"
                    )
                    column_profile = gaussian_filter1d(
                        low.mean(axis=0), line_sigma, axis=0, mode="reflect"
                    )
                    row_profile -= row_profile.mean(axis=0, keepdims=True)
                    column_profile -= column_profile.mean(axis=0, keepdims=True)
                    for direction, profile in (
                        ("row", row_profile),
                        ("column", column_profile),
                    ):
                        luma_scalar, chroma_vector = split_luma_chroma(profile)
                        line_energy[(iso, direction, "luma", "sum_square")][0] += float(
                            np.square(luma_scalar).sum()
                        )
                        line_energy[(iso, direction, "luma", "count")][0] += int(
                            luma_scalar.size
                        )
                        line_energy[(iso, direction, "chroma", "sum_square")][0] += float(
                            np.square(chroma_vector).sum()
                        )
                        line_energy[(iso, direction, "chroma", "count")][0] += int(
                            chroma_vector.size
                        )
        provenance.append(
            {
                "dataset": row["dataset"],
                "scene": row["scene"],
                "split": row["split"],
                "iso": iso,
                "input": row["input"],
                "input_sha256": noisy_sha,
                "clean": row["clean"],
                "clean_sha256": clean_sha,
            }
        )

    observed_profiles: dict[str, dict[str, Any]] = {}
    bin_count = len(bins) - 1
    shadow_bounds = tuple(map(float, calibration["shadow_multiplier_bounds"]))
    for iso in sorted(observed_isos):
        shot_scale, read_variance, diagnostics = [], [], []
        for channel in range(3):
            shot, read, diagnostic = fit_constrained_variance(
                variance_rows[(iso, channel)], calibration
            )
            shot_scale.append(shot)
            read_variance.append(read)
            diagnostics.append(diagnostic)
        count = post_correlation_count[iso]
        global_rgb_correlation = global_rgb_correlation_from_moments(
            post_correlation_sum[iso],
            post_correlation_cross[iso],
            count,
            iso=iso,
        )
        target_document: dict[str, Any] = {"luminance_bins": bins.tolist()}
        global_fields: dict[str, float] = {}
        for band in BAND_NAMES:
            target_document[band] = {}
            for component in ("luma", "chroma"):
                values = fill_missing(
                    [energy_rms(post_energy, iso, band, component, index) for index in range(bin_count)]
                )
                target_document[band][f"{component}_rms"] = values
                global_fields[f"{band}_{component}_rms"] = float(
                    np.sqrt(np.mean(np.square(values)))
                )
            covariances: list[np.ndarray | None] = []
            for bin_index in range(bin_count):
                covariance_key = (iso, band, bin_index)
                count_for_bin = post_band_count[covariance_key]
                if count_for_bin == 0:
                    covariances.append(None)
                    continue
                mean_for_bin = post_band_sum[covariance_key] / count_for_bin
                bin_covariance = (
                    post_band_cross[covariance_key] / count_for_bin
                    - np.outer(mean_for_bin, mean_for_bin)
                )
                covariances.append(positive_covariance(bin_covariance))
            valid_covariances = [
                index for index, covariance in enumerate(covariances) if covariance is not None
            ]
            if not valid_covariances:
                raise RuntimeError(f"No post-ISP covariance observations for ISO {iso}/{band}")
            filled_covariances = [
                covariance
                if covariance is not None
                else covariances[
                    min(valid_covariances, key=lambda candidate: abs(candidate - index))
                ]
                for index, covariance in enumerate(covariances)
            ]
            target_document[band]["rgb_covariance"] = [
                np.asarray(value).tolist() for value in filled_covariances
            ]
            target_document[band]["rgb_correlation"] = [
                covariance_correlation(np.asarray(value)).tolist()
                for value in filled_covariances
            ]
        line_fields = {}
        for direction in ("row", "column"):
            for component in ("luma", "chroma"):
                square = line_energy[(iso, direction, component, "sum_square")][0]
                line_count = int(line_energy[(iso, direction, component, "count")][0])
                line_fields[f"{direction}_{component}_rms"] = math.sqrt(
                    max(0.0, square / max(line_count, 1))
                )
        shadow_energy = sum(
            target_document[band][f"{component}_rms"][0] ** 2
            for band in ("medium", "coarse")
            for component in ("luma", "chroma")
        )
        mid_index = min(bin_count - 1, max(1, bin_count // 2))
        mid_energy = sum(
            target_document[band][f"{component}_rms"][mid_index] ** 2
            for band in ("medium", "coarse")
            for component in ("luma", "chroma")
        )
        shadow_multiplier = math.sqrt(shadow_energy / max(mid_energy, 1e-15))
        shadow_multiplier = min(max(shadow_multiplier, shadow_bounds[0]), shadow_bounds[1])
        observed_profiles[str(iso)] = {
            "iso": iso,
            "source": "observed_snic_train_post_isp",
            "shot_scale": shot_scale,
            "read_variance": read_variance,
            "variance_fit_diagnostics": diagnostics,
            "rgb_correlation": global_rgb_correlation.tolist(),
            "post_isp_band_targets": target_document,
            **global_fields,
            **line_fields,
            "shadow_multiplier": shadow_multiplier,
            "medium_field_sigma": sigmas[1],
            "coarse_field_sigma": sigmas[2],
            "row_column_smoothing_sigma": line_sigma,
            "structured_model": (
                "post_isp_luminance_band_matching_from_linear_heteroscedastic_residual;"
                "no_independent_medium_or_coarse_color_fields"
            ),
            "fit_pairs": sum(int(row["iso"]) == iso for row in selected),
            "fine_variance_observations": [
                len(variance_rows[(iso, channel)]) for channel in range(3)
            ],
            "covariance_pixels": count,
        }

    profiles = extrapolate_profiles(observed_profiles, target_isos, config)
    config_path = Path(config["_config_path"])
    calibration_basis = {
        "source": "SNIC Sony A7R III paired train split only",
        "linear_variance_domain": "decoded_sRGB_inverse_transfer_float32",
        "linear_variance_model": "variance=shot_scale*signal+read_variance",
        "linear_variance_fit": (
            "Huber IRLS bounded least squares with explicit darkest-bin read floor"
        ),
        "post_isp_target_domain": (
            "decoded_sRGB through the synthesis ISP distribution: "
            f"JPEG quality {config['synthesis']['jpeg_quality_range']}, "
            f"subsampling {jpeg_subsampling}, shared color/tone/sharpen profile"
        ),
        "post_isp_jpeg_subsampling": jpeg_subsampling,
        "post_isp_distribution": {
            key: config["synthesis"][key]
            for key in (
                "white_balance_range",
                "color_matrix_jitter",
                "tone_gamma_range",
                "sharpen_amount_range",
                "sharpen_sigma_range",
                "jpeg_quality_range",
                "jpeg_subsampling",
            )
        },
        "post_isp_band_sigmas": list(sigmas),
        "post_isp_bias_removal_sigma": bias_sigma,
        "post_isp_luminance_bins": bins.tolist(),
        "chroma_definition": "RGB residual orthogonal to BT.709 LUMA vector projection",
        "rgb_covariance_definition": (
            "centered per-band RGB covariance within each post-ISP luminance bin"
        ),
        "row_column_smoothing_sigma": line_sigma,
        "fit_split": "train",
        "target_camera_holdout_used": False,
        "fit_support": {
            "pairs": len(provenance),
            "pairs_by_iso": {
                str(iso): sum(int(row["iso"]) == iso for row in provenance)
                for iso in sorted(observed_isos)
            },
            "unique_scenes": len({str(row["scene"]) for row in provenance}),
            "unique_clean_payloads": len(
                {str(row["clean_sha256"]) for row in provenance}
            ),
        },
    }
    payload = {
        "schema_version": 2,
        "profile_version": "snic_train_linear_post_isp_covariance_v3",
        "preprocessing": config["project"]["preprocessing_version"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "fit_dataset": calibration["dataset"],
        "fit_split": calibration["split"],
        "fit_is_limited_smoke": args.limit_pairs_per_iso is not None,
        "holdout_exclusions": exclusions,
        "calibration_basis": calibration_basis,
        "calibration_settings": calibration,
        "fitting_environment": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "seed": int(config["project"]["seed"]),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "source_records": provenance,
        "observed_profiles": observed_profiles,
        "profiles": profiles,
    }
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "source_records": len(provenance),
                "clean_sources": len({row["clean_sha256"] for row in provenance}),
                "observed_isos": sorted(observed_isos),
                "profiles": target_isos,
                "limited_smoke": args.limit_pairs_per_iso is not None,
                "minimum_read_variance": min(
                    min(profile["read_variance"])
                    for profile in observed_profiles.values()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
