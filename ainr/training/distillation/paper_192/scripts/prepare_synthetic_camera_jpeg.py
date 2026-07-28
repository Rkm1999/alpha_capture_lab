#!/usr/bin/env python3
"""Build deterministic exact-192 synthetic high-ISO camera-JPEG samples."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from common import atomic_json, load_config, resolve_paper_path, sha256_file
from fit_synthetic_noise_profiles import (
    apply_isp,
    covariance_correlation,
    draw_isp_profile,
    load_holdout_exclusions,
    reject_holdout,
    source_document,
    stable_seed,
)
from prepare_domain_dataset import (
    ImageSource,
    luminance,
    source_path,
    srgb_to_linear,
    stratified_positions,
)
from src.scunet_teacher import load_scunet_teacher


TILE = 192
DATASET = "synthetic_camera_jpeg"
ARRAY_FIELDS = ("input", "clean", "teacher")
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
LUMA_NORM_SQUARED = float(np.dot(LUMA, LUMA))
BAND_NAMES = ("fine", "medium", "coarse", "very_coarse")
PROFILE_VERSION = "snic_train_linear_post_isp_covariance_v3"
FITTER_SCRIPT = Path(__file__).with_name("fit_synthetic_noise_profiles.py").resolve()


def validate_settings(config: dict[str, Any]) -> None:
    data = config["data"]
    synthesis = config["synthesis"]
    if int(data["tile_size"]) != TILE:
        raise ValueError(f"Synthetic pipeline requires tile_size={TILE}")
    datasets = [str(value) for value in data["datasets"]]
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("data.datasets must be non-empty and unique")
    for dataset in datasets:
        if int(data["patches_per_clean_source"].get(dataset, 0)) < 1:
            raise ValueError(f"Missing positive patch count for {dataset}")
    severities = [int(value) for value in data["severity_isos"]]
    if severities != sorted(set(severities)) or not severities:
        raise ValueError("data.severity_isos must be sorted and unique")
    if int(data["replicates_per_severity"]) < 1 or int(data["crop_candidates"]) < 1:
        raise ValueError("replicate and crop candidate counts must be positive")
    range_keys = (
        "white_balance_range",
        "tone_gamma_range",
        "sharpen_amount_range",
        "sharpen_sigma_range",
        "jpeg_quality_range",
    )
    for key in range_keys:
        values = list(map(float, synthesis[key]))
        if len(values) != 2 or values[0] > values[1]:
            raise ValueError(f"Invalid synthesis.{key}")
    subsampling = [int(value) for value in synthesis["jpeg_subsampling"]]
    if not subsampling or any(value not in {0, 1, 2} for value in subsampling):
        raise ValueError("synthesis.jpeg_subsampling must contain Pillow modes 0, 1, or 2")
    if float(synthesis["shadow_exponent"]) <= 0.0:
        raise ValueError("synthesis.shadow_exponent must be positive")
    gain_bounds = list(map(float, synthesis["post_isp_gain_bounds"]))
    if len(gain_bounds) != 2 or gain_bounds[0] <= 0.0 or gain_bounds[0] > gain_bounds[1]:
        raise ValueError("synthesis.post_isp_gain_bounds must be positive and sorted")
    if float(synthesis["post_isp_gain_smoothing_sigma"]) < 0.0:
        raise ValueError("synthesis.post_isp_gain_smoothing_sigma must be non-negative")
    if int(synthesis["post_isp_match_iterations"]) < 1:
        raise ValueError("synthesis.post_isp_match_iterations must be positive")
    if not 0.0 <= float(synthesis["row_column_component_scale"]) <= 1.0:
        raise ValueError("synthesis.row_column_component_scale must be in [0,1]")
    supervision = config["supervision"]
    for name, value in {
        "default": supervision["default"],
        **supervision.get("by_source_dataset", {}),
    }.items():
        gt = float(value["gt_weight"])
        kd = float(value["kd_weight"])
        if not 0.0 <= gt <= 1.0 or not 0.0 <= kd <= 1.0 or gt + kd <= 0.0:
            raise ValueError(f"Invalid supervision weights for {name}")


def _profile_matrix(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {label}: expected finite 3x3 matrix")
    return result


def validate_severity_profile(profile: dict[str, Any], expected_iso: int | None = None) -> None:
    required_vectors = ("shot_scale", "read_variance")
    for key in required_vectors:
        value = np.asarray(profile.get(key), dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all() or bool((value < 0.0).any()):
            raise ValueError(f"Invalid severity profile {key}")
    correlation = _profile_matrix(profile.get("rgb_correlation"), "rgb_correlation")
    if not np.allclose(np.diag(correlation), 1.0, atol=1e-5):
        raise ValueError("rgb_correlation must have a unit diagonal")
    for key in (
        "medium_luma_rms",
        "medium_chroma_rms",
        "coarse_luma_rms",
        "coarse_chroma_rms",
        "row_luma_rms",
        "row_chroma_rms",
        "column_luma_rms",
        "column_chroma_rms",
        "medium_field_sigma",
        "coarse_field_sigma",
        "row_column_smoothing_sigma",
        "shadow_multiplier",
    ):
        value = float(profile.get(key, math.nan))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Invalid severity profile {key}: {value}")
    if expected_iso is not None and int(profile.get("iso", -1)) != expected_iso:
        raise ValueError(f"Severity profile ISO mismatch: expected {expected_iso}")
    targets = profile.get("post_isp_band_targets")
    if not isinstance(targets, dict):
        raise ValueError("Severity profile has no post_isp_band_targets")
    bins = np.asarray(targets.get("luminance_bins"), dtype=np.float64)
    if bins.ndim != 1 or len(bins) < 2 or not np.all(np.diff(bins) > 0.0):
        raise ValueError("Invalid post-ISP luminance bins")
    if bins[0] != 0.0 or bins[-1] != 1.0:
        raise ValueError("Post-ISP luminance bins must span [0,1]")
    for band in BAND_NAMES:
        if not isinstance(targets.get(band), dict):
            raise ValueError(f"Missing post-ISP target band {band}")
        for component in ("luma", "chroma"):
            values = np.asarray(targets[band].get(f"{component}_rms"), dtype=np.float64)
            if (
                values.shape != (len(bins) - 1,)
                or not np.isfinite(values).all()
                or bool((values < 0.0).any())
            ):
                raise ValueError(f"Invalid post-ISP {band}/{component} targets")
        covariance = np.asarray(targets[band].get("rgb_covariance"), dtype=np.float64)
        correlation = np.asarray(targets[band].get("rgb_correlation"), dtype=np.float64)
        expected_matrix_shape = (len(bins) - 1, 3, 3)
        if (
            covariance.shape != expected_matrix_shape
            or not np.isfinite(covariance).all()
            or correlation.shape != expected_matrix_shape
            or not np.isfinite(correlation).all()
        ):
            raise ValueError(f"Invalid post-ISP {band} RGB covariance targets")
        if bool((np.linalg.eigvalsh(covariance) < -1e-10).any()):
            raise ValueError(f"Post-ISP {band} RGB covariance is not positive semidefinite")
        if not np.allclose(np.diagonal(correlation, axis1=1, axis2=2), 1.0, atol=1e-5):
            raise ValueError(f"Post-ISP {band} RGB correlation must have a unit diagonal")


def load_profiles(
    config: dict[str, Any],
    requested: list[int],
    exclusions: list[dict[str, str]],
    allow_smoke_profile: bool,
    profile_override: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[int, dict[str, Any]]]:
    path = (
        profile_override.expanduser().resolve()
        if profile_override is not None
        else resolve_paper_path(config["data"]["calibration_profile"])
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 2:
        raise ValueError(f"Invalid calibration profile: {path}")
    if document.get("profile_version") != PROFILE_VERSION:
        raise ValueError(
            f"Calibration profile version must be {PROFILE_VERSION}: {path}"
        )
    if document.get("fit_split") != "train":
        raise RuntimeError("Synthetic calibration profile was not fitted on train-only data")
    if document.get("fit_is_limited_smoke") and not allow_smoke_profile:
        raise RuntimeError("Limited smoke calibration cannot build a release cache")
    if document.get("preprocessing") != config["project"]["preprocessing_version"]:
        raise RuntimeError("Calibration preprocessing does not match generator config")
    fitting_environment = document.get("fitting_environment")
    active_fitter_sha = sha256_file(FITTER_SCRIPT)
    if (
        not isinstance(fitting_environment, dict)
        or fitting_environment.get("script_sha256") != active_fitter_sha
    ):
        raise RuntimeError(
            "Calibration profile was not produced by the active fitter: "
            f"profile={None if not isinstance(fitting_environment, dict) else fitting_environment.get('script_sha256')!r}, "
            f"active={active_fitter_sha}"
        )
    config_path = Path(config["_config_path"])
    if document.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("Calibration profile was fitted with a different configuration")
    source_manifest = resolve_paper_path(config["data"]["source_manifest"])
    if document.get("source_manifest_sha256") != sha256_file(source_manifest):
        raise RuntimeError("Calibration profile source manifest has changed")
    if document.get("holdout_exclusions") != exclusions:
        raise RuntimeError("Calibration and generation holdout exclusions differ")
    basis = document.get("calibration_basis")
    if not isinstance(basis, dict) or basis.get("fit_split") != "train":
        raise RuntimeError("Calibration profile has no train-only basis provenance")
    raw_profiles = document.get("profiles")
    if not isinstance(raw_profiles, dict):
        raise ValueError("Calibration profile has no profiles map")
    profiles = {}
    for iso in requested:
        value = raw_profiles.get(str(iso))
        if not isinstance(value, dict):
            raise ValueError(f"Calibration has no ISO {iso} profile")
        validate_severity_profile(value, iso)
        profiles[iso] = value
    return path, document, profiles


def select_clean_sources(
    records: list[dict[str, Any]],
    source_root: Path,
    allowed_datasets: set[str],
    holdout_entries: list[dict[str, str]],
    *,
    limit_per_dataset: int | None = None,
) -> list[dict[str, Any]]:
    """Deduplicate full-size clean sources and enforce SHA/scene split isolation."""

    unique_keys: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    scene_splits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for index, row in enumerate(records):
        if row.get("dataset") not in allowed_datasets:
            continue
        missing = {key for key in ("dataset", "scene", "split", "clean") if key not in row}
        if missing:
            raise ValueError(f"Clean source record {index} is missing {sorted(missing)}")
        split = str(row["split"])
        if split not in {"train", "validation"}:
            raise ValueError(f"Invalid source split {split!r}")
        key = (str(row["dataset"]), str(row["scene"]), str(row["clean"]), split)
        unique_keys.setdefault(key, row)
        scene_splits[(str(row["dataset"]), str(row["scene"]))].add(split)
    leakage = [key for key, splits in scene_splits.items() if len(splits) != 1]
    if leakage:
        raise RuntimeError(f"Source scene spans train and validation: {leakage[0]}")
    candidates = [unique_keys[key] for key in sorted(unique_keys)]
    if limit_per_dataset is not None:
        limited = []
        counts: collections.Counter[tuple[str, str]] = collections.Counter()
        for row in candidates:
            bucket = (str(row["dataset"]), str(row["split"]))
            if counts[bucket] < limit_per_dataset:
                limited.append(row)
                counts[bucket] += 1
        candidates = limited

    path_hashes: dict[Path, str] = {}
    canonical_by_sha: dict[tuple[str, str], dict[str, Any]] = {}
    sha_splits: dict[str, set[str]] = collections.defaultdict(set)
    for row in tqdm(candidates, desc="Auditing unique clean-source payloads"):
        path = source_path(source_root, str(row["clean"]))
        if path not in path_hashes:
            path_hashes[path] = sha256_file(path)
        digest = path_hashes[path]
        reject_holdout(path, digest, holdout_entries)
        split = str(row["split"])
        sha_splits[digest].add(split)
        enriched = dict(row)
        enriched["_clean_path"] = path
        enriched["_clean_sha256"] = digest
        canonical_by_sha.setdefault((split, digest), enriched)
    payload_leakage = [digest for digest, splits in sha_splits.items() if len(splits) != 1]
    if payload_leakage:
        raise RuntimeError(f"Clean payload SHA spans train and validation: {payload_leakage[0]}")
    output = list(canonical_by_sha.values())
    output.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["dataset"]),
            str(row["scene"]),
            str(row["clean"]),
        )
    )
    return output


def split_luma_chroma(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    luma_scalar = np.einsum("...c,c->...", value, LUMA, optimize=True)
    luma_vector = luma_scalar[..., None] * (LUMA / LUMA_NORM_SQUARED)
    return luma_scalar, value - luma_vector


def luma_vector(value: np.ndarray) -> np.ndarray:
    return value[..., None] * (LUMA / LUMA_NORM_SQUARED)


def residual_bands(value: np.ndarray, profile: dict[str, Any]) -> dict[str, np.ndarray]:
    sigmas = (
        1.0,
        float(profile["medium_field_sigma"]),
        float(profile["coarse_field_sigma"]),
    )
    low_fine = gaussian_filter(value, sigma=(sigmas[0], sigmas[0], 0.0), mode="reflect")
    low_medium = gaussian_filter(value, sigma=(sigmas[1], sigmas[1], 0.0), mode="reflect")
    low_coarse = gaussian_filter(value, sigma=(sigmas[2], sigmas[2], 0.0), mode="reflect")
    return {
        "fine": value - low_fine,
        "medium": low_fine - low_medium,
        "coarse": low_medium - low_coarse,
        "very_coarse": low_coarse,
    }


def line_field(
    height: int,
    width: int,
    axis: int,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate calibrated 1D-smoothed luma and LUMA-orthogonal chroma lines."""

    length = height if axis == 0 else width
    raw = rng.normal(0.0, 1.0, (length, 3)).astype(np.float32)
    smooth = gaussian_filter(raw, sigma=(sigma, 0.0), mode="reflect")
    smooth -= smooth.mean(axis=0, keepdims=True)
    scalar, chroma = split_luma_chroma(smooth)
    scalar /= max(float(np.sqrt(np.mean(np.square(scalar)))), 1e-7)
    chroma /= max(float(np.sqrt(np.mean(np.square(chroma)))), 1e-7)
    luma_rgb = luma_vector(scalar)
    if axis == 0:
        return (
            np.repeat(luma_rgb[:, None, :], width, axis=1),
            np.repeat(chroma[:, None, :], width, axis=1),
        )
    return (
        np.repeat(luma_rgb[None, :, :], height, axis=0),
        np.repeat(chroma[None, :, :], height, axis=0),
    )


def component_gain_map(
    component: np.ndarray,
    clean_luma: np.ndarray,
    bins: np.ndarray,
    targets: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    if component.ndim == 2:
        energy = np.square(component)
    else:
        energy = np.mean(np.square(component), axis=2)
    low_gain, high_gain = map(float, config["post_isp_gain_bounds"])
    gain = np.ones_like(clean_luma, dtype=np.float32)
    diagnostics = []
    for index, (low, high) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
        mask = (clean_luma >= low) & (clean_luma < high)
        actual = math.sqrt(float(energy[mask].mean())) if bool(mask.any()) else 0.0
        requested = float(targets[index])
        raw_gain = requested / max(actual, 1e-8)
        bounded = min(max(raw_gain, low_gain), high_gain)
        gain[mask] = bounded
        diagnostics.append(
            {
                "target_rms": requested,
                "before_rms": actual,
                "gain": bounded,
            }
        )
    sigma = float(config["post_isp_gain_smoothing_sigma"])
    if sigma > 0.0:
        gain = gaussian_filter(gain, sigma=sigma, mode="reflect")
    return gain, diagnostics


def positive_matrix_power(value: np.ndarray, exponent: float) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    matrix = np.nan_to_num((matrix + matrix.T) * 0.5, nan=0.0, posinf=0.0, neginf=0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = max(float(np.max(eigenvalues)) * 1e-6, 1e-12)
    powered = np.maximum(eigenvalues, floor) ** exponent
    return (eigenvectors * powered) @ eigenvectors.T


def match_band_covariance(
    band: np.ndarray,
    clean_luma: np.ndarray,
    bins: np.ndarray,
    target_covariances: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Whiten/recolor an existing spatial band without inventing blob fields."""

    output = np.zeros_like(band, dtype=np.float32)
    assigned = np.zeros(clean_luma.shape, dtype=bool)
    low_gain, high_gain = map(float, config["post_isp_gain_bounds"])
    diagnostics = []
    for index, (low, high) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
        mask = (clean_luma >= low) & (clean_luma < high)
        count = int(mask.sum())
        if count < 4:
            output[mask] = band[mask]
            assigned |= mask
            diagnostics.append({"pixels": count, "applied": False})
            continue
        values = band[mask].astype(np.float64)
        mean = values.mean(axis=0)
        centered = values - mean
        current_covariance = centered.T @ centered / count
        target_covariance = np.asarray(target_covariances[index], dtype=np.float64)
        transform = (
            positive_matrix_power(target_covariance, 0.5)
            @ positive_matrix_power(current_covariance, -0.5)
        )
        left, singular, right = np.linalg.svd(transform, full_matrices=False)
        singular = np.clip(singular, low_gain, high_gain)
        transform = (left * singular) @ right
        output[mask] = (centered @ transform.T).astype(np.float32)
        assigned |= mask
        realized = output[mask].astype(np.float64)
        realized_covariance = realized.T @ realized / count
        diagnostics.append(
            {
                "pixels": count,
                "applied": True,
                "maximum_covariance_error": float(
                    np.max(np.abs(realized_covariance - target_covariance))
                ),
            }
        )
    output[~assigned] = band[~assigned]
    return output, diagnostics


def match_post_isp_bands(
    noisy: np.ndarray,
    clean: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    targets = profile["post_isp_band_targets"]
    bins = np.asarray(targets["luminance_bins"], dtype=np.float32)
    clean_luma = luminance(clean)
    current = noisy.astype(np.float32)
    last_diagnostics: dict[str, Any] = {}
    for _ in range(int(config["post_isp_match_iterations"])):
        bands = residual_bands(current - clean, profile)
        matched = np.zeros_like(current, dtype=np.float32)
        last_diagnostics = {}
        for band_name in BAND_NAMES:
            scalar, chroma = split_luma_chroma(bands[band_name])
            luma_gain, luma_diagnostics = component_gain_map(
                scalar,
                clean_luma,
                bins,
                np.asarray(targets[band_name]["luma_rms"], dtype=np.float32),
                config,
            )
            chroma_gain, chroma_diagnostics = component_gain_map(
                chroma,
                clean_luma,
                bins,
                np.asarray(targets[band_name]["chroma_rms"], dtype=np.float32),
                config,
            )
            amplitude_matched = (
                luma_vector(scalar * luma_gain) + chroma * chroma_gain[..., None]
            )
            covariance_matched, covariance_diagnostics = match_band_covariance(
                amplitude_matched,
                clean_luma,
                bins,
                np.asarray(targets[band_name]["rgb_covariance"], dtype=np.float64),
                config,
            )
            matched += covariance_matched
            last_diagnostics[band_name] = {
                "luma": luma_diagnostics,
                "chroma": chroma_diagnostics,
                "covariance": covariance_diagnostics,
            }
        current = np.rint(np.clip(clean + matched, 0.0, 1.0) * 255.0).astype(np.float32) / 255.0
    actual_bands = residual_bands(current - clean, profile)
    summary = {}
    covariance_verification = {}
    for band_name, band in actual_bands.items():
        scalar, chroma = split_luma_chroma(band)
        flattened = band.reshape(-1, 3).astype(np.float64)
        centered = flattened - flattened.mean(axis=0, keepdims=True)
        global_covariance = centered.T @ centered / len(flattened)
        by_luminance_bin = []
        target_covariances = np.asarray(
            targets[band_name]["rgb_covariance"], dtype=np.float64
        )
        target_correlations = np.asarray(
            targets[band_name]["rgb_correlation"], dtype=np.float64
        )
        for index, (low, high) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
            mask = (clean_luma >= low) & (clean_luma < high)
            values = band[mask].astype(np.float64)
            if len(values) < 4:
                by_luminance_bin.append(
                    {"pixels": int(len(values)), "measured": False}
                )
                continue
            centered_values = values - values.mean(axis=0, keepdims=True)
            covariance = centered_values.T @ centered_values / len(values)
            actual_correlation = covariance_correlation(covariance)
            actual_rms = math.sqrt(float(np.square(values).mean()))
            target_rms = math.sqrt(float(np.trace(target_covariances[index]) / 3.0))
            by_luminance_bin.append(
                {
                    "pixels": int(len(values)),
                    "measured": True,
                    "rgb_rms_log_error": abs(
                        math.log(max(actual_rms, 1e-12) / max(target_rms, 1e-12))
                    ),
                    "rgb_correlation_maximum_error": float(
                        np.max(np.abs(actual_correlation - target_correlations[index]))
                    ),
                }
            )
        summary[band_name] = {
            "luma_rms": float(np.sqrt(np.mean(np.square(scalar)))),
            "chroma_rms": float(np.sqrt(np.mean(np.square(chroma)))),
        }
        covariance_verification[band_name] = {
            "rgb_correlation": covariance_correlation(global_covariance).tolist(),
            "by_luminance_bin": by_luminance_bin,
        }
    return current, {
        "iterations": int(config["post_isp_match_iterations"]),
        "bands": summary,
        "covariance_verification": covariance_verification,
    }


def synthesize_pair(
    clean_srgb: np.ndarray,
    severity_profile: dict[str, Any],
    noise_seed: int,
    synthesis_config: dict[str, Any],
    *,
    isp_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return noisy/clean camera-JPEG branches and their realized profile."""

    clean = np.asarray(clean_srgb, dtype=np.float32)
    if clean.shape != (TILE, TILE, 3):
        raise ValueError(f"Expected a {TILE}x{TILE} RGB crop, got {clean.shape}")
    if not np.isfinite(clean).all() or float(clean.min()) < 0.0 or float(clean.max()) > 1.0:
        raise ValueError("Clean crop must contain finite RGB values in [0,1]")
    validate_severity_profile(severity_profile)
    noise_rng = np.random.default_rng(int(noise_seed))
    resolved_isp_seed = (
        int(isp_seed) if isp_seed is not None else stable_seed(int(noise_seed), "isp")
    )
    isp_rng = np.random.default_rng(resolved_isp_seed)
    clean_linear = srgb_to_linear(clean).astype(np.float32)
    height, width = clean_linear.shape[:2]
    correlation = _profile_matrix(
        severity_profile["rgb_correlation"], "rgb_correlation"
    )
    values, vectors = np.linalg.eigh((correlation + correlation.T) * 0.5)
    correlation = (vectors * np.maximum(values, 1e-5)) @ vectors.T
    diagonal = np.sqrt(np.maximum(np.diag(correlation), 1e-8))
    correlation /= np.outer(diagonal, diagonal)
    cholesky = np.linalg.cholesky(correlation + np.eye(3) * 1e-6).astype(np.float32)
    white = noise_rng.normal(0.0, 1.0, clean_linear.shape).astype(np.float32)
    correlated = np.einsum("...c,dc->...d", white, cholesky, optimize=True)
    shot = np.asarray(severity_profile["shot_scale"], dtype=np.float32)
    read = np.asarray(severity_profile["read_variance"], dtype=np.float32)
    variance = read + shot * np.clip(clean_linear, 0.0, 1.0)
    fine = correlated * np.sqrt(np.maximum(variance, 0.0))
    noisy_linear = np.clip(clean_linear + fine, 0.0, 1.0).astype(np.float32)
    isp_profile = draw_isp_profile(isp_rng, synthesis_config)
    clean_target = apply_isp(clean_linear, isp_profile).astype(np.float32)
    noisy_target = apply_isp(noisy_linear, isp_profile).astype(np.float32)

    line_sigma = float(severity_profile["row_column_smoothing_sigma"])
    row_luma, row_chroma = line_field(height, width, 0, line_sigma, noise_rng)
    column_luma, column_chroma = line_field(height, width, 1, line_sigma, noise_rng)
    line_scale = float(synthesis_config["row_column_component_scale"])
    line_residual = line_scale * (
        row_luma * float(severity_profile["row_luma_rms"])
        + row_chroma * float(severity_profile["row_chroma_rms"])
        + column_luma * float(severity_profile["column_luma_rms"])
        + column_chroma * float(severity_profile["column_chroma_rms"])
    )
    preliminary = np.clip(noisy_target + line_residual, 0.0, 1.0).astype(np.float32)
    noisy_target, post_match = match_post_isp_bands(
        preliminary,
        clean_target,
        severity_profile,
        synthesis_config,
    )
    realized = {
        "noise_profile": json.loads(json.dumps(severity_profile, allow_nan=False)),
        "isp_profile": isp_profile,
        "noise_seed": int(noise_seed),
        "isp_seed": resolved_isp_seed,
        "post_isp_match": post_match,
    }
    return noisy_target, clean_target, realized


def supervision_for(config: dict[str, Any], source_dataset: str) -> dict[str, Any]:
    value = config["supervision"].get("by_source_dataset", {}).get(
        source_dataset, config["supervision"]["default"]
    )
    return {
        "supervision": str(value["label"]),
        "gt_weight": float(value["gt_weight"]),
        "kd_weight": float(value["kd_weight"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs/synthetic_camera_jpeg_gate.yaml",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument(
        "--calibration-profile",
        type=Path,
        help="Override the configured profile, primarily for an isolated smoke cache.",
    )
    parser.add_argument(
        "--limit-clean-per-dataset",
        type=int,
        help="Smoke limit applied independently to each dataset/split.",
    )
    parser.add_argument(
        "--patches-per-clean",
        type=int,
        help="Smoke override for every source dataset; never use for a release cache.",
    )
    parser.add_argument("--profiles", help="Comma-separated ISO severities from the profile.")
    parser.add_argument(
        "--allow-smoke-profile",
        action="store_true",
        help="Allow a profile fitted with --limit-pairs-per-iso for a smoke cache.",
    )
    args = parser.parse_args()
    if args.limit_clean_per_dataset is not None and args.limit_clean_per_dataset < 1:
        parser.error("--limit-clean-per-dataset must be positive")
    if args.patches_per_clean is not None and args.patches_per_clean < 1:
        parser.error("--patches-per-clean must be positive")
    if args.allow_smoke_profile and args.limit_clean_per_dataset is None:
        parser.error("--allow-smoke-profile requires --limit-clean-per-dataset")

    config = load_config(args.config)
    validate_settings(config)
    exclusions = load_holdout_exclusions(config)
    requested_isos = (
        [int(value) for value in args.profiles.split(",")]
        if args.profiles
        else [int(value) for value in config["data"]["severity_isos"]]
    )
    if requested_isos != sorted(set(requested_isos)):
        parser.error("--profiles must be sorted and unique")
    profile_path, profile_document, profiles = load_profiles(
        config,
        requested_isos,
        exclusions,
        args.allow_smoke_profile,
        args.calibration_profile,
    )
    source_manifest, source_root, source_records = source_document(config)
    sources = select_clean_sources(
        source_records,
        source_root,
        set(map(str, config["data"]["datasets"])),
        exclusions,
        limit_per_dataset=args.limit_clean_per_dataset,
    )
    if not sources:
        raise RuntimeError("No clean source images remain after deduplication")

    cache_root = (
        args.cache_root.expanduser().resolve()
        if args.cache_root is not None
        else resolve_paper_path(config["data"]["cache_root"])
    )
    output_manifest = (
        args.output_manifest.expanduser().resolve()
        if args.output_manifest is not None
        else resolve_paper_path(config["data"]["manifest"])
    )
    try:
        manifest_relative = output_manifest.relative_to(cache_root)
    except ValueError as error:
        raise ValueError("Output manifest must be inside the synthetic cache root") from error
    build_root = cache_root.with_name(f".{cache_root.name}.building")
    backup_root = cache_root.with_name(f".{cache_root.name}.previous")
    if backup_root.exists() and not cache_root.exists():
        backup_root.rename(cache_root)
    if cache_root.exists() and not args.replace:
        raise FileExistsError(f"Cache exists: {cache_root}; pass --replace")
    if build_root.exists():
        if not args.replace:
            raise FileExistsError(f"Incomplete build exists: {build_root}; pass --replace")
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)

    teacher_repo = resolve_paper_path(config["teacher"]["repository"])
    teacher_checkpoint = resolve_paper_path(config["teacher"]["checkpoint"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required to cache full-precision SCUNet targets")
    teacher = load_scunet_teacher(teacher_repo, teacher_checkpoint, device)
    batch_size = int(config["teacher"]["cache_batch_size"])
    candidate_count = int(config["data"]["crop_candidates"])
    replicates = int(config["data"]["replicates_per_severity"])
    global_seed = int(config["project"]["seed"])
    patch_counts = {
        dataset: (
            args.patches_per_clean
            if args.patches_per_clean is not None
            else int(config["data"]["patches_per_clean_source"][dataset])
        )
        for dataset in map(str, config["data"]["datasets"])
    }
    limited_smoke = args.limit_clean_per_dataset is not None or args.patches_per_clean is not None
    pending: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    records: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        input_batch = np.stack([row[1] for row in pending])
        tensor = torch.from_numpy(input_batch).permute(0, 3, 1, 2).to(device)
        with torch.inference_mode():
            predictions = (
                teacher(tensor).clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy()
            )
        for (metadata, noisy, clean), teacher_target in zip(pending, predictions, strict=True):
            base = Path(metadata["split"]) / DATASET / metadata["id"]
            paths = {
                name: str(base.with_name(base.name + f"_{name}.npy"))
                for name in ARRAY_FIELDS
            }
            for relative in paths.values():
                (build_root / relative).parent.mkdir(parents=True, exist_ok=True)
            np.save(build_root / paths["input"], noisy.astype(np.float32))
            np.save(build_root / paths["clean"], clean.astype(np.float32))
            np.save(build_root / paths["teacher"], teacher_target.astype(np.float16))
            array_sha256 = {
                field: sha256_file(build_root / paths[field]) for field in ARRAY_FIELDS
            }
            records.append({**metadata, **paths, "array_sha256": array_sha256})
        pending.clear()

    for source in tqdm(sources, desc="Generating synthetic camera-JPEG samples"):
        source_dataset = str(source["dataset"])
        clean_path = Path(source["_clean_path"])
        clean_sha = str(source["_clean_sha256"])
        with ImageSource(clean_path) as clean_source:
            thumbnail = clean_source.thumbnail(480)
            positions = stratified_positions(
                thumbnail,
                clean_source.width,
                clean_source.height,
                patch_counts[source_dataset],
                candidate_count,
                stable_seed(
                    global_seed,
                    source_dataset,
                    source["scene"],
                    source["clean"],
                    clean_sha,
                ),
            )
            for crop_index, (left, top, crop_luma) in enumerate(positions):
                raw_clean = clean_source.crop(left, top)
                for severity_iso in requested_isos:
                    severity_profile = profiles[severity_iso]
                    for replicate in range(replicates):
                        isp_seed = stable_seed(
                            global_seed,
                            clean_sha,
                            left,
                            top,
                            "isp",
                        )
                        noise_seed = stable_seed(
                            global_seed,
                            clean_sha,
                            left,
                            top,
                            severity_iso,
                            replicate,
                        )
                        noisy, clean, realized = synthesize_pair(
                            raw_clean,
                            severity_profile,
                            noise_seed,
                            config["synthesis"],
                            isp_seed=isp_seed,
                        )
                        identity = hashlib.sha256(
                            (
                                f"{source_dataset}:{source['scene']}:{clean_sha}:"
                                f"{left}:{top}:{severity_iso}:{replicate}"
                            ).encode("utf-8")
                        ).hexdigest()
                        metadata = {
                            "id": identity[:24],
                            "dataset": DATASET,
                            "source_dataset": source_dataset,
                            "scene": f"{source_dataset}:{source['scene']}",
                            "source_scene": str(source["scene"]),
                            "split": str(source["split"]),
                            "source_clean": str(source["clean"]),
                            "source_clean_sha256": clean_sha,
                            "crop": [left, top, TILE, TILE],
                            "crop_index": crop_index,
                            "crop_mean_luminance": float(crop_luma),
                            "severity_iso": severity_iso,
                            "generation_seed": noise_seed,
                            "noise_seed": noise_seed,
                            "isp_seed": isp_seed,
                            "noise_profile": realized["noise_profile"],
                            "isp_profile": realized["isp_profile"],
                            "post_isp_match": realized["post_isp_match"],
                            "domain": "synthetic camera-processed 8-bit sRGB JPEG",
                            **supervision_for(config, source_dataset),
                            **{
                                key: source[key]
                                for key in (
                                    "camera",
                                    "clean_level",
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

    records.sort(key=lambda row: (row["split"], row["scene"], row["id"]))
    scene_splits: dict[str, set[str]] = collections.defaultdict(set)
    sha_splits: dict[str, set[str]] = collections.defaultdict(set)
    counts: collections.Counter[tuple[str, str, int]] = collections.Counter()
    for row in records:
        scene_splits[row["scene"]].add(row["split"])
        sha_splits[row["source_clean_sha256"]].add(row["split"])
        counts[(row["split"], row["source_dataset"], row["severity_iso"])] += 1
    scene_leakage = [scene for scene, splits in scene_splits.items() if len(splits) != 1]
    sha_leakage = [digest for digest, splits in sha_splits.items() if len(splits) != 1]
    if scene_leakage or sha_leakage:
        raise RuntimeError(
            f"Synthetic cache leakage: scenes={scene_leakage[:1]}, SHA={sha_leakage[:1]}"
        )
    if len({row["id"] for row in records}) != len(records):
        raise RuntimeError("Synthetic record ID collision")

    config_path = Path(config["_config_path"])
    manifest = {
        "schema_version": 2,
        "purpose": "Calibrated synthetic ISO 12800-51200 camera-JPEG training data",
        "preprocessing": config["project"]["preprocessing_version"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "calibration_profile": str(profile_path),
        "calibration_profile_sha256": sha256_file(profile_path),
        "calibration_profile_version": profile_document["profile_version"],
        "calibration_fit_split": profile_document["fit_split"],
        "calibration_is_limited_smoke": bool(profile_document["fit_is_limited_smoke"]),
        "calibration_basis": profile_document["calibration_basis"],
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "holdout_exclusions": exclusions,
        "array_dtypes": {"input": "float32", "clean": "float32", "teacher": "float16"},
        "array_integrity": {
            "field": "array_sha256",
            "algorithm": "SHA-256",
            "scope": "complete .npy file bytes immediately after atomic-cache write",
            "required_arrays": list(ARRAY_FIELDS),
        },
        "generation_environment": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "teacher_inference_dtype": "float32",
            "teacher_batch_size": batch_size,
            "seed": global_seed,
            "limited_smoke": limited_smoke,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "records": records,
    }
    report = {
        "manifest": str(output_manifest),
        "records": len(records),
        "clean_sources": len(sources),
        "unique_clean_payloads": len(sha_splits),
        "scene_groups": len(scene_splits),
        "scene_leakage": 0,
        "clean_sha_leakage": 0,
        "limited_smoke": limited_smoke,
        "calibration_is_limited_smoke": bool(profile_document["fit_is_limited_smoke"]),
        "counts": {
            f"{split}/{dataset}/iso{iso}": count
            for (split, dataset, iso), count in sorted(counts.items())
        },
        "array_dtypes": manifest["array_dtypes"],
        "array_hashes": "per-record SHA-256 of complete .npy file bytes",
        "teacher_inference_dtype": "float32",
    }
    atomic_json(build_root / manifest_relative, manifest)
    atomic_json(build_root / manifest_relative.with_suffix(".report.json"), report)

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
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
