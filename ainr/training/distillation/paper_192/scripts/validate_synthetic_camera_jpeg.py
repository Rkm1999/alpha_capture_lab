#!/usr/bin/env python3
"""Audit and visualize the calibrated synthetic high-ISO JPEG cache.

The release gate is intentionally independent from training.  It verifies the
cache's immutable provenance, excludes protected Sony images by both path and
content hash, checks deterministic reconstruction, and measures whether the
synthetic residual has useful luma/chroma and spatial-frequency structure.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import math
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from common import atomic_json, load_config, resolve_paper_path, sha256_file


TILE = 192
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
ARRAY_KEYS = ("input", "clean", "teacher")
EXPECTED_SPLITS = {"train", "validation"}
FITTER_SCRIPT = Path(__file__).with_name("fit_synthetic_noise_profiles.py").resolve()
GENERATOR_SCRIPT = Path(__file__).with_name("prepare_synthetic_camera_jpeg.py").resolve()
VALIDATOR_SCRIPT = Path(__file__).resolve()
GENERATOR_IDENTITY = "synthetic_camera_jpeg_generator"
VALIDATOR_IDENTITY = "synthetic_camera_jpeg_release_gate"
VALIDATOR_SEMANTIC_VERSION = "synthetic_camera_jpeg_release_gate_v3"
ANALYSIS_SCHEMA_VERSION = 3
Reconstructor = Callable[[dict[str, Any]], Mapping[str, np.ndarray]]
VISUAL_ACCEPTANCE_KEYS = {
    "schema_version",
    "decision",
    "reviewer",
    "accepted_at",
    "manifest_sha256",
    "cache_content_sha256",
    "contact_png_sha256",
    "contact_jpg_sha256",
    "analysis_sha256",
    "reviewed_report_sha256",
}
ANALYSIS_KEYS = {
    "schema_version",
    "semantic_version",
    "validator",
    "generator",
    "manifest",
    "preprocessing",
    "records_loaded",
    "records_measured",
    "cache_content",
    "smoke",
    "protected_holdout",
    "sources_and_splits",
    "reconstruction",
    "severity",
    "calibration",
    "findings",
    "contact_sheet",
}
ANALYSIS_MIRROR_KEYS = (
    "manifest",
    "cache_content",
    "smoke",
    "contact_sheet",
    "preprocessing",
    "records_loaded",
    "records_measured",
    "protected_holdout",
    "sources_and_splits",
    "reconstruction",
    "severity",
    "calibration",
    "findings",
)


class Findings:
    """Collect bounded errors and warnings while completing the whole audit."""

    def __init__(self, examples_per_code: int = 5) -> None:
        self.examples_per_code = examples_per_code
        self.counts = {
            "errors": collections.Counter(),
            "warnings": collections.Counter(),
        }
        self.examples: dict[str, dict[str, list[str]]] = {
            "errors": collections.defaultdict(list),
            "warnings": collections.defaultdict(list),
        }

    def add(self, severity: str, code: str, message: str) -> None:
        self.counts[severity][code] += 1
        examples = self.examples[severity][code]
        if len(examples) < self.examples_per_code:
            examples.append(message)

    def error(self, code: str, message: str) -> None:
        self.add("errors", code, message)

    def warn(self, code: str, message: str) -> None:
        self.add("warnings", code, message)

    @property
    def failed(self) -> bool:
        return bool(self.counts["errors"])

    @property
    def warned(self) -> bool:
        return bool(self.counts["warnings"])

    def report(self) -> dict[str, Any]:
        return {
            severity: {
                "total": int(sum(self.counts[severity].values())),
                "by_code": dict(sorted(self.counts[severity].items())),
                "examples": dict(sorted(self.examples[severity].items())),
            }
            for severity in ("errors", "warnings")
        }


@dataclass(frozen=True)
class GateThresholds:
    reconstruction_atol: float = 1e-7
    minimum_monotonic_fraction: float = 0.95
    minimum_teacher_better_fraction: float = 0.90
    minimum_teacher_gain_median_db: float = 1.0
    minimum_shadow_fraction: float = 0.15
    minimum_structured_band_fraction: float = 0.01
    maximum_calibration_log_rms_error: float = math.log(1.25)
    maximum_calibration_band_l1: float = 0.20
    maximum_calibration_correlation_error: float = 0.15

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "GateThresholds":
        gate = config.get("gate", {})
        values = {}
        for name in cls.__dataclass_fields__:
            if name in gate:
                values[name] = float(gate[name])
        result = cls(**values)
        for name, value in result.__dict__.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"gate.{name} must be finite and non-negative")
        if result.minimum_monotonic_fraction > 1.0:
            raise ValueError("gate.minimum_monotonic_fraction must be <= 1")
        if result.minimum_teacher_better_fraction > 1.0:
            raise ValueError("gate.minimum_teacher_better_fraction must be <= 1")
        return result


@dataclass(frozen=True)
class CalibrationSupportThresholds:
    minimum_pixels: int
    minimum_payloads: int
    minimum_scenes: int
    shadow_luminance_max: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CalibrationSupportThresholds":
        calibration = config.get("calibration", {})
        gate = config.get("gate", {})

        def positive_integer(
            source: Mapping[str, Any], key: str, qualified_name: str
        ) -> int:
            value = source.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{qualified_name} must be a positive integer")
            return value

        shadow_max = float(gate.get("calibration_shadow_luminance_max", math.nan))
        if not math.isfinite(shadow_max) or not 0.0 < shadow_max < 1.0:
            raise ValueError("gate.calibration_shadow_luminance_max must be in (0,1)")
        return cls(
            minimum_pixels=positive_integer(
                calibration,
                "minimum_pixels_per_bin",
                "calibration.minimum_pixels_per_bin",
            ),
            minimum_payloads=positive_integer(
                gate,
                "minimum_calibration_reference_payloads_per_bin",
                "gate.minimum_calibration_reference_payloads_per_bin",
            ),
            minimum_scenes=positive_integer(
                gate,
                "minimum_calibration_reference_scenes_per_bin",
                "gate.minimum_calibration_reference_scenes_per_bin",
            ),
            shadow_luminance_max=shadow_max,
        )


def contained_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def array_content_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(canonical_json(list(contiguous.shape)))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def load_checked_array(
    cache_root: Path,
    record: Mapping[str, Any],
    key: str,
    findings: Findings,
) -> np.ndarray | None:
    identifier = str(record.get("id", "<missing-id>"))
    path = contained_path(cache_root, record.get(key))
    if path is None:
        findings.error("unsafe_array_path", f"{identifier}/{key}: {record.get(key)!r}")
        return None
    if not path.is_file():
        findings.error("missing_array", f"{identifier}/{key}: {path}")
        return None
    try:
        stored = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:  # noqa: BLE001 - aggregate corrupt entries
        findings.error("unreadable_array", f"{identifier}/{key}: {error}")
        return None
    expected_dtype = np.dtype(np.float16 if key == "teacher" else np.float32)
    if stored.shape != (TILE, TILE, 3):
        findings.error("array_shape", f"{identifier}/{key}: {stored.shape}")
        return None
    if stored.dtype != expected_dtype:
        findings.error(
            "array_dtype", f"{identifier}/{key}: {stored.dtype}, expected {expected_dtype}"
        )
    if not np.issubdtype(stored.dtype, np.floating):
        return None
    value = np.asarray(stored, dtype=np.float32)
    if not np.isfinite(value).all():
        findings.error("array_nonfinite", f"{identifier}/{key}")
        return None
    if float(value.min()) < 0.0 or float(value.max()) > 1.0:
        findings.error(
            "array_range",
            f"{identifier}/{key}: [{float(value.min()):.8g}, {float(value.max()):.8g}]",
        )
        return None
    expected_hashes = record.get("array_sha256")
    expected_hash = expected_hashes.get(key) if isinstance(expected_hashes, dict) else None
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        findings.error("missing_array_sha256", f"{identifier}/{key}")
    elif sha256_file(path) != expected_hash.lower():
        findings.error("array_sha256_mismatch", f"{identifier}/{key}: {path}")
    return value


def luminance(value: np.ndarray) -> np.ndarray:
    return value @ LUMA


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    mse = float(np.square(first.astype(np.float64) - second.astype(np.float64)).mean())
    return math.inf if mse == 0.0 else -10.0 * math.log10(mse)


def residual_bands(value: np.ndarray) -> dict[str, np.ndarray]:
    low1 = gaussian_filter(value, sigma=(1.0, 1.0, 0.0), mode="reflect")
    low4 = gaussian_filter(value, sigma=(4.0, 4.0, 0.0), mode="reflect")
    low16 = gaussian_filter(value, sigma=(16.0, 16.0, 0.0), mode="reflect")
    return {
        "fine": value - low1,
        "medium": low1 - low4,
        "coarse": low4 - low16,
        "very_coarse": low16,
    }


def channel_correlation(value: np.ndarray) -> np.ndarray:
    flattened = value.reshape(-1, 3).astype(np.float64)
    if bool(np.all(flattened.std(axis=0) < 1e-12)):
        return np.eye(3, dtype=np.float64)
    result = np.corrcoef(flattened, rowvar=False)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def record_metrics(
    record: dict[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    include_arrays: bool = False,
) -> dict[str, Any]:
    noisy = arrays["input"]
    clean = arrays["clean"]
    teacher = arrays["teacher"]
    residual = noisy - clean
    luma_residual = luminance(residual)
    chroma_residual = np.stack(
        (residual[..., 2] - luma_residual, residual[..., 0] - luma_residual), axis=-1
    )
    bands = residual_bands(residual)
    band_energy = {name: float(np.square(value).mean()) for name, value in bands.items()}
    total_band_energy = sum(band_energy.values())
    band_fraction = {
        name: energy / total_band_energy if total_band_energy > 0.0 else 0.0
        for name, energy in band_energy.items()
    }
    gray = luminance(noisy)
    teacher_gain = psnr(teacher, clean) - psnr(noisy, clean)
    correction = teacher - noisy
    result = {
        "record": record,
        "metrics": {
            "iso": int(record["severity_iso"]),
            "mean_luminance": float(gray.mean()),
            "shadow_fraction": float((gray < 0.25).mean()),
            "noise_rms": float(np.sqrt(np.square(residual).mean())),
            "luma_noise_rms": float(np.sqrt(np.square(luma_residual).mean())),
            "chroma_noise_rms": float(np.sqrt(np.square(chroma_residual).mean())),
            "band_energy": band_energy,
            "band_fraction": band_fraction,
            "rgb_correlation": channel_correlation(residual).tolist(),
            "noisy_clean_psnr": psnr(noisy, clean),
            "teacher_clean_psnr": psnr(teacher, clean),
            "teacher_psnr_gain": teacher_gain,
            "teacher_correction_rms": float(np.sqrt(np.square(correction).mean())),
            "clean_content_sha256": array_content_sha256(clean),
        },
    }
    if include_arrays:
        result["arrays"] = dict(arrays)
    return result


def protected_holdout(
    config: Mapping[str, Any],
    document: Mapping[str, Any],
    findings: Findings,
) -> dict[str, Any]:
    data = config.get("data", {})
    configured = data.get("holdout_exclusions", [])
    cached = document.get("holdout_exclusions", [])
    if configured != cached:
        findings.error("holdout_exclusions_mismatch", "manifest and config exclusions differ")
    if not isinstance(configured, list) or not configured:
        findings.error("protected_holdout_missing", "data.holdout_exclusions must list files")
        return {"paths": [], "sha256": []}
    resolved: list[str] = []
    hashes: list[str] = []
    for entry in configured:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            findings.error("protected_holdout_entry", repr(entry))
            continue
        path = Path(entry["path"]).expanduser().resolve()
        if not path.is_file():
            findings.error("protected_holdout_file_missing", str(path))
            continue
        actual = sha256_file(path)
        if entry.get("sha256") != actual:
            findings.error("protected_holdout_sha256_mismatch", str(path))
        resolved.append(str(path))
        hashes.append(actual)
    if len(hashes) != len(set(hashes)):
        findings.warn("duplicate_holdout_content", "protected holdout contains duplicate files")
    return {"paths": resolved, "sha256": sorted(set(hashes))}


def validate_manifest_provenance(
    document: Mapping[str, Any],
    manifest_path: Path,
    config: Mapping[str, Any],
    findings: Findings,
) -> None:
    if document.get("schema_version") != 2:
        findings.error("manifest_schema", f"expected 2, got {document.get('schema_version')!r}")
    expected_preprocessing = config.get("project", {}).get("preprocessing_version")
    if document.get("preprocessing") != expected_preprocessing:
        findings.error(
            "preprocessing_mismatch",
            f"manifest={document.get('preprocessing')!r}, config={expected_preprocessing!r}",
        )
    config_sha = document.get("config_sha256")
    config_path = Path(str(config.get("_config_path", "")))
    if not config_path.is_file() or config_sha != sha256_file(config_path):
        findings.error("config_sha256_mismatch", str(config_path))
    if document.get("config") != str(config_path):
        findings.error("config_path_mismatch", f"{document.get('config')!r} != {config_path}")
    teacher_path = resolve_paper_path(config.get("teacher", {}).get("checkpoint", ""))
    teacher_sha = document.get("teacher_checkpoint_sha256")
    if not teacher_path.is_file() or teacher_sha != sha256_file(teacher_path):
        findings.error("teacher_checkpoint_sha256_mismatch", str(teacher_path))
    if document.get("teacher_checkpoint") != str(teacher_path):
        findings.error("teacher_checkpoint_path_mismatch", str(teacher_path))
    source_path = resolve_paper_path(document.get("source_manifest", ""))
    if not source_path.is_file() or document.get("source_manifest_sha256") != sha256_file(source_path):
        findings.error("source_manifest_sha256_mismatch", str(source_path))
    configured_source = resolve_paper_path(config.get("data", {}).get("source_manifest", ""))
    if source_path != configured_source:
        findings.error(
            "source_manifest_path_mismatch", f"{source_path} != {configured_source}"
        )
    profile_path = resolve_paper_path(document.get("calibration_profile", ""))
    if (
        not profile_path.is_file()
        or document.get("calibration_profile_sha256") != sha256_file(profile_path)
    ):
        findings.error("calibration_profile_sha256_mismatch", str(profile_path))
    configured_profile = resolve_paper_path(config.get("data", {}).get("calibration_profile", ""))
    if profile_path != configured_profile:
        findings.error(
            "calibration_profile_path_mismatch", f"{profile_path} != {configured_profile}"
        )
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        profile = {}
    if (
        not isinstance(profile, dict)
        or profile.get("schema_version") != 2
        or profile.get("profile_version")
        != "snic_train_linear_post_isp_covariance_v3"
        or profile.get("fit_split") != "train"
        or profile.get("fit_is_limited_smoke") is not False
    ):
        findings.error(
            "calibration_provenance_missing",
            "calibration profile must be schema 2, non-smoke, and declare fit_split=train",
        )
    elif profile.get("holdout_exclusions") != config.get("data", {}).get("holdout_exclusions"):
        findings.error("calibration_holdout_mismatch", str(profile_path))
    if isinstance(profile, dict):
        fitting_environment = profile.get("fitting_environment")
        active_fitter_sha = sha256_file(FITTER_SCRIPT)
        if (
            not isinstance(fitting_environment, dict)
            or fitting_environment.get("script_sha256") != active_fitter_sha
        ):
            findings.error(
                "calibration_fitter_sha256_mismatch",
                f"profile={None if not isinstance(fitting_environment, dict) else fitting_environment.get('script_sha256')!r}, "
                f"active={active_fitter_sha}",
            )
        if profile.get("preprocessing") != expected_preprocessing:
            findings.error("calibration_preprocessing_mismatch", str(profile_path))
        if config_path.is_file() and profile.get("config_sha256") != sha256_file(config_path):
            findings.error("calibration_config_sha256_mismatch", str(profile_path))
        if source_path.is_file() and profile.get("source_manifest_sha256") != sha256_file(source_path):
            findings.error("calibration_source_sha256_mismatch", str(profile_path))
        basis = profile.get("calibration_basis")
        if not isinstance(basis, dict) or basis.get("fit_split") != "train":
            findings.error("calibration_basis_missing", str(profile_path))
        else:
            if document.get("calibration_basis") != basis:
                findings.error("manifest_calibration_basis_mismatch", str(manifest_path))
            support = basis.get("fit_support")
            if (
                not isinstance(support, dict)
                or int(support.get("unique_clean_payloads", 0)) < 12
                or int(support.get("unique_scenes", 0)) < 5
                or int(support.get("pairs_by_iso", {}).get("12800", 0)) < 12
            ):
                findings.error("calibration_fit_support_insufficient", str(profile_path))
            distribution_keys = (
                "white_balance_range",
                "color_matrix_jitter",
                "tone_gamma_range",
                "sharpen_amount_range",
                "sharpen_sigma_range",
                "jpeg_quality_range",
                "jpeg_subsampling",
            )
            expected_distribution = {
                key: config.get("synthesis", {}).get(key) for key in distribution_keys
            }
            if basis.get("post_isp_distribution") != expected_distribution:
                findings.error("calibration_isp_distribution_mismatch", str(profile_path))
    if document.get("calibration_fit_split") != "train":
        findings.error("manifest_calibration_split", repr(document.get("calibration_fit_split")))
    if document.get("calibration_is_limited_smoke") is not False:
        findings.error(
            "manifest_calibration_smoke",
            repr(document.get("calibration_is_limited_smoke")),
        )
    expected_dtypes = {"input": "float32", "clean": "float32", "teacher": "float16"}
    if document.get("array_dtypes") != expected_dtypes:
        findings.error("array_dtype_contract", repr(document.get("array_dtypes")))
    expected_integrity = {
        "field": "array_sha256",
        "algorithm": "SHA-256",
        "scope": "complete .npy file bytes immediately after atomic-cache write",
        "required_arrays": list(ARRAY_KEYS),
    }
    if document.get("array_integrity") != expected_integrity:
        findings.error("array_integrity_contract", repr(document.get("array_integrity")))
    generation_environment = document.get("generation_environment")
    if not isinstance(generation_environment, dict):
        findings.error("generation_environment_missing", str(manifest_path))
    else:
        active_generator = active_generator_identity()
        if generation_environment.get("limited_smoke") is not False:
            findings.error("generation_smoke_cache", str(manifest_path))
        if generation_environment.get("script") != active_generator["path"]:
            findings.error(
                "generation_script_mismatch",
                f"{generation_environment.get('script')!r} != {active_generator['path']!r}",
            )
        if generation_environment.get("script_sha256") != active_generator["sha256"]:
            findings.error(
                "generation_script_sha256_mismatch",
                f"{generation_environment.get('script_sha256')!r} != "
                f"{active_generator['sha256']!r}",
            )


def validate_record_semantics(record: dict[str, Any], findings: Findings) -> None:
    identifier = str(record.get("id", "<missing-id>"))
    required = {
        "id", "dataset", "source_dataset", "source_scene", "scene", "split", "severity_iso",
        "source_clean", "source_clean_sha256", "crop", "crop_mean_luminance", "generation_seed",
        "noise_seed", "isp_seed",
        "noise_profile", "isp_profile", "post_isp_match", "supervision", "gt_weight", "kd_weight",
        "array_sha256", *ARRAY_KEYS,
    }
    missing = sorted(required - record.keys())
    if missing:
        findings.error("record_fields_missing", f"{identifier}: {missing}")
        return
    if record["dataset"] != "synthetic_camera_jpeg":
        findings.error("record_dataset", f"{identifier}: {record['dataset']!r}")
    if record["split"] not in EXPECTED_SPLITS:
        findings.error("record_split", f"{identifier}: {record['split']!r}")
    try:
        iso = int(record["severity_iso"])
    except (TypeError, ValueError):
        findings.error("record_iso", f"{identifier}: {record['severity_iso']!r}")
    else:
        if iso not in {12800, 25600, 51200}:
            findings.error("record_iso", f"{identifier}: {iso}")
    crop = record["crop"]
    if (
        not isinstance(crop, list) or len(crop) != 4
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in crop)
        or crop[0] < 0 or crop[1] < 0 or crop[2:] != [TILE, TILE]
    ):
        findings.error("record_crop", f"{identifier}: {crop!r}")
    if not isinstance(record["generation_seed"], int) or isinstance(record["generation_seed"], bool):
        findings.error("record_seed", f"{identifier}: {record['generation_seed']!r}")
    if record["generation_seed"] != record["noise_seed"]:
        findings.error("record_noise_seed_alias", identifier)
    for key in ("noise_seed", "isp_seed"):
        if not isinstance(record[key], int) or isinstance(record[key], bool):
            findings.error("record_seed", f"{identifier}/{key}: {record[key]!r}")
    expected_scene = f"{record['source_dataset']}:{record['source_scene']}"
    if record["scene"] != expected_scene:
        findings.error("record_scene_identity", f"{identifier}: {record['scene']!r} != {expected_scene!r}")
    if record["supervision"] != "synthetic_paired":
        findings.error("record_supervision", f"{identifier}: {record['supervision']!r}")
    if not math.isclose(float(record["gt_weight"]), 1.0, abs_tol=1e-12):
        findings.error("record_gt_weight", f"{identifier}: {record['gt_weight']!r}")
    if not math.isclose(float(record["kd_weight"]), 0.7, abs_tol=1e-12):
        findings.error("record_kd_weight", f"{identifier}: {record['kd_weight']!r}")
    match = record["post_isp_match"]
    expected_bands = {"fine", "medium", "coarse", "very_coarse"}
    if (
        not isinstance(match, dict)
        or not isinstance(match.get("iterations"), int)
        or isinstance(match.get("iterations"), bool)
        or match["iterations"] < 1
        or not isinstance(match.get("bands"), dict)
        or set(match["bands"]) != expected_bands
    ):
        findings.error("record_post_isp_match", identifier)
    else:
        for band, values in match["bands"].items():
            if not isinstance(values, dict) or set(values) != {"luma_rms", "chroma_rms"}:
                findings.error("record_post_isp_match", f"{identifier}/{band}")
                continue
            for component, value in values.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    findings.error(
                        "record_post_isp_match", f"{identifier}/{band}/{component}"
                    )
                elif not math.isfinite(float(value)) or float(value) < 0.0:
                    findings.error(
                        "record_post_isp_match", f"{identifier}/{band}/{component}"
                    )


def validate_sources_and_splits(
    records: list[dict[str, Any]],
    source_root: Path,
    holdout: Mapping[str, Any],
    findings: Findings,
    *,
    hash_file: Callable[[Path], str] = sha256_file,
) -> dict[str, Any]:
    protected_paths = {str(Path(value).resolve()) for value in holdout["paths"]}
    protected_sha = set(holdout["sha256"])
    ids: set[str] = set()
    array_paths: set[str] = set()
    scene_splits: dict[str, set[str]] = collections.defaultdict(set)
    sha_splits: dict[str, set[str]] = collections.defaultdict(set)
    source_hash_cache: dict[Path, str] = {}
    for record in records:
        identifier = str(record.get("id", "<missing-id>"))
        if identifier in ids:
            findings.error("duplicate_record_id", identifier)
        ids.add(identifier)
        scene_splits[str(record.get("scene"))].add(str(record.get("split")))
        for key in ARRAY_KEYS:
            value = str(record.get(key))
            if value in array_paths:
                findings.error("duplicate_array_path", f"{identifier}/{key}: {value}")
            array_paths.add(value)
        source = contained_path(source_root, record.get("source_clean"))
        if source is None or not source.is_file():
            findings.error("source_clean_missing", f"{identifier}: {source}")
            continue
        actual_sha = source_hash_cache.get(source)
        if actual_sha is None:
            actual_sha = hash_file(source)
            source_hash_cache[source] = actual_sha
        declared_sha = str(record.get("source_clean_sha256", "")).lower()
        if actual_sha != declared_sha:
            findings.error("source_clean_sha256_mismatch", f"{identifier}: {source}")
        sha_splits[actual_sha].add(str(record.get("split")))
        if str(source.resolve()) in protected_paths or actual_sha in protected_sha:
            findings.error("protected_holdout_overlap", f"{identifier}: {source}")
    for scene, splits in scene_splits.items():
        if len(splits) > 1:
            findings.error("scene_split_leakage", f"{scene}: {sorted(splits)}")
    for digest, splits in sha_splits.items():
        if len(splits) > 1:
            findings.error("source_sha_split_leakage", f"{digest}: {sorted(splits)}")
    return {
        "records": len(records),
        "record_ids": len(ids),
        "source_files": len(source_hash_cache),
        "scenes": len(scene_splits),
        "scene_split_leakage": sum(len(value) > 1 for value in scene_splits.values()),
        "source_sha_split_leakage": sum(len(value) > 1 for value in sha_splits.values()),
        "protected_overlap": findings.counts["errors"]["protected_holdout_overlap"],
    }


def reconstruction_stratum(row: dict[str, Any]) -> tuple[str, str, int, str]:
    mean = float(row["metrics"]["mean_luminance"])
    luminance_band = "deep" if mean < 0.10 else "shadow" if mean < 0.25 else "midtone" if mean < 0.75 else "highlight"
    record = row["record"]
    return (
        str(record["split"]),
        str(record["source_dataset"]),
        int(record["severity_iso"]),
        luminance_band,
    )


def deterministic_indices(rows: list[dict[str, Any]], count: int, seed: int) -> list[int]:
    """Round-robin stable-hash selection across split/domain/ISO/luminance strata."""

    if count <= 0 or count >= len(rows):
        return list(range(len(rows)))
    groups: dict[tuple[str, str, int, str], list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        groups[reconstruction_stratum(row)].append(index)
    for key, indices in groups.items():
        indices.sort(
            key=lambda index: hashlib.sha256(
                canonical_json([seed, key, rows[index]["record"]["id"]])
            ).digest()
        )
    selected: list[int] = []
    depth = 0
    ordered_keys = sorted(groups)
    while len(selected) < count:
        added = False
        for key in ordered_keys:
            if depth < len(groups[key]):
                selected.append(groups[key][depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    return sorted(selected)


def cache_content_identity(cache_root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Hash every cached tensor so the report fixes the exact audited payload."""

    digest = hashlib.sha256()
    files = 0
    bytes_total = 0
    for record in sorted(records, key=lambda value: str(value.get("id"))):
        for key in ARRAY_KEYS:
            path = contained_path(cache_root, record.get(key))
            if path is None or not path.is_file():
                continue
            entry = [str(record.get("id")), key, str(record[key]), sha256_file(path)]
            digest.update(canonical_json(entry))
            files += 1
            bytes_total += path.stat().st_size
    return {"files": files, "bytes": bytes_total, "sha256": digest.hexdigest()}


def validate_reconstruction(
    rows: list[dict[str, Any]],
    cache_root: Path,
    reconstructor: Reconstructor | None,
    thresholds: GateThresholds,
    sample_count: int,
    seed: int,
    findings: Findings,
) -> dict[str, Any]:
    if reconstructor is None:
        findings.error("reconstructor_missing", "generator did not expose reconstruct_record")
        return {"records_requested": 0, "records_reconstructed": 0}
    selected = deterministic_indices(rows, sample_count, seed)
    maximum_error = {"input": 0.0, "clean": 0.0}
    checked = 0
    for index in selected:
        row = rows[index]
        identifier = str(row["record"].get("id"))
        try:
            first = reconstructor(row["record"])
            second = reconstructor(row["record"])
        except Exception as error:  # noqa: BLE001 - report every reconstruction failure
            findings.error("reconstruction_failed", f"{identifier}: {error}")
            continue
        for key in ("input", "clean"):
            if key not in first or key not in second:
                findings.error("reconstruction_key_missing", f"{identifier}/{key}")
                continue
            left = np.asarray(first[key], dtype=np.float32)
            right = np.asarray(second[key], dtype=np.float32)
            if left.shape != (TILE, TILE, 3) or right.shape != left.shape:
                findings.error("reconstruction_shape", f"{identifier}/{key}: {left.shape}/{right.shape}")
                continue
            if not np.array_equal(left, right):
                findings.error("reconstruction_nondeterministic", f"{identifier}/{key}")
            cached_path = contained_path(cache_root, row["record"].get(key))
            if cached_path is None or not cached_path.is_file():
                findings.error("reconstruction_cache_missing", f"{identifier}/{key}")
                continue
            cached = np.load(cached_path, allow_pickle=False).astype(np.float32)
            difference = np.abs(left - cached)
            maximum = float(difference.max())
            maximum_error[key] = max(maximum_error[key], maximum)
            if maximum > thresholds.reconstruction_atol:
                findings.error(
                    "reconstruction_cache_mismatch",
                    f"{identifier}/{key}: max={maximum:.8g}, tolerance={thresholds.reconstruction_atol:.8g}",
                )
        checked += 1
    return {
        "records_requested": len(selected),
        "records_reconstructed": checked,
        "maximum_absolute_error": maximum_error,
        "tolerance": thresholds.reconstruction_atol,
        "determinism": "two independent reconstructions must be bit-identical",
        "selection": "stable round-robin over split/source_dataset/ISO/luminance strata",
        "selected_ids_sha256": hashlib.sha256(
            canonical_json([rows[index]["record"]["id"] for index in selected])
        ).hexdigest(),
        "strata": dict(
            sorted(
                collections.Counter(
                    "/".join(map(str, reconstruction_stratum(rows[index]))) for index in selected
                ).items()
            )
        ),
    }


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


def validate_severity(rows: list[dict[str, Any]], thresholds: GateThresholds, findings: Findings) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_iso: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        record = row["record"]
        group_payload = [record["source_clean_sha256"], record["crop"], record["isp_seed"]]
        group = hashlib.sha256(canonical_json(group_payload)).hexdigest()
        by_group[group].append(row)
        by_iso[int(record["severity_iso"])].append(row)
    complete = 0
    monotonic = 0
    clean_identity_failures = 0
    for group, values in by_group.items():
        by_group_iso: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        for value in values:
            by_group_iso[int(value["record"]["severity_iso"])].append(value)
        isos = sorted(by_group_iso)
        replicate_counts = {iso: len(by_group_iso[iso]) for iso in isos}
        if isos != [12800, 25600, 51200] or len(set(replicate_counts.values())) != 1:
            findings.error(
                "severity_triplet_incomplete", f"{group}: isos={isos}, counts={replicate_counts}"
            )
            continue
        complete += 1
        clean_hashes = {value["metrics"]["clean_content_sha256"] for value in values}
        if len(clean_hashes) != 1:
            clean_identity_failures += 1
            findings.error("severity_clean_mismatch", group)
        rms = [
            float(np.median([value["metrics"]["noise_rms"] for value in by_group_iso[iso]]))
            for iso in isos
        ]
        if rms[0] < rms[1] < rms[2]:
            monotonic += 1
    fraction = monotonic / complete if complete else 0.0
    if complete == 0 or fraction < thresholds.minimum_monotonic_fraction:
        findings.error(
            "severity_not_monotonic",
            f"fraction={fraction:.4f}, required={thresholds.minimum_monotonic_fraction:.4f}",
        )
    summaries = {}
    for iso, values in sorted(by_iso.items()):
        gains = [value["metrics"]["teacher_psnr_gain"] for value in values]
        better = float(np.mean(np.asarray(gains) > 0.0)) if gains else 0.0
        gain_median = float(np.median(gains)) if gains else -math.inf
        shadow = float(np.mean([value["metrics"]["shadow_fraction"] for value in values]))
        structured = float(
            np.median(
                [
                    value["metrics"]["band_fraction"]["medium"]
                    + value["metrics"]["band_fraction"]["coarse"]
                    for value in values
                ]
            )
        ) if values else 0.0
        if better < thresholds.minimum_teacher_better_fraction:
            findings.error("teacher_better_fraction", f"ISO {iso}: {better:.4f}")
        if gain_median < thresholds.minimum_teacher_gain_median_db:
            findings.error("teacher_gain_median", f"ISO {iso}: {gain_median:.4f} dB")
        if shadow < thresholds.minimum_shadow_fraction:
            findings.error("shadow_coverage", f"ISO {iso}: {shadow:.4f}")
        if structured < thresholds.minimum_structured_band_fraction:
            findings.error("structured_noise_missing", f"ISO {iso}: {structured:.6f}")
        summaries[str(iso)] = {
            "records": len(values),
            "noise_rms": finite_summary(value["metrics"]["noise_rms"] for value in values),
            "luma_noise_rms": finite_summary(value["metrics"]["luma_noise_rms"] for value in values),
            "chroma_noise_rms": finite_summary(value["metrics"]["chroma_noise_rms"] for value in values),
            "teacher_psnr_gain_db": finite_summary(gains),
            "teacher_better_fraction": better,
            "mean_shadow_fraction": shadow,
            "median_medium_coarse_band_fraction": structured,
            "mean_rgb_correlation": np.mean(
                [value["metrics"]["rgb_correlation"] for value in values], axis=0
            ).tolist(),
            "mean_band_fraction": {
                name: float(np.mean([value["metrics"]["band_fraction"][name] for value in values]))
                for name in ("fine", "medium", "coarse", "very_coarse")
            },
        }
    return {
        "groups": len(by_group),
        "complete_triplets": complete,
        "monotonic_triplets": monotonic,
        "monotonic_fraction": fraction,
        "minimum_monotonic_fraction": thresholds.minimum_monotonic_fraction,
        "clean_identity_failures": clean_identity_failures,
        "by_iso": summaries,
    }


def calibration_bands(value: np.ndarray, sigmas: tuple[float, float, float]) -> dict[str, np.ndarray]:
    fine_smooth = gaussian_filter(value, sigma=(sigmas[0], sigmas[0], 0.0), mode="reflect")
    medium_smooth = gaussian_filter(value, sigma=(sigmas[1], sigmas[1], 0.0), mode="reflect")
    coarse_smooth = gaussian_filter(value, sigma=(sigmas[2], sigmas[2], 0.0), mode="reflect")
    return {
        "fine": value - fine_smooth,
        "medium": fine_smooth - medium_smooth,
        "coarse": medium_smooth - coarse_smooth,
    }


def low_gradient_mask(value: np.ndarray, quantile: float) -> np.ndarray:
    gray = luminance(value)
    vertical, horizontal = np.gradient(gray)
    gradient = np.hypot(vertical, horizontal)
    valid = (gray > 0.002) & (gray < 0.98)
    if int(valid.sum()) < 512:
        return valid
    return valid & (gradient <= float(np.quantile(gradient[valid], quantile)))


def new_domain_accumulator() -> dict[str, Any]:
    return collections.defaultdict(
        lambda: {
            "pixels": 0,
            "source_payloads": set(),
            "source_scenes": set(),
            "bands": {
                name: {
                    "sum_square": 0.0,
                    "sum": np.zeros(3, dtype=np.float64),
                    "cross": np.zeros((3, 3), dtype=np.float64),
                }
                for name in ("fine", "medium", "coarse")
            },
        }
    )


def accumulate_domain_stats(
    accumulator: dict[str, Any],
    noisy: np.ndarray,
    clean: np.ndarray,
    bins: np.ndarray,
    sigmas: tuple[float, float, float],
    low_gradient_quantile: float,
    bias_removal_sigma: float,
    shadow_luminance_max: float,
    *,
    source_payload: str | None = None,
    source_scene: str | None = None,
) -> None:
    residual = noisy.astype(np.float32) - clean.astype(np.float32)
    residual -= gaussian_filter(
        residual,
        sigma=(bias_removal_sigma, bias_removal_sigma, 0.0),
        mode="reflect",
    )
    bands = calibration_bands(residual, sigmas)
    clean_luma = luminance(clean)
    base_mask = low_gradient_mask(srgb_to_linear(clean), low_gradient_quantile)

    def add_mask(label: str, mask: np.ndarray) -> None:
        count = int(mask.sum())
        if count == 0:
            return
        accumulator[label]["pixels"] += count
        if source_payload is not None:
            accumulator[label]["source_payloads"].add(source_payload)
        if source_scene is not None:
            accumulator[label]["source_scenes"].add(source_scene)
        for name, band in bands.items():
            values = band[mask].astype(np.float64)
            accumulator[label]["bands"][name]["sum_square"] += float(np.square(values).sum())
            accumulator[label]["bands"][name]["sum"] += values.sum(axis=0)
            accumulator[label]["bands"][name]["cross"] += values.T @ values

    for index, (low, high) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
        add_mask(
            f"{index}:{low:.6g}-{high:.6g}",
            base_mask & (clean_luma >= low) & (clean_luma < high),
        )
    add_mask(
        f"shadow:0-{shadow_luminance_max:.6g}",
        base_mask & (clean_luma >= 0.0) & (clean_luma < shadow_luminance_max),
    )


def finalize_domain_stats(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for label, row in sorted(accumulator.items()):
        count = int(row["pixels"])
        band_rows = {}
        energies = {}
        for name, values in row["bands"].items():
            rms = math.sqrt(max(0.0, float(values["sum_square"]) / max(count * 3, 1)))
            mean = values["sum"] / max(count, 1)
            covariance = values["cross"] / max(count, 1) - np.outer(mean, mean)
            scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
            correlation = covariance / np.outer(scale, scale)
            correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(correlation, 1.0)
            energies[name] = rms * rms
            band_rows[name] = {"rms": rms, "rgb_correlation": correlation.tolist()}
        total = sum(energies.values())
        for name in band_rows:
            band_rows[name]["energy_fraction"] = energies[name] / total if total > 0.0 else 0.0
        result[label] = {
            "pixels": count,
            "support": {
                "distinct_source_payloads": len(row["source_payloads"]),
                "distinct_scenes": len(row["source_scenes"]),
            },
            "bands": band_rows,
        }
    return result


def compare_domain_stats(
    synthetic: Mapping[str, Any],
    reference: Mapping[str, Any],
    support_thresholds: CalibrationSupportThresholds,
    thresholds: GateThresholds,
    findings: Findings,
) -> dict[str, Any]:
    comparisons = {}
    supported_bins = 0
    passing_bins = 0
    failing_bins = 0
    inconclusive_bins = 0
    shadow_label = f"shadow:0-{support_thresholds.shadow_luminance_max:.6g}"
    for label in sorted(set(synthetic) | set(reference)):
        left = synthetic.get(label)
        right = reference.get(label)
        if left is None or right is None:
            findings.warn("calibration_luminance_bin_unmatched", label)
            if label == shadow_label:
                findings.error("calibration_shadow_missing", label)
            continue
        log_errors = {}
        correlation_errors = {}
        raw_failures = []
        for name in ("fine", "medium", "coarse"):
            synthetic_rms = float(left["bands"][name]["rms"])
            reference_rms = float(right["bands"][name]["rms"])
            error = abs(math.log(max(synthetic_rms, 1e-12) / max(reference_rms, 1e-12)))
            log_errors[name] = error
            if error > thresholds.maximum_calibration_log_rms_error:
                raw_failures.append(
                    {
                        "code": "calibration_log_rms_mismatch",
                        "band": name,
                        "value": error,
                        "maximum": thresholds.maximum_calibration_log_rms_error,
                    }
                )
            synthetic_correlation = np.asarray(
                left["bands"][name]["rgb_correlation"], dtype=np.float64
            )
            reference_correlation = np.asarray(
                right["bands"][name]["rgb_correlation"], dtype=np.float64
            )
            correlation_error = float(
                np.max(np.abs(synthetic_correlation - reference_correlation))
            )
            correlation_errors[name] = correlation_error
            if correlation_error > thresholds.maximum_calibration_correlation_error:
                raw_failures.append(
                    {
                        "code": "calibration_correlation_mismatch",
                        "band": name,
                        "value": correlation_error,
                        "maximum": thresholds.maximum_calibration_correlation_error,
                    }
                )
        band_l1 = sum(
            abs(
                float(left["bands"][name]["energy_fraction"])
                - float(right["bands"][name]["energy_fraction"])
            )
            for name in ("fine", "medium", "coarse")
        )
        if band_l1 > thresholds.maximum_calibration_band_l1:
            raw_failures.append(
                {
                    "code": "calibration_band_l1_mismatch",
                    "band": None,
                    "value": band_l1,
                    "maximum": thresholds.maximum_calibration_band_l1,
                }
            )
        reference_support = right.get("support", {})
        payloads = int(reference_support.get("distinct_source_payloads", 0))
        scenes = int(reference_support.get("distinct_scenes", 0))
        minimum_pixels = min(int(left["pixels"]), int(right["pixels"]))
        support_failures = []
        if minimum_pixels < support_thresholds.minimum_pixels:
            support_failures.append("pixels")
        if payloads < support_thresholds.minimum_payloads:
            support_failures.append("source_payloads")
        if scenes < support_thresholds.minimum_scenes:
            support_failures.append("scenes")
        supported = not support_failures
        is_shadow = label == shadow_label
        if supported:
            if not is_shadow:
                supported_bins += 1
            if raw_failures:
                failing_bins += int(not is_shadow)
                for failure in raw_failures:
                    suffix = f"/{failure['band']}" if failure["band"] else ""
                    findings.error(
                        str(failure["code"]),
                        f"{label}{suffix}: {float(failure['value']):.6f}",
                    )
            elif not is_shadow:
                passing_bins += 1
            decision = "fail" if raw_failures else "pass"
        else:
            inconclusive_bins += int(not is_shadow)
            decision = "inconclusive"
            findings.warn(
                "calibration_bin_inconclusive",
                f"{label}: insufficient={support_failures}, pixels={minimum_pixels}, "
                f"payloads={payloads}, scenes={scenes}, raw_failures={len(raw_failures)}",
            )
            if is_shadow:
                findings.error(
                    "calibration_shadow_support_insufficient",
                    f"{label}: insufficient={support_failures}",
                )
        comparisons[label] = {
            "synthetic_pixels": int(left["pixels"]),
            "reference_pixels": int(right["pixels"]),
            "reference_support": {
                "distinct_source_payloads": payloads,
                "distinct_scenes": scenes,
            },
            "support_failures": support_failures,
            "decision": decision,
            "log_rms_error": log_errors,
            "band_energy_fraction_l1": band_l1,
            "rgb_correlation_maximum_error": correlation_errors,
            "raw_threshold_failures": raw_failures,
        }
    shadow = comparisons.get(shadow_label)
    if shadow is None:
        findings.error("calibration_shadow_missing", shadow_label)
    elif shadow["decision"] == "fail":
        findings.error(
            "calibration_shadow_distribution_mismatch",
            f"{shadow_label}: raw_failures={len(shadow['raw_threshold_failures'])}",
        )
    if supported_bins == 0:
        findings.error(
            "calibration_no_supported_bins",
            "no narrow luminance bin met pixel, payload, and scene support",
        )
    return {
        "eligible_bins": supported_bins,
        "supported_bins": supported_bins,
        "passing_bins": passing_bins,
        "failing_bins": failing_bins,
        "inconclusive_bins": inconclusive_bins,
        "mandatory_shadow_label": shadow_label,
        "mandatory_shadow_decision": shadow["decision"] if shadow else "missing",
        "thresholds": {
            "minimum_pixels": support_thresholds.minimum_pixels,
            "minimum_reference_payloads": support_thresholds.minimum_payloads,
            "minimum_reference_scenes": support_thresholds.minimum_scenes,
            "maximum_log_rms_error": thresholds.maximum_calibration_log_rms_error,
            "maximum_band_energy_fraction_l1": thresholds.maximum_calibration_band_l1,
            "maximum_rgb_correlation_error": thresholds.maximum_calibration_correlation_error,
        },
        "by_luminance_bin": comparisons,
    }


def validate_calibration(
    rows: list[dict[str, Any]],
    cache_root: Path,
    source_root: Path,
    config: Mapping[str, Any],
    thresholds: GateThresholds,
    holdout: Mapping[str, Any],
    findings: Findings,
) -> dict[str, Any]:
    profile_path = resolve_paper_path(config["data"]["calibration_profile"])
    try:
        profile_document = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        findings.error("calibration_reference_unreadable", f"{profile_path}: {error}")
        return {"status": "failed", "path": str(profile_path)}
    profiles = profile_document.get("profiles") if isinstance(profile_document, dict) else None
    if (
        not isinstance(profiles, dict)
        or profile_document.get("schema_version") != 2
        or profile_document.get("profile_version")
        != "snic_train_linear_post_isp_covariance_v3"
        or profile_document.get("fit_split") != "train"
        or profile_document.get("fit_is_limited_smoke") is not False
        or not all(isinstance(profile, dict) for profile in profiles.values())
    ):
        findings.error("calibration_reference_schema", str(profile_path))
        return {"status": "failed", "path": str(profile_path)}
    extrapolated = [
        iso
        for iso, profile in profiles.items()
        if not str(profile.get("source", "")).startswith("observed")
    ]
    if extrapolated:
        findings.warn(
            "extrapolated_calibration_profiles",
            f"ISO profiles require visual acceptance or independent captures: {sorted(extrapolated)}",
        )

    source_manifest = resolve_paper_path(config["data"]["source_manifest"])
    try:
        source_document = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        findings.error("calibration_source_manifest_unreadable", f"{source_manifest}: {error}")
        return {"status": "failed", "path": str(profile_path)}
    source_records = source_document.get("records") if isinstance(source_document, dict) else None
    if not isinstance(source_records, list):
        findings.error("calibration_source_manifest_schema", str(source_manifest))
        return {"status": "failed", "path": str(profile_path)}
    reference_by_clean = {}
    for record in source_records:
        if not isinstance(record, dict):
            continue
        try:
            iso = int(record.get("iso") or -1)
        except (TypeError, ValueError):
            continue
        if (
            record.get("dataset") == "snic_sony"
            and record.get("split") == "validation"
            and iso == 12800
            and isinstance(record.get("clean"), str)
        ):
            reference_by_clean[str(record["clean"])] = record
    candidates = [
        row
        for row in rows
        if row["record"]["source_dataset"] == "snic_sony"
        and row["record"]["split"] == "validation"
        and int(row["record"]["severity_iso"]) == 12800
        and row["record"]["source_clean"] in reference_by_clean
    ]
    if not candidates:
        findings.error("calibration_no_post_isp_pairs", "no held-out SNIC ISO12800 pairs")
        return {"status": "failed", "path": str(profile_path)}

    from prepare_domain_dataset import ImageSource, srgb_to_linear
    from prepare_synthetic_camera_jpeg import apply_isp

    bins = np.asarray(config["calibration"]["luminance_bins"], dtype=np.float32)
    sigmas = (
        float(config["calibration"]["fine_sigma"]),
        float(config["calibration"]["medium_sigma"]),
        float(config["calibration"]["coarse_sigma"]),
    )
    quantile = float(config["calibration"]["low_gradient_quantile"])
    bias_removal_sigma = float(
        config["calibration"]["post_isp_bias_removal_sigma"]
    )
    support_thresholds = CalibrationSupportThresholds.from_config(config)
    synthetic_accumulator = new_domain_accumulator()
    reference_accumulator = new_domain_accumulator()
    protected_sha = set(holdout["sha256"])
    source_hashes: dict[Path, str] = {}
    compared = 0
    for row in candidates:
        record = row["record"]
        source = reference_by_clean[record["source_clean"]]
        noisy_path = contained_path(source_root, source.get("input"))
        clean_path = contained_path(source_root, source.get("clean"))
        if (
            noisy_path is None
            or clean_path is None
            or not noisy_path.is_file()
            or not clean_path.is_file()
        ):
            findings.error("calibration_source_missing", str(source.get("input")))
            continue
        if noisy_path not in source_hashes:
            source_hashes[noisy_path] = sha256_file(noisy_path)
        if clean_path not in source_hashes:
            source_hashes[clean_path] = sha256_file(clean_path)
        noisy_sha = source_hashes[noisy_path]
        clean_sha = source_hashes[clean_path]
        if noisy_sha in protected_sha or clean_sha in protected_sha:
            findings.error("protected_holdout_calibration_overlap", str(noisy_path))
            continue
        try:
            left, top, _, _ = record["crop"]
            with ImageSource(noisy_path) as noisy_source, ImageSource(clean_path) as clean_source:
                real_noisy = noisy_source.crop(left, top)
                real_clean = clean_source.crop(left, top)
            processed_noisy = apply_isp(srgb_to_linear(real_noisy), record["isp_profile"])
            processed_clean = apply_isp(srgb_to_linear(real_clean), record["isp_profile"])
            input_path = contained_path(cache_root, record["input"])
            clean_cache_path = contained_path(cache_root, record["clean"])
            if input_path is None or clean_cache_path is None:
                raise ValueError("unsafe cached calibration path")
            cached_input = np.load(input_path, allow_pickle=False).astype(np.float32)
            cached_clean = np.load(clean_cache_path, allow_pickle=False).astype(np.float32)
        except Exception as error:  # noqa: BLE001 - retain all usable reference pairs
            findings.error("calibration_pair_failed", f"{record.get('id')}: {error}")
            continue
        accumulate_domain_stats(
            synthetic_accumulator,
            cached_input,
            cached_clean,
            bins,
            sigmas,
            quantile,
            bias_removal_sigma,
            support_thresholds.shadow_luminance_max,
            source_payload=clean_sha,
            source_scene=str(source.get("scene", record["source_scene"])),
        )
        accumulate_domain_stats(
            reference_accumulator,
            processed_noisy,
            processed_clean,
            bins,
            sigmas,
            quantile,
            bias_removal_sigma,
            support_thresholds.shadow_luminance_max,
            source_payload=clean_sha,
            source_scene=str(source.get("scene", record["source_scene"])),
        )
        compared += 1
    synthetic_stats = finalize_domain_stats(synthetic_accumulator)
    reference_stats = finalize_domain_stats(reference_accumulator)
    comparison = compare_domain_stats(
        synthetic_stats,
        reference_stats,
        support_thresholds,
        thresholds,
        findings,
    )
    return {
        "status": "provisional_extrapolated" if extrapolated else "compared",
        "domain": "post-ISP 8-bit sRGB JPEG residual",
        "reference": "held-out SNIC validation ISO12800 pairs transformed by each synthetic record ISP",
        "profile": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
        "reference_pairs": compared,
        "luminance_bins": bins.tolist(),
        "band_sigmas": list(sigmas),
        "bias_removal_sigma": bias_removal_sigma,
        "synthetic": synthetic_stats,
        "real_reference": reference_stats,
        "comparison": comparison,
        "extrapolated_isos": sorted(extrapolated),
    }


def font(size: int) -> ImageFont.ImageFont:
    candidate = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(candidate, size) if candidate.is_file() else ImageFont.load_default()


def uint8_image(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB")


def representative_rows(rows: list[dict[str, Any]], maximum: int = 12) -> list[dict[str, Any]]:
    selected = []
    for iso in (12800, 25600, 51200):
        candidates = [row for row in rows if int(row["record"]["severity_iso"]) == iso]
        candidates.sort(
            key=lambda row: (
                abs(row["metrics"]["mean_luminance"] - 0.12),
                -row["metrics"]["teacher_correction_rms"],
                str(row["record"]["id"]),
            )
        )
        selected.extend(candidates[: max(1, maximum // 3)])
    return selected[:maximum]


def render_contact_sheet(
    rows: list[dict[str, Any]], cache_root: Path, destination: Path
) -> None:
    labels = ("Clean", "Synthetic", "SCUNet", "Noise x4", "Correction x4")
    header = 42
    caption = 50
    canvas = Image.new("RGB", (TILE * len(labels), header + (TILE + caption) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(14)
    body_font = font(11)
    for column, label in enumerate(labels):
        draw.text((column * TILE + 6, 12), label, fill="black", font=title_font)
    for row_index, row in enumerate(rows):
        arrays = {}
        for key in ARRAY_KEYS:
            path = contained_path(cache_root, row["record"].get(key))
            if path is None or not path.is_file():
                raise FileNotFoundError(f"contact sheet tensor is missing: {path}")
            arrays[key] = np.load(path, allow_pickle=False).astype(np.float32)
        top = header + row_index * (TILE + caption)
        panels = (
            arrays["clean"],
            arrays["input"],
            arrays["teacher"],
            np.clip(0.5 + 4.0 * (arrays["input"] - arrays["clean"]), 0.0, 1.0),
            np.clip(0.5 + 4.0 * (arrays["teacher"] - arrays["input"]), 0.0, 1.0),
        )
        for column, panel in enumerate(panels):
            canvas.paste(uint8_image(panel), (column * TILE, top))
        metric = row["metrics"]
        caption_text = (
            f"{row['record']['id']} | ISO {metric['iso']} | {row['record']['split']} | "
            f"luma {metric['mean_luminance']:.3f} | RMS {metric['noise_rms']:.4f} | "
            f"teacher gain {metric['teacher_psnr_gain']:+.2f} dB"
        )
        draw.text((6, top + TILE + 7), caption_text, fill="black", font=body_font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", compress_level=6)
    canvas.save(destination.with_suffix(".jpg"), quality=98, subsampling=0, optimize=True)


def analysis_sha256(value: Mapping[str, Any]) -> str:
    """Return the stable identity reviewed by the human acceptance artifact."""

    return hashlib.sha256(canonical_json(value)).hexdigest()


def active_validator_identity() -> dict[str, Any]:
    """Identify the exact validator implementation and analysis semantics in use."""

    return {
        "name": VALIDATOR_IDENTITY,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "semantic_version": VALIDATOR_SEMANTIC_VERSION,
        "path": str(VALIDATOR_SCRIPT),
        "sha256": sha256_file(VALIDATOR_SCRIPT),
    }


def active_generator_identity() -> dict[str, Any]:
    """Identify the exact generator implementation accepted by this gate."""

    return {
        "name": GENERATOR_IDENTITY,
        "path": str(GENERATOR_SCRIPT),
        "sha256": sha256_file(GENERATOR_SCRIPT),
    }


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def findings_contract_errors(findings: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(findings, dict) or set(findings) != {"errors", "warnings"}:
        return ["findings must contain exactly errors and warnings"]
    for severity in ("errors", "warnings"):
        section = findings.get(severity)
        if not isinstance(section, dict) or set(section) != {"total", "by_code", "examples"}:
            errors.append(f"findings.{severity} has an invalid schema")
            continue
        by_code = section.get("by_code")
        examples = section.get("examples")
        total = section.get("total")
        if not isinstance(by_code, dict) or not all(
            isinstance(key, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for key, value in by_code.items()
        ):
            errors.append(f"findings.{severity}.by_code is invalid")
            continue
        if not isinstance(total, int) or isinstance(total, bool) or total != sum(by_code.values()):
            errors.append(f"findings.{severity}.total does not match by_code")
        if (
            not isinstance(examples, dict)
            or set(examples) != set(by_code)
            or not all(
                isinstance(values, list)
                and values
                and len(values) <= by_code[code]
                and all(isinstance(value, str) for value in values)
                for code, values in examples.items()
            )
        ):
            errors.append(f"findings.{severity}.examples is invalid")
    return errors


def analysis_contract_errors(analysis: Any) -> list[str]:
    """Return analysis schema/semantic inconsistencies without hiding gate findings."""

    if not isinstance(analysis, dict):
        return ["analysis is not an object"]
    errors: list[str] = []
    if set(analysis) != ANALYSIS_KEYS:
        errors.append(
            "analysis keys differ: "
            f"missing={sorted(ANALYSIS_KEYS - set(analysis))}, "
            f"unexpected={sorted(set(analysis) - ANALYSIS_KEYS)}"
        )
    if analysis.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        errors.append("analysis schema_version is not active")
    if analysis.get("semantic_version") != VALIDATOR_SEMANTIC_VERSION:
        errors.append("analysis semantic_version is not active")
    if analysis.get("validator") != active_validator_identity():
        errors.append("analysis validator identity is not active")
    if analysis.get("generator") != active_generator_identity():
        errors.append("analysis generator identity is not active")
    if not isinstance(analysis.get("preprocessing"), str) or not analysis[
        "preprocessing"
    ].strip():
        errors.append("analysis preprocessing identity is invalid")
    loaded = analysis.get("records_loaded")
    measured = analysis.get("records_measured")
    if (
        not isinstance(loaded, int)
        or isinstance(loaded, bool)
        or loaded < 0
        or not isinstance(measured, int)
        or isinstance(measured, bool)
        or measured < 0
        or measured > loaded
    ):
        errors.append("analysis record counts are invalid")
    smoke = analysis.get("smoke")
    if (
        not isinstance(smoke, dict)
        or set(smoke) != {"generation", "calibration"}
        or not all(isinstance(smoke.get(key), bool) for key in smoke)
    ):
        errors.append("analysis smoke flags are invalid")
    manifest = analysis.get("manifest")
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"path", "sha256"}
        or not isinstance(manifest.get("path"), str)
        or not is_sha256(manifest.get("sha256"))
    ):
        errors.append("analysis manifest identity is invalid")
    cache = analysis.get("cache_content")
    if (
        not isinstance(cache, dict)
        or set(cache) != {"files", "bytes", "sha256"}
        or not isinstance(cache.get("files"), int)
        or isinstance(cache.get("files"), bool)
        or cache.get("files", -1) < 0
        or not isinstance(cache.get("bytes"), int)
        or isinstance(cache.get("bytes"), bool)
        or cache.get("bytes", -1) < 0
        or not is_sha256(cache.get("sha256"))
    ):
        errors.append("analysis cache-content identity is invalid")
    contact = analysis.get("contact_sheet")
    if contact is not None and (
        not isinstance(contact, dict)
        or set(contact) != {"png", "png_sha256", "jpg", "jpg_sha256"}
        or not isinstance(contact.get("png"), str)
        or not isinstance(contact.get("jpg"), str)
        or not is_sha256(contact.get("png_sha256"))
        or not is_sha256(contact.get("jpg_sha256"))
    ):
        errors.append("analysis contact-sheet identity is invalid")
    for key in (
        "protected_holdout",
        "sources_and_splits",
        "reconstruction",
        "severity",
        "calibration",
    ):
        if not isinstance(analysis.get(key), dict):
            errors.append(f"analysis {key} must be an object")
    errors.extend(findings_contract_errors(analysis.get("findings")))
    return errors


def report_contract_errors(report: Any) -> list[str]:
    if not isinstance(report, dict):
        return ["report is not an object"]
    errors = analysis_contract_errors(report.get("analysis"))
    analysis = report.get("analysis")
    if isinstance(analysis, dict):
        if report.get("analysis_sha256") != analysis_sha256(analysis):
            errors.append("report analysis_sha256 does not match analysis")
        for key in ANALYSIS_MIRROR_KEYS:
            if report.get(key) != analysis.get(key):
                errors.append(f"report {key} differs from analysis.{key}")
    return errors


def expected_visual_acceptance(analysis: Mapping[str, Any]) -> dict[str, str]:
    contact = analysis.get("contact_sheet")
    if not isinstance(contact, dict):
        return {}
    return {
        "manifest_sha256": str(analysis["manifest"]["sha256"]),
        "cache_content_sha256": str(analysis["cache_content"]["sha256"]),
        "contact_png_sha256": str(contact["png_sha256"]),
        "contact_jpg_sha256": str(contact["jpg_sha256"]),
        "analysis_sha256": analysis_sha256(analysis),
    }


def validate_visual_acceptance(
    path: Path, expected: Mapping[str, str]
) -> tuple[bool, dict[str, Any]]:
    """Require a strict, current human signoff without changing analysis identity."""

    result: dict[str, Any] = {"path": str(path)}
    if not path.is_file():
        return False, {**result, "status": "missing"}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, {**result, "status": "invalid", "reason": str(error)}
    if not isinstance(document, dict):
        return False, {**result, "status": "invalid", "reason": "not an object"}
    unexpected = sorted(set(document) - VISUAL_ACCEPTANCE_KEYS)
    missing = sorted(VISUAL_ACCEPTANCE_KEYS - set(document))
    if missing or unexpected:
        return False, {
            **result,
            "status": "invalid",
            "reason": f"missing={missing}, unexpected={unexpected}",
        }
    if document["schema_version"] != 2 or document["decision"] != "accepted":
        return False, {
            **result,
            "status": "invalid",
            "reason": "schema_version must be 2 and decision must be accepted",
        }
    reviewer = document.get("reviewer")
    accepted_at = document.get("accepted_at")
    if not isinstance(reviewer, str) or not reviewer.strip():
        return False, {**result, "status": "invalid", "reason": "reviewer is empty"}
    try:
        parsed_time = datetime.fromisoformat(str(accepted_at).replace("Z", "+00:00"))
    except ValueError:
        parsed_time = None
    if parsed_time is None or parsed_time.utcoffset() is None:
        return False, {
            **result,
            "status": "invalid",
            "reason": "accepted_at must be an ISO-8601 timestamp with timezone",
        }
    reviewed_report_sha = document.get("reviewed_report_sha256")
    if not is_sha256(reviewed_report_sha):
        return False, {
            **result,
            "status": "invalid",
            "reason": "reviewed_report_sha256 must be a lowercase SHA-256",
        }
    mismatches = {
        key: {"expected": value, "actual": document.get(key)}
        for key, value in expected.items()
        if document.get(key) != value
    }
    if mismatches:
        return False, {
            **result,
            "status": "stale",
            "reviewer": reviewer,
            "accepted_at": accepted_at,
            "mismatches": mismatches,
        }
    return True, {
        **result,
        "status": "accepted",
        "reviewer": reviewer,
        "accepted_at": accepted_at,
        "reviewed_report_sha256": reviewed_report_sha,
        "sha256": sha256_file(path),
    }


def write_visual_acceptance(
    report: Mapping[str, Any],
    path: Path,
    reviewer: str,
    *,
    reviewed_report_sha256: str,
) -> None:
    """Write a signoff for a clean, fully rendered validation pass."""

    if not reviewer.strip():
        raise ValueError("reviewer must not be empty")
    if not is_sha256(reviewed_report_sha256):
        raise ValueError("reviewed_report_sha256 must be a lowercase SHA-256")
    contract_errors = report_contract_errors(report)
    if contract_errors:
        raise RuntimeError(f"cannot accept an inconsistent gate report: {contract_errors}")
    if int(report["findings"]["errors"]["total"]) != 0:
        raise RuntimeError("cannot accept a gate report containing hard errors")
    expected = expected_visual_acceptance(report["analysis"])
    if len(expected) != 5:
        raise RuntimeError("cannot accept a report without both contact-sheet artifacts")
    atomic_json(
        path,
        {
            "schema_version": 2,
            "decision": "accepted",
            "reviewer": reviewer.strip(),
            "accepted_at": datetime.now(timezone.utc).isoformat(),
            "reviewed_report_sha256": reviewed_report_sha256,
            **expected,
        },
    )


def validate_cache(
    config: dict[str, Any],
    *,
    reconstructor: Reconstructor | None,
    output_dir: Path | None = None,
    visual_acceptance_path: Path | None = None,
) -> dict[str, Any]:
    thresholds = GateThresholds.from_config(config)
    findings = Findings()
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    source_root = resolve_paper_path(config["data"]["source_root"])
    manifest_path = resolve_paper_path(config["data"]["manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("records"), list):
        raise ValueError(f"Synthetic manifest must be an object containing records: {manifest_path}")
    records = document["records"]
    validate_manifest_provenance(document, manifest_path, config, findings)
    holdout = protected_holdout(config, document, findings)
    for record in records:
        if isinstance(record, dict):
            validate_record_semantics(record, findings)
        else:
            findings.error("record_not_object", repr(record))
    valid_records = [
        record for record in records if isinstance(record, dict) and "severity_iso" in record
    ]
    source_report = validate_sources_and_splits(valid_records, source_root, holdout, findings)
    rows = []
    for record in valid_records:
        arrays = {
            key: load_checked_array(cache_root, record, key, findings) for key in ARRAY_KEYS
        }
        if all(value is not None for value in arrays.values()):
            rows.append(record_metrics(record, arrays))
    if not rows:
        findings.error("no_valid_rows", str(manifest_path))
    gate_config = config.get("gate", {})
    reconstruction = validate_reconstruction(
        rows,
        cache_root,
        reconstructor,
        thresholds,
        int(gate_config.get("reconstruction_samples", 96)),
        int(config.get("project", {}).get("seed", 0)),
        findings,
    )
    severity = validate_severity(rows, thresholds, findings) if rows else {}
    calibration = (
        validate_calibration(
            rows,
            cache_root,
            source_root,
            config,
            thresholds,
            holdout,
            findings,
        )
        if rows
        else {"status": "not_run"}
    )
    destination = output_dir or resolve_paper_path(config["data"]["cache_root"]).parent
    destination.mkdir(parents=True, exist_ok=True)
    contact_path = destination / "synthetic_camera_jpeg_contact_sheet.png"
    contact_rendered = False
    if rows:
        try:
            render_contact_sheet(
                representative_rows(rows, int(gate_config.get("contact_rows", 12))),
                cache_root,
                contact_path,
            )
            contact_rendered = True
        except Exception as error:  # noqa: BLE001 - keep the gate report actionable
            findings.error("contact_sheet_failed", str(error))
    contact_sheet = (
        {
            "png": str(contact_path),
            "png_sha256": sha256_file(contact_path),
            "jpg": str(contact_path.with_suffix(".jpg")),
            "jpg_sha256": sha256_file(contact_path.with_suffix(".jpg")),
        }
        if contact_rendered
        and contact_path.is_file()
        and contact_path.with_suffix(".jpg").is_file()
        else None
    )
    smoke = {
        "generation": bool(
            isinstance(document.get("generation_environment"), dict)
            and document["generation_environment"].get("limited_smoke")
        ),
        "calibration": bool(document.get("calibration_is_limited_smoke")),
    }
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "semantic_version": VALIDATOR_SEMANTIC_VERSION,
        "validator": active_validator_identity(),
        "generator": active_generator_identity(),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "preprocessing": document.get("preprocessing"),
        "records_loaded": len(records),
        "records_measured": len(rows),
        "cache_content": cache_content_identity(cache_root, valid_records),
        "smoke": smoke,
        "protected_holdout": holdout,
        "sources_and_splits": source_report,
        "reconstruction": reconstruction,
        "severity": severity,
        "calibration": calibration,
        "findings": findings.report(),
        "contact_sheet": contact_sheet,
    }
    contract_errors = analysis_contract_errors(analysis)
    if contract_errors:
        raise RuntimeError(f"validator produced an inconsistent analysis: {contract_errors}")
    analysis_digest = analysis_sha256(analysis)
    signoff_path = visual_acceptance_path or destination / "visual_acceptance.json"
    signoff_valid, signoff = validate_visual_acceptance(
        signoff_path, expected_visual_acceptance(analysis)
    )
    if findings.failed:
        status = "failed"
    elif signoff_valid:
        status = "accepted"
    else:
        status = "pending_visual_review"
    report = {
        "schema_version": 2,
        "status": status,
        "release_gate_passed": status == "accepted",
        "manifest": analysis["manifest"],
        "cache_content": analysis["cache_content"],
        "smoke": smoke,
        "contact_sheet": contact_sheet,
        "analysis_sha256": analysis_digest,
        "analysis": analysis,
        "preprocessing": analysis["preprocessing"],
        "records_loaded": analysis["records_loaded"],
        "records_measured": analysis["records_measured"],
        "protected_holdout": analysis["protected_holdout"],
        "sources_and_splits": analysis["sources_and_splits"],
        "reconstruction": analysis["reconstruction"],
        "severity": analysis["severity"],
        "calibration": analysis["calibration"],
        "findings": analysis["findings"],
        "visual_acceptance": signoff,
        "reviewed_report_sha256": (
            signoff.get("reviewed_report_sha256") if signoff_valid else None
        ),
        "decision": (
            "Release gate passed for this exact cache and visual artifact set."
            if status == "accepted"
            else "Training remains blocked until all hard checks pass and the exact contact "
            "sheet is accepted. Extrapolated ISO profiles remain provisional."
        ),
    }
    contract_errors = report_contract_errors(report)
    if contract_errors:
        raise RuntimeError(f"validator produced an inconsistent report: {contract_errors}")
    atomic_json(destination / "synthetic_camera_jpeg_gate_report.json", report)
    return report


def load_reconstructor(config: dict[str, Any]) -> Reconstructor | None:
    try:
        module = importlib.import_module("prepare_synthetic_camera_jpeg")
        synthesize_pair = getattr(module, "synthesize_pair")
    except (ImportError, AttributeError):
        return None

    source_root = resolve_paper_path(config["data"]["source_root"])

    def reconstruct(record: dict[str, Any]) -> Mapping[str, np.ndarray]:
        source_path = contained_path(source_root, record["source_clean"])
        if source_path is None or not source_path.is_file():
            raise FileNotFoundError(record["source_clean"])
        image_source = getattr(module, "ImageSource", None)
        if image_source is None:
            from prepare_domain_dataset import ImageSource as image_source
        left, top, width, height = record["crop"]
        if (width, height) != (TILE, TILE):
            raise ValueError(f"invalid reconstruction crop: {record['crop']!r}")
        with image_source(source_path) as source:
            clean_source = source.crop(left, top, TILE)
        noisy, clean, realized = synthesize_pair(
            clean_source,
            record["noise_profile"],
            int(record["noise_seed"]),
            config["synthesis"],
            isp_seed=int(record["isp_seed"]),
        )
        realized_isp = realized.get("isp_profile") if isinstance(realized, dict) else None
        if realized_isp != record["isp_profile"]:
            raise ValueError(
                f"realized ISP profile differs: {realized_isp!r} != {record['isp_profile']!r}"
            )
        if realized.get("noise_profile") != record["noise_profile"]:
            raise ValueError("realized noise profile differs from manifest")
        if realized.get("noise_seed") != record["noise_seed"]:
            raise ValueError("realized noise seed differs from manifest")
        if realized.get("isp_seed") != record["isp_seed"]:
            raise ValueError("realized ISP seed differs from manifest")
        if realized.get("post_isp_match") != record["post_isp_match"]:
            raise ValueError("realized post-ISP diagnostics differ from manifest")
        return {"input": noisy, "clean": clean}

    return reconstruct


def report_artifact_path(report_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"gate report has no {label} path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (report_path.parent / path).resolve()


def validate_report_for_acceptance(
    config: Mapping[str, Any],
    report_path: Path,
    report: Mapping[str, Any],
    *,
    expected_report_sha256: str,
) -> None:
    """Validate the exact already-rendered pending report before human signoff."""

    if not is_sha256(expected_report_sha256):
        raise RuntimeError("--accept-report-sha256 must be a lowercase SHA-256")
    actual_report_sha = sha256_file(report_path)
    if actual_report_sha != expected_report_sha256:
        raise RuntimeError(
            "visual acceptance refused because the reviewed report changed: "
            f"expected={expected_report_sha256}, actual={actual_report_sha}"
        )
    if (
        report.get("schema_version") != 2
        or report.get("status") != "pending_visual_review"
        or report.get("release_gate_passed") is not False
        or report.get("reviewed_report_sha256") is not None
    ):
        raise RuntimeError(
            "visual acceptance requires an unsigned pending schema-2 gate report"
        )
    contract_errors = report_contract_errors(report)
    if contract_errors:
        raise RuntimeError(f"visual acceptance refused for inconsistent report: {contract_errors}")
    analysis = report["analysis"]
    if analysis["findings"]["errors"]["total"] != 0:
        raise RuntimeError("visual acceptance refused because validation has hard errors")
    if analysis["smoke"] != {"generation": False, "calibration": False}:
        raise RuntimeError("visual acceptance refused for smoke/provisional data")

    manifest_path = resolve_paper_path(config["data"]["manifest"])
    manifest_identity = analysis["manifest"]
    if report_artifact_path(report_path, manifest_identity["path"], "manifest") != (
        manifest_path.resolve()
    ):
        raise RuntimeError("visual acceptance report references a different manifest")
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_identity["sha256"]:
        raise RuntimeError("visual acceptance refused because the manifest changed")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = document.get("records") if isinstance(document, dict) else None
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise RuntimeError("visual acceptance manifest records are invalid")
    if analysis["records_loaded"] != len(records) or analysis["records_measured"] != len(records):
        raise RuntimeError(
            "visual acceptance requires every manifest record to be loaded and measured"
        )
    if analysis["preprocessing"] != document.get("preprocessing") or analysis[
        "preprocessing"
    ] != config.get("project", {}).get("preprocessing_version"):
        raise RuntimeError("visual acceptance preprocessing identity changed")

    config_path = Path(str(config.get("_config_path", ""))).resolve()
    if (
        not config_path.is_file()
        or document.get("config") != str(config_path)
        or document.get("config_sha256") != sha256_file(config_path)
    ):
        raise RuntimeError("visual acceptance refused because the active config changed")

    cache_root = resolve_paper_path(config["data"]["cache_root"])
    actual_cache = cache_content_identity(cache_root, records)
    if actual_cache != analysis["cache_content"]:
        raise RuntimeError("visual acceptance refused because cached arrays changed")
    if actual_cache["files"] != len(records) * len(ARRAY_KEYS):
        raise RuntimeError("visual acceptance cache identity did not cover every record array")

    contact = analysis.get("contact_sheet")
    if not isinstance(contact, dict):
        raise RuntimeError("visual acceptance requires both rendered contact sheets")
    for extension in ("png", "jpg"):
        artifact = report_artifact_path(
            report_path, contact.get(extension), f"contact-sheet {extension}"
        )
        if not artifact.is_file() or sha256_file(artifact) != contact.get(
            f"{extension}_sha256"
        ):
            raise RuntimeError(
                f"visual acceptance refused because contact-sheet {extension} changed"
            )


def accept_existing_report(
    config: Mapping[str, Any],
    *,
    destination: Path,
    signoff_path: Path,
    reviewer: str,
    expected_report_sha256: str,
) -> dict[str, Any]:
    """Sign and finalize an exact prior validation report without re-rendering."""

    report_path = destination / "synthetic_camera_jpeg_gate_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"run the first validation pass before acceptance: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise RuntimeError("pending synthetic gate report is not a JSON object")
    validate_report_for_acceptance(
        config,
        report_path,
        report,
        expected_report_sha256=expected_report_sha256,
    )
    write_visual_acceptance(
        report,
        signoff_path,
        reviewer,
        reviewed_report_sha256=expected_report_sha256,
    )
    valid, signoff = validate_visual_acceptance(
        signoff_path, expected_visual_acceptance(report["analysis"])
    )
    if not valid or signoff.get("reviewed_report_sha256") != expected_report_sha256:
        raise RuntimeError("new visual acceptance did not bind the reviewed report")
    accepted = dict(report)
    accepted.update(
        {
            "status": "accepted",
            "release_gate_passed": True,
            "visual_acceptance": signoff,
            "reviewed_report_sha256": expected_report_sha256,
            "decision": "Release gate passed for this exact cache and visual artifact set.",
        }
    )
    contract_errors = report_contract_errors(accepted)
    if contract_errors:
        raise RuntimeError(f"accepted report is inconsistent: {contract_errors}")
    atomic_json(report_path, accepted)
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs/synthetic_camera_jpeg_gate.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--accept-visual",
        action="store_true",
        help="Accept an exact prior report and its already-rendered contact sheets.",
    )
    parser.add_argument("--reviewer", help="Non-empty reviewer identity for --accept-visual.")
    parser.add_argument(
        "--accept-report-sha256",
        help="Exact SHA-256 printed by the already-inspected first validation pass.",
    )
    args = parser.parse_args()
    if args.accept_visual and (
        not args.reviewer
        or not args.reviewer.strip()
        or not args.accept_report_sha256
    ):
        parser.error("--accept-visual requires --reviewer and --accept-report-sha256")
    if args.accept_report_sha256 and not args.accept_visual:
        parser.error("--accept-report-sha256 requires --accept-visual")
    config = load_config(args.config)
    destination = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve_paper_path(config["data"]["cache_root"]).parent
    )
    signoff_path = destination / "visual_acceptance.json"
    if args.accept_visual:
        report = accept_existing_report(
            config,
            destination=destination,
            signoff_path=signoff_path,
            reviewer=args.reviewer,
            expected_report_sha256=args.accept_report_sha256.lower(),
        )
    else:
        report = validate_cache(
            config,
            reconstructor=load_reconstructor(config),
            output_dir=destination,
            visual_acceptance_path=signoff_path,
        )
    report_path = destination / "synthetic_camera_jpeg_gate_report.json"
    print(json.dumps({
        "status": report["status"],
        "release_gate_passed": report["release_gate_passed"],
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "reviewed_report_sha256": report.get("reviewed_report_sha256"),
        "visual_acceptance": str(signoff_path),
        "records": report["records_measured"],
        "findings": report["findings"],
    }, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
