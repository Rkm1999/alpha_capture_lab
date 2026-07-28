#!/usr/bin/env python3
"""Offline contract tests for the synthetic camera-JPEG cache gate."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from common import sha256_file
from fit_synthetic_noise_profiles import (
    global_rgb_correlation_from_moments,
    positive_correlation,
)
from prepare_synthetic_camera_jpeg import load_profiles, synthesize_pair
from validate_synthetic_camera_jpeg import (
    ANALYSIS_SCHEMA_VERSION,
    VISUAL_ACCEPTANCE_KEYS,
    CalibrationSupportThresholds,
    Findings,
    GateThresholds,
    accept_existing_report,
    active_generator_identity,
    active_validator_identity,
    analysis_sha256,
    compare_domain_stats,
    load_reconstructor,
    record_metrics,
    validate_cache,
    validate_calibration,
    validate_reconstruction,
    validate_severity,
    validate_sources_and_splits,
)


def save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB").save(path)


def save_npy(path: Path, value: np.ndarray, dtype: np.dtype) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value.astype(dtype))
    return sha256_file(path)


def source_pattern(offset: float) -> np.ndarray:
    y, x = np.mgrid[:256, :256]
    base = 0.045 + offset + 0.12 * x / 255.0 + 0.07 * y / 255.0
    texture = 0.006 * np.sin(x / 13.0) * np.cos(y / 17.0)
    return np.stack(
        (base * 1.04 + texture, base, base * 0.93 - texture), axis=-1
    ).clip(0.0, 1.0).astype(np.float32)


def severity_profile(iso: int) -> dict:
    scale = {12800: 1.0, 25600: 2.0, 51200: 4.0}[iso]
    variance_scale = scale * scale
    band_scale = {
        "fine": (0.0060 * scale, 0.0080 * scale),
        "medium": (0.0035 * scale, 0.0045 * scale),
        "coarse": (0.0025 * scale, 0.0030 * scale),
        "very_coarse": (0.0015 * scale, 0.0020 * scale),
    }
    return {
        "iso": iso,
        "source": "observed" if iso == 12800 else "train_only_snic_log2_extrapolation",
        "shot_scale": [0.00018 * variance_scale] * 3,
        "read_variance": [0.000012 * variance_scale] * 3,
        "rgb_correlation": [[1.0, 0.12, 0.04], [0.12, 1.0, 0.08], [0.04, 0.08, 1.0]],
        "medium_luma_rms": 0.0025 * scale,
        "medium_chroma_rms": 0.0015 * scale,
        "coarse_luma_rms": 0.0018 * scale,
        "coarse_chroma_rms": 0.0012 * scale,
        "row_luma_rms": 0.0005 * scale,
        "row_chroma_rms": 0.0003 * scale,
        "column_luma_rms": 0.0004 * scale,
        "column_chroma_rms": 0.00025 * scale,
        "medium_field_sigma": 2.5,
        "coarse_field_sigma": 8.0,
        "row_column_smoothing_sigma": 2.0,
        "shadow_multiplier": 1.6,
        "post_isp_band_targets": {
            "luminance_bins": [0.0, 0.5, 1.0],
            **{
                name: {
                    "luma_rms": [luma_rms, luma_rms],
                    "chroma_rms": [chroma_rms, chroma_rms],
                    "rgb_covariance": [
                        (np.eye(3) * max(luma_rms, chroma_rms) ** 2).tolist()
                        for _ in range(2)
                    ],
                    "rgb_correlation": [np.eye(3).tolist() for _ in range(2)],
                }
                for name, (luma_rms, chroma_rms) in band_scale.items()
            },
        },
    }


def fixture(root: Path) -> tuple[dict, Path]:
    source_root = root / "sources"
    cache_root = root / "cache"
    output_root = root / "gate"
    source_manifest = root / "source_manifest.json"
    calibration_profile = root / "calibration_profiles.json"
    teacher_checkpoint = root / "teacher.pth"
    config_path = root / "config.yaml"
    manifest_path = cache_root / "manifest.json"
    holdout_path = root / "sony_holdout.png"

    teacher_checkpoint.write_bytes(b"frozen SCUNet checkpoint")
    save_rgb(holdout_path, source_pattern(0.16))
    exclusions = [{"path": str(holdout_path.resolve()), "sha256": sha256_file(holdout_path)}]

    source_records = []
    decoded_sources = {}
    for scene_number, split in ((1, "train"), (2, "validation")):
        clean = source_pattern(0.012 * scene_number)
        clean_relative = f"clean/scene_{scene_number}.png"
        noisy_relative = f"noisy/scene_{scene_number}_iso12800.png"
        clean_path = source_root / clean_relative
        noisy_path = source_root / noisy_relative
        save_rgb(clean_path, clean)
        rng = np.random.default_rng(700 + scene_number)
        real_noisy = np.clip(clean + rng.normal(0.0, 0.025, clean.shape), 0.0, 1.0)
        save_rgb(noisy_path, real_noisy)
        with Image.open(clean_path) as image:
            decoded_sources[scene_number] = (
                np.asarray(image.convert("RGB"), dtype=np.float32)[:192, :192] / 255.0
            )
        source_records.append(
            {
                "dataset": "snic_sony",
                "scene": f"scene_{scene_number}",
                "split": split,
                "iso": 12800,
                "input": noisy_relative,
                "clean": clean_relative,
            }
        )
    source_manifest.write_text(
        json.dumps({"schema_version": 1, "records": source_records}), encoding="utf-8"
    )

    synthesis = {
        "shadow_exponent": 1.5,
        "white_balance_range": [0.98, 1.02],
        "color_matrix_jitter": 0.01,
        "tone_gamma_range": [0.98, 1.02],
        "sharpen_amount_range": [0.0, 0.25],
        "sharpen_sigma_range": [0.65, 1.10],
        "jpeg_quality_range": [90, 98],
        "jpeg_subsampling": [0, 2],
        "row_column_component_scale": 0.15,
        "post_isp_gain_bounds": [0.05, 4.0],
        "post_isp_gain_smoothing_sigma": 2.0,
        "post_isp_match_iterations": 3,
    }
    config_file = {
        "project": {
            "name": "synthetic-gate-test",
            "preprocessing_version": "synthetic_camera_jpeg_linear_post_isp_covariance_v3",
            "seed": 97,
        },
        "data": {
            "source_manifest": str(source_manifest),
            "source_root": str(source_root),
            "calibration_profile": str(calibration_profile),
            "cache_root": str(cache_root),
            "manifest": str(manifest_path),
            "holdout_exclusions": exclusions,
        },
        "calibration": {
            "luminance_bins": [0.0, 0.5, 1.0],
            "fine_sigma": 1.0,
            "medium_sigma": 4.0,
            "coarse_sigma": 12.0,
            "post_isp_jpeg_subsampling": [0, 2],
            "post_isp_bias_removal_sigma": 32.0,
            "minimum_pixels_per_bin": 1,
            "low_gradient_quantile": 0.9,
        },
        "synthesis": synthesis,
        "teacher": {"checkpoint": str(teacher_checkpoint)},
        "gate": {
            "reconstruction_samples": 4,
            "contact_rows": 6,
            "minimum_monotonic_fraction": 1.0,
            "minimum_teacher_better_fraction": 1.0,
            "minimum_teacher_gain_median_db": 0.0,
            "minimum_shadow_fraction": 0.0,
            "minimum_structured_band_fraction": 0.0,
            "maximum_calibration_log_rms_error": 50.0,
            "maximum_calibration_band_l1": 3.0,
            "maximum_calibration_correlation_error": 2.0,
            "minimum_calibration_reference_payloads_per_bin": 1,
            "minimum_calibration_reference_scenes_per_bin": 1,
            "calibration_shadow_luminance_max": 0.25,
        },
    }
    config_path.write_text(yaml.safe_dump(config_file, sort_keys=False), encoding="utf-8")
    config = {**config_file, "_config_path": str(config_path.resolve())}

    profiles = {str(iso): severity_profile(iso) for iso in (12800, 25600, 51200)}
    calibration_basis = {
        "source": "offline fixture paired train split only",
        "fit_split": "train",
        "target_camera_holdout_used": False,
        "fit_support": {
            "pairs": 48,
            "pairs_by_iso": {"12800": 12, "25600": 12, "51200": 12},
            "unique_scenes": 5,
            "unique_clean_payloads": 12,
        },
        "post_isp_distribution": {
            key: synthesis[key]
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
    }
    profile_document = {
        "schema_version": 2,
        "profile_version": "snic_train_linear_post_isp_covariance_v3",
        "preprocessing": config["project"]["preprocessing_version"],
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(source_manifest),
        "fit_split": "train",
        "fit_is_limited_smoke": False,
        "holdout_exclusions": exclusions,
        "calibration_basis": calibration_basis,
        "fitting_environment": {
            "script": str(
                Path(__file__).with_name("fit_synthetic_noise_profiles.py").resolve()
            ),
            "script_sha256": sha256_file(
                Path(__file__).with_name("fit_synthetic_noise_profiles.py").resolve()
            ),
        },
        "profiles": profiles,
    }
    calibration_profile.write_text(json.dumps(profile_document), encoding="utf-8")

    records = []
    for scene_number, split in ((1, "train"), (2, "validation")):
        raw_clean = decoded_sources[scene_number]
        source_relative = f"clean/scene_{scene_number}.png"
        source_sha = sha256_file(source_root / source_relative)
        isp_seed = 2000 + scene_number
        for iso_index, iso in enumerate((12800, 25600, 51200)):
            noise_seed = 10000 + scene_number * 10 + iso_index
            noisy, clean, realized = synthesize_pair(
                raw_clean,
                profiles[str(iso)],
                noise_seed,
                synthesis,
                isp_seed=isp_seed,
            )
            teacher = np.clip(clean + 0.08 * (noisy - clean), 0.0, 1.0)
            identifier = f"scene{scene_number}_iso{iso}"
            base = Path(split) / "synthetic_camera_jpeg" / identifier
            paths = {
                key: str(base.with_name(base.name + f"_{key}.npy"))
                for key in ("input", "clean", "teacher")
            }
            hashes = {
                "input": save_npy(cache_root / paths["input"], noisy, np.dtype(np.float32)),
                "clean": save_npy(cache_root / paths["clean"], clean, np.dtype(np.float32)),
                "teacher": save_npy(
                    cache_root / paths["teacher"], teacher, np.dtype(np.float16)
                ),
            }
            records.append(
                {
                    "id": identifier,
                    "dataset": "synthetic_camera_jpeg",
                    "source_dataset": "snic_sony",
                    "source_scene": f"scene_{scene_number}",
                    "scene": f"snic_sony:scene_{scene_number}",
                    "split": split,
                    "source_clean": source_relative,
                    "source_clean_sha256": source_sha,
                    "crop": [0, 0, 192, 192],
                    "crop_mean_luminance": float(
                        (raw_clean @ np.asarray([0.2126, 0.7152, 0.0722])).mean()
                    ),
                    "severity_iso": iso,
                    "generation_seed": noise_seed,
                    "noise_seed": noise_seed,
                    "isp_seed": isp_seed,
                    "noise_profile": realized["noise_profile"],
                    "isp_profile": realized["isp_profile"],
                    "post_isp_match": realized["post_isp_match"],
                    "supervision": "synthetic_paired",
                    "gt_weight": 1.0,
                    "kd_weight": 0.7,
                    **paths,
                    "array_sha256": hashes,
                }
            )

    manifest = {
        "schema_version": 2,
        "purpose": "validator fixture",
        "preprocessing": config["project"]["preprocessing_version"],
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "calibration_profile": str(calibration_profile.resolve()),
        "calibration_profile_sha256": sha256_file(calibration_profile),
        "calibration_fit_split": "train",
        "calibration_is_limited_smoke": False,
        "calibration_basis": calibration_basis,
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "holdout_exclusions": exclusions,
        "array_dtypes": {"input": "float32", "clean": "float32", "teacher": "float16"},
        "array_integrity": {
            "field": "array_sha256",
            "algorithm": "SHA-256",
            "scope": "complete .npy file bytes immediately after atomic-cache write",
            "required_arrays": ["input", "clean", "teacher"],
        },
        "generation_environment": {
            "numpy": np.__version__,
            "limited_smoke": False,
            "script": active_generator_identity()["path"],
            "script_sha256": active_generator_identity()["sha256"],
        },
        "records": records,
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return config, output_root


def write_manifest(config: dict, document: dict) -> None:
    Path(config["data"]["manifest"]).write_text(json.dumps(document), encoding="utf-8")


def measured_rows(config: dict, document: dict) -> list[dict]:
    cache_root = Path(config["data"]["cache_root"])
    rows = []
    for record in document["records"]:
        arrays = {
            key: np.load(cache_root / record[key], allow_pickle=False).astype(np.float32)
            for key in ("input", "clean", "teacher")
        }
        rows.append(record_metrics(record, arrays))
    return rows


def domain_stat_row(
    *,
    correlation: float = 0.0,
    payloads: int = 12,
    scenes: int = 5,
    pixels: int = 4096,
) -> dict:
    matrix = [
        [1.0, correlation, correlation],
        [correlation, 1.0, correlation],
        [correlation, correlation, 1.0],
    ]
    return {
        "pixels": pixels,
        "support": {
            "distinct_source_payloads": payloads,
            "distinct_scenes": scenes,
        },
        "bands": {
            band: {
                "rms": 0.02,
                "rgb_correlation": matrix,
                "energy_fraction": 1.0 / 3.0,
            }
            for band in ("fine", "medium", "coarse")
        },
    }


def validate_global_rgb_correlation_contract() -> None:
    # Keep the global residual covariance deliberately different from the last
    # per-band/bin covariance. This catches accidental loop-variable reuse in
    # the profile serializer.
    global_values = np.asarray(
        [
            [-2.0, -1.8, -0.2],
            [-1.0, -0.9, 0.1],
            [1.0, 0.9, -0.1],
            [2.0, 1.8, 0.2],
        ],
        dtype=np.float64,
    )
    last_bin_values = np.asarray(
        [
            [-2.0, 2.0, -0.2],
            [-1.0, 1.0, 0.1],
            [1.0, -1.0, -0.1],
            [2.0, -2.0, 0.2],
        ],
        dtype=np.float64,
    )
    count = len(global_values)
    serialized = global_rgb_correlation_from_moments(
        global_values.sum(axis=0),
        global_values.T @ global_values,
        count,
        iso=12800,
    )
    global_mean = global_values.mean(axis=0)
    expected = positive_correlation(
        global_values.T @ global_values / count - np.outer(global_mean, global_mean)
    )
    last_bin_mean = last_bin_values.mean(axis=0)
    last_bin = positive_correlation(
        last_bin_values.T @ last_bin_values / count
        - np.outer(last_bin_mean, last_bin_mean)
    )
    np.testing.assert_allclose(serialized, expected, atol=1e-12, rtol=0.0)
    assert not np.allclose(serialized, last_bin, atol=1e-3, rtol=0.0)


def validate_clustered_support_contract() -> None:
    support = CalibrationSupportThresholds(
        minimum_pixels=128,
        minimum_payloads=12,
        minimum_scenes=5,
        shadow_luminance_max=0.25,
    )
    shadow_label = "shadow:0-0.25"
    synthetic = {
        "0:0-0.1": domain_stat_row(correlation=0.5),
        "1:0.1-0.3": domain_stat_row(),
        shadow_label: domain_stat_row(),
    }
    reference = {
        "0:0-0.1": domain_stat_row(payloads=4, scenes=2),
        "1:0.1-0.3": domain_stat_row(),
        shadow_label: domain_stat_row(),
    }

    inconclusive_findings = Findings()
    inconclusive = compare_domain_stats(
        synthetic,
        reference,
        support,
        GateThresholds(),
        inconclusive_findings,
    )
    unsupported = inconclusive["by_luminance_bin"]["0:0-0.1"]
    assert unsupported["decision"] == "inconclusive"
    assert len(unsupported["raw_threshold_failures"]) == 3
    assert inconclusive["supported_bins"] == 1
    assert inconclusive["passing_bins"] == 1
    assert inconclusive["mandatory_shadow_decision"] == "pass"
    assert inconclusive_findings.counts["warnings"]["calibration_bin_inconclusive"] == 1
    assert inconclusive_findings.counts["errors"]["calibration_correlation_mismatch"] == 0

    supported_reference = copy.deepcopy(reference)
    supported_reference["0:0-0.1"]["support"] = {
        "distinct_source_payloads": 12,
        "distinct_scenes": 5,
    }
    supported_findings = Findings()
    supported_result = compare_domain_stats(
        synthetic,
        supported_reference,
        support,
        GateThresholds(),
        supported_findings,
    )
    assert supported_result["by_luminance_bin"]["0:0-0.1"]["decision"] == "fail"
    assert supported_findings.counts["errors"]["calibration_correlation_mismatch"] == 3

    unsupported_shadow = copy.deepcopy(reference)
    unsupported_shadow[shadow_label]["support"]["distinct_source_payloads"] = 11
    shadow_support_findings = Findings()
    shadow_support_result = compare_domain_stats(
        synthetic,
        unsupported_shadow,
        support,
        GateThresholds(),
        shadow_support_findings,
    )
    assert shadow_support_result["mandatory_shadow_decision"] == "inconclusive"
    assert (
        shadow_support_findings.counts["errors"][
            "calibration_shadow_support_insufficient"
        ]
        == 1
    )

    failing_shadow = copy.deepcopy(synthetic)
    failing_shadow[shadow_label] = domain_stat_row(correlation=0.5)
    shadow_failure_findings = Findings()
    shadow_failure_result = compare_domain_stats(
        failing_shadow,
        reference,
        support,
        GateThresholds(),
        shadow_failure_findings,
    )
    assert shadow_failure_result["mandatory_shadow_decision"] == "fail"
    assert (
        shadow_failure_findings.counts["errors"][
            "calibration_shadow_distribution_mismatch"
        ]
        == 1
    )


def main() -> None:
    validate_global_rgb_correlation_contract()
    validate_clustered_support_contract()
    with tempfile.TemporaryDirectory(prefix="synthetic-camera-jpeg-test-") as temporary:
        root = Path(temporary)
        config, output_root = fixture(root)
        exclusions = config["data"]["holdout_exclusions"]
        profile_path, _, loaded_profiles = load_profiles(
            config, [12800, 25600, 51200], exclusions, False
        )
        assert profile_path == Path(config["data"]["calibration_profile"]).resolve()
        assert set(loaded_profiles) == {12800, 25600, 51200}

        profile_document = json.loads(profile_path.read_text(encoding="utf-8"))
        profile_document["fitting_environment"]["script_sha256"] = "0" * 64
        profile_path.write_text(json.dumps(profile_document), encoding="utf-8")
        try:
            load_profiles(config, [12800], exclusions, False)
        except RuntimeError as error:
            assert "active fitter" in str(error)
        else:
            raise AssertionError("A stale calibration fitter hash was accepted")
        profile_document["fitting_environment"]["script_sha256"] = sha256_file(
            Path(__file__).with_name("fit_synthetic_noise_profiles.py").resolve()
        )
        profile_path.write_text(json.dumps(profile_document), encoding="utf-8")

        reconstructor = load_reconstructor(config)
        assert reconstructor is not None

        repeated_template = json.loads(
            Path(config["data"]["manifest"]).read_text(encoding="utf-8")
        )["records"][0]
        repeated_source = (
            Path(config["data"]["source_root"]) / repeated_template["source_clean"]
        ).resolve()
        repeated_sha = sha256_file(repeated_source)
        repeated_records = []
        for index in range(3):
            repeated = copy.deepcopy(repeated_template)
            repeated["id"] = f"repeated-source-{index}"
            repeated["scene"] = "repeated-source-scene"
            repeated["source_clean_sha256"] = repeated_sha
            for key in ("input", "clean", "teacher"):
                repeated[key] = f"repeat-{index}-{key}.npy"
            repeated_records.append(repeated)
        source_hash_calls = 0

        def count_source_hash(path: Path) -> str:
            nonlocal source_hash_calls
            source_hash_calls += 1
            assert path == repeated_source
            return repeated_sha

        source_findings = Findings()
        source_summary = validate_sources_and_splits(
            repeated_records,
            Path(config["data"]["source_root"]),
            {"paths": [], "sha256": []},
            source_findings,
            hash_file=count_source_hash,
        )
        assert source_hash_calls == 1
        assert source_summary["source_files"] == 1
        assert source_findings.counts["errors"]["source_clean_sha256_mismatch"] == 0

        report = validate_cache(config, reconstructor=reconstructor, output_dir=output_root)
        assert report["status"] == "pending_visual_review", report["findings"]
        assert report["release_gate_passed"] is False
        assert report["findings"]["errors"]["total"] == 0, report["findings"]
        assert report["reconstruction"]["records_reconstructed"] == 4
        assert report["calibration"]["reference_pairs"] == 1
        assert report["smoke"] == {"generation": False, "calibration": False}
        assert report["analysis_sha256"] == analysis_sha256(report["analysis"])
        assert report["analysis"]["schema_version"] == ANALYSIS_SCHEMA_VERSION
        assert report["analysis"]["validator"] == active_validator_identity()
        assert report["analysis"]["generator"] == active_generator_identity()
        assert Path(report["contact_sheet"]["png"]).is_file()
        assert Path(report["contact_sheet"]["jpg"]).is_file()

        signoff_path = output_root / "visual_acceptance.json"
        report_path = output_root / "synthetic_camera_jpeg_gate_report.json"
        pending_report_bytes = report_path.read_bytes()
        pending_report_sha = sha256_file(report_path)
        contact_bytes = {
            extension: Path(report["contact_sheet"][extension]).read_bytes()
            for extension in ("png", "jpg")
        }

        try:
            accept_existing_report(
                config,
                destination=output_root,
                signoff_path=signoff_path,
                reviewer="offline fixture reviewer",
                expected_report_sha256="0" * 64,
            )
        except RuntimeError as error:
            assert "reviewed report changed" in str(error)
        else:
            raise AssertionError("visual acceptance ignored the explicit prior report hash")

        contact_png = Path(report["contact_sheet"]["png"])
        contact_png.write_bytes(b"changed after review")
        try:
            accept_existing_report(
                config,
                destination=output_root,
                signoff_path=signoff_path,
                reviewer="offline fixture reviewer",
                expected_report_sha256=pending_report_sha,
            )
        except RuntimeError as error:
            assert "contact-sheet png changed" in str(error)
        else:
            raise AssertionError("visual acceptance ignored a changed rendered artifact")
        contact_png.write_bytes(contact_bytes["png"])

        inconsistent = json.loads(pending_report_bytes)
        inconsistent["records_measured"] += 1
        report_path.write_text(json.dumps(inconsistent), encoding="utf-8")
        try:
            accept_existing_report(
                config,
                destination=output_root,
                signoff_path=signoff_path,
                reviewer="offline fixture reviewer",
                expected_report_sha256=sha256_file(report_path),
            )
        except RuntimeError as error:
            assert "differs from analysis.records_measured" in str(error)
        else:
            raise AssertionError("visual acceptance ignored nested/top-level inconsistency")
        report_path.write_bytes(pending_report_bytes)

        stale_validator = json.loads(pending_report_bytes)
        stale_validator["analysis"]["validator"]["sha256"] = "0" * 64
        stale_validator["analysis_sha256"] = analysis_sha256(stale_validator["analysis"])
        report_path.write_text(json.dumps(stale_validator), encoding="utf-8")
        try:
            accept_existing_report(
                config,
                destination=output_root,
                signoff_path=signoff_path,
                reviewer="offline fixture reviewer",
                expected_report_sha256=sha256_file(report_path),
            )
        except RuntimeError as error:
            assert "validator identity is not active" in str(error)
        else:
            raise AssertionError("visual acceptance ignored a stale validator identity")
        report_path.write_bytes(pending_report_bytes)

        accepted = accept_existing_report(
            config,
            destination=output_root,
            signoff_path=signoff_path,
            reviewer="offline fixture reviewer",
            expected_report_sha256=pending_report_sha,
        )
        signoff = json.loads(signoff_path.read_text(encoding="utf-8"))
        assert set(signoff) == VISUAL_ACCEPTANCE_KEYS
        assert accepted["status"] == "accepted", accepted["visual_acceptance"]
        assert accepted["release_gate_passed"] is True
        assert accepted["analysis_sha256"] == report["analysis_sha256"]
        assert accepted["reviewed_report_sha256"] == pending_report_sha
        assert signoff["reviewed_report_sha256"] == pending_report_sha
        assert {
            extension: Path(report["contact_sheet"][extension]).read_bytes()
            for extension in ("png", "jpg")
        } == contact_bytes

        original_signoff = signoff_path.read_bytes()
        invalid_signoff = dict(signoff)
        invalid_signoff["unexpected"] = True
        signoff_path.write_text(json.dumps(invalid_signoff), encoding="utf-8")
        invalid = validate_cache(config, reconstructor=reconstructor, output_dir=output_root)
        assert invalid["status"] == "pending_visual_review"
        assert invalid["visual_acceptance"]["status"] == "invalid"
        signoff_path.write_bytes(original_signoff)

        stale_signoff = dict(signoff)
        stale_signoff["contact_png_sha256"] = "0" * 64
        signoff_path.write_text(json.dumps(stale_signoff), encoding="utf-8")
        stale = validate_cache(config, reconstructor=reconstructor, output_dir=output_root)
        assert stale["status"] == "pending_visual_review"
        assert stale["visual_acceptance"]["status"] == "stale"
        signoff_path.write_bytes(original_signoff)

        manifest_path = Path(config["data"]["manifest"])
        original = json.loads(manifest_path.read_text(encoding="utf-8"))

        stale_generator = copy.deepcopy(original)
        stale_generator["generation_environment"]["script_sha256"] = "0" * 64
        write_manifest(config, stale_generator)
        stale_generator_report = validate_cache(
            config, reconstructor=reconstructor, output_dir=output_root
        )
        assert (
            stale_generator_report["findings"]["errors"]["by_code"][
                "generation_script_sha256_mismatch"
            ]
            == 1
        )

        protected_source = Path(config["data"]["source_root"]) / "clean/protected-copy.png"
        protected_source.write_bytes((root / "sony_holdout.png").read_bytes())
        leaked = json.loads(json.dumps(original))
        leaked["records"][0]["source_clean"] = "clean/protected-copy.png"
        leaked["records"][0]["source_clean_sha256"] = sha256_file(protected_source)
        write_manifest(config, leaked)
        leaked_report = validate_cache(config, reconstructor=reconstructor, output_dir=output_root)
        assert leaked_report["findings"]["errors"]["by_code"]["protected_holdout_overlap"] == 1

        split_leak = json.loads(json.dumps(original))
        split_leak["records"][0]["split"] = "validation"
        write_manifest(config, split_leak)
        split_report = validate_cache(config, reconstructor=reconstructor, output_dir=output_root)
        assert split_report["findings"]["errors"]["by_code"]["scene_split_leakage"] == 1

        missing_hash = json.loads(json.dumps(original))
        del missing_hash["records"][0]["array_sha256"]
        write_manifest(config, missing_hash)
        missing_hash_report = validate_cache(
            config, reconstructor=reconstructor, output_dir=output_root
        )
        missing_codes = missing_hash_report["findings"]["errors"]["by_code"]
        assert missing_codes["record_fields_missing"] == 1
        assert missing_codes["missing_array_sha256"] == 3

        write_manifest(config, original)
        first = original["records"][0]
        input_path = Path(config["data"]["cache_root"]) / first["input"]
        pristine = np.load(input_path, allow_pickle=False)
        np.save(input_path, np.clip(pristine + 0.01, 0.0, 1.0).astype(np.float32))
        corrupt_report = validate_cache(config, reconstructor=reconstructor, output_dir=output_root)
        codes = corrupt_report["findings"]["errors"]["by_code"]
        assert codes["array_sha256_mismatch"] == 1
        assert codes["reconstruction_cache_mismatch"] >= 1
        assert corrupt_report["visual_acceptance"]["status"] == "stale"
        np.save(input_path, pristine)
        assert sha256_file(input_path) == first["array_sha256"]["input"]

        rows = measured_rows(config, original)
        calls = 0

        def nondeterministic(record: dict) -> dict[str, np.ndarray]:
            nonlocal calls
            calls += 1
            values = {key: value.copy() for key, value in reconstructor(record).items()}
            values["input"] = np.clip(values["input"] + calls * 1e-5, 0.0, 1.0)
            return values

        findings = Findings()
        validate_reconstruction(
            rows,
            Path(config["data"]["cache_root"]),
            nondeterministic,
            GateThresholds(),
            1,
            3,
            findings,
        )
        assert findings.counts["errors"]["reconstruction_nondeterministic"] == 1

        calibration_findings = Findings()
        validate_calibration(
            rows,
            Path(config["data"]["cache_root"]),
            Path(config["data"]["source_root"]),
            config,
            GateThresholds(
                maximum_calibration_log_rms_error=0.0,
                maximum_calibration_band_l1=0.0,
                maximum_calibration_correlation_error=0.0,
            ),
            {
                "paths": [entry["path"] for entry in config["data"]["holdout_exclusions"]],
                "sha256": [entry["sha256"] for entry in config["data"]["holdout_exclusions"]],
            },
            calibration_findings,
        )
        mismatch_total = sum(
            calibration_findings.counts["errors"][code]
            for code in (
                "calibration_log_rms_mismatch",
                "calibration_band_l1_mismatch",
                "calibration_correlation_mismatch",
            )
        )
        assert mismatch_total > 0

        zero_rows = []
        cache_root = Path(config["data"]["cache_root"])
        for row in rows:
            record = row["record"]
            clean = np.load(cache_root / record["clean"], allow_pickle=False).astype(np.float32)
            teacher = np.load(cache_root / record["teacher"], allow_pickle=False).astype(np.float32)
            zero_rows.append(
                record_metrics(record, {"input": clean.copy(), "clean": clean, "teacher": teacher})
            )
        frequency_findings = Findings()
        validate_severity(
            zero_rows,
            GateThresholds(
                minimum_teacher_better_fraction=0.0,
                minimum_teacher_gain_median_db=0.0,
                minimum_shadow_fraction=0.0,
                minimum_structured_band_fraction=0.001,
            ),
            frequency_findings,
        )
        assert frequency_findings.counts["errors"]["structured_noise_missing"] == 3

    print("Synthetic camera-JPEG gate checks passed")


if __name__ == "__main__":
    main()
