#!/usr/bin/env python3
"""Focused offline checks for the UHD-LL/SNIC cache validator."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_domain_dataset import (
    ImageSource,
    alignment_gate,
    apply_local_gain,
    build_local_gain_field,
    build_uhd_hybrid_target,
    crop_seed,
    sample_thumbnail_field,
    stratified_positions,
)

from validate_domain_gate import (
    Findings,
    array_metrics,
    load_checked_array,
    render_contact_sheet,
    representative_rows,
    reconstruct_uhd_cache,
    stratified_subset,
    validate_record_semantics,
)


TARGET_CONFIG = {
    "thumbnail_width": 480,
    "illumination_sigma_full_resolution": 128.0,
    "channel_denominator_floor": 0.003,
    "channel_confidence_scale": 0.025,
    "minimum_gain": 0.005,
    "maximum_gain": 1.25,
    "gain_smoothing_sigma_thumbnail": 1.5,
    "teacher_lowpass_sigma": 8.0,
    "clean_detail_gain": 1.0,
    "alignment_prefilter_sigma": 0.7,
    "alignment_search_radius": 3,
    "minimum_texture": 0.012,
    "minimum_zero_shift_correlation": 0.50,
    "maximum_nonzero_shift_gain": 0.018,
}


def record(
    identifier: str,
    dataset: str,
    split: str,
    supervision: str,
    gt_weight: float,
    kd_weight: float,
    gate: dict | None,
) -> dict:
    base = Path(split) / dataset / identifier
    return {
        "id": identifier,
        "dataset": dataset,
        "split": split,
        "scene": f"scene_{identifier}",
        "supervision": supervision,
        "gt_weight": gt_weight,
        "kd_weight": kd_weight,
        "crop": [0, 0, 192, 192],
        "crop_mean_luminance": 0.2,
        "source_input": f"{dataset}/{identifier}_noisy.tiff",
        "source_clean": f"{dataset}/{identifier}_clean.tiff",
        "input": str(base.with_name(base.name + "_input.npy")),
        "teacher": str(base.with_name(base.name + "_teacher.npy")),
        "clean": str(base.with_name(base.name + "_clean.npy")),
        "uhd_hybrid_gate": gate,
        "clean_target": (
            "native_paired_reference"
            if dataset == "snic_sony"
            else "teacher_lowpass_plus_local_clean_highpass_linear_rgb"
            if supervision == "uhd_hybrid_paired"
            else "local_ratio_candidate_ignored"
        ),
        "noise_level": "6400",
        "iso": 6400 if dataset == "snic_sony" else None,
        "jpeg_quality": 95 if dataset == "snic_sony" else None,
    }


def save_arrays(root: Path, metadata: dict, offset: float) -> dict[str, np.ndarray]:
    y, x = np.mgrid[:192, :192]
    pattern = 0.12 + 0.55 * x[..., None] / 191.0 + 0.08 * np.sin(y[..., None] / 7.0)
    pattern = np.repeat(pattern, 3, axis=2).astype(np.float32)
    noisy = np.clip(pattern + offset, 0.0, 1.0).astype(np.float32)
    teacher = np.clip(pattern + offset * 0.25, 0.0, 1.0).astype(np.float32)
    clean = pattern.astype(np.float32)
    arrays = {"input": noisy, "teacher": teacher, "clean": clean}
    for key, value in arrays.items():
        path = root / metadata[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value.astype(np.float16 if key == "teacher" else np.float32))
    return arrays


def test_source_reconstruction(root: Path) -> None:
    source_root = root / "reconstruction_sources"
    cache_root = root / "reconstruction_cache"
    rng = np.random.default_rng(913)
    clean = rng.uniform(0.04, 0.88, (256, 320, 3)).astype(np.float32)
    noisy = apply_local_gain(
        clean,
        np.broadcast_to(np.asarray([0.24, 0.18, 0.13], dtype=np.float32), clean.shape),
    )
    input_path = source_root / "uhd/input.png"
    clean_path = source_root / "uhd/clean.png"
    input_path.parent.mkdir(parents=True)
    Image.fromarray(np.rint(noisy * 255.0).astype(np.uint8), "RGB").save(input_path)
    Image.fromarray(np.rint(clean * 255.0).astype(np.uint8), "RGB").save(clean_path)
    source = {
        "dataset": "uhd_ll",
        "scene": "reconstruction_scene",
        "split": "train",
        "input": "uhd/input.png",
        "clean": "uhd/clean.png",
    }
    target_config = {**TARGET_CONFIG, "thumbnail_width": 160}
    config = {
        "project": {"seed": 20260719},
        "data": {
            "patches_per_pair": {"uhd_ll": 1},
            "crop_candidates": 32,
            "jpeg_quality": {},
        },
        "uhd_hybrid_target": target_config,
    }
    with ImageSource(input_path) as noisy_source, ImageSource(clean_path) as clean_source:
        noisy_thumbnail = noisy_source.thumbnail(160)
        clean_thumbnail = clean_source.thumbnail(160)
        gain_field = build_local_gain_field(
            noisy_thumbnail, clean_thumbnail, noisy_source.width, target_config
        )
        [(left, top, crop_luma)] = stratified_positions(
            noisy_thumbnail,
            noisy_source.width,
            noisy_source.height,
            1,
            32,
            crop_seed(source, 20260719),
        )
        noisy_crop = noisy_source.crop(left, top)
        gain = sample_thumbnail_field(
            gain_field, left, top, noisy_source.width, noisy_source.height
        )
        mapped_clean = apply_local_gain(clean_source.crop(left, top), gain)
    gate = alignment_gate(noisy_crop, mapped_clean, target_config)
    assert gate["passed"], gate
    gate.update(
        {
            "local_gain_minimum": float(gain.min()),
            "local_gain_mean": gain.mean(axis=(0, 1)).tolist(),
            "local_gain_maximum": float(gain.max()),
        }
    )
    teacher = np.clip(noisy_crop + 0.045, 0.0, 1.0).astype(np.float32)
    cached_teacher = teacher.astype(np.float16)
    hybrid = build_uhd_hybrid_target(
        cached_teacher.astype(np.float32), mapped_clean, target_config
    )
    source_hash = hashlib.sha256(
        f"uhd_ll:{source['input']}:{source['clean']}".encode()
    ).hexdigest()
    identifier = f"{source_hash[:16]}_00"
    metadata = record(
        identifier,
        "uhd_ll",
        "train",
        "uhd_hybrid_paired",
        1.0,
        0.7,
        gate,
    )
    metadata.update(
        {
            "scene": source["scene"],
            "source_input": source["input"],
            "source_clean": source["clean"],
            "crop": [left, top, 192, 192],
            "crop_mean_luminance": crop_luma,
        }
    )
    arrays = {"input": noisy_crop, "teacher": cached_teacher, "clean": hybrid}
    for key, value in arrays.items():
        destination = cache_root / metadata[key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, value.astype(np.float16 if key == "teacher" else np.float32))

    source_key = ("uhd_ll", source["input"], source["clean"])
    findings = Findings()
    report = reconstruct_uhd_cache(
        [metadata],
        {source_key: source},
        source_root,
        cache_root,
        config,
        1e-4,
        findings,
    )
    assert report["records_reconstructed"] == 1
    assert not findings.failed, findings.report()

    np.save(cache_root / metadata["clean"], mapped_clean.astype(np.float32))
    stale_findings = Findings()
    reconstruct_uhd_cache(
        [metadata],
        {source_key: source},
        source_root,
        cache_root,
        config,
        1e-4,
        stale_findings,
    )
    assert stale_findings.counts["errors"]["reconstructed_clean_target_mismatch"] == 1


def main() -> None:
    passed_gate = {
        "texture": 0.04,
        "zero_edge_correlation": 0.95,
        "best_edge_correlation": 0.95,
        "best_shift": [0, 0],
        "alignment_gain": 0.0,
        "local_gain_minimum": 0.12,
        "local_gain_mean": [0.18, 0.17, 0.16],
        "local_gain_maximum": 0.22,
        "failure_reasons": [],
        "passed": True,
    }
    failed_gate = {
        "texture": 0.005,
        "zero_edge_correlation": 0.30,
        "best_edge_correlation": 0.70,
        "best_shift": [0, 2],
        "alignment_gain": 0.40,
        "local_gain_minimum": 0.08,
        "local_gain_mean": [0.11, 0.10, 0.09],
        "local_gain_maximum": 0.14,
        "failure_reasons": [
            "insufficient_texture",
            "zero_shift_correlation",
            "nonzero_shift_improvement",
        ],
        "passed": False,
    }
    records = [
        record("uhd_pass", "uhd_ll", "train", "uhd_hybrid_paired", 1.0, 0.7, passed_gate),
        record("uhd_fail", "uhd_ll", "validation", "teacher_only", 0.0, 1.0, failed_gate),
        record("snic", "snic_sony", "train", "paired", 1.0, 0.7, None),
    ]

    strata = []
    for dataset in ("uhd_ll", "snic_sony"):
        for split in ("train", "validation"):
            for level in range(3):
                strata.append(
                    {
                        "id": f"{dataset}_{split}_{level}",
                        "dataset": dataset,
                        "split": split,
                        "supervision": (
                            "paired"
                            if dataset == "snic_sony"
                            else "uhd_hybrid_paired" if level == 0 else "teacher_only"
                        ),
                        "noise_level": str(1600 * (2**level)),
                    }
                )
    capped_two = stratified_subset(strata, maximum=2, seed=731)
    assert {row["dataset"] for row in capped_two} == {"uhd_ll", "snic_sony"}
    assert {row["split"] for row in capped_two} == {"train", "validation"}
    capped_four = stratified_subset(strata, maximum=4, seed=731)
    assert {(row["dataset"], row["split"]) for row in capped_four} == {
        ("uhd_ll", "train"),
        ("uhd_ll", "validation"),
        ("snic_sony", "train"),
        ("snic_sony", "validation"),
    }
    capped_eight = stratified_subset(strata, maximum=8, seed=731)
    coarse_counts = {
        key: sum((row["dataset"], row["split"]) == key for row in capped_eight)
        for key in {(row["dataset"], row["split"]) for row in strata}
    }
    assert set(coarse_counts.values()) == {2}
    assert [row["id"] for row in capped_eight] == [
        row["id"] for row in stratified_subset(strata, maximum=8, seed=731)
    ]
    try:
        stratified_subset(strata, maximum=0, seed=731)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-size stratified subsets must be rejected")

    findings = Findings()
    for metadata in records:
        validate_record_semantics(
            metadata,
            {"uhd_ll", "snic_sony"},
            TARGET_CONFIG,
            {"snic_sony": 95},
            findings,
        )
    assert not findings.failed, findings.report()

    float32_boundary = dict(records[1])
    float32_boundary["uhd_hybrid_gate"] = {
        **failed_gate,
        "local_gain_minimum": float(np.float32(TARGET_CONFIG["minimum_gain"])),
    }
    boundary_findings = Findings()
    validate_record_semantics(
        float32_boundary,
        {"uhd_ll", "snic_sony"},
        TARGET_CONFIG,
        {"snic_sony": 95},
        boundary_findings,
    )
    assert not boundary_findings.failed, boundary_findings.report()

    invalid_gain = dict(float32_boundary)
    invalid_gain["uhd_hybrid_gate"] = {
        **float32_boundary["uhd_hybrid_gate"],
        "local_gain_minimum": TARGET_CONFIG["minimum_gain"] - 1e-4,
    }
    invalid_gain_findings = Findings()
    validate_record_semantics(
        invalid_gain,
        {"uhd_ll", "snic_sony"},
        TARGET_CONFIG,
        {"snic_sony": 95},
        invalid_gain_findings,
    )
    assert invalid_gain_findings.counts["errors"]["invalid_local_gain_range"] == 1

    malformed = dict(records[0], kd_weight=1.0)
    malformed_findings = Findings()
    validate_record_semantics(
        malformed,
        {"uhd_ll", "snic_sony"},
        TARGET_CONFIG,
        {"snic_sony": 95},
        malformed_findings,
    )
    assert malformed_findings.counts["errors"]["kd_weight_mismatch"] == 1

    with tempfile.TemporaryDirectory(prefix="domain-gate-smoke-") as temporary:
        root = Path(temporary)
        test_source_reconstruction(root)
        rows = []
        digest = hashlib.sha256()
        for index, metadata in enumerate(records):
            save_arrays(root, metadata, 0.03 + 0.01 * index)
            arrays = {
                key: load_checked_array(
                    root,
                    metadata,
                    key,
                    np.dtype(np.float16 if key == "teacher" else np.float32),
                    findings,
                    digest if key == "teacher" else None,
                )
                for key in ("input", "teacher", "clean")
            }
            assert all(value is not None for value in arrays.values())
            rows.append(array_metrics(metadata, arrays))
        assert not findings.failed, findings.report()
        assert digest.hexdigest() != hashlib.sha256().hexdigest()
        selected = representative_rows(rows, maximum=3)
        destination = root / "contact.png"
        render_contact_sheet(selected, root, destination)
        assert destination.is_file() and destination.stat().st_size > 0
        assert destination.with_suffix(".jpg").is_file()

        unsafe = dict(records[0], input="../escape.npy")
        unsafe_findings = Findings()
        assert (
            load_checked_array(
                root,
                unsafe,
                "input",
                np.dtype(np.float32),
                unsafe_findings,
            )
            is None
        )
        assert unsafe_findings.counts["errors"]["unsafe_array_path"] == 1

    print("UHD-LL/SNIC domain-gate validation checks passed")


if __name__ == "__main__":
    main()
