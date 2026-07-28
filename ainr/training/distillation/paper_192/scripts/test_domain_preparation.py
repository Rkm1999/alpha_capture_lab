#!/usr/bin/env python3
"""Focused checks for UHD-LL/SNIC acquisition and source preparation."""

from __future__ import annotations

import json
import io
import sys
import tempfile
import types
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from build_domain_manifest import (
    EXPECTED_SNIC_ARCHIVES,
    EXPECTED_SNIC_ISOS,
    completed_snic_inputs,
    deterministic_split,
)
try:
    import remotezip  # noqa: F401
except ImportError:
    sys.modules["remotezip"] = types.SimpleNamespace(RemoteZip=object)
from download_snic_subset import extract_entry
from download_uhd_listing import jpeg_has_complete_markers, safe_relative_path
from prepare_domain_dataset import (
    TILE,
    UHD_CLEAN_TRANSFORM,
    alignment_gate,
    apply_local_gain,
    audit_sources,
    build_local_gain_field,
    build_uhd_hybrid_target,
    gaussian_rgb,
    sample_thumbnail_field,
    srgb_to_linear,
    source_path,
    stratified_positions,
)


def expect_error(function, exception: type[Exception]) -> None:  # noqa: ANN001
    try:
        function()
    except exception:
        return
    raise AssertionError(f"Expected {exception.__name__}")


def test_paths_and_complete_markers(root: Path) -> None:
    assert safe_relative_path("training_set/input/1_UHD_LL.JPG").as_posix().startswith(
        "training_set/"
    )
    for unsafe in ("../escape.jpg", "/absolute.jpg", "folder/../escape.jpg"):
        expect_error(lambda value=unsafe: safe_relative_path(value), ValueError)

    image_path = root / "valid.jpg"
    Image.new("RGB", (TILE, TILE), (20, 30, 40)).save(image_path, quality=95)
    assert jpeg_has_complete_markers(image_path)
    truncated = root / "truncated.jpg"
    truncated.write_bytes(image_path.read_bytes()[:-2])
    assert not jpeg_has_complete_markers(truncated)

    contained = root / "contained.bin"
    contained.write_bytes(b"data")
    assert source_path(root, "contained.bin") == contained.resolve()
    expect_error(lambda: source_path(root, "../outside.bin"), ValueError)


def test_deterministic_crops() -> None:
    thumbnail = np.linspace(0.0, 1.0, 240 * 320 * 3, dtype=np.float32).reshape(240, 320, 3)
    first = stratified_positions(thumbnail, 1920, 1080, 8, 192, 12345)
    second = stratified_positions(thumbnail, 1920, 1080, 8, 192, 12345)
    assert first == second
    assert len(first) == 8
    assert len({position[:2] for position in first}) == 8
    assert deterministic_split("dataset", "scene", 0.2) == deterministic_split(
        "dataset", "scene", 0.2
    )


def hybrid_config() -> dict:
    return {
        "project": {"seed": 20260719},
        "data": {
            "patches_per_pair": {"uhd_ll": 1},
            "crop_candidates": 32,
        },
        "uhd_hybrid_target": {
            "thumbnail_width": 160,
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
        },
    }


def test_uhd_hybrid_preflight(root: Path) -> None:
    rng = np.random.default_rng(77)
    clean = rng.uniform(0.04, 0.82, (256, 320, 3)).astype(np.float32)
    expected_gain = np.asarray([0.24, 0.19, 0.15], dtype=np.float32)
    noisy = apply_local_gain(clean, np.broadcast_to(expected_gain, clean.shape))
    config = hybrid_config()

    thumbnail_size = (160, 128)
    clean_thumbnail = np.asarray(
        Image.fromarray(np.rint(clean * 255).astype(np.uint8)).resize(
            thumbnail_size, Image.Resampling.BOX
        ),
        dtype=np.float32,
    ) / 255.0
    noisy_thumbnail = np.asarray(
        Image.fromarray(np.rint(noisy * 255).astype(np.uint8)).resize(
            thumbnail_size, Image.Resampling.BOX
        ),
        dtype=np.float32,
    ) / 255.0
    gain_field = build_local_gain_field(
        noisy_thumbnail,
        clean_thumbnail,
        clean.shape[1],
        config["uhd_hybrid_target"],
    )
    crop_gain = sample_thumbnail_field(gain_field, 64, 32, 320, 256)
    np.testing.assert_allclose(crop_gain.mean(axis=(0, 1)), expected_gain, atol=0.015)
    mapped_clean = apply_local_gain(clean[32:224, 64:256], crop_gain)
    noisy_crop = noisy[32:224, 64:256]
    passed = alignment_gate(noisy_crop, mapped_clean, config["uhd_hybrid_target"])
    assert passed["passed"], passed

    shifted = np.empty_like(mapped_clean)
    shifted[:, 2:] = mapped_clean[:, :-2]
    shifted[:, :2] = mapped_clean[:, :1]
    failed = alignment_gate(noisy_crop, shifted, config["uhd_hybrid_target"])
    assert not failed["passed"]
    assert "nonzero_shift_improvement" in failed["failure_reasons"]

    teacher = np.clip(
        noisy_crop + rng.normal(0.0, 0.002, noisy_crop.shape), 0.0, 1.0
    ).astype(np.float32)
    target = build_uhd_hybrid_target(
        teacher,
        mapped_clean,
        config["uhd_hybrid_target"],
    )
    teacher_low = gaussian_rgb(srgb_to_linear(teacher), 8.0)
    target_low = gaussian_rgb(srgb_to_linear(target), 8.0)
    assert float(np.mean(np.abs(target_low - teacher_low))) < 0.003
    assert target.shape == (TILE, TILE, 3)
    assert 0.0 <= float(target.min()) <= float(target.max()) <= 1.0

    source_root = root / "sources"
    noisy_path = source_root / "uhd/input.jpg"
    clean_path = source_root / "uhd/clean.jpg"
    noisy_path.parent.mkdir(parents=True)
    Image.fromarray(np.rint(noisy * 255).astype(np.uint8)).save(noisy_path, quality=100)
    Image.fromarray(np.rint(clean * 255).astype(np.uint8)).save(clean_path, quality=100)
    records = [
        {
            "dataset": "uhd_ll",
            "scene": "synthetic",
            "split": "train",
            "input": "uhd/input.jpg",
            "clean": "uhd/clean.jpg",
            "clean_transform": UHD_CLEAN_TRANSFORM,
        }
    ]
    report = audit_sources(records, source_root, config)
    assert report["source_pairs"] == 1
    assert report["uhd_hybrid_gate"]["crops"] == 1
    assert report["uhd_hybrid_gate"]["passed"] == 1
    assert report["uhd_hybrid_gate"]["failed"] == 0


def write_complete_snic_manifest(root: Path) -> Path:
    archives = []
    for file_id, stem in EXPECTED_SNIC_ARCHIVES.items():
        entries = []
        for iso in sorted(EXPECTED_SNIC_ISOS):
            noisy_name = f"set/{stem}/{stem}_24mm_{iso:05d}_noisy_01.tiff"
            clean_name = f"set/{stem}/{stem}_24mm_00100_clean_01.tiff"
            for name in (noisy_name, clean_name):
                destination = root / stem / name
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(b"II*\x00\x08\x00\x00\x00")
                if not any(row["name"] == name for row in entries):
                    entries.append({"name": name, "size": destination.stat().st_size})
        archives.append(
            {
                "datafile_id": file_id,
                "name": f"{stem}.zip",
                "entries": len(entries),
                "states": {"downloaded": len(entries), "existing": 0},
                "zip_entries": entries,
            }
        )
    manifest_path = root / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "complete": True,
                "integrity": "size_and_zip_crc32_verified",
                "noisy_isos": sorted(EXPECTED_SNIC_ISOS),
                "archives": archives,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_snic_completion_guard(root: Path) -> None:
    snic_root = root / "snic"
    manifest_path = write_complete_snic_manifest(snic_root)
    assert len(completed_snic_inputs(snic_root)) == len(EXPECTED_SNIC_ARCHIVES) * len(
        EXPECTED_SNIC_ISOS
    )

    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["complete"] = False
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    expect_error(lambda: completed_snic_inputs(snic_root), RuntimeError)
    document["complete"] = True
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    partial = snic_root / "interrupted.tiff.part"
    partial.write_bytes(b"")
    expect_error(lambda: completed_snic_inputs(snic_root), RuntimeError)
    partial.unlink()

    first_entry = document["archives"][0]["zip_entries"][0]
    missing = snic_root / EXPECTED_SNIC_ARCHIVES[document["archives"][0]["datafile_id"]] / first_entry[
        "name"
    ]
    missing.unlink()
    expect_error(lambda: completed_snic_inputs(snic_root), RuntimeError)


def test_snic_crc_guard(root: Path) -> None:
    payload = b"SNIC TIFF payload" * 1024
    info = types.SimpleNamespace(
        filename="set/scene/image.tiff",
        file_size=len(payload),
        CRC=zlib.crc32(payload) & 0xFFFFFFFF,
    )

    class FakeRemote:
        def open(self, unused_info):  # noqa: ANN001
            assert unused_info is info
            return io.BytesIO(payload)

    destination = root / info.filename
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    stale_partial = destination.with_suffix(".tiff.part")
    stale_partial.write_bytes(b"stale")
    assert extract_entry(FakeRemote(), info, root) == "existing"
    assert not stale_partial.exists()

    destination.write_bytes(b"X" * len(payload))
    assert extract_entry(FakeRemote(), info, root) == "downloaded"
    assert destination.read_bytes() == payload


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="domain-preparation-test-") as temporary:
        root = Path(temporary)
        test_paths_and_complete_markers(root)
        test_deterministic_crops()
        test_uhd_hybrid_preflight(root)
        test_snic_completion_guard(root)
        test_snic_crc_guard(root / "crc")
    print("UHD-LL/SNIC domain-preparation checks passed")


if __name__ == "__main__":
    main()
