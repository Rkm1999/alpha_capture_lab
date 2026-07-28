#!/usr/bin/env python3
"""Fast contracts for UHD/SNIC photometry, loss weighting, and sampling."""

from __future__ import annotations

import math
import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader

import build_mixed_cache as build_mixed_cache_module
from prepare_domain_dataset import (
    alignment_gate,
    apply_local_gain,
    build_local_gain_field,
    build_uhd_hybrid_target,
    gaussian_rgb,
    srgb_to_linear,
)
from build_mixed_cache import (
    cache_content_identity,
    configured_sample_weight,
    main as build_mixed_cache_main,
    require_preprocessing,
    require_teacher_hash,
    safe_relative_path,
    validate_synthetic_acceptance,
)
from src.losses import compute_distillation_loss
from src.mixed_dataset import (
    MixedDistillationDataset,
    MixedManifestRecord,
    MixedManifestSnapshot,
)
from src.noise_conditioning import (
    conditioned_input,
    estimate_noise_strength,
    model_input_from_config,
)
from src.student import LiteDenoiseNet
from src.weighted_losses import compute_weighted_distillation_loss
from src.width_expansion import expansion_gradient_masks
from train_mixed import (
    apply_paired_kd_weight_override,
    balanced_sample_weights,
    blend_sample_weights,
    correction_loss_config_for_epoch,
    difficulty_strength_for_epoch,
    load_compatible_model_state,
    load_model_only_checkpoint,
    rank_validation,
    run_fingerprint,
    save_ranked_checkpoints,
    state_dict_digest,
    stratified_validation_indices,
    target_correction_tensors,
    target_record_selected,
    update_early_stopping,
    validate,
    validate_mixed_array_integrity,
    validate_training_contract,
    pin_verified_synthetic_arrays,
)
from validate_synthetic_camera_jpeg import (
    ANALYSIS_SCHEMA_VERSION,
    VALIDATOR_SEMANTIC_VERSION,
    active_generator_identity,
    active_validator_identity,
    cache_content_identity as gate_cache_content_identity,
)


@dataclass
class Record:
    dataset: str
    sample_weight: float = 1.0


@dataclass
class ContractRecord:
    dataset: str
    split: str
    gt_weight: float
    kd_weight: float
    iso: int = -1


def expect_raises(error_type: type[BaseException], callback) -> None:
    try:
        callback()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def main() -> None:
    expansion_source = {
        "input_conv.weight": torch.zeros(24, 3, 3, 3),
        "input_conv.bias": torch.zeros(24),
    }
    expansion_target = {
        "input_conv.weight": torch.zeros(32, 3, 3, 3),
        "input_conv.bias": torch.zeros(32),
    }
    expansion_masks = expansion_gradient_masks(
        expansion_source,
        expansion_target,
        24,
        32,
    )
    assert not bool(expansion_masks["input_conv.weight"][:24].any())
    assert bool(expansion_masks["input_conv.weight"][24:].all())
    assert not bool(expansion_masks["input_conv.bias"][:24].any())
    assert bool(expansion_masks["input_conv.bias"][24:].all())
    generator = torch.Generator().manual_seed(710)
    student = torch.rand((2, 3, 16, 16), generator=generator, requires_grad=True)
    teacher = torch.rand((2, 3, 16, 16), generator=generator)
    clean = torch.rand((2, 3, 16, 16), generator=generator)
    paper = compute_distillation_loss(student, teacher, clean, alpha=0.7)
    paired = compute_weighted_distillation_loss(
        student,
        teacher,
        clean,
        torch.ones(2),
        torch.full((2,), 0.7),
        alpha=0.7,
    )
    assert torch.allclose(paired.total, paper.total)
    assert torch.allclose(paired.gt_mse, paper.gt_mse)
    assert torch.allclose(paired.gt_l1, paper.gt_l1)
    assert torch.allclose(paired.kd_mse, paper.kd_mse * 0.7)

    per_kd = (student - teacher).square().flatten(1).mean(1)
    teacher_only = compute_weighted_distillation_loss(
        student,
        teacher,
        clean,
        torch.zeros(2),
        torch.ones(2),
        alpha=0.7,
    )
    assert torch.allclose(teacher_only.total, 1000.0 * per_kd.mean())
    assert teacher_only.gt_mse == 0.0 and teacher_only.gt_l1 == 0.0
    teacher_only.total.backward()
    assert torch.isfinite(student.grad).all()
    correction_config = {
        "enabled": True,
        "shadow_luminance_threshold": 0.25,
        "fine_sigma": 1.0,
        "medium_sigma": 2.0,
        "coarse_sigma": 3.0,
        "very_coarse_sigma": 4.0,
        "shadow_lambda": 10.0,
        "medium_coarse_lambda": 20.0,
        "normalized_shadow_lambda": 0.08,
        "normalized_shadow_chroma_lambda": 0.12,
        "shadow_medium_coarse_chroma_lambda": 20.0,
        "flat_shadow_chroma_lambda": 25.0,
        "flat_gradient_threshold": 0.03,
        "very_coarse_chroma_lambda": 25.0,
        "row_column_chroma_lambda": 10.0,
        "normalized_very_coarse_chroma_lambda": 0.25,
        "normalized_row_column_chroma_lambda": 0.25,
        "target_chroma_magnitude_floor": 0.0001,
        "pyramid_chroma_lambda": 20.0,
        "pyramid_chroma_scales": [1, 2, 4, 8],
        "paired_detail_lambda": 5.0,
    }
    correction = compute_weighted_distillation_loss(
        student,
        teacher,
        clean,
        torch.ones(2),
        torch.full((2,), 0.7),
        alpha=0.7,
        noisy=torch.zeros_like(student),
        correction_config=correction_config,
    )
    assert correction.total > paired.total
    assert correction.shadow_kd_l1 > 0.0
    assert correction.medium_coarse_kd_l1 > 0.0
    assert correction.normalized_shadow_kd_l1 > 0.0
    assert correction.normalized_shadow_chroma_kd_l1 > 0.0
    assert correction.shadow_medium_coarse_chroma_kd_l1 > 0.0
    assert correction.flat_shadow_chroma_kd_l1 > 0.0
    assert correction.very_coarse_chroma_kd_l1 > 0.0
    assert correction.row_column_chroma_kd_l1 > 0.0
    assert correction.normalized_very_coarse_chroma_kd_l1 > 0.0
    assert correction.normalized_row_column_chroma_kd_l1 > 0.0
    assert correction.pyramid_chroma_kd_l1 > 0.0
    assert correction.paired_detail_l1 > 0.0
    width24 = LiteDenoiseNet(base_width=24)
    assert sum(parameter.numel() for parameter in width24.parameters()) == 4_415_643
    conditioned_width24 = LiteDenoiseNet(base_width=24, input_channels=4)
    assert sum(parameter.numel() for parameter in conditioned_width24.parameters()) == 4_415_859
    assert (
        load_compatible_model_state(conditioned_width24, width24.state_dict())
        == "rgb_to_noise_conditioned_zero_initialized"
    )
    assert torch.count_nonzero(conditioned_width24.input_conv.weight[:, 3]) == 0
    test_noisy = torch.rand((2, 3, 192, 192), generator=generator)
    strength = estimate_noise_strength(test_noisy, {})
    assert strength.shape == (2,) and bool(((strength >= 0) & (strength <= 1)).all())
    conditioned_output = conditioned_width24(conditioned_input(test_noisy, strength))
    assert conditioned_output.shape == test_noisy.shape
    adapter_width24 = LiteDenoiseNet(
        base_width=24,
        input_channels=4,
        noise_adapter_channels=12,
    )
    assert sum(parameter.numel() for parameter in adapter_width24.parameters()) == 4_418_682
    assert (
        load_compatible_model_state(adapter_width24, width24.state_dict())
        == "backbone_to_residual_adapter_zero_initialized"
    )
    with torch.inference_mode():
        baseline_output = width24(test_noisy)
        adapter_output = adapter_width24(conditioned_input(test_noisy, strength))
    assert torch.equal(adapter_output, baseline_output)
    spatial_config = {
        "enabled": True,
        "spatial_maps": {
            "enabled": True,
            "shadow_luminance_limit": 0.5,
            "chroma_kernel_size": 9,
            "chroma_scale": 0.04,
        },
    }
    spatial_input = model_input_from_config(
        test_noisy,
        {"noise_conditioning": spatial_config},
    )
    assert spatial_input.shape == (2, 6, 192, 192)
    assert bool(((spatial_input[:, 4:] >= 0) & (spatial_input[:, 4:] <= 1)).all())
    multiscale_width24 = LiteDenoiseNet(
        base_width=24,
        input_channels=6,
        noise_adapter_channels=12,
        multiscale_adapter_channels=16,
        multiscale_spatial_gate=True,
        multiscale_chroma_floor=0.15,
    )
    chroma_head_width24 = LiteDenoiseNet(
        base_width=24,
        input_channels=6,
        noise_adapter_channels=12,
        multiscale_adapter_channels=16,
        multiscale_spatial_gate=True,
        chroma_head_channels=24,
    )
    assert (
        load_compatible_model_state(
            chroma_head_width24, multiscale_width24.state_dict()
        )
        == "multiscale_to_chroma_head_zero_initialized"
    )
    with torch.inference_mode():
        assert torch.equal(
            chroma_head_width24(spatial_input),
            multiscale_width24(spatial_input),
        )
    global_chroma_head_width24 = LiteDenoiseNet(
        base_width=24,
        input_channels=6,
        noise_adapter_channels=12,
        multiscale_adapter_channels=16,
        multiscale_spatial_gate=True,
        chroma_head_channels=24,
        global_chroma_head_channels=32,
        global_chroma_head_blocks=5,
        global_chroma_head_use_bottleneck=True,
    )
    assert (
        load_compatible_model_state(
            global_chroma_head_width24, chroma_head_width24.state_dict()
        )
        == "chroma_head_to_global_chroma_head_zero_initialized"
    )
    with torch.inference_mode():
        assert torch.equal(
            global_chroma_head_width24(spatial_input),
            chroma_head_width24(spatial_input),
        )
    chroma_unet_head_width24 = LiteDenoiseNet(
        base_width=24,
        input_channels=6,
        noise_adapter_channels=12,
        multiscale_adapter_channels=16,
        multiscale_spatial_gate=True,
        chroma_head_channels=24,
        global_chroma_head_channels=32,
        global_chroma_head_blocks=5,
        global_chroma_head_use_bottleneck=True,
        chroma_unet_head_channels=12,
    )
    assert (
        load_compatible_model_state(
            chroma_unet_head_width24,
            global_chroma_head_width24.state_dict(),
        )
        == "global_to_chroma_unet_head_zero_initialized"
    )
    with torch.inference_mode():
        assert torch.equal(
            chroma_unet_head_width24(spatial_input),
            global_chroma_head_width24(spatial_input),
        )
    chroma_profile_head_width24 = LiteDenoiseNet(
        base_width=24,
        input_channels=6,
        noise_adapter_channels=12,
        multiscale_adapter_channels=16,
        multiscale_spatial_gate=True,
        chroma_head_channels=24,
        global_chroma_head_channels=32,
        global_chroma_head_blocks=5,
        global_chroma_head_use_bottleneck=True,
        chroma_unet_head_channels=12,
        chroma_profile_head_channels=24,
    )
    assert (
        load_compatible_model_state(
            chroma_profile_head_width24,
            chroma_unet_head_width24.state_dict(),
        )
        == "chroma_unet_to_profile_head_zero_initialized"
    )
    with torch.inference_mode():
        assert torch.equal(
            chroma_profile_head_width24(spatial_input),
            chroma_unet_head_width24(spatial_input),
        )
    restored_profile_head_width24 = LiteDenoiseNet(
        base_width=24,
        input_channels=6,
        noise_adapter_channels=12,
        multiscale_adapter_channels=16,
        multiscale_spatial_gate=True,
        chroma_head_channels=24,
        global_chroma_head_channels=32,
        global_chroma_head_blocks=5,
        global_chroma_head_use_bottleneck=True,
        chroma_unet_head_channels=12,
        chroma_profile_head_channels=24,
        chroma_profile_use_restored=True,
    )
    assert (
        load_compatible_model_state(
            restored_profile_head_width24,
            chroma_profile_head_width24.state_dict(),
        )
        == "profile_head_restoration_input_zero_initialized"
    )
    with torch.inference_mode():
        assert torch.equal(
            restored_profile_head_width24(spatial_input),
            chroma_profile_head_width24(spatial_input),
        )
    refined_profile_head_width24 = LiteDenoiseNet(
        base_width=24,
        input_channels=6,
        noise_adapter_channels=12,
        multiscale_adapter_channels=16,
        multiscale_spatial_gate=True,
        chroma_head_channels=24,
        global_chroma_head_channels=32,
        global_chroma_head_blocks=5,
        global_chroma_head_use_bottleneck=True,
        chroma_unet_head_channels=12,
        chroma_profile_head_channels=24,
        chroma_profile_refinement_blocks=2,
    )
    assert (
        load_compatible_model_state(
            refined_profile_head_width24,
            chroma_profile_head_width24.state_dict(),
        )
        == "profile_head_refinement_zero_initialized"
    )
    with torch.inference_mode():
        assert torch.equal(
            refined_profile_head_width24(spatial_input),
            chroma_profile_head_width24(spatial_input),
        )
    chroma_refinement_width24 = LiteDenoiseNet(
        base_width=24,
        input_channels=6,
        noise_adapter_channels=12,
        multiscale_adapter_channels=16,
        multiscale_spatial_gate=True,
        chroma_head_channels=24,
        global_chroma_head_channels=32,
        global_chroma_head_blocks=5,
        global_chroma_head_use_bottleneck=True,
        chroma_unet_head_channels=12,
        chroma_refinement_head_channels=24,
    )
    assert (
        load_compatible_model_state(
            chroma_refinement_width24,
            chroma_unet_head_width24.state_dict(),
        )
        == "chroma_unet_to_refinement_head_zero_initialized"
    )
    with torch.inference_mode():
        assert torch.equal(
            chroma_refinement_width24(spatial_input),
            chroma_unet_head_width24(spatial_input),
        )
    assert sum(parameter.numel() for parameter in multiscale_width24.parameters()) == 4_445_523
    assert (
        load_compatible_model_state(
            multiscale_width24,
            adapter_width24.state_dict(),
        )
        == "residual_adapter_to_multiscale_zero_initialized"
    )
    with torch.inference_mode():
        multiscale_output = multiscale_width24(spatial_input)
    assert torch.equal(multiscale_output, adapter_output)
    for branch in multiscale_width24.multiscale_adapters.values():
        branch[-1].bias.data.fill_(0.01)
    masked_input = spatial_input.clone()
    masked_input[:, 3] = 1.0
    masked_input[:, 4] = 0.0
    with torch.inference_mode():
        masked_baseline = multiscale_width24(masked_input)
    for branch in multiscale_width24.multiscale_adapters.values():
        branch[-1].bias.data.zero_()
    with torch.inference_mode():
        expected_masked = multiscale_width24(masked_input)
    assert torch.equal(masked_baseline, expected_masked)
    expect_raises(ValueError, lambda: LiteDenoiseNet(base_width=15))
    curriculum_config = {
        "training": {
            "difficulty_sampling": {
                "enabled": True,
                "warmup_epochs": 2,
                "ramp_epochs": 4,
                "final_strength": 0.8,
            }
        }
    }
    assert difficulty_strength_for_epoch(curriculum_config, 2) == 0.0
    assert math.isclose(difficulty_strength_for_epoch(curriculum_config, 4), 0.4)
    assert math.isclose(difficulty_strength_for_epoch(curriculum_config, 8), 0.8)
    ordinary_weights = torch.tensor([0.25, 0.75], dtype=torch.double)
    hard_weights = torch.tensor([0.75, 0.25], dtype=torch.double)
    assert torch.equal(
        blend_sample_weights(ordinary_weights, hard_weights, 0.0),
        ordinary_weights,
    )
    assert torch.equal(
        blend_sample_weights(ordinary_weights, hard_weights, 1.0),
        hard_weights,
    )
    expect_raises(
        ValueError,
        lambda: compute_weighted_distillation_loss(
            student.detach(),
            teacher,
            clean,
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 0.7]),
            alpha=0.7,
        ),
    )
    target_diagnostics = [
        ContractRecord("snic_sony", "validation", 1.0, 0.7, 6400),
        ContractRecord("snic_sony", "validation", 1.0, 0.7, 12800),
        ContractRecord("nind", "validation", 1.0, 0.7),
    ]
    target_indices = stratified_validation_indices(
        target_diagnostics,
        maximum_samples=2,
        clean_required={"snic_sony"},
        target_config={
            "enabled": True,
            "datasets": {"nind": {}, "snic_sony": {"isos": [12800]}},
        },
    )
    selected_targets = [target_diagnostics[index] for index in target_indices]
    assert {record.dataset for record in selected_targets} == {"nind", "snic_sony"}
    assert next(record.iso for record in selected_targets if record.dataset == "snic_sony") == 12800

    no_supervision = {
        "dataset": "test",
        "scene": "scene",
        "split": "train",
        "input": "input.npy",
        "clean": "clean.npy",
        "teacher": "teacher.npy",
        "supervision": "none",
        "gt_weight": 0.0,
        "kd_weight": 0.0,
    }
    expect_raises(ValueError, lambda: MixedManifestRecord.from_mapping(no_supervision))
    default_weight = MixedManifestRecord.from_mapping(
        {
            **no_supervision,
            "gt_weight": 1.0,
            "kd_weight": 0.7,
        }
    )
    assert default_weight.sample_weight == 1.0
    assert default_weight.iso is None
    with_iso = MixedManifestRecord.from_mapping(
        {
            **no_supervision,
            "gt_weight": 1.0,
            "kd_weight": 0.7,
            "iso": 12800,
        }
    )
    assert with_iso.iso == 12800
    expect_raises(
        ValueError,
        lambda: MixedManifestRecord.from_mapping(
            {
                **no_supervision,
                "gt_weight": 1.0,
                "kd_weight": 0.7,
                "sample_weight": 0.0,
            }
        ),
    )

    records = [Record("a") for _ in range(90)] + [Record("b") for _ in range(10)]
    weights, counts = balanced_sample_weights(records, {"a": 0.25, "b": 0.75})
    assert counts == {"a": 90, "b": 10}
    assert math.isclose(float(weights[:90].sum()), 0.25, abs_tol=1e-12)
    assert math.isclose(float(weights[90:].sum()), 0.75, abs_tol=1e-12)
    weighted_records = [Record("a", 1.0), Record("a", 3.0), Record("b", 2.0)]
    weights, _ = balanced_sample_weights(weighted_records, {"a": 0.25, "b": 0.75})
    torch.testing.assert_close(
        weights, torch.tensor([0.0625, 0.1875, 0.75], dtype=torch.double)
    )
    assert math.isclose(float(weights[:2].sum()), 0.25, abs_tol=1e-12)
    assert math.isclose(float(weights[2:].sum()), 0.75, abs_tol=1e-12)
    scene_records = [
        type("SceneRecord", (), {"dataset": "a", "scene": "one", "sample_weight": 1.0, "input": "1"})(),
        type("SceneRecord", (), {"dataset": "a", "scene": "one", "sample_weight": 1.0, "input": "2"})(),
        type("SceneRecord", (), {"dataset": "a", "scene": "two", "sample_weight": 1.0, "input": "3"})(),
    ]
    scene_weights, _ = balanced_sample_weights(
        scene_records,
        {"a": 1.0},
        scene_balanced_datasets={"a"},
    )
    torch.testing.assert_close(
        scene_weights,
        torch.tensor([0.25, 0.25, 0.5], dtype=torch.double),
    )
    expect_raises(
        ValueError,
        lambda: balanced_sample_weights(
            [Record("a", 0.0)], {"a": 1.0}
        ),
    )

    sampling_config = {
        "training": {
            "record_sampling": {
                "snic_sony": {
                    "field": "iso",
                    "default": 1.0,
                    "values": {12800: 2.0},
                }
            }
        }
    }
    assert configured_sample_weight(
        {"dataset": "snic_sony", "scene": "a", "split": "train", "iso": 12800},
        sampling_config,
    ) == 2.0
    assert configured_sample_weight(
        {
            "dataset": "snic_sony",
            "scene": "a",
            "split": "validation",
            "iso": 12800,
            "sample_weight": 4.0,
        },
        sampling_config,
    ) == 1.0

    best, stale, improved = update_early_stopping(38.0, float("-inf"), 0, 0.003)
    assert (best, stale, improved) == (38.0, 0, True)
    best, stale, improved = update_early_stopping(38.002, best, stale, 0.003)
    assert (best, stale, improved) == (38.0, 1, False)
    best, stale, improved = update_early_stopping(38.004, best, stale, 0.003)
    assert (best, stale, improved) == (38.004, 0, True)

    noisy = torch.full((2, 3, 16, 16), 0.10)
    teacher = torch.full((2, 3, 16, 16), 0.20)
    perfect_target = target_correction_tensors(
        teacher,
        noisy,
        teacher,
        shadow_luminance_threshold=0.25,
        fine_sigma=0.5,
        medium_sigma=1.0,
        coarse_sigma=2.0,
        very_coarse_sigma=3.0,
    )
    missed_target = target_correction_tensors(
        noisy,
        noisy,
        teacher,
        shadow_luminance_threshold=0.25,
        fine_sigma=0.5,
        medium_sigma=1.0,
        coarse_sigma=2.0,
        very_coarse_sigma=3.0,
    )
    torch.testing.assert_close(
        perfect_target["shadow_teacher_correction_capture"], torch.ones(2)
    )
    torch.testing.assert_close(
        perfect_target["medium_coarse_teacher_correction_capture"], torch.ones(2)
    )
    torch.testing.assert_close(
        perfect_target["shadow_chroma_teacher_correction_capture"], torch.ones(2)
    )
    torch.testing.assert_close(
        perfect_target["medium_coarse_chroma_teacher_correction_capture"],
        torch.ones(2),
    )
    torch.testing.assert_close(
        perfect_target["very_coarse_chroma_teacher_correction_capture"],
        torch.ones(2),
    )
    torch.testing.assert_close(
        perfect_target["row_column_chroma_teacher_correction_capture"],
        torch.ones(2),
    )
    torch.testing.assert_close(
        missed_target["shadow_teacher_correction_capture"], torch.zeros(2), atol=1e-6, rtol=0
    )
    no_shadow_target = target_correction_tensors(
        torch.full((1, 3, 16, 16), 0.60),
        torch.full((1, 3, 16, 16), 0.50),
        torch.full((1, 3, 16, 16), 0.60),
        shadow_luminance_threshold=0.25,
        fine_sigma=0.5,
        medium_sigma=1.0,
        coarse_sigma=2.0,
    )
    assert not bool(no_shadow_target["shadow_sample_valid"].item())
    assert math.isnan(
        float(no_shadow_target["shadow_teacher_correction_capture"].item())
    )

    target_rows = [
        {
            "noisy": torch.full((3, 16, 16), 0.10),
            "clean": torch.full((3, 16, 16), 0.11),
            "teacher": torch.full((3, 16, 16), 0.20),
            "dataset": "real_high_iso",
            "iso": 12800,
            "gt_weight": 1.0,
        },
        {
            "noisy": torch.full((3, 16, 16), 0.50),
            "clean": torch.full((3, 16, 16), 0.51),
            "teacher": torch.full((3, 16, 16), 0.60),
            "dataset": "real_high_iso",
            "iso": 12800,
            "gt_weight": 1.0,
        },
    ]
    target_validation = validate(
        torch.nn.Identity(),
        DataLoader(target_rows, batch_size=2),
        torch.device("cpu"),
        border=0,
        window_size=3,
        sigma=1.0,
        selection_datasets={"real_high_iso"},
        target_config={
            "enabled": True,
            "datasets": {"real_high_iso": {"isos": [12800]}},
            "shadow_luminance_threshold": 0.25,
            "fine_sigma": 0.5,
            "medium_sigma": 1.0,
            "coarse_sigma": 2.0,
        },
        max_batches=None,
    )["target_validation"]
    assert target_validation["samples"] == 2
    assert target_validation["shadow_contributing_samples"] == 1
    assert (
        target_validation["metric_samples"]["shadow_teacher_correction_capture"]
        == 1
    )
    assert target_validation["by_dataset"]["real_high_iso"]["samples"] == 2
    assert (
        target_validation["by_dataset"]["real_high_iso"][
            "shadow_contributing_samples"
        ]
        == 1
    )
    assert math.isclose(
        target_validation["metrics"]["shadow_teacher_correction_capture"],
        0.0,
        abs_tol=1e-6,
    )

    initialized_ranking = rank_validation(
        {
            "selection_student_psnr": 38.415,
            "target_validation": {"score": 0.780},
        },
        best_psnr=float("-inf"),
        best_target_score=None,
        target_enabled=True,
        general_psnr_guardrail=38.0,
        general_ssim_guardrail=0.90,
        early_stopping_metric="target",
        early_stopping_best=float("-inf"),
        epochs_without_improvement=0,
        early_stopping_min_delta=0.002,
        count_guardrail_failure=False,
    )
    assert initialized_ranking["general_improved"]
    assert initialized_ranking["target_improved"]
    assert initialized_ranking["best_psnr"] == 38.415
    assert initialized_ranking["best_target_score"] == 0.780
    assert initialized_ranking["early_stopping_best"] == 0.780
    assert initialized_ranking["epochs_without_improvement"] == 0
    degraded_ranking = rank_validation(
        {
            "selection_student_psnr": 38.400,
            "target_validation": {"score": 0.779},
        },
        best_psnr=initialized_ranking["best_psnr"],
        best_target_score=initialized_ranking["best_target_score"],
        target_enabled=True,
        general_psnr_guardrail=38.0,
        general_ssim_guardrail=0.90,
        early_stopping_metric="target",
        early_stopping_best=initialized_ranking["early_stopping_best"],
        epochs_without_improvement=initialized_ranking["epochs_without_improvement"],
        early_stopping_min_delta=0.002,
    )
    assert not degraded_ranking["general_improved"]
    assert not degraded_ranking["target_improved"]
    assert degraded_ranking["best_psnr"] == 38.415
    assert degraded_ranking["early_stopping_best"] == 0.780
    assert degraded_ranking["epochs_without_improvement"] == 1

    ineligible_initial_ranking = rank_validation(
        {
            "selection_student_psnr": 37.9,
            "target_validation": {"score": 0.9},
        },
        best_psnr=float("-inf"),
        best_target_score=None,
        target_enabled=True,
        general_psnr_guardrail=38.0,
        general_ssim_guardrail=0.90,
        early_stopping_metric="target",
        early_stopping_best=float("-inf"),
        epochs_without_improvement=0,
        early_stopping_min_delta=0.002,
        count_guardrail_failure=False,
    )
    assert ineligible_initial_ranking["best_target_score"] is None
    assert ineligible_initial_ranking["early_stopping_best"] == float("-inf")
    assert ineligible_initial_ranking["epochs_without_improvement"] == 0
    ssim_ineligible = rank_validation(
        {
            "selection_student_psnr": 38.8,
            "selection_student_ssim": 0.899,
            "target_validation": {"score": 0.95},
        },
        best_psnr=float("-inf"),
        best_target_score=None,
        target_enabled=True,
        general_psnr_guardrail=38.7,
        general_ssim_guardrail=0.900,
        early_stopping_metric="target",
        early_stopping_best=float("-inf"),
        epochs_without_improvement=0,
        early_stopping_min_delta=0.001,
        count_guardrail_failure=False,
    )
    assert not ssim_ineligible["general_guardrail_passed"]
    assert not ssim_ineligible["target_improved"]
    component_ineligible = rank_validation(
        {
            "selection_student_psnr": 38.8,
            "selection_student_ssim": 0.91,
            "target_validation": {
                "score": 0.81,
                "metric_dataset_means": {
                    "shadow_teacher_correction_capture": 0.72,
                    "medium_coarse_teacher_correction_capture": 0.85,
                },
            },
        },
        best_psnr=float("-inf"),
        best_target_score=None,
        target_enabled=True,
        general_psnr_guardrail=38.7,
        general_ssim_guardrail=0.900,
        early_stopping_metric="target",
        early_stopping_best=float("-inf"),
        epochs_without_improvement=0,
        early_stopping_min_delta=0.001,
        target_component_minimums={
            "shadow_teacher_correction_capture": 0.73,
            "medium_coarse_teacher_correction_capture": 0.78,
        },
        count_guardrail_failure=False,
    )
    assert component_ineligible["general_guardrail_passed"]
    assert not component_ineligible["target_component_guardrail_passed"]
    assert not component_ineligible["target_improved"]
    correction_schedule = {
        "enabled": True,
        "warmup_epochs": 2,
        "ramp_epochs": 4,
        "shadow_lambda": 10.0,
        "medium_coarse_lambda": 20.0,
        "paired_detail_lambda": 5.0,
    }
    assert correction_loss_config_for_epoch(correction_schedule, 2)["shadow_lambda"] == 0.0
    assert correction_loss_config_for_epoch(correction_schedule, 3)["shadow_lambda"] == 2.5
    assert correction_loss_config_for_epoch(correction_schedule, 4)["paired_detail_lambda"] == 2.5
    assert correction_loss_config_for_epoch(correction_schedule, 6)["shadow_lambda"] == 10.0
    assert correction_loss_config_for_epoch(correction_schedule, 20)["schedule_scale"] == 1.0
    correction_schedule["initial_scale"] = 0.5
    assert correction_loss_config_for_epoch(correction_schedule, 1)["schedule_scale"] == 0.5
    assert correction_loss_config_for_epoch(correction_schedule, 3)["schedule_scale"] == 0.625
    overridden_kd = apply_paired_kd_weight_override(
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.7, 1.0]),
        0.9,
    )
    assert torch.equal(overridden_kd, torch.tensor([0.9, 1.0]))
    target_config = {
        "enabled": True,
        "datasets": {"nind": {}, "snic_sony": {"isos": [12800]}},
    }
    assert target_record_selected("nind", -1, target_config)
    assert target_record_selected("snic_sony", 12800, target_config)
    assert not target_record_selected("snic_sony", 6400, target_config)
    assert not target_record_selected("synthetic_camera_jpeg", 12800, target_config)

    diagnostic_records = [
        ContractRecord("a", "validation", 0.0, 1.0),
        ContractRecord("a", "validation", 1.0, 0.7),
        ContractRecord("b", "validation", 1.0, 0.7),
        ContractRecord("c", "validation", 0.0, 1.0),
    ]
    diagnostic_indices = stratified_validation_indices(
        diagnostic_records, maximum_samples=3, clean_required={"a", "b"}
    )
    selected_records = [diagnostic_records[index] for index in diagnostic_indices]
    assert {record.dataset for record in selected_records} == {"a", "b", "c"}
    assert all(
        any(record.dataset == dataset and record.gt_weight > 0.0 for record in selected_records)
        for dataset in ("a", "b")
    )
    expect_raises(
        ValueError,
        lambda: stratified_validation_indices(
            diagnostic_records, maximum_samples=2, clean_required={"a", "b"}
        ),
    )

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_model = LiteDenoiseNet()
        initialization_path = root / "initial.pt"
        torch.save(
            {
                "model": source_model.state_dict(),
                "epoch": 195,
                "run_name": "source_run",
                "optimizer": {"must_not": "be_loaded"},
            },
            initialization_path,
        )
        loaded_state, provenance = load_model_only_checkpoint(initialization_path)
        assert state_dict_digest(loaded_state) == state_dict_digest(source_model.state_dict())
        assert provenance["source_epoch"] == 195
        assert provenance["source_run_name"] == "source_run"
        assert provenance["optimizer"] == "reset"
        assert provenance["scheduler"] == "reset"

        initial_state = {"epoch": 0, "model": source_model.state_dict()}
        ranked_run = root / "ranked"
        ranked_run.mkdir()
        save_ranked_checkpoints(
            ranked_run,
            initial_state,
            general_improved=True,
            target_improved=True,
        )
        assert {path.name for path in ranked_run.iterdir()} == {
            "last.pt",
            "best.pt",
            "general-best.pt",
            "target-best.pt",
        }
        for name in ("last.pt", "best.pt", "general-best.pt", "target-best.pt"):
            saved = torch.load(ranked_run / name, map_location="cpu", weights_only=False)
            assert saved["epoch"] == 0

        synthetic_root = root / "synthetic" / "cache"
        synthetic_root.mkdir(parents=True)
        synthetic_record = {
            "id": "synthetic-1",
            "dataset": "synthetic_camera_jpeg",
            "scene": "synthetic-scene",
            "split": "train",
            "input": "train/input.npy",
            "clean": "train/clean.npy",
            "teacher": "train/teacher.npy",
            "supervision": "paired",
            "gt_weight": 1.0,
            "kd_weight": 0.7,
        }
        synthetic_values = {"input": 0.125, "clean": 0.25, "teacher": 0.375}
        for key in ("input", "clean", "teacher"):
            path = synthetic_root / synthetic_record[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(
                path,
                np.full((192, 192, 3), synthetic_values[key], dtype=np.float32),
            )
        synthetic_record["array_sha256"] = {
            key: hashlib.sha256(
                (synthetic_root / synthetic_record[key]).read_bytes()
            ).hexdigest()
            for key in ("input", "clean", "teacher")
        }
        synthetic_manifest = synthetic_root / "manifest.json"
        synthetic_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "preprocessing": "synthetic_camera_jpeg_linear_post_isp_covariance_v3",
                    "teacher_checkpoint_sha256": "1" * 64,
                    "records": [synthetic_record],
                }
            ),
            encoding="utf-8",
        )
        content_identity = cache_content_identity(synthetic_root, [synthetic_record])
        assert content_identity == gate_cache_content_identity(
            synthetic_root, [synthetic_record]
        )
        contact_png = synthetic_root.parent / "contact.png"
        contact_jpg = synthetic_root.parent / "contact.jpg"
        contact_png.write_bytes(b"png")
        contact_jpg.write_bytes(b"jpg")
        analysis = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "semantic_version": VALIDATOR_SEMANTIC_VERSION,
            "validator": active_validator_identity(),
            "generator": active_generator_identity(),
            "manifest": {
                "path": str(synthetic_manifest),
                "sha256": hashlib.sha256(synthetic_manifest.read_bytes()).hexdigest(),
            },
            "preprocessing": "synthetic_camera_jpeg_linear_post_isp_covariance_v3",
            "records_loaded": 1,
            "records_measured": 1,
            "cache_content": content_identity,
            "smoke": {"generation": False, "calibration": False},
            "protected_holdout": {"paths": [], "sha256": []},
            "sources_and_splits": {"fixture": True},
            "reconstruction": {"fixture": True},
            "severity": {"fixture": True},
            "calibration": {"fixture": True},
            "findings": {
                "errors": {"total": 0, "by_code": {}, "examples": {}},
                "warnings": {"total": 0, "by_code": {}, "examples": {}},
            },
            "contact_sheet": {
                "png": str(contact_png),
                "png_sha256": hashlib.sha256(contact_png.read_bytes()).hexdigest(),
                "jpg": str(contact_jpg),
                "jpg_sha256": hashlib.sha256(contact_jpg.read_bytes()).hexdigest(),
            },
        }
        analysis_sha = hashlib.sha256(
            json.dumps(analysis, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        gate_report = synthetic_root.parent / "synthetic_camera_jpeg_gate_report.json"
        visual_acceptance = synthetic_root.parent / "visual_acceptance.json"
        visual_acceptance.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "decision": "accepted",
                    "reviewer": "test",
                    "accepted_at": "2026-07-19T00:00:00Z",
                    "manifest_sha256": hashlib.sha256(
                        synthetic_manifest.read_bytes()
                    ).hexdigest(),
                    "cache_content_sha256": content_identity["sha256"],
                    "contact_png_sha256": hashlib.sha256(contact_png.read_bytes()).hexdigest(),
                    "contact_jpg_sha256": hashlib.sha256(contact_jpg.read_bytes()).hexdigest(),
                    "analysis_sha256": analysis_sha,
                    "reviewed_report_sha256": "2" * 64,
                }
            ),
            encoding="utf-8",
        )
        gate_report.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "status": "accepted",
                    "release_gate_passed": True,
                    "smoke": {"generation": False, "calibration": False},
                    "manifest": {
                        "path": str(synthetic_manifest),
                        "sha256": hashlib.sha256(synthetic_manifest.read_bytes()).hexdigest(),
                    },
                    "cache_content": content_identity,
                    "contact_sheet": {
                        "png": str(contact_png),
                        "png_sha256": hashlib.sha256(contact_png.read_bytes()).hexdigest(),
                        "jpg": str(contact_jpg),
                        "jpg_sha256": hashlib.sha256(contact_jpg.read_bytes()).hexdigest(),
                    },
                    "analysis_sha256": analysis_sha,
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
                    "reviewed_report_sha256": "2" * 64,
                    "visual_acceptance": {
                        "status": "accepted",
                        "reviewer": "test",
                        "accepted_at": "2026-07-19T00:00:00Z",
                        "reviewed_report_sha256": "2" * 64,
                        "sha256": hashlib.sha256(visual_acceptance.read_bytes()).hexdigest(),
                    },
                }
            ),
            encoding="utf-8",
        )
        acceptance = validate_synthetic_acceptance(
            synthetic_root,
            synthetic_manifest,
            [synthetic_record],
            gate_report,
            visual_acceptance,
        )
        assert acceptance["status"] == "accepted"

        accepted_report_bytes = gate_report.read_bytes()
        stale_validator_report = json.loads(accepted_report_bytes)
        stale_validator_report["analysis"]["validator"]["sha256"] = "0" * 64
        stale_validator_report["analysis_sha256"] = hashlib.sha256(
            json.dumps(
                stale_validator_report["analysis"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        gate_report.write_text(json.dumps(stale_validator_report), encoding="utf-8")
        expect_raises(
            RuntimeError,
            lambda: validate_synthetic_acceptance(
                synthetic_root,
                synthetic_manifest,
                [synthetic_record],
                gate_report,
                visual_acceptance,
            ),
        )
        gate_report.write_bytes(accepted_report_bytes)

        incomplete_report = json.loads(accepted_report_bytes)
        incomplete_report["analysis"]["records_loaded"] = 0
        incomplete_report["analysis"]["records_measured"] = 0
        incomplete_report["records_loaded"] = 0
        incomplete_report["records_measured"] = 0
        incomplete_report["analysis_sha256"] = hashlib.sha256(
            json.dumps(
                incomplete_report["analysis"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        gate_report.write_text(json.dumps(incomplete_report), encoding="utf-8")
        expect_raises(
            RuntimeError,
            lambda: validate_synthetic_acceptance(
                synthetic_root,
                synthetic_manifest,
                [synthetic_record],
                gate_report,
                visual_acceptance,
            ),
        )
        gate_report.write_bytes(accepted_report_bytes)

        def write_source_cache(
            cache_root: Path,
            *,
            dataset: str,
            scene: str,
            preprocessing: str,
        ) -> None:
            cache_root.mkdir(parents=True)
            record = {
                "id": f"{dataset}-1",
                "dataset": dataset,
                "scene": scene,
                "split": "train",
                "input": "train/input.npy",
                "clean": "train/clean.npy",
                "teacher": "train/teacher.npy",
                "supervision": "paired",
                "gt_weight": 1.0,
                "kd_weight": 0.7,
            }
            for key in ("input", "clean", "teacher"):
                path = cache_root / record[key]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{dataset}-{key}".encode())
            (cache_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "preprocessing": preprocessing,
                        "teacher_checkpoint_sha256": "1" * 64,
                        "records": [record],
                    }
                ),
                encoding="utf-8",
            )

        legacy_root = root / "legacy"
        domain_root = root / "domain"
        write_source_cache(
            legacy_root,
            dataset="midd",
            scene="legacy-scene",
            preprocessing="legacy_v1",
        )
        write_source_cache(
            domain_root,
            dataset="snic_sony",
            scene="domain-scene",
            preprocessing="domain_v3",
        )
        mixed_config = root / "mixed_config.json"
        mixed_config.write_text(
            json.dumps(
                {
                    "project": {"preprocessing_version": "mixed_targeted_v2"},
                    "data": {
                        "source_preprocessing": {
                            "legacy": "legacy_v1",
                            "domain": "domain_v3",
                            "synthetic": "synthetic_camera_jpeg_linear_post_isp_covariance_v3",
                        }
                    },
                    "training": {},
                }
            ),
            encoding="utf-8",
        )
        mixed_output = root / "mixed"
        original_argv = sys.argv
        sys.argv = [
            "build_mixed_cache.py",
            "--legacy-root",
            str(legacy_root),
            "--domain-root",
            str(domain_root),
            "--synthetic-root",
            str(synthetic_root),
            "--synthetic-gate-report",
            str(gate_report),
            "--synthetic-visual-acceptance",
            str(visual_acceptance),
            "--config",
            str(mixed_config),
            "--output-root",
            str(mixed_output),
        ]
        try:
            with redirect_stdout(io.StringIO()):
                build_mixed_cache_main()
        finally:
            sys.argv = original_argv
        mixed_document = json.loads(
            (mixed_output / "manifest.json").read_text(encoding="utf-8")
        )
        assert {record["mixed_source"] for record in mixed_document["records"]} == {
            "legacy",
            "domain",
            "synthetic",
        }
        assert mixed_document["sources"]["synthetic"]["acceptance"]["status"] == (
            "accepted"
        )
        integrity = validate_mixed_array_integrity(
            mixed_output / "manifest.json", mixed_output
        )
        assert integrity["status"] == "verified"
        assert integrity["synthetic_files_verified"] == 3

        published_manifest_bytes = (mixed_output / "manifest.json").read_bytes()
        published_report_bytes = (mixed_output / "manifest.report.json").read_bytes()
        build_arguments = [
            "build_mixed_cache.py",
            "--legacy-root",
            str(legacy_root),
            "--domain-root",
            str(domain_root),
            "--synthetic-root",
            str(synthetic_root),
            "--synthetic-gate-report",
            str(gate_report),
            "--synthetic-visual-acceptance",
            str(visual_acceptance),
            "--config",
            str(mixed_config),
            "--output-root",
            str(mixed_output),
            "--replace",
        ]
        original_materialize = build_mixed_cache_module.materialize
        materialize_calls = 0

        def fail_during_replacement(source: Path, destination: Path) -> str:
            nonlocal materialize_calls
            materialize_calls += 1
            if materialize_calls == 2:
                raise RuntimeError("injected mixed-cache build interruption")
            return original_materialize(source, destination)

        build_mixed_cache_module.materialize = fail_during_replacement
        sys.argv = build_arguments
        try:
            expect_raises(
                RuntimeError,
                lambda: build_mixed_cache_main(),
            )
        finally:
            build_mixed_cache_module.materialize = original_materialize
            sys.argv = original_argv
        assert (mixed_output / "manifest.json").read_bytes() == published_manifest_bytes
        assert (mixed_output / "manifest.report.json").read_bytes() == published_report_bytes
        assert mixed_output.with_name(f".{mixed_output.name}.building").is_dir()
        assert not mixed_output.with_name(f".{mixed_output.name}.previous").exists()

        sys.argv = build_arguments
        try:
            with redirect_stdout(io.StringIO()):
                build_mixed_cache_main()
        finally:
            sys.argv = original_argv
        assert not mixed_output.with_name(f".{mixed_output.name}.building").exists()
        assert not mixed_output.with_name(f".{mixed_output.name}.previous").exists()
        rebuilt_report = json.loads(
            (mixed_output / "manifest.report.json").read_text(encoding="utf-8")
        )
        assert rebuilt_report["manifest"] == str(mixed_output / "manifest.json")
        assert rebuilt_report["manifest_sha256"] == hashlib.sha256(
            (mixed_output / "manifest.json").read_bytes()
        ).hexdigest()

        mixed_manifest_path = mixed_output / "manifest.json"
        manifest_snapshot = MixedManifestSnapshot.load(mixed_manifest_path)
        pinned_report, pinned_arrays = pin_verified_synthetic_arrays(
            manifest_snapshot, mixed_output
        )
        assert pinned_report["pinning"]["status"] == "active"
        assert pinned_report["manifest_sha256"] == manifest_snapshot.sha256
        assert pinned_arrays is not None and len(pinned_arrays) == 3

        replaced_manifest = json.loads(mixed_manifest_path.read_text(encoding="utf-8"))
        replaced_manifest["records"][0]["scene"] = "replacement-manifest-scene"
        replacement_path = mixed_manifest_path.with_suffix(".replacement.json")
        replacement_path.write_text(json.dumps(replaced_manifest), encoding="utf-8")
        replacement_path.replace(mixed_manifest_path)
        pinned_dataset = MixedDistillationDataset(
            manifest_snapshot,
            root=mixed_output,
            split="train",
            datasets={"synthetic_camera_jpeg"},
            verified_synthetic_arrays=pinned_arrays,
        )
        assert pinned_dataset.manifest_snapshot is manifest_snapshot
        assert pinned_dataset.manifest_sha256 == manifest_snapshot.sha256
        assert pinned_dataset.records[0].scene == synthetic_record["scene"]
        path_dataset = MixedDistillationDataset(
            mixed_manifest_path,
            root=mixed_output,
            split="train",
        )
        assert any(
            record.scene == "replacement-manifest-scene"
            for record in path_dataset.records
        )
        mixed_manifest_path.write_bytes(manifest_snapshot.payload)
        mixed_synthetic_input = mixed_output / "synthetic" / synthetic_record["input"]
        original_payload = mixed_synthetic_input.read_bytes()
        replacement = mixed_synthetic_input.with_suffix(".replacement.npy")
        np.save(replacement, np.full((192, 192, 3), 0.875, dtype=np.float32))
        replacement.replace(mixed_synthetic_input)
        # The worker must still consume the verified inode, not the replacement path.
        pinned_batch = next(
            iter(DataLoader(pinned_dataset, batch_size=1, num_workers=2))
        )
        assert torch.allclose(
            pinned_batch["noisy"], torch.full_like(pinned_batch["noisy"], 0.125)
        )
        # An in-place change to the pinned inode is detected before NumPy decodes it.
        source_synthetic_input = synthetic_root / synthetic_record["input"]
        source_synthetic_input.write_bytes(b"changed-pinned-inode")
        expect_raises(RuntimeError, lambda: pinned_dataset[0])
        source_synthetic_input.write_bytes(original_payload)
        restore = mixed_synthetic_input.with_suffix(".restore.npy")
        restore.write_bytes(original_payload)
        restore.replace(mixed_synthetic_input)
        assert torch.allclose(
            pinned_dataset[0]["noisy"],
            torch.full_like(pinned_dataset[0]["noisy"], 0.125),
        )
        pinned_arrays.close()

        original_mixed_manifest = mixed_manifest_path.read_text(encoding="utf-8")
        missing_hash_document = json.loads(original_mixed_manifest)
        next(
            record
            for record in missing_hash_document["records"]
            if record["mixed_source"] == "synthetic"
        ).pop("array_sha256")
        mixed_manifest_path.write_text(
            json.dumps(missing_hash_document), encoding="utf-8"
        )
        expect_raises(
            RuntimeError,
            lambda: validate_mixed_array_integrity(mixed_manifest_path, mixed_output),
        )
        mixed_manifest_path.write_text(original_mixed_manifest, encoding="utf-8")

        original_payload = mixed_synthetic_input.read_bytes()
        mixed_synthetic_input.write_bytes(b"mutated-after-acceptance")
        expect_raises(
            RuntimeError,
            lambda: validate_mixed_array_integrity(mixed_manifest_path, mixed_output),
        )
        mixed_synthetic_input.write_bytes(original_payload)
        assert validate_mixed_array_integrity(mixed_manifest_path, mixed_output)[
            "status"
        ] == "verified"

        report_document = json.loads(gate_report.read_text(encoding="utf-8"))
        report_document["status"] = "pending_visual_review"
        gate_report.write_text(json.dumps(report_document), encoding="utf-8")
        expect_raises(
            RuntimeError,
            lambda: validate_synthetic_acceptance(
                synthetic_root,
                synthetic_manifest,
                [synthetic_record],
                gate_report,
                visual_acceptance,
            ),
        )
        report_document["status"] = "accepted"
        report_document["release_gate_passed"] = True
        report_document["smoke"] = {"generation": True, "calibration": False}
        gate_report.write_text(json.dumps(report_document), encoding="utf-8")
        expect_raises(
            RuntimeError,
            lambda: validate_synthetic_acceptance(
                synthetic_root,
                synthetic_manifest,
                [synthetic_record],
                gate_report,
                visual_acceptance,
            ),
        )

        checkpoint = root / "teacher.pth"
        checkpoint.write_bytes(b"teacher checkpoint")
        teacher_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "preprocessing": "mixed_v3",
                    "source_preprocessing": {"legacy": "legacy_v1", "domain": "domain_v3"},
                    "teacher_checkpoint_sha256": teacher_hash,
                    "sources": {
                        "legacy": {"preprocessing": "legacy_v1"},
                        "domain": {"preprocessing": "domain_v3"},
                    },
                    "records": [],
                }
            ),
            encoding="utf-8",
        )
        config = {
            "project": {"preprocessing_version": "mixed_v3", "seed": 1337},
            "model": {},
            "metrics": {},
            "teacher": {"checkpoint": str(checkpoint)},
            "data": {
                "datasets": ["a", "b"],
                "source_preprocessing": {
                    "legacy": "legacy_v1",
                    "domain": "domain_v3",
                },
            },
            "training": {
                "alpha": 0.7,
                "dataset_sampling_weights": {"a": 0.5, "b": 0.5},
                "selection_datasets": ["a", "b"],
            },
        }
        train_records = [
            ContractRecord("a", "train", 1.0, 0.7),
            ContractRecord("b", "train", 0.0, 1.0),
        ]
        validation_records = [
            ContractRecord("a", "validation", 1.0, 0.7),
            ContractRecord("b", "validation", 1.0, 0.7),
        ]
        contract = validate_training_contract(
            config, manifest, train_records, validation_records
        )
        assert contract["cached_teacher_checkpoint_sha256"] == teacher_hash
        contract_snapshot = MixedManifestSnapshot.load(manifest)
        original_fingerprint, original_fingerprint_payload = run_fingerprint(
            config,
            contract_snapshot,
            1,
            None,
            None,
        )
        bad_weights = train_records + [ContractRecord("a", "train", 1.0, 1.0)]
        expect_raises(
            ValueError,
            lambda: validate_training_contract(
                config, manifest, bad_weights, validation_records
            ),
        )
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "preprocessing": "mixed_v3",
                    "source_preprocessing": {"legacy": "legacy_v1", "domain": "domain_v3"},
                    "teacher_checkpoint_sha256": "0" * 64,
                    "sources": {
                        "legacy": {"preprocessing": "legacy_v1"},
                        "domain": {"preprocessing": "domain_v3"},
                    },
                    "records": [],
                }
            ),
            encoding="utf-8",
        )
        snapshot_contract = validate_training_contract(
            config,
            contract_snapshot,
            train_records,
            validation_records,
        )
        assert snapshot_contract["manifest_sha256"] == contract_snapshot.sha256
        snapshot_fingerprint, snapshot_fingerprint_payload = run_fingerprint(
            config,
            contract_snapshot,
            1,
            None,
            None,
        )
        assert snapshot_fingerprint == original_fingerprint
        assert snapshot_fingerprint_payload == original_fingerprint_payload
        path_fingerprint, path_fingerprint_payload = run_fingerprint(
            config,
            manifest,
            1,
            None,
            None,
        )
        assert path_fingerprint != original_fingerprint
        assert path_fingerprint_payload["manifest_sha256"] != contract_snapshot.sha256
        expect_raises(
            ValueError,
            lambda: validate_training_contract(
                config, manifest, train_records, validation_records
            ),
        )
        assert require_teacher_hash(
            {"teacher_checkpoint_sha256": teacher_hash}, manifest
        ) == teacher_hash
        assert require_preprocessing({"preprocessing": "domain_v3"}, manifest) == "domain_v3"
        expect_raises(
            RuntimeError,
            lambda: require_preprocessing({}, manifest),
        )
        stale_manifest = {
            "schema_version": 2,
            "preprocessing": "mixed_v2",
            "source_preprocessing": {"legacy": "legacy_v1", "domain": "domain_v2"},
            "teacher_checkpoint_sha256": teacher_hash,
            "sources": {
                "legacy": {"preprocessing": "legacy_v1"},
                "domain": {"preprocessing": "domain_v2"},
            },
            "records": [],
        }
        manifest.write_text(json.dumps(stale_manifest), encoding="utf-8")
        expect_raises(
            ValueError,
            lambda: validate_training_contract(
                config, manifest, train_records, validation_records
            ),
        )
        expect_raises(
            ValueError,
            lambda: safe_relative_path(
                "../escape.npy", manifest_path=manifest, field="input"
            ),
        )

    rng = np.random.default_rng(99)
    clean_rgb = rng.uniform(0.03, 0.85, (192, 192, 3)).astype(np.float32)
    expected_gain = np.asarray([0.20, 0.14, 0.09], dtype=np.float32)
    noisy_rgb = apply_local_gain(clean_rgb, np.broadcast_to(expected_gain, clean_rgb.shape))
    target_config = {
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
    gain = build_local_gain_field(noisy_rgb, clean_rgb, 192, target_config)
    recovered = apply_local_gain(clean_rgb, gain)
    assert alignment_gate(noisy_rgb, recovered, target_config)["passed"]
    teacher_rgb = np.clip(
        noisy_rgb + rng.normal(0.0, 0.001, noisy_rgb.shape), 0.0, 1.0
    ).astype(np.float32)
    hybrid = build_uhd_hybrid_target(teacher_rgb, recovered, target_config)
    lowpass_error = np.mean(
        np.abs(
            gaussian_rgb(srgb_to_linear(hybrid), 8.0)
            - gaussian_rgb(srgb_to_linear(teacher_rgb), 8.0)
        )
    )
    assert float(lowpass_error) < 0.003
    print("UHD/SNIC mixed-training smoke checks passed")


if __name__ == "__main__":
    main()
