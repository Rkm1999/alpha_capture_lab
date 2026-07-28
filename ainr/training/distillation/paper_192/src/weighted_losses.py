"""Per-sample clean/KD weighting for mixed paired and teacher-only data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class WeightedLossTerms:
    total: torch.Tensor
    gt_mse: torch.Tensor
    kd_mse: torch.Tensor
    gt_l1: torch.Tensor
    shadow_kd_l1: torch.Tensor
    medium_coarse_kd_l1: torch.Tensor
    normalized_shadow_kd_l1: torch.Tensor
    normalized_shadow_chroma_kd_l1: torch.Tensor
    shadow_medium_coarse_chroma_kd_l1: torch.Tensor
    flat_shadow_chroma_kd_l1: torch.Tensor
    very_coarse_chroma_kd_l1: torch.Tensor
    row_column_chroma_kd_l1: torch.Tensor
    normalized_very_coarse_chroma_kd_l1: torch.Tensor
    weighted_very_coarse_chroma_kd_l1: torch.Tensor
    normalized_row_column_chroma_kd_l1: torch.Tensor
    normalized_very_coarse_chroma_kd_mse: torch.Tensor
    normalized_row_column_chroma_kd_mse: torch.Tensor
    pyramid_chroma_kd_l1: torch.Tensor
    paired_detail_l1: torch.Tensor
    gt_weight_mean: torch.Tensor
    kd_weight_mean: torch.Tensor


def _gaussian_blur(value: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError(f"Gaussian sigma must be positive, got {sigma}")
    radius = max(1, int(3.0 * sigma + 0.999999))
    if radius >= min(value.shape[-2:]):
        raise ValueError("Gaussian kernel is too large for the training tile")
    positions = torch.arange(-radius, radius + 1, device=value.device, dtype=value.dtype)
    kernel = torch.exp(-(positions * positions) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = value.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    blurred = F.conv2d(
        F.pad(value, (radius, radius, 0, 0), mode="reflect"),
        horizontal,
        groups=channels,
    )
    return F.conv2d(
        F.pad(blurred, (0, 0, radius, radius), mode="reflect"),
        vertical,
        groups=channels,
    )


def _chroma_projection(value: torch.Tensor) -> torch.Tensor:
    luma = value.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    projected_luma = (value * luma).sum(1, keepdim=True) * (
        luma / luma.square().sum()
    )
    return value - projected_luma


def _ycbcr_chroma(value: torch.Tensor) -> torch.Tensor:
    matrix = value.new_tensor(
        (
            (-0.114572, -0.385428, 0.5),
            (0.5, -0.454153, -0.045847),
        )
    )
    return torch.einsum("oc,bchw->bohw", matrix, value)


def _pyramid_chroma_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    kd_weight: torch.Tensor,
    config: dict[str, Any],
    noise_strength: torch.Tensor | None,
) -> torch.Tensor:
    scales = tuple(int(value) for value in config.get("pyramid_chroma_scales", (1, 2, 4, 8)))
    if not scales or scales[0] != 1 or any(scale < 1 for scale in scales):
        raise ValueError("pyramid_chroma_scales must start at 1 and contain positive integers")
    if len(scales) != len(set(scales)):
        raise ValueError("pyramid_chroma_scales must not contain duplicates")
    chroma_error = _ycbcr_chroma(
        student_output.float()
    ) - _ycbcr_chroma(teacher_output.float())
    per_sample = chroma_error.new_zeros((chroma_error.shape[0],))
    for scale in scales:
        scaled = (
            chroma_error
            if scale == 1
            else F.avg_pool2d(chroma_error, kernel_size=scale, stride=scale)
        )
        per_sample = per_sample + scaled.abs().flatten(1).mean(1)
    per_sample = per_sample / len(scales)
    effective_weight = kd_weight.float()
    if noise_strength is not None:
        from .noise_conditioning import severity_gate

        effective_weight = effective_weight * severity_gate(noise_strength, config)
    return (per_sample * effective_weight).mean()


def _correction_losses(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    noisy: torch.Tensor,
    kd_weight: torch.Tensor,
    config: dict[str, Any],
    noise_strength: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    threshold = float(config["shadow_luminance_threshold"])
    fine_sigma = float(config["fine_sigma"])
    medium_sigma = float(config["medium_sigma"])
    coarse_sigma = float(config["coarse_sigma"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("shadow_luminance_threshold must be in (0,1)")
    if not 0.0 < fine_sigma < medium_sigma < coarse_sigma:
        raise ValueError("correction-loss sigmas must satisfy fine < medium < coarse")

    # Compare teacher/student corrections in float32 so AMP does not erase
    # weak shadow residuals and low-frequency differences.
    student = student_output.float()
    teacher = teacher_output.float()
    source = noisy.float()
    correction_error = (student - source) - (teacher - source)
    luma_weights = source.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    shadow = ((source * luma_weights).sum(1, keepdim=True) < threshold).to(source)
    shadow_denominator = (shadow.flatten(1).sum(1) * source.shape[1]).clamp_min(1.0)
    shadow_per_sample = (
        correction_error.abs() * shadow
    ).flatten(1).sum(1) / shadow_denominator
    shadow_loss = (shadow_per_sample * kd_weight.float()).mean()
    teacher_correction = (teacher - source).abs()
    teacher_shadow_per_sample = (
        teacher_correction * shadow
    ).flatten(1).sum(1) / shadow_denominator
    magnitude_floor = float(config.get("correction_magnitude_floor", 0.005))
    magnitude_reference = float(config.get("correction_weight_reference", 0.02))
    magnitude_max = float(config.get("correction_weight_max", 2.0))
    if magnitude_floor <= 0.0 or magnitude_reference <= 0.0 or magnitude_max < 1.0:
        raise ValueError("invalid normalized shadow correction parameters")
    normalized_error = shadow_per_sample / teacher_shadow_per_sample.clamp_min(
        magnitude_floor
    )
    difficulty_weight = (teacher_shadow_per_sample / magnitude_reference).clamp(
        min=0.25, max=magnitude_max
    )
    normalized_shadow_loss = (
        normalized_error.clamp_max(2.0) * difficulty_weight * kd_weight.float()
    ).mean()
    chroma_error = _chroma_projection(correction_error)
    teacher_chroma_residual = _chroma_projection(teacher - source)
    teacher_chroma_correction = teacher_chroma_residual.abs()
    masked_chroma_error = (
        chroma_error.abs() * shadow
    ).flatten(1).sum(1) / shadow_denominator
    teacher_shadow_chroma = (
        teacher_chroma_correction * shadow
    ).flatten(1).sum(1) / shadow_denominator
    normalized_shadow_chroma = (
        masked_chroma_error
        / teacher_shadow_chroma.clamp_min(magnitude_floor)
    )
    chroma_difficulty = (
        teacher_shadow_chroma / magnitude_reference
    ).clamp(min=0.25, max=magnitude_max)
    normalized_shadow_chroma_loss = (
        normalized_shadow_chroma.clamp_max(2.0)
        * chroma_difficulty
        * kd_weight.float()
    ).mean()

    fine = _gaussian_blur(correction_error, fine_sigma)
    medium = _gaussian_blur(correction_error, medium_sigma)
    coarse = _gaussian_blur(correction_error, coarse_sigma)
    medium_band = fine - medium
    coarse_band = medium - coarse
    multiscale_per_sample = (
        medium_band.abs().flatten(1).mean(1)
        + coarse_band.abs().flatten(1).mean(1)
    )
    multiscale_loss = (multiscale_per_sample * kd_weight.float()).mean()
    chroma_fine = _gaussian_blur(chroma_error, fine_sigma)
    chroma_medium = _gaussian_blur(chroma_error, medium_sigma)
    chroma_coarse = _gaussian_blur(chroma_error, coarse_sigma)
    shadow_chroma_bands = (
        (chroma_fine - chroma_medium).abs()
        + (chroma_medium - chroma_coarse).abs()
    )
    shadow_medium_coarse_chroma_per_sample = (
        shadow_chroma_bands * shadow
    ).flatten(1).sum(1) / shadow_denominator
    extreme_weight = kd_weight.float()
    if noise_strength is not None:
        from .noise_conditioning import severity_gate

        if noise_strength.shape != kd_weight.shape:
            raise ValueError("noise strength and KD weight shapes must match")
        extreme_weight = extreme_weight * severity_gate(noise_strength, config)
    shadow_medium_coarse_chroma_loss = (
        shadow_medium_coarse_chroma_per_sample * extreme_weight
    ).mean()
    teacher_luma = (teacher * luma_weights).sum(1, keepdim=True)
    horizontal_gradient = F.pad(
        (teacher_luma[..., 1:] - teacher_luma[..., :-1]).abs(),
        (0, 1, 0, 0),
        mode="replicate",
    )
    vertical_gradient = F.pad(
        (teacher_luma[..., 1:, :] - teacher_luma[..., :-1, :]).abs(),
        (0, 0, 0, 1),
        mode="replicate",
    )
    flat_gradient_threshold = float(config.get("flat_gradient_threshold", 0.03))
    if not 0.0 < flat_gradient_threshold < 1.0:
        raise ValueError("flat_gradient_threshold must be in (0,1)")
    flat_shadow = (
        shadow
        * (
            0.5 * (horizontal_gradient + vertical_gradient)
            < flat_gradient_threshold
        ).to(source)
    )
    flat_shadow_denominator = (
        flat_shadow.flatten(1).sum(1) * source.shape[1]
    ).clamp_min(1.0)
    flat_shadow_error = (
        chroma_error.abs() * flat_shadow
    ).flatten(1).sum(1) / flat_shadow_denominator
    teacher_flat_shadow_chroma = (
        teacher_chroma_correction * flat_shadow
    ).flatten(1).sum(1) / flat_shadow_denominator
    flat_shadow_normalized_error = (
        flat_shadow_error
        / teacher_flat_shadow_chroma.clamp_min(magnitude_floor)
    )
    flat_shadow_difficulty = (
        teacher_flat_shadow_chroma / magnitude_reference
    ).clamp(min=0.25, max=magnitude_max)
    flat_shadow_chroma_loss = (
        flat_shadow_normalized_error.clamp_max(2.0)
        * flat_shadow_difficulty
        * extreme_weight
    ).mean()
    very_coarse_sigma = config.get("very_coarse_sigma")
    if very_coarse_sigma is None:
        very_coarse_chroma_loss = student_output.new_zeros(())
        row_column_chroma_loss = student_output.new_zeros(())
        normalized_very_coarse_chroma_loss = student_output.new_zeros(())
        weighted_very_coarse_chroma_loss = student_output.new_zeros(())
        normalized_row_column_chroma_loss = student_output.new_zeros(())
        normalized_very_coarse_chroma_mse = student_output.new_zeros(())
        normalized_row_column_chroma_mse = student_output.new_zeros(())
    else:
        very_coarse_sigma = float(very_coarse_sigma)
        if very_coarse_sigma <= coarse_sigma:
            raise ValueError("very_coarse_sigma must be greater than coarse_sigma")
        chroma_coarse = _gaussian_blur(chroma_error, coarse_sigma)
        chroma_very_coarse = _gaussian_blur(chroma_error, very_coarse_sigma)
        very_coarse_per_sample = (
            chroma_coarse - chroma_very_coarse
        ).abs().flatten(1).mean(1)
        very_coarse_chroma_loss = (
            very_coarse_per_sample * extreme_weight
        ).mean()
        teacher_chroma_coarse = _gaussian_blur(
            teacher_chroma_residual, coarse_sigma
        )
        teacher_chroma_very_coarse = _gaussian_blur(
            teacher_chroma_residual, very_coarse_sigma
        )
        teacher_very_coarse_per_sample = (
            teacher_chroma_coarse - teacher_chroma_very_coarse
        ).abs().flatten(1).mean(1)
        teacher_very_coarse = (
            teacher_chroma_coarse - teacher_chroma_very_coarse
        )
        target_magnitude_floor = float(
            config.get("target_chroma_magnitude_floor", 0.0001)
        )
        if target_magnitude_floor <= 0.0:
            raise ValueError("target_chroma_magnitude_floor must be positive")
        normalized_l1_max_ratio = float(
            config.get("normalized_weak_band_l1_max_ratio", 2.0)
        )
        normalized_mse_max_ratio = float(
            config.get("normalized_weak_band_mse_max_ratio", 4.0)
        )
        if normalized_l1_max_ratio <= 0.0 or normalized_mse_max_ratio <= 0.0:
            raise ValueError("normalized weak-band caps must be positive")
        normalized_very_coarse_chroma_loss = (
            (
                very_coarse_per_sample
                / teacher_very_coarse_per_sample.clamp_min(
                    target_magnitude_floor
                )
            ).clamp_max(normalized_l1_max_ratio)
            * kd_weight.float()
        ).mean()
        teacher_band_importance = teacher_very_coarse.abs().mean(
            dim=1, keepdim=True
        )
        importance_reference = teacher_band_importance.flatten(1).mean(1).view(
            -1, 1, 1, 1
        )
        teacher_band_weight = (
            teacher_band_importance
            / importance_reference.clamp_min(target_magnitude_floor)
        ).clamp(min=0.25, max=4.0)
        weighted_error_per_sample = (
            (chroma_coarse - chroma_very_coarse).abs() * teacher_band_weight
        ).flatten(1).mean(1)
        weighted_teacher_per_sample = (
            teacher_very_coarse.abs() * teacher_band_weight
        ).flatten(1).mean(1)
        weighted_very_coarse_chroma_loss = (
            (
                weighted_error_per_sample
                / weighted_teacher_per_sample.clamp_min(
                    target_magnitude_floor
                )
            ).clamp_max(normalized_l1_max_ratio)
            * kd_weight.float()
        ).mean()
        row_profile = chroma_error.mean(dim=3).abs().flatten(1).mean(1)
        column_profile = chroma_error.mean(dim=2).abs().flatten(1).mean(1)
        row_column_chroma_loss = (
            0.5 * (row_profile + column_profile) * extreme_weight
        ).mean()
        teacher_row_profile = (
            teacher_chroma_residual.mean(dim=3).abs().flatten(1).mean(1)
        )
        teacher_column_profile = (
            teacher_chroma_residual.mean(dim=2).abs().flatten(1).mean(1)
        )
        row_column_per_sample = 0.5 * (row_profile + column_profile)
        teacher_row_column_per_sample = 0.5 * (
            teacher_row_profile + teacher_column_profile
        )
        normalized_row_column_chroma_loss = (
            (
                row_column_per_sample
                / teacher_row_column_per_sample.clamp_min(
                    target_magnitude_floor
                )
            ).clamp_max(normalized_l1_max_ratio)
            * kd_weight.float()
        ).mean()
        normalized_very_coarse_chroma_mse = (
            (
                (chroma_coarse - chroma_very_coarse).square().flatten(1).mean(1)
                / teacher_very_coarse.square().flatten(1).mean(1).clamp_min(
                    target_magnitude_floor * target_magnitude_floor
                )
            ).clamp_max(normalized_mse_max_ratio)
            * kd_weight.float()
        ).mean()
        row_error_mse = chroma_error.mean(dim=3).square().flatten(1).mean(1)
        column_error_mse = chroma_error.mean(dim=2).square().flatten(1).mean(1)
        teacher_row_mse = (
            teacher_chroma_residual.mean(dim=3).square().flatten(1).mean(1)
        )
        teacher_column_mse = (
            teacher_chroma_residual.mean(dim=2).square().flatten(1).mean(1)
        )
        normalized_row_column_chroma_mse = (
            (
                0.5 * (row_error_mse + column_error_mse)
                / (
                    0.5 * (teacher_row_mse + teacher_column_mse)
                ).clamp_min(target_magnitude_floor * target_magnitude_floor)
            ).clamp_max(normalized_mse_max_ratio)
            * kd_weight.float()
        ).mean()
    return (
        shadow_loss,
        multiscale_loss,
        normalized_shadow_loss,
        normalized_shadow_chroma_loss,
        shadow_medium_coarse_chroma_loss,
        flat_shadow_chroma_loss,
        very_coarse_chroma_loss,
        row_column_chroma_loss,
        normalized_very_coarse_chroma_loss,
        weighted_very_coarse_chroma_loss,
        normalized_row_column_chroma_loss,
        normalized_very_coarse_chroma_mse,
        normalized_row_column_chroma_mse,
    )


def _paired_detail_loss(
    student_output: torch.Tensor,
    clean: torch.Tensor,
    gt_weight: torch.Tensor,
) -> torch.Tensor:
    student = student_output.float()
    target = clean.float()
    horizontal = (student[..., 1:] - student[..., :-1]) - (
        target[..., 1:] - target[..., :-1]
    )
    vertical = (student[..., 1:, :] - student[..., :-1, :]) - (
        target[..., 1:, :] - target[..., :-1, :]
    )
    per_sample = 0.5 * (
        horizontal.abs().flatten(1).mean(1) + vertical.abs().flatten(1).mean(1)
    )
    return (per_sample * gt_weight.float()).mean()


def compute_weighted_distillation_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    clean: torch.Tensor,
    gt_weight: torch.Tensor,
    kd_weight: torch.Tensor,
    *,
    alpha: float = 0.7,
    mse_scale: float = 1000.0,
    lambda_l1: float = 50.0,
    noisy: torch.Tensor | None = None,
    correction_config: dict[str, Any] | None = None,
    noise_strength: torch.Tensor | None = None,
) -> WeightedLossTerms:
    """Use paper loss on paired rows and full-strength KD on teacher-only rows.

    ``kd_weight`` is an absolute fraction of ``mse_scale``. Paired rows use
    ``alpha`` (0.7), while teacher-only rows use 1.0 so removing unavailable
    clean terms does not also reduce that record's total training influence.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    if student_output.shape != teacher_output.shape or student_output.shape != clean.shape:
        raise ValueError("student, teacher, and clean shapes must match")
    expected = (student_output.shape[0],)
    if gt_weight.shape != expected or kd_weight.shape != expected:
        raise ValueError(f"loss weights must have shape {expected}")
    for name, weight in (("gt", gt_weight), ("kd", kd_weight)):
        if not torch.isfinite(weight).all() or bool((weight < 0).any()) or bool((weight > 1).any()):
            raise ValueError(f"{name}_weight must contain finite values in [0,1]")
    if bool(((gt_weight + kd_weight) <= 0).any()):
        raise ValueError("every sample must have clean or teacher supervision")
    gt_weight = gt_weight.to(student_output)
    kd_weight = kd_weight.to(student_output)
    per_gt_mse = F.mse_loss(student_output, clean, reduction="none").flatten(1).mean(1)
    per_kd_mse = F.mse_loss(student_output, teacher_output, reduction="none").flatten(1).mean(1)
    per_gt_l1 = F.l1_loss(student_output, clean, reduction="none").flatten(1).mean(1)
    gt_mse = (per_gt_mse * gt_weight).mean()
    kd_mse = (per_kd_mse * kd_weight).mean()
    gt_l1 = (per_gt_l1 * gt_weight).mean()
    zero = student_output.new_zeros(())
    shadow_kd_l1 = zero
    medium_coarse_kd_l1 = zero
    normalized_shadow_kd_l1 = zero
    normalized_shadow_chroma_kd_l1 = zero
    shadow_medium_coarse_chroma_kd_l1 = zero
    flat_shadow_chroma_kd_l1 = zero
    very_coarse_chroma_kd_l1 = zero
    row_column_chroma_kd_l1 = zero
    normalized_very_coarse_chroma_kd_l1 = zero
    weighted_very_coarse_chroma_kd_l1 = zero
    normalized_row_column_chroma_kd_l1 = zero
    normalized_very_coarse_chroma_kd_mse = zero
    normalized_row_column_chroma_kd_mse = zero
    pyramid_chroma_kd_l1 = zero
    paired_detail_l1 = zero
    correction_total = zero
    if correction_config and bool(correction_config.get("enabled", False)):
        if noisy is None or noisy.shape != student_output.shape:
            raise ValueError("enabled correction loss requires a matching noisy tensor")
        shadow_lambda = float(correction_config["shadow_lambda"])
        medium_coarse_lambda = float(correction_config["medium_coarse_lambda"])
        if shadow_lambda < 0.0 or medium_coarse_lambda < 0.0:
            raise ValueError("correction-loss coefficients must be non-negative")
        (
            shadow_kd_l1,
            medium_coarse_kd_l1,
            normalized_shadow_kd_l1,
            normalized_shadow_chroma_kd_l1,
            shadow_medium_coarse_chroma_kd_l1,
            flat_shadow_chroma_kd_l1,
            very_coarse_chroma_kd_l1,
            row_column_chroma_kd_l1,
            normalized_very_coarse_chroma_kd_l1,
            weighted_very_coarse_chroma_kd_l1,
            normalized_row_column_chroma_kd_l1,
            normalized_very_coarse_chroma_kd_mse,
            normalized_row_column_chroma_kd_mse,
        ) = _correction_losses(
            student_output,
            teacher_output,
            noisy,
            kd_weight,
            correction_config,
            noise_strength,
        )
        correction_total = (
            shadow_lambda * shadow_kd_l1
            + medium_coarse_lambda * medium_coarse_kd_l1
            + float(correction_config.get("normalized_shadow_lambda", 0.0))
            * normalized_shadow_kd_l1
            + float(correction_config.get("normalized_shadow_chroma_lambda", 0.0))
            * normalized_shadow_chroma_kd_l1
            + float(
                correction_config.get(
                    "shadow_medium_coarse_chroma_lambda", 0.0
                )
            )
            * shadow_medium_coarse_chroma_kd_l1
            + float(correction_config.get("flat_shadow_chroma_lambda", 0.0))
            * flat_shadow_chroma_kd_l1
            + float(correction_config.get("very_coarse_chroma_lambda", 0.0))
            * very_coarse_chroma_kd_l1
            + float(correction_config.get("row_column_chroma_lambda", 0.0))
            * row_column_chroma_kd_l1
            + float(
                correction_config.get(
                    "normalized_very_coarse_chroma_lambda", 0.0
                )
            )
            * normalized_very_coarse_chroma_kd_l1
            + float(
                correction_config.get(
                    "weighted_very_coarse_chroma_lambda", 0.0
                )
            )
            * weighted_very_coarse_chroma_kd_l1
            + float(
                correction_config.get(
                    "normalized_row_column_chroma_lambda", 0.0
                )
            )
            * normalized_row_column_chroma_kd_l1
            + float(
                correction_config.get(
                    "normalized_very_coarse_chroma_mse_lambda", 0.0
                )
            )
            * normalized_very_coarse_chroma_kd_mse
            + float(
                correction_config.get(
                    "normalized_row_column_chroma_mse_lambda", 0.0
                )
            )
            * normalized_row_column_chroma_kd_mse
        )
        pyramid_chroma_lambda = float(
            correction_config.get("pyramid_chroma_lambda", 0.0)
        )
        if pyramid_chroma_lambda < 0.0:
            raise ValueError("pyramid_chroma_lambda must be non-negative")
        if pyramid_chroma_lambda > 0.0:
            pyramid_chroma_kd_l1 = _pyramid_chroma_loss(
                student_output,
                teacher_output,
                kd_weight,
                correction_config,
                noise_strength,
            )
            correction_total = (
                correction_total
                + pyramid_chroma_lambda * pyramid_chroma_kd_l1
            )
        detail_lambda = float(correction_config.get("paired_detail_lambda", 0.0))
        if detail_lambda < 0.0:
            raise ValueError("paired_detail_lambda must be non-negative")
        if detail_lambda > 0.0:
            paired_detail_l1 = _paired_detail_loss(
                student_output, clean, gt_weight
            )
            correction_total = correction_total + detail_lambda * paired_detail_l1
    total = (
        mse_scale * (1.0 - alpha) * gt_mse
        + mse_scale * kd_mse
        + lambda_l1 * gt_l1
        + correction_total
    )
    return WeightedLossTerms(
        total=total,
        gt_mse=gt_mse,
        kd_mse=kd_mse,
        gt_l1=gt_l1,
        shadow_kd_l1=shadow_kd_l1,
        medium_coarse_kd_l1=medium_coarse_kd_l1,
        normalized_shadow_kd_l1=normalized_shadow_kd_l1,
        normalized_shadow_chroma_kd_l1=normalized_shadow_chroma_kd_l1,
        shadow_medium_coarse_chroma_kd_l1=(
            shadow_medium_coarse_chroma_kd_l1
        ),
        flat_shadow_chroma_kd_l1=flat_shadow_chroma_kd_l1,
        very_coarse_chroma_kd_l1=very_coarse_chroma_kd_l1,
        row_column_chroma_kd_l1=row_column_chroma_kd_l1,
        normalized_very_coarse_chroma_kd_l1=(
            normalized_very_coarse_chroma_kd_l1
        ),
        weighted_very_coarse_chroma_kd_l1=(
            weighted_very_coarse_chroma_kd_l1
        ),
        normalized_row_column_chroma_kd_l1=(
            normalized_row_column_chroma_kd_l1
        ),
        normalized_very_coarse_chroma_kd_mse=(
            normalized_very_coarse_chroma_kd_mse
        ),
        normalized_row_column_chroma_kd_mse=(
            normalized_row_column_chroma_kd_mse
        ),
        pyramid_chroma_kd_l1=pyramid_chroma_kd_l1,
        paired_detail_l1=paired_detail_l1,
        gt_weight_mean=gt_weight.mean(),
        kd_weight_mean=kd_weight.mean(),
    )
