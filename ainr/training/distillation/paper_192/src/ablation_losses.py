"""Per-sample masked loss for the NIND supervision ablation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class MaskedDistillationLossTerms:
    """Scalar objective terms averaged over the complete batch."""

    total: torch.Tensor
    gt_mse: torch.Tensor
    kd_mse: torch.Tensor
    gt_l1: torch.Tensor
    gt_weight_mean: torch.Tensor


def compute_masked_distillation_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    clean: torch.Tensor,
    gt_weight: torch.Tensor,
    *,
    alpha: float = 0.7,
    mse_scale: float = 1000.0,
    lambda_l1: float = 50.0,
) -> MaskedDistillationLossTerms:
    """Apply clean losses per sample while keeping KD active for every sample.

    Ground-truth terms are averaged over the full batch after masking. This
    keeps each record's contribution explicit and avoids amplifying paired
    records in batches that happen to contain more teacher-only NIND samples.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if student_output.shape != teacher_output.shape or student_output.shape != clean.shape:
        raise ValueError(
            "student, teacher, and clean tensors must have identical shapes; "
            f"got {student_output.shape}, {teacher_output.shape}, and {clean.shape}"
        )
    if gt_weight.shape != (student_output.shape[0],):
        raise ValueError(
            f"gt_weight must have shape ({student_output.shape[0]},), got {gt_weight.shape}"
        )
    if not torch.isfinite(gt_weight).all() or bool((gt_weight < 0.0).any()) or bool(
        (gt_weight > 1.0).any()
    ):
        raise ValueError("gt_weight must contain finite values in [0, 1]")

    per_gt_mse = F.mse_loss(student_output, clean, reduction="none").flatten(1).mean(1)
    per_kd_mse = F.mse_loss(student_output, teacher_output, reduction="none").flatten(1).mean(1)
    per_gt_l1 = F.l1_loss(student_output, clean, reduction="none").flatten(1).mean(1)
    weight = gt_weight.to(device=student_output.device, dtype=student_output.dtype)
    gt_mse = (per_gt_mse * weight).mean()
    kd_mse = per_kd_mse.mean()
    gt_l1 = (per_gt_l1 * weight).mean()
    total = (
        mse_scale * (1.0 - alpha) * gt_mse
        + mse_scale * alpha * kd_mse
        + lambda_l1 * gt_l1
    )
    return MaskedDistillationLossTerms(
        total=total,
        gt_mse=gt_mse,
        kd_mse=kd_mse,
        gt_l1=gt_l1,
        gt_weight_mean=weight.mean(),
    )
