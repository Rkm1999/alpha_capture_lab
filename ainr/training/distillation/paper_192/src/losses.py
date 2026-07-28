"""Losses for output-level SCUNet response distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class DistillationLossTerms:
    """The total objective and its unweighted component losses."""

    total: torch.Tensor
    gt_mse: torch.Tensor
    kd_mse: torch.Tensor
    gt_l1: torch.Tensor


def compute_distillation_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    clean: torch.Tensor,
    *,
    alpha: float = 0.9,
    mse_scale: float = 1000.0,
    lambda_l1: float = 50.0,
) -> DistillationLossTerms:
    """Compute the guide's exact clean/KD MSE plus clean L1 objective.

    At ``alpha=0.9`` the three coefficients are respectively 100, 900,
    and 50. ``alpha=0.0`` gives the required clean-supervised baseline.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if student_output.shape != teacher_output.shape or student_output.shape != clean.shape:
        raise ValueError(
            "student, teacher, and clean tensors must have identical shapes; "
            f"got {student_output.shape}, {teacher_output.shape}, and {clean.shape}"
        )

    gt_mse = F.mse_loss(student_output, clean)
    kd_mse = F.mse_loss(student_output, teacher_output)
    gt_l1 = F.l1_loss(student_output, clean)
    total = (
        mse_scale * (1.0 - alpha) * gt_mse
        + mse_scale * alpha * kd_mse
        + lambda_l1 * gt_l1
    )
    return DistillationLossTerms(
        total=total,
        gt_mse=gt_mse,
        kd_mse=kd_mse,
        gt_l1=gt_l1,
    )


def distillation_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    clean: torch.Tensor,
    *,
    alpha: float = 0.9,
    mse_scale: float = 1000.0,
    lambda_l1: float = 50.0,
) -> torch.Tensor:
    """Return only the scalar total for the paper's distillation objective."""

    return compute_distillation_loss(
        student_output,
        teacher_output,
        clean,
        alpha=alpha,
        mse_scale=mse_scale,
        lambda_l1=lambda_l1,
    ).total
