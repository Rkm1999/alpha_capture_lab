#!/usr/bin/env python3
"""Fast contracts for the NIND supervision ablation."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import torch

from common import load_config, resolve_paper_path, seed_everything
from src.ablation_losses import compute_masked_distillation_loss
from src.losses import compute_distillation_loss
from src.student import LiteDenoiseNet
from train_ablation import model_digest, supervision_name


def main() -> None:
    generator = torch.Generator().manual_seed(91)
    student = torch.rand((2, 3, 16, 16), generator=generator, requires_grad=True)
    teacher = torch.rand((2, 3, 16, 16), generator=generator)
    clean = torch.rand((2, 3, 16, 16), generator=generator)
    legacy = compute_distillation_loss(student, teacher, clean, alpha=0.7)
    masked = compute_masked_distillation_loss(
        student, teacher, clean, torch.ones(2), alpha=0.7
    )
    assert torch.allclose(masked.total, legacy.total)
    assert torch.allclose(masked.gt_mse, legacy.gt_mse)
    assert torch.allclose(masked.kd_mse, legacy.kd_mse)
    assert torch.allclose(masked.gt_l1, legacy.gt_l1)

    weights = torch.tensor([0.0, 1.0])
    mixed = compute_masked_distillation_loss(student, teacher, clean, weights, alpha=0.7)
    per_gt_mse = (student - clean).square().flatten(1).mean(1)
    per_kd_mse = (student - teacher).square().flatten(1).mean(1)
    per_gt_l1 = (student - clean).abs().flatten(1).mean(1)
    expected = (
        300.0 * (per_gt_mse * weights).mean()
        + 700.0 * per_kd_mse.mean()
        + 50.0 * (per_gt_l1 * weights).mean()
    )
    assert torch.allclose(mixed.total, expected)
    assert torch.allclose(mixed.gt_weight_mean, torch.tensor(0.5))

    teacher_only = compute_masked_distillation_loss(
        student, teacher, clean, torch.zeros(2), alpha=0.7
    )
    assert torch.allclose(teacher_only.total, 700.0 * per_kd_mse.mean())
    assert teacher_only.gt_mse == 0.0 and teacher_only.gt_l1 == 0.0
    teacher_only.total.backward()
    assert torch.isfinite(student.grad).all()

    teacher_config = load_config(
        Path(__file__).parents[1] / "configs/high_iso_ablation_teacher_only.yaml"
    )
    full_config = load_config(
        Path(__file__).parents[1] / "configs/high_iso_ablation_full_reference.yaml"
    )
    teacher_copy = copy.deepcopy(teacher_config)
    full_copy = copy.deepcopy(full_config)
    teacher_copy.pop("_config_path")
    full_copy.pop("_config_path")
    assert teacher_copy["training"].pop("nind_gt_weight") == 0.0
    assert full_copy["training"].pop("nind_gt_weight") == 1.0
    assert teacher_copy == full_copy
    assert supervision_name(0.0) == "nind_teacher_only"
    assert supervision_name(1.0) == "nind_full_reference"

    seed_everything(int(teacher_config["project"]["seed"]))
    first_hash = model_digest(LiteDenoiseNet())
    seed_everything(int(full_config["project"]["seed"]))
    second_hash = model_digest(LiteDenoiseNet())
    assert first_hash == second_hash

    manifest = resolve_paper_path(teacher_config["data"]["manifest"])
    if manifest.is_file():
        document = json.loads(manifest.read_text(encoding="utf-8"))
        records = document["records"]
        counts = Counter((row["split"], row["dataset"]) for row in records)
        assert len(records) == 5208
        assert counts[("train", "nind")] == 516
        assert counts[("train", "polyu_sony")] == 96
        assert counts[("validation", "nind")] == 164
        assert counts[("validation", "polyu_sony")] == 32
    print("NIND supervision ablation smoke checks passed")


if __name__ == "__main__":
    main()
