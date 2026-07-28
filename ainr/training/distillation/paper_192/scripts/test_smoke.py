#!/usr/bin/env python3
"""Fast graph, loss, metric, and cached-sample smoke checks."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from common import load_config, resolve_paper_path, sha256_file
from evaluate import (
    checkpoint_identity,
    evaluation_identifier,
    ordered_checkpoint_specs,
    parse_checkpoint,
    validate_checkpoint_run,
)
from src.dataset import DistillationDataset
from src.losses import compute_distillation_loss
from src.metrics import gaussian_ssim_per_image, psnr_per_image
from src.student import LiteDenoiseNet
from visual_review import load_bound_rows


def expect_contract_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except (argparse.ArgumentTypeError, ValueError):
        return
    raise AssertionError(f"{function.__name__} accepted an invalid evaluation contract")


def main() -> None:
    config = load_config(Path(__file__).parents[1] / "configs/default.yaml")
    model = LiteDenoiseNet()
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_963_411
    noisy = torch.rand(2, 3, 192, 192)
    clean = torch.rand_like(noisy)
    teacher = torch.rand_like(noisy)
    output = model(noisy)
    assert output.shape == noisy.shape
    detached = output.detach()
    assert 0.0 <= float(detached.min()) <= float(detached.max()) <= 1.0
    terms = compute_distillation_loss(output, teacher, clean, alpha=0.9)
    terms.total.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert torch.isfinite(terms.total)
    assert torch.isinf(psnr_per_image(clean, clean)).all()
    assert torch.allclose(gaussian_ssim_per_image(clean, clean), torch.ones(2), atol=1e-5)

    manifest = resolve_paper_path(config["data"]["manifest"])
    if manifest.is_file():
        dataset = DistillationDataset(
            manifest,
            root=resolve_paper_path(config["data"]["cache_root"]),
            split="train",
            augment=True,
            augmentation_seed=int(config["project"]["seed"]),
        )
        sample = dataset[0]
        assert sample["noisy"].shape == (3, 192, 192)
        assert sample["clean"].shape == sample["teacher"].shape == sample["noisy"].shape

    baseline = Path(__file__).parents[1] / "runs/alpha_0p0/best.pt"
    if baseline.is_file() and manifest.is_file():
        spec = parse_checkpoint(f"alpha_0p0={baseline}")
        checkpoint = torch.load(baseline, map_location="cpu", weights_only=False)
        identity = checkpoint_identity("alpha_0p0", baseline.resolve(), checkpoint)
        assert identity["alpha"] == 0.0
        assert identity["run_fingerprint"] == checkpoint["run_fingerprint"]
        validate_checkpoint_run(baseline.resolve(), checkpoint, config, manifest)
        expect_contract_error(parse_checkpoint, f"teacher={baseline}")
        expect_contract_error(parse_checkpoint, f"alpha_0p5={baseline}")
        expect_contract_error(ordered_checkpoint_specs, [spec, spec])
        expect_contract_error(
            ordered_checkpoint_specs, [spec], require_paper_matrix=True
        )
        expect_contract_error(
            checkpoint_identity, "alpha_0p7", baseline.resolve(), checkpoint
        )
        reversed_matrix = [
            ("alpha_0p9", baseline.resolve()),
            ("alpha_0p7", baseline.resolve()),
            spec,
        ]
        assert [
            label
            for label, _ in ordered_checkpoint_specs(
                reversed_matrix, require_paper_matrix=True
            )
        ] == ["alpha_0p0", "alpha_0p7", "alpha_0p9"]
        diagnostic = Path(__file__).parents[1] / "smoke_run/best.pt"
        if diagnostic.is_file():
            diagnostic_checkpoint = torch.load(
                diagnostic, map_location="cpu", weights_only=False
            )
            expect_contract_error(
                validate_checkpoint_run,
                diagnostic.resolve(),
                diagnostic_checkpoint,
                config,
                manifest,
            )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        per_image = root / "per_image.jsonl"
        provenance = {"schema_version": 1, "checkpoints": {}}
        evaluation_id = evaluation_identifier(provenance)
        content = json.dumps({"evaluation_id": evaluation_id, "index": 0}) + "\n"
        per_image.write_text(content, encoding="utf-8")
        summary = {
            "evaluation_id": evaluation_id,
            "provenance": provenance,
            "checkpoints": {},
            "per_image": {
                "path": str(per_image),
                "sha256": sha256_file(per_image),
                "rows": 1,
                "evaluation_id": evaluation_id,
            },
        }
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        _, bound_rows = load_bound_rows(summary_path, per_image)
        assert bound_rows[0]["evaluation_id"] == evaluation_id
        per_image.write_text(content + " ", encoding="utf-8")
        expect_contract_error(load_bound_rows, summary_path, per_image)
    print("paper_192 smoke checks passed")


if __name__ == "__main__":
    main()
