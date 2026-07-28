"""Frozen SCUNet teacher loading for exact 192 px cache checks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


def load_scunet_teacher(
    repository: Path,
    checkpoint: Path,
    device: torch.device,
) -> nn.Module:
    repository = repository.resolve()
    checkpoint = checkpoint.resolve()
    if not (repository / "models" / "network_scunet.py").is_file():
        raise FileNotFoundError(f"SCUNet repository is incomplete: {repository}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SCUNet checkpoint is missing: {checkpoint}")

    repository_text = str(repository)
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)
    from models.network_scunet import SCUNet  # type: ignore[import-not-found]

    model = SCUNet(
        in_nc=3,
        config=[4, 4, 4, 4, 4, 4, 4],
        dim=64,
        input_resolution=192,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)
