#!/usr/bin/env python3
"""Export a conditioned LiteDenoiseNet checkpoint for the iOS app."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))

from src.student import LiteDenoiseNet, checkpoint_model_kwargs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--precomputed-noise-gate",
        action="store_true",
        help="Export a noise-adapter model with a delegate-friendly fifth gate plane",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kwargs = checkpoint_model_kwargs(checkpoint)
    if args.precomputed_noise_gate:
        if not kwargs.get("noise_adapter_channels"):
            raise ValueError("precomputed noise gate requires a noise-adapter checkpoint")
        kwargs["input_channels"] = 5
        kwargs["precomputed_noise_gate"] = True
    model = LiteDenoiseNet(**kwargs, clamp_output=False).eval()
    model.load_state_dict(checkpoint["model"], strict=True)

    sample = torch.rand(
        1,
        kwargs["input_channels"],
        LiteDenoiseNet.INPUT_SIZE,
        LiteDenoiseNet.INPUT_SIZE,
        generator=torch.Generator().manual_seed(1337),
    )
    traced = torch.jit.trace(model, sample)
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        inputs=[ct.TensorType(name="input", shape=sample.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
    )

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))
    print(
        f"Exported epoch {checkpoint['epoch']} with "
        f"{kwargs['input_channels']} input channels to {args.output}"
    )


if __name__ == "__main__":
    main()
