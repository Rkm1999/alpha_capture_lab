#!/usr/bin/env python3
"""Expand a LiteDenoiseNet backbone width without changing its initial output."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import torch

from common import PAPER_ROOT
from src.student import LiteDenoiseNet, checkpoint_model_kwargs
from src.width_expansion import convolution_maps

del PAPER_ROOT


def expanded_state(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    source_width: int,
    target_width: int,
) -> dict[str, torch.Tensor]:
    expanded = {name: value.clone() for name, value in target.items()}
    for name, source_value in source.items():
        target_value = expanded[name]
        if source_value.shape == target_value.shape:
            expanded[name] = source_value.clone()
            continue
        if name.endswith(".weight") and source_value.ndim == 4:
            output_map, input_map = convolution_maps(
                name,
                source_value.shape,
                source_width,
                target_width,
            )
            if len(output_map) != source_value.shape[0]:
                raise RuntimeError(f"Invalid output map for {name}")
            if len(input_map) != source_value.shape[1]:
                raise RuntimeError(f"Invalid input map for {name}")
            for source_output, target_output in enumerate(output_map):
                target_value[target_output].zero_()
                target_value[target_output, input_map] = source_value[source_output]
            continue
        if name.endswith(".bias") and source_value.ndim == 1:
            target_value[: source_value.shape[0]].copy_(source_value)
            continue
        raise RuntimeError(
            f"Unsupported width expansion for {name}: "
            f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
        )
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-width", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    source_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_kwargs = checkpoint_model_kwargs(checkpoint)
    source_width = int(source_kwargs["base_width"])
    target_width = int(args.target_width)
    if target_width <= source_width:
        raise ValueError("target width must be greater than source width")

    torch.manual_seed(args.seed)
    source_model = LiteDenoiseNet(**source_kwargs).eval()
    source_model.load_state_dict(checkpoint["model"], strict=True)
    target_kwargs = dict(source_kwargs)
    target_kwargs["base_width"] = target_width
    target_model = LiteDenoiseNet(**target_kwargs).eval()
    target_model.load_state_dict(
        expanded_state(
            checkpoint["model"],
            target_model.state_dict(),
            source_width,
            target_width,
        ),
        strict=True,
    )

    generator = torch.Generator().manual_seed(args.seed + 1)
    sample = torch.rand(
        (2, source_model.input_channels, 192, 192),
        generator=generator,
    )
    with torch.inference_mode():
        source_output = source_model(sample)
        target_output = target_model(sample)
    maximum_error = float((source_output - target_output).abs().max())
    if maximum_error > 2e-6:
        raise RuntimeError(
            f"Width expansion changed model output: max_abs_error={maximum_error}"
        )

    output_checkpoint = copy.deepcopy(checkpoint)
    output_checkpoint["model"] = target_model.state_dict()
    output_checkpoint["config"]["model"]["base_width"] = target_width
    output_checkpoint["width_expansion"] = {
        "source_checkpoint": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_width": source_width,
        "target_width": target_width,
        "seed": args.seed,
        "verification_max_abs_error": maximum_error,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(output_checkpoint, temporary)
    temporary.replace(output)
    print(
        f"wrote {output} width={source_width}->{target_width} "
        f"max_abs_error={maximum_error:.3e}"
    )


if __name__ == "__main__":
    main()
