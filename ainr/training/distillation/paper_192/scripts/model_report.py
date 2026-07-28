#!/usr/bin/env python3
"""Verify the guide's fixed student shape, parameter count, and convolution MACs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from common import atomic_json, load_config
from src.student import LiteDenoiseNet


def convolution_macs(model: nn.Module, sample: torch.Tensor) -> int:
    total = 0
    hooks = []

    def count(module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal total
        convolution = module
        assert isinstance(convolution, nn.Conv2d)
        batch, output_channels, output_height, output_width = output.shape
        kernel_height, kernel_width = convolution.kernel_size
        operations = (
            batch
            * output_channels
            * output_height
            * output_width
            * (convolution.in_channels // convolution.groups)
            * kernel_height
            * kernel_width
        )
        total += operations

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(count))
    with torch.inference_mode():
        model(sample)
    for hook in hooks:
        hook.remove()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parents[1] / "configs/default.yaml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    shape = tuple(config["model"]["input_shape"])
    model_config = config["model"]
    model = LiteDenoiseNet(
        base_width=int(model_config.get("base_width", 16)),
        input_channels=int(model_config.get("input_channels", 3)),
        noise_adapter_channels=int(model_config.get("noise_adapter_channels", 0)),
        multiscale_adapter_channels=int(
            model_config.get("multiscale_adapter_channels", 0)
        ),
        multiscale_spatial_gate=bool(
            model_config.get("multiscale_spatial_gate", False)
        ),
        multiscale_chroma_floor=float(
            model_config.get("multiscale_chroma_floor", 0.15)
        ),
        chroma_head_channels=int(
            model_config.get("chroma_head_channels", 0)
        ),
        chroma_head_spatial_floor=float(
            model_config.get("chroma_head_spatial_floor", 0.15)
        ),
        chroma_head_noise_floor=float(
            model_config.get("chroma_head_noise_floor", 0.0)
        ),
        chroma_head_use_rgb=bool(
            model_config.get("chroma_head_use_rgb", False)
        ),
        chroma_head_dilations=tuple(
            int(value)
            for value in model_config.get("chroma_head_dilations", (2,))
        ),
        global_chroma_head_channels=int(
            model_config.get("global_chroma_head_channels", 0)
        ),
        global_chroma_head_blocks=int(
            model_config.get("global_chroma_head_blocks", 4)
        ),
        global_chroma_head_use_bottleneck=bool(
            model_config.get("global_chroma_head_use_bottleneck", False)
        ),
        chroma_unet_head_channels=int(
            model_config.get("chroma_unet_head_channels", 0)
        ),
        chroma_profile_head_channels=int(
            model_config.get("chroma_profile_head_channels", 0)
        ),
        chroma_profile_use_restored=bool(
            model_config.get("chroma_profile_use_restored", False)
        ),
        chroma_profile_refinement_blocks=int(
            model_config.get("chroma_profile_refinement_blocks", 0)
        ),
        chroma_refinement_head_channels=int(
            model_config.get("chroma_refinement_head_channels", 0)
        ),
        chroma_refinement_use_restored=bool(
            model_config.get("chroma_refinement_use_restored", False)
        ),
        noise_gate_start=float(model_config.get("noise_gate_start", 0.35)),
        noise_gate_end=float(model_config.get("noise_gate_end", 0.75)),
    ).eval()
    sample = torch.rand(*shape)
    with torch.inference_mode():
        output = model(sample)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    macs = convolution_macs(model, sample)
    report = {
        "input_shape": list(sample.shape),
        "output_shape": list(output.shape),
        "output_min": float(output.min()),
        "output_max": float(output.max()),
        "trainable_parameters": parameters,
        "convolution_macs": macs,
        "convolution_gmacs": macs / 1e9,
        "fp32_weight_mib": parameters * 4 / (1024**2),
        "fp16_weight_mib": parameters * 2 / (1024**2),
    }
    expected_parameters = int(config["model"]["expected_parameters"])
    expected_macs = int(config["model"]["expected_conv_macs"])
    expected_output_shape = (sample.shape[0], 3, sample.shape[2], sample.shape[3])
    if output.shape != expected_output_shape:
        raise AssertionError(
            f"Output shape {output.shape} does not match {expected_output_shape}"
        )
    if parameters != expected_parameters:
        raise AssertionError(f"Parameter count {parameters} != {expected_parameters}")
    if macs != expected_macs:
        raise AssertionError(f"Convolution MACs {macs} != {expected_macs}")
    atomic_json(Path(__file__).parents[1] / "model_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
