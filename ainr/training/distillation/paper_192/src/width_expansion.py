"""Function-preserving width expansion mappings and warm-up masks."""

from __future__ import annotations

import torch


def concatenated_map(
    source_first: int,
    target_first: int,
    source_second: int,
) -> list[int]:
    return [
        *range(source_first),
        *range(target_first, target_first + source_second),
    ]


def convolution_maps(
    name: str,
    source_shape: torch.Size,
    source_width: int,
    target_width: int,
) -> tuple[list[int], list[int]]:
    source_features = [source_width * scale for scale in (1, 2, 4, 8, 16)]
    target_features = [target_width * scale for scale in (1, 2, 4, 8, 16)]
    output_map = list(range(source_shape[0]))
    input_map = list(range(source_shape[1]))

    up_levels = {
        "up3.weight": (4, 3),
        "up2.weight": (3, 2),
        "up1.weight": (2, 1),
        "up0.weight": (1, 0),
    }
    if name in up_levels:
        first, second = up_levels[name]
        input_map = concatenated_map(
            source_features[first],
            target_features[first],
            source_features[second],
        )
    elif name == "noise_adapter.0.weight":
        input_map = [
            *range(source_features[0]),
            target_features[0],
        ]
    elif name.startswith("multiscale_adapters.") and name.endswith(".0.weight"):
        scale_name = name.split(".")[1]
        level = {"scale0": 0, "scale1": 1, "scale2": 2}[scale_name]
        input_map = [
            *range(source_features[level]),
            *range(target_features[level], target_features[level] + 3),
        ]
    elif name == "chroma_head.0.weight":
        input_map = [
            *range(source_features[2]),
            *range(target_features[2], target_features[2] + 6),
        ]
    elif name == "global_chroma_head.0.weight":
        input_map = [
            *range(6),
            *range(6, 6 + source_features[4]),
        ]
    return output_map, input_map


def expansion_gradient_masks(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    source_width: int,
    target_width: int,
) -> dict[str, torch.Tensor]:
    """Return masks that train only newly added channels and connections."""

    masks: dict[str, torch.Tensor] = {}
    for name, target_value in target.items():
        source_value = source[name]
        if source_value.shape == target_value.shape:
            masks[name] = torch.zeros_like(target_value)
            continue
        mask = torch.ones_like(target_value)
        if name.endswith(".weight") and source_value.ndim == 4:
            output_map, input_map = convolution_maps(
                name,
                source_value.shape,
                source_width,
                target_width,
            )
            for source_output, target_output in enumerate(output_map):
                del source_output
                mask[target_output, input_map] = 0
        elif name.endswith(".bias") and source_value.ndim == 1:
            mask[: source_value.shape[0]] = 0
        else:
            raise RuntimeError(
                f"Unsupported width-expansion mask for {name}: "
                f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
            )
        masks[name] = mask
    return masks
