"""Input-only noise estimation for the conditioned mobile denoiser."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def estimate_noise_strength(
    noisy: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    """Estimate one normalized noise level per RGB tile.

    The estimator uses dark, low-gradient luma samples so image detail has less
    influence than it would in a global high-pass statistic. It intentionally
    uses only the noisy input, allowing the same scalar to be computed in the
    mobile application without EXIF or teacher access.
    """

    if noisy.ndim != 4 or noisy.shape[1] != 3:
        raise ValueError("noise estimation expects NCHW RGB input")
    sigma_min = float(config.get("sigma_min", 0.0015))
    sigma_max = float(config.get("sigma_max", 0.035))
    shadow_limit = float(config.get("shadow_luminance_limit", 0.5))
    flat_fraction = float(config.get("flat_fraction", 0.4))
    if not 0.0 < sigma_min < sigma_max:
        raise ValueError("noise sigma bounds must satisfy 0 < min < max")
    if not 0.0 < shadow_limit <= 1.0:
        raise ValueError("shadow_luminance_limit must be in (0,1]")
    if not 0.0 < flat_fraction < 1.0:
        raise ValueError("flat_fraction must be in (0,1)")

    source = noisy.float()
    weights = source.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    luma = (source * weights).sum(1, keepdim=True)
    local_mean = F.avg_pool2d(luma, kernel_size=3, stride=1, padding=1)
    residual = (luma - local_mean).abs()
    horizontal = F.pad(
        (luma[..., 1:] - luma[..., :-1]).abs(), (0, 1, 0, 0), mode="replicate"
    )
    vertical = F.pad(
        (luma[..., 1:, :] - luma[..., :-1, :]).abs(), (0, 0, 0, 1), mode="replicate"
    )
    gradient = 0.5 * (horizontal + vertical)

    strengths: list[torch.Tensor] = []
    for sample_luma, sample_gradient, sample_residual in zip(
        luma, gradient, residual, strict=True
    ):
        shadow = sample_luma.flatten() < shadow_limit
        shadow_gradients = sample_gradient.flatten()[shadow]
        if shadow_gradients.numel() < 64:
            shadow = torch.ones_like(sample_luma.flatten(), dtype=torch.bool)
            shadow_gradients = sample_gradient.flatten()
        gradient_limit = torch.quantile(shadow_gradients, flat_fraction)
        selected = shadow & (sample_gradient.flatten() <= gradient_limit)
        candidates = sample_residual.flatten()[selected]
        if candidates.numel() < 64:
            candidates = sample_residual.flatten()
        # Median absolute high-pass response divided by the Gaussian MAD
        # constant provides a stable, inexpensive sigma proxy.
        sigma = torch.quantile(candidates, 0.5) / 0.67448975
        log_position = (
            torch.log(sigma.clamp_min(sigma_min)) - torch.log(source.new_tensor(sigma_min))
        ) / (torch.log(source.new_tensor(sigma_max)) - torch.log(source.new_tensor(sigma_min)))
        strengths.append(log_position.clamp(0.0, 1.0))
    return torch.stack(strengths)


def conditioned_input(
    noisy: torch.Tensor,
    strength: torch.Tensor,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    if strength.shape != (noisy.shape[0],):
        raise ValueError(f"noise strength must have shape {(noisy.shape[0],)}")
    plane = strength.to(noisy).view(-1, 1, 1, 1).expand(
        -1, 1, noisy.shape[-2], noisy.shape[-1]
    )
    if not config or not bool(config.get("spatial_maps", {}).get("enabled", False)):
        conditioned = torch.cat((noisy, plane), dim=1)
        return _append_precomputed_gate(conditioned, strength, config)
    spatial_config = config["spatial_maps"]
    shadow_limit = float(spatial_config.get("shadow_luminance_limit", 0.5))
    chroma_kernel = int(spatial_config.get("chroma_kernel_size", 9))
    chroma_scale = float(spatial_config.get("chroma_scale", 0.04))
    if not 0.0 < shadow_limit <= 1.0:
        raise ValueError("spatial shadow luminance limit must be in (0,1]")
    if chroma_kernel < 3 or chroma_kernel % 2 == 0:
        raise ValueError("spatial chroma kernel size must be odd and at least 3")
    if chroma_scale <= 0.0:
        raise ValueError("spatial chroma scale must be positive")

    source = noisy.float()
    luma_weights = source.new_tensor((0.2126, 0.7152, 0.0722)).view(
        1, 3, 1, 1
    )
    luma = (source * luma_weights).sum(1, keepdim=True)
    shadow = ((shadow_limit - luma) / shadow_limit).clamp(0.0, 1.0)

    opponent = torch.stack(
        (source[:, 0] - source[:, 1], source[:, 2] - source[:, 1]),
        dim=1,
    )
    padding = chroma_kernel // 2
    local_mean = F.avg_pool2d(
        F.pad(opponent, (padding,) * 4, mode="replicate"),
        kernel_size=chroma_kernel,
        stride=1,
    )
    chroma = (opponent - local_mean).abs().mean(1, keepdim=True)
    chroma = (chroma / chroma_scale).clamp(0.0, 1.0)
    conditioned = torch.cat(
        (noisy, plane, shadow.to(noisy), chroma.to(noisy)),
        dim=1,
    )
    return _append_precomputed_gate(conditioned, strength, config)


def _append_precomputed_gate(
    conditioned: torch.Tensor,
    strength: torch.Tensor,
    config: dict[str, Any] | None,
) -> torch.Tensor:
    """Append the deployment gate so the quantized graph needs no float clamp."""

    if not config or not bool(config.get("precomputed_gate", False)):
        return conditioned
    start = float(config.get("precomputed_gate_start", 0.35))
    end = float(config.get("precomputed_gate_end", 0.75))
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("precomputed gate bounds must satisfy 0 <= start < end <= 1")
    position = ((strength.float() - start) / (end - start)).clamp(0.0, 1.0)
    gate = position.square() * (3.0 - 2.0 * position)
    gate_plane = gate.to(conditioned).view(-1, 1, 1, 1).expand(
        -1,
        1,
        conditioned.shape[-2],
        conditioned.shape[-1],
    )
    return torch.cat((conditioned, gate_plane), dim=1)


def training_noise_strength(
    noisy: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the true estimate and its regularized training-time version."""

    strength = estimate_noise_strength(noisy, config)
    conditioned = strength
    jitter = float(config.get("jitter", 0.0))
    dropout = float(config.get("dropout", 0.0))
    if jitter < 0.0 or not 0.0 <= dropout < 1.0:
        raise ValueError("noise conditioning requires jitter >= 0 and dropout in [0,1)")
    if jitter:
        conditioned = conditioned + torch.randn_like(conditioned) * jitter
    if dropout:
        keep = torch.rand_like(conditioned) >= dropout
        conditioned = torch.where(keep, conditioned, torch.zeros_like(conditioned))
    return strength, conditioned.clamp(0.0, 1.0)


def severity_gate(strength: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    start = float(config.get("gate_start", 0.35))
    end = float(config.get("gate_end", 0.75))
    floor = float(config.get("gate_floor", 0.1))
    if not 0.0 <= start < end <= 1.0 or not 0.0 <= floor <= 1.0:
        raise ValueError("invalid noise severity gate configuration")
    position = ((strength.float() - start) / (end - start)).clamp(0.0, 1.0)
    smooth = position.square() * (3.0 - 2.0 * position)
    return floor + (1.0 - floor) * smooth


def model_input_from_config(
    noisy: torch.Tensor,
    model_config: dict[str, Any],
) -> torch.Tensor:
    conditioning = model_config.get("noise_conditioning")
    if not isinstance(conditioning, dict) or not bool(conditioning.get("enabled", False)):
        return noisy
    return conditioned_input(
        noisy,
        estimate_noise_strength(noisy, conditioning),
        conditioning,
    )
