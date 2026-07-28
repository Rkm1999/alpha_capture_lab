"""Per-image RGB metrics used by the paper-compliant evaluator."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _validate_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError(
            "metrics require NCHW RGB tensors shaped [N, 3, H, W]; "
            f"got {prediction.shape}"
        )


def _crop_border(value: torch.Tensor, border: int) -> torch.Tensor:
    if border < 0:
        raise ValueError(f"border must be non-negative, got {border}")
    if border == 0:
        return value
    if value.shape[-2] <= 2 * border or value.shape[-1] <= 2 * border:
        raise ValueError(
            f"border {border} leaves no pixels for spatial shape {value.shape[-2:]}"
        )
    return value[..., border:-border, border:-border]


def psnr_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    border: int = 1,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Return RGB PSNR for each image after cropping ``border`` pixels."""

    _validate_pair(prediction, target)
    if data_range <= 0.0:
        raise ValueError(f"data_range must be positive, got {data_range}")

    prediction = _crop_border(prediction.float(), border)
    target = _crop_border(target.float(), border)
    mse = (prediction - target).square().mean(dim=(1, 2, 3))
    finite_psnr = 10.0 * torch.log10((data_range * data_range) / mse)
    return torch.where(mse == 0.0, torch.full_like(mse, torch.inf), finite_psnr)


def _gaussian_kernel(
    *,
    size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def gaussian_ssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    border: int = 1,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Return Gaussian-window RGB SSIM for each image.

    The default uses the conventional 11x11 window with sigma 1.5. Window
    filtering is valid (no synthetic boundary padding), and channel/spatial
    SSIM values are averaged into one value per RGB image.
    """

    _validate_pair(prediction, target)
    if data_range <= 0.0:
        raise ValueError(f"data_range must be positive, got {data_range}")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    prediction = _crop_border(prediction.float(), border)
    target = _crop_border(target.float(), border)
    if prediction.shape[-2] < window_size or prediction.shape[-1] < window_size:
        raise ValueError(
            f"SSIM window {window_size} exceeds cropped shape {prediction.shape[-2:]}"
        )

    channels = prediction.shape[1]
    window = _gaussian_kernel(
        size=window_size,
        sigma=sigma,
        channels=channels,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    mu_prediction = F.conv2d(prediction, window, groups=channels)
    mu_target = F.conv2d(target, window, groups=channels)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target

    variance_prediction = (
        F.conv2d(prediction.square(), window, groups=channels) - mu_prediction_sq
    )
    variance_target = F.conv2d(target.square(), window, groups=channels) - mu_target_sq
    covariance = F.conv2d(prediction * target, window, groups=channels) - mu_product

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * mu_product + c1) * (2.0 * covariance + c2)
    denominator = (mu_prediction_sq + mu_target_sq + c1) * (
        variance_prediction + variance_target + c2
    )
    ssim_map = numerator / denominator
    return ssim_map.mean(dim=(1, 2, 3))
