"""The mobile-friendly LiteDenoiseNet student."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def checkpoint_base_width(checkpoint: dict) -> int:
    """Return the configured width, defaulting legacy checkpoints to width 16."""

    config = checkpoint.get("config")
    model = config.get("model") if isinstance(config, dict) else None
    return int(model.get("base_width", 16)) if isinstance(model, dict) else 16


def checkpoint_input_channels(checkpoint: dict) -> int:
    """Return configured channels, inferring legacy checkpoints from weights."""

    config = checkpoint.get("config")
    model = config.get("model") if isinstance(config, dict) else None
    if isinstance(model, dict) and "input_channels" in model:
        return int(model["input_channels"])
    state = checkpoint.get("model")
    weight = state.get("input_conv.weight") if isinstance(state, dict) else None
    return int(weight.shape[1]) if isinstance(weight, torch.Tensor) else 3


def checkpoint_model_kwargs(checkpoint: dict) -> dict:
    """Return architecture arguments recorded in a training checkpoint."""

    config = checkpoint.get("config")
    model = config.get("model") if isinstance(config, dict) else None
    model = model if isinstance(model, dict) else {}
    conditioning = model.get("noise_conditioning")
    conditioned = isinstance(conditioning, dict) and bool(
        conditioning.get("enabled", False)
    )
    return {
        "base_width": checkpoint_base_width(checkpoint),
        "input_channels": int(
            model.get(
                "input_channels",
                4 if conditioned else checkpoint_input_channels(checkpoint),
            )
        ),
        "noise_adapter_channels": int(model.get("noise_adapter_channels", 0)),
        "multiscale_adapter_channels": int(
            model.get("multiscale_adapter_channels", 0)
        ),
        "multiscale_spatial_gate": bool(
            model.get("multiscale_spatial_gate", False)
        ),
        "multiscale_chroma_floor": float(
            model.get("multiscale_chroma_floor", 0.15)
        ),
        "chroma_head_channels": int(model.get("chroma_head_channels", 0)),
        "chroma_head_spatial_floor": float(
            model.get("chroma_head_spatial_floor", 0.15)
        ),
        "chroma_head_noise_floor": float(
            model.get("chroma_head_noise_floor", 0.0)
        ),
        "chroma_head_use_rgb": bool(
            model.get("chroma_head_use_rgb", False)
        ),
        "chroma_head_dilations": tuple(
            int(value)
            for value in model.get("chroma_head_dilations", (2,))
        ),
        "global_chroma_head_channels": int(
            model.get("global_chroma_head_channels", 0)
        ),
        "global_chroma_head_blocks": int(
            model.get("global_chroma_head_blocks", 4)
        ),
        "global_chroma_head_use_bottleneck": bool(
            model.get("global_chroma_head_use_bottleneck", False)
        ),
        "global_chroma_head_bilinear_upsample": bool(
            model.get("global_chroma_head_bilinear_upsample", False)
        ),
        "chroma_unet_head_channels": int(
            model.get("chroma_unet_head_channels", 0)
        ),
        "chroma_profile_head_channels": int(
            model.get("chroma_profile_head_channels", 0)
        ),
        "chroma_profile_use_restored": bool(
            model.get("chroma_profile_use_restored", False)
        ),
        "chroma_profile_refinement_blocks": int(
            model.get("chroma_profile_refinement_blocks", 0)
        ),
        "chroma_refinement_head_channels": int(
            model.get("chroma_refinement_head_channels", 0)
        ),
        "chroma_refinement_use_restored": bool(
            model.get("chroma_refinement_use_restored", False)
        ),
        "noise_gate_start": float(model.get("noise_gate_start", 0.35)),
        "noise_gate_end": float(model.get("noise_gate_end", 0.75)),
        "precomputed_noise_gate": bool(
            model.get("precomputed_noise_gate", False)
        ),
    }


def student_from_checkpoint(checkpoint: dict) -> "LiteDenoiseNet":
    model = LiteDenoiseNet(**checkpoint_model_kwargs(checkpoint))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


class LiteDenoisingBlock(nn.Module):
    """Two convolutions with a residual connection and final ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = channels // 2
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.relu(self.conv1(value)))
        return self.relu(value + residual)


class MobileChromaUNetHead(nn.Module):
    """Full-resolution mobile U-Net that predicts a two-channel chroma residual."""

    def __init__(self, input_channels: int, base_channels: int) -> None:
        super().__init__()
        c0, c1, c2, c3 = (
            base_channels * scale for scale in (1, 2, 4, 8)
        )
        self.input_conv = nn.Conv2d(input_channels, c0, kernel_size=3, padding=1)
        self.encoder0 = LiteDenoisingBlock(c0)
        self.down0 = LiteDenoiseNet._downsample(c0, c1)
        self.encoder1 = LiteDenoisingBlock(c1)
        self.down1 = LiteDenoiseNet._downsample(c1, c2)
        self.encoder2 = LiteDenoisingBlock(c2)
        self.down2 = LiteDenoiseNet._downsample(c2, c3)
        self.encoder3 = LiteDenoisingBlock(c3)
        self.down3 = LiteDenoiseNet._downsample(c3, c3)
        self.bottleneck = LiteDenoisingBlock(c3)
        self.up3 = nn.Conv2d(c3 + c3, c3, kernel_size=3, padding=1)
        self.decoder3 = LiteDenoisingBlock(c3)
        self.up2 = nn.Conv2d(c3 + c2, c2, kernel_size=3, padding=1)
        self.decoder2 = LiteDenoisingBlock(c2)
        self.up1 = nn.Conv2d(c2 + c1, c1, kernel_size=3, padding=1)
        self.decoder1 = LiteDenoisingBlock(c1)
        self.up0 = nn.Conv2d(c1 + c0, c0, kernel_size=3, padding=1)
        self.decoder0 = LiteDenoisingBlock(c0)
        self.output_conv = nn.Conv2d(c0, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    @staticmethod
    def _upsample(value: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, scale_factor=2.0, mode="nearest")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        level0 = self.encoder0(self.input_conv(value))
        level1 = self.encoder1(self.down0(level0))
        level2 = self.encoder2(self.down1(level1))
        level3 = self.encoder3(self.down2(level2))
        bottleneck = self.bottleneck(self.down3(level3))
        decoded3 = self.decoder3(
            self.up3(torch.cat((self._upsample(bottleneck), level3), dim=1))
        )
        decoded2 = self.decoder2(
            self.up2(torch.cat((self._upsample(decoded3), level2), dim=1))
        )
        decoded1 = self.decoder1(
            self.up1(torch.cat((self._upsample(decoded2), level1), dim=1))
        )
        decoded0 = self.decoder0(
            self.up0(torch.cat((self._upsample(decoded1), level0), dim=1))
        )
        return self.output_conv(decoded0)


class MobileChromaProfileResidualBlock(nn.Module):
    """Identity-initialized low-resolution profile refinement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.body(value)


class MobileChromaProfileHead(nn.Module):
    """Predict smooth spatial, row, and column chroma correction profiles."""

    def __init__(
        self, input_channels: int, channels: int, refinement_blocks: int = 0
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.refinement = nn.Sequential(
            *(
                MobileChromaProfileResidualBlock(channels)
                for _ in range(refinement_blocks)
            )
        )
        self.spatial = nn.Conv2d(channels, 2, kernel_size=3, padding=1)
        self.row = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, 2, kernel_size=1),
        )
        self.column = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, 2, kernel_size=1),
        )
        for layer in (self.spatial, self.row[-1], self.column[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = self.refinement(self.input_projection(value))
        row = self.row(features.mean(dim=3, keepdim=True))
        column = self.column(features.mean(dim=2, keepdim=True))
        return self.spatial(features) + row + column


class LiteDenoiseNet(nn.Module):
    """Configurable-width mobile denoiser from the distillation guide.

    The network predicts an RGB residual, adds it to the noisy input, and
    clamps the restored image to ``[0, 1]``. The topology intentionally uses
    only mobile-friendly convolutions, ReLUs, nearest-neighbor resizing,
    concatenation, addition, and clamp.
    """

    DEFAULT_BASE_WIDTH = 16
    BASE_WIDTH = DEFAULT_BASE_WIDTH
    EXPECTED_PARAMETERS = 1_963_411
    INPUT_SIZE = 192

    def __init__(
        self,
        *,
        base_width: int = DEFAULT_BASE_WIDTH,
        input_channels: int = 3,
        noise_adapter_channels: int = 0,
        multiscale_adapter_channels: int = 0,
        multiscale_spatial_gate: bool = False,
        multiscale_chroma_floor: float = 0.15,
        chroma_head_channels: int = 0,
        chroma_head_spatial_floor: float = 0.15,
        chroma_head_noise_floor: float = 0.0,
        chroma_head_use_rgb: bool = False,
        chroma_head_dilations: tuple[int, ...] = (2,),
        global_chroma_head_channels: int = 0,
        global_chroma_head_blocks: int = 4,
        global_chroma_head_use_bottleneck: bool = False,
        global_chroma_head_bilinear_upsample: bool = False,
        chroma_unet_head_channels: int = 0,
        chroma_profile_head_channels: int = 0,
        chroma_profile_use_restored: bool = False,
        chroma_profile_refinement_blocks: int = 0,
        chroma_refinement_head_channels: int = 0,
        chroma_refinement_use_restored: bool = False,
        noise_gate_start: float = 0.35,
        noise_gate_end: float = 0.75,
        precomputed_noise_gate: bool = False,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        if base_width < 4 or base_width % 2:
            raise ValueError("base_width must be an even integer of at least 4")
        self.base_width = int(base_width)
        if not 3 <= input_channels <= 8:
            raise ValueError("input_channels must be between 3 and 8")
        self.input_channels = int(input_channels)
        if noise_adapter_channels < 0:
            raise ValueError("noise_adapter_channels must be non-negative")
        if multiscale_adapter_channels < 0:
            raise ValueError("multiscale_adapter_channels must be non-negative")
        if chroma_head_channels < 0:
            raise ValueError("chroma_head_channels must be non-negative")
        if global_chroma_head_channels < 0:
            raise ValueError("global_chroma_head_channels must be non-negative")
        if chroma_unet_head_channels < 0 or (
            chroma_unet_head_channels and chroma_unet_head_channels % 2
        ):
            raise ValueError(
                "chroma_unet_head_channels must be zero or a positive even integer"
            )
        if chroma_profile_head_channels < 0:
            raise ValueError("chroma_profile_head_channels must be non-negative")
        if chroma_profile_refinement_blocks < 0:
            raise ValueError("chroma_profile_refinement_blocks must be non-negative")
        if chroma_refinement_head_channels < 0 or (
            chroma_refinement_head_channels
            and chroma_refinement_head_channels % 2
        ):
            raise ValueError(
                "chroma_refinement_head_channels must be zero or a positive even integer"
            )
        if global_chroma_head_blocks < 1:
            raise ValueError("global_chroma_head_blocks must be positive")
        if not chroma_head_dilations or any(
            dilation < 1 for dilation in chroma_head_dilations
        ):
            raise ValueError("chroma_head_dilations must contain positive integers")
        if noise_adapter_channels and input_channels not in (4, 5):
            if input_channels < 4:
                raise ValueError("noise adapter requires a conditioned model input")
        if multiscale_adapter_channels and input_channels < 6:
            raise ValueError(
                "multiscale adapter requires global, shadow, and chroma conditioning"
            )
        if not 0.0 <= multiscale_chroma_floor <= 1.0:
            raise ValueError("multiscale_chroma_floor must be in [0,1]")
        if not 0.0 <= chroma_head_spatial_floor <= 1.0:
            raise ValueError("chroma_head_spatial_floor must be in [0,1]")
        if not 0.0 <= chroma_head_noise_floor <= 1.0:
            raise ValueError("chroma_head_noise_floor must be in [0,1]")
        if chroma_head_channels and input_channels < 6:
            raise ValueError(
                "chroma head requires global, shadow, and chroma conditioning"
            )
        if global_chroma_head_channels and input_channels < 6:
            raise ValueError(
                "global chroma head requires global, shadow, and chroma conditioning"
            )
        if chroma_unet_head_channels and input_channels < 6:
            raise ValueError(
                "chroma U-Net head requires global, shadow, and chroma conditioning"
            )
        if chroma_profile_head_channels and input_channels < 6:
            raise ValueError(
                "chroma profile head requires global, shadow, and chroma conditioning"
            )
        if chroma_refinement_head_channels and input_channels < 6:
            raise ValueError(
                "chroma refinement head requires global, shadow, and chroma conditioning"
            )
        if precomputed_noise_gate and input_channels not in (5, 7):
            raise ValueError(
                "a precomputed noise gate requires five or seven input channels"
            )
        if not 0.0 <= noise_gate_start < noise_gate_end <= 1.0:
            raise ValueError("noise gate bounds must satisfy 0 <= start < end <= 1")
        self.noise_adapter_channels = int(noise_adapter_channels)
        self.multiscale_adapter_channels = int(multiscale_adapter_channels)
        self.multiscale_spatial_gate = bool(multiscale_spatial_gate)
        self.multiscale_chroma_floor = float(multiscale_chroma_floor)
        self.chroma_head_channels = int(chroma_head_channels)
        self.chroma_head_spatial_floor = float(chroma_head_spatial_floor)
        self.chroma_head_noise_floor = float(chroma_head_noise_floor)
        self.chroma_head_use_rgb = bool(chroma_head_use_rgb)
        self.chroma_head_dilations = tuple(
            int(dilation) for dilation in chroma_head_dilations
        )
        self.global_chroma_head_channels = int(global_chroma_head_channels)
        self.global_chroma_head_blocks = int(global_chroma_head_blocks)
        self.global_chroma_head_use_bottleneck = bool(
            global_chroma_head_use_bottleneck
        )
        self.global_chroma_head_bilinear_upsample = bool(
            global_chroma_head_bilinear_upsample
        )
        self.chroma_unet_head_channels = int(chroma_unet_head_channels)
        self.chroma_profile_head_channels = int(chroma_profile_head_channels)
        self.chroma_profile_use_restored = bool(chroma_profile_use_restored)
        self.chroma_profile_refinement_blocks = int(
            chroma_profile_refinement_blocks
        )
        self.chroma_refinement_head_channels = int(
            chroma_refinement_head_channels
        )
        self.chroma_refinement_use_restored = bool(
            chroma_refinement_use_restored
        )
        self.noise_gate_start = float(noise_gate_start)
        self.noise_gate_end = float(noise_gate_end)
        self.precomputed_noise_gate = bool(precomputed_noise_gate)
        self.clamp_output = clamp_output
        f0, f1, f2, f3, f4 = (
            self.base_width * scale for scale in (1, 2, 4, 8, 16)
        )

        backbone_input_channels = 3 if self.noise_adapter_channels else self.input_channels
        self.input_conv = nn.Conv2d(
            backbone_input_channels, f0, kernel_size=3, padding=1
        )
        self.encoder0 = LiteDenoisingBlock(f0)

        self.down0 = self._downsample(f0, f1)
        self.encoder1 = LiteDenoisingBlock(f1)
        self.down1 = self._downsample(f1, f2)
        self.encoder2 = LiteDenoisingBlock(f2)
        self.down2 = self._downsample(f2, f3)
        self.encoder3 = LiteDenoisingBlock(f3)
        self.down3 = self._downsample(f3, f4)
        self.bottleneck = LiteDenoisingBlock(f4)

        self.up3 = nn.Conv2d(f4 + f3, f3, kernel_size=3, padding=1)
        self.decoder3 = LiteDenoisingBlock(f3)
        self.up2 = nn.Conv2d(f3 + f2, f2, kernel_size=3, padding=1)
        self.decoder2 = LiteDenoisingBlock(f2)
        self.up1 = nn.Conv2d(f2 + f1, f1, kernel_size=3, padding=1)
        self.decoder1 = LiteDenoisingBlock(f1)
        self.up0 = nn.Conv2d(f1 + f0, f0, kernel_size=3, padding=1)
        self.decoder0 = LiteDenoisingBlock(f0)

        self.output_conv = nn.Conv2d(f0, 3, kernel_size=3, padding=1)
        if self.noise_adapter_channels:
            self.noise_adapter = nn.Sequential(
                nn.Conv2d(
                    f0 + 1,
                    self.noise_adapter_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    self.noise_adapter_channels,
                    3,
                    kernel_size=3,
                    padding=1,
                ),
            )
            nn.init.zeros_(self.noise_adapter[-1].weight)
            nn.init.zeros_(self.noise_adapter[-1].bias)
        else:
            self.noise_adapter = None
        conditioned_input_channels = self.input_channels - (
            1 if self.precomputed_noise_gate else 0
        )
        if self.multiscale_adapter_channels:
            condition_channels = conditioned_input_channels - 3

            def adapter(feature_channels: int) -> nn.Sequential:
                branch = nn.Sequential(
                    nn.Conv2d(
                        feature_channels + condition_channels,
                        self.multiscale_adapter_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(
                        self.multiscale_adapter_channels,
                        3,
                        kernel_size=3,
                        padding=1,
                    ),
                )
                nn.init.zeros_(branch[-1].weight)
                nn.init.zeros_(branch[-1].bias)
                return branch

            self.multiscale_adapters = nn.ModuleDict(
                {
                    "scale2": adapter(f2),
                    "scale1": adapter(f1),
                    "scale0": adapter(f0),
                }
            )
        else:
            self.multiscale_adapters = None
        if self.chroma_head_channels:
            condition_channels = conditioned_input_channels - 3
            head_input_channels = (
                f2 + condition_channels + (3 if self.chroma_head_use_rgb else 0)
            )
            head_layers: list[nn.Module] = [
                nn.Conv2d(
                    head_input_channels,
                    self.chroma_head_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(inplace=False),
            ]
            for dilation in self.chroma_head_dilations:
                head_layers.extend(
                    (
                        nn.Conv2d(
                            self.chroma_head_channels,
                            self.chroma_head_channels,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                        ),
                        nn.ReLU(inplace=False),
                    )
                )
            head_layers.append(
                nn.Conv2d(
                    self.chroma_head_channels,
                    2,
                    kernel_size=3,
                    padding=1,
                )
            )
            self.chroma_head = nn.Sequential(*head_layers)
            nn.init.zeros_(self.chroma_head[-1].weight)
            nn.init.zeros_(self.chroma_head[-1].bias)
        else:
            self.chroma_head = None
        if self.global_chroma_head_channels:
            global_input_channels = conditioned_input_channels + (
                f4 if self.global_chroma_head_use_bottleneck else 0
            )
            global_layers: list[nn.Module] = [
                nn.Conv2d(
                    global_input_channels,
                    self.global_chroma_head_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(inplace=False),
            ]
            for _ in range(self.global_chroma_head_blocks - 1):
                global_layers.extend(
                    (
                        nn.Conv2d(
                            self.global_chroma_head_channels,
                            self.global_chroma_head_channels,
                            kernel_size=3,
                            padding=1,
                        ),
                        nn.ReLU(inplace=False),
                    )
                )
            global_layers.append(
                nn.Conv2d(
                    self.global_chroma_head_channels,
                    2,
                    kernel_size=3,
                    padding=1,
                )
            )
            self.global_chroma_head = nn.Sequential(*global_layers)
            nn.init.zeros_(self.global_chroma_head[-1].weight)
            nn.init.zeros_(self.global_chroma_head[-1].bias)
        else:
            self.global_chroma_head = None
        self.chroma_unet_head = (
            MobileChromaUNetHead(
                # The last input plane is a precomputed control gate, not an
                # image feature consumed by the chroma branch.
                conditioned_input_channels,
                self.chroma_unet_head_channels,
            )
            if self.chroma_unet_head_channels
            else None
        )
        self.chroma_profile_head = (
            MobileChromaProfileHead(
                conditioned_input_channels
                + f4
                + (6 if self.chroma_profile_use_restored else 0),
                self.chroma_profile_head_channels,
                self.chroma_profile_refinement_blocks,
            )
            if self.chroma_profile_head_channels
            else None
        )
        self.chroma_refinement_head = (
            MobileChromaUNetHead(
                conditioned_input_channels
                + (6 if self.chroma_refinement_use_restored else 0),
                self.chroma_refinement_head_channels,
            )
            if self.chroma_refinement_head_channels
            else None
        )

    @staticmethod
    def _downsample(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=False),
        )

    @staticmethod
    def _upsample(value: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, scale_factor=2.0, mode="nearest")

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        if not isinstance(noisy, torch.fx.Proxy) and (
            noisy.ndim != 4 or noisy.shape[1] != self.input_channels
        ):
            raise ValueError(
                f"expected NCHW input with {self.input_channels} channels, "
                f"got {tuple(noisy.shape)}"
            )
        conditioned = noisy[:, :-1] if self.precomputed_noise_gate else noisy
        rgb = conditioned[:, :3]
        backbone_input = rgb if self.noise_adapter is not None else conditioned
        level0 = self.encoder0(self.input_conv(backbone_input))
        level1 = self.encoder1(self.down0(level0))
        level2 = self.encoder2(self.down1(level1))
        level3 = self.encoder3(self.down2(level2))
        bottleneck = self.bottleneck(self.down3(level3))

        decoded3 = self.decoder3(
            self.up3(torch.cat((self._upsample(bottleneck), level3), dim=1))
        )
        decoded2 = self.decoder2(
            self.up2(torch.cat((self._upsample(decoded3), level2), dim=1))
        )
        decoded1 = self.decoder1(
            self.up1(torch.cat((self._upsample(decoded2), level1), dim=1))
        )
        decoded0 = self.decoder0(
            self.up0(torch.cat((self._upsample(decoded1), level0), dim=1))
        )

        restored = rgb + self.output_conv(decoded0)
        if self.noise_adapter is not None or self.multiscale_adapters is not None:
            strength_plane = noisy[:, 3:4]
            if self.precomputed_noise_gate:
                gate = noisy[:, -1:]
            else:
                position = (
                    (strength_plane - self.noise_gate_start)
                    / (self.noise_gate_end - self.noise_gate_start)
                ).clamp(0.0, 1.0)
                gate = position.square() * (3.0 - 2.0 * position)
            if self.noise_adapter is not None:
                adapter_input = torch.cat((decoded0, strength_plane), dim=1)
                restored = restored + gate * self.noise_adapter(adapter_input)
            if self.multiscale_adapters is not None:
                condition = conditioned[:, 3:]
                condition2 = F.interpolate(
                    condition, scale_factor=0.25, mode="nearest"
                )
                condition1 = F.interpolate(
                    condition, scale_factor=0.5, mode="nearest"
                )
                residual2 = F.interpolate(
                    self.multiscale_adapters["scale2"](
                        torch.cat((decoded2, condition2), dim=1)
                    ),
                    scale_factor=4.0,
                    mode="nearest",
                )
                residual1 = F.interpolate(
                    self.multiscale_adapters["scale1"](
                        torch.cat((decoded1, condition1), dim=1)
                    ),
                    scale_factor=2.0,
                    mode="nearest",
                )
                residual0 = self.multiscale_adapters["scale0"](
                    torch.cat((decoded0, condition), dim=1)
                )
                spatial_gate: torch.Tensor | float = 1.0
                if self.multiscale_spatial_gate:
                    shadow = condition[:, 1:2]
                    chroma = condition[:, 2:3]
                    spatial_gate = shadow * (
                        self.multiscale_chroma_floor
                        + (1.0 - self.multiscale_chroma_floor) * chroma
                    )
                restored = restored + gate * spatial_gate * (
                    residual2 + residual1 + residual0
                )
            if self.chroma_head is not None:
                condition = conditioned[:, 3:]
                condition2 = F.interpolate(
                    condition, scale_factor=0.25, mode="nearest"
                )
                head_inputs = [decoded2, condition2]
                if self.chroma_head_use_rgb:
                    rgb2 = F.avg_pool2d(rgb, kernel_size=4, stride=4)
                    if not isinstance(rgb2, torch.fx.Proxy) and (
                        rgb2.shape[-2:] != decoded2.shape[-2:]
                    ):
                        rgb2 = F.interpolate(
                            rgb2, size=decoded2.shape[-2:], mode="nearest"
                        )
                    head_inputs.append(rgb2)
                cbcr = self.chroma_head(
                    torch.cat(head_inputs, dim=1)
                )
                cb, cr = cbcr[:, 0:1], cbcr[:, 1:2]
                chroma_residual = torch.cat(
                    (
                        1.5748 * cr,
                        -0.1873 * cb - 0.4681 * cr,
                        1.8556 * cb,
                    ),
                    dim=1,
                )
                chroma_residual = F.interpolate(
                    chroma_residual,
                    scale_factor=4.0,
                    mode="nearest",
                )
                shadow_chroma = condition[:, 1:2] * condition[:, 2:3]
                spatial_gate = self.chroma_head_spatial_floor + (
                    1.0 - self.chroma_head_spatial_floor
                ) * shadow_chroma
                chroma_noise_gate = self.chroma_head_noise_floor + (
                    1.0 - self.chroma_head_noise_floor
                ) * gate
                restored = (
                    restored
                    + chroma_noise_gate * spatial_gate * chroma_residual
                )
            if self.global_chroma_head is not None:
                global_input = F.avg_pool2d(
                    conditioned, kernel_size=16, stride=16
                )
                if self.global_chroma_head_use_bottleneck:
                    global_input = torch.cat((global_input, bottleneck), dim=1)
                global_cbcr = self.global_chroma_head(global_input)
                global_cb, global_cr = (
                    global_cbcr[:, 0:1],
                    global_cbcr[:, 1:2],
                )
                global_chroma_residual = torch.cat(
                    (
                        1.5748 * global_cr,
                        -0.1873 * global_cb - 0.4681 * global_cr,
                        1.8556 * global_cb,
                    ),
                    dim=1,
                )
                if self.global_chroma_head_bilinear_upsample:
                    global_chroma_residual = F.interpolate(
                        global_chroma_residual,
                        scale_factor=16.0,
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    global_chroma_residual = F.interpolate(
                        global_chroma_residual,
                        scale_factor=16.0,
                        mode="nearest",
                    )
                restored = restored + global_chroma_residual
            if self.chroma_unet_head is not None:
                unet_cbcr = self.chroma_unet_head(conditioned)
                unet_cb, unet_cr = unet_cbcr[:, 0:1], unet_cbcr[:, 1:2]
                restored = restored + torch.cat(
                    (
                        1.5748 * unet_cr,
                        -0.1873 * unet_cb - 0.4681 * unet_cr,
                        1.8556 * unet_cb,
                    ),
                    dim=1,
                )
            if self.chroma_profile_head is not None:
                profile_inputs = [
                    F.avg_pool2d(conditioned, kernel_size=16, stride=16),
                    bottleneck,
                ]
                if self.chroma_profile_use_restored:
                    profile_inputs.extend(
                        (
                            F.avg_pool2d(
                                restored,
                                kernel_size=16,
                                stride=16,
                            ),
                            F.avg_pool2d(
                                restored - rgb,
                                kernel_size=16,
                                stride=16,
                            ),
                        )
                    )
                profile_input = torch.cat(profile_inputs, dim=1)
                profile_cbcr = F.interpolate(
                    self.chroma_profile_head(profile_input),
                    scale_factor=16.0,
                    mode="bilinear",
                    align_corners=False,
                )
                profile_cb, profile_cr = (
                    profile_cbcr[:, 0:1],
                    profile_cbcr[:, 1:2],
                )
                restored = restored + torch.cat(
                    (
                        1.5748 * profile_cr,
                        -0.1873 * profile_cb - 0.4681 * profile_cr,
                        1.8556 * profile_cb,
                    ),
                    dim=1,
                )
            if self.chroma_refinement_head is not None:
                refinement_input = conditioned
                if self.chroma_refinement_use_restored:
                    refinement_input = torch.cat(
                        (conditioned, restored, restored - rgb),
                        dim=1,
                    )
                refinement_cbcr = self.chroma_refinement_head(
                    refinement_input
                )
                refinement_cb, refinement_cr = (
                    refinement_cbcr[:, 0:1],
                    refinement_cbcr[:, 1:2],
                )
                restored = restored + torch.cat(
                    (
                        1.5748 * refinement_cr,
                        -0.1873 * refinement_cb - 0.4681 * refinement_cr,
                        1.8556 * refinement_cb,
                    ),
                    dim=1,
                )
        return restored.clamp(0.0, 1.0) if self.clamp_output else restored
