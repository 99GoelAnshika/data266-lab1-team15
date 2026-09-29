"""Neural-network components for the DATA 266 Task 3 CycleGAN.

This module contains only from-scratch CycleGAN network definitions.
It does not load pretrained weights, access datasets, train models,
perform evaluation, or create Kaggle outputs.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


__all__ = [
    "ResidualBlock",
    "ResNetGenerator",
    "PatchGANDiscriminator",
    "initialize_weights",
    "build_cyclegan_models",
    "count_trainable_parameters",
]


class ResidualBlock(nn.Module):
    """CycleGAN residual block with reflection padding and InstanceNorm."""

    def __init__(self, channels: int) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.InstanceNorm2d(
                channels,
                affine=False,
                track_running_stats=False,
            ),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.InstanceNorm2d(
                channels,
                affine=False,
                track_running_stats=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """256x256 CycleGAN generator using nine residual blocks by default."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        base_channels: int = 64,
        residual_blocks: int = 9,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError(
                "input_channels must be positive"
            )

        if output_channels <= 0:
            raise ValueError(
                "output_channels must be positive"
            )

        if base_channels <= 0:
            raise ValueError(
                "base_channels must be positive"
            )

        if residual_blocks < 0:
            raise ValueError(
                "residual_blocks cannot be negative"
            )

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                input_channels,
                base_channels,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.InstanceNorm2d(
                base_channels,
                affine=False,
                track_running_stats=False,
            ),
            nn.ReLU(inplace=True),
        ]

        channels = base_channels

        # Two stride-2 downsampling stages:
        # 256 -> 128 -> 64 spatial resolution.
        for _ in range(2):
            next_channels = channels * 2

            layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        next_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                    nn.InstanceNorm2d(
                        next_channels,
                        affine=False,
                        track_running_stats=False,
                    ),
                    nn.ReLU(inplace=True),
                ]
            )

            channels = next_channels

        for _ in range(residual_blocks):
            layers.append(
                ResidualBlock(
                    channels
                )
            )

        # Two learned upsampling stages:
        # 64 -> 128 -> 256 spatial resolution.
        for _ in range(2):
            next_channels = channels // 2

            layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels,
                        next_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=True,
                    ),
                    nn.InstanceNorm2d(
                        next_channels,
                        affine=False,
                        track_running_stats=False,
                    ),
                    nn.ReLU(inplace=True),
                ]
            )

            channels = next_channels

        layers.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(
                    channels,
                    output_channels,
                    kernel_size=7,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.Tanh(),
            ]
        )

        self.model = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator used by the original CycleGAN design."""

    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError(
                "input_channels must be positive"
            )

        if base_channels <= 0:
            raise ValueError(
                "base_channels must be positive"
            )

        if n_layers < 1:
            raise ValueError(
                "n_layers must be at least 1"
            )

        kernel_size = 4
        padding = 1

        layers: list[nn.Module] = [
            nn.Conv2d(
                input_channels,
                base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        ]

        previous_multiplier = 1

        for layer_index in range(
            1,
            n_layers,
        ):
            multiplier = min(
                2**layer_index,
                8,
            )

            layers.extend(
                [
                    nn.Conv2d(
                        base_channels
                        * previous_multiplier,
                        base_channels
                        * multiplier,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        bias=True,
                    ),
                    nn.InstanceNorm2d(
                        base_channels
                        * multiplier,
                        affine=False,
                        track_running_stats=False,
                    ),
                    nn.LeakyReLU(
                        negative_slope=0.2,
                        inplace=True,
                    ),
                ]
            )

            previous_multiplier = (
                multiplier
            )

        multiplier = min(
            2**n_layers,
            8,
        )

        layers.extend(
            [
                nn.Conv2d(
                    base_channels
                    * previous_multiplier,
                    base_channels
                    * multiplier,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=True,
                ),
                nn.InstanceNorm2d(
                    base_channels
                    * multiplier,
                    affine=False,
                    track_running_stats=False,
                ),
                nn.LeakyReLU(
                    negative_slope=0.2,
                    inplace=True,
                ),
                nn.Conv2d(
                    base_channels
                    * multiplier,
                    1,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=True,
                ),
            ]
        )

        self.model = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(x)


def initialize_weights(
    module: nn.Module,
    *,
    mean: float = 0.0,
    std: float = 0.02,
) -> None:
    """Initialize convolution layers with CycleGAN-style normal weights."""

    if std <= 0:
        raise ValueError(
            "std must be positive"
        )

    for layer in module.modules():

        if isinstance(
            layer,
            (
                nn.Conv2d,
                nn.ConvTranspose2d,
            ),
        ):
            nn.init.normal_(
                layer.weight,
                mean=mean,
                std=std,
            )

            if layer.bias is not None:
                nn.init.constant_(
                    layer.bias,
                    0.0,
                )

        elif isinstance(
            layer,
            nn.InstanceNorm2d,
        ):
            if (
                layer.affine
                and layer.weight
                is not None
            ):
                nn.init.normal_(
                    layer.weight,
                    mean=1.0,
                    std=std,
                )

            if (
                layer.affine
                and layer.bias
                is not None
            ):
                nn.init.constant_(
                    layer.bias,
                    0.0,
                )


def count_trainable_parameters(
    module: nn.Module,
) -> int:
    """Return the number of parameters requiring gradients."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def build_cyclegan_models(
    *,
    input_channels: int = 3,
    output_channels: int = 3,
    generator_channels: int = 64,
    discriminator_channels: int = 64,
    residual_blocks: int = 9,
    discriminator_layers: int = 3,
) -> Mapping[str, nn.Module]:
    """Build and randomly initialize the four CycleGAN networks."""

    generator_a_to_b = ResNetGenerator(
        input_channels=input_channels,
        output_channels=output_channels,
        base_channels=generator_channels,
        residual_blocks=residual_blocks,
    )

    generator_b_to_a = ResNetGenerator(
        input_channels=output_channels,
        output_channels=input_channels,
        base_channels=generator_channels,
        residual_blocks=residual_blocks,
    )

    discriminator_a = PatchGANDiscriminator(
        input_channels=input_channels,
        base_channels=discriminator_channels,
        n_layers=discriminator_layers,
    )

    discriminator_b = PatchGANDiscriminator(
        input_channels=output_channels,
        base_channels=discriminator_channels,
        n_layers=discriminator_layers,
    )

    networks = {
        "generator_a_to_b":
            generator_a_to_b,
        "generator_b_to_a":
            generator_b_to_a,
        "discriminator_a":
            discriminator_a,
        "discriminator_b":
            discriminator_b,
    }

    for network in networks.values():
        initialize_weights(
            network
        )

    return networks
