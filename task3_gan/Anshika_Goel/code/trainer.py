"""Core one-batch CycleGAN optimization engine for DATA 266 Task 3.

This module implements the optimization mechanics used by the eventual
training loop:

- canonical 100-epoch constant + 100-epoch linear-decay learning rate;
- Adam optimizers for both generators and each discriminator;
- generator adversarial/cycle/identity update;
- discriminator A and discriminator B updates;
- replay-buffer integration;
- gradient L2 norms and non-finite gradient counting.

It does not discover/load datasets, execute epoch loops, write logs,
save checkpoints, access Kaggle, or perform any work at import time.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch
from torch import nn
from torch.optim import Adam, Optimizer

from training_utils import (
    CycleGANLoss,
    ImagePool,
    set_requires_grad,
)


__all__ = [
    "learning_rate_for_epoch",
    "set_optimizer_learning_rate",
    "apply_epoch_learning_rate",
    "gradient_l2_norm",
    "count_nonfinite_gradients",
    "build_optimizers",
    "CycleGANTrainer",
]


_REQUIRED_NETWORK_KEYS = {
    "generator_a_to_b",
    "generator_b_to_a",
    "discriminator_a",
    "discriminator_b",
}


_REQUIRED_OPTIMIZER_KEYS = {
    "generator",
    "discriminator_a",
    "discriminator_b",
}


def learning_rate_for_epoch(
    epoch: int,
    *,
    base_lr: float = 2e-4,
    constant_lr_epochs: int = 100,
    linear_decay_epochs: int = 100,
) -> float:
    """Return the 1-based CycleGAN learning rate for one epoch."""

    if base_lr < 0.0:
        raise ValueError(
            "base_lr cannot be negative"
        )

    if constant_lr_epochs <= 0:
        raise ValueError(
            "constant_lr_epochs must be positive"
        )

    if linear_decay_epochs <= 0:
        raise ValueError(
            "linear_decay_epochs must be positive"
        )

    total_epochs = (
        constant_lr_epochs
        + linear_decay_epochs
    )

    if not (
        1
        <= epoch
        <= total_epochs
    ):
        raise ValueError(
            "epoch is outside the configured schedule"
        )

    if epoch <= constant_lr_epochs:
        return float(
            base_lr
        )

    progress = (
        epoch
        - constant_lr_epochs
    ) / linear_decay_epochs

    return float(
        base_lr
        * max(
            0.0,
            1.0 - progress,
        )
    )


def set_optimizer_learning_rate(
    optimizer: Optimizer,
    learning_rate: float,
) -> None:
    """Set every parameter group's learning rate."""

    if learning_rate < 0.0:
        raise ValueError(
            "learning_rate cannot be negative"
        )

    for group in optimizer.param_groups:
        group["lr"] = float(
            learning_rate
        )


def apply_epoch_learning_rate(
    optimizers: Mapping[str, Optimizer],
    *,
    epoch: int,
    base_lr: float = 2e-4,
    constant_lr_epochs: int = 100,
    linear_decay_epochs: int = 100,
) -> float:
    """Apply the canonical epoch LR to all CycleGAN optimizers."""

    if set(
        optimizers
    ) != _REQUIRED_OPTIMIZER_KEYS:
        raise ValueError(
            "optimizers must contain exactly generator, "
            "discriminator_a, and discriminator_b"
        )

    learning_rate = learning_rate_for_epoch(
        epoch,
        base_lr=base_lr,
        constant_lr_epochs=constant_lr_epochs,
        linear_decay_epochs=linear_decay_epochs,
    )

    for optimizer in optimizers.values():
        set_optimizer_learning_rate(
            optimizer,
            learning_rate,
        )

    return learning_rate


def _as_module_list(
    modules: nn.Module | Iterable[nn.Module],
) -> list[nn.Module]:
    if isinstance(
        modules,
        nn.Module,
    ):
        result = [
            modules
        ]
    else:
        result = list(
            modules
        )

    if not result:
        raise ValueError(
            "at least one module is required"
        )

    for module in result:
        if not isinstance(
            module,
            nn.Module,
        ):
            raise TypeError(
                "all items must be torch.nn.Module instances"
            )

    return result


def gradient_l2_norm(
    modules: nn.Module | Iterable[nn.Module],
) -> float:
    """Return the global L2 norm of currently populated gradients."""

    module_list = _as_module_list(
        modules
    )

    squared_sum = 0.0

    for module in module_list:
        for parameter in module.parameters():
            if parameter.grad is None:
                continue

            value = float(
                parameter.grad
                .detach()
                .float()
                .norm(
                    p=2
                )
                .item()
            )

            squared_sum += (
                value
                * value
            )

    return math.sqrt(
        squared_sum
    )


def count_nonfinite_gradients(
    modules: nn.Module | Iterable[nn.Module],
) -> int:
    """Count non-finite scalar gradient elements."""

    module_list = _as_module_list(
        modules
    )

    count = 0

    for module in module_list:
        for parameter in module.parameters():
            if parameter.grad is None:
                continue

            count += int(
                (
                    ~torch.isfinite(
                        parameter.grad
                    )
                )
                .sum()
                .item()
            )

    return count


def build_optimizers(
    networks: Mapping[str, nn.Module],
    *,
    learning_rate: float = 2e-4,
    beta1: float = 0.5,
    beta2: float = 0.999,
    weight_decay: float = 0.0,
) -> dict[str, Optimizer]:
    """Build canonical Adam optimizers with disjoint parameter ownership."""

    if set(
        networks
    ) != _REQUIRED_NETWORK_KEYS:
        raise ValueError(
            "networks must contain exactly the four CycleGAN networks"
        )

    if learning_rate < 0.0:
        raise ValueError(
            "learning_rate cannot be negative"
        )

    if not (
        0.0
        <= beta1
        < 1.0
    ):
        raise ValueError(
            "beta1 must be in [0, 1)"
        )

    if not (
        0.0
        <= beta2
        < 1.0
    ):
        raise ValueError(
            "beta2 must be in [0, 1)"
        )

    if weight_decay < 0.0:
        raise ValueError(
            "weight_decay cannot be negative"
        )

    generator_parameters = list(
        networks[
            "generator_a_to_b"
        ].parameters()
    ) + list(
        networks[
            "generator_b_to_a"
        ].parameters()
    )

    discriminator_a_parameters = list(
        networks[
            "discriminator_a"
        ].parameters()
    )

    discriminator_b_parameters = list(
        networks[
            "discriminator_b"
        ].parameters()
    )

    id_sets = {
        "generator": {
            id(parameter)
            for parameter
            in generator_parameters
        },
        "discriminator_a": {
            id(parameter)
            for parameter
            in discriminator_a_parameters
        },
        "discriminator_b": {
            id(parameter)
            for parameter
            in discriminator_b_parameters
        },
    }

    if (
        id_sets["generator"]
        & id_sets["discriminator_a"]
    ):
        raise RuntimeError(
            "Generator and discriminator A parameter ownership overlaps"
        )

    if (
        id_sets["generator"]
        & id_sets["discriminator_b"]
    ):
        raise RuntimeError(
            "Generator and discriminator B parameter ownership overlaps"
        )

    if (
        id_sets["discriminator_a"]
        & id_sets["discriminator_b"]
    ):
        raise RuntimeError(
            "Discriminator parameter ownership overlaps"
        )

    optimizer_kwargs = {
        "lr": float(
            learning_rate
        ),
        "betas": (
            float(
                beta1
            ),
            float(
                beta2
            ),
        ),
        "weight_decay": float(
            weight_decay
        ),
    }

    return {
        "generator": Adam(
            generator_parameters,
            **optimizer_kwargs,
        ),
        "discriminator_a": Adam(
            discriminator_a_parameters,
            **optimizer_kwargs,
        ),
        "discriminator_b": Adam(
            discriminator_b_parameters,
            **optimizer_kwargs,
        ),
    }


class CycleGANTrainer:
    """Execute one canonical CycleGAN optimization step."""

    def __init__(
        self,
        *,
        networks: Mapping[str, nn.Module],
        optimizers: Mapping[str, Optimizer],
        losses: CycleGANLoss,
        fake_a_pool: ImagePool,
        fake_b_pool: ImagePool,
    ) -> None:

        if set(
            networks
        ) != _REQUIRED_NETWORK_KEYS:
            raise ValueError(
                "networks must contain exactly the four CycleGAN networks"
            )

        if set(
            optimizers
        ) != _REQUIRED_OPTIMIZER_KEYS:
            raise ValueError(
                "optimizers must contain exactly the three CycleGAN optimizers"
            )

        if not isinstance(
            losses,
            CycleGANLoss,
        ):
            raise TypeError(
                "losses must be CycleGANLoss"
            )

        if not isinstance(
            fake_a_pool,
            ImagePool,
        ):
            raise TypeError(
                "fake_a_pool must be ImagePool"
            )

        if not isinstance(
            fake_b_pool,
            ImagePool,
        ):
            raise TypeError(
                "fake_b_pool must be ImagePool"
            )

        self.networks = dict(
            networks
        )

        self.optimizers = dict(
            optimizers
        )

        self.losses = losses
        self.fake_a_pool = fake_a_pool
        self.fake_b_pool = fake_b_pool

    @property
    def generators(
        self,
    ) -> list[nn.Module]:
        return [
            self.networks[
                "generator_a_to_b"
            ],
            self.networks[
                "generator_b_to_a"
            ],
        ]

    @property
    def discriminators(
        self,
    ) -> list[nn.Module]:
        return [
            self.networks[
                "discriminator_a"
            ],
            self.networks[
                "discriminator_b"
            ],
        ]

    def train_mode(
        self,
    ) -> None:
        for network in self.networks.values():
            network.train()

    def _validate_real_batch(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
    ) -> None:

        for name, tensor in (
            (
                "real_a",
                real_a,
            ),
            (
                "real_b",
                real_b,
            ),
        ):
            if not isinstance(
                tensor,
                torch.Tensor,
            ):
                raise TypeError(
                    f"{name} must be a torch.Tensor"
                )

            if tensor.ndim != 4:
                raise ValueError(
                    f"{name} must have NCHW rank 4"
                )

            if tensor.shape[1] != 3:
                raise ValueError(
                    f"{name} must have exactly 3 channels"
                )

            if tensor.shape[0] <= 0:
                raise ValueError(
                    f"{name} batch cannot be empty"
                )

            if not tensor.is_floating_point():
                raise TypeError(
                    f"{name} must use a floating dtype"
                )

            if not torch.isfinite(
                tensor
            ).all():
                raise ValueError(
                    f"{name} contains non-finite values"
                )

        if real_a.shape[0] != real_b.shape[0]:
            raise ValueError(
                "real_a and real_b batch sizes must match"
            )

        generator_device = next(
            self.networks[
                "generator_a_to_b"
            ].parameters()
        ).device

        if real_a.device != generator_device:
            raise ValueError(
                "real_a device does not match model device"
            )

        if real_b.device != generator_device:
            raise ValueError(
                "real_b device does not match model device"
            )

    @staticmethod
    def _require_finite_loss(
        value: torch.Tensor,
        *,
        name: str,
    ) -> None:

        if value.ndim != 0:
            raise ValueError(
                f"{name} must be a scalar tensor"
            )

        if not torch.isfinite(
            value
        ):
            raise FloatingPointError(
                f"{name} is non-finite"
            )

    def train_step(
        self,
        *,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
    ) -> dict[str, float | int]:
        """Run one generator update and one update for each discriminator."""

        self._validate_real_batch(
            real_a,
            real_b,
        )

        self.train_mode()

        generator_a_to_b = self.networks[
            "generator_a_to_b"
        ]

        generator_b_to_a = self.networks[
            "generator_b_to_a"
        ]

        discriminator_a = self.networks[
            "discriminator_a"
        ]

        discriminator_b = self.networks[
            "discriminator_b"
        ]

        optimizer_g = self.optimizers[
            "generator"
        ]

        optimizer_d_a = self.optimizers[
            "discriminator_a"
        ]

        optimizer_d_b = self.optimizers[
            "discriminator_b"
        ]

        # --------------------------------------------------------------
        # Generator update
        # --------------------------------------------------------------

        set_requires_grad(
            self.discriminators,
            requires_grad=False,
        )

        try:
            optimizer_g.zero_grad(
                set_to_none=True
            )

            fake_b = generator_a_to_b(
                real_a
            )

            reconstructed_a = generator_b_to_a(
                fake_b
            )

            fake_a = generator_b_to_a(
                real_b
            )

            reconstructed_b = generator_a_to_b(
                fake_a
            )

            identity_a = generator_b_to_a(
                real_a
            )

            identity_b = generator_a_to_b(
                real_b
            )

            fake_b_prediction = discriminator_b(
                fake_b
            )

            fake_a_prediction = discriminator_a(
                fake_a
            )

            generator_total, generator_components = (
                self.losses.generator_loss(
                    fake_a_prediction=fake_a_prediction,
                    fake_b_prediction=fake_b_prediction,
                    reconstructed_a=reconstructed_a,
                    real_a=real_a,
                    reconstructed_b=reconstructed_b,
                    real_b=real_b,
                    identity_a=identity_a,
                    identity_b=identity_b,
                )
            )

            self._require_finite_loss(
                generator_total,
                name="generator_total",
            )

            generator_total.backward()

            generator_nonfinite = (
                count_nonfinite_gradients(
                    self.generators
                )
            )

            generator_grad_norm = (
                gradient_l2_norm(
                    self.generators
                )
            )

            if generator_nonfinite != 0:
                raise FloatingPointError(
                    "Generator gradients contain non-finite values"
                )

            if not math.isfinite(
                generator_grad_norm
            ):
                raise FloatingPointError(
                    "Generator gradient norm is non-finite"
                )

            optimizer_g.step()

        finally:
            set_requires_grad(
                self.discriminators,
                requires_grad=True,
            )

        # --------------------------------------------------------------
        # Discriminator A update
        # --------------------------------------------------------------

        pooled_fake_a = self.fake_a_pool.query(
            fake_a
        )

        optimizer_d_a.zero_grad(
            set_to_none=True
        )

        real_a_prediction = discriminator_a(
            real_a
        )

        fake_a_pool_prediction = discriminator_a(
            pooled_fake_a
        )

        discriminator_a_loss = (
            self.losses.discriminator_loss(
                real_a_prediction,
                fake_a_pool_prediction,
            )
        )

        self._require_finite_loss(
            discriminator_a_loss,
            name="discriminator_a_loss",
        )

        discriminator_a_loss.backward()

        discriminator_a_nonfinite = (
            count_nonfinite_gradients(
                discriminator_a
            )
        )

        discriminator_a_grad_norm = (
            gradient_l2_norm(
                discriminator_a
            )
        )

        if discriminator_a_nonfinite != 0:
            raise FloatingPointError(
                "Discriminator A gradients contain non-finite values"
            )

        if not math.isfinite(
            discriminator_a_grad_norm
        ):
            raise FloatingPointError(
                "Discriminator A gradient norm is non-finite"
            )

        optimizer_d_a.step()

        # --------------------------------------------------------------
        # Discriminator B update
        # --------------------------------------------------------------

        pooled_fake_b = self.fake_b_pool.query(
            fake_b
        )

        optimizer_d_b.zero_grad(
            set_to_none=True
        )

        real_b_prediction = discriminator_b(
            real_b
        )

        fake_b_pool_prediction = discriminator_b(
            pooled_fake_b
        )

        discriminator_b_loss = (
            self.losses.discriminator_loss(
                real_b_prediction,
                fake_b_pool_prediction,
            )
        )

        self._require_finite_loss(
            discriminator_b_loss,
            name="discriminator_b_loss",
        )

        discriminator_b_loss.backward()

        discriminator_b_nonfinite = (
            count_nonfinite_gradients(
                discriminator_b
            )
        )

        discriminator_b_grad_norm = (
            gradient_l2_norm(
                discriminator_b
            )
        )

        if discriminator_b_nonfinite != 0:
            raise FloatingPointError(
                "Discriminator B gradients contain non-finite values"
            )

        if not math.isfinite(
            discriminator_b_grad_norm
        ):
            raise FloatingPointError(
                "Discriminator B gradient norm is non-finite"
            )

        optimizer_d_b.step()

        # --------------------------------------------------------------
        # Detached metrics only
        # --------------------------------------------------------------

        metrics: dict[
            str,
            float | int,
        ] = {
            "loss_generator_total": float(
                generator_total.detach().item()
            ),
            "loss_gan_a_to_b": float(
                generator_components[
                    "gan_a_to_b"
                ].detach().item()
            ),
            "loss_gan_b_to_a": float(
                generator_components[
                    "gan_b_to_a"
                ].detach().item()
            ),
            "loss_cycle_a": float(
                generator_components[
                    "cycle_a"
                ].detach().item()
            ),
            "loss_cycle_b": float(
                generator_components[
                    "cycle_b"
                ].detach().item()
            ),
            "loss_identity_a": float(
                generator_components[
                    "identity_a"
                ].detach().item()
            ),
            "loss_identity_b": float(
                generator_components[
                    "identity_b"
                ].detach().item()
            ),
            "loss_discriminator_a": float(
                discriminator_a_loss.detach().item()
            ),
            "loss_discriminator_b": float(
                discriminator_b_loss.detach().item()
            ),
            "grad_norm_generator": float(
                generator_grad_norm
            ),
            "grad_norm_discriminator_a": float(
                discriminator_a_grad_norm
            ),
            "grad_norm_discriminator_b": float(
                discriminator_b_grad_norm
            ),
            "nonfinite_gradient_count_generator": int(
                generator_nonfinite
            ),
            "nonfinite_gradient_count_discriminator_a": int(
                discriminator_a_nonfinite
            ),
            "nonfinite_gradient_count_discriminator_b": int(
                discriminator_b_nonfinite
            ),
            "fake_a_pool_size": int(
                len(
                    self.fake_a_pool
                )
            ),
            "fake_b_pool_size": int(
                len(
                    self.fake_b_pool
                )
            ),
            "fake_a_min": float(
                fake_a.detach().min().item()
            ),
            "fake_a_max": float(
                fake_a.detach().max().item()
            ),
            "fake_b_min": float(
                fake_b.detach().min().item()
            ),
            "fake_b_max": float(
                fake_b.detach().max().item()
            ),
        }

        for name, value in metrics.items():
            if isinstance(
                value,
                float,
            ) and not math.isfinite(
                value
            ):
                raise FloatingPointError(
                    f"Non-finite metric: {name}"
                )

        return metrics
