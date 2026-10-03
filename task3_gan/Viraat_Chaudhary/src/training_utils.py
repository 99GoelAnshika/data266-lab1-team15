"""CycleGAN optimization primitives for DATA 266 Task 3.

This module contains the mathematical training utilities needed by the
CycleGAN training loop:

- least-squares GAN loss;
- cycle-consistency L1 loss;
- identity L1 loss;
- discriminator loss;
- module gradient enable/disable helper;
- deterministic image replay pool.

It does not load datasets, instantiate optimizers, perform optimizer
steps, train models, save checkpoints, or access Kaggle.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping

import torch
from torch import nn


__all__ = [
    "CycleGANLoss",
    "ImagePool",
    "set_requires_grad",
]


class CycleGANLoss(nn.Module):
    """Canonical CycleGAN objective components.

    The adversarial term uses least-squares GAN (LSGAN) mean-squared
    error. Cycle-consistency and identity terms use L1 distance.
    """

    def __init__(
        self,
        *,
        lambda_cycle_a: float = 10.0,
        lambda_cycle_b: float = 10.0,
        lambda_identity: float = 0.5,
    ) -> None:
        super().__init__()

        if lambda_cycle_a < 0:
            raise ValueError(
                "lambda_cycle_a cannot be negative"
            )

        if lambda_cycle_b < 0:
            raise ValueError(
                "lambda_cycle_b cannot be negative"
            )

        if lambda_identity < 0:
            raise ValueError(
                "lambda_identity cannot be negative"
            )

        self.lambda_cycle_a = float(
            lambda_cycle_a
        )

        self.lambda_cycle_b = float(
            lambda_cycle_b
        )

        self.lambda_identity = float(
            lambda_identity
        )

        self.gan_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()

    @staticmethod
    def _target_like(
        prediction: torch.Tensor,
        *,
        is_real: bool,
    ) -> torch.Tensor:
        value = (
            1.0
            if is_real
            else 0.0
        )

        return torch.full_like(
            prediction,
            fill_value=value,
        )

    def adversarial_loss(
        self,
        prediction: torch.Tensor,
        *,
        is_real: bool,
    ) -> torch.Tensor:
        """Return LSGAN MSE against a real-one or fake-zero target."""

        if prediction.numel() == 0:
            raise ValueError(
                "prediction cannot be empty"
            )

        target = self._target_like(
            prediction,
            is_real=is_real,
        )

        return self.gan_criterion(
            prediction,
            target,
        )

    def cycle_loss(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        *,
        domain: str,
    ) -> torch.Tensor:
        """Return weighted cycle-consistency L1 loss."""

        if reconstructed.shape != original.shape:
            raise ValueError(
                "cycle tensors must have identical shapes"
            )

        normalized_domain = domain.upper()

        if normalized_domain == "A":
            weight = self.lambda_cycle_a
        elif normalized_domain == "B":
            weight = self.lambda_cycle_b
        else:
            raise ValueError(
                "domain must be 'A' or 'B'"
            )

        return (
            self.l1_criterion(
                reconstructed,
                original,
            )
            * weight
        )

    def identity_loss(
        self,
        identity_output: torch.Tensor,
        original: torch.Tensor,
        *,
        domain: str,
    ) -> torch.Tensor:
        """Return weighted CycleGAN identity-preservation L1 loss."""

        if identity_output.shape != original.shape:
            raise ValueError(
                "identity tensors must have identical shapes"
            )

        normalized_domain = domain.upper()

        if normalized_domain == "A":
            cycle_weight = self.lambda_cycle_a
        elif normalized_domain == "B":
            cycle_weight = self.lambda_cycle_b
        else:
            raise ValueError(
                "domain must be 'A' or 'B'"
            )

        return (
            self.l1_criterion(
                identity_output,
                original,
            )
            * cycle_weight
            * self.lambda_identity
        )

    def discriminator_loss(
        self,
        real_prediction: torch.Tensor,
        fake_prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Return the canonical average real/fake discriminator loss."""

        real_loss = self.adversarial_loss(
            real_prediction,
            is_real=True,
        )

        fake_loss = self.adversarial_loss(
            fake_prediction,
            is_real=False,
        )

        return (
            real_loss
            + fake_loss
        ) * 0.5

    def generator_loss(
        self,
        *,
        fake_a_prediction: torch.Tensor,
        fake_b_prediction: torch.Tensor,
        reconstructed_a: torch.Tensor,
        real_a: torch.Tensor,
        reconstructed_b: torch.Tensor,
        real_b: torch.Tensor,
        identity_a: torch.Tensor,
        identity_b: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        Mapping[str, torch.Tensor],
    ]:
        """Return total two-direction generator loss and components."""

        gan_a_to_b = self.adversarial_loss(
            fake_b_prediction,
            is_real=True,
        )

        gan_b_to_a = self.adversarial_loss(
            fake_a_prediction,
            is_real=True,
        )

        cycle_a = self.cycle_loss(
            reconstructed_a,
            real_a,
            domain="A",
        )

        cycle_b = self.cycle_loss(
            reconstructed_b,
            real_b,
            domain="B",
        )

        identity_a_loss = self.identity_loss(
            identity_a,
            real_a,
            domain="A",
        )

        identity_b_loss = self.identity_loss(
            identity_b,
            real_b,
            domain="B",
        )

        total = (
            gan_a_to_b
            + gan_b_to_a
            + cycle_a
            + cycle_b
            + identity_a_loss
            + identity_b_loss
        )

        components = {
            "gan_a_to_b": gan_a_to_b,
            "gan_b_to_a": gan_b_to_a,
            "cycle_a": cycle_a,
            "cycle_b": cycle_b,
            "identity_a": identity_a_loss,
            "identity_b": identity_b_loss,
            "generator_total": total,
        }

        return total, components


def set_requires_grad(
    modules: nn.Module | Iterable[nn.Module],
    *,
    requires_grad: bool,
) -> None:
    """Enable or disable parameter-gradient computation."""

    if isinstance(
        modules,
        nn.Module,
    ):
        iterable = [
            modules
        ]
    else:
        iterable = list(
            modules
        )

    for module in iterable:
        if not isinstance(
            module,
            nn.Module,
        ):
            raise TypeError(
                "all items must be torch.nn.Module instances"
            )

        for parameter in module.parameters():
            parameter.requires_grad = (
                requires_grad
            )


class ImagePool:
    """Replay buffer for previously generated CycleGAN images.

    Once full, each incoming image has a 50% chance of being returned
    directly. Otherwise, a uniformly selected historical image is
    returned and replaced by the incoming image.

    Stored and returned images are detached from autograd.
    """

    def __init__(
        self,
        pool_size: int = 50,
        *,
        seed: int = 0,
    ) -> None:

        if pool_size < 0:
            raise ValueError(
                "pool_size cannot be negative"
            )

        self.pool_size = int(
            pool_size
        )

        self._rng = random.Random(
            seed
        )

        self._images: list[
            torch.Tensor
        ] = []

    def __len__(self) -> int:
        return len(
            self._images
        )

    def query(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Return replayed/current detached images with batch preserved."""

        if images.ndim < 1:
            raise ValueError(
                "images must include a batch dimension"
            )

        if images.shape[0] == 0:
            raise ValueError(
                "images batch cannot be empty"
            )

        detached = images.detach()

        if self.pool_size == 0:
            return detached

        returned: list[
            torch.Tensor
        ] = []

        for image in detached:
            current = image.unsqueeze(
                0
            )

            if len(
                self._images
            ) < self.pool_size:

                self._images.append(
                    current.clone()
                )

                returned.append(
                    current
                )

                continue

            use_history = (
                self._rng.random()
                > 0.5
            )

            if use_history:
                index = self._rng.randrange(
                    self.pool_size
                )

                historical = (
                    self._images[
                        index
                    ].clone()
                )

                self._images[
                    index
                ] = current.clone()

                returned.append(
                    historical
                )

            else:
                returned.append(
                    current
                )

        return torch.cat(
            returned,
            dim=0,
        )
