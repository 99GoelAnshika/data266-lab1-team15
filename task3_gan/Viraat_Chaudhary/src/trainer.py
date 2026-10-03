"""Core one-batch CycleGAN optimization engine for DATA 266 Task 3.

This module implements the optimization mechanics used by the eventual
training loop:

- configured constant + linear-decay learning rate;
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


def learning_rate_for_epoch(epoch, *, base_lr=0.0002, constant_lr_epochs=20, linear_decay_epochs=40):
    """Keep LR constant, then decay; the final training epoch still updates."""
    if base_lr <= 0 or constant_lr_epochs < 1 or linear_decay_epochs < 1:
        raise ValueError('Invalid learning-rate schedule')
    if not 1 <= epoch <= constant_lr_epochs + linear_decay_epochs:
        raise ValueError('epoch is outside the configured schedule')
    if epoch <= constant_lr_epochs:
        return float(base_lr)
    return float(base_lr * (1 - (epoch - constant_lr_epochs) / (linear_decay_epochs + 1)))


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


def gradient_l2_norm(modules):
    """Global L2 norm with one host synchronization for all parameters."""
    grads = [p.grad.detach() for m in _as_module_list(modules)
             for p in m.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.stack([g.float().norm(2) for g in grads]).norm(2).item())


def count_nonfinite_gradients(modules):
    """Count nonfinite gradient entries with a single host synchronization."""
    grads = [p.grad.detach() for m in _as_module_list(modules)
             for p in m.parameters() if p.grad is not None]
    if not grads:
        return 0
    return int(torch.stack([(~torch.isfinite(g)).sum() for g in grads]).sum().item())


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
        mixed_precision: bool = False,
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
        device = next(self.networks['generator_a_to_b'].parameters()).device
        self.amp_enabled = bool(mixed_precision and device.type == 'cuda'
                                and torch.cuda.is_bf16_supported())

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

    def train_step(self, *, real_a, real_b):
        """CycleGAN updates; BF16 autocast only on GPUs supporting BF16.

        Parameters, Adam state, losses and recorded gradients remain FP32.
        BF16 has FP32's exponent range and does not require a GradScaler.
        CPU and unsupported CUDA devices transparently use FP32.
        """
        self._validate_real_batch(real_a, real_b)
        self.train_mode()
        ga, gb = self.generators
        da, db = self.discriminators
        og = self.optimizers['generator']
        oda, odb = self.optimizers['discriminator_a'], self.optimizers['discriminator_b']

        def amp():
            return torch.autocast(device_type=real_a.device.type, dtype=torch.bfloat16,
                                  enabled=self.amp_enabled)

        def checked_backward(loss, modules, optimizer, name):
            self._require_finite_loss(loss, name=name)
            loss.backward()
            nonfinite = count_nonfinite_gradients(modules)
            norm = gradient_l2_norm(modules)
            if nonfinite or not math.isfinite(norm):
                raise FloatingPointError(name + ' has nonfinite gradients')
            optimizer.step()
            return norm, nonfinite

        set_requires_grad(self.discriminators, requires_grad=False)
        try:
            og.zero_grad(set_to_none=True)
            with amp():
                fake_b, fake_a = ga(real_a), gb(real_b)
                reconstructed_a, reconstructed_b = gb(fake_b), ga(fake_a)
                identity_a, identity_b = gb(real_a), ga(real_b)
                # Explicit FP32 loss reduction also handles BF16 L1 subtraction.
                total, components = self.losses.generator_loss(
                    fake_a_prediction=da(fake_a).float(), fake_b_prediction=db(fake_b).float(),
                    reconstructed_a=reconstructed_a.float(), real_a=real_a.float(),
                    reconstructed_b=reconstructed_b.float(), real_b=real_b.float(),
                    identity_a=identity_a.float(), identity_b=identity_b.float())
            g_norm, g_nonfinite = checked_backward(total, self.generators, og, 'generator_total')
        finally:
            set_requires_grad(self.discriminators, requires_grad=True)

        oda.zero_grad(set_to_none=True)
        with amp():
            loss_da = self.losses.discriminator_loss(da(real_a).float(),
                        da(self.fake_a_pool.query(fake_a)).float())
        da_norm, da_nonfinite = checked_backward(loss_da, da, oda, 'discriminator_a_loss')

        odb.zero_grad(set_to_none=True)
        with amp():
            loss_db = self.losses.discriminator_loss(db(real_b).float(),
                        db(self.fake_b_pool.query(fake_b)).float())
        db_norm, db_nonfinite = checked_backward(loss_db, db, odb, 'discriminator_b_loss')

        metrics = {'loss_generator_total': float(total.detach()),
            'loss_gan_a_to_b': float(components['gan_a_to_b'].detach()),
            'loss_gan_b_to_a': float(components['gan_b_to_a'].detach()),
            'loss_cycle_a': float(components['cycle_a'].detach()),
            'loss_cycle_b': float(components['cycle_b'].detach()),
            'loss_identity_a': float(components['identity_a'].detach()),
            'loss_identity_b': float(components['identity_b'].detach()),
            'loss_discriminator_a': float(loss_da.detach()), 'loss_discriminator_b': float(loss_db.detach()),
            'grad_norm_generator': g_norm, 'grad_norm_discriminator_a': da_norm,
            'grad_norm_discriminator_b': db_norm,
            'nonfinite_gradient_count_generator': g_nonfinite,
            'nonfinite_gradient_count_discriminator_a': da_nonfinite,
            'nonfinite_gradient_count_discriminator_b': db_nonfinite,
            'fake_a_pool_size': len(self.fake_a_pool), 'fake_b_pool_size': len(self.fake_b_pool),
            'fake_a_min': float(fake_a.detach().min()), 'fake_a_max': float(fake_a.detach().max()),
            'fake_b_min': float(fake_b.detach().min()), 'fake_b_max': float(fake_b.detach().max())}
        if any(not math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('A recorded training metric is nonfinite')
        return metrics
