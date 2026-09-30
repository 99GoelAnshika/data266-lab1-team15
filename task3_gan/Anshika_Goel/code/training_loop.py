"""Deterministic epoch-level orchestration for Task 3 CycleGAN.

This module intentionally does not discover datasets, parse command-line
arguments, write logs, or create permanent checkpoints.

The eventual training entry point supplies an epoch-specific loader
factory. At the beginning of every epoch, this module derives a stable
epoch seed and resets Python/Torch/CUDA process RNG state. A real-data
loader factory must also pass this epoch seed to the Task 3 DataLoader.

That design gives the project an explicit epoch-boundary resume contract:
after a checkpoint that records ``epoch_completed = N``, reconstruction
starts at epoch ``N + 1`` using the deterministic seed for that epoch.
It does not claim arbitrary mid-epoch bit-exact continuation.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch

from trainer import (
    CycleGANTrainer,
    apply_epoch_learning_rate,
)


__all__ = [
    "derive_epoch_seed",
    "seed_process_for_epoch",
    "next_epoch_after_checkpoint",
    "run_training_epoch",
    "run_epoch_range",
]


_NONFINITE_KEYS = {
    "nonfinite_gradient_count_generator",
    "nonfinite_gradient_count_discriminator_a",
    "nonfinite_gradient_count_discriminator_b",
}

_POOL_SIZE_KEYS = {
    "fake_a_pool_size",
    "fake_b_pool_size",
}

_MIN_KEYS = {
    "fake_a_min",
    "fake_b_min",
}

_MAX_KEYS = {
    "fake_a_max",
    "fake_b_max",
}


def derive_epoch_seed(
    base_seed: int,
    epoch: int,
) -> int:
    """Derive a stable positive 63-bit seed from base seed and epoch."""

    if not isinstance(
        base_seed,
        int,
    ):
        raise TypeError(
            "base_seed must be int"
        )

    if base_seed < 0:
        raise ValueError(
            "base_seed cannot be negative"
        )

    if not isinstance(
        epoch,
        int,
    ):
        raise TypeError(
            "epoch must be int"
        )

    if epoch <= 0:
        raise ValueError(
            "epoch must be positive"
        )

    material = (
        f"DATA266_TASK3_CYCLEGAN|{base_seed}|{epoch}"
        .encode(
            "utf-8"
        )
    )

    digest = hashlib.sha256(
        material
    ).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return value % (
        2**63
        - 1
    )


def seed_process_for_epoch(
    epoch_seed: int,
) -> None:
    """Seed Python, Torch CPU and all currently visible CUDA devices."""

    if not isinstance(
        epoch_seed,
        int,
    ):
        raise TypeError(
            "epoch_seed must be int"
        )

    if not (
        0
        <= epoch_seed
        < 2**63
    ):
        raise ValueError(
            "epoch_seed must be in [0, 2**63)"
        )

    random.seed(
        epoch_seed
    )

    torch.manual_seed(
        epoch_seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            epoch_seed
        )


def next_epoch_after_checkpoint(
    epoch_completed: int,
    *,
    total_epochs: int,
) -> int | None:
    """Return the next whole epoch, or None when training is complete."""

    if not isinstance(
        epoch_completed,
        int,
    ):
        raise TypeError(
            "epoch_completed must be int"
        )

    if not isinstance(
        total_epochs,
        int,
    ):
        raise TypeError(
            "total_epochs must be int"
        )

    if total_epochs <= 0:
        raise ValueError(
            "total_epochs must be positive"
        )

    if not (
        0
        <= epoch_completed
        <= total_epochs
    ):
        raise ValueError(
            "epoch_completed is outside the configured range"
        )

    if epoch_completed == total_epochs:
        return None

    return epoch_completed + 1


def _extract_domain_pair(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract domain A/B tensors from a supported batch structure."""

    if isinstance(
        batch,
        Mapping,
    ):

        if set(
            batch
        ) != {
            "real_a",
            "real_b",
        }:
            raise ValueError(
                "Mapping batches must contain exactly real_a and real_b"
            )

        real_a = batch[
            "real_a"
        ]

        real_b = batch[
            "real_b"
        ]

    elif isinstance(
        batch,
        (tuple, list),
    ):

        if len(
            batch
        ) != 2:
            raise ValueError(
                "Tuple/list batches must contain exactly two tensors"
            )

        real_a = batch[
            0
        ]

        real_b = batch[
            1
        ]

    else:
        raise TypeError(
            "batch must be a mapping or two-item tuple/list"
        )

    if not isinstance(
        real_a,
        torch.Tensor,
    ):
        raise TypeError(
            "Domain A batch must be a torch.Tensor"
        )

    if not isinstance(
        real_b,
        torch.Tensor,
    ):
        raise TypeError(
            "Domain B batch must be a torch.Tensor"
        )

    return (
        real_a,
        real_b,
    )


def _cuda_peak_mib(
    device: torch.device,
) -> float:
    if device.type != "cuda":
        return 0.0

    return float(
        torch.cuda.max_memory_allocated(
            device
        )
        / 1024**2
    )


def run_training_epoch(
    *,
    trainer: CycleGANTrainer,
    batches: Iterable[Any],
    device: str | torch.device,
    epoch: int,
    epoch_seed: int,
    global_step: int,
    base_lr: float,
    constant_lr_epochs: int,
    linear_decay_epochs: int,
) -> dict[str, Any]:
    """Execute one already-seeded epoch and aggregate trainer metrics."""

    if not isinstance(
        trainer,
        CycleGANTrainer,
    ):
        raise TypeError(
            "trainer must be CycleGANTrainer"
        )

    if not isinstance(
        epoch,
        int,
    ) or epoch <= 0:
        raise ValueError(
            "epoch must be a positive integer"
        )

    if not isinstance(
        global_step,
        int,
    ) or global_step < 0:
        raise ValueError(
            "global_step must be a non-negative integer"
        )

    torch_device = torch.device(
        device
    )

    learning_rate = apply_epoch_learning_rate(
        trainer.optimizers,
        epoch=epoch,
        base_lr=base_lr,
        constant_lr_epochs=constant_lr_epochs,
        linear_decay_epochs=linear_decay_epochs,
    )

    if torch_device.type == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but CUDA is unavailable"
            )

        torch.cuda.synchronize(
            torch_device
        )

        torch.cuda.reset_peak_memory_stats(
            torch_device
        )

    started = time.perf_counter()

    metric_sums: dict[str, float] = {}

    nonfinite_sums = {
        key: 0
        for key in _NONFINITE_KEYS
    }

    pool_sizes: dict[
        str,
        int,
    ] = {}

    extrema_min: dict[
        str,
        float,
    ] = {}

    extrema_max: dict[
        str,
        float,
    ] = {}

    metric_schema: set[str] | None = None

    batch_count = 0
    image_count = 0

    global_step_start = global_step

    for batch in batches:

        real_a, real_b = (
            _extract_domain_pair(
                batch
            )
        )

        real_a = real_a.to(
            torch_device,
            non_blocking=(
                torch_device.type
                == "cuda"
            ),
        )

        real_b = real_b.to(
            torch_device,
            non_blocking=(
                torch_device.type
                == "cuda"
            ),
        )

        metrics = trainer.train_step(
            real_a=real_a,
            real_b=real_b,
        )

        current_schema = set(
            metrics
        )

        if metric_schema is None:
            metric_schema = current_schema
        elif current_schema != metric_schema:
            raise RuntimeError(
                "Trainer metric schema changed within an epoch"
            )

        batch_count += 1

        image_count += int(
            real_a.shape[
                0
            ]
        )

        image_count += int(
            real_b.shape[
                0
            ]
        )

        global_step += 1

        for key, raw_value in metrics.items():

            if key in _NONFINITE_KEYS:

                value = int(
                    raw_value
                )

                if value < 0:
                    raise RuntimeError(
                        "Nonfinite-gradient count cannot be negative"
                    )

                nonfinite_sums[
                    key
                ] += value

                continue

            if key in _POOL_SIZE_KEYS:

                pool_sizes[
                    key
                ] = int(
                    raw_value
                )

                continue

            value = float(
                raw_value
            )

            if not math.isfinite(
                value
            ):
                raise FloatingPointError(
                    "Non-finite epoch metric: "
                    + key
                )

            if key in _MIN_KEYS:

                if (
                    key not in extrema_min
                    or value
                    < extrema_min[
                        key
                    ]
                ):
                    extrema_min[
                        key
                    ] = value

                continue

            if key in _MAX_KEYS:

                if (
                    key not in extrema_max
                    or value
                    > extrema_max[
                        key
                    ]
                ):
                    extrema_max[
                        key
                    ] = value

                continue

            metric_sums[
                key
            ] = (
                metric_sums.get(
                    key,
                    0.0,
                )
                + value
            )

    if batch_count <= 0:
        raise ValueError(
            "Epoch received zero batches"
        )

    if torch_device.type == "cuda":
        torch.cuda.synchronize(
            torch_device
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    if elapsed <= 0.0:
        raise RuntimeError(
            "Measured epoch duration must be positive"
        )

    metric_means = {
        key:
            value
            / batch_count
        for key, value
        in metric_sums.items()
    }

    images_per_second = (
        image_count
        / elapsed
    )

    if not math.isfinite(
        images_per_second
    ):
        raise RuntimeError(
            "images_per_second is non-finite"
        )

    return {
        "epoch": epoch,
        "epoch_seed": epoch_seed,
        "learning_rate": float(
            learning_rate
        ),
        "global_step_start": global_step_start,
        "global_step_end": global_step,
        "batch_count": batch_count,
        "image_count": image_count,
        "duration_seconds": float(
            elapsed
        ),
        "images_per_second": float(
            images_per_second
        ),
        "peak_cuda_memory_mib": (
            _cuda_peak_mib(
                torch_device
            )
        ),
        "metric_means": metric_means,
        "nonfinite_gradient_sums": (
            nonfinite_sums
        ),
        "final_pool_sizes": (
            pool_sizes
        ),
        "output_minima": (
            extrema_min
        ),
        "output_maxima": (
            extrema_max
        ),
    }


def run_epoch_range(
    *,
    trainer: CycleGANTrainer,
    loader_factory: Callable[
        [
            int,
            int,
        ],
        Iterable[Any],
    ],
    device: str | torch.device,
    start_epoch: int,
    end_epoch: int,
    global_step: int,
    base_seed: int,
    base_lr: float,
    constant_lr_epochs: int,
    linear_decay_epochs: int,
    on_epoch_complete: (
        Callable[
            [
                dict[str, Any],
                CycleGANTrainer,
            ],
            None,
        ]
        | None
    ) = None,
) -> list[dict[str, Any]]:
    """Execute an inclusive range of whole epochs deterministically."""

    if not callable(
        loader_factory
    ):
        raise TypeError(
            "loader_factory must be callable"
        )

    if not isinstance(
        start_epoch,
        int,
    ) or start_epoch <= 0:
        raise ValueError(
            "start_epoch must be positive"
        )

    if not isinstance(
        end_epoch,
        int,
    ) or end_epoch < start_epoch:
        raise ValueError(
            "end_epoch must be >= start_epoch"
        )

    if not isinstance(
        global_step,
        int,
    ) or global_step < 0:
        raise ValueError(
            "global_step must be non-negative"
        )

    total_schedule_epochs = (
        constant_lr_epochs
        + linear_decay_epochs
    )

    if end_epoch > total_schedule_epochs:
        raise ValueError(
            "end_epoch exceeds learning-rate schedule"
        )

    results: list[
        dict[
            str,
            Any,
        ]
    ] = []

    current_global_step = global_step

    for epoch in range(
        start_epoch,
        end_epoch + 1,
    ):

        epoch_seed = derive_epoch_seed(
            base_seed,
            epoch,
        )

        seed_process_for_epoch(
            epoch_seed
        )

        batches = loader_factory(
            epoch,
            epoch_seed,
        )

        result = run_training_epoch(
            trainer=trainer,
            batches=batches,
            device=device,
            epoch=epoch,
            epoch_seed=epoch_seed,
            global_step=current_global_step,
            base_lr=base_lr,
            constant_lr_epochs=constant_lr_epochs,
            linear_decay_epochs=linear_decay_epochs,
        )

        current_global_step = int(
            result[
                "global_step_end"
            ]
        )

        results.append(
            result
        )

        if on_epoch_complete is not None:
            on_epoch_complete(
                result,
                trainer,
            )

    return results
