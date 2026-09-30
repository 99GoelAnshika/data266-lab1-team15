"""Atomic checkpoint and resume support for the Task 3 CycleGAN.

Checkpoints produced by this module contain:

- all four CycleGAN network state dictionaries;
- all three Adam optimizer state dictionaries;
- both ImagePool replay-buffer objects and their internal RNG state;
- Python random state;
- PyTorch CPU RNG state;
- PyTorch CUDA RNG states when CUDA is available;
- completed epoch and global optimization-step counters;
- the SHA-256 fingerprint of the training configuration;
- optional plain metadata.

Each checkpoint is written to a temporary file in the destination
directory and atomically moved into place. A SHA-256 sidecar is also
written and verified before deserializing a checkpoint.

Only load checkpoints created by this project. ``torch.load`` uses
Python pickle semantics for the replay-pool objects.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from training_utils import ImagePool


__all__ = [
    "checkpoint_sha256_path",
    "file_sha256",
    "capture_rng_state",
    "restore_rng_state",
    "save_training_checkpoint",
    "verify_training_checkpoint",
    "load_training_checkpoint",
]


_SCHEMA_VERSION = 1

_NETWORK_KEYS = {
    "generator_a_to_b",
    "generator_b_to_a",
    "discriminator_a",
    "discriminator_b",
}

_OPTIMIZER_KEYS = {
    "generator",
    "discriminator_a",
    "discriminator_b",
}

_SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


def checkpoint_sha256_path(
    checkpoint_path: str | Path,
) -> Path:
    """Return the checksum-sidecar path for a checkpoint."""

    path = Path(
        checkpoint_path
    )

    return Path(
        str(path)
        + ".sha256"
    )


def file_sha256(
    path: str | Path,
) -> str:
    """Return a streaming SHA-256 digest for one file."""

    source = Path(
        path
    )

    digest = hashlib.sha256()

    with source.open(
        "rb"
    ) as handle:

        while True:
            chunk = handle.read(
                1024
                * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def _validate_sha256(
    value: str,
    *,
    name: str,
) -> str:

    normalized = str(
        value
    ).strip().lower()

    if not _SHA256_PATTERN.fullmatch(
        normalized
    ):
        raise ValueError(
            f"{name} must be a 64-character lowercase SHA-256 hex digest"
        )

    return normalized


def _validate_networks(
    networks: Mapping[str, nn.Module],
) -> None:

    if set(
        networks
    ) != _NETWORK_KEYS:
        raise ValueError(
            "networks must contain exactly the four CycleGAN networks"
        )

    for name, network in networks.items():

        if not isinstance(
            network,
            nn.Module,
        ):
            raise TypeError(
                f"{name} must be torch.nn.Module"
            )


def _validate_optimizers(
    optimizers: Mapping[str, Optimizer],
) -> None:

    if set(
        optimizers
    ) != _OPTIMIZER_KEYS:
        raise ValueError(
            "optimizers must contain exactly generator, "
            "discriminator_a, and discriminator_b"
        )

    for name, optimizer in optimizers.items():

        if not isinstance(
            optimizer,
            Optimizer,
        ):
            raise TypeError(
                f"{name} must be torch.optim.Optimizer"
            )


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, CPU Torch, and available CUDA RNG state."""

    cuda_states: list[torch.Tensor] = []

    if torch.cuda.is_available():
        cuda_states = [
            state.cpu()
            for state
            in torch.cuda.get_rng_state_all()
        ]

    return {
        "python": random.getstate(),
        "torch_cpu": (
            torch.get_rng_state()
            .cpu()
        ),
        "cuda_device_count": (
            torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0
        ),
        "torch_cuda": cuda_states,
    }


def restore_rng_state(
    state: Mapping[str, Any],
) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""

    required = {
        "python",
        "torch_cpu",
        "cuda_device_count",
        "torch_cuda",
    }

    if set(
        state
    ) != required:
        raise ValueError(
            "RNG-state structure is invalid"
        )

    random.setstate(
        state[
            "python"
        ]
    )

    cpu_state = state[
        "torch_cpu"
    ]

    if not isinstance(
        cpu_state,
        torch.Tensor,
    ):
        raise TypeError(
            "torch_cpu RNG state must be a tensor"
        )

    torch.set_rng_state(
        cpu_state.cpu()
    )

    saved_cuda_count = int(
        state[
            "cuda_device_count"
        ]
    )

    cuda_states = list(
        state[
            "torch_cuda"
        ]
    )

    if saved_cuda_count != len(
        cuda_states
    ):
        raise ValueError(
            "Saved CUDA RNG metadata is internally inconsistent"
        )

    if saved_cuda_count == 0:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Checkpoint contains CUDA RNG state but CUDA is unavailable"
        )

    current_cuda_count = (
        torch.cuda.device_count()
    )

    if current_cuda_count != saved_cuda_count:
        raise RuntimeError(
            "CUDA device count differs from the saved checkpoint; "
            "exact CUDA RNG restoration is not possible"
        )

    torch.cuda.set_rng_state_all(
        [
            state_tensor.cpu()
            for state_tensor
            in cuda_states
        ]
    )


def _atomic_write_text(
    path: Path,
    text: str,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = (
        tempfile.mkstemp(
            prefix=(
                path.name
                + "."
            ),
            suffix=".tmp",
            dir=str(
                path.parent
            ),
        )
    )

    temporary_path = Path(
        temporary_name
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:

            handle.write(
                text
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        os.replace(
            temporary_path,
            path,
        )

    except BaseException:
        try:
            temporary_path.unlink(
                missing_ok=True
            )
        finally:
            raise


def _atomic_torch_save(
    payload: Mapping[str, Any],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = (
        tempfile.mkstemp(
            prefix=(
                path.name
                + "."
            ),
            suffix=".tmp",
            dir=str(
                path.parent
            ),
        )
    )

    os.close(
        descriptor
    )

    temporary_path = Path(
        temporary_name
    )

    try:
        torch.save(
            dict(
                payload
            ),
            temporary_path,
        )

        os.replace(
            temporary_path,
            path,
        )

    except BaseException:
        try:
            temporary_path.unlink(
                missing_ok=True
            )
        finally:
            raise


def save_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    networks: Mapping[str, nn.Module],
    optimizers: Mapping[str, Optimizer],
    fake_a_pool: ImagePool,
    fake_b_pool: ImagePool,
    epoch_completed: int,
    global_step: int,
    config_sha256: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically save a trusted project checkpoint and checksum."""

    path = Path(
        checkpoint_path
    )

    _validate_networks(
        networks
    )

    _validate_optimizers(
        optimizers
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

    if not isinstance(
        epoch_completed,
        int,
    ):
        raise TypeError(
            "epoch_completed must be int"
        )

    if epoch_completed < 0:
        raise ValueError(
            "epoch_completed cannot be negative"
        )

    if not isinstance(
        global_step,
        int,
    ):
        raise TypeError(
            "global_step must be int"
        )

    if global_step < 0:
        raise ValueError(
            "global_step cannot be negative"
        )

    normalized_config_hash = (
        _validate_sha256(
            config_sha256,
            name="config_sha256",
        )
    )

    metadata_dict = (
        {}
        if metadata is None
        else dict(
            metadata
        )
    )

    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "epoch_completed": epoch_completed,
        "global_step": global_step,
        "config_sha256": normalized_config_hash,
        "network_state_dicts": {
            name:
                network.state_dict()
            for name, network
            in networks.items()
        },
        "optimizer_state_dicts": {
            name:
                optimizer.state_dict()
            for name, optimizer
            in optimizers.items()
        },
        "replay_pools": {
            "fake_a_pool": fake_a_pool,
            "fake_b_pool": fake_b_pool,
        },
        "rng_state": capture_rng_state(),
        "metadata": metadata_dict,
    }

    _atomic_torch_save(
        payload,
        path,
    )

    digest = file_sha256(
        path
    )

    sidecar = checkpoint_sha256_path(
        path
    )

    _atomic_write_text(
        sidecar,
        digest
        + "\n",
    )

    return {
        "checkpoint_path": path,
        "sha256_path": sidecar,
        "sha256": digest,
        "epoch_completed": epoch_completed,
        "global_step": global_step,
        "config_sha256": normalized_config_hash,
    }


def verify_training_checkpoint(
    checkpoint_path: str | Path,
) -> str:
    """Verify a checkpoint against its SHA-256 sidecar."""

    path = Path(
        checkpoint_path
    )

    sidecar = checkpoint_sha256_path(
        path
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}"
        )

    if not sidecar.is_file():
        raise FileNotFoundError(
            f"Checkpoint SHA-256 sidecar does not exist: {sidecar}"
        )

    expected = _validate_sha256(
        sidecar.read_text(
            encoding="utf-8"
        ).strip(),
        name="checkpoint sidecar SHA-256",
    )

    actual = file_sha256(
        path
    )

    if actual != expected:
        raise RuntimeError(
            "Checkpoint SHA-256 verification failed"
        )

    return actual


def _validate_loaded_payload(
    payload: Mapping[str, Any],
) -> None:

    required = {
        "schema_version",
        "epoch_completed",
        "global_step",
        "config_sha256",
        "network_state_dicts",
        "optimizer_state_dicts",
        "replay_pools",
        "rng_state",
        "metadata",
    }

    if set(
        payload
    ) != required:
        raise RuntimeError(
            "Checkpoint payload structure is invalid"
        )

    if payload[
        "schema_version"
    ] != _SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported checkpoint schema version"
        )

    if set(
        payload[
            "network_state_dicts"
        ]
    ) != _NETWORK_KEYS:
        raise RuntimeError(
            "Checkpoint network-state structure is invalid"
        )

    if set(
        payload[
            "optimizer_state_dicts"
        ]
    ) != _OPTIMIZER_KEYS:
        raise RuntimeError(
            "Checkpoint optimizer-state structure is invalid"
        )

    replay_pools = payload[
        "replay_pools"
    ]

    if set(
        replay_pools
    ) != {
        "fake_a_pool",
        "fake_b_pool",
    }:
        raise RuntimeError(
            "Checkpoint replay-pool structure is invalid"
        )

    if not isinstance(
        replay_pools[
            "fake_a_pool"
        ],
        ImagePool,
    ):
        raise RuntimeError(
            "Checkpoint fake_a_pool type is invalid"
        )

    if not isinstance(
        replay_pools[
            "fake_b_pool"
        ],
        ImagePool,
    ):
        raise RuntimeError(
            "Checkpoint fake_b_pool type is invalid"
        )

    if int(
        payload[
            "epoch_completed"
        ]
    ) < 0:
        raise RuntimeError(
            "Checkpoint epoch is invalid"
        )

    if int(
        payload[
            "global_step"
        ]
    ) < 0:
        raise RuntimeError(
            "Checkpoint global step is invalid"
        )

    _validate_sha256(
        payload[
            "config_sha256"
        ],
        name="checkpoint config_sha256",
    )


def load_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    networks: Mapping[str, nn.Module],
    optimizers: Mapping[str, Optimizer],
    map_location: str | torch.device,
    expected_config_sha256: str | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Verify and load a trusted project checkpoint.

    The supplied network and optimizer objects are mutated in place.
    The restored replay-pool objects are returned to the caller.
    """

    path = Path(
        checkpoint_path
    )

    _validate_networks(
        networks
    )

    _validate_optimizers(
        optimizers
    )

    digest = verify_training_checkpoint(
        path
    )

    # This project intentionally stores ImagePool Python objects.
    # Therefore only self-created trusted checkpoints may be loaded.
    payload = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(
        payload,
        Mapping,
    ):
        raise RuntimeError(
            "Checkpoint payload must be a mapping"
        )

    _validate_loaded_payload(
        payload
    )

    checkpoint_config_hash = (
        _validate_sha256(
            payload[
                "config_sha256"
            ],
            name="checkpoint config_sha256",
        )
    )

    if expected_config_sha256 is not None:

        expected = _validate_sha256(
            expected_config_sha256,
            name="expected_config_sha256",
        )

        if checkpoint_config_hash != expected:
            raise RuntimeError(
                "Checkpoint configuration fingerprint does not match"
            )

    for name, network in networks.items():

        network.load_state_dict(
            payload[
                "network_state_dicts"
            ][
                name
            ],
            strict=True,
        )

    for name, optimizer in optimizers.items():

        optimizer.load_state_dict(
            payload[
                "optimizer_state_dicts"
            ][
                name
            ]
        )

    if restore_rng:
        restore_rng_state(
            payload[
                "rng_state"
            ]
        )

    replay_pools = payload[
        "replay_pools"
    ]

    return {
        "schema_version": int(
            payload[
                "schema_version"
            ]
        ),
        "epoch_completed": int(
            payload[
                "epoch_completed"
            ]
        ),
        "global_step": int(
            payload[
                "global_step"
            ]
        ),
        "config_sha256": checkpoint_config_hash,
        "checkpoint_sha256": digest,
        "fake_a_pool": replay_pools[
            "fake_a_pool"
        ],
        "fake_b_pool": replay_pools[
            "fake_b_pool"
        ],
        "metadata": dict(
            payload[
                "metadata"
            ]
        ),
    }
