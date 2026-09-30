"""Unpaired image-data utilities for DATA 266 Task 3 CycleGAN.

This module defines:

- deterministic image-file discovery;
- independent unpaired domain sampling;
- canonical CycleGAN train/evaluation transforms;
- reproducible DataLoader construction.

It does not download data, train models, save checkpoints, access Kaggle,
or assume any user-specific absolute filesystem path.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import (
    DataLoader,
    Dataset,
    get_worker_info,
)
from torchvision import transforms


__all__ = [
    "IMAGE_EXTENSIONS",
    "discover_images",
    "build_train_transform",
    "build_eval_transform",
    "UnpairedImageDataset",
    "seed_worker",
    "build_dataloader",
]


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def discover_images(
    directory: str | Path,
) -> list[Path]:
    """Return sorted supported image files from one directory."""

    root = Path(
        directory
    )

    if not root.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Image path is not a directory: {root}"
        )

    files = sorted(
        path
        for path in root.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not files:
        raise RuntimeError(
            f"No supported image files found in: {root}"
        )

    return files


def build_train_transform(
    *,
    resize_size: int = 286,
    crop_size: int = 256,
    horizontal_flip_probability: float = 0.5,
) -> transforms.Compose:
    """Build canonical stochastic CycleGAN training preprocessing."""

    if resize_size <= 0:
        raise ValueError(
            "resize_size must be positive"
        )

    if crop_size <= 0:
        raise ValueError(
            "crop_size must be positive"
        )

    if resize_size < crop_size:
        raise ValueError(
            "resize_size must be >= crop_size"
        )

    if not (
        0.0
        <= horizontal_flip_probability
        <= 1.0
    ):
        raise ValueError(
            "horizontal_flip_probability must be in [0, 1]"
        )

    return transforms.Compose(
        [
            transforms.Resize(
                (
                    resize_size,
                    resize_size,
                ),
                interpolation=(
                    transforms.InterpolationMode.BICUBIC
                ),
            ),
            transforms.RandomCrop(
                crop_size
            ),
            transforms.RandomHorizontalFlip(
                p=horizontal_flip_probability
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(
                    0.5,
                    0.5,
                    0.5,
                ),
                std=(
                    0.5,
                    0.5,
                    0.5,
                ),
            ),
        ]
    )


def build_eval_transform(
    *,
    image_size: int = 256,
) -> transforms.Compose:
    """Build deterministic CycleGAN evaluation preprocessing."""

    if image_size <= 0:
        raise ValueError(
            "image_size must be positive"
        )

    return transforms.Compose(
        [
            transforms.Resize(
                (
                    image_size,
                    image_size,
                ),
                interpolation=(
                    transforms.InterpolationMode.BICUBIC
                ),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(
                    0.5,
                    0.5,
                    0.5,
                ),
                std=(
                    0.5,
                    0.5,
                    0.5,
                ),
            ),
        ]
    )


class UnpairedImageDataset(
    Dataset[dict[str, Any]]
):
    """Two-domain unpaired image dataset.

    Domain A is indexed deterministically by the incoming sample index.
    During training, domain B is sampled independently. Direct
    single-process access uses the dataset-local seeded RNG, while
    DataLoader workers use PyTorch's worker-specific seeded RNG.
    During deterministic/evaluation mode, both domains use modulo
    indexing so repeated evaluation is reproducible.

    Dataset length is max(len(A), len(B)), which gives every image from
    the larger domain one visit per logical epoch.
    """

    def __init__(
        self,
        domain_a_dir: str | Path,
        domain_b_dir: str | Path,
        *,
        transform: Any,
        training: bool,
        seed: int = 0,
    ) -> None:
        super().__init__()

        if transform is None:
            raise ValueError(
                "transform cannot be None"
            )

        self.domain_a_files = discover_images(
            domain_a_dir
        )

        self.domain_b_files = discover_images(
            domain_b_dir
        )

        self.transform = transform
        self.training = bool(
            training
        )

        self._rng = random.Random(
            seed
        )

    def __len__(self) -> int:
        return max(
            len(
                self.domain_a_files
            ),
            len(
                self.domain_b_files
            ),
        )

    @staticmethod
    def _load_rgb(
        path: Path,
    ) -> Image.Image:
        with Image.open(path) as image:
            return image.convert(
                "RGB"
            )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        if not isinstance(
            index,
            int,
        ):
            raise TypeError(
                "index must be an integer"
            )

        if index < 0:
            index = (
                len(self)
                + index
            )

        if not (
            0
            <= index
            < len(self)
        ):
            raise IndexError(
                "dataset index out of range"
            )

        a_index = (
            index
            % len(
                self.domain_a_files
            )
        )

        if self.training:
            worker_info = get_worker_info()

            if worker_info is None:
                b_index = self._rng.randrange(
                    len(
                        self.domain_b_files
                    )
                )
            else:
                b_index = int(
                    torch.randint(
                        low=0,
                        high=len(
                            self.domain_b_files
                        ),
                        size=(1,),
                    ).item()
                )
        else:
            b_index = (
                index
                % len(
                    self.domain_b_files
                )
            )

        a_path = self.domain_a_files[
            a_index
        ]

        b_path = self.domain_b_files[
            b_index
        ]

        image_a = self.transform(
            self._load_rgb(
                a_path
            )
        )

        image_b = self.transform(
            self._load_rgb(
                b_path
            )
        )

        if not isinstance(
            image_a,
            torch.Tensor,
        ):
            raise TypeError(
                "transform must return a torch.Tensor for domain A"
            )

        if not isinstance(
            image_b,
            torch.Tensor,
        ):
            raise TypeError(
                "transform must return a torch.Tensor for domain B"
            )

        return {
            "A": image_a,
            "B": image_b,
            "A_path": str(
                a_path
            ),
            "B_path": str(
                b_path
            ),
            "A_index": a_index,
            "B_index": b_index,
        }


def seed_worker(
    worker_id: int,
) -> None:
    """Seed Python RNG from the PyTorch-assigned worker seed."""

    worker_seed = (
        torch.initial_seed()
        % 2**32
    )

    random.seed(
        worker_seed
    )


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Build a reproducibly seeded PyTorch DataLoader."""

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative"
        )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(
            num_workers > 0
        ),
    )
