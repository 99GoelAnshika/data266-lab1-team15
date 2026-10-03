"""Production CycleGAN training entry point for DATA 266 Lab 1 Task 3.

The launcher composes the separately validated Task 3 modules. Full training requires CUDA. The actual GPU and precision are recorded;
smoke runs may use CPU. This is an explicitly attributed adaptation of team code.

The canonical JSON configuration is treated as immutable run identity:
its full-file SHA-256 is recorded in the run manifest and checkpoints.
Resume is supported only at completed epoch boundaries.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch
from torchvision.utils import save_image

from checkpointing import (
    file_sha256,
    load_training_checkpoint,
    save_training_checkpoint,
    verify_training_checkpoint,
)
from data import (
    UnpairedImageDataset,
    build_dataloader,
    build_eval_transform,
    build_train_transform,
    discover_images,
)
from models import (
    build_cyclegan_models,
    count_trainable_parameters,
)
from run_evidence import (
    RunEvidenceWriter,
    collect_runtime_environment,
)
from trainer import (
    CycleGANTrainer,
    build_optimizers,
)
from training_loop import (
    next_epoch_after_checkpoint,
    run_epoch_range,
)
from training_utils import (
    CycleGANLoss,
    ImagePool,
)


RUN_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)

DEFAULT_CONFIG_RELATIVE = Path(
    "task3_gan"
) / "Viraat_Chaudhary" / "configs" / "cyclegan_viraat.json"

TASK3_MEMBER_RELATIVE = (
    Path(
        "task3_gan"
    )
    / "Viraat_Chaudhary"
)

LATEST_CHECKPOINT_NAME = "latest.pt"


def repository_root() -> Path:
    """Return repository root from this file's committed location."""

    return Path(
        __file__
    ).resolve().parents[
        3
    ]


def resolve_repository_path(
    root: Path,
    value: str | Path,
) -> Path:
    """Resolve a repository-relative path without changing the config."""

    path = Path(
        value
    )

    if path.is_absolute():
        return path.resolve()

    return (
        root
        / path
    ).resolve()


def repository_relative_string(
    root: Path,
    path: Path,
) -> str:
    """Return a slash-normalized repository-relative path."""

    try:
        return path.resolve().relative_to(
            root.resolve()
        ).as_posix()

    except ValueError:
        return "<external-smoke-path>"


def validate_run_id(
    run_id: str,
) -> str:
    """Validate a filesystem-safe run identifier."""

    if not isinstance(
        run_id,
        str,
    ):
        raise TypeError(
            "run_id must be a string"
        )

    if RUN_ID_PATTERN.fullmatch(
        run_id
    ) is None:
        raise ValueError(
            "run_id must match "
            "[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )

    return run_id


def load_config(
    config_path: Path,
) -> tuple[
    dict[str, Any],
    str,
]:
    """Load canonical configuration and compute immutable SHA-256."""

    if not config_path.is_file():
        raise FileNotFoundError(
            "Configuration file not found: "
            + str(
                config_path
            )
        )

    config_bytes = config_path.read_bytes()

    try:
        config = json.loads(
            config_bytes.decode(
                "utf-8"
            )
        )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            "Configuration is not valid UTF-8 JSON"
        ) from exc

    if not isinstance(
        config,
        dict,
    ):
        raise RuntimeError(
            "Top-level configuration must be an object"
        )

    return (
        config,
        file_sha256(
            config_path
        ),
    )


def require_mapping(
    parent: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Return a required mapping-valued configuration field."""

    value = parent.get(
        key
    )

    if not isinstance(
        value,
        Mapping,
    ):
        raise RuntimeError(
            "Configuration field must be an object: "
            + key
        )

    return value


def validate_canonical_config(config):
    """Validate this experiment rather than silently imposing baseline settings."""
    if config.get('schema_version') != 1:
        raise RuntimeError('Unsupported config schema')
    for key in ['domains', 'model', 'preprocessing', 'training', 'loss', 'dataloader',
                'checkpointing', 'monitoring', 'integrity', 'experiment']:
        require_mapping(config, key)
    g, d = config['model']['generator'], config['model']['discriminator']
    t, loss = config['training'], config['loss']
    if config['domains'].get('unpaired_training') is not True:
        raise RuntimeError('Unpaired training is required')
    if g['architecture'] != 'resnet_6block_resizeconv' or g['residual_blocks'] != 6:
        raise RuntimeError('Expected the six-block resize-convolution experiment')
    if g.get('upsampling') != 'nearest_resize_conv':
        raise RuntimeError('Generator upsampling config mismatch')
    if d['architecture'] != 'patchgan_70x70' or d['layers'] != 3:
        raise RuntimeError('Expected a 70x70 PatchGAN')
    if g['input_channels'] != 3 or g['output_channels'] != 3 or d['input_channels'] != 3:
        raise RuntimeError('This task uses RGB images')
    if t['optimizer'] != 'adam' or t['precision_dtype'] != 'bfloat16':
        raise RuntimeError('Unsupported optimizer/precision')
    if t['gradient_clip_norm'] is not None:
        raise RuntimeError('Gradient clipping is not enabled in this experiment')
    if t['epochs'] != t['constant_lr_epochs'] + t['linear_decay_epochs']:
        raise RuntimeError('Epoch count must cover the entire learning-rate schedule')
    if min(t['epochs'], t['batch_size'], t['constant_lr_epochs'], t['linear_decay_epochs']) < 1:
        raise RuntimeError('Training sizes must be positive')
    if t['image_pool_size'] < 0 or config['dataloader']['num_workers'] < 0:
        raise RuntimeError('Negative replay-pool/worker count')
    if loss['adversarial'] != 'least_squares_gan_mse' or loss['cycle'] != 'l1' or loss['identity'] != 'l1':
        raise RuntimeError('Unsupported loss')
    if min(loss['lambda_cycle_a'], loss['lambda_cycle_b'], loss['lambda_identity_ratio']) < 0:
        raise RuntimeError('Loss weights cannot be negative')
    for suffix in ['a', 'b']:
        if abs(loss['effective_identity_weight_' + suffix] -
               loss['lambda_cycle_' + suffix] * loss['lambda_identity_ratio']) > 1e-10:
            raise RuntimeError('Identity-weight metadata disagrees with the objective')
    if not config['checkpointing']['atomic_writes'] or not config['checkpointing']['resume_supported']:
        raise RuntimeError('Atomic checkpoints and resume must remain enabled')
    for k in ['save_every_epochs', 'save_latest_every_epochs']:
        if config['checkpointing'][k] < 1:
            raise RuntimeError('Checkpoint intervals must be positive')
    for k in ['log_every_steps', 'sample_every_epochs']:
        if config['monitoring'][k] < 1:
            raise RuntimeError('Monitoring intervals must be positive')
    expected = {'train_from_scratch': True, 'pretrained_generation_models': False,
        'foundation_models_for_submission_generation': False, 'test_pairing_access': False,
        'manual_submission_image_editing': False,
        'submission_source': 'direct_output_of_user_trained_cyclegan'}
    for k, v in expected.items():
        if config['integrity'].get(k) != v:
            raise RuntimeError('Integrity field mismatch: ' + k)


def resolve_device(*, mode, requested_device):
    """Accept the actual lab CUDA GPU; never substitute CPU for full training."""
    if mode not in {'smoke', 'full'} or requested_device not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('Invalid mode/device')
    if mode == 'full' or requested_device == 'cuda':
        if requested_device == 'cpu' or not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; check the Colab local runtime connection')
        torch.cuda.set_device(0)
        return torch.device('cuda:0')
    return torch.device('cpu')


def resolve_domain_directories(
    *,
    root: Path,
    config: Mapping[str, Any],
    mode: str,
    domain_a_override: str | None,
    domain_b_override: str | None,
) -> tuple[
    Path,
    Path,
]:
    """Resolve training domains while protecting canonical full mode."""

    domains = require_mapping(
        config,
        "domains",
    )

    if mode == "full" and (
        domain_a_override is not None
        or domain_b_override is not None
    ):
        raise RuntimeError(
            "Full mode does not permit domain directory overrides"
        )

    domain_a_value = (
        domain_a_override
        if domain_a_override is not None
        else domains[
            "domain_a_directory"
        ]
    )

    domain_b_value = (
        domain_b_override
        if domain_b_override is not None
        else domains[
            "domain_b_directory"
        ]
    )

    return (
        resolve_repository_path(
            root,
            domain_a_value,
        ),
        resolve_repository_path(
            root,
            domain_b_value,
        ),
    )


def validate_dataset(
    domain_a_dir: Path,
    domain_b_dir: Path,
) -> tuple[
    list[Path],
    list[Path],
]:
    """Discover both unpaired domains and reject empty datasets."""

    domain_a_images = discover_images(
        domain_a_dir
    )

    domain_b_images = discover_images(
        domain_b_dir
    )

    if not domain_a_images:
        raise RuntimeError(
            "Domain A contains no supported images"
        )

    if not domain_b_images:
        raise RuntimeError(
            "Domain B contains no supported images"
        )

    return (
        domain_a_images,
        domain_b_images,
    )


def build_networks(
    *,
    config: Mapping[str, Any],
    device: torch.device,
) -> Mapping[
    str,
    torch.nn.Module,
]:
    """Build and move canonical CycleGAN networks to selected device."""

    model = require_mapping(
        config,
        "model",
    )

    generator = require_mapping(
        model,
        "generator",
    )

    discriminator = require_mapping(
        model,
        "discriminator",
    )

    networks = build_cyclegan_models(
        input_channels=int(
            generator[
                "input_channels"
            ]
        ),
        output_channels=int(
            generator[
                "output_channels"
            ]
        ),
        generator_channels=int(
            generator[
                "base_channels"
            ]
        ),
        discriminator_channels=int(
            discriminator[
                "base_channels"
            ]
        ),
        residual_blocks=int(
            generator[
                "residual_blocks"
            ]
        ),
        discriminator_layers=int(
            discriminator[
                "layers"
            ]
        ),
        upsampling=generator["upsampling"],
    )

    for network in networks.values():
        network.to(
            device
        )

    return networks


def build_optimizer_mapping(
    *,
    config: Mapping[str, Any],
    networks: Mapping[
        str,
        torch.nn.Module,
    ],
):
    """Build validated optimizer mapping."""

    training = require_mapping(
        config,
        "training",
    )

    return build_optimizers(
        networks,
        learning_rate=float(
            training[
                "learning_rate"
            ]
        ),
        beta1=float(
            training[
                "beta1"
            ]
        ),
        beta2=float(
            training[
                "beta2"
            ]
        ),
        weight_decay=float(
            training[
                "weight_decay"
            ]
        ),
    )


def build_loss_bundle(
    config: Mapping[str, Any],
) -> CycleGANLoss:
    """Build validated CycleGAN loss bundle."""

    loss = require_mapping(
        config,
        "loss",
    )

    return CycleGANLoss(
        lambda_cycle_a=float(
            loss[
                "lambda_cycle_a"
            ]
        ),
        lambda_cycle_b=float(
            loss[
                "lambda_cycle_b"
            ]
        ),
        lambda_identity=float(
            loss[
                "lambda_identity_ratio"
            ]
        ),
    )


def make_new_pools(
    config: Mapping[str, Any],
) -> tuple[
    ImagePool,
    ImagePool,
]:
    """Create deterministic replay pools for a fresh run."""

    training = require_mapping(
        config,
        "training",
    )

    experiment = require_mapping(
        config,
        "experiment",
    )

    pool_size = int(
        training[
            "image_pool_size"
        ]
    )

    base_seed = int(
        experiment[
            "seed"
        ]
    )

    return (
        ImagePool(
            pool_size=pool_size,
            seed=base_seed + 101,
        ),
        ImagePool(
            pool_size=pool_size,
            seed=base_seed + 202,
        ),
    )


def ensure_fresh_destination(
    path: Path,
    *,
    label: str,
) -> None:
    """Reject accidental overwrite of an existing run destination."""

    if path.exists():
        raise FileExistsError(
            label
            + " already exists: "
            + str(
                path
            )
        )


def build_epoch_loader_factory(
    *,
    config: Mapping[str, Any],
    domain_a_dir: Path,
    domain_b_dir: Path,
    mode: str,
    device: torch.device,
):
    """Return deterministic per-epoch DataLoader factory."""

    preprocessing = require_mapping(
        config,
        "preprocessing",
    )

    training = require_mapping(
        config,
        "training",
    )

    dataloader = require_mapping(
        config,
        "dataloader",
    )

    configured_workers = int(
        dataloader[
            "num_workers"
        ]
    )

    actual_workers = (
        configured_workers
        if mode == "full"
        else 0
    )

    actual_pin_memory = (
        bool(
            dataloader[
                "pin_memory"
            ]
        )
        and device.type == "cuda"
    )

    def loader_factory(
        epoch: int,
        epoch_seed: int,
    ):
        del epoch

        transform = build_train_transform(
            resize_size=int(
                preprocessing[
                    "train_resize"
                ]
            ),
            crop_size=int(
                preprocessing[
                    "train_crop"
                ]
            ),
            horizontal_flip_probability=float(
                preprocessing[
                    "horizontal_flip_probability"
                ]
            ),
        )

        dataset = UnpairedImageDataset(
            domain_a_dir,
            domain_b_dir,
            transform=transform,
            training=True,
            seed=epoch_seed,
        )

        return build_dataloader(
            dataset,
            batch_size=int(
                training[
                    "batch_size"
                ]
            ),
            shuffle=bool(
                dataloader[
                    "shuffle"
                ]
            ),
            num_workers=actual_workers,
            seed=epoch_seed,
            pin_memory=actual_pin_memory,
            drop_last=bool(
                dataloader[
                    "drop_last"
                ]
            ),
        )

    return (
        loader_factory,
        actual_workers,
        actual_pin_memory,
    )


def build_monitoring_dataset(
    *,
    config: Mapping[str, Any],
    domain_a_dir: Path,
    domain_b_dir: Path,
) -> UnpairedImageDataset:
    """Build deterministic evaluation-transform dataset for samples."""

    preprocessing = require_mapping(
        config,
        "preprocessing",
    )

    experiment = require_mapping(
        config,
        "experiment",
    )

    transform = build_eval_transform(
        image_size=int(
            preprocessing[
                "evaluation_size"
            ]
        )
    )

    return UnpairedImageDataset(
        domain_a_dir,
        domain_b_dir,
        transform=transform,
        training=False,
        seed=int(
            experiment[
                "seed"
            ]
        ),
    )


def save_monitoring_samples(
    *,
    trainer: CycleGANTrainer,
    dataset: UnpairedImageDataset,
    device: torch.device,
    epoch: int,
    sample_dir: Path,
    sample_count: int,
) -> list[str]:
    """Save deterministic direct CycleGAN monitoring grids."""

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = min(
        sample_count,
        len(
            dataset
        ),
    )

    if count <= 0:
        raise RuntimeError(
            "Monitoring dataset is empty"
        )

    generator_a_to_b = trainer.networks[
        "generator_a_to_b"
    ]

    generator_b_to_a = trainer.networks[
        "generator_b_to_a"
    ]

    generator_a_to_b.eval()
    generator_b_to_a.eval()

    written: list[str] = []

    try:

        with torch.no_grad():

            for index in range(
                count
            ):

                sample = dataset[
                    index
                ]

                real_a = sample[
                    "A"
                ].unsqueeze(
                    0
                ).to(
                    device
                )

                real_b = sample[
                    "B"
                ].unsqueeze(
                    0
                ).to(
                    device
                )

                fake_b = generator_a_to_b(
                    real_a
                )

                recovered_a = generator_b_to_a(
                    fake_b
                )

                fake_a = generator_b_to_a(
                    real_b
                )

                recovered_b = generator_a_to_b(
                    fake_a
                )

                grid = torch.cat(
                    [
                        real_a,
                        fake_b,
                        recovered_a,
                        real_b,
                        fake_a,
                        recovered_b,
                    ],
                    dim=0,
                )

                grid = (
                    (
                        grid.detach().cpu()
                        + 1.0
                    )
                    / 2.0
                ).clamp(
                    0.0,
                    1.0,
                )

                filename = (
                    f"epoch_{epoch:04d}_"
                    f"sample_{index:03d}.png"
                )

                path = (
                    sample_dir
                    / filename
                )

                save_image(
                    grid,
                    path,
                    nrow=3,
                )

                written.append(
                    filename
                )

    finally:
        trainer.train_mode()

    return written


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the DATA 266 Task 3 CycleGAN resize-convolution experiment."
        )
    )

    parser.add_argument(
        "--config",
        default=str(
            DEFAULT_CONFIG_RELATIVE
        ),
        help=(
            "Canonical JSON config path relative to repository root."
        ),
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help=(
            "Unique run identifier used for logs, checkpoints, and outputs."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "full",
        ],
        required=True,
        help=(
            "smoke for development validation; "
            "full for serious CUDA training."
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help=(
            "Smoke-mode device selector. "
            "Full mode always resolves to cuda:0."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume this run ID from its verified latest checkpoint."
        ),
    )

    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=1,
        help=(
            "Number of epochs in smoke mode. Ignored in full mode."
        ),
    )

    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help=(
            "Number of deterministic monitoring examples saved "
            "when sampling is due."
        ),
    )

    parser.add_argument(
        "--domain-a-dir",
        default=None,
        help=(
            "Smoke-only domain A override."
        ),
    )

    parser.add_argument(
        "--domain-b-dir",
        default=None,
        help=(
            "Smoke-only domain B override."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Execute fresh or whole-epoch-resumed CycleGAN training."""

    args = parse_arguments()

    root = repository_root()

    run_id = validate_run_id(
        args.run_id
    )

    if args.smoke_epochs <= 0:
        raise ValueError(
            "--smoke-epochs must be positive"
        )

    if args.sample_count <= 0:
        raise ValueError(
            "--sample-count must be positive"
        )

    config_path = resolve_repository_path(
        root,
        args.config,
    )

    config, config_sha256 = load_config(
        config_path
    )

    validate_canonical_config(
        config
    )

    device = resolve_device(
        mode=args.mode,
        requested_device=args.device,
    )

    domain_a_dir, domain_b_dir = (
        resolve_domain_directories(
            root=root,
            config=config,
            mode=args.mode,
            domain_a_override=args.domain_a_dir,
            domain_b_override=args.domain_b_dir,
        )
    )

    domain_a_images, domain_b_images = (
        validate_dataset(
            domain_a_dir,
            domain_b_dir,
        )
    )

    member_root = (
        root
        / TASK3_MEMBER_RELATIVE
    )

    checkpoint_dir = (
        member_root
        / "checkpoints"
        / run_id
    )

    evidence_dir = (
        member_root
        / "logs"
        / run_id
    )

    output_dir = (
        member_root
        / "outputs"
        / run_id
    )

    sample_dir = (
        output_dir
        / "samples"
    )

    latest_checkpoint = (
        checkpoint_dir
        / LATEST_CHECKPOINT_NAME
    )

    training = require_mapping(
        config,
        "training",
    )

    experiment = require_mapping(
        config,
        "experiment",
    )

    checkpointing = require_mapping(
        config,
        "checkpointing",
    )

    monitoring = require_mapping(
        config,
        "monitoring",
    )

    initialization_seed = int(
        experiment[
            "seed"
        ]
    )

    torch.manual_seed(
        initialization_seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            initialization_seed
        )

    networks = build_networks(
        config=config,
        device=device,
    )

    optimizers = build_optimizer_mapping(
        config=config,
        networks=networks,
    )

    losses = build_loss_bundle(
        config
    )

    parameter_counts = {
        name: count_trainable_parameters(
            network
        )
        for name, network
        in networks.items()
    }

    total_parameters = sum(
        parameter_counts.values()
    )

    total_epochs = int(
        training[
            "epochs"
        ]
    )

    if args.mode == "smoke":
        requested_end_epoch = min(
            total_epochs,
            args.smoke_epochs,
        )

    else:
        requested_end_epoch = total_epochs

    loader_factory, actual_workers, actual_pin_memory = (
        build_epoch_loader_factory(
            config=config,
            domain_a_dir=domain_a_dir,
            domain_b_dir=domain_b_dir,
            mode=args.mode,
            device=device,
        )
    )

    monitoring_dataset = build_monitoring_dataset(
        config=config,
        domain_a_dir=domain_a_dir,
        domain_b_dir=domain_b_dir,
    )

    if args.resume:

        if not evidence_dir.is_dir():
            raise FileNotFoundError(
                "Evidence directory missing for resume: "
                + str(
                    evidence_dir
                )
            )

        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                "Checkpoint directory missing for resume: "
                + str(
                    checkpoint_dir
                )
            )

        if not latest_checkpoint.is_file():
            raise FileNotFoundError(
                "Latest checkpoint missing for resume: "
                + str(
                    latest_checkpoint
                )
            )

        writer = RunEvidenceWriter.resume(
            evidence_dir,
            run_id=run_id,
        )

        existing_manifest = json.loads(
            writer.manifest_path.read_text(
                encoding="utf-8"
            )
        )

        existing_manifest_payload = (
            existing_manifest.get(
                "manifest"
            )
        )

        if not isinstance(
            existing_manifest_payload,
            dict,
        ):
            raise RuntimeError(
                "Run manifest payload is invalid"
            )

        if existing_manifest_payload.get(
            "config_sha256"
        ) != config_sha256:
            raise RuntimeError(
                "Run manifest configuration fingerprint does not match"
            )

        loaded = load_training_checkpoint(
            latest_checkpoint,
            networks=networks,
            optimizers=optimizers,
            map_location=device,
            expected_config_sha256=config_sha256,
            restore_rng=True,
        )

        fake_a_pool = loaded[
            "fake_a_pool"
        ]

        fake_b_pool = loaded[
            "fake_b_pool"
        ]

        global_step = int(
            loaded[
                "global_step"
            ]
        )

        start_epoch = next_epoch_after_checkpoint(
            int(
                loaded[
                    "epoch_completed"
                ]
            ),
            total_epochs=total_epochs,
        )

        writer.append_event(
            "run_resume",
            {
                "checkpoint": LATEST_CHECKPOINT_NAME,
                "checkpoint_sha256": loaded[
                    "checkpoint_sha256"
                ],
                "epoch_completed": int(
                    loaded[
                        "epoch_completed"
                    ]
                ),
                "global_step": global_step,
                "next_epoch": start_epoch,
            },
        )

    else:

        ensure_fresh_destination(
            evidence_dir,
            label="Evidence run directory",
        )

        ensure_fresh_destination(
            checkpoint_dir,
            label="Checkpoint run directory",
        )

        ensure_fresh_destination(
            output_dir,
            label="Output run directory",
        )

        fake_a_pool, fake_b_pool = (
            make_new_pools(
                config
            )
        )

        start_epoch = 1
        global_step = 0

        runtime = collect_runtime_environment(
            selected_device=device,
        )

        manifest = {
            "purpose": "cyclegan_training_run",
            "run_id": run_id,
            "mode": args.mode,
            "config_path": repository_relative_string(
                root,
                config_path,
            ),
            "config_sha256": config_sha256,
            "runtime": runtime,
            "dataset": {
                "domain_a_name": require_mapping(
                    config,
                    "domains",
                )[
                    "domain_a"
                ],
                "domain_b_name": require_mapping(
                    config,
                    "domains",
                )[
                    "domain_b"
                ],
                "domain_a_count": len(
                    domain_a_images
                ),
                "domain_b_count": len(
                    domain_b_images
                ),
                "unpaired_training": True,
            },
            "training": {
                "configured_total_epochs": total_epochs,
                "requested_end_epoch": requested_end_epoch,
                "batch_size": int(
                    training[
                        "batch_size"
                    ]
                ),
                "base_learning_rate": float(
                    training[
                        "learning_rate"
                    ]
                ),
                "constant_lr_epochs": int(
                    training[
                        "constant_lr_epochs"
                    ]
                ),
                "linear_decay_epochs": int(
                    training[
                        "linear_decay_epochs"
                    ]
                ),
                "actual_num_workers": actual_workers,
                "actual_pin_memory": actual_pin_memory,
                "sample_count": args.sample_count,
            },
            "parameter_counts": parameter_counts,
            "total_trainable_parameters": total_parameters,
            "integrity": dict(
                require_mapping(
                    config,
                    "integrity",
                )
            ),
        }

        writer = RunEvidenceWriter.create(
            evidence_dir,
            run_id=run_id,
            manifest=manifest,
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        sample_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        writer.append_event(
            "run_start",
            {
                "start_epoch": start_epoch,
                "requested_end_epoch": requested_end_epoch,
                "global_step": global_step,
            },
        )

    trainer = CycleGANTrainer(
        networks=networks,
        optimizers=optimizers,
        losses=losses,
        fake_a_pool=fake_a_pool,
        fake_b_pool=fake_b_pool,
        mixed_precision=bool(training["mixed_precision"]),
    )

    writer.append_event('precision', {
        'requested_bf16': bool(training['mixed_precision']),
        'actual_bf16': trainer.amp_enabled, 'parameter_dtype': 'float32'})
    print('Actual BF16 autocast:', trainer.amp_enabled, flush=True)
    if args.resume:
        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if start_epoch is None:

        writer.append_event(
            "run_already_complete",
            {
                "configured_total_epochs": total_epochs,
                "global_step": global_step,
            },
        )

        print(
            "Run already completed all configured epochs."
        )

        return

    if args.mode == "smoke":

        end_epoch = min(
            total_epochs,
            start_epoch
            + args.smoke_epochs
            - 1,
        )

    else:
        end_epoch = total_epochs

    if end_epoch < start_epoch:
        raise RuntimeError(
            "Resolved end_epoch is earlier than start_epoch"
        )

    log_every_steps = int(
        monitoring[
            "log_every_steps"
        ]
    )

    sample_every_epochs = int(
        monitoring[
            "sample_every_epochs"
        ]
    )

    archive_every_epochs = int(
        checkpointing[
            "save_every_epochs"
        ]
    )

    latest_every_epochs = int(
        checkpointing[
            "save_latest_every_epochs"
        ]
    )

    def on_step_complete(
        step_result: dict[str, Any],
    ) -> None:

        if (
            int(
                step_result[
                    "global_step"
                ]
            )
            % log_every_steps
            == 0
        ):

            writer.append_event(
                "step",
                step_result,
            )
            print(f"epoch={step_result['epoch']} step={step_result['global_step']} "
                  f"G={step_result['metrics']['loss_generator_total']:.4f} "
                  f"D_A={step_result['metrics']['loss_discriminator_a']:.4f} "
                  f"D_B={step_result['metrics']['loss_discriminator_b']:.4f}", flush=True)


    def save_checkpoint(
        *,
        path: Path,
        epoch_completed: int,
        current_global_step: int,
        checkpoint_kind: str,
    ) -> None:

        save_training_checkpoint(
            path,
            networks=trainer.networks,
            optimizers=trainer.optimizers,
            fake_a_pool=trainer.fake_a_pool,
            fake_b_pool=trainer.fake_b_pool,
            epoch_completed=epoch_completed,
            global_step=current_global_step,
            config_sha256=config_sha256,
            metadata={
                "run_id": run_id,
                "mode": args.mode,
                "checkpoint_kind": checkpoint_kind,
            },
        )

        digest = verify_training_checkpoint(
            path
        )

        writer.append_event(
            "checkpoint",
            {
                "epoch_completed": epoch_completed,
                "global_step": current_global_step,
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_file": path.name,
                "checkpoint_sha256": digest,
            },
        )

    def on_epoch_complete(
        epoch_result: dict[str, Any],
        callback_trainer: CycleGANTrainer,
    ) -> None:

        if callback_trainer is not trainer:
            raise RuntimeError(
                "Epoch callback received unexpected trainer object"
            )

        epoch = int(
            epoch_result[
                "epoch"
            ]
        )

        current_global_step = int(
            epoch_result[
                "global_step_end"
            ]
        )

        writer.append_event(
            "epoch_end",
            epoch_result,
        )
        print(f"EPOCH {epoch}/{total_epochs} DONE: "
              f"{epoch_result['duration_seconds']:.1f}s "
              f"lr={epoch_result['learning_rate']:.8f} "
              f"images/sec={epoch_result['images_per_second']:.2f}", flush=True)


        should_sample = (
            epoch
            % sample_every_epochs
            == 0
        )

        if (
            args.mode == "smoke"
            and epoch == end_epoch
        ):
            should_sample = True

        if should_sample:

            written = save_monitoring_samples(
                trainer=trainer,
                dataset=monitoring_dataset,
                device=device,
                epoch=epoch,
                sample_dir=sample_dir,
                sample_count=args.sample_count,
            )

            writer.append_event(
                "monitoring_samples",
                {
                    "epoch": epoch,
                    "files": written,
                    "sample_count": len(
                        written
                    ),
                    "source": (
                        "direct_output_of_user_trained_cyclegan"
                    ),
                },
            )

        if (
            epoch
            % archive_every_epochs
            == 0
        ):

            archive_path = (
                checkpoint_dir
                / f"epoch_{epoch:04d}.pt"
            )

            save_checkpoint(
                path=archive_path,
                epoch_completed=epoch,
                current_global_step=current_global_step,
                checkpoint_kind="archive",
            )

        should_save_latest = (
            epoch
            % latest_every_epochs
            == 0
        )

        if args.mode == "smoke":
            should_save_latest = True

        if should_save_latest:

            save_checkpoint(
                path=latest_checkpoint,
                epoch_completed=epoch,
                current_global_step=current_global_step,
                checkpoint_kind="latest",
            )

    print(
        "======================================================================"
    )

    print(
        "CYCLEGAN TRAINING RUN"
    )

    print(
        "Run ID:",
        run_id,
    )

    print(
        "Mode:",
        args.mode,
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                device
            ),
        )

    print(
        "Config SHA256:",
        config_sha256,
    )

    print(
        "Domain A images:",
        len(
            domain_a_images
        ),
    )

    print(
        "Domain B images:",
        len(
            domain_b_images
        ),
    )

    print(
        "Trainable parameters:",
        total_parameters,
    )

    print(
        "Epoch range:",
        start_epoch,
        "->",
        end_epoch,
    )

    print(
        "Starting global step:",
        global_step,
    )

    print(
        "======================================================================"
    )

    try:

        results = run_epoch_range(
            trainer=trainer,
            loader_factory=loader_factory,
            device=device,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            global_step=global_step,
            base_seed=int(
                experiment[
                    "seed"
                ]
            ),
            base_lr=float(
                training[
                    "learning_rate"
                ]
            ),
            constant_lr_epochs=int(
                training[
                    "constant_lr_epochs"
                ]
            ),
            linear_decay_epochs=int(
                training[
                    "linear_decay_epochs"
                ]
            ),
            on_step_complete=on_step_complete,
            on_epoch_complete=on_epoch_complete,
        )

    except BaseException as exc:

        try:
            writer.append_event(
                "run_failure",
                {
                    "exception_type": type(
                        exc
                    ).__name__,
                    "message": str(
                        exc
                    ),
                },
            )

        except Exception:
            pass

        raise

    if not results:
        raise RuntimeError(
            "Training returned no epoch results"
        )

    final_result = results[
        -1
    ]

    final_global_step = int(
        final_result[
            "global_step_end"
        ]
    )

    writer.append_event(
        "run_complete",
        {
            "first_epoch": int(
                results[
                    0
                ][
                    "epoch"
                ]
            ),
            "last_epoch": int(
                final_result[
                    "epoch"
                ]
            ),
            "epochs_executed": len(
                results
            ),
            "global_step": final_global_step,
            "mode": args.mode,
        },
    )

    print(
        "======================================================================"
    )

    print(
        "RUN COMPLETE"
    )

    print(
        "Last completed epoch:",
        final_result[
            "epoch"
        ],
    )

    print(
        "Final global step:",
        final_global_step,
    )

    print(
        "Latest checkpoint:",
        latest_checkpoint,
    )

    print(
        "Evidence directory:",
        evidence_dir,
    )

    print(
        "Monitoring output directory:",
        sample_dir,
    )

    print(
        "======================================================================"
    )


if __name__ == "__main__":
    main()
