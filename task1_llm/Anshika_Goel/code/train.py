from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import psutil
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import (
    CharVocabulary,
    SequenceTensorDataset,
    repository_root,
    resolve_project_path,
    sha256_file,
)
from model import CharacterGPT, build_model_from_experiment_config


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def relative_to_repository(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def build_optimizer(
    model: CharacterGPT,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
) -> torch.optim.AdamW:
    decay_parameters: list[torch.nn.Parameter] = []
    no_decay_parameters: list[torch.nn.Parameter] = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        parameter_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
    )


def scheduled_learning_rate(
    optimization_step: int,
    total_optimization_steps: int,
    warmup_steps: int,
    maximum_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    if total_optimization_steps <= 0:
        raise ValueError("Total optimization steps must be positive.")

    if optimization_step < warmup_steps:
        return maximum_learning_rate * (
            (optimization_step + 1) / max(1, warmup_steps)
        )

    decay_steps = max(1, total_optimization_steps - warmup_steps)
    decay_progress = min(
        1.0,
        max(
            0.0,
            (
                optimization_step
                - warmup_steps
                + 1
            )
            / decay_steps,
        ),
    )
    cosine_multiplier = 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )

    return minimum_learning_rate + (
        maximum_learning_rate - minimum_learning_rate
    ) * cosine_multiplier


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def make_train_loader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def make_validation_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(
    model: CharacterGPT,
    data_loader: DataLoader,
    device: torch.device,
    use_mixed_precision: bool,
    maximum_batches: int | None = None,
) -> dict[str, float | int]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for batch_index, (inputs, targets) in enumerate(data_loader):
        if (
            maximum_batches is not None
            and batch_index >= maximum_batches
        ):
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_mixed_precision,
        ):
            outputs = model(inputs, targets)
            loss = outputs["loss"]

        token_count = targets.numel()
        total_loss += float(loss.item()) * token_count
        predictions = outputs["logits"].argmax(dim=-1)
        total_correct += int((predictions == targets).sum().item())
        total_tokens += token_count

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_seconds = time.perf_counter() - start_time

    if total_tokens == 0:
        raise RuntimeError("Validation processed zero tokens.")

    return {
        "loss": total_loss / total_tokens,
        "top1_accuracy": total_correct / total_tokens,
        "tokens": total_tokens,
        "seconds": elapsed_seconds,
        "tokens_per_second": total_tokens / elapsed_seconds,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_curves(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    epochs = [record["epoch"] for record in history]
    training_losses = [
        record["training_loss"]
        for record in history
    ]
    validation_losses = [
        record["validation_loss"]
        for record in history
    ]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        epochs,
        training_losses,
        marker="o",
        label="Training loss",
    )
    axis.plot(
        epochs,
        validation_losses,
        marker="o",
        label="Validation loss",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Character GPT Training and Validation Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def atomic_torch_save(payload: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)


def current_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def create_checkpoint(
    model: CharacterGPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    history: list[dict[str, Any]],
    step_records: list[dict[str, Any]],
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
        "step_records": step_records,
        "experiment_config": experiment_config,
        "rng_state": current_rng_state(),
    }


def train(
    config_path: Path,
    resume_path: Path | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    repo_root = repository_root()
    member_root = config_path.resolve().parent.parent

    with config_path.open("r", encoding="utf-8") as handle:
        experiment_config = yaml.safe_load(handle)

    data_config = experiment_config["data"]
    training_config = experiment_config["training"]

    training_seed = int(training_config["training_seed"])
    set_reproducible_seed(training_seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_mixed_precision = bool(
        training_config["mixed_precision"]
        and device.type == "cuda"
    )

    processed_directory = resolve_project_path(
        repo_root,
        data_config["processed_dir"],
    )
    train_tensor_path = (
        processed_directory / "train_sequences.pt"
    )
    validation_tensor_path = (
        processed_directory / "validation_sequences.pt"
    )
    vocabulary_path = resolve_project_path(
        repo_root,
        data_config["vocabulary_file"],
    )

    required_paths = [
        train_tensor_path,
        validation_tensor_path,
        vocabulary_path,
    ]
    missing_paths = [
        path for path in required_paths if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(
            relative_to_repository(path, repo_root)
            for path in missing_paths
        )
        raise FileNotFoundError(
            f"Missing preprocessing artifacts: {missing}"
        )

    vocabulary = CharVocabulary.load(vocabulary_path)
    train_dataset: Dataset = SequenceTensorDataset.from_file(
        train_tensor_path
    )
    validation_dataset: Dataset = (
        SequenceTensorDataset.from_file(
            validation_tensor_path
        )
    )

    maximum_train_batches = None
    maximum_validation_batches = None

    if smoke_test:
        train_dataset = Subset(
            train_dataset,
            range(min(64, len(train_dataset))),
        )
        validation_dataset = Subset(
            validation_dataset,
            range(min(64, len(validation_dataset))),
        )
        epochs = 1
        maximum_train_batches = 2
        maximum_validation_batches = 2
    else:
        epochs = int(training_config["epochs"])

    batch_size = int(training_config["batch_size"])
    gradient_accumulation_steps = int(
        training_config["gradient_accumulation_steps"]
    )
    num_workers = int(training_config["num_workers"])
    pin_memory = device.type == "cuda"

    validation_loader = make_validation_loader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader_length = math.ceil(
        len(train_dataset) / batch_size
    )
    if maximum_train_batches is not None:
        train_loader_length = min(
            train_loader_length,
            maximum_train_batches,
        )

    updates_per_epoch = math.ceil(
        train_loader_length / gradient_accumulation_steps
    )
    total_optimization_steps = epochs * updates_per_epoch
    warmup_steps = max(
        1,
        int(
            total_optimization_steps
            * float(training_config["warmup_ratio"])
        ),
    )

    model = build_model_from_experiment_config(
        experiment_config,
        vocab_size=vocabulary.size,
    ).to(device)

    optimizer = build_optimizer(
        model=model,
        learning_rate=float(
            training_config["learning_rate"]
        ),
        weight_decay=float(training_config["weight_decay"]),
        beta1=float(training_config["adam_beta1"]),
        beta2=float(training_config["adam_beta2"]),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_mixed_precision,
    )

    checkpoints_directory = member_root / "checkpoints"
    metrics_directory = member_root / "outputs" / "metrics"
    plots_directory = member_root / "outputs" / "plots"

    best_checkpoint_path = (
        checkpoints_directory / "best_model.pt"
    )
    last_checkpoint_path = (
        checkpoints_directory / "last_checkpoint.pt"
    )
    history_csv_path = (
        metrics_directory / "training_history.csv"
    )
    history_json_path = (
        metrics_directory / "training_history.json"
    )
    step_metrics_path = (
        metrics_directory / "step_metrics.csv"
    )
    summary_path = (
        metrics_directory / "training_summary.json"
    )
    loss_plot_path = plots_directory / "loss_curves.png"

    if (
        not smoke_test
        and resume_path is None
        and (
            last_checkpoint_path.exists()
            or summary_path.exists()
        )
    ):
        raise FileExistsError(
            "Training outputs already exist. Resume from the last "
            "checkpoint or archive the existing run before starting "
            "another full run."
        )

    start_epoch = 1
    global_step = 0
    best_validation_loss = math.inf
    history: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []

    if resume_path is not None:
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        if checkpoint["model_config"] != model.config.to_dict():
            raise ValueError(
                "Checkpoint model configuration does not match."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_validation_loss = float(
            checkpoint["best_validation_loss"]
        )
        history = list(checkpoint.get("history", []))
        step_records = list(
            checkpoint.get("step_records", [])
        )
        restore_rng_state(checkpoint["rng_state"])

    if start_epoch > epochs:
        raise ValueError(
            "The checkpoint has already completed all configured epochs."
        )

    run_metadata = {
        "event": "RUN_START",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment_config["experiment"]["name"],
        "smoke_test": smoke_test,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "mixed_precision": use_mixed_precision,
        "parameter_count": model.parameter_count(),
        "vocabulary_size": vocabulary.size,
        "train_sequences": len(train_dataset),
        "validation_sequences": len(validation_dataset),
        "batch_size": batch_size,
        "epochs": epochs,
        "updates_per_epoch": updates_per_epoch,
        "total_optimization_steps": total_optimization_steps,
        "warmup_steps": warmup_steps,
        "config": relative_to_repository(
            config_path,
            repo_root,
        ),
    }
    print(json.dumps(run_metadata, ensure_ascii=False))

    process = psutil.Process()
    peak_cpu_rss_bytes = process.memory_info().rss

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    recent_losses: deque[float] = deque(maxlen=20)
    gradient_norms: list[float] = []
    loss_spike_count = 0
    nonfinite_loss_count = 0
    nonfinite_gradient_count = 0
    maximum_batch_loss = -math.inf
    total_training_tokens = 0
    accumulated_training_seconds = 0.0

    full_training_start = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        train_loader = make_train_loader(
            train_dataset,
            batch_size=batch_size,
            seed=training_seed + epoch,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss_sum = 0.0
        epoch_token_count = 0

        batches_this_epoch = len(train_loader)
        if maximum_train_batches is not None:
            batches_this_epoch = min(
                batches_this_epoch,
                maximum_train_batches,
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_training_start = time.perf_counter()

        for batch_index, (inputs, targets) in enumerate(
            train_loader
        ):
            if batch_index >= batches_this_epoch:
                break

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_mixed_precision,
            ):
                outputs = model(inputs, targets)
                loss = outputs["loss"]

            raw_loss = float(loss.item())
            token_count = targets.numel()

            if not math.isfinite(raw_loss):
                nonfinite_loss_count += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    json.dumps(
                        {
                            "event": "NONFINITE_LOSS",
                            "epoch": epoch,
                            "batch": batch_index + 1,
                        }
                    )
                )
                continue

            maximum_batch_loss = max(
                maximum_batch_loss,
                raw_loss,
            )

            if len(recent_losses) == recent_losses.maxlen:
                rolling_median = statistics.median(
                    recent_losses
                )
                if raw_loss > 1.5 * rolling_median:
                    loss_spike_count += 1

            recent_losses.append(raw_loss)

            epoch_loss_sum += raw_loss * token_count
            epoch_token_count += token_count
            total_training_tokens += token_count

            scaled_loss = (
                loss / gradient_accumulation_steps
            )
            scaler.scale(scaled_loss).backward()

            is_accumulation_boundary = (
                (batch_index + 1)
                % gradient_accumulation_steps
                == 0
            )
            is_last_batch = (
                batch_index + 1 == batches_this_epoch
            )

            if is_accumulation_boundary or is_last_batch:
                learning_rate = scheduled_learning_rate(
                    optimization_step=global_step,
                    total_optimization_steps=(
                        total_optimization_steps
                    ),
                    warmup_steps=warmup_steps,
                    maximum_learning_rate=float(
                        training_config["learning_rate"]
                    ),
                    minimum_learning_rate=float(
                        training_config[
                            "minimum_learning_rate"
                        ]
                    ),
                )
                set_optimizer_learning_rate(
                    optimizer,
                    learning_rate,
                )

                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(
                        training_config[
                            "gradient_clip_norm"
                        ]
                    ),
                )
                gradient_norm_value = float(
                    gradient_norm.item()
                )

                if not math.isfinite(gradient_norm_value):
                    nonfinite_gradient_count += 1
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

                gradient_norms.append(gradient_norm_value)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                log_every_steps = int(
                    training_config["log_every_steps"]
                )

                if (
                    global_step % log_every_steps == 0
                    or global_step == 1
                    or is_last_batch
                ):
                    elapsed = max(
                        time.perf_counter()
                        - epoch_training_start,
                        1.0e-9,
                    )
                    epoch_tokens_per_second = (
                        epoch_token_count / elapsed
                    )
                    peak_cpu_rss_bytes = max(
                        peak_cpu_rss_bytes,
                        process.memory_info().rss,
                    )
                    gpu_allocated_mb = (
                        torch.cuda.memory_allocated()
                        / (1024**2)
                        if device.type == "cuda"
                        else 0.0
                    )

                    step_record = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "batch": batch_index + 1,
                        "loss": raw_loss,
                        "learning_rate": learning_rate,
                        "gradient_norm": (
                            gradient_norm_value
                        ),
                        "tokens_per_second": (
                            epoch_tokens_per_second
                        ),
                        "gpu_allocated_mb": (
                            gpu_allocated_mb
                        ),
                    }
                    step_records.append(step_record)
                    print(
                        json.dumps(
                            {
                                "event": "STEP",
                                **step_record,
                            }
                        )
                    )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_training_seconds = (
            time.perf_counter() - epoch_training_start
        )
        accumulated_training_seconds += (
            epoch_training_seconds
        )

        if epoch_token_count == 0:
            raise RuntimeError(
                "Training processed zero valid tokens."
            )

        training_loss = (
            epoch_loss_sum / epoch_token_count
        )
        training_tokens_per_second = (
            epoch_token_count / epoch_training_seconds
        )

        validation_metrics = evaluate(
            model=model,
            data_loader=validation_loader,
            device=device,
            use_mixed_precision=use_mixed_precision,
            maximum_batches=maximum_validation_batches,
        )

        validation_loss = float(
            validation_metrics["loss"]
        )
        perplexity = math.exp(min(validation_loss, 20.0))
        bits_per_character = validation_loss / math.log(2.0)
        generalization_gap = (
            validation_loss - training_loss
        )

        epoch_record = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "perplexity": perplexity,
            "bits_per_character": bits_per_character,
            "generalization_gap": generalization_gap,
            "validation_top1_accuracy": float(
                validation_metrics["top1_accuracy"]
            ),
            "training_tokens_per_second": (
                training_tokens_per_second
            ),
            "validation_tokens_per_second": float(
                validation_metrics["tokens_per_second"]
            ),
            "epoch_training_seconds": (
                epoch_training_seconds
            ),
            "global_step": global_step,
        }
        history.append(epoch_record)

        peak_cpu_rss_bytes = max(
            peak_cpu_rss_bytes,
            process.memory_info().rss,
        )

        epoch_event = {
            "event": "EPOCH_COMPLETE",
            **epoch_record,
        }
        print(json.dumps(epoch_event))

        if not smoke_test:
            write_csv(history_csv_path, history)
            write_json(history_json_path, history)
            write_csv(step_metrics_path, step_records)
            plot_loss_curves(history, loss_plot_path)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss

                best_payload = {
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "model_config": model.config.to_dict(),
                    "model_state_dict": model.state_dict(),
                    "experiment_config": experiment_config,
                    "vocabulary_sha256": sha256_file(
                        vocabulary_path
                    ),
                }
                atomic_torch_save(
                    best_payload,
                    best_checkpoint_path,
                )

            last_payload = create_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                history=history,
                step_records=step_records,
                experiment_config=experiment_config,
            )
            atomic_torch_save(
                last_payload,
                last_checkpoint_path,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_wall_seconds = (
        time.perf_counter() - full_training_start
    )

    peak_gpu_allocated_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    peak_gpu_reserved_mb = (
        torch.cuda.max_memory_reserved() / (1024**2)
        if device.type == "cuda"
        else 0.0
    )

    final_record = history[-1]
    summary = {
        "experiment": experiment_config["experiment"]["name"],
        "member_name": experiment_config["experiment"][
            "member_name"
        ],
        "smoke_test": smoke_test,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "parameter_count": model.parameter_count(),
        "epochs_completed": len(history),
        "global_steps": global_step,
        "final_training_loss": final_record[
            "training_loss"
        ],
        "final_validation_loss": final_record[
            "validation_loss"
        ],
        "final_perplexity": final_record["perplexity"],
        "final_bits_per_character": final_record[
            "bits_per_character"
        ],
        "final_generalization_gap": final_record[
            "generalization_gap"
        ],
        "final_validation_top1_accuracy": final_record[
            "validation_top1_accuracy"
        ],
        "training_tokens": total_training_tokens,
        "training_tokens_per_second": (
            total_training_tokens
            / max(accumulated_training_seconds, 1.0e-9)
        ),
        "total_training_seconds": total_wall_seconds,
        "peak_gpu_allocated_mb": peak_gpu_allocated_mb,
        "peak_gpu_reserved_mb": peak_gpu_reserved_mb,
        "peak_cpu_rss_mb": peak_cpu_rss_bytes / (1024**2),
        "gradient_norm_mean": (
            statistics.mean(gradient_norms)
            if gradient_norms
            else None
        ),
        "gradient_norm_max": (
            max(gradient_norms)
            if gradient_norms
            else None
        ),
        "loss_spike_count": loss_spike_count,
        "maximum_batch_loss": maximum_batch_loss,
        "nonfinite_loss_count": nonfinite_loss_count,
        "nonfinite_gradient_count": (
            nonfinite_gradient_count
        ),
        "best_validation_loss": (
            min(
                record["validation_loss"]
                for record in history
            )
        ),
        "config_sha256": sha256_file(config_path),
    }

    if not smoke_test:
        summary["best_checkpoint"] = (
            relative_to_repository(
                best_checkpoint_path,
                repo_root,
            )
        )
        summary["last_checkpoint"] = (
            relative_to_repository(
                last_checkpoint_path,
                repo_root,
            )
        )
        write_json(summary_path, summary)

    print(
        json.dumps(
            {
                "event": "TRAINING_COMPLETE",
                **summary,
            }
        )
    )

    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the custom character-level GPT model."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML experiment configuration.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional last-checkpoint path for resuming.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run two train and validation batches without saving.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    train(
        config_path=arguments.config.resolve(),
        resume_path=(
            arguments.resume.resolve()
            if arguments.resume is not None
            else None
        ),
        smoke_test=arguments.smoke_test,
    )


if __name__ == "__main__":
    main()