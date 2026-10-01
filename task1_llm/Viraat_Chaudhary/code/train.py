"""Train ten complete epochs, retaining raw logs and resumable epoch checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from common import (
    atomic_checkpoint,
    environment,
    event,
    member_root,
    memory_reader,
    raw_log,
    read_config,
    sha256_file,
    synchronize,
    write_json,
)
from data import CharacterSequences, CharacterVocabulary, verify_processed
from model import CharGPT, DecoderConfig


def learning_rate(step: int, total: int, warmup: int, maximum: float, minimum: float):
    if step < warmup:
        return maximum * (step + 1) / warmup
    fraction = min(1.0, (step - warmup + 1) / max(1, total - warmup))
    return minimum + 0.5 * (maximum - minimum) * (1 + math.cos(math.pi * fraction))


def loader(dataset, batch_size, seed=None, workers=0, cuda=False):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=seed is not None,
        generator=generator,
        num_workers=workers,
        pin_memory=cuda,
        drop_last=False,
    )


def precision_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(
        device_type=device.type, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    )


def choose_precision(device, configured):
    if device.type != "cuda":
        return "fp32"
    if configured == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if configured == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Configured bfloat16 requires a compatible GPU.")
    return configured


@torch.inference_mode()
def score(model, batches, device, precision="fp32"):
    """Token-weighted loss/accuracy; inference mode removes training dropout."""
    model.eval()
    loss_sum, correct, tokens = 0.0, 0, 0
    synchronize(device)
    start = time.perf_counter()
    for inputs, targets in batches:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
        with precision_context(device, precision):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.float().reshape(-1, model.config.vocab_size),
                targets.reshape(-1),
                reduction="sum",
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite evaluation loss.")
        loss_sum += loss.item()
        correct += (logits.argmax(-1) == targets).sum().item()
        tokens += targets.numel()
    synchronize(device)
    seconds = time.perf_counter() - start
    if tokens == 0:
        raise ValueError("Evaluation processed no target characters.")
    return {
        "loss": loss_sum / tokens,
        "accuracy": correct / tokens,
        "tokens": tokens,
        "seconds": seconds,
        "tokens_per_second": tokens / max(seconds, 1e-12),
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(history, path):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "task1_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(
        [r["epoch"] for r in history],
        [r["training_loss"] for r in history],
        marker="o",
        label="Epoch training loss (dropout active)",
    )
    axis.plot(
        [r["epoch"] for r in history],
        [r["validation_loss"] for r in history],
        marker="o",
        label="Validation loss (dropout disabled)",
    )
    axis.set(
        xlabel="Completed epoch",
        ylabel="Cross-entropy (nats/character)",
        title="Viraat: post-norm character decoder",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train(
    root: Path, config: dict, config_hash: str, resume=False, smoke=False, stop_after_epoch=None
):
    if stop_after_epoch is not None and not smoke:
        raise ValueError("Early stopping here is permitted only for the synthetic resume check.")
    if not smoke and (
        config["data"]["train_sequences"] != 100000
        or config["data"]["validation_sequences"] != 10000
        or config["training"]["epochs"] < 10
    ):
        raise ValueError("Reduced runs cannot be published as the assessed Team 15 training.")
    manifest = verify_processed(root, config)
    vocabulary_path = root / "outputs/metrics/vocabulary.json"
    vocabulary = CharacterVocabulary.load(vocabulary_path)
    train_data = CharacterSequences.load(root / "data_processed/train.npy")
    val_data = CharacterSequences.load(root / "data_processed/validation.npy")
    if (
        len(train_data) != config["data"]["train_sequences"]
        or len(val_data) != config["data"]["validation_sequences"]
    ):
        raise ValueError("Processed sequence counts do not match the configuration.")
    settings = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if settings["require_cuda"] and device.type != "cuda" and not smoke:
        raise RuntimeError("Connect a Colab GPU before the assessed full run.")
    precision = choose_precision(device, settings["precision"])
    torch.manual_seed(settings["training_seed"])
    np.random.seed(settings["training_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings["training_seed"])
        torch.backends.cudnn.benchmark = False
    model = CharGPT(DecoderConfig.from_experiment(config, len(vocabulary))).to(device)
    groups = [
        {
            "params": [p for p in model.parameters() if p.ndim >= 2],
            "weight_decay": settings["weight_decay"],
        },
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        groups, lr=settings["learning_rate"], betas=(settings["adam_beta1"], settings["adam_beta2"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    epochs = 2 if smoke else settings["epochs"]
    batches_per_epoch = math.ceil(len(train_data) / settings["batch_size"])
    total_steps = batches_per_epoch * epochs
    warmup = max(1, int(total_steps * settings["warmup_ratio"]))
    last_path, best_path = (
        root / "checkpoints/last_checkpoint.pt",
        root / "checkpoints/best_model.pt",
    )
    state = {
        "history": [],
        "steps": [],
        "updates": 0,
        "training_tokens": 0,
        "training_seconds": 0.0,
        "total_training_seconds": 0.0,
        "gradient_sum": 0.0,
        "gradient_count": 0,
        "gradient_max": 0.0,
        "loss_spike_count": 0,
        "nonfinite_loss_count": 0,
        "nonfinite_gradient_count": 0,
        "recent_losses": [],
        "peak_cpu_rss_mib": 0.0,
        "peak_gpu_allocated_mib": 0.0,
        "peak_gpu_reserved_mib": 0.0,
    }
    best_loss, start_epoch = math.inf, 1
    if resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if checkpoint["config_sha256"] != config_hash or checkpoint[
            "split_manifest_sha256"
        ] != sha256_file(root / "outputs/metrics/split_manifest.json"):
            raise ValueError("Resume configuration or preprocessing differs from the checkpoint.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        state, best_loss = checkpoint["state"], checkpoint["best_validation_loss"]
        start_epoch = checkpoint["epoch"] + 1
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        if start_epoch > epochs:
            event("ALREADY_COMPLETE", completed_epochs=len(state["history"]))
            return json.loads((root / "outputs/metrics/training_summary.json").read_text())
    elif last_path.exists() or (root / "outputs/metrics/training_summary.json").exists():
        raise FileExistsError(
            "Training artifacts already exist. Use --resume or archive the run before retraining."
        )
    write_json(root / "outputs/metrics/run_configuration.json", config)
    hardware = environment()
    # A separate snapshot preserves hardware provenance for each training/resume session.
    hardware_path = root / "outputs/metrics" / f"environment_epoch_{start_epoch:02d}.json"
    write_json(hardware_path, hardware)
    event(
        "TRAINING_CONFIGURATION",
        smoke_test=smoke,
        device=device.type,
        gpu=hardware["gpu"],
        precision=precision,
        epochs=epochs,
        start_epoch=start_epoch,
        train_sequences=len(train_data),
        validation_sequences=len(val_data),
        parameters=model.parameter_count(),
        updates_per_epoch=batches_per_epoch,
        total_planned_updates=total_steps,
    )
    validation_loader = loader(
        val_data,
        settings["batch_size"],
        workers=settings["num_workers"],
        cuda=device.type == "cuda",
    )
    read_memory = memory_reader()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, epochs + 1):
        synchronize(device)
        epoch_wall_start = time.perf_counter()
        epoch_training_start = epoch_wall_start
        training_loader = loader(
            train_data,
            settings["batch_size"],
            settings["training_seed"] + epoch,
            settings["num_workers"],
            device.type == "cuda",
        )
        model.train()
        loss_sum, epoch_tokens, epoch_examples = 0.0, 0, 0
        for batch_index, (inputs, targets) in enumerate(training_loader, 1):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )
            rate = learning_rate(
                state["updates"],
                total_steps,
                warmup,
                settings["learning_rate"],
                settings["minimum_learning_rate"],
            )
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            with precision_context(device, precision):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, len(vocabulary)), targets.reshape(-1)
                )
            value = loss.item()
            if not math.isfinite(value):
                state["nonfinite_loss_count"] += 1
                event("NONFINITE_LOSS", epoch=epoch, batch=batch_index)
                write_json(root / "outputs/metrics/interrupted_stability.json", state)
                raise FloatingPointError(
                    "Nonfinite loss; raw logs retained. Investigate before resuming."
                )
            recent = state["recent_losses"]
            if len(recent) == 20 and value > 1.5 * statistics.median(recent):
                state["loss_spike_count"] += 1
                event("LOSS_SPIKE", epoch=epoch, batch=batch_index, loss=value)
            state["recent_losses"] = (recent + [value])[-20:]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["gradient_clip_norm"]
            )
            norm_value = norm.item()
            if not math.isfinite(norm_value):
                state["nonfinite_gradient_count"] += 1
                event("NONFINITE_GRADIENT", epoch=epoch, batch=batch_index)
                write_json(root / "outputs/metrics/interrupted_stability.json", state)
                raise FloatingPointError(
                    "Nonfinite gradient; raw logs retained. Investigate before resuming."
                )
            scaler.step(optimizer)
            scaler.update()
            state["updates"] += 1
            state["gradient_sum"] += norm_value
            state["gradient_count"] += 1
            state["gradient_max"] = max(state["gradient_max"], norm_value)
            tokens = targets.numel()
            loss_sum += value * tokens
            epoch_tokens += tokens
            epoch_examples += inputs.shape[0]
            state["peak_cpu_rss_mib"] = max(state["peak_cpu_rss_mib"], read_memory())
            record = {
                "epoch": epoch,
                "update": state["updates"],
                "batch": batch_index,
                "loss": value,
                "learning_rate": rate,
                "gradient_norm_before_clipping": norm_value,
            }
            state["steps"].append(record)
            if (
                batch_index == 1
                or batch_index == batches_per_epoch
                or batch_index % settings["log_every_steps"] == 0
            ):
                event("STEP", **record)
        synchronize(device)
        training_seconds = time.perf_counter() - epoch_training_start
        if epoch_examples != len(train_data):
            raise RuntimeError("Epoch did not visit every training example exactly once.")
        state["training_tokens"] += epoch_tokens
        state["training_seconds"] += training_seconds
        validation = score(model, validation_loader, device, precision)
        entry = {
            "epoch": epoch,
            "training_loss": loss_sum / epoch_tokens,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "training_examples": epoch_examples,
            "training_tokens": epoch_tokens,
            "training_seconds": training_seconds,
            "validation_seconds": validation["seconds"],
            "training_tokens_per_second": epoch_tokens / training_seconds,
            "updates": state["updates"],
        }
        state["history"].append(entry)
        event("EPOCH_COMPLETE", **entry)
        provenance = {
            "model_config": model.configuration(),
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config_sha256": config_hash,
            "vocabulary_sha256": sha256_file(vocabulary_path),
            "split_manifest_sha256": sha256_file(root / "outputs/metrics/split_manifest.json"),
            "experiment_config": config,
            "smoke_test": smoke,
        }
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            atomic_checkpoint({**provenance, "validation_loss": best_loss}, best_path)
        write_csv(root / "outputs/metrics/training_history.csv", state["history"])
        write_csv(root / "outputs/metrics/step_metrics.csv", state["steps"])
        write_json(root / "outputs/metrics/training_history.json", state["history"])
        plot_history(state["history"], root / "outputs/plots/loss_curves.png")
        if device.type == "cuda":
            state["peak_gpu_allocated_mib"] = max(
                state["peak_gpu_allocated_mib"], torch.cuda.max_memory_allocated() / 1024**2
            )
            state["peak_gpu_reserved_mib"] = max(
                state["peak_gpu_reserved_mib"], torch.cuda.max_memory_reserved() / 1024**2
            )
        synchronize(device)
        state["total_training_seconds"] += time.perf_counter() - epoch_wall_start
        atomic_checkpoint(
            {
                **provenance,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "state": state,
                "best_validation_loss": best_loss,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
            last_path,
        )
        # Persist summary each epoch so a completed checkpoint remains usable after interruption.
        summary = {
            "member_name": config["experiment"]["member_name"],
            "experiment": config["experiment"]["name"],
            "smoke_test": smoke,
            "epochs_completed": len(state["history"]),
            "required_epochs": epochs,
            "completed": len(state["history"]) == epochs,
            "global_steps": state["updates"],
            "parameter_count": model.parameter_count(),
            "gpu": hardware["gpu"],
            "precision": precision,
            "final_training_loss": entry["training_loss"],
            "final_validation_loss": entry["validation_loss"],
            "best_validation_loss": best_loss,
            "training_tokens": state["training_tokens"],
            "training_tokens_per_second": state["training_tokens"] / state["training_seconds"],
            "total_training_seconds": state["total_training_seconds"],
            "time_definition": "Sum of completed epoch training, validation, best-checkpoint and plot/metric writes; excludes preprocessing, setup, last-checkpoint write and uncheckpointed interrupted work",
            "peak_gpu_allocated_mb": state["peak_gpu_allocated_mib"],
            "peak_gpu_reserved_mb": state["peak_gpu_reserved_mib"],
            "peak_cpu_rss_mb": state["peak_cpu_rss_mib"],
            "gradient_norm_mean": state["gradient_sum"] / state["gradient_count"],
            "gradient_norm_max": state["gradient_max"],
            "loss_spike_count": state["loss_spike_count"],
            "nonfinite_loss_count": state["nonfinite_loss_count"],
            "nonfinite_gradient_count": state["nonfinite_gradient_count"],
            "config_sha256": config_hash,
            "best_checkpoint_sha256": sha256_file(best_path),
            "last_checkpoint_sha256": sha256_file(last_path),
        }
        write_json(root / "outputs/metrics/training_summary.json", summary)
        if stop_after_epoch == epoch:
            event("SYNTHETIC_RESUME_CHECK_PAUSED", epoch=epoch)
            return summary
    event(
        "TRAINING_COMPLETE",
        epochs_completed=summary["epochs_completed"],
        training_tokens=summary["training_tokens"],
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=member_root() / "configs/gpt_char.yaml")
    parser.add_argument(
        "--resume", action="store_true", help="Continue from the last completed epoch."
    )
    args = parser.parse_args()
    root = args.config.resolve().parents[1]
    with raw_log(root, "training_resume" if args.resume else "training"):
        train(root, read_config(args.config), sha256_file(args.config), args.resume)


if __name__ == "__main__":
    main()
