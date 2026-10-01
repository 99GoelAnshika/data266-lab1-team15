"""Small utilities shared by Viraat's Task 1 command-line programs."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
import yaml


def member_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    data, model, training = (config[k] for k in ("data", "model", "training"))
    if model["normalization"] != "post_layer_norm" or model["activation"] != "relu":
        raise ValueError("This implementation is the documented post-norm/ReLU decoder.")
    if model["tie_token_and_output_embeddings"] or model["positional_embeddings"] != "learned":
        raise ValueError("Use independent output weights and learned positional embeddings.")
    if model["d_model"] <= 0 or model["num_heads"] <= 0 or model["d_model"] % model["num_heads"]:
        raise ValueError("Positive model width must be divisible by the number of heads.")
    if (
        min(
            data["train_sequences"],
            data["validation_sequences"],
            data["sequence_length"],
            training["batch_size"],
            model["num_layers"],
        )
        <= 0
    ):
        raise ValueError("Dataset sizes, context, batch size and layer count must be positive.")
    if data["train_sequences"] != 100000 or data["validation_sequences"] != 10000:
        raise ValueError(
            "The assessed Team 15 protocol requires 100K training / 10K validation sequences."
        )
    if training["epochs"] < 10:
        raise ValueError(
            "The assessed run requires at least ten full epochs; use smoke_test.py for checks."
        )
    if not 0 <= model["dropout"] < 1 or not 0 < training["warmup_ratio"] < 1:
        raise ValueError("Invalid dropout or warm-up ratio.")
    if training["scheduler"] != "cosine" or training["precision"] not in {
        "auto",
        "fp32",
        "bf16",
        "fp16",
    }:
        raise ValueError("Unsupported scheduler or precision.")
    if not 0 <= training["minimum_learning_rate"] <= training["learning_rate"]:
        raise ValueError("Invalid learning rate bounds.")
    return config


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()
        return len(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


@contextlib.contextmanager
def raw_log(root: Path, stage: str):
    """Capture stdout/stderr once in a new file; never rewrite a previous log."""
    directory = root / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{stage}_{stamp}.log"
    with path.open("x", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(Tee(sys.stdout, handle)), contextlib.redirect_stderr(
            Tee(sys.stderr, handle)
        ):
            event("RUN_START", stage=stage, created_utc=datetime.now(timezone.utc).isoformat())
            try:
                yield path
            except BaseException:
                traceback.print_exc()
                event("RUN_FAILED", stage=stage)
                raise
            else:
                event("RUN_END", stage=stage)


def event(name: str, **fields) -> None:
    print(json.dumps({"event": name, **fields}, ensure_ascii=False, allow_nan=False), flush=True)


def synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment() -> dict:
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {"name": properties.name, "total_memory_mib": properties.total_memory / 1024**2}
    versions = {}
    for name in ("torch", "datasets", "numpy", "matplotlib", "PyYAML", "psutil", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    driver = None
    if gpu:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
            )
            driver = result.stdout.strip() if result.returncode == 0 else "unavailable"
        except FileNotFoundError:
            driver = "nvidia-smi unavailable"
    return {
        "python": platform.python_version(),
        "os": platform.system(),
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "system_ram_mib": psutil.virtual_memory().total / 1024**2,
        "gpu": gpu,
        "driver": driver,
        "pytorch_cuda": torch.version.cuda,
        "packages": versions,
    }


def memory_reader():
    try:
        process = psutil.Process()
        process.memory_info()
        return lambda: process.memory_info().rss / 1024**2
    except (psutil.Error, OSError):
        # Some isolated runtimes hide the process table while exposing getrusage.
        import resource

        divisor = 1024**2 if platform.system() == "Darwin" else 1024
        return lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def atomic_checkpoint(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
