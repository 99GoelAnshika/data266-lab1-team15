"""One-command synthetic training/reload/evaluation check; never writes assessed outputs."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import torch

from common import member_root, raw_log, read_config
from data import prepare
from evaluate_generate import evaluate
from report import generate_report
from train import train


def main():
    torch.set_num_threads(1)
    config = copy.deepcopy(read_config(member_root() / "configs/gpt_char.yaml"))
    config["data"].update(train_sequences=16, validation_sequences=4, sequence_length=8)
    config["model"].update(d_model=16, num_heads=4, num_layers=2, d_ff=32, dropout=0.0)
    config["training"].update(
        batch_size=4, num_workers=0, require_cuda=False, precision="fp32", log_every_steps=4
    )
    config["generation"].update(prompts=["Once"], max_new_characters=8, samples_per_temperature=1)
    with tempfile.TemporaryDirectory(prefix="task1_smoke_") as directory:
        root = Path(directory)
        (root / "analysis").mkdir()
        (root / "analysis/failure_cases.json").write_text('{"cases": []}\n')
        source = [{"text": ("Once upon a time Lily found a little dog. " * 6)} for _ in range(4)]
        # Retain every new smoke attempt log; only its datasets/checkpoints
        # are temporary and can never replace assessed-run evidence.
        with raw_log(member_root(), "smoke"):
            prepare(root, config, source=source)
            partial = train(
                root,
                config,
                config_hash="synthetic-smoke-configuration",
                smoke=True,
                stop_after_epoch=1,
            )
            assert partial["epochs_completed"] == 1 and not partial["completed"]
            summary = train(
                root, config, config_hash="synthetic-smoke-configuration", resume=True, smoke=True
            )
            assert summary["epochs_completed"] == 2
            assert summary["training_tokens"] == 16 * 8 * 2
            assert train(root, config, "synthetic-smoke-configuration", resume=True, smoke=True)[
                "completed"
            ]
            metrics = evaluate(root, config, smoke=True)
            assert metrics["generation"]["sample_count"] == 4
        generate_report(root)
        assert (root / "metrics_report.csv").exists()
        assert json.loads((root / "artifact_manifest.json").read_text())["checkpoint"]["epoch"] in (
            1,
            2,
        )
    print(
        "SMOKE CHECK PASSED: synthetic preprocessing, training, reload, evaluation, generation, report and hashes."
    )
    print("Only a labelled smoke log was retained; assessed-run artifacts were unchanged.")


if __name__ == "__main__":
    main()
