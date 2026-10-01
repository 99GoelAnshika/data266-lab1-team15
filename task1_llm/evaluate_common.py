"""Evaluate both trained decoders on the same official TinyStories validation text."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import yaml


def import_model(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def verify_vocabulary_hash(checkpoint, vocabulary_path):
    """Match frozen bytes, allowing only an LF/CRLF checkout conversion."""
    if checkpoint.get("smoke_test"):
        raise ValueError("A synthetic smoke-test checkpoint cannot enter the common evaluation.")
    expected = checkpoint.get("vocabulary_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("The checkpoint must record its frozen vocabulary SHA256.")
    raw = vocabulary_path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    candidates = {
        "exact_bytes": hashlib.sha256(raw).hexdigest(),
        "lf_line_endings": hashlib.sha256(lf).hexdigest(),
        "crlf_line_endings": hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest(),
    }
    for representation, digest in candidates.items():
        if digest == expected:
            return {
                "checkpoint_vocabulary_sha256": expected,
                "vocabulary_file_sha256": candidates["exact_bytes"],
                "hash_match_representation": representation,
            }
    raise ValueError(
        "Checkpoint vocabulary mismatch beyond LF/CRLF line endings: "
        f"expected={expected}; observed={candidates}"
    )


def run(task_root, partner_root=None):
    viraat = task_root / "Viraat_Chaudhary"
    anshika = partner_root or task_root / "Anshika_Goel"
    sys.path.insert(0, str(viraat / "code"))
    from common import environment, sha256_file, write_json
    from data import CharacterSequences, CharacterVocabulary, collect_text
    from train import loader, score

    members = [("Anshika_Goel", anshika), ("Viraat_Chaudhary", viraat)]
    configs = [
        yaml.safe_load((folder / "configs/gpt_char.yaml").read_text()) for _, folder in members
    ]
    for c in configs:
        if c["data"]["source_split"] != "train":
            raise ValueError(
                "This common holdout requires both training corpora to come only from the official training split."
            )
    if configs[0]["data"]["dataset_revision"] != configs[1]["data"]["dataset_revision"]:
        raise ValueError("Both models must reference the same frozen dataset revision.")
    if any(c["data"]["sequence_length"] != 256 for c in configs):
        raise ValueError("This Team 15 benchmark is fixed at 256-character context.")
    # Validate frozen inputs before any dataset download or evaluation.
    frozen_inputs = []
    for name, folder in members:
        checkpoint_path = folder / "checkpoints/best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"A trained best checkpoint is required for {folder.name}.")
        vocabulary_path = folder / "outputs/metrics/vocabulary.json"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        verification = verify_vocabulary_hash(checkpoint, vocabulary_path)
        vocabulary = CharacterVocabulary.load(vocabulary_path)
        if len(vocabulary) != checkpoint["model_config"]["vocab_size"]:
            raise ValueError(f"Checkpoint vocabulary size differs for {name}.")
        frozen_inputs.append((checkpoint, vocabulary, checkpoint_path, verification))
        print(json.dumps({"event": "FROZEN_VOCABULARY_VERIFIED", "member": name, **verification}))
    from datasets import load_dataset

    cache = task_root / "data/processed/common_evaluation"
    cache.mkdir(parents=True, exist_ok=True)
    c = configs[0]["data"]
    source = load_dataset(
        c["dataset_id"],
        revision=c["dataset_revision"],
        split="validation",
        cache_dir=os.environ.get("TASK1_HF_CACHE", str(cache / "hf_cache")),
    )
    # Independent official split; no tuning or checkpoint selection on this evaluation.
    text, end, stories = collect_text(source, 0, 10000 * 257, c["text_field"], "\n\n")
    (cache / "common_text.txt").write_text(text, encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    for index, (name, folder) in enumerate(members):
        checkpoint, vocabulary, checkpoint_path, verification = frozen_inputs[index]
        module = import_model(folder / "code/model.py", f"common_task1_model_{index}")
        if name == "Anshika_Goel":
            model = module.CharacterGPT(module.GPTConfig(**checkpoint["model_config"]))
        else:
            model = module.CharGPT(module.DecoderConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        # Adapt the original dict output without changing either member's source.
        if name == "Anshika_Goel":

            class LogitAdapter(torch.nn.Module):
                def __init__(self, wrapped):
                    super().__init__()
                    self.wrapped, self.config = wrapped, wrapped.config

                def forward(self, inputs):
                    return self.wrapped(inputs)["logits"]

            model = LogitAdapter(model).to(device).eval()
        ids = vocabulary.encode(text)
        rows = ids.reshape(10000, 257)
        result = score(
            model, loader(CharacterSequences(rows), 32, cuda=device.type == "cuda"), device, "fp32"
        )
        records.append(
            {
                "member": name,
                "common_validation_ce": result["loss"],
                "common_next_character_accuracy": result["accuracy"],
                "unknown_character_count": int((ids == 0).sum()),
                "checkpoint_epoch": checkpoint["epoch"],
                "checkpoint_sha256": sha256_file(checkpoint_path),
                **verification,
            }
        )
        print(json.dumps(records[-1]))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = task_root / "comparison"
    output.mkdir(exist_ok=True)
    with (output / "common_validation_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(
        output / "common_evaluation_manifest.json",
        {
            "dataset_id": c["dataset_id"],
            "revision": c["dataset_revision"],
            "official_split": "validation",
            "source_record_range": [0, end],
            "stories_used": stories,
            "sequences": 10000,
            "context_length": 256,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "evaluation_precision": "fp32",
            "evaluation_batch_size": 32,
            "evaluator_sha256": sha256_file(Path(__file__)),
            "vocabulary_hash_policy": "Frozen bytes or exact LF/CRLF line-ending conversion; vocabulary files remain unchanged.",
            "environment": environment(),
            "results": records,
            "selection_boundary": "Evaluate after each member freezes their own best checkpoint; do not tune on this common holdout. Report observed scores and OOV coverage before making a selection.",
        },
    )
    print(
        "Common held-out evaluation saved. Review coverage and the fixed selection criterion before naming the preferred model."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--partner-root", type=Path)
    args = parser.parse_args()
    task_root = args.task_root.resolve()
    sys.path.insert(0, str(task_root / "Viraat_Chaudhary/code"))
    from common import raw_log

    with raw_log(task_root / "Viraat_Chaudhary", "common_evaluation"):
        run(task_root, args.partner_root.resolve() if args.partner_root else None)
