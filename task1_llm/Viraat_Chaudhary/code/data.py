"""Character preprocessing using Team 15's existing sequence-count protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from common import event, member_root, raw_log, read_config, sha256_file, write_json


class CharacterVocabulary:
    def __init__(self, characters: list[str]):
        if not characters or characters[0] != "<UNK>" or len(set(characters)) != len(characters):
            raise ValueError("Vocabulary must contain unique symbols with <UNK> at index zero.")
        self.idx_to_char = dict(enumerate(characters))
        self.char_to_idx = {character: index for index, character in self.idx_to_char.items()}

    @classmethod
    def from_text(cls, training_text: str):
        return cls(["<UNK>", *sorted(set(training_text))])

    def __len__(self):
        return len(self.idx_to_char)

    def encode(self, text: str) -> np.ndarray:
        return np.fromiter(
            (self.char_to_idx.get(c, 0) for c in text), dtype=np.int64, count=len(text)
        )

    def decode(self, ids) -> str:
        return "".join(self.idx_to_char[int(i)] for i in ids)

    def save(self, path: Path):
        write_json(
            path,
            {
                "unknown_token": "<UNK>",
                "vocab_size": len(self),
                "char_to_idx": self.char_to_idx,
                "idx_to_char": self.idx_to_char,
            },
        )

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls([value["idx_to_char"][str(i)] for i in range(value["vocab_size"])])


def collect_text(dataset, start: int, needed: int, field: str, separator: str):
    """Consume complete story records before assigning subsequent records to validation."""
    pieces, size, stories = [], 0, 0
    position = start
    while size < needed:
        if position >= len(dataset):
            raise ValueError("Insufficient stories for the configured sequence counts.")
        text = dataset[position][field]
        position += 1
        if not isinstance(text, str):
            raise ValueError("A dataset story is not text.")
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            continue
        piece = (separator if pieces else "") + text
        pieces.append(piece)
        size += len(piece)
        stories += 1
    return "".join(pieces)[:needed], position, stories


def make_rows(text: str, count: int, context: int, vocabulary: CharacterVocabulary):
    needed = count * (context + 1)
    if len(text) < needed:
        raise ValueError("Not enough characters for input-target rows.")
    if len(vocabulary) > np.iinfo(np.uint16).max:
        raise ValueError("Vocabulary exceeds uint16 storage capacity.")
    return vocabulary.encode(text[:needed]).astype(np.uint16).reshape(count, context + 1)


class CharacterSequences(Dataset):
    def __init__(self, rows):
        if rows.ndim != 2 or rows.shape[1] < 2:
            raise ValueError("Expected rows of shape [examples, context + 1].")
        self.rows = rows

    @classmethod
    def load(cls, path: Path):
        # About 56 MiB total for the prescribed uint16 rows: load once into RAM,
        # avoiding random per-example reads from Colab's Drive filesystem.
        return cls(np.load(path, allow_pickle=False))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # Copy only one compact row; embedding/cross-entropy indices must be int64.
        row = torch.from_numpy(np.asarray(self.rows[index], dtype=np.int64).copy())
        return row[:-1], row[1:]


def verify_processed(root: Path, config: dict) -> dict:
    manifest = json.loads(
        (root / "outputs/metrics/split_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["data_configuration"] != config["data"]:
        raise ValueError("Existing preprocessing does not match the data configuration.")
    for name, digest in manifest["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Preprocessing hash mismatch: {name}")
    return manifest


def prepare(root: Path, config: dict, source=None) -> dict:
    artifacts = [
        root / "data_processed/train.npy",
        root / "data_processed/validation.npy",
        root / "outputs/metrics/vocabulary.json",
        root / "outputs/metrics/split_manifest.json",
    ]
    frozen_manifest = None
    if all(p.exists() for p in artifacts):
        manifest = verify_processed(root, config)
        event("PREPROCESSING_REUSED", manifest_sha256=sha256_file(artifacts[-1]))
        return manifest
    if any(p.exists() for p in artifacts):
        # A clone/export retains the vocabulary and manifest but omits large
        # processed arrays. Rebuild only missing arrays against frozen hashes.
        if not (artifacts[2].exists() and artifacts[3].exists()):
            raise ValueError(
                "Incomplete preprocessing outputs; archive the incomplete run before rebuilding."
            )
        frozen_manifest = json.loads(artifacts[3].read_text(encoding="utf-8"))
        if frozen_manifest["data_configuration"] != config["data"]:
            raise ValueError("Existing preprocessing does not match the data configuration.")
        for name, digest in frozen_manifest["files"].items():
            path = root / name
            if path.exists() and sha256_file(path) != digest:
                raise ValueError(f"Preprocessing hash mismatch: {name}")
    data = config["data"]
    if source is None:
        from datasets import load_dataset

        source = load_dataset(
            data["dataset_id"],
            revision=data["dataset_revision"],
            split=data["source_split"],
            cache_dir=os.environ.get("TASK1_HF_CACHE", str(root / "hf_cache")),
        )
        source = source.shuffle(seed=int(data["split_seed"]))
    row_size = int(data["sequence_length"]) + 1
    train_text, next_record, train_stories = collect_text(
        source, 0, data["train_sequences"] * row_size, data["text_field"], data["separator"]
    )
    val_text, end_record, val_stories = collect_text(
        source,
        next_record,
        data["validation_sequences"] * row_size,
        data["text_field"],
        data["separator"],
    )
    vocabulary = CharacterVocabulary.from_text(train_text)
    train_rows = make_rows(train_text, data["train_sequences"], data["sequence_length"], vocabulary)
    val_rows = make_rows(
        val_text, data["validation_sequences"], data["sequence_length"], vocabulary
    )
    manifest = {
        "member_name": config["experiment"]["member_name"],
        "data_configuration": data,
        "protocol": "100K/10K fixed-length sequences, following the existing Team 15 implementation; counts are not story counts",
        "source_fingerprint": getattr(source, "_fingerprint", "synthetic-smoke-fixture"),
        "train_shuffled_index_range": [0, next_record],
        "validation_shuffled_index_range": [next_record, end_record],
        "story_record_overlap": False,
        "train_stories_used": train_stories,
        "validation_stories_used": val_stories,
        "train_shape": list(train_rows.shape),
        "validation_shape": list(val_rows.shape),
        "vocabulary_size": len(vocabulary),
        "unknown_validation_characters": sorted(set(val_text) - vocabulary.char_to_idx.keys()),
        "unknown_validation_character_count": sum(
            c not in vocabulary.char_to_idx for c in val_text
        ),
        "train_text_sha256": hashlib.sha256(train_text.encode()).hexdigest(),
        "validation_text_sha256": hashlib.sha256(val_text.encode()).hexdigest(),
    }
    root.mkdir(parents=True, exist_ok=True)
    # Validate all candidate artifacts before publishing any restored file.
    with tempfile.TemporaryDirectory(prefix="task1_preprocessing_", dir=root) as directory:
        stage = Path(directory)
        candidate_paths = [stage / p.relative_to(root) for p in artifacts[:3]]
        candidate_paths[0].parent.mkdir(parents=True, exist_ok=True)
        np.save(candidate_paths[0], train_rows, allow_pickle=False)
        np.save(candidate_paths[1], val_rows, allow_pickle=False)
        vocabulary.save(candidate_paths[2])
        manifest["files"] = {
            p.relative_to(stage).as_posix(): sha256_file(p) for p in candidate_paths
        }
        if frozen_manifest is not None:
            # Dataset library fingerprints can vary across versions. The
            # source revision, selected text, shapes, vocabulary and all file
            # hashes must agree; retain the observed original fingerprint.
            def comparable(value):
                return {
                    key: item for key, item in value.items() if key != "source_fingerprint"
                }
            if comparable(manifest) != comparable(frozen_manifest):
                raise ValueError("Rebuilt preprocessing differs from the frozen manifest.")
            for target, candidate in zip(artifacts[:2], candidate_paths[:2]):
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    candidate.replace(target)
            manifest = verify_processed(root, config)
            event("PREPROCESSING_RESTORED", manifest_sha256=sha256_file(artifacts[-1]))
            return manifest
        for target, candidate in zip(artifacts[:3], candidate_paths):
            target.parent.mkdir(parents=True, exist_ok=True)
            candidate.replace(target)
        write_json(artifacts[-1], manifest)
    event(
        "PREPROCESSING_COMPLETE",
        train_sequences=len(train_rows),
        validation_sequences=len(val_rows),
        vocabulary_size=len(vocabulary),
        train_stories=train_stories,
        validation_stories=val_stories,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=member_root() / "configs/gpt_char.yaml")
    args = parser.parse_args()
    root = args.config.resolve().parents[1]
    with raw_log(root, "preprocessing"):
        prepare(root, read_config(args.config))


if __name__ == "__main__":
    main()
