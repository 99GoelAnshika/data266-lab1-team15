from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_dataset
from torch.utils.data import Dataset


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_project_path(repo_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else repo_root / path


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CharVocabulary:
    def __init__(self, idx_to_char: list[str], unknown_token: str = "<UNK>") -> None:
        if not idx_to_char or idx_to_char[0] != unknown_token:
            raise ValueError("The unknown token must occupy vocabulary index 0.")

        self.idx_to_char = list(idx_to_char)
        self.char_to_idx = {
            character: index
            for index, character in enumerate(self.idx_to_char)
        }
        self.unknown_token = unknown_token
        self.unknown_index = self.char_to_idx[unknown_token]

    @classmethod
    def from_training_text(
        cls,
        training_text: str,
        unknown_token: str = "<UNK>",
    ) -> "CharVocabulary":
        characters = sorted(set(training_text))
        return cls([unknown_token, *characters], unknown_token=unknown_token)

    @property
    def size(self) -> int:
        return len(self.idx_to_char)

    def encode(self, text: str) -> np.ndarray:
        mapping = self.char_to_idx
        unknown_index = self.unknown_index
        return np.fromiter(
            (mapping.get(character, unknown_index) for character in text),
            dtype=np.int32,
            count=len(text),
        )

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        return "".join(
            self.idx_to_char[int(token_id)]
            if 0 <= int(token_id) < self.size
            else self.unknown_token
            for token_id in token_ids
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "unknown_token": self.unknown_token,
            "unknown_index": self.unknown_index,
            "vocab_size": self.size,
            "char_to_idx": self.char_to_idx,
            "idx_to_char": {
                str(index): character
                for index, character in enumerate(self.idx_to_char)
            },
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "CharVocabulary":
        payload = json.loads(path.read_text(encoding="utf-8"))
        ordered_characters = [
            character
            for _, character in sorted(
                payload["idx_to_char"].items(),
                key=lambda item: int(item[0]),
            )
        ]
        return cls(
            ordered_characters,
            unknown_token=payload["unknown_token"],
        )


class SequenceTensorDataset(Dataset):
    def __init__(self, sequence_tensor: torch.Tensor) -> None:
        if sequence_tensor.ndim != 2:
            raise ValueError("Sequence tensor must be two-dimensional.")
        if sequence_tensor.shape[1] < 2:
            raise ValueError("Each row must contain at least two token IDs.")

        self.sequence_tensor = sequence_tensor

    @classmethod
    def from_file(cls, path: Path) -> "SequenceTensorDataset":
        tensor = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        return cls(tensor)

    def __len__(self) -> int:
        return self.sequence_tensor.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.sequence_tensor[index]
        inputs = row[:-1].to(dtype=torch.long)
        targets = row[1:].to(dtype=torch.long)
        return inputs, targets


def normalize_story(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def collect_disjoint_story_text(
    dataset: Any,
    start_index: int,
    required_characters: int,
    text_field: str,
    separator: str,
) -> tuple[str, int, int]:
    pieces: list[str] = []
    character_count = 0
    dataset_index = start_index
    stories_used = 0

    while character_count < required_characters:
        if dataset_index >= len(dataset):
            raise RuntimeError(
                "The dataset ended before enough characters were collected."
            )

        story = normalize_story(dataset[dataset_index][text_field])
        dataset_index += 1

        if not story:
            continue

        piece = story if not pieces else separator + story
        pieces.append(piece)
        character_count += len(piece)
        stories_used += 1

    corpus = "".join(pieces)

    if len(corpus) < required_characters:
        raise RuntimeError("Collected corpus is unexpectedly too short.")

    return corpus[:required_characters], dataset_index, stories_used


def build_sequence_tensor(
    text: str,
    number_of_sequences: int,
    sequence_length: int,
    vocabulary: CharVocabulary,
) -> torch.Tensor:
    row_length = sequence_length + 1
    required_characters = number_of_sequences * row_length

    if len(text) < required_characters:
        raise ValueError(
            f"Need {required_characters:,} characters but received "
            f"{len(text):,}."
        )

    encoded = vocabulary.encode(text[:required_characters])
    tensor = torch.from_numpy(encoded.copy())
    return tensor.reshape(number_of_sequences, row_length)


def prepare_data(config_path: Path, force: bool = False) -> dict[str, Any]:
    repo_root = repository_root()
    config = load_yaml_config(config_path)
    data_config = config["data"]

    processed_directory = resolve_project_path(
        repo_root,
        data_config["processed_dir"],
    )
    vocabulary_path = resolve_project_path(
        repo_root,
        data_config["vocabulary_file"],
    )
    manifest_path = resolve_project_path(
        repo_root,
        data_config["split_manifest_file"],
    )

    train_tensor_path = processed_directory / "train_sequences.pt"
    validation_tensor_path = (
        processed_directory / "validation_sequences.pt"
    )

    output_paths = [
        train_tensor_path,
        validation_tensor_path,
        vocabulary_path,
        manifest_path,
    ]

    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs and not force:
        existing_names = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            "Preprocessed outputs already exist. Use --force only when "
            f"you intentionally want to replace them: {existing_names}"
        )

    cache_override = os.environ.get("HF_DATASETS_CACHE")
    if cache_override:
        cache_directory = Path(cache_override)
    else:
        cache_directory = resolve_project_path(
            repo_root,
            data_config["cache_dir"],
        )

    cache_directory.mkdir(parents=True, exist_ok=True)
    processed_directory.mkdir(parents=True, exist_ok=True)

    source_dataset = load_dataset(
        data_config["dataset_id"],
        split=data_config["source_split"],
        revision=data_config["dataset_revision"],
        cache_dir=str(cache_directory),
    )

    shuffled_dataset = source_dataset.shuffle(
        seed=int(data_config["split_seed"])
    )

    sequence_length = int(data_config["sequence_length"])
    train_sequence_count = int(data_config["train_sequences"])
    validation_sequence_count = int(
        data_config["validation_sequences"]
    )
    row_length = sequence_length + 1

    train_required_characters = train_sequence_count * row_length
    validation_required_characters = (
        validation_sequence_count * row_length
    )

    train_text, next_index, train_stories = collect_disjoint_story_text(
        dataset=shuffled_dataset,
        start_index=0,
        required_characters=train_required_characters,
        text_field=data_config["text_field"],
        separator=data_config["separator"],
    )

    validation_text, final_index, validation_stories = (
        collect_disjoint_story_text(
            dataset=shuffled_dataset,
            start_index=next_index,
            required_characters=validation_required_characters,
            text_field=data_config["text_field"],
            separator=data_config["separator"],
        )
    )

    vocabulary = CharVocabulary.from_training_text(train_text)

    validation_unknown_characters = sorted(
        set(validation_text) - set(vocabulary.char_to_idx)
    )
    validation_unknown_count = sum(
        character not in vocabulary.char_to_idx
        for character in validation_text
    )

    train_tensor = build_sequence_tensor(
        text=train_text,
        number_of_sequences=train_sequence_count,
        sequence_length=sequence_length,
        vocabulary=vocabulary,
    )
    validation_tensor = build_sequence_tensor(
        text=validation_text,
        number_of_sequences=validation_sequence_count,
        sequence_length=sequence_length,
        vocabulary=vocabulary,
    )

    torch.save(train_tensor, train_tensor_path)
    torch.save(validation_tensor, validation_tensor_path)
    vocabulary.save(vocabulary_path)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "member_name": config["experiment"]["member_name"],
        "experiment_name": config["experiment"]["name"],
        "dataset_id": data_config["dataset_id"],
        "dataset_revision": data_config["dataset_revision"],
        "source_split": data_config["source_split"],
        "source_dataset_fingerprint": source_dataset._fingerprint,
        "shuffled_dataset_fingerprint": shuffled_dataset._fingerprint,
        "split_seed": int(data_config["split_seed"]),
        "text_field": data_config["text_field"],
        "separator_repr": repr(data_config["separator"]),
        "sequence_length": sequence_length,
        "row_length_including_target_shift": row_length,
        "train_sequences": train_sequence_count,
        "validation_sequences": validation_sequence_count,
        "train_characters": len(train_text),
        "validation_characters": len(validation_text),
        "train_stories_used": train_stories,
        "validation_stories_used": validation_stories,
        "train_shuffled_index_range": [0, next_index],
        "validation_shuffled_index_range": [next_index, final_index],
        "story_overlap_between_splits": False,
        "vocabulary_size": vocabulary.size,
        "unknown_token": vocabulary.unknown_token,
        "validation_unknown_characters": validation_unknown_characters,
        "validation_unknown_character_count": validation_unknown_count,
        "train_tensor_shape": list(train_tensor.shape),
        "validation_tensor_shape": list(validation_tensor.shape),
        "tensor_dtype": str(train_tensor.dtype),
        "train_text_sha256": sha256_text(train_text),
        "validation_text_sha256": sha256_text(validation_text),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest["train_tensor_sha256"] = sha256_file(train_tensor_path)
    manifest["validation_tensor_sha256"] = sha256_file(
        validation_tensor_path
    )
    manifest["vocabulary_sha256"] = sha256_file(vocabulary_path)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "train_tensor_path": train_tensor_path.relative_to(repo_root).as_posix(),
        "validation_tensor_path": validation_tensor_path.relative_to(repo_root).as_posix(),
        "vocabulary_path": vocabulary_path.relative_to(repo_root).as_posix(),
        "manifest_path": manifest_path.relative_to(repo_root).as_posix(),
        "train_shape": list(train_tensor.shape),
        "validation_shape": list(validation_tensor.shape),
        "vocabulary_size": vocabulary.size,
        "train_stories": train_stories,
        "validation_stories": validation_stories,
        "validation_unknown_count": validation_unknown_count,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare character-level TinyStories sequences."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML experiment configuration.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing preprocessed outputs.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    summary = prepare_data(
        config_path=arguments.config.resolve(),
        force=arguments.force,
    )

    print("Preprocessing completed successfully.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()