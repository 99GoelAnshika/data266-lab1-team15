from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from data import (
    CharVocabulary,
    SequenceTensorDataset,
    repository_root,
    resolve_project_path,
    sha256_file,
)
from model import CharacterGPT, GPTConfig
from train import (
    evaluate,
    make_validation_loader,
    set_reproducible_seed,
)


def relative_to_repository(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(
        repo_root.resolve()
    ).as_posix()


def distinct_n(
    token_sequences: list[list[int]],
    n: int,
) -> float:
    if n <= 0:
        raise ValueError("n must be positive.")

    unique_ngrams: set[tuple[int, ...]] = set()
    total_ngrams = 0

    for sequence in token_sequences:
        sequence_ngram_count = max(
            0,
            len(sequence) - n + 1,
        )
        total_ngrams += sequence_ngram_count

        for start in range(sequence_ngram_count):
            unique_ngrams.add(
                tuple(sequence[start : start + n])
            )

    if total_ngrams == 0:
        return 0.0

    return len(unique_ngrams) / total_ngrams


def repeated_ngram_rate(
    token_sequence: list[int],
    n: int,
) -> float:
    if n <= 0:
        raise ValueError("n must be positive.")

    ngram_count = max(
        0,
        len(token_sequence) - n + 1,
    )

    if ngram_count == 0:
        return 0.0

    ngrams = [
        tuple(token_sequence[start : start + n])
        for start in range(ngram_count)
    ]
    repeated_count = len(ngrams) - len(set(ngrams))

    return repeated_count / len(ngrams)


def diversity_summary(
    token_sequences: list[list[int]],
) -> dict[str, float | int]:
    repeated_rates = [
        repeated_ngram_rate(sequence, n=4)
        for sequence in token_sequences
    ]

    return {
        "sample_count": len(token_sequences),
        "generated_tokens": sum(
            len(sequence)
            for sequence in token_sequences
        ),
        "distinct_1": distinct_n(token_sequences, n=1),
        "distinct_2": distinct_n(token_sequences, n=2),
        "distinct_3": distinct_n(token_sequences, n=3),
        "repeated_4gram_rate": (
            statistics.mean(repeated_rates)
            if repeated_rates
            else 0.0
        ),
        "minimum_sample_repeated_4gram_rate": (
            min(repeated_rates)
            if repeated_rates
            else 0.0
        ),
        "maximum_sample_repeated_4gram_rate": (
            max(repeated_rates)
            if repeated_rates
            else 0.0
        ),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def normalize_display_text(text: str) -> str:
    normalized = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    return "\n".join(
        line.rstrip(" `t")
        for line in normalized.split("\n")
    )


def write_sample_text(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    lines: list[str] = []

    for record in records:
        lines.extend(
            [
                "=" * 80,
                f"Sample ID: {record['sample_id']}",
                f"Prompt: {record['prompt']}",
                f"Mode: {record['mode']}",
                f"Temperature: {record['temperature']}",
                f"Sample index: {record['sample_index']}",
                "",
                "Generated continuation:",
                normalize_display_text(record["continuation"]),
                "",
                "Complete text:",
                normalize_display_text(record["complete_text"]),
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def timed_generate(
    model: CharacterGPT,
    prompt_batch: torch.Tensor,
    max_new_characters: int,
    temperature: float,
    greedy: bool,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    generated = model.generate(
        input_ids=prompt_batch,
        max_new_characters=max_new_characters,
        temperature=temperature,
        greedy=greedy,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_seconds = time.perf_counter() - start_time

    return generated, elapsed_seconds


def evaluate_and_generate(
    config_path: Path,
    checkpoint_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    repo_root = repository_root()
    member_root = config_path.resolve().parent.parent

    with config_path.open("r", encoding="utf-8") as handle:
        experiment_config = yaml.safe_load(handle)

    data_config = experiment_config["data"]
    training_config = experiment_config["training"]
    generation_config = experiment_config["generation"]

    vocabulary_path = resolve_project_path(
        repo_root,
        data_config["vocabulary_file"],
    )
    processed_directory = resolve_project_path(
        repo_root,
        data_config["processed_dir"],
    )
    validation_tensor_path = (
        processed_directory / "validation_sequences.pt"
    )
    training_summary_path = (
        member_root
        / "outputs"
        / "metrics"
        / "training_summary.json"
    )

    evaluation_metrics_path = (
        member_root
        / "outputs"
        / "metrics"
        / "evaluation_metrics.json"
    )
    samples_json_path = (
        member_root
        / "outputs"
        / "samples"
        / "generated_samples.json"
    )
    samples_text_path = (
        member_root
        / "outputs"
        / "samples"
        / "generated_samples.txt"
    )

    output_paths = [
        evaluation_metrics_path,
        samples_json_path,
        samples_text_path,
    ]

    existing_outputs = [
        path for path in output_paths
        if path.exists()
    ]

    if existing_outputs and not force:
        existing_names = ", ".join(
            relative_to_repository(path, repo_root)
            for path in existing_outputs
        )
        raise FileExistsError(
            "Evaluation outputs already exist. Use --force only "
            f"when intentionally replacing them: {existing_names}"
        )

    required_paths = [
        vocabulary_path,
        validation_tensor_path,
        checkpoint_path,
        training_summary_path,
    ]

    missing_paths = [
        path for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        missing_names = ", ".join(
            relative_to_repository(path, repo_root)
            for path in missing_paths
        )
        raise FileNotFoundError(
            f"Missing required artifacts: {missing_names}"
        )

    vocabulary = CharVocabulary.load(vocabulary_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    expected_vocabulary_hash = checkpoint.get(
        "vocabulary_sha256"
    )
    actual_vocabulary_hash = sha256_file(vocabulary_path)

    if (
        expected_vocabulary_hash is not None
        and expected_vocabulary_hash
        != actual_vocabulary_hash
    ):
        raise ValueError(
            "Checkpoint vocabulary hash does not match the "
            "current vocabulary file."
        )

    model = CharacterGPT(
        GPTConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if not all(
        torch.isfinite(parameter).all().item()
        for parameter in model.parameters()
    ):
        raise ValueError(
            "The checkpoint contains nonfinite model parameters."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_mixed_precision = bool(
        training_config["mixed_precision"]
        and device.type == "cuda"
    )

    model = model.to(device)
    model.eval()

    validation_dataset = SequenceTensorDataset.from_file(
        validation_tensor_path
    )
    validation_loader = make_validation_loader(
        validation_dataset,
        batch_size=int(training_config["batch_size"]),
        num_workers=int(training_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    validation_metrics = evaluate(
        model=model,
        data_loader=validation_loader,
        device=device,
        use_mixed_precision=use_mixed_precision,
    )

    validation_loss = float(
        validation_metrics["loss"]
    )

    training_summary = json.loads(
        training_summary_path.read_text(
            encoding="utf-8"
        )
    )

    generation_seed = (
        int(training_config["training_seed"]) + 1
    )
    set_reproducible_seed(generation_seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    prompts = list(generation_config["prompts"])
    temperatures = [
        float(value)
        for value in generation_config["temperatures"]
    ]
    samples_per_temperature = int(
        generation_config["samples_per_temperature"]
    )
    max_new_characters = int(
        generation_config["max_new_characters"]
    )
    include_greedy = bool(
        generation_config["include_greedy_sample"]
    )

    sample_records: list[dict[str, Any]] = []
    sampled_token_sequences: list[list[int]] = []
    sampled_tokens_by_temperature: dict[
        str,
        list[list[int]],
    ] = {
        str(temperature): []
        for temperature in temperatures
    }
    timing_records: list[dict[str, Any]] = []

    total_generated_tokens = 0
    total_generation_seconds = 0.0
    sample_number = 0

    for prompt_number, prompt in enumerate(
        prompts,
        start=1,
    ):
        prompt_token_ids = vocabulary.encode(
            prompt
        ).tolist()

        if vocabulary.unknown_index in prompt_token_ids:
            raise ValueError(
                f"Prompt contains a character outside the "
                f"training vocabulary: {prompt!r}"
            )

        prompt_tensor = torch.tensor(
            prompt_token_ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        generation_conditions: list[
            tuple[str, float | None, int]
        ] = []

        if include_greedy:
            generation_conditions.append(
                ("greedy", None, 1)
            )

        generation_conditions.extend(
            (
                "sampled",
                temperature,
                samples_per_temperature,
            )
            for temperature in temperatures
        )

        for (
            mode,
            temperature,
            sample_count,
        ) in generation_conditions:
            prompt_batch = prompt_tensor.repeat(
                sample_count,
                1,
            )

            generation_temperature = (
                1.0
                if temperature is None
                else temperature
            )

            generated_batch, elapsed_seconds = (
                timed_generate(
                    model=model,
                    prompt_batch=prompt_batch,
                    max_new_characters=(
                        max_new_characters
                    ),
                    temperature=(
                        generation_temperature
                    ),
                    greedy=mode == "greedy",
                    device=device,
                )
            )

            generated_token_count = (
                sample_count * max_new_characters
            )
            total_generated_tokens += (
                generated_token_count
            )
            total_generation_seconds += (
                elapsed_seconds
            )

            timing_records.append(
                {
                    "prompt_number": prompt_number,
                    "mode": mode,
                    "temperature": temperature,
                    "sample_count": sample_count,
                    "generated_tokens": (
                        generated_token_count
                    ),
                    "seconds": elapsed_seconds,
                    "tokens_per_second": (
                        generated_token_count
                        / elapsed_seconds
                    ),
                }
            )

            for sample_index in range(
                sample_count
            ):
                sample_number += 1

                complete_ids = (
                    generated_batch[sample_index]
                    .detach()
                    .cpu()
                    .tolist()
                )
                continuation_ids = complete_ids[
                    len(prompt_token_ids) :
                ]

                record = {
                    "sample_id": (
                        f"sample_{sample_number:03d}"
                    ),
                    "prompt_number": prompt_number,
                    "prompt": prompt,
                    "mode": mode,
                    "temperature": temperature,
                    "sample_index": (
                        sample_index + 1
                    ),
                    "prompt_token_count": len(
                        prompt_token_ids
                    ),
                    "continuation_token_count": len(
                        continuation_ids
                    ),
                    "continuation_token_ids": (
                        continuation_ids
                    ),
                    "continuation": vocabulary.decode(
                        continuation_ids
                    ),
                    "complete_text": vocabulary.decode(
                        complete_ids
                    ),
                    "repeated_4gram_rate": (
                        repeated_ngram_rate(
                            continuation_ids,
                            n=4,
                        )
                    ),
                }
                sample_records.append(record)

                if mode == "sampled":
                    sampled_token_sequences.append(
                        continuation_ids
                    )
                    sampled_tokens_by_temperature[
                        str(temperature)
                    ].append(continuation_ids)

    generation_peak_gpu_allocated_mb = (
        torch.cuda.max_memory_allocated()
        / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    generation_peak_gpu_reserved_mb = (
        torch.cuda.max_memory_reserved()
        / (1024**2)
        if device.type == "cuda"
        else 0.0
    )

    diversity_by_temperature = {
        temperature: diversity_summary(sequences)
        for temperature, sequences
        in sampled_tokens_by_temperature.items()
    }

    sample_payload = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment": experiment_config[
            "experiment"
        ]["name"],
        "member_name": experiment_config[
            "experiment"
        ]["member_name"],
        "generation_seed": generation_seed,
        "checkpoint": relative_to_repository(
            checkpoint_path,
            repo_root,
        ),
        "samples": sample_records,
    }

    evaluation_payload = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment": experiment_config[
            "experiment"
        ]["name"],
        "member_name": experiment_config[
            "experiment"
        ]["member_name"],
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "checkpoint": {
            "path": relative_to_repository(
                checkpoint_path,
                repo_root,
            ),
            "sha256": sha256_file(
                checkpoint_path
            ),
            "epoch": int(checkpoint["epoch"]),
            "validation_loss_at_save": float(
                checkpoint["validation_loss"]
            ),
        },
        "configuration": {
            "path": relative_to_repository(
                config_path,
                repo_root,
            ),
            "sha256": sha256_file(config_path),
            "generation_seed": generation_seed,
            "prompts": prompts,
            "temperatures": temperatures,
            "samples_per_temperature": (
                samples_per_temperature
            ),
            "include_greedy_sample": include_greedy,
            "max_new_characters": (
                max_new_characters
            ),
        },
        "model": {
            "parameter_count": (
                model.parameter_count()
            ),
            "vocabulary_size": vocabulary.size,
        },
        "training": {
            "final_training_loss": (
                training_summary[
                    "final_training_loss"
                ]
            ),
            "training_tokens_per_second": (
                training_summary[
                    "training_tokens_per_second"
                ]
            ),
            "total_training_seconds": (
                training_summary[
                    "total_training_seconds"
                ]
            ),
            "peak_gpu_allocated_mb": (
                training_summary[
                    "peak_gpu_allocated_mb"
                ]
            ),
            "peak_gpu_reserved_mb": (
                training_summary[
                    "peak_gpu_reserved_mb"
                ]
            ),
            "gradient_norm_mean": (
                training_summary[
                    "gradient_norm_mean"
                ]
            ),
            "gradient_norm_max": (
                training_summary[
                    "gradient_norm_max"
                ]
            ),
            "loss_spike_count": (
                training_summary[
                    "loss_spike_count"
                ]
            ),
            "nonfinite_loss_count": (
                training_summary[
                    "nonfinite_loss_count"
                ]
            ),
            "nonfinite_gradient_count": (
                training_summary[
                    "nonfinite_gradient_count"
                ]
            ),
        },
        "validation": {
            "cross_entropy_loss": validation_loss,
            "perplexity": math.exp(
                min(validation_loss, 20.0)
            ),
            "bits_per_character": (
                validation_loss / math.log(2.0)
            ),
            "generalization_gap": (
                validation_loss
                - float(
                    training_summary[
                        "final_training_loss"
                    ]
                )
            ),
            "top1_accuracy": float(
                validation_metrics[
                    "top1_accuracy"
                ]
            ),
            "tokens": int(
                validation_metrics["tokens"]
            ),
            "seconds": float(
                validation_metrics["seconds"]
            ),
            "tokens_per_second": float(
                validation_metrics[
                    "tokens_per_second"
                ]
            ),
        },
        "generation": {
            "sample_count": len(sample_records),
            "sampled_sample_count": len(
                sampled_token_sequences
            ),
            "total_generated_tokens": (
                total_generated_tokens
            ),
            "total_generation_seconds": (
                total_generation_seconds
            ),
            "tokens_per_second": (
                total_generated_tokens
                / total_generation_seconds
            ),
            "peak_gpu_allocated_mb": (
                generation_peak_gpu_allocated_mb
            ),
            "peak_gpu_reserved_mb": (
                generation_peak_gpu_reserved_mb
            ),
            "timing_records": timing_records,
        },
        "diversity": {
            "all_sampled_outputs": (
                diversity_summary(
                    sampled_token_sequences
                )
            ),
            "by_temperature": (
                diversity_by_temperature
            ),
        },
    }

    write_json(
        samples_json_path,
        sample_payload,
    )
    write_sample_text(
        samples_text_path,
        sample_records,
    )
    write_json(
        evaluation_metrics_path,
        evaluation_payload,
    )

    result_summary = {
        "validation_loss": validation_loss,
        "validation_perplexity": (
            evaluation_payload[
                "validation"
            ]["perplexity"]
        ),
        "validation_top1_accuracy": (
            evaluation_payload[
                "validation"
            ]["top1_accuracy"]
        ),
        "sample_count": len(sample_records),
        "generation_tokens_per_second": (
            evaluation_payload[
                "generation"
            ]["tokens_per_second"]
        ),
        "diversity": (
            evaluation_payload[
                "diversity"
            ]["all_sampled_outputs"]
        ),
        "evaluation_metrics": (
            relative_to_repository(
                evaluation_metrics_path,
                repo_root,
            )
        ),
        "samples_json": (
            relative_to_repository(
                samples_json_path,
                repo_root,
            )
        ),
        "samples_text": (
            relative_to_repository(
                samples_text_path,
                repo_root,
            )
        ),
    }

    print(
        "Evaluation and generation completed successfully."
    )
    print(
        json.dumps(
            result_summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    return result_summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the best character GPT checkpoint "
            "and generate text samples."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML experiment configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the best model checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing evaluation outputs.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    evaluate_and_generate(
        config_path=arguments.config.resolve(),
        checkpoint_path=(
            arguments.checkpoint.resolve()
        ),
        force=arguments.force,
    )


if __name__ == "__main__":
    main()