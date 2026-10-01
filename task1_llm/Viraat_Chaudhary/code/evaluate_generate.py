"""Evaluate the selected checkpoint and preserve all thirty generated continuations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch

from common import (
    environment,
    event,
    member_root,
    raw_log,
    read_config,
    sha256_file,
    synchronize,
    write_json,
)
from data import CharacterSequences, CharacterVocabulary, verify_processed
from model import CharGPT, DecoderConfig
from train import loader, score


def distinct_n(sequences, n):
    if n <= 0:
        raise ValueError("n must be positive.")
    unique, count = set(), 0
    for sequence in sequences:
        grams = [tuple(sequence[i : i + n]) for i in range(max(0, len(sequence) - n + 1))]
        unique.update(grams)
        count += len(grams)
    return len(unique) / count if count else 0.0


def repeated_ngram_rate(sequence, n=4):
    if n <= 0:
        raise ValueError("n must be positive.")
    grams = [tuple(sequence[i : i + n]) for i in range(max(0, len(sequence) - n + 1))]
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.0


def diversity(sequences):
    return {
        "sample_count": len(sequences),
        "distinct_1": distinct_n(sequences, 1),
        "distinct_2": distinct_n(sequences, 2),
        "distinct_3": distinct_n(sequences, 3),
        "repeated_4gram_rate": (
            statistics.mean(repeated_ngram_rate(s) for s in sequences) if sequences else 0.0
        ),
    }


def evaluate(root, config, smoke=False):
    verify_processed(root, config)
    summary = json.loads((root / "outputs/metrics/training_summary.json").read_text())
    if not summary["completed"] or (summary["smoke_test"] and not smoke):
        raise ValueError("Run the complete assessed training before publishing final metrics.")
    metrics_path = root / "outputs/metrics/evaluation_metrics.json"
    if metrics_path.exists():
        raise FileExistsError("Evaluation already exists; archive it before an intentional rerun.")
    checkpoint_path = root / "checkpoints/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vocabulary_path = root / "outputs/metrics/vocabulary.json"
    if checkpoint["vocabulary_sha256"] != sha256_file(vocabulary_path):
        raise ValueError("Checkpoint and vocabulary hashes differ.")
    if checkpoint["experiment_config"] != config:
        raise ValueError("The evaluation configuration differs from the training snapshot.")
    vocabulary = CharacterVocabulary.load(vocabulary_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharGPT(DecoderConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("The checkpoint contains nonfinite weights.")
    model.to(device).eval()
    batch_size = config["training"]["batch_size"]
    scores = {}
    # Both losses below use the same best checkpoint, FP32 and disabled dropout.
    for name in ("train", "validation"):
        dataset = CharacterSequences.load(root / f"data_processed/{name}.npy")
        scores[name] = score(model, loader(dataset, batch_size, cuda=device.type == "cuda"), device)
    generation = config["generation"]
    torch.manual_seed(generation["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(generation["seed"])
        torch.cuda.reset_peak_memory_stats()
    samples, sampled, by_temperature, timing = [], [], {}, []
    for number, prompt in enumerate(generation["prompts"], 1):
        if not prompt or any(c not in vocabulary.char_to_idx for c in prompt):
            raise ValueError(
                "Generation prompts must be nonempty and fully in the training vocabulary."
            )
        modes = [(True, None, 1)] if generation["include_greedy_sample"] else []
        modes += [
            (False, float(t), generation["samples_per_temperature"])
            for t in generation["temperatures"]
        ]
        for greedy, temperature, count in modes:
            prompt_ids = (
                torch.tensor(vocabulary.encode(prompt), dtype=torch.long, device=device)
                .unsqueeze(0)
                .repeat(count, 1)
            )
            synchronize(device)
            start = time.perf_counter()
            output = model.generate(
                prompt_ids, generation["max_new_characters"], temperature or 1.0, greedy
            )
            synchronize(device)
            seconds = time.perf_counter() - start
            continuations = output[:, prompt_ids.shape[1] :].cpu().tolist()
            timing.append(
                {
                    "prompt_number": number,
                    "mode": "greedy" if greedy else "sampled",
                    "temperature": temperature,
                    "sample_count": count,
                    "generated_tokens": sum(map(len, continuations)),
                    "seconds": seconds,
                }
            )
            for ids in continuations:
                record = {
                    "sample_id": f"sample_{len(samples)+1:03d}",
                    "prompt": prompt,
                    "mode": "greedy" if greedy else "sampled",
                    "temperature": temperature,
                    "continuation_token_ids": ids,
                    "continuation": vocabulary.decode(ids),
                    "repeated_4gram_rate": repeated_ngram_rate(ids),
                }
                samples.append(record)
                if not greedy:
                    sampled.append(ids)
                    by_temperature.setdefault(str(temperature), []).append(ids)
                event(
                    "GENERATED_SAMPLE",
                    sample_id=record["sample_id"],
                    temperature=temperature,
                    repeated_4gram_rate=record["repeated_4gram_rate"],
                )
    total_time = sum(r["seconds"] for r in timing)
    total_tokens = sum(r["generated_tokens"] for r in timing)
    val_loss = scores["validation"]["loss"]
    if val_loss > 700:
        raise FloatingPointError("Perplexity overflows float64; retain the raw loss for diagnosis.")
    payload = {
        "schema_version": 1,
        "member_name": config["experiment"]["member_name"],
        "smoke_test": smoke,
        "experiment": config["experiment"]["name"],
        "environment": environment(),
        "checkpoint": {
            "path": "checkpoints/best_model.pt",
            "sha256": sha256_file(checkpoint_path),
            "epoch": checkpoint["epoch"],
        },
        "configuration": {
            **generation,
            "dataset_split_seed": config["data"]["split_seed"],
            "metric_protocol": "Character n-grams over continuations, no prompt/cross-sample n-grams; 27 sampled outputs; mean per-sample repeated 4-gram rate",
        },
        "model": {"parameter_count": model.parameter_count(), "vocabulary_size": len(vocabulary)},
        "training": {
            **summary,
            "selected_checkpoint_training_cross_entropy": scores["train"]["loss"],
            "loss_definition": "FP32 inference on the full training set at the selected best checkpoint; dropout disabled",
            "epoch_average_loss_definition": "training_summary.final_training_loss is final-epoch online loss with dropout active",
        },
        "validation": {
            "cross_entropy_loss": val_loss,
            "perplexity": math.exp(val_loss),
            "bits_per_character": val_loss / math.log(2),
            "generalization_gap": val_loss - scores["train"]["loss"],
            "legacy_epoch_average_gap": val_loss - summary["final_training_loss"],
            "top1_accuracy": scores["validation"]["accuracy"],
            "tokens": scores["validation"]["tokens"],
        },
        "generation": {
            "sample_count": len(samples),
            "sampled_sample_count": len(sampled),
            "total_generated_tokens": total_tokens,
            "total_generation_seconds": total_time,
            "tokens_per_second": total_tokens / max(total_time, 1e-12),
            "timing_records": timing,
        },
        "diversity": {
            "all_sampled_outputs": diversity(sampled),
            "by_temperature": {t: diversity(ids) for t, ids in by_temperature.items()},
        },
    }
    if device.type == "cuda":
        payload["generation"]["peak_gpu_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    write_json(
        root / "outputs/samples/generated_samples.json",
        {"generation_seed": generation["seed"], "samples": samples},
    )
    text_path = root / "outputs/samples/generated_samples.txt"
    text_path.write_text(
        "\n\n".join(
            f"{s['sample_id']} | {s['mode']} | temperature={s['temperature']}\nPrompt: {s['prompt']}\n{s['continuation']}"
            for s in samples
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(metrics_path, payload)
    metrics = {
        "training_cross_entropy_loss": scores["train"]["loss"],
        "final_epoch_training_cross_entropy_loss": summary["final_training_loss"],
        "validation_cross_entropy_loss": val_loss,
        "perplexity": math.exp(val_loss),
        "bits_per_character": val_loss / math.log(2),
        "generalization_gap": payload["validation"]["generalization_gap"],
        "top1_next_character_accuracy": scores["validation"]["accuracy"],
        **{k: v for k, v in diversity(sampled).items() if k != "sample_count"},
        **{
            k: summary[k]
            for k in (
                "gradient_norm_mean",
                "gradient_norm_max",
                "loss_spike_count",
                "nonfinite_loss_count",
                "nonfinite_gradient_count",
                "parameter_count",
                "training_tokens_per_second",
                "total_training_seconds",
                "peak_gpu_allocated_mb",
                "peak_gpu_reserved_mb",
                "peak_cpu_rss_mb",
            )
        },
        "generation_tokens_per_second": payload["generation"]["tokens_per_second"],
    }
    with (root / "metrics_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "checkpoint_epoch", "evidence"])
        for name, value in metrics.items():
            writer.writerow(
                [name, value, checkpoint["epoch"], "outputs/metrics/evaluation_metrics.json"]
            )
    event(
        "EVALUATION_COMPLETE",
        validation_cross_entropy=val_loss,
        samples=len(samples),
        diversity=diversity(sampled),
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=member_root() / "configs/gpt_char.yaml")
    args = parser.parse_args()
    root = args.config.resolve().parents[1]
    with raw_log(root, "evaluation_generation"):
        evaluate(root, read_config(args.config))


if __name__ == "__main__":
    main()
