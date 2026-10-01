"""Compare recorded member evidence without inventing results or a common-test winner."""

import argparse
import csv
import json
from pathlib import Path

import yaml


def compare(root: Path):
    folders = [root / name for name in ("Anshika_Goel", "Viraat_Chaudhary")]
    rows = []
    for folder in folders:
        path = folder / "outputs/metrics/evaluation_metrics.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing measured evaluation: {folder.name}. Run both members before comparison."
            )
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("smoke_test"):
            raise ValueError("Synthetic smoke results cannot enter the team comparison.")
        config = yaml.safe_load((folder / "configs/gpt_char.yaml").read_text())
        model, training, validation = value["model"], value["training"], value["validation"]
        diversity = value["diversity"]["all_sampled_outputs"]
        architecture = config["model"]
        row = {
            "member": folder.name,
            "layers": architecture["num_layers"],
            "width": architecture["d_model"],
            "heads": architecture["num_heads"],
            "feed_forward_width": architecture["d_ff"],
            "dropout": architecture["dropout"],
            "normalization": architecture["normalization"],
            "activation": architecture.get("activation", "gelu"),
            "tied_embeddings": architecture["tie_token_and_output_embeddings"],
            "batch_size": config["training"]["batch_size"],
            "epochs": config["training"]["epochs"],
            "learning_rate": config["training"]["learning_rate"],
            "warmup_ratio": config["training"]["warmup_ratio"],
            "split_seed": config["data"]["split_seed"],
            "epoch_average_training_ce": training["final_training_loss"],
            "selected_checkpoint_training_ce": training.get(
                "selected_checkpoint_training_cross_entropy"
            ),
            "inference_generalization_gap": (
                validation["generalization_gap"]
                if "selected_checkpoint_training_cross_entropy" in training
                else None
            ),
            "validation_ce": validation["cross_entropy_loss"],
            "perplexity": validation["perplexity"],
            "bits_per_character": validation["bits_per_character"],
            "legacy_epoch_average_gap": validation.get(
                "legacy_epoch_average_gap", validation["generalization_gap"]
            ),
            "next_character_accuracy": validation["top1_accuracy"],
            **{
                k: diversity[k]
                for k in ("distinct_1", "distinct_2", "distinct_3", "repeated_4gram_rate")
            },
            **{
                k: training[k]
                for k in (
                    "gradient_norm_mean",
                    "gradient_norm_max",
                    "loss_spike_count",
                    "nonfinite_loss_count",
                    "nonfinite_gradient_count",
                    "training_tokens_per_second",
                    "total_training_seconds",
                    "peak_gpu_allocated_mb",
                    "peak_gpu_reserved_mb",
                )
            },
            "parameters": model["parameter_count"],
            "peak_cpu_rss_mb": json.loads(
                (folder / "outputs/metrics/training_summary.json").read_text(encoding="utf-8")
            ).get("peak_cpu_rss_mb"),
            "generation_tokens_per_second": value["generation"]["tokens_per_second"],
            "gpu": training.get("gpu", value.get("gpu")),
            "evidence": f"{folder.name}/outputs/metrics/evaluation_metrics.json",
        }
        if isinstance(row["gpu"], dict):
            row["gpu"] = row["gpu"]["name"]
        rows.append(row)
    output = root / "comparison"
    output.mkdir(exist_ok=True)
    with (output / "task1_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Task 1 - Team 15 comparison", "", "| Field | Anshika | Viraat |", "|---|---|---|"]
    for key in rows[0]:
        lines.append(f"| {key} | {rows[0][key]} | {rows[1][key]} |")
    lines += [
        "",
        "Both rows use actual saved evidence. Validation splits differ; these are descriptive own-split results, not a controlled common-test quality ranking.",
        "The table uses final-epoch online training CE and the corresponding legacy-style gap for consistency with Anshika's saved definitions. Viraat also reports full-training-set inference CE and an inference-mode gap separately.",
        "None in the supplementary inference columns means that quantity was not recorded for that member; it is not an estimated value.",
        "Training/generation speeds and memory were measured on different GPUs and are hardware-specific; architecture alone does not explain a speed difference.",
        "The recorded common held-out evaluation and checkpoint selection are in [best_model_selection.md](best_model_selection.md). The complete report section, including both members' actual failure snippets, is [task1_team_report.md](task1_team_report.md). Retain both individual runs in the combined lab report.",
    ]
    # Markdown tables require contiguous header/separator/data rows.
    (output / "task1_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Comparison written to task1_llm/comparison/. No common-test winner inferred.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=Path(__file__).resolve().parent)
    compare(parser.parse_args().task_root.resolve())
