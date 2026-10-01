"""Produce traceable tables; human failure observations remain explicit inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from common import member_root, sha256_file, write_json


def markdown_blocks(blocks):
    """Keep adjacent Markdown table rows together while separating prose."""
    parts = []
    previous = ""
    for block in blocks:
        if not block:
            continue
        separator = "\n" if block.startswith("|") and previous.startswith("|") else "\n\n"
        parts.append((separator if parts else "") + block)
        previous = block
    return "".join(parts) + "\n"


def validate_cases(samples, cases):
    if len(cases) != 3 or len({c.get("sample_id") for c in cases}) != 3:
        raise ValueError("Provide exactly three different generated sample IDs.")
    lookup = {sample["sample_id"]: sample for sample in samples}
    for case in cases:
        for key in ("sample_id", "excerpt", "failure_type", "observation", "testable_fix"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"Each case needs a nonempty {key}.")
        if (
            case["sample_id"] not in lookup
            or case["excerpt"] not in lookup[case["sample_id"]]["continuation"]
        ):
            raise ValueError(
                "Failure excerpts must be exact substrings of the referenced generated continuation."
            )
    return lookup


def generate_report(root, require_failures=False):
    metrics = json.loads(
        (root / "outputs/metrics/evaluation_metrics.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (root / "outputs/metrics/run_configuration.json").read_text(encoding="utf-8")
    )
    samples = json.loads(
        (root / "outputs/samples/generated_samples.json").read_text(encoding="utf-8")
    )["samples"]
    cases = json.loads((root / "analysis/failure_cases.json").read_text(encoding="utf-8"))["cases"]
    interpretation_path = root / "analysis/interpretation.json"
    interpretation = (
        json.loads(interpretation_path.read_text(encoding="utf-8"))
        if interpretation_path.exists()
        else {}
    )
    interpretation_keys = (
        "architecture_justification",
        "hyperparameter_justification",
        "observations",
        "limitations_and_next_steps",
    )
    interpretation_complete = all(
        isinstance(interpretation.get(k), str) and interpretation[k].strip()
        for k in interpretation_keys
    )
    if require_failures and not interpretation_complete:
        raise ValueError(
            "Finalization requires your architecture/hyperparameter justifications and observations/limitations in analysis/interpretation.json."
        )
    if require_failures or cases:
        lookup = validate_cases(samples, cases)
        paragraphs = [
            "# Task 1 Failure Analysis - Viraat Chaudhary",
            "Three manually reviewed failures from the saved model outputs.",
        ]
        for n, case in enumerate(cases, 1):
            source = lookup[case["sample_id"]]
            quoted = "\n".join("> " + line for line in case["excerpt"].splitlines())
            paragraphs.append(
                f"## Case {n}: {case['failure_type']}\n\nSample: `{case['sample_id']}`; decoding: {source['mode']}; temperature: {source['temperature']}.\n\n{quoted}\n\nObservation: {case['observation']}\n\nTestable fix: {case['testable_fix']}\n\nEvidence: outputs/samples/generated_samples.json"
            )
        (root / "failure_analysis.md").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    status = (
        "Full training/evaluation and manual write-ups present; review the executed notebook and team comparison before submission."
        if cases and interpretation_complete
        else "Full training and evaluation complete; manual failure/design interpretation is still pending."
    )
    if metrics["smoke_test"]:
        status = "SYNTHETIC SMOKE TEST ONLY - not assessed training or submission evidence."
    m, t, d = config["model"], metrics["training"], config["data"]
    lines = [
        "# Task 1 Results - Viraat Chaudhary",
        f"Status: {status}",
        "## Data and architecture",
        f"Dataset: `{d['dataset_id']}`; revision: `{d['dataset_revision']}`; split seed: {d['split_seed']}.",
        f"The existing Team 15 protocol counts fixed-length sequences: {d['train_sequences']:,} training / {d['validation_sequences']:,} validation, context {d['sequence_length']}. Story-record groups are disjoint before sequence construction; only training text builds the vocabulary.",
        f"{m['num_layers']} post-norm blocks; width {m['d_model']}; heads {m['num_heads']}; feed-forward width {m['d_ff']}; ReLU; dropout {m['dropout']}; learned positions; untied output head. Attention, causal masking and layer normalization use explicit tensor operations.",
        "This differs from Anshika's pre-norm/GELU/tied-output decoder. The comparison changes several factors and is not a controlled one-factor ablation.",
        "## Training",
        f"Completed epochs: {t['epochs_completed']}; batch size: {config['training']['batch_size']}; optimizer: AdamW; peak LR: {config['training']['learning_rate']}; minimum LR: {config['training']['minimum_learning_rate']}; warm-up: {config['training']['warmup_ratio']:.0%}; cosine decay; precision: {t['precision']}; best checkpoint epoch: {metrics['checkpoint']['epoch']}.",
        "## Required metrics",
        "| Metric | Value |",
        "|---|---:|",
    ]
    import csv

    with (root / "metrics_report.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lines.append(f"| {row['metric']} | {float(row['value']):.8g} |")
    lines += [
        "",
        "## Metric definitions and evidence",
        f"Epoch validation during training uses {t['precision']} inference with dropout disabled. The final reported checkpoint metrics are recomputed in FP32, so small precision-related differences from the epoch curve are expected.",
        "Primary training/validation cross-entropy and generalization gap use FP32 inference with dropout disabled on both full sets at the same selected checkpoint. The final-epoch online training loss is reported separately. Anshika's saved training loss/gap instead use epoch-average training loss, so these two definitions must not be silently mixed.",
        "Perplexity = exp(validation CE); bits per character = validation CE / ln(2); generalization gap = validation CE minus training CE. Accuracy counts correct next-character predictions across all validation target positions.",
        "Distinct-n is unique character n-grams divided by their total occurrences over the 27 sampled continuations. Prompts and cross-sample boundaries are excluded. Repeated 4-gram rate is the mean per-continuation duplicate-occurrence fraction. Greedy samples are preserved but excluded from sampled diversity totals.",
        "Gradient norms are measured before clipping. A loss spike exceeds 1.5 times the median of the previous 20 minibatch losses. Nonfinite losses/gradients stop the run and are recorded in untouched logs. Training throughput uses processed target characters divided by completed training-pass time; generation throughput includes both greedy and sampled generated characters.",
        t["time_definition"],
        "Evidence: metrics_report.csv; outputs/metrics/evaluation_metrics.json; training_summary.json; training_history.csv; step_metrics.csv; split_manifest.json; outputs/plots/loss_curves.png; outputs/samples/generated_samples.json; logs/; checkpoints/; environment_manifest.txt.",
        "## Interpretation to review and defend",
        "Explain why you selected the post-norm/ReLU/untied architecture and these hyperparameters. Inspect the curves and generated samples before writing conclusions about generalization, diversity, grammar or coherence. State the own-split and hardware limitations when comparing the models. The implementation does not infer qualitative failure observations from numeric metrics.",
        "## References",
        "Vaswani et al. (2017), Attention Is All You Need: https://arxiv.org/abs/1706.03762",
        "Eldan and Li (2023), TinyStories: https://arxiv.org/abs/2305.07759",
    ]
    if interpretation_complete:
        lines += ["## Member's reviewed interpretation"] + [
            f"### {key.replace('_', ' ').capitalize()}\n\n{interpretation[key]}"
            for key in interpretation_keys
        ]
    common_path = root.parent / "comparison/common_evaluation_manifest.json"
    if common_path.exists() and not metrics["smoke_test"]:
        common = json.loads(common_path.read_text(encoding="utf-8"))
        records = {record["member"]: record for record in common["results"]}
        own = records["Viraat_Chaudhary"]
        if own["checkpoint_sha256"] != metrics["checkpoint"]["sha256"]:
            raise ValueError("Common evaluation refers to a different member checkpoint.")
        lines += [
            "## Common held-out comparison",
            "| Metric | Anshika | Viraat |",
            "|---|---:|---:|",
            f"| Common CE | {records['Anshika_Goel']['common_validation_ce']:.8f} | {own['common_validation_ce']:.8f} |",
            f"| Common next-character accuracy | {100*records['Anshika_Goel']['common_next_character_accuracy']:.4f}% | {100*own['common_next_character_accuracy']:.4f}% |",
            "Use the documented quality criterion and coverage limitations in [the team selection write-up](../comparison/best_model_selection.md). Keep both members' individual results.",
        ]
    (root / "results.md").write_text(markdown_blocks(lines), encoding="utf-8")
    lock_path = root / "requirements.lock.txt"
    if not lock_path.exists():
        locked = sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib.metadata.distributions()
            if dist.metadata.get("Name")
        )
        lock_path.write_text(
            "# Actual executing environment; provenance, not a cross-platform Colab installation command.\n"
            + "\n".join(locked)
            + "\n",
            encoding="utf-8",
        )
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and not any(
            part
            in {"data_processed", "hf_cache", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
            for part in p.relative_to(root).parts
        )
        and p.name not in {"environment_manifest.txt", "artifact_manifest.json"}
        and not p.name.endswith(".tmp")
    ]
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(files)}
    write_json(
        root / "artifact_manifest.json",
        {
            "member": "Viraat Chaudhary",
            "team": 15,
            "checkpoint": metrics["checkpoint"],
            "files": hashes,
            "failure_analysis_present": len(cases) == 3,
            "design_interpretation_present": interpretation_complete,
            "personal_review_status": "not_attested_by_automated_check",
        },
    )
    env = metrics["environment"]
    manifest = [
        "DATA 266 Lab 1 - Viraat Chaudhary - Team 15",
        f"Status: {status}",
        "Hardware/package versions are observed values from the evaluating runtime:",
        json.dumps(env, indent=2),
        "Checkpoint/result mapping and SHA256:",
    ]
    manifest += [
        f"{name}: {digest}"
        for name, digest in hashes.items()
        if name.startswith(("checkpoints/", "logs/", "outputs/metrics/", "configs/"))
    ]
    (root / "environment_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-root", type=Path, default=member_root())
    parser.add_argument("--require-failures", action="store_true")
    args = parser.parse_args()
    generate_report(args.member_root.resolve(), args.require_failures)


if __name__ == "__main__":
    main()
