"""Verify the frozen Task 1 handoff using only the Python standard library."""

import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root):
    task = root / "task1_llm"
    member = task / "Viraat_Chaudhary"
    partner = task / "Anshika_Goel"
    require((root / "README.md").is_file(), "Run from the actual team repository.")
    require(partner.is_dir(), "Anshika_Goel must remain alongside Viraat_Chaudhary.")
    member_manifest = load(member / "artifact_manifest.json")
    for relative, expected in member_manifest["files"].items():
        require(digest(member / relative) == expected, f"Member artifact differs: {relative}")
    shared_manifest = load(task / "comparison/artifact_manifest.json")
    for relative, expected in shared_manifest["files"].items():
        require(digest(root / relative) == expected, f"Shared artifact differs: {relative}")

    summary = load(member / "outputs/metrics/training_summary.json")
    metrics = load(member / "outputs/metrics/evaluation_metrics.json")
    require(summary["completed"] and not summary["smoke_test"], "Real training must be complete.")
    require(summary["epochs_completed"] >= 10, "At least ten full epochs are required.")
    require(summary["global_steps"] == 15630, "The submitted run counter has changed.")
    validation = metrics["validation"]
    require(math.isclose(validation["perplexity"], math.exp(validation["cross_entropy_loss"]), rel_tol=1e-12), "Perplexity definition differs.")
    require(math.isclose(validation["bits_per_character"], validation["cross_entropy_loss"] / math.log(2), rel_tol=1e-12), "BPC definition differs.")
    require(digest(member / "checkpoints/best_model.pt") == metrics["checkpoint"]["sha256"], "Best checkpoint differs.")
    require(digest(member / "checkpoints/last_checkpoint.pt") == summary["last_checkpoint_sha256"], "Last checkpoint differs.")

    notebook = load(member / "code/task1_colab.ipynb")
    require(any(c.get("execution_count") is not None and c.get("outputs") for c in notebook["cells"]), "Executed notebook outputs are missing.")
    require(not any(o.get("output_type") == "error" for c in notebook["cells"] for o in c.get("outputs", [])), "Notebook contains saved error outputs.")
    for path in member.glob("code/**/*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    for path in (task / "compare_task1.py", task / "evaluate_common.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    common = load(task / "comparison/common_evaluation_manifest.json")
    rows = {r["member"]: r for r in common["results"]}
    require(set(rows) == {"Anshika_Goel", "Viraat_Chaudhary"}, "Both common evaluation rows are required.")
    for name, row in rows.items():
        require(digest(task / name / "checkpoints/best_model.pt") == row["checkpoint_sha256"], f"Common checkpoint differs: {name}")
        vocabulary = (task / name / "outputs/metrics/vocabulary.json").read_bytes()
        require(hashlib.sha256(vocabulary).hexdigest() == row["vocabulary_file_sha256"], f"Vocabulary bytes differ from common evidence: {name}")
        expected = row["checkpoint_vocabulary_sha256"]
        forms = (vocabulary, vocabulary.replace(b"\r\n", b"\n"), vocabulary.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        require(any(hashlib.sha256(form).hexdigest() == expected for form in forms), f"Vocabulary/checkpoint mismatch beyond LF/CRLF: {name}")
    require(rows["Anshika_Goel"]["common_validation_ce"] < rows["Viraat_Chaudhary"]["common_validation_ce"], "Selection report must be reviewed after changed common scores.")

    sources = {}
    for name in rows:
        samples = load(task / name / "outputs/samples/generated_samples.json")["samples"]
        sources[name] = {s["sample_id"]: s["continuation"] for s in samples}
    cases = load(task / "comparison/team_failure_examples.json")["cases"]
    require(len(cases) == 6, "The report must contain three cases per member.")
    for name in rows:
        selected = [c for c in cases if c["member"] == name]
        require(len(selected) == 3 and len({c["sample_id"] for c in selected}) == 3, f"Three distinct cases required: {name}")
    for case in cases:
        require(case["excerpt"] in sources[case["member"]][case["sample_id"]], "A team-report excerpt differs from actual generation.")

    partner_metrics = load(partner / "outputs/metrics/evaluation_metrics.json")
    expected = {
        "training_cross_entropy_loss": partner_metrics["training"]["final_training_loss"],
        "validation_cross_entropy_loss": partner_metrics["validation"]["cross_entropy_loss"],
        "perplexity": partner_metrics["validation"]["perplexity"],
        "bits_per_character": partner_metrics["validation"]["bits_per_character"],
        "generalization_gap": partner_metrics["validation"]["generalization_gap"],
        "top1_next_character_accuracy": partner_metrics["validation"]["top1_accuracy"],
        "parameter_count": partner_metrics["model"]["parameter_count"],
        "generation_tokens_per_second": partner_metrics["generation"]["tokens_per_second"],
    }
    expected.update({k: partner_metrics["diversity"]["all_sampled_outputs"][k] for k in ("distinct_1", "distinct_2", "distinct_3", "repeated_4gram_rate")})
    expected.update({k: partner_metrics["training"][k] for k in ("gradient_norm_mean", "gradient_norm_max", "loss_spike_count", "nonfinite_loss_count", "nonfinite_gradient_count", "training_tokens_per_second", "total_training_seconds", "peak_gpu_allocated_mb", "peak_gpu_reserved_mb")})
    with (task / "comparison/anshika_metrics_report.csv").open(newline="", encoding="utf-8") as handle:
        derived = list(csv.DictReader(handle))
    require(len(derived) == len(expected) and {r["metric"] for r in derived} == set(expected), "Derived partner CSV coverage differs.")
    for row in derived:
        require(math.isclose(float(row["value"]), expected[row["metric"]], rel_tol=1e-12, abs_tol=1e-12), f"Derived partner metric differs: {row['metric']}")

    readme = (root / "README.md").read_text(encoding="utf-8")
    require("task1_llm/Viraat_Chaudhary/code/smoke_test.py" in readme, "Update the root README with the one-command smoke test.")
    require("task1_llm/comparison/task1_team_report.md" in readme, "Update the root README with the completed Task 1 report section.")
    print("PASS: frozen artifacts, both checkpoints, executed notebook, metrics, six real failure excerpts and README links verified.")
    print("This is a packaging/evidence check; it does not rerun GPU inference or attest to authorship, team approval or Git pushes.")
    if not (partner / "metrics_report.csv").exists():
        print("Partner item: Anshika should copy comparison/anshika_metrics_report.csv to her folder and commit it herself.")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[3])
