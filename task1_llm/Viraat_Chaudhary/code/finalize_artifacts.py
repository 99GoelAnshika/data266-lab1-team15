"""Refresh hashes after saving the executed notebook; preserve observed Colab versions."""

import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finalize(root):
    metrics = json.loads(
        (root / "outputs/metrics/evaluation_metrics.json").read_text(encoding="utf-8")
    )
    if metrics.get("smoke_test") or not metrics["training"]["completed"]:
        raise ValueError("Finalization requires completed real training/evaluation.")
    summary = json.loads(
        (root / "outputs/metrics/training_summary.json").read_text(encoding="utf-8")
    )
    if summary["epochs_completed"] < 10 or not summary["completed"] or summary["smoke_test"]:
        raise ValueError("At least ten complete real epochs are required.")
    for relative, expected in (
        (metrics["checkpoint"]["path"], metrics["checkpoint"]["sha256"]),
        ("checkpoints/last_checkpoint.pt", summary["last_checkpoint_sha256"]),
        ("configs/gpt_char.yaml", summary["config_sha256"]),
    ):
        if digest(root / relative) != expected:
            raise ValueError(f"Frozen checkpoint/configuration differs: {relative}")
    cases = json.loads((root / "analysis/failure_cases.json").read_text(encoding="utf-8"))["cases"]
    samples = json.loads(
        (root / "outputs/samples/generated_samples.json").read_text(encoding="utf-8")
    )["samples"]
    lookup = {s["sample_id"]: s["continuation"] for s in samples}
    if len(cases) != 3 or len({c.get("sample_id") for c in cases}) != 3:
        raise ValueError("Three different real failure cases are required.")
    for case in cases:
        if any(
            not isinstance(case.get(k), str) or not case[k].strip()
            for k in ("sample_id", "excerpt", "failure_type", "observation", "testable_fix")
        ):
            raise ValueError("Failure write-up is incomplete.")
        if case["sample_id"] not in lookup or case["excerpt"] not in lookup[case["sample_id"]]:
            raise ValueError("An excerpt differs from the actual saved generation.")
    notes = json.loads((root / "analysis/interpretation.json").read_text(encoding="utf-8"))
    if any(
        not isinstance(notes.get(k), str) or not notes[k].strip()
        for k in (
            "architecture_justification",
            "hyperparameter_justification",
            "observations",
            "limitations_and_next_steps",
        )
    ):
        raise ValueError("Complete the reviewed design/results interpretation.")
    notebook = json.loads((root / "code/task1_colab.ipynb").read_text(encoding="utf-8"))
    if any(
        o.get("output_type") == "error" for c in notebook["cells"] for o in c.get("outputs", [])
    ):
        raise ValueError(
            "The saved notebook contains error outputs; retain debugging logs and save the clean verified run."
        )
    if not any(
        c.get("execution_count") is not None and c.get("outputs")
        for c in notebook["cells"]
        if c["cell_type"] == "code"
    ):
        raise ValueError("Save the executed Colab notebook with outputs before finalizing.")
    excluded = {"data_processed", "hf_cache", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and not (set(p.relative_to(root).parts) & excluded)
        and p.name not in {"artifact_manifest.json", "environment_manifest.txt"}
        and not p.name.endswith(".tmp")
    ]
    hashes = {p.relative_to(root).as_posix(): digest(p) for p in sorted(files)}
    artifact = {
        "member": "Viraat Chaudhary",
        "team": 15,
        "checkpoint": metrics["checkpoint"],
        "files": hashes,
        "failure_analysis_present": True,
        "design_interpretation_present": True,
        "personal_review_status": "not_attested_by_automated_check",
        "executed_notebook_present": True,
        "verification_scope": "Completeness and evidence consistency; these flags do not attest to personal authorship or understanding.",
    }
    (root / "artifact_manifest.json").write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "DATA 266 Lab 1 - Viraat Chaudhary - Team 15",
        "Observed evaluation environment (retained from Colab):",
        json.dumps(metrics["environment"], indent=2),
        "Artifact/result mapping and SHA256:",
    ]
    lines += [f"{name}: {value}" for name, value in hashes.items()]
    (root / "environment_manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "Executed notebook and written evidence validated; artifact hashes refreshed. Colab package versions retained."
    )


if __name__ == "__main__":
    finalize(Path(__file__).resolve().parents[1])
