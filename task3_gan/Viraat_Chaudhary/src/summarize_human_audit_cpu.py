"""Summarize two completed real-rater forms using only Python's standard library.

From the repository root:
  python3 task3_gan/Viraat_Chaudhary/src/summarize_human_audit_cpu.py
No model loading, training, inference, or GPU is involved.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

CRITERIA = ("style", "content", "artifacts")


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_scores(path, expected_ids):
    rows = read_rows(path)
    found = {}
    for row in rows:
        audit_id = row.get("audit_id", "").strip()
        if audit_id not in expected_ids or audit_id in found:
            raise ValueError(f"{path.name}: unknown or duplicate audit ID {audit_id!r}")
        found[audit_id] = {}
        for criterion in CRITERIA:
            value = (row.get(criterion) or "").strip()
            if value not in {"1", "2", "3", "4", "5"}:
                raise ValueError(f"{path.name}: {audit_id} {criterion} needs an integer 1-5")
            found[audit_id][criterion] = int(value)
    if set(found) != set(expected_ids):
        raise ValueError(f"{path.name}: all 30 fixed audit IDs must be scored")
    return found


def agreement(first, second):
    if not first or len(first) != len(second):
        raise ValueError("Agreement requires equal-length, nonempty score vectors")
    count = len(first)
    observed = sum(x == y for x, y in zip(first, second)) / count
    a, b = Counter(first), Counter(second)
    expected = sum(a[k] * b[k] for k in set(a) | set(b)) / (count * count)
    kappa = None if expected == 1.0 else (observed - expected) / (1.0 - expected)
    return observed, kappa


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def update_metric_file(path, output, evidence):
    if not path.exists():
        return
    rows = read_rows(path)
    columns = list(rows[0])
    scores = {"human_style_score": "style", "human_content_score": "content",
              "human_artifacts_score": "artifacts"}
    for row in rows:
        metric = row["metric"]
        direction = row["direction"]
        if direction not in output or not metric.startswith("human_"):
            continue
        if metric in scores:
            value = output[direction][scores[metric]]["mean_score"]
        elif metric == "human_inter_rater_agreement":
            value = output[direction]["pooled"]["agreement_fraction"]
        elif metric == "human_cohens_kappa":
            value = output[direction]["pooled"]["cohens_kappa"]
        else:
            continue
        row["value"] = "" if value is None else str(value)
        row["status"] = "undefined_constant_ratings" if value is None else "computed_real_raters"
        row["evidence"] = evidence
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(member, run_id):
    folder = member / "outputs" / run_id / "human_audit"
    private = read_rows(folder / "private_manifest.csv")
    directions = {row["audit_id"]: row["direction"] for row in private}
    if len(private) != 30 or len(directions) != 30:
        raise ValueError("The private manifest must contain exactly 30 unique fixed audit IDs")
    if Counter(directions.values()) != Counter({"A2B": 15, "B2A": 15}):
        raise ValueError("The audit must retain 15 fixed samples in each direction")
    first = read_scores(folder / "rater1_scores.csv", directions)
    second = read_scores(folder / "rater2_scores.csv", directions)
    output = {}
    for direction in ("A2B", "B2A", "both"):
        ids = sorted(key for key in directions if direction == "both" or directions[key] == direction)
        result = {"sample_count": len(ids)}
        for criterion in CRITERIA:
            a = [first[key][criterion] for key in ids]
            b = [second[key][criterion] for key in ids]
            observed, kappa = agreement(a, b)
            result[criterion] = {
                "mean_score": statistics.mean(a + b),
                "std_score": statistics.stdev(a + b),
                "rater1_mean": statistics.mean(a), "rater2_mean": statistics.mean(b),
                "agreement_fraction": observed, "agreement_percent": observed * 100,
                "cohens_kappa": kappa,
            }
        a = [first[key][criterion] for key in ids for criterion in CRITERIA]
        b = [second[key][criterion] for key in ids for criterion in CRITERIA]
        observed, kappa = agreement(a, b)
        result["pooled"] = {
            "paired_judgment_count": len(a), "mean_score": statistics.mean(a + b),
            "agreement_fraction": observed, "agreement_percent": observed * 100,
            "cohens_kappa": kappa,
        }
        output[direction] = result
    atomic_json(folder / "human_audit_results.json", output)
    evidence = f"outputs/{run_id}/human_audit/human_audit_results.json"
    for name in ("metrics_report.csv", "full_metrics_report.csv"):
        update_metric_file(member / name, output, evidence)
    summary_path = member / "outputs" / run_id / "metrics" / "evaluation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["real_human_audit_completed"] = True
        summary["human_audit_evidence"] = evidence
        atomic_json(summary_path, summary)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="viraat_resizeconv6_run001")
    parser.add_argument("--member-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = summarize(args.member_root, args.run_id)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Audit not summarized: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
