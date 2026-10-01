"""Update only Viraat/Task 1 status while retaining the team's other README text."""

import json
import re
from pathlib import Path


def update(root):
    readme = root / "README.md"
    if not readme.exists():
        raise FileNotFoundError("Use the actual team repository containing README.md.")
    member = root / "task1_llm/Viraat_Chaudhary"
    metrics = json.loads((member / "outputs/metrics/evaluation_metrics.json").read_text())
    if metrics["smoke_test"] or not metrics["training"]["completed"]:
        raise ValueError("Complete the real member run before updating team status.")
    manifest_path = root / "task1_llm/comparison/common_evaluation_manifest.json"
    common = json.loads(manifest_path.read_text())
    records = {r["member"]: r for r in common["results"]}
    if set(records) != {"Anshika_Goel", "Viraat_Chaudhary"}:
        raise ValueError("Both common comparison rows are required.")
    a, v = records["Anshika_Goel"], records["Viraat_Chaudhary"]
    section = f"""## Viraat - Task 1 Colab workflow

Both members' Task 1 training/evaluation artifacts are present. Viraat completed
ten full epochs on an NVIDIA A100-SXM4-40GB using a separate post-norm/ReLU
decoder with an untied output head. His
[member README](task1_llm/Viraat_Chaudhary/README.md),
[results](task1_llm/Viraat_Chaudhary/results.md),
[all metrics](task1_llm/Viraat_Chaudhary/metrics_report.csv),
[failure analysis](task1_llm/Viraat_Chaudhary/failure_analysis.md) and
[executed notebook](task1_llm/Viraat_Chaudhary/code/task1_colab.ipynb)
provide the individual evidence.

Install the member requirements in an appropriate PyTorch environment:

```bash
python -m pip install -r task1_llm/Viraat_Chaudhary/requirements.txt
```

The grader's one-command synthetic smoke check is:

```bash
python task1_llm/Viraat_Chaudhary/code/smoke_test.py
```

The smoke test needs no TinyStories download. To restore the omitted processed
arrays from the frozen dataset and verify their recorded hashes:

```bash
python task1_llm/Viraat_Chaudhary/code/data.py
```

After restoration, `python task1_llm/Viraat_Chaudhary/code/train.py --resume`
verifies the completed checkpoint and reports `ALREADY_COMPLETE`. For a new
training experiment, use a separate clean run directory as explained in the
member README; retain the original logs and weights.

The [recorded-results table](task1_llm/comparison/task1_comparison.md) and
[common held-out selection](task1_llm/comparison/best_model_selection.md)
retain both members. The [Task 1 report section](task1_llm/comparison/task1_team_report.md)
contains the complete comparison, evidence links, both members' real failure
examples and proposed next experiments. It does not attest to another member's approval.
On the common official validation text, Anshika has CE
{a['common_validation_ce']:.8f} and accuracy
{100*a['common_next_character_accuracy']:.4f}%; Viraat has CE
{v['common_validation_ce']:.8f} and accuracy
{100*v['common_next_character_accuracy']:.4f}%. Use the documented common quality
criteria when choosing the model; different-GPU training speeds cannot isolate
architecture effects.

Regenerate the shared evaluations from the complete team repository:

```bash
python task1_llm/compare_task1.py
python task1_llm/evaluate_common.py
```

The [run/commit guide](task1_llm/Viraat_Chaudhary/RUN_GUIDE.md) gives eight
logical commits from the repository root. Task 1 completion does not complete
the remaining individual tasks, the combined Tasks 1-3 PDF, or the viva.
"""
    text = readme.read_text(encoding="utf-8")
    old_status = "| Task 1 - GPT-style LLM from scratch | Anshika implementation, evidence, `results.md`, and `failure_analysis.md` complete |"
    new_status = "| Task 1 - GPT-style LLM from scratch | Both recorded runs and common comparison present; Task 1 report section prepared; member understanding and final team PDF remain |"
    text = text.replace(old_status, new_status)
    previous_status = "| Task 1 - GPT-style LLM from scratch | Both members' training/evaluation, individual analyses and common held-out comparison complete; see the Task 1 member section below |"
    text = text.replace(previous_status, new_status)
    old_members = "At the current `main` commit, the tracked Task 1 and Task 2 implementation folders contain `Anshika_Goel`; the remaining teammate/task deliverables must be added before final submission."
    new_members = "Task 1 includes both `Anshika_Goel` and `Viraat_Chaudhary` with independent configurations and evidence. Other remaining individual/task deliverables must still be added before the combined lab submission."
    text = text.replace(old_members, new_members)
    tree_anchor = "    results.md\ntask2_sentiment/"
    tree_update = """    results.md
  Viraat_Chaudhary/
    code/task1_colab.ipynb                # exact executed Colab notebook
    configs/
    checkpoints/
    logs/
    outputs/
    metrics_report.csv
    failure_analysis.md
    results.md
  compare_task1.py
  evaluate_common.py
  comparison/                            # both frozen models and selection
task2_sentiment/"""
    text = text.replace(tree_anchor, tree_update, 1)
    marker = "## Viraat - Task 1 Colab workflow"
    if marker in text:
        text = re.sub(
            r"(?ms)^## Viraat - Task 1 Colab workflow\n.*?(?=^## |\Z)",
            lambda _: section.rstrip() + "\n\n", text,
        )
    else:
        text = text.rstrip() + "\n\n" + section
    readme.write_text(text.rstrip() + "\n", encoding="utf-8")
    print("Updated Viraat/Task 1 README status; retained the team's other sections.")


if __name__ == "__main__":
    update(Path(__file__).resolve().parents[3])
