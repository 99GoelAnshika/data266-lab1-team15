# Task 1 - reviewed merge and eight logical commits

The training/evaluation is already complete. Use **Viraat_Task1_Reviewed.zip**
for the final merge; it includes the executed notebook, current source, real
checkpoints/logs/metrics and shared comparison. Earlier source/fix ZIPs need no
separate overlay. Anshika's original source/weights and Tasks 2/3 are excluded. The only added
partner file is the missing metrics CSV derived from her existing evaluation JSON.

The PDF requests clear, frequent commits but gives no numeric minimum.
These eight groups reflect the user's request for multiple meaningful commits.
They organize completed work honestly; do not backdate commits or pretend
the grouping is an earlier experiment timeline.

## 1. Merge from your existing repository root

Keep your original downloaded run ZIP/notebook as backups. Open Terminal at
the root of your existing `data266-lab1-team15` clone. Place the reviewed ZIP
one directory above that root. Check the current files/identity/branch:

```bash
pwd
git status --short
git branch --show-current
git config user.name
git config user.email
git diff --cached --name-only
```

Use your own Git identity and the branch agreed with your teammate. Inspect any
existing local changes before overlaying the package; keep intended changes.
If the clone is clean before merging, update it with `git pull --ff-only`.
If existing changes are present, preserve and review them; do not reset or
throw away work. Start each commit group with no unrelated staged changes.
Use your own repository-local Git name and GitHub-verified email; if necessary,
set `git config user.name "Viraat Chaudhary"` and set `user.email` to your actual
verified address. Do not guess an email or use your teammate's identity.

The archive paths start with `task1_llm/`. Extract at the repo root: this
merges files without deleting Anshika's existing folder. Do not create a
nested `task1_llm/task1_llm` or replace Anshika's whole folder in Finder:

```bash
unzip -o ../Viraat_Task1_Reviewed.zip
python3 task1_llm/Viraat_Chaudhary/code/finalize_artifacts.py
python3 task1_llm/Viraat_Chaudhary/code/update_root_readme.py
python3 task1_llm/Viraat_Chaudhary/code/verify_submission.py
git status --short
```

All three Python helpers use the standard library. No GPU or new training is needed for this handoff. Finalization validates the
downloaded notebook and frozen checkpoints, refreshes hashes, and retains
observed Colab versions. The README helper updates only Task 1/member status
and its own section, retaining unrelated team content. No separate notebook
replacement is needed: the reviewed ZIP already includes the executed copy.

Do not retrain merely to create commits. Keep every raw log unchanged.
If a later intentional smoke test writes a new log, rerun finalization before
staging its evidence. No dataset/cache or earlier ZIP should be committed.

## 2. Confirm the upstream before the first push

```bash
git rev-parse --abbrev-ref --symbolic-full-name '@{u}'
```

If it shows the agreed upstream, use `git push` after each commit below.
If no upstream is set, first identify the real branch name with
`git branch --show-current`, then use `git push -u origin ACTUAL_BRANCH_NAME`
for the first push, substituting the actual agreed branch. Do not type the
placeholder literally. Subsequent pushes use `git push`.

The provided GitHub screenshot shows `main`. If your local branch is also
`main` and has no upstream, the first push is `git push -u origin main`;
subsequent pushes use `git push`. For another agreed branch, use its real name.
If any command fails, stop that sequence and inspect its error. If a group
has no changes because it was already committed, skip it; do not invent an
empty commit or pretend these reporting-time commits are earlier experiments.

Read your results/failure analysis and `comparison/task1_team_report.md` before
committing; correct statements you cannot personally support, then rerun
finalization and the verifier. No unrecorded teammate approval is claimed and
no additional approval gate is imposed by this guide.

Run every command below from the repository root. Stage only each named group;
do not use `git add .`. Inspect the staged list before every commit.

## Commit 1 - Configuration and setup

```bash
git add task1_llm/.gitattributes task1_llm/Viraat_Chaudhary/configs task1_llm/Viraat_Chaudhary/requirements.txt task1_llm/Viraat_Chaudhary/.gitignore task1_llm/Viraat_Chaudhary/.gitattributes task1_llm/Viraat_Chaudhary/README.md task1_llm/Viraat_Chaudhary/RUN_GUIDE.md task1_llm/Viraat_Chaudhary/REPO_REVIEW.md task1_llm/Viraat_Chaudhary/code/update_root_readme.py
git diff --cached --name-only
git diff --cached --stat
git commit -m "task1: add Viraat experiment configuration and setup"
git push
```

## Commit 2 - Character preprocessing and frozen-data restoration

```bash
git add task1_llm/Viraat_Chaudhary/code/common.py task1_llm/Viraat_Chaudhary/code/data.py
git diff --cached --name-only
git diff --cached --stat
git commit -m "task1: implement Viraat character preprocessing and frozen split restoration"
git push
```

## Commit 3 - Separate scratch decoder architecture

```bash
git add task1_llm/Viraat_Chaudhary/code/model.py
git diff --cached --name-only
git diff --cached --stat
git commit -m "task1: implement Viraat post-norm causal decoder from scratch"
git push
```

## Commit 4 - Training, scheduling and resumable checkpoints

```bash
git add task1_llm/Viraat_Chaudhary/code/train.py
git diff --cached --name-only
git diff --cached --stat
git commit -m "task1: add full-epoch training scheduling and checkpoint resume"
git push
```

## Commit 5 - Evaluation and reporting tools

```bash
git add task1_llm/Viraat_Chaudhary/code/evaluate_generate.py task1_llm/Viraat_Chaudhary/code/report.py task1_llm/Viraat_Chaudhary/code/finalize_artifacts.py task1_llm/Viraat_Chaudhary/analysis task1_llm/compare_task1.py task1_llm/evaluate_common.py
git diff --cached --name-only
git diff --cached --stat
git commit -m "task1: add required metrics generation and team comparison tools"
git push
```

## Commit 6 - Correctness checks and authentic executed notebook

```bash
git add task1_llm/Viraat_Chaudhary/code/tests task1_llm/Viraat_Chaudhary/code/smoke_test.py task1_llm/Viraat_Chaudhary/code/task1_colab.ipynb task1_llm/Viraat_Chaudhary/VERIFICATION.md task1_llm/Viraat_Chaudhary/code/verify_submission.py
git diff --cached --name-only
git diff --cached --stat
git commit -m "task1: add correctness checks and executed Colab evidence"
git push
```

## Commit 7 - Actual checkpoints, logs and measured evidence

```bash
git add task1_llm/Viraat_Chaudhary/checkpoints task1_llm/Viraat_Chaudhary/logs task1_llm/Viraat_Chaudhary/outputs task1_llm/Viraat_Chaudhary/metrics_report.csv task1_llm/Viraat_Chaudhary/environment_manifest.txt task1_llm/Viraat_Chaudhary/requirements.lock.txt task1_llm/Viraat_Chaudhary/artifact_manifest.json
git diff --cached --name-only
git diff --cached --stat
git commit -m "task1: record Viraat trained checkpoints logs and full metrics"
git push
```

## Commit 8 - Results, failures and best-model comparison

```bash
git add task1_llm/Viraat_Chaudhary/results.md task1_llm/Viraat_Chaudhary/failure_analysis.md README.md task1_llm/comparison task1_llm/Anshika_Goel/metrics_report.csv
git diff --cached --name-only
git diff --cached --stat
git commit -m "task1: document results failures and common-holdout model selection"
git push
```

## 3. Verify the final push

```bash
python3 task1_llm/Viraat_Chaudhary/code/verify_submission.py
git status --short
git log -8 --oneline
git rev-list --left-right --count '@{u}...HEAD'
git ls-files task1_llm/Viraat_Chaudhary/data_processed task1_llm/Viraat_Chaudhary/hf_cache
```

Expected: no remaining intended changes; the eight actual commits are visible;
ahead/behind counts are `0 0`; no processed arrays/caches are listed. Confirm
the named folder, executed notebook and comparison files on GitHub.
These checks require your actual local clone and push; the export review does
not claim to have created commits or accessed the private remote.

The completed `task1_llm/comparison/task1_team_report.md` contains the full
architecture/hyperparameter/all-metric table, common model selection, six real
failure snippets, limitations, evidence links and proposed next experiments.
It is ready to incorporate into the eventual combined PDF; Tasks 2/3 and the
actual members' personal review/viva remain separate responsibilities.

Commit 8 adds Anshika's missing metrics CSV from her existing JSON. This is a
packaging correction, not a change to her source/model/weights/logs or an
assignment of her experiment to Viraat. No Task 2/3 paths are staged here.
