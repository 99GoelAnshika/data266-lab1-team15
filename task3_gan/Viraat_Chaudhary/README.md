# Viraat Chaudhary — Task 3, Team 15

The six-block resize-convolution CycleGAN completed 60 epochs on an RTX 4090.
This member folder retains the trained checkpoint, all 600 direct predictions,
all raw logs, all 60 training sample panels, 30 fixed anonymous audit panels,
executed evaluation notebooks, metrics, plots, configuration and analysis.
See [results.md](results.md), [failure_analysis.md](failure_analysis.md),
[team_comparison.md](team_comparison.md), and [file selection](REPOSITORY_SELECTION.md).
Shared code provenance is recorded in `configs/code_provenance.json`.

## Data, checkpoint and environment

Use the team's shared raw dataset once at `task3_gan/data/monet_jpg` (300 images)
and `task3_gan/data/photo_jpg` (7,038 images). Keep the existing dataset retrieval
instructions in the root README; this member preparation does not upload or
replace the raw dataset. `configs/evaluator_input_manifest.json` records SHA256
hashes of the fixed 300 evaluation inputs per domain.

The final checkpoint is `checkpoints/viraat_resizeconv6_run001/latest.pt`, SHA256
`138f9ae9edef412125ddfad643d2fc672c3de2d609819caf0b623979d3765252`.
It is approximately 294 MB and must be retrieved through Git LFS after checkout.
Before committing it, configure LFS for this exact path and explicitly force-add
it because the member `.gitignore` excludes `.pt` files. Check the LFS pointer
and staged paths before pushing. The actual weights must remain accessible to
the grader; a checksum alone is insufficient.

`logs/viraat_resizeconv6_run001/run_manifest.json` records the production hardware,
Python 3.10.13, PyTorch 2.1.2, CUDA 12.1 and cuDNN 8902.
`configs/requirements_task3.txt` records evaluation dependencies; notebook outputs
also retain the observed package-install evidence. Preserve a compatible
PyTorch/torchvision environment. This preparation does not install packages or
claim a complete frozen package environment from information not captured.

## CPU review and smoke test

From the repository root, verify this preparation snapshot with Python's
standard library:

```bash
python3 task3_gan/Viraat_Chaudhary/src/verify_saved_files.py
```

The verifier checks every recorded file hash, including the actual checkpoint
and all predictions. It does not run the model. Later completed human forms or
metric updates legitimately change this snapshot and must be reviewed separately.
`src/03_review_completed_run.ipynb` reads the saved evidence without training.

With compatible PyTorch/torchvision installed, the single-command CPU smoke test is:

```bash
python task3_gan/Viraat_Chaudhary/src/smoke_check.py
```

It uses temporary synthetic data and reduced channels to check the optimizer,
checkpoint and resume path. Historical CPU test evidence is retained unchanged
in `VALIDATION.txt`; this preparation did not rerun model inference or that test.

## Instructor evaluation workspace

The repository avoids 1,200 duplicate evaluator image copies. Before rerunning
the saved instructor notebook, recreate those inputs from the existing shared
raw data and retained predictions:

```bash
python3 task3_gan/Viraat_Chaudhary/src/prepare_evaluator_workspace.py
```

This checks the first 300 sorted raw inputs and all 600 prediction hashes before
copying unchanged image bytes. It performs no inference, training, image editing,
checkpoint selection or metric calculation. Existing differing inputs are not
overwritten. The rebuilt directory is ignored by Git. Its 1,200 files were tested
against the original saved evaluator workspace and matched byte-for-byte.

Set the notebook kernel's working directory to
`task3_gan/Viraat_Chaudhary/outputs/viraat_resizeconv6_run001/evaluator_workspace`,
then run `src/Part3_Evaluation_Script.ipynb` from that working directory.
For example, add a temporary setup cell specifying that working directory before
rerunning; the saved metric cells remain unchanged. The archived instructor
notebook's `%pip pandas` install is historical environment setup, not a new
mandatory package replacement. Preserve the captured outputs for provenance.

No evaluation rerun is needed to use the existing `submission.csv`.
The member-root CSV is byte-identical to the saved evaluator CSV; its FID/MiFID
already average both directions. Submit it unchanged, without another division
or sign change. The saved notebooks retain their actual executed outputs.

## Real human audit

Use the separate anonymous `Audit_Rater_1.zip` and `Audit_Rater_2.zip` packets
from the completion pack for two independent real raters. Share only each
anonymous packet, not the author-specific README or private direction manifest.
Each packet contains all 30 unchanged panels and one blank `ratings.csv`.

After receiving both completed forms, copy them to
`outputs/viraat_resizeconv6_run001/human_audit/rater1_scores.csv` and
`rater2_scores.csv`, then run from the repository root:

```bash
python3 task3_gan/Viraat_Chaudhary/src/summarize_human_audit_cpu.py
```

The CPU-only script validates all 30 IDs and 1–5 integer scores, pairs responses
by ID and calculates criterion/direction means, exact agreement and unweighted
Cohen's kappa. It updates the metric reports/status. Constant-rating kappa is
undefined rather than fabricated. The currently blank forms are not completed
evidence. The script's validation does not establish real-rater provenance.

## Remaining submission work

Viraat's real human scores/agreement and actual Kaggle result/rank remain pending.
Anshika's supplied final report retains simulated epoch-50 ratings as historical
provenance; these do not establish a real human audit of her final epoch-125 model.
Her current KID variability and peak allocated memory are not in the supplied
three-file evidence archive. The team's root README, full environment capture,
combined PDF, checkpoint accessibility and final Git/LFS state need repository
review before declaring the submission complete.

## Reference

Zhu et al. (2017), *Unpaired Image-to-Image Translation using Cycle-Consistent
Adversarial Networks*, ICCV. https://arxiv.org/abs/1703.10593
