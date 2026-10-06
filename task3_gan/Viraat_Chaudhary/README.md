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

The required real-human audit has now been completed using the fixed set of
30 anonymous panels: 15 Monet -> Photo and 15 Photo -> Monet. Two independent
human raters scored every panel on three integer 1-5 scales: style quality,
content preservation and artifact-free quality. Model identity was hidden behind
anonymous Set_A / Set_B assignments until both completed responses had been
returned and validated.

### Final overall human-audit results

| Criterion | Two-rater mean | Exact agreement | Cohen's kappa |
|---|---:|---:|---:|
| Style quality | 3.2500 | 40.00% | 0.2275 |
| Content preservation | 3.7500 | 43.33% | 0.2630 |
| Artifact-free quality | 3.6833 | 36.67% | 0.1775 |

Pooled across all three dimensions, exact agreement was **40.00%** and
unweighted Cohen's kappa was **0.2357**.

Direction-specific two-rater means were:

| Direction | Style | Content | Artifact-free |
|---|---:|---:|---:|
| Monet -> Photo | 2.8667 | 3.5000 | 3.5000 |
| Photo -> Monet | 3.6333 | 4.0000 | 3.8667 |

Permanent real-human evidence is stored under
`outputs/viraat_resizeconv6_run001/human_audit/real_human_audit/`.

Team-level summary, agreement and provenance files are stored under
`../human_audit_analysis/`.

The original blank rating forms and historical audit infrastructure remain
preserved for provenance. They are not substituted for the completed real-human
evidence.

## Remaining submission work

The real-human audit requirement is complete. Historical Kaggle and run
artifacts remain unchanged; historical logs are not rewritten to represent
events that occurred after those artifacts were captured. The combined report
and final Git state still require final review before submission.

## Reference

Zhu et al. (2017), *Unpaired Image-to-Image Translation using Cycle-Consistent
Adversarial Networks*, ICCV. https://arxiv.org/abs/1703.10593
