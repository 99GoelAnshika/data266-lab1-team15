# Reviewed Task 3 repository selection

This preparation contains **757 files** in `task3_gan/Viraat_Chaudhary`.
It preserves the production model, all predictions, training/evaluation evidence
and necessary source. The original 1,951-file verified backup remains untouched.

| Deliverable | Included evidence |
| --- | --- |
| Source and notebooks | All original training/inference/evaluation modules; four notebooks with captured outputs where available; CPU review, audit, workspace and verification helpers |
| Distinct architecture/configuration | Six-block nearest resize-convolution generators, two PatchGANs, canonical config, code provenance and completed-run identities |
| Trained weights | `checkpoints/viraat_resizeconv6_run001/latest.pt` and checksum; actual weights require Git LFS |
| Direct generated outputs | All 300 A2B and all 300 B2A JPEG predictions plus inference manifest |
| Training record | All unedited raw logs, runtime/hardware/run manifests and all 60 sample panels |
| Automatic evaluation | Both metric CSVs, official and local CSVs, per-sample results, saved features, four plots, summaries and instructor notebook outputs |
| Failure analysis | Existing six-case analysis and fixed panel links |
| Human audit preparation | All 30 fixed anonymous panels, two blank forms, private manifest and instructions; real ratings still pending |
| Current team comparison | Updated epoch-125 report/config snapshots, byte hashes, side-by-side metrics and limitations |
| Reproduction | Member README, smoke test, byte verification and tested evaluator workspace reconstruction |

## What is omitted

Exactly 1,200 evaluator image copies are omitted: 600 raw evaluation inputs already
provided by shared `task3_gan/data`, and 600 duplicates of retained predictions.
`src/prepare_evaluator_workspace.py` rebuilds these copies after checking hashes;
all 1,200 reconstructed files were compared successfully to the saved originals.
The small saved evaluator `submission.csv` is retained. No direct prediction,
checkpoint, raw log or progress panel is removed.

Two superseded epoch-50 comparison snapshots are replaced by the supplied current
report/results/configuration snapshots. The current provenance JSON replaces the
old comparison provenance. Historical copies remain in the full completion backup.
OS metadata and Python caches are excluded. Backup ZIPs, the copy-preparation
utility and its full source-reference manifest stay outside the Git repository.

The checkpoint is 293,948,004 bytes (280.33 MiB).
Other retained original/update files total approximately
108.00 MiB, plus this document and the small
repository manifest. File count is driven mainly by the 600 required direct
predictions, not cache files.

## Verification scope and remaining requirements

`repository_manifest.json` records hashes for every prepared file except itself.
`src/verify_saved_files.py` checks them and ignores only regenerated evaluator
copies/OS caches. The outside preparation tool also verifies the manifest file's
own hash. Later real-rating/metric updates must be reviewed as legitimate changes.

Byte identity verifies the saved evidence, not model quality, authorship, live
leaderboard rank or real-rater provenance. This environment did not rerun the
PyTorch model; saved executed evaluator outputs and earlier validation are retained.

Before final submission, collect two actual independent real-rater forms, calculate
means/agreement, record Viraat's actual Kaggle result, check all final teammate
evidence, obtain missing current efficiency/package evidence where available,
finish the team root README and `report/DATA266_Lab1_Report_Team_15.pdf`, and verify
staged paths plus accessible LFS weights. This package is prepared evidence, not a
claim that every grading requirement is already complete.
