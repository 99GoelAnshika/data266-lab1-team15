# Task 3 ? CycleGAN Image Style Transfer ? Anshika Goel

## Final Status

Task 3 is complete.

The final selected model is **epoch 125** of:

`cyclegan_baseline_rtx4090_run001`

Final checkpoint:

`checkpoints/cyclegan_baseline_rtx4090_run001/epoch_0125.pt`

Checkpoint SHA-256:

`3216b4997ed51962590251170ca5cbd6d00fb444479a8b5cd9480f0cfbeab7ec`

The checkpoint binary is retained locally because its size exceeds normal
GitHub blob limits. Its SHA-256 sidecar and checkpoint manifests are committed.

## Final Kaggle Result

- Team: `PairProgramming_Team_15`
- Final score: **-48.9365**
- Rank snapshot: **14**
- Previous best: **-49.2903**
- Kaggle displayed the epoch-125 submission as a **new personal best**.

The rank is a time-specific leaderboard snapshot.

## Official Instructor Evaluation

The unchanged instructor notebook produced:

- FID: **97.466148498**
- MiFID: **0.406865570**

Root submission:

`../../submission.csv`

Executed evaluator evidence:

- `outputs/evaluation/Part3_Evaluation_Script_executed.ipynb`
- `outputs/evaluation/Part3_Evaluation_Script_epoch125_executed.ipynb`
- `outputs/evaluation/part3_epoch125_evaluator.log`
- `outputs/evaluation/final_epoch125_submission_manifest.json`
- `outputs/evaluation/kaggle_epoch125_result.json`

## Final Automatic Metrics

| Metric | A2B / A2B2A | B2A / B2A2B | Mean |
|---|---:|---:|---:|
| FID | 96.068450 | 98.864013 | 97.466148 official |
| MiFID | 0.413548 | 0.400183 | 0.406866 official |
| KID | 0.017190 | 0.007900 | 0.012545 |
| Precision | 0.7800 | 0.4333 | 0.6067 |
| Recall | 0.3333 | 0.6300 | 0.4817 |
| Content cosine | 0.783451 | 0.754759 | 0.769105 |
| Cycle L1 | 0.032256 | 0.034622 | 0.033439 |
| Cycle LPIPS | 0.182873 | 0.134368 | 0.158620 |

## Training

- completed epochs: **125**
- final global step: **879750**
- final learning rate: **0.00015000**
- non-finite gradient events: **0**
- trainable parameters: **28285832**

Training evidence:

- `logs/cyclegan_baseline_rtx4090_run001/events.jsonl`
- `logs/cyclegan_baseline_rtx4090_run001/run_manifest.json`
- `logs/cyclegan_baseline_rtx4090_run001/improvement_epoch50_to75_training.log`
- `logs/cyclegan_baseline_rtx4090_run001/improvement_epoch75_to125_training.log`
- `outputs/metrics/training_epoch_metrics.csv`
- `outputs/metrics/training_run_summary.json`
- `outputs/plots/`

## Final Predictions

The direct epoch-125 CycleGAN outputs submitted to the evaluation pipeline are
preserved under:

`outputs/cyclegan_baseline_rtx4090_run001/official_epoch125_predictions/`

This directory contains:

- 300 Monet ? Photo predictions
- 300 Photo ? Monet predictions
- `inference_manifest.json`

No manually edited, copied, externally generated, hardcoded, lookup-table, or
test-pair-peeking outputs were used.

## Model

The implementation contains:

- two ResNet generators;
- nine residual blocks per generator;
- two PatchGAN discriminators;
- instance normalization;
- least-squares GAN loss;
- cycle-consistency L1 loss;
- identity L1 loss;
- Adam optimization.

Canonical configuration:

`configs/cyclegan_baseline.json`

Config SHA-256:

`4795c20ee2d0307d26659da578aa6e42834cd10def8ca1bd088285c0216bfae5`

## Reproducibility

Primary final evidence:

- `results.md`
- `failure_analysis.md`
- `full_metrics_report.csv`
- `outputs/metrics/automatic_metrics_report.csv`
- `outputs/evaluation/`
- `checkpoints/checkpoint_manifest.csv`
- `checkpoints/checkpoint_manifest.json`
- `outputs/improvement_checkpoint_comparison_epoch80_to125/checkpoint_comparison_summary.csv`

The earlier epoch-50/75 evaluation artifacts are retained where useful as
historical provenance. Final-model claims in this README refer to epoch 125.
