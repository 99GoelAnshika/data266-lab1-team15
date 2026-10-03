# Task 3 Final Results ? CycleGAN ? Anshika Goel

## 1. Final Model

The final selected production checkpoint is **epoch 125** at global step
**879750** from `cyclegan_baseline_rtx4090_run001`.

Checkpoint SHA-256:

`3216b4997ed51962590251170ca5cbd6d00fb444479a8b5cd9480f0cfbeab7ec`

The checkpoint was selected after deterministic local comparison of epochs
80, 85, 90, 95, 100, 105, 110, 115, 120 and 125, with epoch 50 retained as a
reproduction control.

The epoch-50 control reproduced its historical metrics, supporting consistency
of the comparison protocol.

## 2. Kaggle Result

The epoch-125 submission produced:

- team: `PairProgramming_Team_15`
- score: **-48.9365**
- leaderboard rank snapshot: **14**
- previous best score: **-49.2903**

Kaggle identified this submission as a **new personal best**.

## 3. Official FID / MiFID

The unchanged instructor evaluator produced:

| Metric | Final value |
|---|---:|
| FID | 97.466148498 |
| MiFID | 0.406865570 |

The direct comparison pipeline produced directional values:

| Metric | Monet ? Photo | Photo ? Monet | Mean |
|---|---:|---:|---:|
| FID | 96.068450 | 98.864013 | 97.466231 |
| MiFID | 0.413548 | 0.400183 | 0.406866 |
| KID | 0.017190 | 0.007900 | 0.012545 |
| Precision | 0.7800 | 0.4333 | 0.6067 |
| Recall | 0.3333 | 0.6300 | 0.4817 |

The very small difference between the comparison FID/MiFID mean and the
official notebook values is numerical implementation tolerance; the root
`submission.csv` contains the official instructor-evaluator result.

## 4. Cycle Consistency

| Metric | A2B2A | B2A2B | Mean |
|---|---:|---:|---:|
| Cycle L1 | 0.032256 | 0.034622 | 0.033439 |
| LPIPS AlexNet | 0.182873 ? 0.057226 | 0.134368 ? 0.042938 | 0.158620 |

LPIPS uses `lpips==0.1.4` with the AlexNet backbone on the same fixed 300
source images per direction.

## 5. Content Preservation

| Direction | Inception cosine similarity |
|---|---:|
| Monet ? Photo | 0.783451 |
| Photo ? Monet | 0.754759 |
| Mean | 0.769105 |

This is a feature-space proxy and should not be interpreted as a perfect
semantic-content measurement.

## 6. Training Stability

The durable event log contains a contiguous sequence from epoch 1 through
epoch 125.

- completed epochs: 125
- final global step: 879750
- training duration from durable epoch events: 27.8685 hours
- weighted throughput: 17.5377 images/sec
- non-finite gradient events: 0

Final epoch means:

- `grad_norm_discriminator_a`: 4.317097
- `grad_norm_discriminator_b`: 6.441369
- `grad_norm_generator`: 34.309808
- `loss_cycle_a`: 0.543312
- `loss_cycle_b`: 0.627509
- `loss_discriminator_a`: 0.055120
- `loss_discriminator_b`: 0.157722
- `loss_gan_a_to_b`: 0.475969
- `loss_gan_b_to_a`: 0.812843
- `loss_generator_total`: 2.811135
- `loss_identity_a`: 0.147671
- `loss_identity_b`: 0.203830


Training plots are preserved in `outputs/plots/`.

## 7. Final Prediction Provenance

The final prediction bundle is:

`outputs/cyclegan_baseline_rtx4090_run001/official_epoch125_predictions/`

It contains exactly 300 direct translations in each direction.

The source checkpoint, aggregate hashes, Kaggle result and file counts are
recorded in its `inference_manifest.json`.

## 8. Improvement Relative to Earlier Checkpoints

Epoch 125 improved the official FID from the epoch-75 value
`98.17002258187867` to `97.466148498`.

It also improved the official MiFID from
`0.41062747340586747` to `0.406865570`.

Kaggle score improved from `-49.2903` to
`-48.9365`.

The improvement was not uniform across every auxiliary metric. This is
documented in `failure_analysis.md`.

## 9. Audit Provenance

The repository retains the earlier evaluator-approved simulated/AI-assisted
30-sample audit artifacts for provenance.

Those simulated ratings were created before the final epoch-125 checkpoint
selection and are therefore explicitly treated as **historical audit evidence**,
not as newly collected human ratings of the epoch-125 model.

No claim is made that those scores were produced by external human raters.

## 10. Final Evidence Map

- `../../submission.csv`
- `outputs/evaluation/Part3_Evaluation_Script_executed.ipynb`
- `outputs/evaluation/Part3_Evaluation_Script_epoch125_executed.ipynb`
- `outputs/evaluation/final_epoch125_submission_manifest.json`
- `outputs/evaluation/kaggle_epoch125_result.json`
- `outputs/metrics/`
- `outputs/plots/`
- `outputs/cyclegan_baseline_rtx4090_run001/official_epoch125_predictions/`
- `outputs/improvement_checkpoint_comparison_epoch80_to125/`
- `checkpoints/checkpoint_manifest.csv`
- `checkpoints/checkpoint_manifest.json`
