# Task 3 — CycleGAN Image Style Transfer — Anshika Goel

## Status

Task 3 is complete.

A custom CycleGAN was implemented and trained for unpaired bidirectional image
translation between:

- Monet paintings → photographs (`A2B`)
- photographs → Monet-style paintings (`B2A`)

The final reported experiment is:

`cyclegan_baseline_rtx4090_run001`

The preserved evaluated checkpoint is epoch 50 at global step 351900.

---

## Model

The implementation uses the standard CycleGAN structure:

- two generators:
  - `G_A2B`
  - `G_B2A`
- two PatchGAN discriminators:
  - `D_A`
  - `D_B`

Generator configuration:

- ResNet generator
- 9 residual blocks
- base channels: 64
- instance normalization
- tanh output

Discriminator configuration:

- 70×70 PatchGAN
- base channels: 64
- 3 discriminator layers
- instance normalization

Losses:

- least-squares GAN objective
- cycle-consistency L1 loss
- identity L1 loss

Canonical configuration:

`configs/cyclegan_baseline.json`

Config SHA-256:

`4795c20ee2d0307d26659da578aa6e42834cd10def8ca1bd088285c0216bfae5`

---

## Training

Production training hardware:

- NVIDIA GeForce RTX 4090
- 24 GB VRAM
- Python 3.10.13
- PyTorch 2.1.2
- torchvision 0.16.2
- CUDA runtime 12.1

Training configuration:

- image size: 256×256
- training resize: 286
- random crop: 256
- horizontal flip probability: 0.5
- batch size: 1
- optimizer: Adam
- learning rate: 0.0002
- beta1: 0.5
- beta2: 0.999
- image replay pool: 50
- cycle-loss weight: 10
- effective identity-loss weight: 5

The canonical configuration defines a longer schedule, but the preserved
production experiment was stopped and evaluated at epoch 50.

Final production-run statistics:

- completed epochs: 50
- final global step: 351900
- training duration: approximately 11.01 hours
- total images processed: 703800
- effective throughput: approximately 17.76 images/second
- peak CUDA memory recorded by the training process: approximately 2542 MiB
- non-finite gradient events: 0

Detailed training evidence:

- `logs/cyclegan_baseline_rtx4090_run001/events.jsonl`
- `outputs/metrics/training_epoch_metrics.csv`
- `outputs/metrics/training_run_summary.json`

---

## Final Checkpoint

The evaluated checkpoint is:

`checkpoints/cyclegan_baseline_rtx4090_run001/epoch_0050.pt`

SHA-256:

`a66c5beafb3037167df5d34b72c39dea0b3b1d282f564dd7680cecc7585f592b`

Checkpoint binaries are intentionally retained locally rather than committed to
Git because each checkpoint is approximately 399 MiB and exceeds normal GitHub
blob limits.

All 11 checkpoint identities, byte sizes, and SHA-256 hashes are preserved in:

- `checkpoints/checkpoint_manifest.csv`
- `checkpoints/checkpoint_manifest.json`

The checkpoint binaries themselves have not been deleted.

---

## Frozen Evaluation Predictions

Official frozen epoch-50 predictions are stored under:

`outputs/cyclegan_baseline_rtx4090_run001/official_epoch50_predictions/`

with:

- `pred_A2B/` — 300 Monet → Photo translations
- `pred_B2A/` — 300 Photo → Monet translations

These predictions were generated directly by the trained CycleGAN and were
reused for the reported quantitative evaluations.

No hand-picked replacement images, manual image edits, external generators,
lookup-table outputs, or test-pair peeking were used.

---

## Main Quantitative Results

### Official FID / MiFID

Using the unchanged instructor evaluation notebook:

| Direction | FID | MiFID |
|---|---:|---:|
| Photo → Monet | 99.0412 | 0.406675 |
| Monet → Photo | 100.3974 | 0.418696 |
| Official average | 99.7193 | 0.412685 |

The root Kaggle submission file is:

`../../submission.csv`

Recorded Kaggle result at the time of submission:

- team: `PairProgramming_Team_15`
- rank snapshot: 12
- score: `-50.0659`

Leaderboard rank is a time-dependent snapshot and may change as additional
submissions are made.

### KID

- Photo → Monet: 0.006703
- Monet → Photo: 0.016918
- mean: 0.011810

### Generative Precision / Recall

Photo → Monet:

- precision: 0.4267
- recall: 0.6433

Monet → Photo:

- precision: 0.6467
- recall: 0.3300

### Cycle-Reconstruction L1

Mean L1 in RGB [0,1] space:

- A2B2A: 0.034629
- B2A2B: 0.045006
- mean: 0.039817

### Cycle LPIPS

AlexNet LPIPS:

- A2B2A: 0.209752
- B2A2B: 0.171835
- mean: 0.190794

### Content-Preservation Cosine Similarity

Inception feature cosine similarity between source image and direct translation:

- Monet → Photo: 0.835613
- Photo → Monet: 0.778093
- mean: 0.806853

The consolidated metric tables are:

- `full_metrics_report.csv`
- `outputs/metrics/automatic_metrics_report.csv`

---

## Training Stability Evidence

Final-epoch mean losses include:

- generator total loss: 3.149617
- GAN A2B: 0.491896
- GAN B2A: 0.649156
- cycle A: 0.638300
- cycle B: 0.842586
- identity A: 0.219875
- identity B: 0.307804
- discriminator A: 0.097711
- discriminator B: 0.150821

Gradient monitoring recorded:

- generator gradient norm: 23.407706
- discriminator-A gradient norm: 5.694744
- discriminator-B gradient norm: 6.954282
- non-finite gradient events: 0

Plots:

- `outputs/plots/generator_losses_by_epoch.png`
- `outputs/plots/discriminator_losses_by_epoch.png`
- `outputs/plots/cycle_consistency_losses_by_epoch.png`
- `outputs/plots/identity_losses_by_epoch.png`
- `outputs/plots/gradient_norms_by_epoch.png`

---

## Failure Analysis

Six deterministic metric-selected failure cases were analyzed:

- 3 Monet → Photo
- 3 Photo → Monet

Selection combined within-direction percentiles for:

- low content cosine similarity;
- high cycle L1;
- high cycle LPIPS.

The analysis demonstrates several failure modes including:

- content/geometry drift;
- incomplete target-domain conversion;
- texture smearing;
- color bleeding;
- streaking artifacts;
- loss of fine detail.

See:

- `failure_analysis.md`
- `outputs/failure_analysis/selected_failure_cases.csv`
- `outputs/failure_analysis/failure_cases_contact_sheet.png`

The failure analysis also documents proposed follow-up experiments. Those
experiments are recommendations and were not presented as completed runs.

---

## Blinded Audit Substitute

A fixed 30-sample blinded audit set was constructed:

- 15 Monet → Photo examples
- 15 Photo → Monet examples

The assignment requested two-rater style/content/artifact scoring. For this
submission, an evaluator-approved simulated/AI-assisted two-rater audit
substitute was generated rather than representing the scores as ratings from
external human participants.

Important provenance:

- external human raters were not used;
- the simulated ratings are disclosed as simulated;
- the fixed blinded image pairs and scoring protocol are preserved;
- agreement statistics are reported only as agreement between the two
  simulated rating streams.

Artifacts:

- `outputs/human_audit/audit_private_manifest.csv`
- `outputs/human_audit/blinded_pairs/`
- `outputs/human_audit/human_audit_instructions.txt`
- `outputs/human_audit/human_audit_protocol.json`
- `outputs/human_audit/rater1_simulated_scores.csv`
- `outputs/human_audit/rater2_simulated_scores.csv`
- `outputs/human_audit/simulated_audit_per_sample.csv`
- `outputs/human_audit/simulated_audit_results.json`
- `outputs/human_audit/simulated_rater_metadata.json`

---

## Reproducibility Files

Canonical production/evaluation environment:

`configs/environment_manifest.json`

Portable version-pinned production dependencies:

`configs/requirements_task3.txt`

Raw `pip freeze` from the RTX 4090 environment:

`configs/pip_freeze_task3.txt`

Some Conda-origin packages in the raw freeze use PEP 508 direct references such
as `package @ file:///...`; the exact portable versions are separately pinned
in `requirements_task3.txt`.

Historical local-development environment records are preserved as:

- `configs/environment_manifest_local_development.json`
- `configs/requirements_task3_local_development.txt`
- `configs/pip_freeze_task3_local_development.txt`

The historical files describe the earlier RTX 3060 development/smoke-test
machine and are not the production-training environment.

---

## Notebook

The main implementation/training/evaluation notebook is:

`notebooks/01_train_cyclegan_rtx4090.ipynb`

It contains the Task 3 workflow used to implement, train, infer, evaluate, and
generate the preserved evidence artifacts.

---

## Data

Expected local data directories:

- `../data/monet_jpg/`
- `../data/photo_jpg/`

The task uses unpaired Monet and photograph images.

Bulk source data are intentionally excluded from Git.

---

## Instructor Evaluation Notebook

An unchanged source copy of the instructor evaluator is preserved at the
repository root:

`../../Part3_Evaluation_Script.ipynb`

The evaluator source cells were not rewritten for the reported official
FID/MiFID evaluation.

A temporary compatibility workspace named `Part 3/` was created locally so the
evaluator could access the expected directory layout. That workspace is
ignored by Git because it duplicates images already represented by the real
source/prediction artifacts.

---

## Repository Evidence Map

Primary result documentation:

- `results.md`
- `failure_analysis.md`
- `full_metrics_report.csv`

Training provenance:

- `configs/cyclegan_baseline.json`
- `configs/environment_manifest.json`
- `logs/cyclegan_baseline_rtx4090_run001/`
- `checkpoints/checkpoint_manifest.json`
- `checkpoints/checkpoint_manifest.csv`

Quantitative evidence:

- `outputs/metrics/`
- `outputs/plots/`

Generated-image evidence:

- `outputs/cyclegan_baseline_rtx4090_run001/`
- `outputs/failure_analysis/`
- `outputs/human_audit/`

Submission evidence:

- repository-root `submission.csv`

---

## CycleGAN Reference

Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017).

**Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
Networks.**

Proceedings of the IEEE International Conference on Computer Vision (ICCV).

Paper:

https://arxiv.org/abs/1703.10593

---

## Important Interpretation Boundary

The reported metrics evaluate different properties and should not be treated as
interchangeable:

- FID/KID evaluate generated and real feature distributions;
- generative precision/recall evaluate manifold fidelity and coverage;
- cycle L1 evaluates pixel-space reconstruction;
- LPIPS evaluates perceptual reconstruction;
- content cosine similarity is a feature-space content-preservation proxy;
- qualitative failure analysis evaluates visible translation behavior.

Accordingly, no single metric is treated as a complete measure of CycleGAN
quality.
