# Task 3 Results — Anshika Goel

## 1. Task and Model

I implemented and trained an unpaired CycleGAN for bidirectional image translation between the Monet and photograph domains.

The two translation directions are:

- **A → B:** Monet → Photo
- **B → A:** Photo → Monet

The implementation contains two generators and two discriminators and was trained from scratch. The Kaggle submission was produced directly from the frozen CycleGAN generators without manual image editing, hand-picking, external generated images, lookup tables, or access to test-set pairings.

The canonical experiment configuration is:

`configs/cyclegan_baseline.json`

The evaluated production run is:

`cyclegan_baseline_rtx4090_run001`

The evaluated checkpoint is:

`checkpoints/cyclegan_baseline_rtx4090_run001/epoch_0050.pt`

The final evaluated checkpoint corresponds to **epoch 50** and **global step 351900**.

> **Important run/config distinction:** the canonical configuration defines a maximum schedule of 200 epochs, including the configured learning-rate schedule, but the preserved model evaluated for this submission is the actual epoch-50 checkpoint. All results in this document therefore refer to epoch 50; I do not claim that the full configured 200-epoch schedule was completed.

---

## 2. Dataset and Unpaired Training Setup

The two training domains were:

- **Monet:** 300 images
- **Photo:** 7,038 images

Training was explicitly unpaired. No source image was matched to a ground-truth target image during training.

The training preprocessing pipeline used:

- resize to 286 pixels;
- random crop to 256 × 256;
- horizontal flip probability = 0.5;
- normalization mean = [0.5, 0.5, 0.5];
- normalization standard deviation = [0.5, 0.5, 0.5];
- normalized model-input range = [-1.0, 1.0].

Evaluation used deterministic 256 × 256 images.

---

## 3. Architecture

### Generators

Both translation directions use the same generator architecture:

- architecture: **resnet_9block**
- residual blocks: **9**
- base channels: **64**
- input channels: **3**
- output channels: **3**
- normalization: **instance_norm**
- output activation: **tanh**

A 9-block ResNet generator was selected because CycleGAN-style image translation requires the network to modify domain appearance while retaining spatial structure. Residual blocks provide a direct path for preserving source structure while learning the domain transformation.

### Discriminators

Both domains use a PatchGAN discriminator:

- architecture: **patchgan_70x70**
- base channels: **64**
- layers: **3**
- input channels: **3**
- normalization: **instance_norm**

PatchGAN evaluates local image patches rather than assigning only one global real/fake decision, making it appropriate for learning local texture and style characteristics.

### Weight initialization

All trainable networks were initialized from scratch using:

- Normal distribution
- mean = 0.0
- standard deviation = 0.02

Total trainable parameters across both generators and both discriminators:

**28,285,832 parameters**

---

## 4. Loss Functions

The adversarial objective used **least-squares GAN / MSE loss**.

Cycle consistency used L1 loss with:

- lambda cycle A = **10.0**
- lambda cycle B = **10.0**

Identity preservation also used L1 loss. The configured identity ratio was **0.5**, producing effective identity weights:

- identity A = **5.0**
- identity B = **5.0**

Cycle consistency is essential because the training data are unpaired. For example, a Monet image translated to the photo domain should recover the original Monet content after applying the reverse generator.

---

## 5. Optimization and Training Configuration

The optimizer configuration was:

- optimizer: **adam**
- learning rate: **0.0002**
- beta1: **0.5**
- beta2: **0.999**
- weight decay: **0.0**
- batch size: **1**
- image replay-pool size: **50**
- mixed precision: **False**
- gradient clipping: **None**

The canonical configuration defines 100 constant-learning-rate epochs followed by 100 linear-decay epochs. The evaluated production artifact is the epoch-50 checkpoint, so the full configured decay schedule was not reached.

---

## 6. Hardware and Runtime

The serious training run was executed on an **NVIDIA GeForce RTX 4090**.

Measured run-level statistics:

| Metric | Result |
|---|---:|
| Actual completed epochs | 50 |
| Final global step | 351,900 |
| Trainable parameters | 28,285,832 |
| Training time | 11.0058 hours |
| Weighted throughput | 17.7634 images/sec |
| Peak CUDA memory | 2542.12 MiB |

Runtime evidence is preserved in:

- `logs/cyclegan_baseline_rtx4090_run001/events.jsonl`
- `logs/cyclegan_baseline_rtx4090_run001/run_manifest.json`
- `outputs/metrics/training_run_summary.json`
- `outputs/metrics/training_epoch_metrics.csv`

---

## 7. Distribution-Level Translation Metrics

Evaluation used 300 frozen epoch-50 outputs per direction.

| Metric | Monet → Photo (A2B) | Photo → Monet (B2A) | Two-direction mean |
|---|---:|---:|---:|
| FID ↓ | 100.397360 | 99.041190 | 99.719275 |
| MiFID ↓ | 0.418696 | 0.406675 | 0.412685 |
| KID ↓ | 0.016918 ± 0.002164 | 0.006703 ± 0.001331 | 0.011810 |
| Generative precision ↑ | 0.6467 | 0.4267 | 0.5367 |
| Generative recall ↑ | 0.3300 | 0.6433 | 0.4867 |

The official instructor evaluator produced the submission averages:

- **FID = 99.719274736**
- **MiFID = 0.412685258**

The directional FID/MiFID reproduction agreed with the instructor-generated `submission.csv` within numerical tolerance.

### Interpretation

Photo → Monet produced the lower FID and KID, indicating closer distribution-level alignment to the real Monet evaluation set under the selected Inception feature representation.

The precision/recall pattern differed between directions. Monet → Photo obtained higher precision (0.647) but lower recall (0.330), whereas Photo → Monet obtained lower precision (0.427) and higher recall (0.643). Under the k-nearest-neighbor manifold estimator used here, this suggests that the A2B outputs more frequently lie within the estimated real-photo manifold but cover a smaller portion of it, while B2A covers a broader portion of the estimated Monet manifold at the cost of lower precision.

Detailed artifacts:

- `outputs/metrics/directional_fid_mifid.json`
- `outputs/metrics/kid_metrics.json`
- `outputs/metrics/generative_precision_recall.json`

---

## 8. Cycle Consistency and Perceptual Reconstruction

### Cycle-reconstruction L1

| Direction | Mean L1 [0,1] |
|---|---:|
| Monet → Photo → Monet | 0.034629 |
| Photo → Monet → Photo | 0.045006 |
| Two-direction mean | 0.039817 |

The Monet → Photo → Monet cycle achieved the lower pixel-space reconstruction error.

### LPIPS

LPIPS was computed between each original source image and its cycle reconstruction using `lpips==0.1.4` with the AlexNet backbone.

| Direction | LPIPS mean ± std |
|---|---:|
| Monet → Photo → Monet | 0.209752 ± 0.074125 |
| Photo → Monet → Photo | 0.171835 ± 0.055726 |
| Two-direction mean | 0.190794 |

Interestingly, the direction with the lower pixel L1 was not the direction with the lower LPIPS. This is not contradictory: L1 directly measures pixel differences, whereas LPIPS compares perceptual feature representations.

Artifacts:

- `outputs/metrics/cycle_reconstruction_l1.json`
- `outputs/metrics/cycle_reconstruction_l1_per_sample.csv`
- `outputs/metrics/cycle_reconstruction_lpips.json`
- `outputs/metrics/cycle_reconstruction_lpips_per_sample.csv`

---

## 9. Content Preservation

Content preservation was measured as matched-image cosine similarity between the source and direct translation using 2048-dimensional ImageNet-pretrained Inception-v3 features.

| Direction | Mean cosine similarity |
|---|---:|
| Monet → Photo | 0.835613 |
| Photo → Monet | 0.778093 |
| Two-direction mean | 0.806853 |

Monet → Photo retained the higher feature-space similarity to its source images under this protocol.

Artifact:

`outputs/metrics/content_preservation_cosine.json`

Because the embedding network can respond to both semantic content and visual style, this metric should be interpreted as a feature-space content-preservation proxy rather than a perfect semantic-content measurement.

---

## 10. Training Losses at Epoch 50

The final epoch means from the original unedited training event log were:

| Metric | Epoch-50 mean |
|---|---:|
| Generator total loss | 3.149617 |
| GAN loss A2B | 0.491896 |
| GAN loss B2A | 0.649156 |
| Cycle-consistency loss A | 0.638300 |
| Cycle-consistency loss B | 0.842586 |
| Identity loss A | 0.219875 |
| Identity loss B | 0.307804 |
| Discriminator A loss | 0.097711 |
| Discriminator B loss | 0.150821 |

Training curves:

- [Generator losses](outputs/plots/generator_losses_by_epoch.png)
- [Discriminator losses](outputs/plots/discriminator_losses_by_epoch.png)
- [Cycle-consistency losses](outputs/plots/cycle_consistency_losses_by_epoch.png)
- [Identity losses](outputs/plots/identity_losses_by_epoch.png)
- [Gradient norms](outputs/plots/gradient_norms_by_epoch.png)

The plotted losses show a strong initial reduction followed by slower evolution over the remaining epochs. The adversarial losses continue to fluctuate, as expected for joint generator/discriminator optimization, while the cycle and identity components decrease substantially from their early-run values.

---

## 11. Training Stability

Epoch-50 mean gradient norms were:

| Network | Mean gradient norm |
|---|---:|
| Generators | 23.407706 |
| Discriminator A | 5.694744 |
| Discriminator B | 6.954282 |

The preserved training log recorded:

**Total non-finite gradient events = 0**

No logged non-finite gradient event occurred during the 50-epoch production run.

The gradient histories remained finite throughout the recorded run. This supports numerical stability of the completed training run; it does not by itself imply that the adversarial game reached a global optimum.

---

## 12. Blinded Audit Substitute

The original evaluation protocol requested a fixed 30-sample, two-rater audit covering style quality, content preservation, and visible artifacts.

External raters were unavailable. With evaluator permission, I used an **evaluator-approved simulated/AI-assisted substitute** on the same fixed 30-sample audit structure:

- 30 fixed samples;
- 15 Monet → Photo;
- 15 Photo → Monet;
- three 1–5 ordinal dimensions;
- two reproducible simulated rater vectors;
- independent deterministic seeds;
- agreement calculated with exact agreement and unweighted Cohen's kappa.

**These values are not presented as ratings from two external human participants.**

| Audit metric | Result |
|---|---:|
| Overall style quality | 3.5167 / 5 |
| Overall content preservation | 3.7833 / 5 |
| Overall artifact-free quality | 3.4000 / 5 |
| Combined audit score | 3.5667 / 5 |
| A2B combined audit score | 3.6444 / 5 |
| B2A combined audit score | 3.4889 / 5 |
| Pooled exact agreement | 44.44% |
| Pooled Cohen's kappa | 0.1336 |

Agreement by dimension:

| Dimension | Cohen's kappa |
|---|---:|
| Style quality | 0.2174 |
| Content preservation | 0.2424 |
| Artifact-free quality | -0.1091 |

The low pooled kappa indicates weak agreement between the two simulated rating vectors. The artifact-free dimension produced a negative kappa, meaning its observed exact agreement was lower than the chance agreement implied by the simulated raters' marginal score distributions.

These audit values should therefore be treated only as the evaluator-approved audit substitute and **not as empirical human perceptual-study evidence**.

Audit artifacts:

- `outputs/human_audit/blinded_pairs/`
- `outputs/human_audit/audit_private_manifest.csv`
- `outputs/human_audit/simulated_rater_metadata.json`
- `outputs/human_audit/simulated_audit_per_sample.csv`
- `outputs/human_audit/simulated_audit_results.json`

---

## 13. Kaggle Submission

The official Kaggle submission was generated from the direct outputs of the frozen epoch-50 CycleGAN.

Recorded leaderboard information at evaluation time:

- Team: **PairProgramming_Team_15**
- Competition score: **-50.0659**
- Recorded leaderboard rank: **12**

The submitted prediction directories contain exactly:

- 300 direct Monet → Photo outputs;
- 300 direct Photo → Monet outputs.

Prediction provenance is preserved in:

`outputs/cyclegan_baseline_rtx4090_run001/official_epoch50_predictions/inference_manifest.json`

The root-level Kaggle file is:

`../../submission.csv`

The recorded rank is a snapshot and may change if additional competition submissions are made.

---

## 14. Main Observations

1. **Distribution alignment differs by direction.** Photo → Monet achieved lower FID and KID, while Monet → Photo achieved higher manifold precision.

2. **Coverage differs strongly by direction.** B2A recall (0.643) was notably higher than A2B recall (0.330), suggesting different diversity/coverage behavior between the two generators.

3. **Pixel and perceptual cycle metrics tell different stories.** A2B2A obtained the lower cycle L1, while B2A2B obtained the lower LPIPS. Both measurements are useful because they capture different reconstruction properties.

4. **Direct A2B translation preserved more Inception-feature similarity.** A2B content cosine similarity was 0.836 compared with 0.778 for B2A.

5. **The completed training run was numerically stable.** All recorded gradient-norm values remained finite and the total logged non-finite gradient count was zero.

6. **The model remained imperfect.** FID remained around 99.72, generative recall was asymmetric, and the content/perceptual metrics show that translation can alter source information. These issues are examined separately in `failure_analysis.md`.

---

## 15. Limitations

Several limitations are important when interpreting these results:

- The canonical configuration defines 200 epochs, but the evaluated production checkpoint completed only 50 epochs.
- Distribution metrics were computed from 300 evaluation images per direction, so estimates are based on a relatively small sample.
- The data are unpaired, so there is no ground-truth translated counterpart for direct per-image target-domain comparison.
- Inception-feature cosine similarity is only a proxy for semantic content preservation.
- LPIPS was evaluated on cycle reconstructions, not against nonexistent paired target-domain ground truth.
- Generative precision/recall depends on the selected feature representation and k-nearest-neighbor manifold definition.
- The 30-sample audit is an evaluator-approved simulated substitute rather than an empirical study with two external human raters.
- The recorded Kaggle rank is a time-specific leaderboard snapshot.

---

## 16. Reproducibility and Evidence

Primary evidence artifacts:

- `configs/cyclegan_baseline.json`
- `logs/cyclegan_baseline_rtx4090_run001/events.jsonl`
- `logs/cyclegan_baseline_rtx4090_run001/run_manifest.json`
- `checkpoints/cyclegan_baseline_rtx4090_run001/epoch_0050.pt`
- `outputs/metrics/training_epoch_metrics.csv`
- `outputs/metrics/training_run_summary.json`
- `outputs/metrics/directional_fid_mifid.json`
- `outputs/metrics/kid_metrics.json`
- `outputs/metrics/generative_precision_recall.json`
- `outputs/metrics/cycle_reconstruction_l1.json`
- `outputs/metrics/cycle_reconstruction_lpips.json`
- `outputs/metrics/content_preservation_cosine.json`
- `outputs/plots/`
- `outputs/cyclegan_baseline_rtx4090_run001/official_epoch50_predictions/inference_manifest.json`
- `outputs/human_audit/`
- `full_metrics_report.csv`

Final checkpoint identity:

- epoch: **50**
- global step: **351900**
- checkpoint SHA256: **a66c5beafb3037167df5d34b72c39dea0b3b1d282f564dd7680cecc7585f592b**
- config SHA256: **4795c20ee2d0307d26659da578aa6e42834cd10def8ca1bd088285c0216bfae5**

---

## 17. Reference

Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. Proceedings of the IEEE International Conference on Computer Vision (ICCV).

