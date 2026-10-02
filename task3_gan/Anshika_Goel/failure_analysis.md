# Task 3 Failure and Limitation Analysis ? Final Epoch 125

## Final-model scope

This analysis refers to the selected epoch-125 CycleGAN.

The final model improved the official FID/MiFID and Kaggle score, but the
auxiliary metrics show that additional training did not improve every property
of the generated images simultaneously.

## Distribution quality

Final epoch-125 values:

- official FID: **97.466148**
- official MiFID: **0.406866**
- KID mean: **0.012545**
- precision mean: **0.6067**
- recall mean: **0.4817**

The lower FID/MiFID and improved Kaggle score support better target-domain
distribution alignment under the competition metric.

However, KID does not improve in the same way. This illustrates that model
selection using only one distribution statistic can hide trade-offs.

## Content preservation

Epoch-125 mean Inception content cosine similarity is:

**0.769105**

This is lower than the epoch-50 control value `0.806853`.

Thus, stronger target-domain alignment at epoch 125 is accompanied by weaker
source/translation feature similarity under this particular proxy.

This does not prove semantic content loss for every image, but it is a concrete
warning that continued adversarial/style optimization may alter source
features.

## Cycle consistency

Epoch-125 mean cycle L1 is:

**0.033439**

The final cycle LPIPS mean is:

**0.158620**

Both provide complementary evidence because L1 measures pixel-space
reconstruction while LPIPS measures perceptual feature distance.

## Precision / recall trade-off

Final average precision is **0.6067** and final
average recall is **0.4817**.

The direction-specific values remain asymmetric:

- A2B precision: 0.7800
- A2B recall: 0.3333
- B2A precision: 0.4333
- B2A recall: 0.6300

This means distribution coverage remains direction dependent.

## Training stability

The durable training record contains epochs 1 through 125 with
**0 non-finite gradient events**.

Finite gradients support numerical stability, but do not imply convergence to
a globally optimal adversarial equilibrium.

## Historical visual failure artifacts

The repository contains earlier deterministic failure-case images and the
earlier 30-sample simulated audit artifacts.

They are retained for provenance, but they predate final epoch-125 selection.
They should not be represented as newly scored epoch-125 human evaluations.

## Final model-selection rationale

Epoch 125 was retained because it produced:

- a lower official FID than epoch 75;
- a lower official MiFID than epoch 75;
- a better Kaggle score (`-48.9365` vs `-49.2903`);
- a verified, direct CycleGAN inference path;
- reproducible final checkpoint/evaluator/submission evidence.

The final choice therefore reflects the assignment's official evaluator and
leaderboard evidence while still documenting auxiliary-metric trade-offs.


## Epoch 75 ? Epoch 125 auxiliary comparison

| Metric | Epoch 75 | Epoch 125 |
|---|---:|---:|
| FID mean | 98.170107 | 97.466231 |
| MiFID mean | 0.410627 | 0.406866 |
| KID mean | 0.011541 | 0.012545 |
| Precision mean | 0.6050 | 0.6067 |
| Recall mean | 0.5000 | 0.4817 |
| Content cosine | 0.795510 | 0.769105 |
| Cycle L1 | 0.038552 | 0.033439 |

This table demonstrates why the final checkpoint should not be described as
uniformly better on every metric.
