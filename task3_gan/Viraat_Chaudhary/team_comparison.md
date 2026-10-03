# Task 3 comparison — Team 15

## Evidence and selected runs

Anshika: `cyclegan_baseline_rtx4090_run001`, selected epoch **125**, global step
879750. Viraat: `viraat_resizeconv6_run001`, completed epoch **60**, global step
211140. The previous epoch-50 comparison is superseded.

Anshika's columns below come from the three newly supplied evidence files. Their
unchanged snapshots and file hashes are retained under
`outputs/viraat_resizeconv6_run001/comparison`. This review did not receive her
current raw logs, selected checkpoint, generated images, executed evaluator or
Kaggle result JSON. The values are reported evidence, not an independent rerun.
The configuration's scaffold `status` flags are stale; the results/metrics identify
selected epoch 125. Teammate files are not edited by this preparation.

See [source provenance](outputs/viraat_resizeconv6_run001/comparison/source_provenance.json)
and [current report snapshot](outputs/viraat_resizeconv6_run001/comparison/anshika_epoch125_full_metrics_snapshot.csv).

## Architecture and hyperparameters

| Setting | Anshika (reported) | Viraat (saved run) |
| --- | --- | --- |
| Generator | 9 residual blocks, 64 base channels | 6 residual blocks, 64 base channels; nearest resize-convolution |
| Discriminators | Two 70×70 PatchGANs | Two 70×70 PatchGANs |
| Parameters | 28,285,832 | 21,204,872 |
| Selected/evaluated epoch | 125; configuration plans 200 | 60 of 60 |
| Batch size / precision | 1 / FP32 | 2 / BF16 autocast, FP32 weights/Adam |
| Configured LR schedule | 100 constant epochs; 100 decay epochs | 20 constant epochs; 40 decay epochs |
| Base LR / Adam betas | 0.0002 / (0.5, 0.999) | 0.0002 / (0.5, 0.999) |
| Identity ratio / effective weight | 0.5 / 5 | 0.25 / 2.5 |
| Cycle weight | 10 each direction | 10 each direction |
| Replay pool | 50 | 50 |
| Seed | 20260929 | 2661510 |
| Training GPU | Configuration names lab RTX 4090 | RTX 4090 recorded in run manifest |

The earlier supplied baseline source used transposed-convolution upsampling; the
new configuration specifies nine residual blocks but does not separately name its
upsampling operator. Viraat's source uses nearest resize-convolution. These are
separate trained configurations, not a controlled single-factor ablation.

## Both-direction automatic metrics

A2B is Monet → Photo; B2A is Photo → Monet. Cycle measurements compare each source
with its round trip. MiFID is the class evaluator's fixed-index feature cosine
distance, not conventional memorization-informed FID.

| Direction | Metric | Anshika epoch 125 (reported) | Viraat epoch 60 (local) |
| --- | --- | --- | --- |
| A2B | FID | 96.068450 | 103.036489 |
| A2B | MiFID | 0.413548 | 0.416142 |
| A2B | KID | 0.017190 | 0.024615 |
| A2B | Generative precision | 0.780000 | 0.623333 |
| A2B | Generative recall | 0.333333 | 0.426667 |
| A2B | Cycle reconstruction L1 | 0.032256 | 0.043438 |
| A2B | Cycle LPIPS AlexNet | 0.182873 | 0.420099 |
| A2B | Inception cosine similarity | 0.783451 | 0.789136 |
| B2A | FID | 98.864013 | 95.615238 |
| B2A | MiFID | 0.400183 | 0.405585 |
| B2A | KID | 0.007900 | 0.007861 |
| B2A | Generative precision | 0.433333 | 0.546667 |
| B2A | Generative recall | 0.630000 | 0.596667 |
| B2A | Cycle reconstruction L1 | 0.034622 | 0.046902 |
| B2A | Cycle LPIPS AlexNet | 0.134368 | 0.309908 |
| B2A | Inception cosine similarity | 0.754759 | 0.793367 |

Anshika's KID subset standard deviations are absent from the supplied epoch-125
report. Viraat's are 0.003407 (A2B) and
0.001244 (B2A). Subset variability is not a confidence
interval establishing statistical significance.

| Evaluator summary | Anshika epoch 125 (reported) | Viraat epoch 60 |
| --- | --- | --- |
| Official professor FID average | 97.466148498151 | 99.328525404097 |
| Official professor MiFID average | 0.406865569646 | 0.410869756014 |
| Mean of local directional FIDs | 97.466231244915 | 99.325863503503 |

Viraat's official average FID is **1.862377 higher** than Anshika's selected
model. B2A FID is lower, but A2B FID is higher. A2B KID is higher; B2A KID is almost
the same and slightly lower. Viraat has higher B2A precision and A2B recall, with
lower A2B precision and B2A recall. Both cycle L1 and LPIPS are higher; both
input/translation cosine similarities are higher. No run wins every metric.
Use the unchanged professor CSV for submission, without an extra division or
sign change; numerical differences between local/professor FID implementations
are recorded separately.

## Efficiency and final training state

| Metric | Anshika epoch 125 (reported) | Viraat epoch 60 |
| --- | --- | --- |
| Recorded epoch-duration sum (hours) | 27.868480 | 5.266705 |
| Recorded domain images/sec | 17.537734 | 44.543976 |
| Peak allocated CUDA memory (MiB) | Not supplied for epoch 125 | 2587.737305 |
| Logged nonfinite gradient count | 0.000000 | 0.000000 |
| loss_generator_total | 2.811135 | 3.640399 |
| loss_gan_a_to_b | 0.475969 | 0.848098 |
| loss_gan_b_to_a | 0.812843 | 0.867927 |
| loss_discriminator_a | 0.055120 | 0.038158 |
| loss_discriminator_b | 0.157722 | 0.042537 |
| loss_cycle_a | 0.543312 | 0.743626 |
| loss_cycle_b | 0.627509 | 0.822668 |
| loss_identity_a | 0.147671 | 0.155582 |
| loss_identity_b | 0.203830 | 0.202499 |
| grad_norm_generator | 34.309808 | 32.479166 |
| grad_norm_discriminator_a | 4.317097 | 4.104792 |
| grad_norm_discriminator_b | 6.441369 | 5.933052 |

The run durations cover different amounts of training (125 versus 60 epochs),
batch sizes and schedules. Throughput combines architecture, batching, precision
and sampling changes; it cannot establish a causal effect of one component.
Epoch-50 GPU memory is not substituted for missing epoch-125 memory. Weighted
loss totals are not comparable image-quality scores because identity weights
and training schedules differ. Zero logged nonfinite gradients supports numerical
stability, not proof of ideal adversarial convergence.

## Human audit, leaderboard and team report

Viraat's two real-rater forms are blank. Anshika's current results explicitly retain
simulated/AI-assisted scores from epoch 50 as historical evidence and state that
external human raters were not used. Those scores are not final-model ratings from
two independent humans. The supplied archive contains no professor permission
document for an exception to the rubric's real-human audit requirement.

Anshika reports a Kaggle score of **-48.9365** and rank snapshot **14** for
`PairProgramming_Team_15`. This is the supplied report's snapshot, not a live rank
check. Her Kaggle result JSON was not included here. Viraat's new CSV has no
supplied Kaggle result yet, so no leaderboard improvement is claimed.

The evidence supports a smaller, higher-throughput Viraat configuration with
mixed quality metrics and greater cycle reconstruction error. A follow-up study
should separate block count, identity weight and batch/precision effects under a
common training/evaluation budget and several seeds. Training-domain evaluation
of 300 images per direction does not establish held-out generalization.

For the combined Lab 1 PDF, add the final human/Kaggle evidence, missing current
baseline efficiency evidence, checkpoint/result links, ownership statement and
joint interpretation across Tasks 1–3. This member comparison does not replace
`report/DATA266_Lab1_Report_Team_15.pdf`.
