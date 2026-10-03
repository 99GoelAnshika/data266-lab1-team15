# Task 3 results — Viraat Chaudhary, Team 15

## Completed run and provenance

Run `viraat_resizeconv6_run001` completed 60 epochs and 211,140 optimizer-loop steps. Both generators
and both discriminators were trained from scratch. The training, checkpointing and
logging infrastructure adapts Anshika Goel's supplied team implementation;
`configs/code_provenance.json` records the unchanged and modified modules. The
completed run's identities are in `configs/completed_run_evidence.json`.

The production checkpoint is `checkpoints/viraat_resizeconv6_run001/latest.pt`, SHA256
`138f9ae9edef412125ddfad643d2fc672c3de2d609819caf0b623979d3765252`. The config SHA256 is
`7f864f63aa4e7221c63625fd6b5e7de50c9819fb419bf176b8ce923ed7311aa5`. Raw logs are retained without edits.

## Architecture and configuration

Two RGB ResNet generators use six residual blocks, 64 base channels, instance
normalization, nearest-neighbor resize followed by reflection-padded convolutions,
and tanh outputs. Two three-layer 70×70 PatchGAN discriminators use 64 base channels.
Each generator has 7,837,699 parameters and each discriminator has 2,764,737;
the total is **21,204,872**. The nine-block team baseline has 28,285,832 parameters.

The smaller generator reduces compute and capacity. Resize-convolution changes
the upsampling operator rather than merely renaming the baseline. The tradeoff is
potential loss of fine reconstruction detail. These changes were trained together
with the settings below, so the results cannot isolate the effect of one change.

| Setting | Value |
| --- | --- |
| Training domains | 300 Monet images; 7,038 photographs; unpaired |
| Seed | 2661510 |
| Training transform | Resize 286; random crop 256; horizontal flip probability 0.5 |
| Normalization | RGB [0,1] to [-1,1] |
| Batch size / precision | 2 / BF16 autocast; FP32 model and Adam states |
| Optimizer | Adam; learning rate 0.0002; betas (0.5, 0.999) |
| Learning-rate schedule | 20 constant epochs followed by 40 decay epochs |
| Final epoch learning rate | 0.00000487804878048781 |
| Adversarial loss | Least-squares GAN / MSE |
| Cycle loss | L1; weight 10 in both directions |
| Identity loss | L1; ratio 0.25; effective weight 2.5 in both domains |
| Replay pool | 50 generated images per domain |
| Sampling | Shuffle all photographs once per epoch; randomly sample the smaller Monet domain |

## Hardware and training behaviour

Training used one NVIDIA GeForce RTX 4090, Python 3.10.13, PyTorch 2.1.2 and CUDA 12.1.
The sum of recorded epoch durations is **18960.139 s**
(5.2667 h); the run-start to run-complete interval
is **19633.226 s** (5.4537 h), including additional overhead.
Recorded throughput is **44.544 domain images/s**,
counting the two domain inputs, not unique dataset images. Peak allocated CUDA
memory is **2587.737 MiB**. This differs from total
process/device memory displayed by nvidia-smi.

All 60 epoch events and the final `run_complete` event are present. Logged nonfinite
gradient count is zero. Final mean gradient norms are 32.479166 for the generators,
4.104792 for discriminator A and 5.933052 for discriminator B. This supports
numerical stability; it does not prove adversarial convergence or ideal image quality.

| Logged component | Epoch 1 mean | Epoch 60 mean |
| --- | --- | --- |
| loss_cycle_a | 2.227211 | 0.743626 |
| loss_cycle_b | 2.399329 | 0.822668 |
| loss_discriminator_a | 0.212143 | 0.038158 |
| loss_discriminator_b | 0.232427 | 0.042537 |
| loss_gan_a_to_b | 0.451873 | 0.848098 |
| loss_gan_b_to_a | 0.491637 | 0.867927 |
| loss_generator_total | 6.683079 | 3.640399 |
| loss_identity_a | 0.549718 | 0.155582 |
| loss_identity_b | 0.563311 | 0.202499 |

Cycle and identity reconstruction components decreased. Discriminator losses also
decreased while generator adversarial losses increased. This pattern is consistent
with stronger discriminators and a remaining realism challenge; low discriminator
loss alone is not evidence of better translations. Logged cycle/identity components
include their configured weights and are not the unweighted evaluation RGB L1.

Evidence: [generator losses](outputs/viraat_resizeconv6_run001/metrics/generator_losses.png),
[discriminator losses](outputs/viraat_resizeconv6_run001/metrics/discriminator_losses.png),
[cycle/identity losses](outputs/viraat_resizeconv6_run001/metrics/cycle_identity_losses.png),
[gradient norms](outputs/viraat_resizeconv6_run001/metrics/gradient_norms.png), and
[epoch metrics](outputs/viraat_resizeconv6_run001/metrics/training_epoch_metrics.csv).

## Evaluation protocol and automatic results

The shared reference protocol uses the first 300 sorted images per domain,
deterministic direct inference at 256×256, and ImageNet Inception-v3 features.
Feature preprocessing is Resize(299), CenterCrop(299), tensor conversion and
ImageNet normalization. KID uses 50 deterministic subsets of 100, seed 266;
generative precision/recall uses k=3. Content cosine compares each input with its
own translation. Cycle L1 and AlexNet LPIPS compare each source with its round trip.
Pretrained feature networks are used for measurement, not image generation.

| Metric | A2B: Monet → Photo | B2A: Photo → Monet |
| --- | --- | --- |
| Local FID | 103.036489 | 95.615238 |
| Class MiFID | 0.416142 | 0.405585 |
| KID mean | 0.024615 | 0.007861 |
| KID subset std | 0.003407 | 0.001244 |
| Generative precision | 0.623333 | 0.546667 |
| Generative recall | 0.426667 | 0.596667 |
| Cycle L1, RGB [0,1] | 0.043438 | 0.046902 |
| Cycle LPIPS, AlexNet | 0.420099 | 0.309908 |
| Input/translation content cosine | 0.789136 | 0.793367 |

The professor's unchanged metric cells were executed successfully on all four
300-image sets. Their displayed directional FIDs are 103.040 for A2B and 95.617
for B2A; these figures are rounded notebook output. The exact generated CSV row is:

```csv
ID,FID,MiFID
1,99.32852540409687,0.410869756014433
```

These CSV values already average the two directions. No further division or sign
change is applied. They are evaluator metrics, not an observed Kaggle competition
score. Local FID uses an equivalent sample-space covariance calculation and differs
slightly numerically from the professor's SciPy calculation; the professor's CSV
is retained for submission. The class MiFID is the supplied evaluator's fixed-index
cosine-distance measurement, not a claim to the conventional memorization-informed
FID definition.

Evidence: [executed professor notebook](src/Part3_Evaluation_Script.ipynb),
[official verification](outputs/viraat_resizeconv6_run001/metrics/official_evaluation_verification.json),
[submission CSV](submission.csv), [full metrics](full_metrics_report.csv), and
[per-sample cycle metrics](outputs/viraat_resizeconv6_run001/metrics/cycle_metrics_per_sample.csv).

## Interpretation and limitations

B2A has lower FID and KID than A2B in this run, but A2B has higher generative
precision. The directions therefore trade estimated target-manifold fidelity and
coverage differently. Content cosine is about 0.79 in both directions, so source
information is retained imperfectly. Mean RGB cycle L1 is relatively small while
LPIPS remains substantial, showing that pixel reconstruction and perceptual
reconstruction are distinct measurements.

The quantitative comparison with Anshika is in [team_comparison.md](team_comparison.md).
Against her newly supplied selected epoch-125 model, this run's official average
FID is 1.862377 higher (99.328525 versus 97.466148). B2A FID is lower,
but A2B FID is higher. Both cycle L1/LPIPS values are higher, while both content
cosine values are higher. The results show tradeoffs, not an overall quality gain.
Anshika's current values are reported evidence from the supplied files; her
current checkpoint, raw logs and evaluator were not independently checked here.

Evaluation uses training-domain images under the supplied protocol; it is not
held-out generalization evidence. There are only 300 examples per direction,
one seed per completed run, no controlled single-factor ablation and no paired
ground-truth target translations. Feature-space similarity is a proxy for content.
Specific observed failures and proposed experiments are in
[failure_analysis.md](failure_analysis.md).

## Human audit and Kaggle evidence

Thirty fixed anonymous panels, 15 per direction, and two blank rating forms are
preserved. Two independent real-rater forms have not yet been supplied, so no
human mean or agreement is claimed. After both forms are complete, the CPU-only
audit script reports per-criterion means, exact agreement and unweighted Cohen's
kappa, plus clearly labelled pooled results, and updates the metrics files.

The new CSV's actual Kaggle public/private scores and rank have not yet been
provided. The existing baseline's recorded rank is a historical snapshot and is
not substituted for this run's leaderboard evidence.

## Reference

Zhu, J.-Y., Park, T., Isola, P., and Efros, A. A. (2017). *Unpaired Image-to-Image
Translation using Cycle-Consistent Adversarial Networks*. ICCV, pp. 2223–2232.
https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html
