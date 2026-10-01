# Task 1 — Team 15 technical report section

This section is complete as a technical reporting artifact for the eventual
combined lab PDF. It does not attest to an unrecorded teammate approval.

## Contributions and objective

Anshika Goel's recorded run uses a pre-norm/GELU decoder with tied embeddings.
Viraat Chaudhary's separately configured run uses a post-norm/ReLU decoder with
an untied head. Both completed ten full epochs and retain their own
configurations, checkpoints, metrics, logs and failure analyses. Implementation
support and report preparation for Viraat used AI assistance, as disclosed in
his member README; personal independent reasoning and viva understanding cannot
be established by an artifact check.

Both reviewed implementations predict the next character using scratch causal
attention, explicit layer normalization, feed-forward sublayers, residual
connections, learned token/position embeddings and a vocabulary projection.
No pretrained weights or prebuilt Transformer/attention modules are used.

## Preprocessing and training protocol

The existing Team 15 convention counts 100,000 training and 10,000 validation
windows from TinyStories, with context 256 and stored row length 257. They are
sequence counts, not counts of complete stories. The assignment does not
explicitly identify the unit of its 100K/10K requirement. Different shuffle
seeds divide official training records before window construction; each member
builds the vocabulary using training text only. Frozen split manifests retain
record ranges and hashes. The vocabulary sizes are 101 and 96 respectively.

Both use cross-entropy, AdamW with betas 0.90/0.95, weight decay 0.10,
gradient clipping at 1.0, peak LR 0.0003, minimum LR 0.00003 and cosine decay.
Warm-up is 5% for Anshika and 10% for Viraat. Architecture, batch size, precision
and hardware differ, so the runs do not isolate a single causal factor.
The source remains under the existing repository's `code/` convention; the
assignment's illustrated folder template uses `src/`. The actual source
locations and reproduction commands are documented consistently.

## Complete recorded architecture, hyperparameters and metric table

Own validation splits differ. The paired training row uses final-epoch online
cross-entropy for both members, and the paired gap uses validation minus that
online training loss. Viraat additionally reports full-training-set FP32,
dropout-off loss/gap at the selected checkpoint. Anshika's corresponding
inference training measurement was not recorded; `None` means unmeasured.

| Field | Anshika | Viraat |
|---|---|---|
| member | Anshika_Goel | Viraat_Chaudhary |
| layers | 5 | 4 |
| width | 240 | 256 |
| heads | 6 | 8 |
| feed_forward_width | 960 | 1024 |
| dropout | 0.1 | 0.15 |
| normalization | pre_layer_norm | post_layer_norm |
| activation | gelu | relu |
| tied_embeddings | True | False |
| batch_size | 32 | 64 |
| epochs | 10 | 10 |
| learning_rate | 0.0003 | 0.0003 |
| warmup_ratio | 0.05 | 0.1 |
| split_seed | 2661501 | 2661502 |
| epoch_average_training_ce | 0.7292673282814026 | 0.9326997169876099 |
| selected_checkpoint_training_ce | None | 0.8575841665267945 |
| inference_generalization_gap | None | 0.006117999744415292 |
| validation_ce | 0.6937300154685974 | 0.8637021662712098 |
| perplexity | 2.00116600957984 | 2.3719257223049572 |
| bits_per_character | 1.0008408530323694 | 1.2460588320845294 |
| legacy_epoch_average_gap | -0.035537312812805255 | -0.06899755071640012 |
| next_character_accuracy | 0.78033671875 | 0.727812109375 |
| distinct_1 | 0.00437037037037037 | 0.0045925925925925926 |
| distinct_2 | 0.03970904772507979 | 0.04215839085578565 |
| distinct_3 | 0.15818830879071843 | 0.16659229510635132 |
| repeated_4gram_rate | 0.22237126462478576 | 0.17467769580445636 |
| gradient_norm_mean | 0.7671757246292508 | 0.6247545246332789 |
| gradient_norm_max | 11.639734268188477 | 3.3099968433380127 |
| loss_spike_count | 0 | 0 |
| nonfinite_loss_count | 0 | 0 |
| nonfinite_gradient_count | 10 | 0 |
| training_tokens_per_second | 80067.67031501204 | 590870.7275939166 |
| total_training_seconds | 3305.320511900005 | 451.62236405800104 |
| peak_gpu_allocated_mb | 1586.033203125 | 1758.94873046875 |
| peak_gpu_reserved_mb | 1754.0 | 2034.0 |
| parameters | 3557861 | 3273824 |
| peak_cpu_rss_mb | 1660.1640625 | 1674.8984375 |
| generation_tokens_per_second | 518.0794406968815 | 696.0130767304037 |
| gpu | NVIDIA GeForce RTX 3060 Laptop GPU | NVIDIA A100-SXM4-40GB |
| evidence | Anshika_Goel/outputs/metrics/evaluation_metrics.json | Viraat_Chaudhary/outputs/metrics/evaluation_metrics.json |

Both rows use actual saved evidence. Validation splits differ; these are descriptive own-split results, not a controlled common-test quality ranking.
The table uses final-epoch online training CE and the corresponding legacy-style gap for consistency with Anshika's saved definitions. Viraat also reports full-training-set inference CE and an inference-mode gap separately.
None in the supplementary inference columns means that quantity was not recorded for that member; it is not an estimated value.
Training/generation speeds and memory were measured on different GPUs and are hardware-specific; architecture alone does not explain a speed difference.
The recorded common held-out evaluation and checkpoint selection are in [best_model_selection.md](best_model_selection.md). The complete report section, including both members' actual failure snippets, is [task1_team_report.md](task1_team_report.md). Retain both individual runs in the combined lab report.

### Metric interpretation

Perplexity is exp(validation CE); bits per character is validation CE / ln(2).
Accuracy counts correct next-character predictions over all validation targets.
Online training loss includes changing parameters and dropout; a negative
online-style gap does not by itself establish better generalization.
Viraat's same-checkpoint inference training CE is 0.85758417 and inference gap
is 0.00611800. They must not be mixed with Anshika's online definition.

Distinct-n is unique character n-grams divided by occurrences across the 27
sampled continuations, excluding prompts and sample boundaries. Repeated
4-gram rate is the mean per-continuation duplicate-occurrence fraction. Three
greedy outputs are retained but excluded from sampled diversity aggregates.
These are character-level, not word-level or semantic-coherence measures.

Both retain three prompts, one greedy output per prompt and three sampled
outputs per prompt/temperature at 0.7, 1.0 and 1.3, with 500 new characters
per output and generation seed 20260925. All thirty real outputs are preserved.

Gradient norms are before clipping. Both record zero loss spikes and zero
nonfinite losses. Anshika records ten skipped FP16 updates with nonfinite
gradients; Viraat records zero nonfinite gradients. GPU allocated/reserved
memory and process CPU RSS are separate. Throughput counts target characters.
The different GPUs, precision, batch sizes and disclosed timing boundaries
prevent an architecture-only speed or memory conclusion.

## Shared holdout, model selection and measured tradeoffs


For this report, use lower cross-entropy on the common held-out text as the
quality criterion, with next-character accuracy as corroborating evidence.
This criterion is documented at reporting time; it is not claimed to have been
preregistered before training. Both checkpoints were frozen before the common
evaluation. No architecture-only causal conclusion follows from the comparison.

**Preferred Task 1 checkpoint: Anshika's epoch-10 model.** It has lower common
CE and higher common accuracy. Retain Viraat's implementation, metrics,
checkpoint and failure analysis as his separate individual contribution.

| Common held-out metric | Anshika | Viraat |
|---|---:|---:|
| Cross-entropy, nats/target character | 0.6944072781 | 0.8609892454 |
| Next-character accuracy | 78.1023% | 72.9764% |
| Unknown encoded characters | 5 | 8 |
| Frozen checkpoint epoch | 10 | 10 |

Anshika's accuracy is higher by
5.1259
percentage points. Her common CE is lower by
0.16658197 nats/character.

## Shared evaluation protocol

- Dataset: `roneneldan/TinyStories` at revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
- Official validation records `[0, 3092)`; neither
  training implementation uses this official source split for its training data.
- Same 2,570,000 normalized characters form 10,000 rows of length 257.
  Each row contributes 256 input/target positions: 2,560,000 evaluated targets.
- Context 256, FP32 inference, dropout disabled, evaluation batch size 32.
- Each model retains its own training-only vocabulary. Unknown counts above
  cover all encoded characters, including first-input positions. Anshika's
  vocabulary has 101 entries and Viraat's 96; coverage differences remain a limitation.
- Holdout text SHA256: `e1a567c53c47a55438e4718e21e717c5b370fd0df263c17bda6f38e20624a234`.
- Executed evaluator SHA256: `2ea63d33a211a32a4b57dcc79eee21e56ec881f545481e16ebd923980a97a935`.

The shared benchmark was added after the independent training runs. It provides
a common comparison for the frozen checkpoints; it does not turn the runs into
a controlled or preregistered experiment. Keep the common holdout reserved and
do not use it for subsequent tuning.

## Architecture and measured tradeoffs

| Setting | Anshika | Viraat |
|---|---|---|
| Blocks / width / heads / FF width | 5 / 240 / 6 / 960 | 4 / 256 / 8 / 1024 |
| Normalization / activation | pre-norm / GELU | post-norm / ReLU |
| Output weights | tied to token embeddings | untied |
| Dropout | 0.10 | 0.15 |
| Parameters | 3,557,861 | 3,273,824 |
| Training batch / epochs | 32 / 10 | 64 / 10 |
| Peak LR / minimum LR | 0.0003 / 0.00003 | 0.0003 / 0.00003 |
| Warm-up / scheduler | 5% / cosine | 10% / cosine |
| Training GPU / precision | RTX 3060 Laptop / FP16 | A100-SXM4-40GB / BF16 |
| Own split seed | 2661501 | 2661502 |
| Sampled repeated 4-gram rate | 0.22237126 | 0.17467770 |
| Training target characters/sec | 80,067.67 | 590,870.73 |
| Recorded total training seconds | 3,305.32 | 451.62 |

Viraat uses fewer parameters and his saved sampled continuations have a lower
repeated 4-gram rate. Those facts do not establish better grammar, narrative
coherence or a general resource advantage. Training throughput, time and memory
were recorded on different GPUs, precisions and batch sizes. No same-hardware
training benchmark or measured weighted quality/cost objective was run.

The complete architecture/hyperparameter/all-metric table is
[task1_comparison.md](task1_comparison.md), with numeric rows in
[task1_comparison.csv](task1_comparison.csv). Both online training losses and
their legacy-style gaps are shown consistently. Viraat additionally records
full-train FP32 inference loss/gap; that supplementary quantity is not available
for Anshika and is left unmeasured rather than inferred.

## Failures and next experiments

Review both members' actual failure analyses and generated text before describing
story quality. Viraat's three saved cases show a greedy repetition loop,
ambiguous/unexplained character identity and malformed words/grammar.
His proposed decoding and name-consistency fixes have not been tested.
Higher character diversity alone is not a coherence metric.

For future work, change one factor at a time on training/development data, keep
other settings fixed, and repeat across training seeds. A broader prompt set
and a fixed human review rubric would support stronger generation-quality claims.
These are proposed experiments; no improvement is asserted.

## Evidence and checkpoint mapping

| Model | Checkpoint | SHA256 |
|---|---|---|
| Anshika | [best_model.pt](../Anshika_Goel/checkpoints/best_model.pt) | `64ad99efc0bf2e49959d3ef54bf5373a4c3a974f0fa333ae4a37e62fac9d2ba3` |
| Viraat | [best_model.pt](../Viraat_Chaudhary/checkpoints/best_model.pt) | `52e2bddd0dadea877d0230f95e5e267cdd87c2f9c4401ff5817a5bc92d0093f1` |

- [Common CSV](common_validation_comparison.csv) and
  [protocol/environment manifest](common_evaluation_manifest.json).
- [Viraat common raw log](../Viraat_Chaudhary/logs/common_evaluation_20260930T230014655953Z.log).
- [Anshika results](../Anshika_Goel/results.md),
  [metrics](../Anshika_Goel/outputs/metrics/evaluation_metrics.json),
  [failures](../Anshika_Goel/failure_analysis.md) and
  [samples](../Anshika_Goel/outputs/samples/generated_samples.json).
- [Viraat results](../Viraat_Chaudhary/results.md),
  [metrics](../Viraat_Chaudhary/metrics_report.csv),
  [failures](../Viraat_Chaudhary/failure_analysis.md),
  [samples](../Viraat_Chaudhary/outputs/samples/generated_samples.json) and
  [loss curves](../Viraat_Chaudhary/outputs/plots/loss_curves.png).

Anshika's vocabulary/checkpoint hash difference is exactly an LF/CRLF checkout
conversion. The evaluator accepts only that verified byte transformation,
records both hashes and preserves both vocabularies and checkpoint files.

## References

- Vaswani et al. (2017), Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Eldan and Li (2023), TinyStories: https://arxiv.org/abs/2305.07759

## Partner CSV packaging correction

The original uploaded partner folder omitted its required root metrics CSV.
The package adds `task1_llm/Anshika_Goel/metrics_report.csv` using only the
values already present in her evaluation JSON. An identical derived copy
remains in this comparison folder. Her source, checkpoints, results, samples
and raw logs are unchanged. The shared reporting commit records this packaging
correction; it is not a new experiment or a claim that Viraat trained her model.

## Six real failure examples

Every excerpt is an exact substring of its saved continuation. The examples
illustrate observed failures; they do not quantify overall grammar quality
or prove a particular hyperparameter caused an error. Full individual analyses
include proposed, untested fixes.

### 1. Anshika_Goel — Sentence repetition

Sample `sample_001`; decoding `greedy`; temperature `None`.

> She said they were going to the park to play with it. She said they were going to the park to play with it.

The same sentence repeats instead of advancing the story.

[Saved generation evidence](../Anshika_Goel/outputs/samples/generated_samples.json).

### 2. Anshika_Goel — Word repetition

Sample `sample_005`; decoding `sampled`; temperature `1.0`.

> she found a box of blue blue blue blue blue balls.

Repeated use of blue interrupts narrative progression despite temperature sampling.

[Saved generation evidence](../Anshika_Goel/outputs/samples/generated_samples.json).

### 3. Anshika_Goel — Malformed words and coherence

Sample `sample_008`; decoding `sampled`; temperature `1.3`.

> The nlight was nothing happened, hugged and ran off for bed.

The malformed word and broken sentence make the event hard to interpret.

[Saved generation evidence](../Anshika_Goel/outputs/samples/generated_samples.json).

### 4. Viraat_Chaudhary — Greedy repetition

Sample `sample_001`; decoding `greedy`; temperature `None`.

> You are very sad. You are a good friend. You are very sad. You are a good friend.

A sentence pair repeats without developing the story.

[Saved generation evidence](../Viraat_Chaudhary/outputs/samples/generated_samples.json).

### 5. Viraat_Chaudhary — Ambiguous character identity

Sample `sample_006`; decoding `sampled`; temperature `1.0`.

> , there was a little girl called Jelie, "Zoomp! I love you. Thank you, Grandma!" Lily said.

The text introduces Jelie but attributes the speech to Lily without explaining who Lily is.

[Saved generation evidence](../Viraat_Chaudhary/outputs/samples/generated_samples.json).

### 6. Viraat_Chaudhary — Malformed words and grammar

Sample `sample_018`; decoding `sampled`; temperature `1.3`.

> Lily noticed down the tib legs. She splashed the stame. She's notice and kicked some chone flowersh.

Nonstandard words and broken grammar reduce readability.

[Saved generation evidence](../Viraat_Chaudhary/outputs/samples/generated_samples.json).

## Synthesis and next experiments

Anshika has stronger next-character prediction on the common holdout. Viraat
provides a smaller alternative with lower recorded sampled repetition. Both
still exhibit repetition, malformed language or incoherent transitions in
actual outputs. Greater character diversity is not enough to establish better
story quality.

A next decoding experiment can compare greedy decoding, lower temperatures and
a repetition penalty on fixed development prompts, using the same checkpoint,
length and recorded seeds. Measure repeated sentences, malformed words and
human coherence ratings together. A next architecture experiment should change
one factor at a time and repeat across training seeds, using training/development
data rather than tuning on the common holdout. These are proposals; no fix or
performance improvement has been demonstrated.

## Evidence index and submission boundaries

- [Anshika results](../Anshika_Goel/results.md), [metrics CSV](../Anshika_Goel/metrics_report.csv), [failure analysis](../Anshika_Goel/failure_analysis.md), [loss plot](../Anshika_Goel/outputs/plots/loss_curves.png), [raw training log](../Anshika_Goel/logs/training_run_001.log), and [environment/checkpoint mappings](../Anshika_Goel/environment_manifest.txt).
- [Viraat results](../Viraat_Chaudhary/results.md), [metrics CSV](../Viraat_Chaudhary/metrics_report.csv), [failure analysis](../Viraat_Chaudhary/failure_analysis.md), [loss plot](../Viraat_Chaudhary/outputs/plots/loss_curves.png), [raw training log](../Viraat_Chaudhary/logs/training_20260930T192119034828Z.log), [executed notebook](../Viraat_Chaudhary/code/task1_colab.ipynb), and [environment/checkpoint mappings](../Viraat_Chaudhary/environment_manifest.txt).
- [Common scores](common_validation_comparison.csv), [common protocol/environment manifest](common_evaluation_manifest.json), [all recorded metric rows](task1_comparison.csv), [six exact failure excerpts](team_failure_examples.json), and [review scope](../Viraat_Chaudhary/VERIFICATION.md).

This section covers Task 1 only. The combined PDF still needs Tasks 2/3, the
repository link, the team ownership statement and all required references.
Actual Git history and individual viva understanding remain student/team
responsibilities. Eight reporting-time commits organize completed work; they
cannot recreate an earlier development timeline. Sequence-count ambiguity,
the code-folder convention and the AI-assistance restriction remain disclosed.
Automated checks do not certify independent authorship or teammate approval.
