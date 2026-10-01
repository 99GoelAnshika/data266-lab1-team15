# Task 1 - Team 15 model selection

Completed technical selection write-up derived from both frozen runs. The
[Task 1 report section](task1_team_report.md) combines the comparison, evidence
and six real failure examples for the eventual combined Tasks 1-3 PDF.
This document does not record another member's approval.

## Selection criterion and result

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
