# Repository review - Task 1 only

Inspected the uploaded Team 15 repository. Members: Anshika Goel and Viraat Chaudhary.
Anshika's Task 1 source, YAML configuration, two notebooks, tests, metrics,
generations, raw logs, failure write-up and checkpoints are present.

## Observed baseline

- Frozen TinyStories revision: f54c09fd23315a6f9c86f9dc80f725de7d8f9c64.
- Existing protocol: 100,000 training / 10,000 validation sequences, 256-character context.
- Disjoint shuffled story-record groups; training-only vocabulary (101 symbols).
- Five custom pre-norm blocks, width 240, heads 6, feed-forward width 960,
  GELU, dropout 0.10, learned positions and tied embedding/output weights.
- Ten recorded epochs, batch 32, AdamW, peak LR 0.0003, 5% warm-up/cosine decay.
- GPU recorded in the evidence: NVIDIA GeForce RTX 3060 Laptop GPU.
- Validation CE 0.6937300155; PPL 2.0011660096; BPC 1.0008408530;
  next-character accuracy 0.7803367188; 3,557,861 parameters.

## Checks actually performed

Anshika's 22 tests passed in the review environment. Her best checkpoint loaded
with weights_only=True; all weights were finite and parameter count matched.
Both checkpoint SHA256 values matched the documented hashes. PPL/BPC formulas
matched. Distinct-1/2/3 and repeated 4-gram rate recomputed from the actual saved
27 sampled outputs matched the stored evaluation metrics. Ten training-history
rows are present. The original char-GPT demo has executed outputs and no error
outputs; the second evaluator demo is unexecuted.

This verifies saved evidence consistency and implementation checks; it does not
reproduce her complete training run or independently re-score her validation
corpus, which was excluded from the archive.

## Conventions and limits retained explicitly

The uploaded assignment and repo's PDF give the same Task 1 technical requirements.
The PDFs are different byte versions; the originally uploaded assignment remains
the grading reference. Existing `code/`/`logs/` conventions are followed for the
member implementation, with root metrics_report.csv and individual preprocessing.

The wording 100K/10K does not explicitly name the counted unit; this implementation
follows the existing partner's sequence-count protocol rather than presenting
that interpretation as a new instructor clarification.

Own validation splits differ; original own-split CE values alone do not establish
a controlled best-model ranking. GPU speeds differ as well. The new shared
holdout evaluator preserves both source implementations and evaluates frozen
checkpoints on the same official-validation text. Both member rows remain in
the report even when one model is selected for the demo.

No Anshika Task 1 source/evidence was changed. Tasks 2 and 3 were not modified.
Viraat's assessed training, generated failures, executed notebook and final
metrics remain pending the Colab run. The source checks use a separate CPU
PyTorch 2.8.0 environment and synthetic data; they are not A100 training results.
