# DATA 266 Lab 1 - Team 15

LLM pretraining from scratch, Yelp Polarity sentiment classification, and CycleGAN image style transfer.

This top-level README documents the current repository state, setup, reproducibility commands, result locations, evidence trails, scientific boundaries, and remaining submission work required by the DATA 266 Lab 1 brief.

## Team

- Anshika Goel
- Viraat Chaudhary

Each team member is required to independently design, train, evaluate, and defend their own models for all three tasks. At the current `main` commit, the tracked Task 1 and Task 2 implementation folders contain `Anshika_Goel`; the remaining teammate/task deliverables must be added before final submission.

## Current submission status

| Deliverable | Current status |
|---|---|
| Task 1 - GPT-style LLM from scratch | Anshika implementation, evidence, `results.md`, and `failure_analysis.md` complete |
| Task 2 - Yelp Polarity sentiment classification | Anshika three-model evaluation, evidence, `results.md`, and `failure_analysis.md` complete |
| Task 3 - CycleGAN | **Pending** - `task3_gan/` does not yet exist on `main` |
| Kaggle Task 3 submission / rank | **Pending** |
| Combined team PDF report | **Pending** - `report/` does not yet exist on `main` |
| Teammate Task 2 numerical comparison | **Unavailable from current repository evidence; not inferred or fabricated** |

## Repository-level submission rules

- Use one shared GitHub repository with a named member folder inside each task.
- Keep raw training logs unedited after each run; they are part of the grading evidence trail.
- Keep configurations, environment/package information, checkpoint provenance, results, and failure/error analysis with each member's work.
- Do not commit personal filesystem paths, credentials, API keys, raw local caches, virtual environments, or other machine-specific secrets.
- Every reported number should be traceable to a committed metric artifact, log, plot/sample, and checkpoint where applicable.
- The final team report must contain per-task member comparison tables, joint analysis, failure/error evidence, ownership information, and references.

## Repository structure

```text
README.md
docs/
  DATA266_Lab1_Fall_2026.pdf
task1_llm/
  data/                                  # local/ignored dataset and cache material
  Anshika_Goel/
    code/
      data.py
      model.py
      train.py
      evaluate_generate.py
      task1_char_gpt_demo.ipynb
      tests/
    configs/gpt_char.yaml
    checkpoints/
    logs/
    outputs/
    environment_manifest.txt
    failure_analysis.md
    requirements.txt
    results.md
task2_sentiment/
  data/                                  # local/ignored dataset/cache material
  Anshika_Goel/
    notebooks/
      01_data_analysis_preprocessing.ipynb
      02_train_baseline_mean_pool.ipynb
      03_train_textcnn.ipynb
      04_train_bigru_attention.ipynb
      05_evaluate_compare_models.ipynb
      06_manual_error_analysis.ipynb
    configs/
    checkpoints/
    logs/
    outputs/
    environment_manifest.txt
    failure_analysis.md
    requirements.txt
    results.md
task3_gan/                                # PENDING
report/                                   # PENDING
```

## Clone the repository

From PowerShell, clone the shared Team 15 repository and enter the repository root:

```powershell
git clone https://github.com/99GoelAnshika/data266-lab1-team15.git
Set-Location data266-lab1-team15
```
## Environment setup

The verified Anshika runs used Python 3.11 on Windows with CUDA-enabled PyTorch.

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r task1_llm\Anshika_Goel\requirements.txt
python -m pip install -r task2_sentiment\Anshika_Goel\requirements.txt
```

Exact package/hardware provenance is recorded separately in [Task 1 environment_manifest.txt](task1_llm/Anshika_Goel/environment_manifest.txt) and [Task 2 environment_manifest.txt](task2_sentiment/Anshika_Goel/environment_manifest.txt).

## Grader one-command smoke test

After installing the Task 1 requirements, the following single command runs the committed synthetic-data implementation test suite without downloading TinyStories or loading a full training checkpoint:

```powershell
python -c "import sys,pytest; sys.path.insert(0,r'task1_llm\Anshika_Goel\code'); raise SystemExit(pytest.main([r'task1_llm\Anshika_Goel\code\tests','-q']))"
```

Expected verified result: `22 passed`.

The tests exercise vocabulary handling, input/target shifting, train/validation story isolation, custom causal attention, future-token isolation, manual layer normalization, weight tying, gradient flow, deterministic data loading, learning-rate scheduling, optimizer parameter grouping, diversity metrics, and deterministic generation.

## Task 1 - GPT-style character language model from scratch

Anshika's Task 1 implementation trains a character-level GPT-style model from scratch on TinyStories. No prebuilt Transformer block, Transformer encoder/decoder, `nn.MultiheadAttention`, or scaled-dot-product attention helper is used.

### Task 1 data protocol

- Dataset: `roneneldan/TinyStories`.
- Frozen dataset revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
- Training sequences: `100,000`.
- Validation sequences: `10,000`.
- Sequence/context length: `256` characters.
- Split seed: `2661501`.
- Character vocabulary is constructed from training text only.
- Training and validation story groups are disjoint.

### Task 1 architecture and hyperparameters

| Setting | Value |
|---|---:|
| Tokenization | character |
| Vocabulary size | 101 |
| Context length | 256 |
| Model dimension | 240 |
| Attention heads | 6 |
| Transformer blocks | 5 |
| Feed-forward dimension | 960 |
| Dropout | 0.1 |
| Position embeddings | learned |
| Normalization | pre_layer_norm |
| Weight tying | True |
| Epochs | 10 |
| Batch size | 32 |
| Learning rate | 0.0003 |
| Minimum learning rate | 3e-05 |
| Warm-up ratio | 0.05 |
| Scheduler | cosine |
| Weight decay | 0.1 |
| Trainable parameters | 3,557,861 |

### Task 1 required evaluation metrics

| Metric | Verified value |
|---|---:|
| Training cross-entropy | 0.729267 |
| Validation cross-entropy | 0.693730 |
| Validation perplexity | 2.001166 |
| Bits per character | 1.000841 |
| Generalization gap | -0.035537 |
| Validation top-1 next-character accuracy | 78.03% |
| Distinct-1 | 0.004370 |
| Distinct-2 | 0.039709 |
| Distinct-3 | 0.158188 |
| Repeated 4-gram rate | 0.222371 |
| Mean gradient norm | 0.767176 |
| Maximum gradient norm | 11.639734 |
| Detected loss spikes | 0 |
| Nonfinite losses | 0 |
| AMP-skipped nonfinite-gradient updates | 10 |
| Parameter count | 3,557,861 |
| Training characters/second | 80067.67 |
| Validation characters/second | 257370.58 |
| Generation characters/second | 518.08 |
| Peak GPU allocated memory (MB) | 1586.03 |
| Peak GPU reserved memory (MB) | 1754.00 |
| Total training time (minutes) | 55.09 |

Generation evaluation contains 30 outputs: three greedy outputs plus sampled outputs at temperatures 0.7, 1.0, and 1.3. Diversity statistics above use the 27 sampled outputs.

### Task 1 reproduction

Preprocess TinyStories:

```powershell
python task1_llm\Anshika_Goel\code\data.py --config task1_llm\Anshika_Goel\configs\gpt_char.yaml
```

Dedicated training smoke test after preprocessing:

```powershell
python task1_llm\Anshika_Goel\code\train.py --config task1_llm\Anshika_Goel\configs\gpt_char.yaml --smoke-test
```

Full training on a fresh output state:

```powershell
python task1_llm\Anshika_Goel\code\train.py --config task1_llm\Anshika_Goel\configs\gpt_char.yaml
```

Resume from the committed resumable checkpoint:

```powershell
python task1_llm\Anshika_Goel\code\train.py --config task1_llm\Anshika_Goel\configs\gpt_char.yaml --resume task1_llm\Anshika_Goel\checkpoints\last_checkpoint.pt
```

Evaluate the best checkpoint and generate samples:

```powershell
python task1_llm\Anshika_Goel\code\evaluate_generate.py --config task1_llm\Anshika_Goel\configs\gpt_char.yaml --checkpoint task1_llm\Anshika_Goel\checkpoints\best_model.pt
```

### Task 1 evidence locations

- [Task 1 results](task1_llm/Anshika_Goel/results.md)
- [Task 1 failure analysis](task1_llm/Anshika_Goel/failure_analysis.md)
- [Task 1 training summary](task1_llm/Anshika_Goel/outputs/metrics/training_summary.json)
- [Task 1 evaluation metrics](task1_llm/Anshika_Goel/outputs/metrics/evaluation_metrics.json)
- [Task 1 split manifest](task1_llm/Anshika_Goel/outputs/metrics/split_manifest.json)
- [Task 1 generated samples](task1_llm/Anshika_Goel/outputs/samples/generated_samples.txt)
- [Task 1 loss curve](task1_llm/Anshika_Goel/outputs/plots/loss_curves.png)
- [Task 1 training log](task1_llm/Anshika_Goel/logs/training_run_001.log)
- [Task 1 evaluation/generation log](task1_llm/Anshika_Goel/logs/evaluation_generation_run_001.log)
- [Task 1 best checkpoint](task1_llm/Anshika_Goel/checkpoints/best_model.pt)
- [Task 1 executed demo notebook](task1_llm/Anshika_Goel/code/task1_char_gpt_demo.ipynb)

## Task 2 - Yelp Polarity sentiment classification

Task 2 uses Yelp Polarity binary sentiment classification with textual embeddings learned from scratch. No pretrained embeddings and no pretrained language models are used.

### Task 2 data and preprocessing protocol

- Dataset: `fancyzhx/yelp_polarity` / `plain_text`.
- Frozen dataset revision: `bbf1c97a1f0cf005e5aded43839fd814654a1557`.
- Official training population: `560,000` reviews.
- Frozen training subset: `504,000` reviews.
- Frozen validation subset: `56,000` reviews (10% of the official training split).
- Official test set: `38,000` reviews, consumed once for final evaluation.
- Train/validation split seed: `2662501`.
- Vocabulary size: `50,000`.
- Sequence length: `256` tokens.
- Preprocessing includes lowercasing, HTML/whitespace normalization, negation-contraction expansion, punctuation/special-character removal, stopword removal while preserving negations, Porter stemming, and tokenization.
- Data analysis includes review-length distribution, class distribution/balance, and malformed/missing-entry checks.

### Task 2 model lineup

| Model | Architecture | Embedding | Architecture details | Dropout | Batch | Max epochs | LR |
|---|---|---:|---|---:|---:|---:|---:|
| Baseline Mean Pool | `learned_embedding_masked_mean_linear` | 128 | masked mean pooling + linear classifier | 0.2 | 512 | 5 | 0.001 |
| TextCNN | `learned_embedding_multikernel_textcnn` | 128 | kernels 3/4/5; 128 filters/kernel | 0.5 | 256 | 5 | 0.001 |
| BiGRU-Attention | `learned_embedding_bidirectional_gru_attention` | 128 | BiGRU hidden=128; layers=1; attention=128 | 0.3 | 128 | 5 | 0.0005 |

### Task 2 final official-test metrics

| Model | Accuracy | P-macro | P-micro | P-weighted | R-macro | R-micro | R-weighted | F1-macro | F1-micro | F1-weighted | MCC | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Mean Pool | 0.929474 | 0.929474 | 0.929474 | 0.929474 | 0.929474 | 0.929474 | 0.929474 | 0.929474 | 0.929474 | 0.929474 | 0.858947 | 0.977648 | 0.977576 | 0.053777 | 0.008339 |
| TextCNN | 0.941737 | 0.941754 | 0.941737 | 0.941754 | 0.941737 | 0.941737 | 0.941737 | 0.941736 | 0.941737 | 0.941736 | 0.883491 | 0.985671 | 0.986079 | 0.044814 | 0.017341 |
| BiGRU-Attention | 0.944868 | 0.945000 | 0.944868 | 0.945000 | 0.944868 | 0.944868 | 0.944868 | 0.944864 | 0.944868 | 0.944864 | 0.889869 | 0.988415 | 0.988652 | 0.040506 | 0.007061 |

### Task 2 confusion matrices

Format: `[[TN, FP], [FN, TP]]`.

- **Baseline Mean Pool:** `[[17661, 1339], [1341, 17659]]`
- **TextCNN:** `[[17833, 1167], [1047, 17953]]`
- **BiGRU-Attention:** `[[17789, 1211], [884, 18116]]`

### Task 2 95% bootstrap confidence intervals

Protocol: nonparametric percentile bootstrap with replacement, `2000` replicates, replicate size `38,000`, confidence level `95%`, seed `2662502`.

| Model | Accuracy 95% CI | Macro-F1 95% CI | MCC 95% CI |
|---|---|---|---|
| Baseline Mean Pool | [0.926973, 0.931974] | [0.926970, 0.931970] | [0.853945, 0.863941] |
| TextCNN | [0.939315, 0.944053] | [0.939310, 0.944052] | [0.878639, 0.888126] |
| BiGRU-Attention | [0.942500, 0.947079] | [0.942499, 0.947073] | [0.885147, 0.894284] |

### Task 2 paired McNemar tests

The frozen planned family contains only baseline-vs-TextCNN and baseline-vs-BiGRU-Attention, with Holm correction over two comparisons.

| Comparison | Baseline correct / experimental wrong | Baseline wrong / experimental correct | Discordant pairs | Raw p | Holm-adjusted p | Reject at 0.05 |
|---|---:|---:|---:|---:|---:|---|
| baseline_mean_pool_vs_textcnn | 805 | 1271 | 2076 | 1.871274e-24 | 1.871274e-24 | True |
| baseline_mean_pool_vs_bigru_attention | 648 | 1233 | 1881 | 2.500459e-41 | 5.000917e-41 | True |

**Statistical boundary:** no TextCNN-vs-BiGRU-Attention McNemar test was performed, so no pairwise significance claim is made between those two models.

### Task 2 robustness slices

| Model | Slice | Count | Macro-F1 | Error rate |
|---|---|---:|---:|---:|
| Baseline Mean Pool | short_reviews | 9996 | 0.922356 | 0.075030 |
| Baseline Mean Pool | medium_reviews | 18591 | 0.931577 | 0.068420 |
| Baseline Mean Pool | long_reviews | 9413 | 0.927679 | 0.069903 |
| Baseline Mean Pool | contains_negation | 27951 | 0.921872 | 0.075489 |
| Baseline Mean Pool | high_oov_rate | 2336 | 0.913829 | 0.082192 |
| TextCNN | short_reviews | 9996 | 0.934229 | 0.063525 |
| TextCNN | medium_reviews | 18591 | 0.945296 | 0.054704 |
| TextCNN | long_reviews | 9413 | 0.938362 | 0.059705 |
| TextCNN | contains_negation | 27951 | 0.938360 | 0.059676 |
| TextCNN | high_oov_rate | 2336 | 0.924423 | 0.071918 |
| BiGRU-Attention | short_reviews | 9996 | 0.938494 | 0.059324 |
| BiGRU-Attention | medium_reviews | 18591 | 0.947394 | 0.052606 |
| BiGRU-Attention | long_reviews | 9413 | 0.942682 | 0.055668 |
| BiGRU-Attention | contains_negation | 27951 | 0.941969 | 0.056313 |
| BiGRU-Attention | high_oov_rate | 2336 | 0.925263 | 0.071062 |

### Task 2 resource measurements

| Model | Parameters | Training seconds | Train examples/s | Peak GPU allocated MB | Peak GPU reserved MB | Final-test examples/s |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Mean Pool | 6,400,129 | 72.6784 | 36636.5305 | 269.0864 | 308.0000 | 37655.4414 |
| TextCNN | 6,597,377 | 358.8451 | 7310.5459 | 308.8823 | 402.0000 | 15632.2678 |
| BiGRU-Attention | 6,631,425 | 1545.9143 | 1401.1242 | 223.4473 | 486.0000 | 6407.5701 |

### Task 2 interpretation and scientific boundaries

**BiGRU-Attention had the highest observed final-test macro-F1 among the three evaluated models.**

This is an observed descriptive comparison. It is not a claim that BiGRU-Attention significantly outperformed TextCNN because no TextCNN-vs-BiGRU-Attention McNemar comparison was part of the frozen statistical family.

The official Yelp Polarity test set has already been consumed. Final-test metrics, robustness findings, and the 20-example manual review are evaluation evidence only and must not be used for retraining, threshold selection, hyperparameter tuning, slice redefinition, or checkpoint reselection.

Report teammate comparison as unavailable from the current repository evidence. Do not infer, fabricate, rank, or estimate teammate model results.

### Task 2 evidence locations

- [Task 2 results](task2_sentiment/Anshika_Goel/results.md)
- [Task 2 failure analysis](task2_sentiment/Anshika_Goel/failure_analysis.md)
- [Final-test metrics](task2_sentiment/Anshika_Goel/outputs/metrics/final_test_metrics.json)
- [Bootstrap confidence intervals](task2_sentiment/Anshika_Goel/outputs/metrics/bootstrap_confidence_intervals.json)
- [McNemar tests](task2_sentiment/Anshika_Goel/outputs/metrics/mcnemar_tests.json)
- [Robustness slices](task2_sentiment/Anshika_Goel/outputs/metrics/robustness_slice_metrics.json)
- [Frozen final-test predictions](task2_sentiment/Anshika_Goel/outputs/predictions/final_test_predictions.csv)
- [Manual-review selection manifest](task2_sentiment/Anshika_Goel/outputs/manual_error_analysis/manual_review_selection_manifest.json)
- [Manual-review frozen text snapshot](task2_sentiment/Anshika_Goel/outputs/manual_error_analysis/manual_review_text_snapshot.json)
- [Teammate-evidence availability](task2_sentiment/Anshika_Goel/outputs/comparison/teammate_evidence_availability.json)
- [Final evaluation notebook](task2_sentiment/Anshika_Goel/notebooks/05_evaluate_compare_models.ipynb)
- [Manual error-analysis notebook](task2_sentiment/Anshika_Goel/notebooks/06_manual_error_analysis.ipynb)

### Task 2 raw logs

Raw logs are retained rather than rewritten. In particular, `final_evaluation_run_001.log` preserves the original metric-alias failure, and `final_evaluation_run_002.log` preserves the corrected successful evaluation.

- [baseline_mean_pool_training_run_001.log](task2_sentiment/Anshika_Goel/logs/baseline_mean_pool_training_run_001.log)
- [textcnn_training_run_001.log](task2_sentiment/Anshika_Goel/logs/textcnn_training_run_001.log)
- [bigru_attention_training_run_001.log](task2_sentiment/Anshika_Goel/logs/bigru_attention_training_run_001.log)
- [final_evaluation_run_001.log](task2_sentiment/Anshika_Goel/logs/final_evaluation_run_001.log)
- [final_evaluation_run_002.log](task2_sentiment/Anshika_Goel/logs/final_evaluation_run_002.log)
- [manual_error_analysis_run_001.log](task2_sentiment/Anshika_Goel/logs/manual_error_analysis_run_001.log)

## Task 3 - CycleGAN image style transfer

**Status: scaffold and reproducibility environment complete; implementation and training pending.** The Task 3 workspace is present under [`task3_gan/Anshika_Goel/`](task3_gan/Anshika_Goel/), including the assignment/integrity contract and verified environment records. No Task 3 model training or official evaluation has been performed yet.

The Lab 1 brief requires each member to train their own CycleGAN using two unpaired image domains, two generators, two discriminators, adversarial loss, cycle-consistency loss, image translation in both directions, and training-stability analysis.

Required Task 3 evaluation/reporting includes:

- FID in both directions.
- KID in both directions.
- Generative precision/recall or density/coverage.
- Cycle-reconstruction L1 distance.
- LPIPS perceptual similarity.
- Content-preservation cosine similarity.
- Generator and discriminator loss curves.
- Cycle-consistency and identity loss values.
- Gradient norms and NaN/stability counts.
- Blinded human audit of 30 fixed samples with two raters.
- Inter-rater agreement such as Cohen's kappa or percentage agreement.
- Parameter count, training time, images/second, and peak memory.
- Kaggle public/private leaderboard score and recorded rank.

**Leaderboard integrity:** the Kaggle submission must be direct inference output from the member's own trained CycleGAN. No manually edited, hand-picked, copied, externally sourced, lookup-table, pretrained/foundation-model, or test-pair-peeking outputs may be used.

## Final combined team report

**Status: pending.** The required `report/` directory and combined PDF are not yet present on the current `main` branch.

Before final submission, the team report must be committed at `report/DATA266_Lab1_Report_Team_15.pdf` and should include:

- A one-paragraph team ownership statement.
- A per-task comparison table covering every member.
- Architecture summaries and hyperparameters for every member/model.
- All required metrics for Tasks 1-3.
- Joint strengths, weaknesses, limitations, and future-work analysis for each task.
- Evidence links for reported numbers, including logs, plots/samples, and checkpoint IDs.
- Individual failure/error analyses with actual text/image snippets.
- References to Attention Is All You Need, TinyStories, and CycleGAN.
- The repository link.
- Task 3 Kaggle submission score/rank once available.

## Demo / Viva preparation

Each member must be ready to open their own checkpoints and `results.md` files and explain:

- Architecture and hyperparameter choices, including why each choice was made.
- How every reported metric was computed and interpreted.
- Task 1 failure cases and Task 2 manual error-review findings.
- How their own models differ from teammates' models.
- What the comparisons do and do not support.
- Limitations and what would be changed in a future experiment.

## Evidence preservation and reproducibility policy

Raw training logs are grading evidence and should remain unedited. New corrections or reruns should create new logs rather than replacing historical logs.

Configuration-driven runs are preferred over hard-coded machine paths. The tracked repository has been scanned for obvious personal filesystem paths and common secret/token patterns; none were found at the README preflight commit.

Raw datasets, processed local tensors, caches, and virtual environments remain local/ignored unless the assignment explicitly requires otherwise.

## Primary member documents

- [Task 1 results](task1_llm/Anshika_Goel/results.md)
- [Task 1 failure analysis](task1_llm/Anshika_Goel/failure_analysis.md)
- [Task 1 environment manifest](task1_llm/Anshika_Goel/environment_manifest.txt)
- [Task 2 results](task2_sentiment/Anshika_Goel/results.md)
- [Task 2 failure analysis](task2_sentiment/Anshika_Goel/failure_analysis.md)
- [Task 2 environment manifest](task2_sentiment/Anshika_Goel/environment_manifest.txt)

## Final pre-submission checklist

- [x] Anshika Task 1 code/config/logs/checkpoints/results/failure analysis present.
- [x] Anshika Task 1 uses custom attention rather than prebuilt Transformer/attention modules.
- [x] Anshika Task 2 trained all three required models.
- [x] Anshika Task 2 uses no pretrained embeddings or pretrained language models.
- [x] Anshika Task 2 required final metrics and statistical/robustness analyses are present.
- [x] Anshika Task 1 and Task 2 raw logs and environment manifests are committed.
- [x] One-command repository smoke-test command is documented above.
- [x] No tracked personal paths or obvious secrets found in the current preflight scan.
- [ ] Every team member has a completed folder under Task 1, Task 2, and Task 3.
- [ ] Team model architectures/hyperparameters are confirmed meaningfully different.
- [ ] Task 3 member implementations and required metrics are complete.
- [ ] Task 3 Kaggle submission is complete and rank is recorded.
- [ ] Team comparison tables are complete for all three tasks.
- [ ] Combined PDF report is committed under `report/`.
- [ ] Every member is prepared for the individual Demo/Viva.
