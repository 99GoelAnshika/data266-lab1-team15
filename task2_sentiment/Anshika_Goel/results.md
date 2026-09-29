# Task 2 - Yelp Polarity Sentiment Classification Results

**Member:** Anshika Goel
**Team:** 15
**Task:** Yelp Polarity binary sentiment classification
**Final evaluation population:** official Yelp Polarity test split, `n=38,000`

This document consolidates the frozen Task 2 architecture, hyperparameters, final-test metrics, statistical analyses, robustness results, resource measurements, provenance, and scientific reporting boundaries. Values are copied from the committed machine-readable artifacts; no metric is newly tuned or selected in this document.

## 1. Dataset and split protocol

- Dataset: `fancyzhx/yelp_polarity` / `plain_text`.
- Frozen dataset revision: `bbf1c97a1f0cf005e5aded43839fd814654a1557`.
- Official source training split: `560,000` examples.
- Training subset after the frozen validation split: `504,000` examples.
- Validation subset: `56,000` examples (10% of the official training split).
- Official test split: `38,000` examples.
- Train/validation split seed: `2662501`.
- Vocabulary size: `50,000`; minimum token frequency: `2`.
- Fixed sequence length: `256` tokens.
- Vocabulary and text embeddings were learned from the training data; pretrained embeddings and pretrained language models were not used.

### Preprocessing

- Lowercase: `True`
- HTML normalization: `True`
- Whitespace normalization: `True`
- Negation-contraction expansion: `True`
- Punctuation removal: `True`
- Special-character removal: `True`
- Stopword removal: `True`
- Preserved negations: `no`, `nor`, `not`, `never`
- Stemming: `porter`
- Tokenization: `regex_word_tokens`

## 2. Models and design rationale

### Baseline Mean Pool
A learned token embedding is masked-mean pooled over the model-visible sequence and passed to a linear classifier. This intentionally simple order-insensitive baseline tests how far learned lexical representations alone can go before adding local or sequential structure.

### TextCNN
A learned embedding feeds parallel one-dimensional convolution banks with kernel widths 3, 4, and 5, followed by global max pooling and classification. The design targets local phrase and n-gram patterns while remaining highly parallel.

### BiGRU-Attention
A learned embedding feeds a one-layer bidirectional GRU. Additive attention aggregates contextual token representations before classification. This design introduces word-order and bidirectional sequence context while allowing the classifier to weight different contextual states unequally.

### Frozen model hyperparameters

| Model | Architecture | Embedding | Main architecture settings | Dropout | Batch | Max epochs | LR | Weight decay |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Baseline Mean Pool | `learned_embedding_masked_mean_linear` | 128 | masked mean pooling + linear classifier | 0.2 | 512 | 5 | 0.001 | 0.0001 |
| TextCNN | `learned_embedding_multikernel_textcnn` | 128 | kernels=3/4/5; 128 filters/kernel | 0.5 | 256 | 5 | 0.001 | 0.0001 |
| BiGRU-Attention | `learned_embedding_bidirectional_gru_attention` | 128 | hidden=128; layers=1; bidirectional=True; attention=128 | 0.3 | 128 | 5 | 0.0005 | 0.0001 |

### Shared training controls

- Training seed: `20260924`
- Mixed precision: `True`
- Gradient clipping norm: `1.0`
- Warm-up ratio: `0.05`
- Minimum learning rate: `1e-05`
- Early-stopping patience: `2`
- Frozen decision threshold: `0.5`

## 3. Final official-test classification metrics

| Model | Accuracy | Precision macro | Precision micro | Precision weighted | Recall macro | Recall micro | Recall weighted | F1 macro | F1 micro | F1 weighted | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Mean Pool | 0.929474 | 0.929474 | 0.929474 | 0.929474 | 0.929474 | 0.929474 | 0.929474 | 0.929474 | 0.929474 | 0.929474 | 0.858947 |
| TextCNN | 0.941737 | 0.941754 | 0.941737 | 0.941754 | 0.941737 | 0.941737 | 0.941737 | 0.941736 | 0.941737 | 0.941736 | 0.883491 |
| BiGRU-Attention | 0.944868 | 0.945000 | 0.944868 | 0.945000 | 0.944868 | 0.944868 | 0.944868 | 0.944864 | 0.944868 | 0.944864 | 0.889869 |

### Ranking, calibration, and confusion matrices

| Model | ROC-AUC | PR-AUC | Brier | ECE (15 bins) | Confusion matrix `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---|
| Baseline Mean Pool | 0.977648 | 0.977576 | 0.053777 | 0.008339 | `[[17661, 1339], [1341, 17659]]` |
| TextCNN | 0.985671 | 0.986079 | 0.044814 | 0.017341 | `[[17833, 1167], [1047, 17953]]` |
| BiGRU-Attention | 0.988415 | 0.988652 | 0.040506 | 0.007061 | `[[17789, 1211], [884, 18116]]` |

## 4. 95% bootstrap confidence intervals

The committed protocol uses a nonparametric percentile bootstrap with replacement, 2,000 replicates, replicate size 38,000, 95% confidence, and seed `2662502`.

| Model | Accuracy 95% CI | Macro-F1 95% CI | MCC 95% CI |
|---|---|---|---|
| Baseline Mean Pool | [0.926973, 0.931974] | [0.926970, 0.931970] | [0.853945, 0.863941] |
| TextCNN | [0.939315, 0.944053] | [0.939310, 0.944052] | [0.878639, 0.888126] |
| BiGRU-Attention | [0.942500, 0.947079] | [0.942499, 0.947073] | [0.885147, 0.894284] |

## 5. Planned paired McNemar comparisons

The frozen planned comparison family contains only baseline-vs-TextCNN and baseline-vs-BiGRU-Attention. Continuity correction was used, followed by Holm correction across the two planned comparisons.

| Comparison | Baseline correct / experimental wrong | Baseline wrong / experimental correct | Discordant | Chi-square | Raw p | Holm-adjusted p | Reject at 0.05 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_mean_pool_vs_textcnn | 805 | 1271 | 2076 | 104.154624 | 1.871274e-24 | 1.871274e-24 | True |
| baseline_mean_pool_vs_bigru_attention | 648 | 1233 | 1881 | 181.316321 | 2.500459e-41 | 5.000917e-41 | True |

**Statistical boundary:** no TextCNN-vs-BiGRU-Attention McNemar test was performed. Therefore no pairwise statistical significance claim is made between those two experimental models.

## 6. Robustness slices

| Model | Slice | Count | Accuracy | Error rate | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Baseline Mean Pool | short_reviews | 9996 | 0.924970 | 0.075030 | 0.922356 |
| Baseline Mean Pool | medium_reviews | 18591 | 0.931580 | 0.068420 | 0.931577 |
| Baseline Mean Pool | long_reviews | 9413 | 0.930097 | 0.069903 | 0.927679 |
| Baseline Mean Pool | contains_negation | 27951 | 0.924511 | 0.075489 | 0.921872 |
| Baseline Mean Pool | high_oov_rate | 2336 | 0.917808 | 0.082192 | 0.913829 |
| TextCNN | short_reviews | 9996 | 0.936475 | 0.063525 | 0.934229 |
| TextCNN | medium_reviews | 18591 | 0.945296 | 0.054704 | 0.945296 |
| TextCNN | long_reviews | 9413 | 0.940295 | 0.059705 | 0.938362 |
| TextCNN | contains_negation | 27951 | 0.940324 | 0.059676 | 0.938360 |
| TextCNN | high_oov_rate | 2336 | 0.928082 | 0.071918 | 0.924423 |
| BiGRU-Attention | short_reviews | 9996 | 0.940676 | 0.059324 | 0.938494 |
| BiGRU-Attention | medium_reviews | 18591 | 0.947394 | 0.052606 | 0.947394 |
| BiGRU-Attention | long_reviews | 9413 | 0.944332 | 0.055668 | 0.942682 |
| BiGRU-Attention | contains_negation | 27951 | 0.943687 | 0.056313 | 0.941969 |
| BiGRU-Attention | high_oov_rate | 2336 | 0.928938 | 0.071062 | 0.925263 |

## 7. Resource measurements

| Model | Parameters | Training seconds | Mean train examples/s | Peak GPU allocated MB | Peak GPU reserved MB | Final-test examples/s |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Mean Pool | 6,400,129 | 72.6784 | 36636.5305 | 269.0864 | 308.0000 | 37655.4414 |
| TextCNN | 6,597,377 | 358.8451 | 7310.5459 | 308.8823 | 402.0000 | 15632.2678 |
| BiGRU-Attention | 6,631,425 | 1545.9143 | 1401.1242 | 223.4473 | 486.0000 | 6407.5701 |

## 8. Comparative observations

BiGRU-Attention had the highest observed final-test macro-F1 among the three evaluated models.
TextCNN and BiGRU-Attention both improved the observed classification metrics relative to the masked-mean baseline. The two planned McNemar comparisons also reject equal paired error behavior between the baseline and each experimental model after Holm correction.
BiGRU-Attention also had the lowest observed Brier score and lowest observed 15-bin ECE of the three evaluated models, while the baseline had the highest throughput and shortest training time.
The robustness slices show that high-OOV and short/ambiguous examples remain difficult. The manual review additionally shows mixed sentiment, discourse reversal, sarcasm, sentiment scope, short-input ambiguity, possible label/text disagreement, and fixed-length truncation as recurring qualitative limitations.

## 9. Teammate-comparison boundary

Report teammate comparison as unavailable from the current repository evidence. Do not infer, fabricate, rank, or estimate teammate model results.

The frozen evidence search covered seven repository scopes. No teammate Task 2 model/result evidence was available in those scopes, so teammate numerical ranking, architecture comparison, and hyperparameter comparison are not fabricated or inferred.

## 10. Checkpoints and provenance

| Model | Checkpoint | SHA256 |
|---|---|---|
| Baseline Mean Pool | `task2_sentiment/Anshika_Goel/checkpoints/baseline_mean_pool_best.pt` | `4ba3b1585ff34e40cf26d74dd7a56f59d38fc8010e795046772472bf72bae724` |
| TextCNN | `task2_sentiment/Anshika_Goel/checkpoints/textcnn_best.pt` | `6f65361355ab9219177dec3b6cb482d7f7e3103ecf0f9110c3eefda42b7e2c1a` |
| BiGRU-Attention | `task2_sentiment/Anshika_Goel/checkpoints/bigru_attention_best.pt` | `da6841121a69416601fbc9fde5e2e9c32fb4cc5da4cd4a9efb758e35ab66657a` |

### Primary evidence artifacts

- `task2_sentiment/Anshika_Goel/configs/sentiment.yaml`
- `task2_sentiment/Anshika_Goel/configs/final_evaluation.yaml`
- `task2_sentiment/Anshika_Goel/outputs/metrics/final_test_metrics.json`
- `task2_sentiment/Anshika_Goel/outputs/metrics/bootstrap_confidence_intervals.json`
- `task2_sentiment/Anshika_Goel/outputs/metrics/mcnemar_tests.json`
- `task2_sentiment/Anshika_Goel/outputs/metrics/robustness_slice_metrics.json`
- `task2_sentiment/Anshika_Goel/outputs/predictions/final_test_predictions.csv`
- `task2_sentiment/Anshika_Goel/outputs/manual_error_analysis/manual_review_selection_manifest.json`
- `task2_sentiment/Anshika_Goel/outputs/manual_error_analysis/manual_review_text_snapshot.json`
- `task2_sentiment/Anshika_Goel/outputs/comparison/teammate_evidence_availability.json`
- `task2_sentiment/Anshika_Goel/notebooks/05_evaluate_compare_models.ipynb`
- `task2_sentiment/Anshika_Goel/notebooks/06_manual_error_analysis.ipynb`
- `task2_sentiment/Anshika_Goel/logs/final_evaluation_run_001.log`
- `task2_sentiment/Anshika_Goel/logs/final_evaluation_run_002.log`
- `task2_sentiment/Anshika_Goel/logs/manual_error_analysis_run_001.log`
- `task2_sentiment/Anshika_Goel/environment_manifest.txt`

The first final-evaluation log is intentionally retained as the raw failed attempt caused by the metric-alias KeyError. The second final-evaluation log records the corrected successful execution. Neither raw log is edited.

## 11. Scientific boundary

The official test set has already been consumed. Its metrics, robustness observations, and manual error cases are final evaluation evidence only. They must not be used to retrain models, change hyperparameters, change the 0.50 threshold, redefine slices, or select a different checkpoint.

The manual error analysis is retrospective and descriptive; it does not establish causal explanations of internal model reasoning.
