# Task 2 — Viraat Chaudhary

Yelp Polarity; embeddings and all classifier weights learned from scratch.

Status: training and frozen evaluation completed; Viraat confirmed review of the explanations and analysis. The combined team PDF is deferred until all three tasks are complete.

## Design choices

### baseline

I selected a plain RNN as the baseline because it is a simple sequence model that learns patterns from word order. It provides a reference for evaluating the LSTM and TCN. Its difficulty retaining information across long sequences motivates the comparison with LSTM's gated memory. Keeping the RNN and LSTM embedding size, hidden size, pooling, and training settings the same makes their comparison more controlled.

Configuration: {"amp": true, "architecture": "rnn", "batch_size": 512, "dropout": 0.25, "early_stopping_patience": 2, "embedding_dim": 128, "embedding_dropout": 0.1, "gradient_clip_norm": 1.0, "head_dim": 64, "hidden_dim": 128, "learning_rate": 0.001, "max_epochs": 5, "name": "baseline", "pooling": "masked mean plus masked maximum", "pretrained_weights": false, "seed": 2662503, "selection": "validation macro-F1, then lower validation BCE", "settings_provenance": "assistant-proposed starting values; student reviews before first training", "weight_decay": 0.0001}

### experiment_1

I want to test whether the LSTM's gates help retain useful sentiment information across longer word sequences and improve performance compared with the plain RNN. Both models use the same data split, embedding size, hidden size, pooling, and training settings. I will compare their classification metrics, including performance on long reviews, and record parameter counts and training time to assess the tradeoff.

Configuration: {"amp": true, "architecture": "lstm", "batch_size": 512, "dropout": 0.25, "early_stopping_patience": 2, "embedding_dim": 128, "embedding_dropout": 0.1, "gradient_clip_norm": 1.0, "head_dim": 64, "hidden_dim": 128, "learning_rate": 0.001, "max_epochs": 5, "name": "experiment_1", "pooling": "masked mean plus masked maximum", "pretrained_weights": false, "seed": 2662503, "selection": "validation macro-F1, then lower validation BCE", "settings_provenance": "assistant-proposed starting values; student reviews before first training", "weight_decay": 0.0001}

### experiment_2

I want to test whether a TCN can classify sentiment effectively using dilated convolutions instead of recurrent processing. Dilations combine information from words at different distances, while residual connections support optimization. I will compare its classification metrics, performance on long reviews, training time, throughput, and memory usage with the RNN and LSTM to assess the tradeoff between predictive performance and computational cost.

Configuration: {"amp": true, "architecture": "tcn", "batch_size": 512, "channels": 128, "dilations": [1, 2, 4, 8, 16, 32], "dropout": 0.25, "early_stopping_patience": 2, "embedding_dim": 128, "embedding_dropout": 0.1, "gradient_clip_norm": 1.0, "head_dim": 64, "hidden_dim": 128, "kernel_size": 3, "learning_rate": 0.001, "max_epochs": 5, "name": "experiment_2", "pooling": "masked mean plus masked maximum", "pretrained_weights": false, "seed": 2662503, "selection": "validation macro-F1, then lower validation BCE", "settings_provenance": "assistant-proposed starting values; student reviews before first training", "weight_decay": 0.0001}

Embedding reason: I use randomly initialized, trainable embeddings because the assignment prohibits pretrained embeddings and language models. Each word has a 128-dimensional vector learned during sentiment training. This size provides a practical balance between representation capacity and computational cost, although it is a starting choice rather than a proven optimum. All three models use the same vocabulary and embedding dimension to keep their input representation consistent.

Hyperparameter reason: I use a batch size of 512 to process many reviews per update and make use of the A100 GPU. The learning rate of 0.001 is a starting value that balances update size and training stability. Training is capped at five epochs to control runtime, with early stopping after two epochs without improvement in the validation selection criterion. The best checkpoint is selected by validation macro-F1, using lower validation binary cross-entropy to break ties. Test results do not guide these choices.

Comparison question: My experiments will test whether the LSTM or TCN improves sentiment classification enough to justify its computational cost compared with the plain RNN. I will compare accuracy and macro-F1 alongside training time, examples per second, and peak memory usage. The results will show the performance and efficiency tradeoffs; I will not assume either experimental model is better before evaluation.

## Preprocessing evidence

Verified development preprocessing statistics:

```json
{
  "population": "training membership only",
  "examples": 504000,
  "minimum": 0,
  "maximum": 953,
  "mean": 68.49914682539682,
  "percentiles": {
    "25": 27.0,
    "50": 50.0,
    "75": 89.0,
    "90": 144.0,
    "95": 190.0,
    "99": 312.0
  },
  "sequence_length": 256,
  "truncated_reviews": 9948,
  "truncation_fraction": 0.01973809523809524,
  "empty_processed_reviews": 31,
  "training_slice_counts": {
    "short_reviews": 128920,
    "medium_reviews": 250581,
    "long_reviews": 124499,
    "contains_negation": 377234,
    "high_oov_rate": 25556
  }
}
```

## Final official-test metrics

All fields, including precision/recall/F1 averages, confidence intervals, paired tests, slices and resources, are in metrics_report.csv.

| model | accuracy | f1_macro | mcc | roc_auc | pr_auc_average_precision | brier_score | ece_15_bins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.9426578947368421 | 0.9426570131082743 | 0.8853430135768008 | 0.986512369806094 | 0.9869001206536869 | 0.04287902251349408 | 0.0023501019760868084 |
| experiment_1 | 0.9484736842105264 | 0.9484716287851925 | 0.8970189340052213 | 0.9891207991689751 | 0.9893550165963445 | 0.03873529061247016 | 0.008853935875636508 |
| experiment_2 | 0.9480263157894737 | 0.9480208125829976 | 0.8962424279062094 | 0.9891215110803324 | 0.9894920304736363 | 0.03944809373885799 | 0.011681590799034965 |

| model | total_parameter_count | training_wall_seconds | training_examples_per_second | peak_cuda_allocated_mib | cpu_model | gpu_model |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 6450049 | 46.98384015100055 | 58008.285551501154 | 581.1484375 | Intel(R) Xeon(R) CPU @ 2.20GHz | NVIDIA A100-SXM4-40GB |
| experiment_1 | 6549121 | 57.14593579000001 | 48179.360860937275 | 712.029296875 | Intel(R) Xeon(R) CPU @ 2.20GHz | NVIDIA A100-SXM4-40GB |
| experiment_2 | 7027969 | 280.528348735 | 9414.18459262025 | 2738.2275390625 | Intel(R) Xeon(R) CPU @ 2.20GHz | NVIDIA A100-SXM4-40GB |

Checkpoint-to-config/result provenance: outputs/metrics/checkpoint_result_mapping.json.

- baseline: [training curves](outputs/plots/baseline_training_curves.png), [confusion matrix](outputs/plots/baseline_confusion_matrix.png), [ROC/PR](outputs/plots/baseline_roc_pr.png), [calibration](outputs/plots/baseline_reliability.png), [raw log](logs/baseline_training.jsonl).
- experiment_1: [training curves](outputs/plots/experiment_1_training_curves.png), [confusion matrix](outputs/plots/experiment_1_confusion_matrix.png), [ROC/PR](outputs/plots/experiment_1_roc_pr.png), [calibration](outputs/plots/experiment_1_reliability.png), [raw log](logs/experiment_1_training.jsonl).
- experiment_2: [training curves](outputs/plots/experiment_2_training_curves.png), [confusion matrix](outputs/plots/experiment_2_confusion_matrix.png), [ROC/PR](outputs/plots/experiment_2_roc_pr.png), [calibration](outputs/plots/experiment_2_reliability.png), [raw log](logs/experiment_2_training.jsonl).

## Team comparison

| member | model | accuracy | f1_macro | mcc | roc_auc | pr_auc_average_precision | brier_score | ece_15_bins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Viraat_Chaudhary | baseline | 0.9426578947368421 | 0.9426570131082743 | 0.8853430135768008 | 0.986512369806094 | 0.9869001206536869 | 0.04287902251349408 | 0.0023501019760868084 |
| Viraat_Chaudhary | experiment_1 | 0.9484736842105264 | 0.9484716287851925 | 0.8970189340052213 | 0.9891207991689751 | 0.9893550165963445 | 0.03873529061247016 | 0.008853935875636508 |
| Viraat_Chaudhary | experiment_2 | 0.9480263157894737 | 0.9480208125829976 | 0.8962424279062094 | 0.9891215110803324 | 0.9894920304736363 | 0.03944809373885799 | 0.011681590799034965 |
| Anshika_Goel | baseline_mean_pool | 0.929474 | 0.929474 | 0.858947 | 0.977648 | 0.977576 | 0.0537766 | 0.00833923 |
| Anshika_Goel | textcnn | 0.941737 | 0.941736 | 0.883491 | 0.985671 | 0.986079 | 0.0448143 | 0.0173414 |
| Anshika_Goel | bigru_attention | 0.944868 | 0.944864 | 0.889869 | 0.988415 | 0.988652 | 0.0405061 | 0.00706147 |

Anshika values are transcribed from her uploaded executed evaluation notebook. Her code is not included. Overall official-test metrics share the population; tokenizer-dependent slices differ. Exact teammate CPU/GPU details require her hardware manifest. Cross-member runtime is not a controlled architecture benchmark.

## Student-written findings

### Preprocessing findings

The approved revision-2 preprocessing uses a stratified development split of 504,000 training reviews and 56,000 validation reviews, with balanced classes and no overlapping source IDs. The recorded audit found no malformed training rows or exact stripped-text duplicates within or across the development memberships. Literal escaped whitespace was normalized before tokenization. The training-only vocabulary contains 50,000 entries, with PAD, UNK and EMPTY reserved. Sequences are capped at 256 tokens. Recorded training truncation is 9,948 reviews (1.97381%); 1,080 validation reviews are truncated. Mean encoded OOV rates are approximately 0.003219 for training and 0.003973 for validation. There are 31 training and four validation token-empty reviews represented by EMPTY. Frozen length cutoffs define short as at most 27 processed tokens, medium as 28-89 and long as more than 89. Learned embeddings are initialized randomly and trained for sentiment classification. Stopword removal and punctuation removal may reduce discourse information even when negation tokens are retained. The official test was processed only after all selected checkpoints were frozen.

### Own model comparison

All three models were evaluated on the same 38,000 official test reviews. RNN accuracy is 0.942657895, macro-F1 0.942657013 and MCC 0.885343014. LSTM accuracy is 0.948473684, macro-F1 0.948471629 and MCC 0.897018934. TCN accuracy is 0.948026316, macro-F1 0.948020813 and MCC 0.896242428. LSTM improves accuracy over RNN by approximately 0.582 percentage points (221 additional correct predictions); TCN improves it by approximately 0.537 points (204 additional correct predictions). The baseline-versus-LSTM and baseline-versus-TCN McNemar tests have Holm-adjusted p-values of approximately 1.0951e-8 and 1.1310e-8. Both reject equal paired error rates at 0.05. LSTM has 17 more correct predictions than TCN, a 0.0447-point accuracy difference. No LSTM-versus-TCN significance test was reported, so that small difference does not establish superiority. TCN has slightly higher ROC-AUC (0.989122 versus 0.989121) and average precision (0.989492 versus 0.989355), while LSTM has lower Brier score (0.038735 versus 0.039448). RNN has the lowest recorded ECE (0.002350 versus 0.008854 for LSTM and 0.011682 for TCN), showing that a higher accuracy does not guarantee a lower ECE. On long reviews TCN macro-F1 is 0.948040, LSTM 0.943835 and RNN 0.937921. High-OOV macro-F1 is lower for all models: 0.922524, 0.927306 and 0.926472 respectively. Training wall times are 46.983840 s for RNN, 57.145936 s for LSTM and 280.528349 s for TCN on A100-SXM4-40GB. Their training-phase throughputs are approximately 58,008, 48,179 and 9,414 examples/s. Peak allocated CUDA memory is approximately 581.15, 712.03 and 2,738.23 MiB. Parameter counts are 6,450,049, 6,549,121 and 7,027,969. LSTM offers a strong observed accuracy/time balance in this run. TCN costs approximately 4.91 times the LSTM training wall time while improving the long-review slice. LSTM was selected for manual review using validation macro-F1 before any test results were accessed.

### Team model comparison

The six-model table records Anshika's mean-pooling baseline accuracy/macro-F1 as 0.929474/0.929474, TextCNN as 0.941737/0.941736 and BiGRU with attention as 0.944868/0.944864. My RNN, LSTM and TCN record 0.942658/0.942657, 0.948474/0.948472 and 0.948026/0.948021 respectively. Descriptively, my LSTM and TCN have the highest accuracy and macro-F1 in the displayed table. Anshika's BiGRU-attention model performs best among her three. The RNN accuracy is above TextCNN but below BiGRU-attention. Anshika's ROC-AUC values are 0.977648, 0.985671 and 0.988415, and average precision values are 0.977576, 0.986079 and 0.988652. My corresponding ROC-AUC values are 0.986512, 0.989121 and 0.989122, and average precision values are 0.986900, 0.989355 and 0.989492. These descriptive differences cannot isolate architecture effects because member-specific preprocessing, vocabulary and training settings differ. Slice memberships depend on each tokenizer and cannot be treated as identical merely because their names match. Exact teammate CPU/GPU disclosure and matching development source-ID hashes remain to be verified from her manifests. Her table values were imported from her executed evaluation output without rerunning her models. Cross-member paired significance was not established from aligned predictions, and raw training speeds are not a controlled comparison without exact hardware and timing scopes.

### Limitations

These are single-seed, five-epoch runs and do not measure variability across random initializations or exhaust hyperparameter choices. The models have different parameter counts, so matched embedding and hidden sizes do not imply equal capacity. The 256-token limit can discard information, and preprocessing may remove useful punctuation or discourse words. High-OOV reviews remain a difficult slice. The manual sample is deliberately selected by error type and confidence, so it does not estimate the population frequency of failure mechanisms. Several reviews have mixed sentiment, temporal updates or apparently inconsistent text and labels. A label-text mismatch is a hypothesis requiring independent audit, not permission to change the official label. Slice membership and a mistaken prediction do not prove a causal effect of OOV, negation or truncation. Full raw-versus-encoded comparisons and controlled development ablations would be needed to support such mechanisms. LSTM-versus-TCN significance and cross-member paired tests were not reported. Exact teammate hardware and split equivalence are still unverified. Confidence intervals and remaining per-model metrics should be checked in the exported metric artifacts before final report submission. The annotations and this prose were prepared with AI assistance. Viraat confirmed review of the recorded explanations and analysis before repository integration.

### Future work

Treat all fixes in the annotation CSV as untested proposals. Keep the current frozen checkpoints, official labels and test results unchanged. Use training and validation data for future ablations, with a new independent final holdout if model development continues after this test has been examined. Priorities are to retain contrast and temporal markers, represent sentence order and updated opinions, distinguish sentiment toward staff from sentiment toward other customers, and compare a from-scratch subword vocabulary for rare words. Verify actual retained lengths before changing the sequence limit. Audit suspected label issues only on independently checked development data. Repeat runs with multiple seeds and compare performance against wall time, throughput and memory. Evaluate probability calibration on validation data, with temperature scaling as a calibration experiment; it does not necessarily fix incorrect binary decisions at a fixed threshold. A future abstention option should report coverage-risk tradeoffs rather than silently remove difficult examples. Obtain complete teammate manifests and aligned predictions before stronger cross-member or slice comparisons.

### Individual contribution

My Task 2 work covers executing the corrected Yelp preprocessing, reviewing the proposed RNN/LSTM/TCN configurations, running all three models on Colab A100-SXM4-40GB, preserving executed notebooks and Drive artifacts, and running the frozen-model evaluation and error-selection workflow. The supplied implementation, proposed starting settings and initial explanation/annotation drafts were prepared with AI assistance.

## Evidence and references

Original raw logs are preserved. Training checkpoints are selected on validation; test predictions are cached and never used for retuning. Twenty-error evidence and student fixes are in failure_analysis.md.

- Bai, Kolter and Koltun (2018), An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling: https://arxiv.org/abs/1803.01271
- PyTorch RNN/LSTM documentation: https://docs.pytorch.org/docs/2.11/generated/torch.nn.LSTM.html
- Yelp source: https://huggingface.co/datasets/fancyzhx/yelp_polarity

Assistant assistance is disclosed in AI_use.md; student justifications and annotations are user-entered.
