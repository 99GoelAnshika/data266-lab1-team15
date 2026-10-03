## Task 2 — Viraat findings

Member folder: `task2_sentiment/Viraat_Chaudhary/`. The trained models are a plain RNN baseline, a unidirectional LSTM and a residual dilated causal TCN. Each learned its embeddings from scratch. All three runs recorded NVIDIA A100-SXM4-40GB and Intel(R) Xeon(R) CPU @ 2.20GHz; exact software versions and per-run timing scopes are preserved in the member manifests.

### Setup and checkpoint smoke reproduction

After cloning, run these commands from the repository root in an environment with Python and pip. The first installs dependencies; the second is the single checkpoint smoke command.

```bash
python3 -m pip install -r task2_sentiment/Viraat_Chaudhary/requirements.txt
```

```bash
python3 task2_sentiment/Viraat_Chaudhary/src/evaluator.py --smoke --config task2_sentiment/Viraat_Chaudhary/configs/baseline.json
```

The smoke command loads the saved RNN checkpoint on CPU with its frozen vocabulary, checks checkpoint/config integrity and emits synthetic-text probabilities. It requires no dataset download and does not recompute test accuracy. The supplied executed notebook 07 records CPU smoke execution for all three checkpoints. Executed notebook copies belong in the member's `src/`; retain the original `notebooks/` copies for the recorded export paths.

### Results and evidence

Complete metrics are in `task2_sentiment/Viraat_Chaudhary/metrics_report.csv`. The following paths are relative to that member folder: plots and predictions under `outputs/`, checkpoint-to-result mapping in `outputs/metrics/checkpoint_result_mapping.json`, paired comparisons in `outputs/metrics/mcnemar_tests.csv`, intervals in `outputs/metrics/bootstrap_confidence_intervals.json`, and slice results in `outputs/metrics/robustness_slice_metrics.csv`. Raw logs are in `logs/`; per-run environment records are in `environments/`.

The numerical results below were checked against saved run evidence. Viraat confirmed review of the recorded explanations and analysis; assistance is documented in the member's `AI_use.md`. Teammate exact hardware and split/checkpoint provenance remain unverified. The reviewed Task 2 analysis and actual error cases must also be integrated into the combined team PDF report.

| model | accuracy | f1_macro | mcc | roc_auc | pr_auc_average_precision | brier_score | ece_15_bins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.9426578947368421 | 0.9426570131082743 | 0.8853430135768008 | 0.986512369806094 | 0.9869001206536869 | 0.04287902251349408 | 0.0023501019760868084 |
| experiment_1 | 0.9484736842105264 | 0.9484716287851925 | 0.8970189340052213 | 0.9891207991689751 | 0.9893550165963445 | 0.03873529061247016 | 0.008853935875636508 |
| experiment_2 | 0.9480263157894737 | 0.9480208125829976 | 0.8962424279062094 | 0.9891215110803324 | 0.9894920304736363 | 0.03944809373885799 | 0.011681590799034965 |

All three models were evaluated on the same 38,000 official test reviews. RNN accuracy is 0.942657895, macro-F1 0.942657013 and MCC 0.885343014. LSTM accuracy is 0.948473684, macro-F1 0.948471629 and MCC 0.897018934. TCN accuracy is 0.948026316, macro-F1 0.948020813 and MCC 0.896242428. LSTM improves accuracy over RNN by approximately 0.582 percentage points (221 additional correct predictions); TCN improves it by approximately 0.537 points (204 additional correct predictions). The baseline-versus-LSTM and baseline-versus-TCN McNemar tests have Holm-adjusted p-values of approximately 1.0951e-8 and 1.1310e-8. Both reject equal paired error rates at 0.05. LSTM has 17 more correct predictions than TCN, a 0.0447-point accuracy difference. No LSTM-versus-TCN significance test was reported, so that small difference does not establish superiority. TCN has slightly higher ROC-AUC (0.989122 versus 0.989121) and average precision (0.989492 versus 0.989355), while LSTM has lower Brier score (0.038735 versus 0.039448). RNN has the lowest recorded ECE (0.002350 versus 0.008854 for LSTM and 0.011682 for TCN), showing that a higher accuracy does not guarantee a lower ECE. On long reviews TCN macro-F1 is 0.948040, LSTM 0.943835 and RNN 0.937921. High-OOV macro-F1 is lower for all models: 0.922524, 0.927306 and 0.926472 respectively. Training wall times are 46.983840 s for RNN, 57.145936 s for LSTM and 280.528349 s for TCN on A100-SXM4-40GB. Their training-phase throughputs are approximately 58,008, 48,179 and 9,414 examples/s. Peak allocated CUDA memory is approximately 581.15, 712.03 and 2,738.23 MiB. Parameter counts are 6,450,049, 6,549,121 and 7,027,969. LSTM offers a strong observed accuracy/time balance in this run. TCN costs approximately 4.91 times the LSTM training wall time while improving the long-review slice. LSTM was selected for manual review using validation macro-F1 before any test results were accessed.
