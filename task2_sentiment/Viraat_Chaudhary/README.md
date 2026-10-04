# Task 2 — Viraat Chaudhary

## Dataset reference

[Yelp Polarity dataset ZIP folder](https://drive.google.com/drive/folders/1wzaSwbXnBjHUu1dOBT6sX-9PF2jeoovB)

Dataset: `fancyzhx/yelp_polarity`; config: `plain_text`; revision: `bbf1c97a1f0cf005e5aded43839fd814654a1557`.

ZIP: `yelp_polarity_bbf1c97a1f0c_20261001T203954337938Z.zip`.

The user confirmed access and a download option in a private browser window. Dataset files are stored in Drive. After extracting the ZIP, restore with `datasets.load_from_disk("yelp_polarity")`.

<!-- VIRAAT_TASK2_PREPROCESSING -->
## Preprocessing artifacts

The executed preprocessing notebook is `Viraat_01_data_analysis_preprocessing.ipynb`. All seven executed notebooks have byte-identical copies in `src/`. The original `notebooks/` copies are retained because the export workflow and preservation manifest reference that location.

Raw dataset: [Yelp Polarity ZIP folder](https://drive.google.com/drive/folders/1wzaSwbXnBjHUu1dOBT6sX-9PF2jeoovB). ZIP: `yelp_polarity_bbf1c97a1f0c_20261001T203954337938Z.zip`. Read access and download were confirmed by the user in a private browser window.

Official training rows were split into 504,000 training and 56,000 validation reviews with stratified seed 2662501. Training-only vocabulary limit: 50,000 including PAD/UNK/EMPTY; minimum frequency: 2; sequence length: 256. Literal escaped newlines, carriage returns and tabs were normalized before tokenization. Embeddings were randomly initialized and learned from scratch.

The official test set was evaluated in notebook 05 after all three validation-selected models were frozen. Actual preprocessing statistics and the duplicate audit are in `outputs/metrics/processed_training_analysis.json` and `outputs/metrics/duplicate_training_audit.json`. Results for all three models are present in `metrics_report.csv` and `outputs/metrics/final_test_metrics.json`; saved predictions, plots and statistical comparisons are included under `outputs/`.

AI assistance included proposed model families and settings, implementation, initial explanation/annotation drafts, packaging and verification. Team reference notebooks were consulted for split/metric conventions and reported results. See [AI_use.md](AI_use.md) for the scope of assistance. Viraat confirmed review of the recorded explanations and analysis before repository integration. The assistance disclosure remains part of the submission.


## Complete training and evaluation workflow

Notebooks 01–07 have executed copies with outputs in the checked results package. Documentation and repository-layout updates do not require retraining or another official-test evaluation. [RUN_ORDER.md](RUN_ORDER.md) records the Colab workflow and recovery instructions. Re-running notebook 06 or an export can regenerate documents; preserve these post-export corrections if that happens.

The trained lineup is plain RNN, unidirectional LSTM, and residual dilated causal TCN. All model parameters and embeddings started randomly. Settings are recorded in `configs/`. Each run used NVIDIA A100-SXM4-40GB; the recorded CPU is Intel(R) Xeon(R) CPU @ 2.20GHz. Exact per-run hardware, software versions, checkpoint hashes, selected epochs and timing scopes are in `outputs/metrics/*_training_manifest.json` and `environments/`.

Notebook 06 collected twenty unique selected error cases and uploaded analysis. `results.md`, `failure_analysis.md`, `REPORT_TASK2_SECTION.md` and `SHARED_README_APPEND.md` contain actual numerical evidence and AI-assisted explanations. Viraat confirmed review of the explanations and analysis before repository integration. Original preprocessing README bytes remain in `provenance/preprocessing_original_readme.md`.

The repository integration adds this member's section to the shared root README and preserves the relevant log, environment, configuration, and manifest evidence under the task-scoped `task2_sentiment/reproducibility/Viraat_Chaudhary/` directory. Teammate hardware and split/checkpoint provenance still need to be confirmed when assembling the combined report. The final `report/DATA266_Lab1_Report_Team_15.pdf` is deferred until all three tasks are complete. The original checked ZIP and its `EXPORT_MANIFEST.json` describe the pre-update snapshot; retain them as that historical record.

### Evaluator smoke test after training

For a separate evaluator environment, install dependencies from the repository root:

```bash
python3 -m pip install -r task2_sentiment/Viraat_Chaudhary/requirements.txt
```

Then run the documented smoke command from the repository root:

```bash
python3 task2_sentiment/Viraat_Chaudhary/src/evaluator.py --smoke --config task2_sentiment/Viraat_Chaudhary/configs/baseline.json
```

This loads the selected RNN checkpoint on CPU and processes synthetic sentences with the frozen tokenizer/vocabulary. No dataset download is required. The checkpoint and its config hashes are verified. It is not a new test accuracy measurement. The supplied executed notebook 07 records a successful CPU smoke demonstration for all three checkpoints. Main dependencies are in `requirements.txt`; actual per-run versions are under `environments/`.

Historical verification/export records describe earlier snapshots and remain unchanged. The current repository documentation records the subsequent review and integration.
