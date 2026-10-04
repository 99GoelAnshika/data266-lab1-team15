# Viraat — Task 2 Colab run package

The checked results package contains executed notebooks 01–07, three trained checkpoints and the actual evaluation outputs. The steps below record the original Colab execution and recovery workflow. Documentation and repository-layout updates require no retraining or repeat official-test evaluation.
Dataset: **Yelp Polarity**, following the TA correction. No IMDB, pretrained embeddings or pretrained language models are used.

## Recorded Colab workflow

1. Unzip this package on your computer. Open its notebooks in Colab one at a time.
2. Keep the existing raw and corrected processed dataset backups in Drive. Their original source folders and checksums are already configured. Do not upload datasets to GitHub.
3. Optionally upload the original `Viraat_Task2_Run_Package.zip` to `MyDrive/DATA266_Lab1_Task2/Viraat_Chaudhary/`. If absent, the first setup cell asks for this code ZIP once. Later notebooks use the installed Drive copy.
4. Use A100 for notebooks 02–05. CPU is sufficient for 06–07. Keep the same `RUN_NAME` and Drive base in every notebook.
5. The original training workflow started at **02** after the approved notebook 01 had completed. All seven stages now have executed copies in the checked results package.

For the brief's repository layout, byte-identical copies of all seven executed notebooks are present under `task2_sentiment/Viraat_Chaudhary/src/`. Keep the original `notebooks/` copies because the Colab export workflow and preservation manifest reference that location. Copying does not change notebook code, execution counts or saved outputs.

| Order | Notebook | What happens |
|---|---|---|
| 01 | `Viraat_01_data_analysis_preprocessing.ipynb` | Already completed and verified; retained byte-for-byte |
| 02 | `Viraat_02_train_baseline.ipynb` | Review settings, enter six own-word design explanations, restore processed data, train/resume RNN |
| 03 | `Viraat_03_train_experiment_1.ipynb` | Train/resume the unidirectional LSTM with the same pooling/head defaults |
| 04 | `Viraat_04_train_experiment_2.ipynb` | Train/resume residual causal TCN; freeze all three validation-selected checkpoints |
| 05 | `Viraat_05_evaluate_compare_models.ipynb` | Final frozen test inference, all required metrics/statistics/slices, own and team tables |
| 06 | `Viraat_06_manual_error_analysis.ipynb` | Download 20-error CSV and analysis JSON; fill in your own words, upload both, validate |
| 07 | `Viraat_07_evaluator_demo.ipynb` | Demo CPU checkpoint evaluator; upload executed 02–06; audit/export all results |

After each notebook, use **File → Download → Download .ipynb** to save the version with outputs. Keep its agreed filename. Keep those five downloads from 02–06 for the upload cell in 07.

## Model defaults to review before training

All three use a 50,000-entry training-only vocabulary, 256-token cap, learned embedding dimension 128, masked mean/max pooling, head dimension 64, batch size 512, AdamW learning rate 0.001, weight decay 0.0001, at most 5 epochs, patience 2, gradient clip 1, seed 2662503, and CUDA float16 AMP. Embedding dropout is 0.1; other dropout is 0.25.

RNN and LSTM are one-layer, unidirectional, hidden size 128 and have no attention. TCN has channels 128, six residual blocks, two kernel-3 convolutions per block, and dilations 1, 2, 4, 8, 16, 32. Its receptive field is 253 tokens; pooling aggregates all retained positions. Increasing the layer count does not guarantee a better model.

These were assistant-proposed initial numeric settings, not proven optimal settings. The saved run configs are frozen and must stay consistent with their checkpoints and manifests. The brief reserves core design decisions and analysis to the student. Uploaded design explanations do not establish independent authorship. See `AI_use.md`; personally review and be prepared to justify the recorded choices.

No runtime or accuracy improvement is promised. Epoch logs show actual validation performance, time, throughput and memory. Selection is maximum validation macro-F1, ties lower validation BCE. Validation loss controls learning-rate reduction. Test results must not be used to retune.

## Recovery and storage

The working member folder lives at `MyDrive/DATA266_Lab1_Task2/Viraat_Chaudhary/task2_models_rnn_lstm_tcn_v1/task2_sentiment/Viraat_Chaudhary/`. The path is controlled by the setup variables. Each notebook works in a fresh Colab session; it depends on saved files, not on earlier notebook variables.

Source/configs, weights, results and raw logs persist in that Drive folder. Arrays restore into local Colab cache from the approved backup. No vocabulary fitting is repeated. After a disconnect, rerun the same notebook with the same run name. It resumes after the last saved complete epoch; an interrupted partial epoch restarts. A completed model run is reused. Settings/code/data changes are rejected against an existing run identity.

Run sequentially, not concurrently. After test access, further training is blocked in the run. New run names are for genuine pre-test experiments, never a way to tune using already-observed test outcomes.

## Human work in notebook 06

The CSV contains actual full reviews, IDs, labels, probabilities and slice memberships. Fill only `error_type`, `explanation`, `testable_fix`; retain all evidence fields and 20 unique rows. Five most confident FP/FN errors are ranked within their respective groups; near-threshold errors are the closest remaining errors to 0.5. Report their actual confidence values. Slice-specific examples use frozen training-derived slices.

The supplied run already uploaded the annotation CSV and analysis JSON. Their populated content includes AI-assisted drafts; the validator checks fields and evidence consistency, not independent manual review. Viraat confirmed review of the Task 2 explanations and analysis before repository integration. For any future genuine revisions, inspect the actual reviews and results before changing explanations or findings. Re-running notebook 06 can regenerate `failure_analysis.md`, `results.md` and README/report contributions, so preserve the post-export documentation corrections when incorporating genuine student revisions.

## Original export and checked snapshot

Notebook 07 downloads **`Viraat_Task2_Run_Results.zip`**, including selected checkpoints, executed notebooks 01–06, configs/source, raw logs, metrics, predictions, plots and manual/student write-ups. It excludes datasets and optimizer resume files.

The separately downloaded **executed `Viraat_07_evaluator_demo.ipynb`** has already been incorporated into `Viraat_Task2_Checked_Results.zip`. The original ZIP and its `EXPORT_MANIFEST.json` describe that checked snapshot. Keep them as historical evidence; documentation changes and additional `src/` notebook copies create a subsequent repository state.

## GitHub and team integration

Copy only `task2_sentiment/Viraat_Chaudhary/` into the matching member directory in your repository. Preserve the shared README's existing content and Anshika's directory. Append the reviewed `SHARED_README_APPEND.md` section below her existing findings. `REPORT_TASK2_SECTION.md` supplies the Task 2 material for the jointly written combined report at `report/DATA266_Lab1_Report_Team_15.pdf`; it is not the required final PDF.

Preserve byte-identical copies of this member's logs under `task2_sentiment/reproducibility/Viraat_Chaudhary/logs/`. Preserve the environment captures, root environment manifests, `package_installation.json`, training manifests, checkpoint-result mapping, frozen-model metadata, configurations, and preprocessing provenance under `task2_sentiment/reproducibility/Viraat_Chaudhary/`, retaining their member-relative paths where applicable. Paths inside copied manifests still resolve against the canonical `task2_sentiment/Viraat_Chaudhary/` member directory. Retain the original evidence files there.

This package includes Anshika's reported numerical results from her uploaded executed evaluation notebook, with provenance. Her source/model code is not copied. Exact teammate CPU/GPU details were not supplied; obtain her hardware manifest before completing the team report. Slice membership and hardware/timing differences must be stated in the comparison.

The existing dataset link is user-confirmed readable: https://drive.google.com/drive/folders/1wzaSwbXnBjHUu1dOBT6sX-9PF2jeoovB . It references the raw ZIP by filename in the member README. Read access remains your responsibility at submission.

## One-command evaluator smoke test after training

Install the listed dependencies if needed, then use the single smoke command from the repository root:

```bash
python3 -m pip install -r task2_sentiment/Viraat_Chaudhary/requirements.txt
```

```bash
python3 task2_sentiment/Viraat_Chaudhary/src/evaluator.py --smoke --config task2_sentiment/Viraat_Chaudhary/configs/baseline.json
```

This loads a saved checkpoint on CPU and frozen preprocessing for synthetic text. No dataset download is needed. It checks inference operation/probability validity; it is not a new accuracy measurement.
