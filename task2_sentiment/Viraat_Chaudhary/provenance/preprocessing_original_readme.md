# Task 2 — Viraat Chaudhary

## Dataset reference

[Yelp Polarity dataset ZIP folder](https://drive.google.com/drive/folders/1wzaSwbXnBjHUu1dOBT6sX-9PF2jeoovB)

Dataset: `fancyzhx/yelp_polarity`; config: `plain_text`; revision: `bbf1c97a1f0cf005e5aded43839fd814654a1557`.

ZIP: `yelp_polarity_bbf1c97a1f0c_20261001T203954337938Z.zip`.

The user confirmed access and a download option in a private browser window. Dataset files are stored in Drive. After extracting the ZIP, restore with `datasets.load_from_disk("yelp_polarity")`.

<!-- VIRAAT_TASK2_PREPROCESSING -->
## Preprocessing artifacts

Run `notebooks/01_data_analysis_preprocessing.ipynb` in Colab after mounting Drive.

Raw dataset: [Yelp Polarity ZIP folder](https://drive.google.com/drive/folders/1wzaSwbXnBjHUu1dOBT6sX-9PF2jeoovB). ZIP: `yelp_polarity_bbf1c97a1f0c_20261001T203954337938Z.zip`. Read access and download were confirmed by the user in a private browser window.

Official training rows are split into 504,000 training and 56,000 validation reviews with stratified seed 2662501. Training-only vocabulary limit: 50,000 including PAD/UNK/EMPTY; minimum frequency: 2; sequence length: 256. Literal escaped newlines, carriage returns and tabs are normalized before tokenization. Embeddings will be randomly initialized and learned from scratch.

The official test set is reserved for notebook 05 after all three models are frozen. Actual preprocessing statistics and duplicate audit are in `outputs/metrics/processed_training_analysis.json` and `outputs/metrics/duplicate_training_audit.json`. Student findings and model results are still pending.

Assistant assistance: data handling and checks. Team reference notebooks were consulted for split/metric conventions; teammate model code was not copied. Core architecture choices and manual analysis are written by the student.
