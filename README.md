# DATA 266 Lab 1 - Team 15

This repository contains Team 15's work for DATA 266 Lab 1: LLM pretraining, sentiment classification, and CycleGAN style transfer.

## Team Members

- Anshika Goel
- Viraat Chaudhary

Each team member independently designs, trains, evaluates, and documents a meaningfully different model for every task.

## Task 1: Character-Level GPT

Anshika Goel's completed Task 1 implementation trains a GPT-style character language model from scratch on TinyStories.

The implementation includes:

- A custom character vocabulary and deterministic data split
- Custom multi-head causal self-attention
- A manually created causal mask
- Manual layer normalization and GELU
- Learned token and positional embeddings
- Five pre-layer-normalized Transformer blocks
- Six attention heads per block
- Residual connections and a feed-forward network
- AdamW optimization with warm-up and cosine decay
- Mixed-precision CUDA training
- Checkpointing, logging, loss plots, and reproducibility metadata
- Greedy and temperature-based generation
- Diversity metrics and three evidence-backed failure analyses

No prebuilt Transformer, Transformer block, or multi-head attention module is used.

## Anshika Task 1 Architecture

| Item | Value |
|---|---:|
| Vocabulary size | 101 characters |
| Context length | 256 characters |
| Model dimension | 240 |
| Attention heads | 6 |
| Transformer blocks | 5 |
| Feed-forward dimension | 960 |
| Trainable parameters | 3,557,861 |
| Dropout | 0.10 |
| Token and output weights | Tied |

## Final Task 1 Results

| Metric | Value |
|---|---:|
| Training cross-entropy | 0.729267 |
| Validation cross-entropy | 0.693730 |
| Validation perplexity | 2.001166 |
| Bits per character | 1.000841 |
| Validation top-1 accuracy | 78.03% |
| Training throughput | 80,067.67 characters/second |
| Generation throughput | 518.08 characters/second |
| Peak allocated GPU memory | 1,586.03 MB |
| Total training time | 55.09 minutes |

Complete metrics and interpretation are available in `task1_llm/Anshika_Goel/results.md`.

## Repository Structure

    docs/
        DATA266_Lab1_Fall_2026.pdf

    task1_llm/
        data/
            raw/                       Ignored local raw files
            hf_cache/                  Ignored Hugging Face cache
            processed/                 Ignored processed tensors
        Anshika_Goel/
            code/
                data.py
                model.py
                train.py
                evaluate_generate.py
                task1_char_gpt_demo.ipynb
                tests/
            configs/
                gpt_char.yaml
            checkpoints/
                best_model.pt
                last_checkpoint.pt
            logs/
                preprocessing_run_002.log
                training_run_001.log
                evaluation_generation_run_001.log
            outputs/
                metrics/
                plots/
                samples/
            environment_manifest.txt
            failure_analysis.md
            requirements.txt
            results.md

Viraat Chaudhary will maintain a separate member directory and independently implement a meaningfully different Task 1 model.

## Environment Setup

From the repository root, create and activate a Python 3.11 virtual environment:

    py -3.11 -m venv .venv
    & ".\.venv\Scripts\Activate.ps1"

Install the locked environment:

    python -m pip install -r task1_llm\Anshika_Goel\requirements.txt

Register the Jupyter kernel if notebook access is needed:

    python -m ipykernel install --user --name data266-lab1-anshika --display-name "Python 3.11 (DATA266 Lab1 - Anshika)"

## Windows Hugging Face Cache Setup

The repository path may exceed the traditional Windows path-length limit. Map the data directory to a short drive before downloading TinyStories:

    subst T: "$((Get-Location).Path)\task1_llm\data"
    $env:HF_HOME = "T:\hf_cache"
    $env:HF_DATASETS_CACHE = "T:\hf_cache\datasets"
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

This mapping points to the existing repository data directory. Raw data, cache files, and processed tensors remain excluded from Git.

## Reproduce Preprocessing

From the repository root:

    python task1_llm\Anshika_Goel\code\data.py --config task1_llm\Anshika_Goel\configs\gpt_char.yaml

The preprocessing pipeline creates:

- 100,000 training sequences
- 10,000 validation sequences
- 256 input characters and one shifted target character per row
- A vocabulary derived only from training text
- Disjoint training and validation story groups
- SHA256 hashes for processed tensors and vocabulary metadata

Use `--force` only when intentionally replacing existing local processed tensors.

## Run Automated Tests

Set the source directory and execute the complete test suite:

    $env:PYTHONPATH = "task1_llm\Anshika_Goel\code"
    python -m pytest task1_llm\Anshika_Goel\code\tests -q

Expected result:

    22 passed

The tests cover vocabulary behavior, sequence shifting, split isolation, causal masking, future-token isolation, manual layer normalization, weight tying, model gradients, deterministic loaders, learning-rate scheduling, optimizer parameter grouping, diversity metrics, and deterministic sampled generation.

## Run a Training Smoke Test

    python task1_llm\Anshika_Goel\code\train.py --config task1_llm\Anshika_Goel\configs\gpt_char.yaml --smoke-test

The smoke test runs two training batches and two validation batches without writing or replacing full-run checkpoints.

## Run Full Training

On a fresh output directory:

    python task1_llm\Anshika_Goel\code\train.py --config task1_llm\Anshika_Goel\configs\gpt_char.yaml 2>&1 | Tee-Object -FilePath task1_llm\Anshika_Goel\logs\training_run_001.log

The training script deliberately refuses to overwrite an existing full run. If training is interrupted after a completed epoch, resume with:

    python task1_llm\Anshika_Goel\code\train.py --config task1_llm\Anshika_Goel\configs\gpt_char.yaml --resume task1_llm\Anshika_Goel\checkpoints\last_checkpoint.pt

## Run Evaluation and Generation

    python task1_llm\Anshika_Goel\code\evaluate_generate.py --config task1_llm\Anshika_Goel\configs\gpt_char.yaml --checkpoint task1_llm\Anshika_Goel\checkpoints\best_model.pt 2>&1 | Tee-Object -FilePath task1_llm\Anshika_Goel\logs\evaluation_generation_run_001.log

This command evaluates the complete validation set and produces 30 outputs:

- Three greedy outputs
- Nine samples at temperature 0.7
- Nine samples at temperature 1.0
- Nine samples at temperature 1.3

Use `--force` only when intentionally replacing existing evaluation outputs.

## Evidence and Reproducibility

Every reported result is connected to:

- The exact YAML configuration
- A pinned TinyStories dataset revision
- An unedited training log
- Best and resumable checkpoints
- Machine-readable training and evaluation metrics
- Generated text and decoding metadata
- A loss-curve image
- SHA256 artifact hashes
- An environment and hardware manifest
- An executed task-specific demonstration notebook
- Automated tests
- Genuine model failure examples

## Data and Credentials

Virtual environments, raw datasets, processed tensors, Hugging Face caches, personal filesystem paths, credentials, and API keys must not be committed.