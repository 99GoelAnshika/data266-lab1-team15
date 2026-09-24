# DATA 266 Lab 1 - Team 15

This repository contains Team 15's work for DATA 266 Lab 1: LLM pretraining, sentiment classification, and CycleGAN style transfer.

## Team Members

- Anshika Goel
- Viraat Chaudhary

Each member independently designs, trains, evaluates, and documents their own models for all three tasks.

## Current Scope

Task 1 builds a character-level GPT-style language model from scratch using TinyStories. Prebuilt Transformer and attention modules are not used.

Anshika Goel's Task 1 model uses:

- Five custom Transformer blocks
- Six causal self-attention heads per block
- A 240-dimensional hidden representation
- A 960-dimensional feed-forward layer
- A 256-character context window
- Learnable token and positional embeddings
- Pre-layer normalization and residual connections

Viraat Chaudhary will independently choose and document a meaningfully different architecture and hyperparameter configuration.

## Repository Structure

    docs/
        DATA266_Lab1_Fall_2026.pdf

    task1_llm/
        data/
            raw/
            hf_cache/
        Anshika_Goel/
            code/
                tests/
            configs/
                gpt_char.yaml
            logs/
            checkpoints/
            outputs/
                metrics/
                plots/
                samples/
            requirements.txt

Member folders will also contain `results.md`, `failure_analysis.md`, `environment_manifest.txt`, raw training logs, checkpoints, and generated outputs.

## Anshika Task 1 Setup

Create and activate a Python 3.11 environment:

    py -3.11 -m venv .venv
    & ".\.venv\Scripts\Activate.ps1"

Install the locked environment:

    python -m pip install -r task1_llm\Anshika_Goel\requirements.txt

Register the Jupyter kernel:

    python -m ipykernel install --user --name data266-lab1-anshika --display-name "Python 3.11 (DATA266 Lab1 - Anshika)"

The smoke-test and training commands will be added after implementation is complete.

## Data and Reproducibility

The TinyStories dataset is downloaded through the Hugging Face datasets library. Raw data and caches are excluded from Git.

Every reported result will be linked to:

- The exact YAML configuration
- An unedited raw training log
- A model checkpoint
- Generated text samples
- Machine-readable metric files
- The environment and hardware manifest

No personal filesystem paths, credentials, API keys, or raw datasets should be committed.

## Task 1 Resources

- TinyStories dataset: https://huggingface.co/datasets/roneneldan/TinyStories
- TinyStories paper: https://arxiv.org/abs/2305.07759
- Attention Is All You Need: https://arxiv.org/abs/1706.03762