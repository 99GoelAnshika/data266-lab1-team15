# Task 1 Results - Anshika Goel

## Status

Implementation and training are in progress. Pending fields will be replaced with measured results from the final run.

## Ownership

This model is independently designed, implemented, trained, evaluated, and documented by Anshika Goel.

## Data Configuration

| Item | Value |
|---|---:|
| Dataset | roneneldan/TinyStories |
| Tokenization | Character level |
| Training sequences | 100,000 |
| Validation sequences | 10,000 |
| Sequence length | 256 characters |
| Split seed | 2661501 |
| Input-target relationship | Target shifted one character to the right |

## Architecture

| Component | Configuration |
|---|---|
| Transformer blocks | 5 custom pre-layer-normalized blocks |
| Attention | Custom causal multi-head self-attention |
| Attention heads | 6 |
| Model dimension | 240 |
| Head dimension | 40 |
| Feed-forward dimension | 960 |
| Token embeddings | Learned |
| Positional embeddings | Learned |
| Dropout | 0.10 |
| Residual connections | Attention and feed-forward sublayers |
| Output head | Linear projection to character vocabulary |
| Weight tying | Token embedding and output projection |
| Prebuilt Transformer modules | None |

## Training Configuration

| Item | Value |
|---|---:|
| Epochs | 10 |
| Batch size | 32 |
| Optimizer | AdamW |
| Peak learning rate | 0.0003 |
| Minimum learning rate | 0.00003 |
| Warm-up | First 5% of optimization steps |
| Scheduler | Cosine decay |
| Weight decay | 0.10 |
| Gradient clipping | 1.0 |
| Mixed precision | Enabled on CUDA |
| Training seed | 20260924 |

## Required Metrics

| Metric | Final value | Evidence |
|---|---:|---|
| Training cross-entropy loss | Pending | Pending |
| Validation cross-entropy loss | Pending | Pending |
| Perplexity | Pending | Pending |
| Bits per character | Pending | Pending |
| Generalization gap | Pending | Pending |
| Top-1 next-character accuracy | Pending | Pending |
| Distinct-1 | Pending | Pending |
| Distinct-2 | Pending | Pending |
| Distinct-3 | Pending | Pending |
| Repeated 4-gram rate | Pending | Pending |
| Gradient norm summary | Pending | Pending |
| Loss spikes | Pending | Pending |
| NaN count | Pending | Pending |
| Parameter count | Pending | Pending |
| Training tokens per second | Pending | Pending |
| Generation tokens per second | Pending | Pending |
| Peak GPU memory | Pending | Pending |
| Total training time | Pending | Pending |

## Training Runs

| Run ID | Configuration | Log | Checkpoint | Outcome |
|---|---|---|---|---|
| Pending | configs/gpt_char.yaml | Pending | Pending | Pending |

## Generated Samples

Generated samples and diversity measurements will be saved under `outputs/samples/`.

## Loss Curves

Training and validation loss curves will be saved under `outputs/plots/`.

## Observations and Limitations

Pending final training and evaluation.