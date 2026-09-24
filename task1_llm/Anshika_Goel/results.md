# Task 1 Results - Anshika Goel

## Status

Task 1 implementation, training, validation, generation, quantitative evaluation, and failure analysis are complete.

## Ownership

This model was independently designed, implemented, trained, evaluated, and documented by Anshika Goel for Team 15.

## Data Configuration

| Item | Value |
|---|---:|
| Dataset | `roneneldan/TinyStories` |
| Dataset revision | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| Source split | Official training split |
| Tokenization | Character level |
| Vocabulary size | 101 |
| Training sequences | 100,000 |
| Validation sequences | 10,000 |
| Sequence length | 256 characters |
| Characters per stored row | 257, including shifted target |
| Training stories used | 28,604 |
| Validation stories used | 2,859 |
| Split seed | 2,661,501 |
| Story overlap | None |
| Unknown characters in validation | 0 |

The vocabulary was constructed only from the training text. Training and validation stories occupy disjoint ranges after deterministic shuffling. Each target sequence is the corresponding input shifted one character to the right.

## Architecture

| Component | Configuration |
|---|---|
| Transformer blocks | 5 custom pre-layer-normalized blocks |
| Attention | Custom causal multi-head self-attention |
| Attention heads | 6 |
| Model dimension | 240 |
| Dimension per head | 40 |
| Feed-forward dimension | 960 |
| Token embeddings | Learned |
| Positional embeddings | Learned |
| Maximum context | 256 characters |
| Activation | Manually implemented GELU approximation |
| Normalization | Manually implemented layer normalization |
| Dropout | 0.10 |
| Residual connections | Attention and feed-forward sublayers |
| Output head | Linear character-vocabulary projection |
| Weight tying | Token embedding and output projection |
| Trainable parameters | 3,557,861 |
| Prebuilt Transformer or attention modules | None |

Queries, keys, and values are independently projected and divided into six heads. Scaled dot-product attention, causal masking, softmax, head concatenation, and output projection are implemented directly with tensor operations.

## Training Configuration

| Item | Value |
|---|---:|
| Epochs | 10 |
| Batch size | 32 |
| Gradient accumulation | 1 |
| Optimizer | AdamW |
| Maximum learning rate | 0.0003 |
| Minimum learning rate | 0.00003 |
| Warm-up | First 5% of intended updates |
| Scheduler | Cosine decay |
| Adam beta 1 | 0.90 |
| Adam beta 2 | 0.95 |
| Weight decay | 0.10 |
| Gradient clipping threshold | 1.0 |
| Mixed precision | Enabled |
| Training seed | 20,260,924 |
| Intended optimizer updates | 31,250 |
| Completed optimizer updates | 31,240 |
| Training characters processed | 256,000,000 |

Ten optimizer updates, or 0.032% of attempted updates, were safely skipped after mixed-precision overflow detection found nonfinite gradients. No nonfinite loss was observed, model parameters remained finite, and validation loss improved during every epoch.

## Loss by Epoch

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 1.700381 | 1.017307 |
| 2 | 0.986310 | 0.864268 |
| 3 | 0.881707 | 0.797890 |
| 4 | 0.829790 | 0.765877 |
| 5 | 0.799201 | 0.742357 |
| 6 | 0.776939 | 0.726950 |
| 7 | 0.759546 | 0.713466 |
| 8 | 0.745757 | 0.704207 |
| 9 | 0.735716 | 0.697901 |
| 10 | 0.729267 | 0.693730 |

Both losses decreased smoothly. The final validation loss is slightly lower than the epoch-average training loss because training loss is accumulated while parameters are changing and dropout is active, whereas validation is measured after the epoch with dropout disabled.

## Final Metrics

| Metric | Final value | Evidence |
|---|---:|---|
| Training cross-entropy loss | 0.729267 | `outputs/metrics/training_summary.json` |
| Validation cross-entropy loss | 0.693730 | `outputs/metrics/evaluation_metrics.json` |
| Validation perplexity | 2.001166 | `outputs/metrics/evaluation_metrics.json` |
| Bits per character | 1.000841 | `outputs/metrics/evaluation_metrics.json` |
| Generalization gap | -0.035537 | `outputs/metrics/evaluation_metrics.json` |
| Top-1 next-character accuracy | 0.780337 | `outputs/metrics/evaluation_metrics.json` |
| Distinct-1 | 0.004370 | `outputs/metrics/evaluation_metrics.json` |
| Distinct-2 | 0.039709 | `outputs/metrics/evaluation_metrics.json` |
| Distinct-3 | 0.158188 | `outputs/metrics/evaluation_metrics.json` |
| Mean repeated 4-gram rate | 0.222371 | `outputs/metrics/evaluation_metrics.json` |
| Mean gradient norm before clipping | 0.767176 | `outputs/metrics/training_summary.json` |
| Maximum gradient norm before clipping | 11.639734 | `outputs/metrics/training_summary.json` |
| Detected loss spikes | 0 | `outputs/metrics/training_summary.json` |
| Nonfinite losses | 0 | `outputs/metrics/training_summary.json` |
| Skipped nonfinite-gradient updates | 10 | `outputs/metrics/training_summary.json` |
| Trainable parameters | 3,557,861 | `outputs/metrics/training_summary.json` |
| Training throughput | 80,067.67 characters/second | `outputs/metrics/training_summary.json` |
| Generation throughput | 518.08 characters/second | `outputs/metrics/evaluation_metrics.json` |
| Peak GPU memory allocated | 1,586.03 MB | `outputs/metrics/training_summary.json` |
| Peak GPU memory reserved | 1,754.00 MB | `outputs/metrics/training_summary.json` |
| Peak CPU resident memory | 1,660.16 MB | `outputs/metrics/training_summary.json` |
| Total training time | 3,305.32 seconds, or 55.09 minutes | `outputs/metrics/training_summary.json` |

Distinct-n metrics are calculated over the 27 sampled continuations and exclude the three deterministic greedy continuations. Because tokens are individual characters, Distinct-1 is naturally small and should not be interpreted using word-token expectations.

## Diversity by Temperature

| Temperature | Samples | Distinct-1 | Distinct-2 | Distinct-3 | Mean repeated 4-gram rate |
|---:|---:|---:|---:|---:|---:|
| 0.7 | 9 | 0.010889 | 0.082610 | 0.242079 | 0.274536 |
| 1.0 | 9 | 0.010222 | 0.089290 | 0.282909 | 0.237872 |
| 1.3 | 9 | 0.011778 | 0.103763 | 0.339804 | 0.154706 |

Increasing temperature improved measured diversity and reduced repeated 4-grams. Direct inspection showed that temperature 1.3 also produced more malformed words, broken grammar, and incoherent transitions. The best decoding setting therefore depends on balancing surface diversity against semantic quality.

## Generation Configuration

| Item | Value |
|---|---:|
| Prompts | 3 |
| Greedy samples | 3 |
| Sampled outputs | 27 |
| Samples per prompt and temperature | 3 |
| Temperatures | 0.7, 1.0, 1.3 |
| New characters per output | 500 |
| Total generated characters | 15,000 |
| Generation seed | 20,260,925 |

All generated outputs are preserved in `outputs/samples/generated_samples.json`. A normalized human-readable copy is stored in `outputs/samples/generated_samples.txt`.

## Reproducibility and Evidence

| Artifact | Path |
|---|---|
| Experiment configuration | `configs/gpt_char.yaml` |
| Locked environment | `requirements.txt` |
| Environment and hardware | `environment_manifest.txt` |
| Preprocessing implementation | `code/data.py` |
| Model implementation | `code/model.py` |
| Training implementation | `code/train.py` |
| Evaluation and generation | `code/evaluate_generate.py` |
| Executed demonstration notebook | `code/task1_char_gpt_demo.ipynb` |
| Automated tests | `code/tests/` |
| Preprocessing log | `logs/preprocessing_run_002.log` |
| Raw training log | `logs/training_run_001.log` |
| Evaluation log | `logs/evaluation_generation_run_001.log` |
| Best checkpoint | `checkpoints/best_model.pt` |
| Resume checkpoint | `checkpoints/last_checkpoint.pt` |
| Split manifest | `outputs/metrics/split_manifest.json` |
| Training history | `outputs/metrics/training_history.json` |
| Step metrics | `outputs/metrics/step_metrics.csv` |
| Training summary | `outputs/metrics/training_summary.json` |
| Evaluation metrics | `outputs/metrics/evaluation_metrics.json` |
| Loss plot | `outputs/plots/loss_curves.png` |
| Generated outputs | `outputs/samples/generated_samples.json` |
| Failure analysis | `failure_analysis.md` |

## Checkpoint Integrity

| Checkpoint | SHA256 |
|---|---|
| `best_model.pt` | `64AD99EFC0BF2E49959D3EF54BF5373A4C3A974F0FA333AE4A37E62FAC9D2BA3` |
| `last_checkpoint.pt` | `4C1A3DD0CB9A4547B557E2A975BC86E2DD4844C6B6AF0B79B3A24AAAA24AA4BA` |

The best checkpoint was reloaded independently, all parameters were confirmed finite, and an additional CUDA validation batch produced a finite loss of `0.703475`.

## Limitations

The model operates at the character level and has a 256-character context window. This makes it computationally inexpensive and able to represent arbitrary in-vocabulary strings, but it limits long-range narrative planning and provides no explicit word-level semantic representation. Greedy decoding can enter repetition loops, while high-temperature sampling can damage grammar and coherence. These behaviors are documented with genuine outputs in `failure_analysis.md`.