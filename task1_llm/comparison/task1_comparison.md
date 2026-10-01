# Task 1 - Team 15 comparison

| Field | Anshika | Viraat |
|---|---|---|
| member | Anshika_Goel | Viraat_Chaudhary |
| layers | 5 | 4 |
| width | 240 | 256 |
| heads | 6 | 8 |
| feed_forward_width | 960 | 1024 |
| dropout | 0.1 | 0.15 |
| normalization | pre_layer_norm | post_layer_norm |
| activation | gelu | relu |
| tied_embeddings | True | False |
| batch_size | 32 | 64 |
| epochs | 10 | 10 |
| learning_rate | 0.0003 | 0.0003 |
| warmup_ratio | 0.05 | 0.1 |
| split_seed | 2661501 | 2661502 |
| epoch_average_training_ce | 0.7292673282814026 | 0.9326997169876099 |
| selected_checkpoint_training_ce | None | 0.8575841665267945 |
| inference_generalization_gap | None | 0.006117999744415292 |
| validation_ce | 0.6937300154685974 | 0.8637021662712098 |
| perplexity | 2.00116600957984 | 2.3719257223049572 |
| bits_per_character | 1.0008408530323694 | 1.2460588320845294 |
| legacy_epoch_average_gap | -0.035537312812805255 | -0.06899755071640012 |
| next_character_accuracy | 0.78033671875 | 0.727812109375 |
| distinct_1 | 0.00437037037037037 | 0.0045925925925925926 |
| distinct_2 | 0.03970904772507979 | 0.04215839085578565 |
| distinct_3 | 0.15818830879071843 | 0.16659229510635132 |
| repeated_4gram_rate | 0.22237126462478576 | 0.17467769580445636 |
| gradient_norm_mean | 0.7671757246292508 | 0.6247545246332789 |
| gradient_norm_max | 11.639734268188477 | 3.3099968433380127 |
| loss_spike_count | 0 | 0 |
| nonfinite_loss_count | 0 | 0 |
| nonfinite_gradient_count | 10 | 0 |
| training_tokens_per_second | 80067.67031501204 | 590870.7275939166 |
| total_training_seconds | 3305.320511900005 | 451.62236405800104 |
| peak_gpu_allocated_mb | 1586.033203125 | 1758.94873046875 |
| peak_gpu_reserved_mb | 1754.0 | 2034.0 |
| parameters | 3557861 | 3273824 |
| peak_cpu_rss_mb | 1660.1640625 | 1674.8984375 |
| generation_tokens_per_second | 518.0794406968815 | 696.0130767304037 |
| gpu | NVIDIA GeForce RTX 3060 Laptop GPU | NVIDIA A100-SXM4-40GB |
| evidence | Anshika_Goel/outputs/metrics/evaluation_metrics.json | Viraat_Chaudhary/outputs/metrics/evaluation_metrics.json |

Both rows use actual saved evidence. Validation splits differ; these are descriptive own-split results, not a controlled common-test quality ranking.
The table uses final-epoch online training CE and the corresponding legacy-style gap for consistency with Anshika's saved definitions. Viraat also reports full-training-set inference CE and an inference-mode gap separately.
None in the supplementary inference columns means that quantity was not recorded for that member; it is not an estimated value.
Training/generation speeds and memory were measured on different GPUs and are hardware-specific; architecture alone does not explain a speed difference.
The recorded common held-out evaluation and checkpoint selection are in [best_model_selection.md](best_model_selection.md). The complete report section, including both members' actual failure snippets, is [task1_team_report.md](task1_team_report.md). Retain both individual runs in the combined lab report.
