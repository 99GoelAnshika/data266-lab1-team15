# Task 1 Results - Viraat Chaudhary

Status: Ten-epoch training, individual evaluation, executed notebook, three failure write-ups and the common comparison are present. Technical evidence was reviewed; member understanding and the Git submission remain separate responsibilities.

## Data and architecture

Dataset: `roneneldan/TinyStories`; revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; split seed: 2661502.

The existing Team 15 protocol counts fixed-length sequences: 100,000 training / 10,000 validation, context 256. Story-record groups are disjoint before sequence construction; only training text builds the vocabulary.

4 post-norm blocks; width 256; heads 8; feed-forward width 1024; ReLU; dropout 0.15; learned positions; untied output head. Attention, causal masking and layer normalization use explicit tensor operations.

This differs from Anshika's pre-norm/GELU/tied-output decoder. The comparison changes several factors and is not a controlled one-factor ablation.

## Training

Completed epochs: 10; batch size: 64; optimizer: AdamW; peak LR: 0.0003; minimum LR: 3e-05; warm-up: 10%; cosine decay; precision: bf16; best checkpoint epoch: 10.

## Required metrics

| Metric | Value |
|---|---:|
| training_cross_entropy_loss | 0.85758417 |
| final_epoch_training_cross_entropy_loss | 0.93269972 |
| validation_cross_entropy_loss | 0.86370217 |
| perplexity | 2.3719257 |
| bits_per_character | 1.2460588 |
| generalization_gap | 0.0061179997 |
| top1_next_character_accuracy | 0.72781211 |
| distinct_1 | 0.0045925926 |
| distinct_2 | 0.042158391 |
| distinct_3 | 0.1665923 |
| repeated_4gram_rate | 0.1746777 |
| gradient_norm_mean | 0.62475452 |
| gradient_norm_max | 3.3099968 |
| loss_spike_count | 0 |
| nonfinite_loss_count | 0 |
| nonfinite_gradient_count | 0 |
| parameter_count | 3273824 |
| training_tokens_per_second | 590870.73 |
| total_training_seconds | 451.62236 |
| peak_gpu_allocated_mb | 1758.9487 |
| peak_gpu_reserved_mb | 2034 |
| peak_cpu_rss_mb | 1674.8984 |
| generation_tokens_per_second | 696.01308 |



## Metric definitions and evidence

Epoch validation uses BF16 autocast during training; the final required checkpoint metrics are recomputed in FP32. This accounts for the small difference between epoch-10 validation CE 0.86375192 and final FP32 validation CE 0.86370217. Both evaluations disable dropout.

Primary training/validation cross-entropy and generalization gap use FP32 inference with dropout disabled on both full sets at the same selected checkpoint. The final-epoch online training loss is reported separately. Anshika's saved training loss/gap instead use epoch-average training loss, so these two definitions must not be silently mixed.

Perplexity = exp(validation CE); bits per character = validation CE / ln(2); generalization gap = validation CE minus training CE. Accuracy counts correct next-character predictions across all validation target positions.

Distinct-n is unique character n-grams divided by their total occurrences over the 27 sampled continuations. Prompts and cross-sample boundaries are excluded. Repeated 4-gram rate is the mean per-continuation duplicate-occurrence fraction. Greedy samples are preserved but excluded from sampled diversity totals.

Gradient norms are measured before clipping. A loss spike exceeds 1.5 times the median of the previous 20 minibatch losses. Nonfinite losses/gradients stop the run and are recorded in untouched logs. Training throughput uses processed target characters divided by completed training-pass time; generation throughput includes both greedy and sampled generated characters.

Sum of completed epoch training, validation, best-checkpoint and plot/metric writes; excludes preprocessing, setup, last-checkpoint write and uncheckpointed interrupted work

Evidence: metrics_report.csv; outputs/metrics/evaluation_metrics.json; training_summary.json; training_history.csv; step_metrics.csv; split_manifest.json; outputs/plots/loss_curves.png; outputs/samples/generated_samples.json; logs/; checkpoints/; environment_manifest.txt.

## Interpretation to review and defend

Explain why you selected the post-norm/ReLU/untied architecture and these hyperparameters. Inspect the curves and generated samples before writing conclusions about generalization, diversity, grammar or coherence. State the own-split and hardware limitations when comparing the models. The implementation does not infer qualitative failure observations from numeric metrics.

## References

Vaswani et al. (2017), Attention Is All You Need: https://arxiv.org/abs/1706.03762

Eldan and Li (2023), TinyStories: https://arxiv.org/abs/2305.07759

## Member's reviewed interpretation

### Architecture justification

The implemented configuration is a small causal character decoder with four post-norm blocks, width 256, eight attention heads and feed-forward width 1024. Each head uses 32 features, and the feed-forward layer expands the width by a factor of four before projecting it back. Explicit causal masking restricts each position to its own and earlier characters; residual connections preserve a path around each attention and feed-forward sublayer. Post-norm applies layer normalization after each residual addition. ReLU supplies the feed-forward nonlinearity. Learned token and position embeddings represent character identity and position. The untied output head has its own weights instead of sharing the input embedding weights, allowing the two mappings to be learned separately. For this run, the training vocabulary has 96 entries and the model has 3,273,824 parameters. These are technically reasonable design choices for a small scratch implementation, but this run does not demonstrate that they outperform pre-norm, GELU or tied weights.

### Hyperparameter justification

The context length is 256 characters, following the existing team sequence protocol. Batch size 64 and BF16 execution use the observed A100 GPU while keeping the assessed training at ten full epochs. AdamW uses a peak learning rate of 0.0003, weight decay 0.1 and beta values 0.9 and 0.95. A 10% linear warm-up reaches the peak learning rate gradually, followed by cosine decay to 0.00003. With 1,563 updates per epoch and 15,630 total planned updates, warm-up lasts 1,563 updates. Dropout 0.15 and weight decay provide regularization, and gradient clipping at norm 1.0 limits unusually large gradient updates. Gradient norms in the report are measured before clipping, so a recorded maximum above 1.0 is consistent with that configuration. Ten epochs process 256,000,000 target characters. These settings define this experiment; no hyperparameter search or claim of optimal settings is supported by the saved run.

### Observations

Both epoch training and validation losses decrease throughout the ten-epoch run. Epoch training loss decreases from 2.359281 to 0.932700, and epoch validation loss decreases from 1.602774 to 0.863752. Epoch 10 is the selected checkpoint. At that checkpoint, FP32 evaluation with dropout disabled gives training CE 0.857584 and validation CE 0.863702, hence a generalization gap of 0.006118 nats per character. Validation perplexity is 2.371926, BPC is 1.246059 and next-character accuracy is 72.7812%. The plotted training loss averages predictions made during learning with dropout active; it should not be substituted for the final checkpoint's inference training loss. The curves show no rising-validation-loss trend within these ten epochs, but the small inference gap alone does not establish strong generalization or good story quality. The metrics record zero detected loss spikes, nonfinite losses and nonfinite gradients. Thirty 500-character continuations were saved. Greedy outputs contain strong repetition, while sampled outputs also exhibit ambiguous character identities, grammatical errors and malformed words. Across the 27 sampled continuations only, character-level Distinct-1/2/3 are 0.004593, 0.042158 and 0.166592, and the mean repeated 4-gram rate is 0.174678. These diversity statistics do not measure narrative coherence.

### Limitations and next steps

The counts follow the repository's fixed-sequence convention: 100,000 training sequences and 10,000 validation sequences, rather than those numbers of complete stories. Story-record groups are disjoint before window construction and the vocabulary is built only from training text. This is one architecture, split seed and training seed, with qualitative generation evaluated using three prompts; it is not a controlled ablation or a broad benchmark. A 256-character attention context limits the history available during 500-character generation, although it does not by itself explain every observed failure. Anshika's validation split and training-loss definition differ, and her training hardware also differs, so individual validation metrics and hardware throughput cannot isolate architecture effects. Both frozen models were subsequently evaluated on the same official-validation holdout with FP32 and batch size 32. Anshika obtained common CE 0.69440728 and accuracy 78.1023%; Viraat obtained common CE 0.86098925 and accuracy 72.9764%. Under common CE and accuracy criteria, Anshika is the preferred checkpoint. The complete comparison and vocabulary-coverage limitations are documented in ../comparison/best_model_selection.md. Future work can compare one architecture choice at a time while holding other settings fixed, repeat training across seeds, and test the proposed decoding and name-consistency experiments. Those experiments have not been run, and their outcomes should not be stated as improvements.

## Common held-out comparison

| Metric | Anshika | Viraat |
|---|---:|---:|
| Common CE | 0.69440728 | 0.86098925 |
| Common next-character accuracy | 78.1023% | 72.9764% |

Under the documented common quality criteria, Anshika is the preferred Task 1
checkpoint. The [team selection write-up](../comparison/best_model_selection.md)
preserves both models, evidence links and limitations.

Implementation and analysis drafting used AI assistance. The member must review
and understand the rationale and interpretation; the automated review cannot
attest to personal authorship or satisfy the viva on the member's behalf.
