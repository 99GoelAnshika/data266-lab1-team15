# Task 1 - Viraat Chaudhary - Team 15

The assessed run completed ten full epochs on an NVIDIA A100-SXM4-40GB.
Individual evaluation, thirty generated continuations, three failure cases,
and a common held-out comparison with Anshika are present. Git commits and the
combined Tasks 1-3 report are separate remaining team deliverables.

## Model and data

Four post-norm blocks, width 256, eight heads, feed-forward width 1024,
ReLU, dropout 0.15, learned character/position embeddings and an untied output
head. The training vocabulary has 96 entries; the model has 3,273,824 parameters.
Attention, causal masking and layer normalization are explicit tensor operations.
No pretrained weights or prebuilt Transformer/attention modules are used.

Anshika's decoder uses five pre-norm blocks, width 240, six heads, GELU,
dropout 0.10 and tied embeddings. Multiple factors differ, so this comparison
does not isolate the effect of one architecture decision.

The Team 15 protocol uses 100,000 training and 10,000 validation **sequences**
with 256 input/target positions each. Those are not counts of complete stories.
The brief does not explicitly identify the unit of its 100K/10K counts; this
implementation retains the existing repository's sequence-count convention.
Story-record groups are divided before window construction; the vocabulary is
built from training text only. Split seed 2661502 differs from Anshika's 2661501.
The dataset revision and split hashes are recorded in
[split_manifest.json](outputs/metrics/split_manifest.json).

## Results and evidence

| Individual metric | Value |
|---|---:|
| Selected-checkpoint training CE, FP32/dropout off | 0.85758417 |
| Validation CE, FP32/dropout off | 0.86370217 |
| Perplexity | 2.37192572 |
| Bits per character | 1.24605883 |
| Inference generalization gap | 0.00611800 |
| Next-character accuracy | 72.7812% |

Every required metric is in [metrics_report.csv](metrics_report.csv).
Definitions and interpretation are in [results.md](results.md); the actual
snippets are in [failure_analysis.md](failure_analysis.md). Evidence includes
[loss curves](outputs/plots/loss_curves.png),
[generations](outputs/samples/generated_samples.json), raw logs,
both checkpoints and [artifact_manifest.json](artifact_manifest.json).
The environment manifest and requirements lock retain observed Colab values.

## Reproduce from the repository root

Install Python and an appropriate CPU/CUDA PyTorch environment, then:

```bash
python -m pip install -r task1_llm/Viraat_Chaudhary/requirements.txt
```

Colab already provides CUDA PyTorch. The observed run used Python 3.13.15,
PyTorch 2.11.0+cu128 and BF16 training; evaluation/generation used FP32.
`requirements.lock.txt` records the complete observed environment for provenance;
it is not a cross-platform installation command.

One-command synthetic smoke test, without downloading TinyStories:

```bash
python task1_llm/Viraat_Chaudhary/code/smoke_test.py
```

New smoke attempts retain labelled logs. Their temporary datasets and weights
cannot overwrite the assessed run. The original Colab smoke console is also
preserved, with its provenance explained in [VERIFICATION.md](VERIFICATION.md).

Regenerate the omitted processed arrays from the pinned dataset:

```bash
python task1_llm/Viraat_Chaudhary/code/data.py
```

On a clone, this restores missing arrays only after their text, vocabulary,
split metadata and file hashes match the frozen manifest. Existing vocabulary,
manifest, checkpoints and other evidence are preserved. A mismatch stops the
operation before candidate files are published.

Check that completed training resumes safely:

```bash
python task1_llm/Viraat_Chaudhary/code/train.py --resume
```

The committed run is already complete; this reports `ALREADY_COMPLETE`.
For a genuinely new experiment, use a separate archived/member run folder.
Within that clean experiment, run `data.py`, `train.py` and
`evaluate_generate.py` in order. Evaluation refuses to overwrite existing metrics.
The frozen configuration controls sequence counts, architecture and hyperparameters.

Correctness and preprocessing-restoration tests:

```bash
python -m pytest task1_llm/Viraat_Chaudhary/code/tests -q
```

## Executed notebook

[code/task1_colab.ipynb](code/task1_colab.ipynb) is the exact downloaded
executed notebook, including the recovered original training output, the
successful Section 10 verification, the one-time comparison repair and the
successful common evaluation/export. It is a historical run record. For a fresh
clone, follow the CLI commands above; the comparison repair is already included
in the committed shared script and need not be installed again.

## Team comparison

```bash
python task1_llm/compare_task1.py
python task1_llm/evaluate_common.py
```

The recorded-results table preserves each member's own split and distinguishes
online epoch loss from Viraat's supplementary inference loss/gap. The common
evaluator uses the same 10,000 windows from the official validation split,
context 256, FP32 and batch 32 for both frozen checkpoints. Vocabulary coverage,
checkpoint hashes, evaluator hash and holdout text hash are recorded.

Anshika has lower common CE (0.69440728 versus 0.86098925) and higher common
accuracy (78.1023% versus 72.9764%). Under those quality criteria, present her
checkpoint as the preferred Task 1 model while retaining both members' work.
See [best_model_selection.md](../comparison/best_model_selection.md) and the
completed [Task 1 report section](../comparison/task1_team_report.md).
The shared evaluation was added after the independent training runs; it is
not claimed to be a preregistered ablation. Do not tune on this common holdout.
Training speed/memory comparisons use different GPUs and batch sizes.

## Review and ownership

Implementation support, analysis drafting and artifact review used AI assistance.
The member must personally review and understand the rationale and failure
interpretation for the viva. Automated checks verify evidence consistency;
they cannot certify personal authorship or understanding. The lab brief requires
the core architecture decisions and analysis to reflect the member's own
understanding. No performance or marking guarantee is implied by the checks.

See [RUN_GUIDE.md](RUN_GUIDE.md) for the local merge and eight logical
commit/push groups, and [VERIFICATION.md](VERIFICATION.md) for review limits.

## References

- Vaswani et al. (2017), Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Eldan and Li (2023), TinyStories: https://arxiv.org/abs/2305.07759
- Dataset: https://huggingface.co/datasets/roneneldan/TinyStories/tree/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
