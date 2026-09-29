# Task 3 ? CycleGAN Image Style Transfer ? Anshika Goel

## Status

Repository scaffold created. No Task 3 model has been implemented, trained, or
evaluated yet.

## Problem

Train an independently designed CycleGAN for unpaired translation between:

- Monet images
- Photograph images

The implementation must contain two generators and two discriminators and must
support translation in both directions.

## Evidence policy

All serious training runs will preserve:

- configuration used for the run;
- raw unedited training log;
- exact hardware/software manifest;
- checkpoint identity and hashes;
- training and validation/evaluation metric artifacts;
- representative fixed generated samples;
- stability evidence including gradient norms and NaN counts.

Bulk datasets and generated image directories remain local unless a specific
assignment artifact must be committed.

## Kaggle integrity boundary

Any Kaggle submission must be generated directly by Anshika Goel's own trained
CycleGAN.

No manually edited, hand-picked, copied, externally sourced, hardcoded,
lookup-table, pretrained/foundation-model-generated, or test-pair-peeking output
may be used as the submitted translation.

Evaluation models required for metrics must remain separate from the CycleGAN
submission-generation path.
