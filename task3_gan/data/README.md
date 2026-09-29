# Task 3 Shared Data

The Task 3 image data are local shared inputs and are not committed to Git.

Expected local domain directories:

- `monet_jpg/` ? real Monet-domain images
- `photo_jpg/` ? real photograph-domain images

These two domains are treated as **unpaired** for CycleGAN training.

Do not create or use hidden image pairings between the two domains for training,
model selection, or test-set inspection.

The actual image files are intentionally excluded from Git. Dataset acquisition,
validation, checksums/counts, and split provenance will be recorded separately
before training begins.
