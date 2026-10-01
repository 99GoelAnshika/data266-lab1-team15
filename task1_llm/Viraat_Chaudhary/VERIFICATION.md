# Task 1 verification - completed run and exported evidence

## Observed Colab run

NVIDIA A100-SXM4-40GB; Python 3.13.15; PyTorch 2.11.0+cu128;
CUDA 12.8. BF16 training, FP32 final evaluation/generation.
The correctness command and synthetic GPU pause/resume/reload smoke cell
completed successfully in the executed notebook. The original source suite
contains 13 focused tests; this notebook does not preserve pytest's separate
console summary, so the successful checked command is the available run evidence.

Ten full epochs visited 100,000 training sequences each, with 1,563 updates per
epoch and 256,000,000 total target characters. Epoch 10 is the best checkpoint.
The accidental training-cell rerun reports `ALREADY_COMPLETE`; it does not
replace the original training evidence. The notebook's recovered training output
matches all 184 JSON events in the original unedited training log.

## Export review

The review passed 58 independent evidence checks. Details:
[submission_review_checks.json](outputs/metrics/submission_review_checks.json).

- All step losses, gradient norms, warm-up/cosine values, token counts and
  online epoch losses agree with the logs and summary.
- All 23 required/supplementary metric CSV rows agree with saved JSON. PPL,
  BPC, both gap definitions, every sampled diversity aggregate, per-temperature
  diversity and throughput denominators were independently recomputed.
- Best/last checkpoint hashes, frozen config/vocabulary/split provenance,
  finite tensors, parameter count and optimizer state agree. Best/last weights
  match at epoch 10. Anshika's finite checkpoint has 3,557,861 unique parameters;
  Viraat's has 3,273,824.
- Thirty 500-token continuations decode exactly through the saved vocabulary.
  All three failure excerpts are real exact substrings of different samples.
- The common CSV, manifest, raw log and executed notebook agree on both scores.
  The actual evaluator hash matches the verified LF/CRLF compatibility fix.
- Python sources and all notebook code cells compile; the downloaded notebook
  has no error outputs. A credential/personal-machine-path pattern scan found
  no matching values in the reviewed text artifacts.
- All 30 original Anshika Task 1 files remain byte-identical to the original
  repository ZIP. Tasks 2/3 are outside this package.

## Post-export maintenance

Missing processed arrays can now be restored from the frozen dataset. Candidate
text/splits/vocabulary and file hashes must match before publication; existing
evidence is preserved. Nine focused regression checks passed for restoring both
arrays, restoring one, preserving existing files, and rejecting changed
configuration, source, vocabulary, expected hashes or incomplete metadata.
Only a library fingerprint may vary when all substantive evidence agrees.
See [preprocessing_restore_checks.json](outputs/metrics/preprocessing_restore_checks.json).

These nine committed regression methods executed against the actual pure NumPy
preprocessing functions with PyTorch-dependent imports excluded in the review
environment. The full PyTorch/GPU suite was not rerun here. The model, training
algorithm, checkpoint tensors and recorded performance numbers are unchanged.

The other maintenance changes fix Markdown table spacing, preserve the existing
observed requirements lock during report refresh, retain future smoke attempt
logs, validate frozen checkpoint/config hashes during finalization, and update
stale documentation. The executed notebook is copied byte-for-byte from the
user's download, rather than fabricating new executions for these maintenance changes.

## Log provenance

The assessed preprocessing, training, accidental resume, evaluation/generation
and common-evaluation raw files are unchanged. The earlier common-evaluation
failure is retained as its actual screenshot, not a reconstructed raw log.
`logs/smoke_console_capture_20260930T191443.log` is an exact copy of the smoke
stdout stored in notebook cell 9; the original temporary smoke logfile was
removed by the original script. The copy is explicitly a console capture.
Future smoke attempts keep their original raw files in the member log folder.

## Review limits

The real dataset preprocessing and full model inference were not rerun during
this export review. Their evidence is the uploaded Colab notebook, raw logs,
metrics and manifests. Processed arrays are intentionally omitted from Git;
their hashes can be checked after deterministic restoration. Record isolation
is supported by the code and saved ranges, while exact text duplicates across
all records/splits were not exhaustively audited.

The 100K/10K counts use the existing team's fixed-sequence convention. The brief
does not explicitly identify that unit. Core architecture reasoning and analysis
must reflect the member's own understanding, as required by the assignment;
AI assistance was used here and technical checks cannot certify that requirement.
The member still needs the actual Git commits/pushes, team review/synthesis,
remaining tasks, combined PDF and individual viva. No marking guarantee follows
from passing the technical checks.

## Original development verification

Before the assessed run, CPU development checks used Python 3.12.14 and
PyTorch 2.8.0+cpu. Thirteen own-model and 22 partner tests passed in that earlier
development environment. Those are historical development checks, not a second
A100 training run or a claim of new full-suite execution in this review.

## Final reporting handoff

The current package includes the complete Task 1 technical report section with
all recorded metrics, six exact failure excerpts, common quality selection,
limitations and evidence links. It adds the partner's missing root metrics CSV
from her existing evaluation JSON, preserving all original partner artifacts.
The new standard-library verifier checks the merged repository's frozen files,
both checkpoints, notebook and derived report evidence. It does not rerun GPU
inference or attest to personal understanding, team approval or Git pushes.
The original 58-check run review and nine restoration checks retain their
historical scopes; this is a subsequent reporting/packaging check.
