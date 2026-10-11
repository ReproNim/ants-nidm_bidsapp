# FireANTs registration diagnostics, 2026-09-07/08

These JSON summaries are all that remains of four scratch directories
(`ants-fireants-diag_sep07`, `diag4_sep08`, `diag5_sep08`, `diag6_sep08`),
deleted from `/orcd/scratch/bcs/002/yibei` on 2026-09-25 to reclaim 3.2 GB.

The deleted bulk was warped volumes and per-configuration `*_dseg.nii.gz`
outputs from registration parameter sweeps: SyN against Greedy, masked against
unmasked loss, learning rates, and label-interpolation variants. The summaries
kept here carry the measurements those volumes were produced to yield, so the
conclusions stay checkable without the 3.2 GB.

What the sweeps established, and where it now lives:

- Greedy became the committed default, and the five GPU-only registration
  defects (batch broadcast, OOM, identity init, cross-correlation background
  saturation, folding-gate sign) were found and fixed. Recorded in the project
  memory note "FireANTs registration root causes, Sep 7".
- The reusable part of the harness is `scripts/diag_folding.py`.
- Round 4 (`diag4_sep08__summary_round4_mask_geometry.json`) is the mask
  geometry check.

Regenerating any of the deleted volumes requires a GPU allocation; nothing in
the current test suite depends on them. The ad hoc `--method quick` checks
that read `ants-fireants-diag_sep07` were deleted with it; their question,
the label vocabulary, was answered separately and is fixed on
`fireants-engine`.
