# FireANTs registration engine for ants-nidm BIDS App

**Date:** 2026-09-01
**Branch:** `fireants-engine`
**Status:** approved design, pending implementation plan

## 1. Motivation

`--method fusion` spends nearly all of its wall-clock in a loop of twenty
`ants.registration(type_of_transform='SyNRA')` calls that register each
OASIS-TRT-20 atlas to the target brain
([`src/antspy/wrapper.py:415-439`](../../../src/antspy/wrapper.py)). On CPU this
dominates per-subject cost and sets the ceiling on how fast a BABS array can
clear a site.

[FireANTs](https://github.com/rohitrango/FireANTs) (v1.5.1) performs
diffeomorphic registration on GPU with Riemannian adaptive optimisation,
reporting ~250 ms per image pair and supporting batched registration of many
moving images against one fixed image. That maps exactly onto the shape of the
fusion loop.

## 2. What FireANTs does and does not provide

Audited against a clone of `main` (68 `.py` files, plus docs and notebooks).

**Provides:**

- `RigidRegistration`, `AffineRegistration`, `GreedyRegistration`,
  `SyNRegistration` (`fireants/registration/`)
- `Image` / `BatchedImages` with a `device` argument, `is_segmentation=True`
  label handling, and ANTs-compatible warp export (`write_image` with
  `PERMITTED_ANTS_WARP_EXT`, `utils/util.save_itk_affine`)
- Masked similarity losses (`losses/maskedutils.py`, `io/imagemask.py`)
- An evaluation harness: `scripts/evalutils.py` provides
  `compute_dice_coefficient_onehot`, `compute_surface_distances`,
  `compute_robust_hausdorff`, `compute_surface_dice_at_tolerance`
- `utils/imageutils.jacobian`, giving the fraction of non-positive Jacobian
  determinants — a direct folding / invertibility check
- A reference label-propagation implementation on OASIS with tuned
  hyperparameters (`scripts/pairwise/oasis.py`)
- A fused generic-label grid sampler that returns argmax labels without
  materialising a one-hot volume (`interpolator/fused_grid_sample_genericlabel.py`)

**Does not provide.** A case-insensitive search for
`atropos|n4_bias|biasfield|skull|joint.label|jointfusion|label_fusion` across
the entire repository returns two incidental prose mentions and no
implementation. FireANTs has no bias-field correction, no tissue
classification, no brain extraction, and no label-fusion voting.

Consequently the container keeps ANTs 2.5.4 and `antspyx`. N4,
`antsBrainExtraction.sh`, and `ants.joint_label_fusion` are unchanged by this
work.

## 3. Scope

FireANTs replaces the ANTs registration calls outright. There is no dual
backend and no `--engine` flag: the ANTs registration path is deleted, not
kept alongside. Two phases, sequenced by risk; phase 2 does not begin until
phase 1 passes the tier-3 gate in section 8.

**Phase 1 — the fusion registration loop.** Replace the twenty
`ants.registration` + `ants.apply_transforms` calls at
`wrapper.py:415-439`. This is where the compute is.

**Phase 2 — the template registration in brain extraction.** Replace the
`Affine` + `SyN` pair at `wrapper.py:815-838`. Deferred, not dropped: its
output is `TemplateToSubjectTransforms`, a list of ANTs transform file paths
consumed verbatim by `ants.apply_transforms` downstream. Reproducing it
requires emitting ANTs-compatible warp and affine files with correct
composition order and inversion. The comment block at `wrapper.py:839-851`
records that getting this direction wrong once produced an 11.5 L brain
volume. Attempting it after phase 1 means the round-trip conversion and the
Jacobian gate are already validated.

**Out of scope:** N4, brain extraction, Atropos, fusion voting, the NIDM
submodule, the multi-run T1w glob blocker tracked separately, and any change
to output layout.

## 4. Architecture

`wrapper.py` is 871 lines. Rather than inlining a torch pipeline into it,
extract the seam into a new module:

```
src/registration/
    __init__.py    # warp_atlases(...) — the only public entry point
    fireants.py    # FireANTs implementation
```

### Interface

```python
def warp_atlases(
    target: "ants.ANTsImage",       # masked_brain
    target_mask: "ants.ANTsImage",  # BrainExtractionMask
    atlas_paths: list[Path],
    label_paths: list[Path],
    *,
    device: str = "cuda",
    batch_size: int = 0,            # 0 = all atlases in one batch
    label_interpolation: str = "onehot",   # or "nearest"
) -> tuple[list["ants.ANTsImage"], list["ants.ANTsImage"]]:
    """Return (warped atlas images, warped atlas labels) in target space."""
```

Return types are ANTsImages so `ants.joint_label_fusion` consumes them
unchanged. The fusion loop in `segment_image` collapses to a single
`warp_atlases(...)` call. The module is separate from `wrapper.py` for
testability and to keep an already-large file from growing, not to support a
second backend.

### Registration

Follows `scripts/pairwise/oasis.py`, using the authors' structure rather than
invented settings: `AffineRegistration(scales=[4,2,1],
iterations=[200,100,50])` to initialise, then `SyNRegistration(scales=[4,2,1],
iterations=[200,100,50], deformation_type='compositive', optimizer='Adam')`
with `init_affine` from the affine stage. SyN rather than Greedy, as the
closer analogue to the `SyNRA` it replaces. Learning rate, `cc_kernel_size`,
`smooth_grad_sigma` and `smooth_warp_sigma` are exposed as options with the
oasis.py values as defaults, so tier-3 disagreement can be tuned without code
changes.

`target_mask` is passed to the masked loss path. The ANTs call being replaced
passed no mask; restricting the metric to the brain is expected to help rather
than hurt, and is recorded here as an intentional difference from the
reference derivative.

**Batching.** All twenty atlases register against one fixed target in a single
`BatchedImages`. `batch_size` allows chunking on smaller cards. On `pi_satra`
(A100 80 GB / H100) full batch is the target configuration.

**Label warping.** Default `label_interpolation="onehot"` uses the fused
generic-label sampler, which interpolates one-hot and returns argmax without
materialising 97 channels. The replaced ANTs path used `nearestNeighbor`;
one-hot is generally the better choice, but it is a real source of Dice
difference against the reference derivative and must not be mistaken for a
registration difference when reading tier-3 results. `"nearest"` is retained
precisely so the two effects can be separated — see section 9.

### Conversion boundary

ANTsPy holds images in memory; FireANTs loads through SimpleITK, whose numpy
axis order differs. The ANTsImage → FireANTs → ANTsImage round trip is the
highest-risk part of this change and the most likely home for a silent
orientation flip. Atlas images and labels are read from disk by FireANTs
directly (they exist as files under `self.brains_dir` / `self.labels_dir`);
only `target`, `target_mask`, and the returned warped volumes cross the
boundary. Round-trip fidelity is asserted in tier 1.

## 5. CLI

Added to `parse_arguments` in `src/run.py` and threaded through
`ANTsSegmentation`:

| Flag | Values | Default | Behaviour |
|---|---|---|---|
| `--device` | `cuda`, `cpu` | `cuda` | FireANTs compute device |
| `--fireants-batch-size` | int | `0` (all) | Atlases per registration batch |
| `--label-interpolation` | `onehot`, `nearest` | `onehot` | Atlas label warping method |

No `--engine` flag: FireANTs is the only registration backend.

`--method fusion` with `--device cuda` and no visible GPU **fails before any
processing begins**, with a message naming the missing device. It does not
fall back to CPU: FireANTs on CPU is slower than the ANTs code it replaces,
and a silent fallback inside a BABS array turns a fast job into a many-hour
one while still exiting 0. `--device cpu` remains available as an explicit
opt-in for tests and small debugging runs.

Phase 1 touches only the fusion loop, so `--method quick` still runs without a
GPU. That changes in phase 2, when the template registration also moves.

Rollback, if it is ever needed, is the previously delivered container plus the
existing 38-subject derivative on disk — not a runtime flag.

## 6. Container

One image, per decision in brainstorming. Both `Dockerfile` and `Singularity`
gain, on top of the existing ANTs 2.5.4 payload and `antspyx`:

- CUDA-enabled `torch>=2.3.0`
- `fireants==1.5.1`
- the `fused_ops` CUDA extension, built from source
  (`cd fused_ops && python setup.py build_ext && python setup.py install`).
  `pip install fireants` alone does **not** provide `fireants_fused_ops`, and
  without it the generic-label sampler is unavailable.

Runtime requires `apptainer --nv` and `--gres=gpu:1`. `run_container.sh` gains
an `--nv` passthrough. BABS job configuration targets `pi_satra`
(h100:4 / a100:8, 2-day limit).

`Dockerfile` and `Singularity` are near-duplicates that have drifted before. A
test asserts the ANTs, torch, and fireants versions match across both files.

## 7. Testing

**Tier 1 — CI, no GPU, every push.** Real assertions on CPU tensors, not
mocks:

- `--method fusion --device cuda` on a GPU-less host exits non-zero before any
  image is read, with the missing device named in the message
- `--device cpu` is accepted and does not raise
- **round-trip fidelity**: an ANTsImage with a non-identity direction matrix
  and anisotropic spacing survives ANTsImage → FireANTs → ANTsImage with voxel
  data, spacing, origin, and direction intact
- warped labels contain only values present in the input labels — no
  interpolated intermediates, under both `onehot` and `nearest`
- `batch_size` chunking returns the same number of atlases in the same order
  as a single full batch

**Tier 2 — one GPU job on `pi_satra`, one real T1w.** Full pipeline end to
end; outputs land in the `sub-XX/anat`, `sub-XX/stats` layout with `nidm.ttl`
and `ants_cde.ttl` present.

**Tier 3 — agreement, 3 Caltech subjects.** Reference:
`/orcd/data/satra/002/datasets/simple2_datalad/study-ABIDE/site-Caltech/derivatives/babs-ants-nidm`
(38 subjects, delivered and gate-passed). Metrics computed with FireANTs'
`scripts/evalutils.py`.

The reference segmentations already exist on disk, so no ANTs registration run
is needed at comparison time. This is why deleting the ANTs registration path
costs nothing for validation: the comparison is against delivered files, not
against a live second backend.

## 8. Acceptance criteria

Derived from the 38-subject reference derivative rather than chosen a priori.
Across 38 subjects and 97 labels:

| Observation | Consequence |
|---|---|
| Labels 15 and 2033 present in only 2/38 subjects, median volume 0 | Dice degenerate — exclude |
| Label 69: median 32 mm³, CV 99.8% across subjects under ANTs alone | Already unstable with no change — cannot gate |
| Labels 64, 63: median 126 / 144 mm³, CV 41% / 27% | Small-structure Dice intrinsically low |
| 87 of 97 labels ≥ 2000 mm³; largest structures CV 10–17% | Where 0.85 is a real and fair constraint |

Gate for each of the 3 subjects:

1. **Excluded** — labels present in <90% of reference subjects (15, 2033): not
   evaluated.
2. **Primary, median volume ≥ 2000 mm³ (87 labels):** within each subject, the
   median Dice across these 87 labels ≥ 0.85, and no individual label below
   0.75.
3. **Relaxed, 250–2000 mm³ (labels 50, 55, 30, 56, 49):** each label ≥ 0.70
   in each subject.
4. **Report-only, < 250 mm³ (labels 69, 64, 63):** reported, not gated.
5. **Volume agreement:** per-label |Δvolume| ≤ 2× that label's inter-subject
   standard deviation in the reference cohort. Self-calibrating, so noisy
   structures are not held to the same bar as stable ones.
6. **BVOL** within [1.29, 1.94] L, the observed envelope of the delivered
   cohort (mean 1.63 L).
7. **Diffeomorphism:** fraction of non-positive Jacobian determinants < 0.1%
   for every atlas registration. A folding warp is bad independently of Dice.

Surface Dice and 95th-percentile Hausdorff are reported alongside for every
gated label. They are not gate criteria in this iteration — no reference
distribution exists for them yet — but they catch boundary degradation that
volumetric Dice hides on large structures, and establish the baseline for
future runs.

Failing the gate blocks the merge. With no ANTs registration path left in the
app, there is no half-landed state to fall back on: either the FireANTs output
meets the bar against the delivered derivative, or the branch does not merge.
Hyperparameters (section 4) and `--label-interpolation` are the tuning levers
before declaring failure.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Silent orientation flip in the ANTsImage ↔ FireANTs conversion | Tier-1 round-trip test with non-identity direction and anisotropic spacing |
| Registration converges to a plausible-but-wrong warp; app exits 0 with a well-formed TTL | Jacobian gate, BVOL envelope, per-label Dice against a known-good reference — all three, not one |
| GPU OOM at full 20-atlas batch on smaller cards | `--fireants-batch-size`; default full batch only because `pi_satra` has 80 GB cards |
| `Dockerfile` / `Singularity` drift | Version-parity test across both files |
| `fused_ops` build fails or is skipped, silently disabling the generic-label sampler | Container build asserts `import fireants_fused_ops` succeeds |
| Label-warping method differs (one-hot argmax vs nearestNeighbor), confounding tier-3 attribution | `--label-interpolation nearest` reproduces the reference's method, isolating the registration change from the interpolation change without needing an ANTs engine |
| No in-app fallback if FireANTs regresses on some subject | The delivered 38-subject derivative and its container remain on disk; rollback is redeploying that image, and the tier-3 gate blocks merge rather than discovering this in production |

## 10. Phase 1 deliverables

1. `src/registration/` module implementing `warp_atlases` on FireANTs
2. `wrapper.py` fusion loop reduced to one `warp_atlases` call, with the
   `ants.registration` / `ants.apply_transforms` calls deleted
3. `--device`, `--fireants-batch-size`, `--label-interpolation` in `run.py`
4. Updated `Dockerfile`, `Singularity`, `requirements.txt`, `run_container.sh`
5. Tier-1 tests in `tests/test_registration.py`
6. Tier-2 and tier-3 harnesses as submittable `sbatch` scripts under `scripts/`
7. Tier-3 report: per-label Dice, volume deltas, Jacobian, BVOL for 3 subjects
