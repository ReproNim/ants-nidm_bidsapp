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

FireANTs replaces every registration and transform-application call in the
app. There is no dual backend, no `--engine` flag, and no phasing: the ANTs
registration path is deleted, not kept alongside.

All five sites move:

| `wrapper.py` | Call | Purpose |
|---|---|---|
| 417 | `ants.registration` (`SyNRA`) ×20 | atlas → target, fusion loop |
| 431 | `ants.apply_transforms` | atlas labels → target |
| 815 | `ants.registration` (`Affine`) | subject → template |
| 827 | `ants.registration` (`SyN`) | subject → template |
| 369 | `ants.apply_transforms` | template labels → subject (`quick`) |

Moving all of them together is safer than splitting them, not riskier. The
`TemplateToSubjectTransforms` producer (`wrapper.py:869`) and its sole
consumer (`wrapper.py:365`) are both inside `wrapper.py`. If only the producer
moved, FireANTs would have to emit ANTs-compatible warp and affine files with
correct composition order and inversion so `ants.apply_transforms` could read
them back — the exact serialisation round-trip that the comment block at
`wrapper.py:839-851` records getting wrong once, producing an 11.5 L brain
volume. Moving both ends keeps the transform as an in-memory warp that is
never serialised to ANTs format at all.

**What remains on ANTs**, because FireANTs has no implementation (section 2):
N4 bias correction, `antsBrainExtraction.sh`, `ants.joint_label_fusion`
voting, and `ants.image_read` / `image_write` for I/O. The container therefore
keeps ANTs 2.5.4 and `antspyx`.

**Out of scope:** the NIDM submodule, the multi-run T1w glob blocker tracked
separately, and any change to output layout.

## 4. Architecture

`wrapper.py` is 871 lines. Rather than inlining a torch pipeline into it,
extract the seam into a new module:

```
src/registration/
    __init__.py    # public API: warp_atlases, register_to_template, apply_warp
    fireants.py    # FireANTs implementation
    handoff.py     # file-based ANTs <-> FireANTs handoff + geometry guard
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
    label_interpolation: str = "genericlabel",   # or "nearest"
) -> tuple[list["ants.ANTsImage"], list["ants.ANTsImage"]]:
    """Return (warped atlas images, warped atlas labels) in target space."""
```

Return types are ANTsImages so `ants.joint_label_fusion` consumes them
unchanged. The fusion loop in `segment_image` collapses to a single
`warp_atlases(...)` call. The module is separate from `wrapper.py` for
testability and to keep an already-large file from growing, not to support a
second backend.

Two further entry points replace the template-registration sites:

```python
def register_to_template(
    subject: "ants.ANTsImage",      # brain_image, skull-stripped
    template: "ants.ANTsImage",     # T_template0_BrainCerebellum
    *,
    device: str = "cuda",
) -> "TemplateWarp":
    """Register subject -> template. The returned object carries the
    template -> subject direction, matching what consumers need."""

def apply_warp(
    warp: "TemplateWarp",
    moving: "ants.ANTsImage",
    reference: "ants.ANTsImage",
    *,
    interpolation: str = "genericlabel",
) -> "ants.ANTsImage":
    ...
```

`TemplateWarp` is an in-memory object holding the composed affine and
displacement field as torch tensors. It replaces
`thickness_results['TemplateToSubjectTransforms']`, which today is a list of
ANTs transform *file paths*. Because both the producer and the consumer move
in the same change, the warp is never written to disk in ANTs format and the
`whichtoinvert` inference that `ants.apply_transforms` performs disappears
along with it.

**Direction is the load-bearing detail.** `register_to_template` registers
subject → template but exposes the template → subject direction, because every
consumer pulls template-space images into subject space. `wrapper.py:839-851`
documents an 11.5 L brain volume caused by getting exactly this backwards.
`TemplateWarp` therefore exposes one direction only, rather than a
transform list whose orientation depends on which attribute the caller
picked.

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

`target_mask` is carried in the signature because `ants.joint_label_fusion`
needs it downstream, but it is **not** wired into the similarity loss. An
earlier draft of this spec proposed doing so as an improvement over the ANTs
call, which passes no mask. That was redundant: `masked_brain` is already
`BrainSegmentationN4 * BrainExtractionMask` (`wrapper.py:352`) and the
OASIS-TRT-20 atlases are brain-only volumes, so both sides of the metric are
already skull-stripped. Adding a mask would change nothing and would be one
more difference from the reference derivative for no gain.

**Batching.** All twenty atlases register against one fixed target in a single
`BatchedImages`. `batch_size` allows chunking on smaller cards. On `pi_satra`
(A100 80 GB / H100) full batch is the target configuration.

**Label warping.** Two strategies, both required:

- `genericlabel` (default) — `Image(is_segmentation=True, is_onehot=False)`,
  which sets `interpolation_mode='genericlabel'` and routes to the fused
  sampler. It returns argmax labels directly, shape `[1, 1, H, W, D]`, without
  materialising 97 channels.
- `nearest` — plain nearest-neighbour `grid_sample` on the raw label volume.
  Reproduces the `interpolator='nearestNeighbor'` behaviour of the ANTs path
  being replaced, and is the tier-3 control.

**The `genericlabel` path silently changes shape without FFO, and the caller
must handle it.** `fireants/io/image.py:145-147` reads:

```python
# When FFO is not available, force one-hot mode so generic-label path is never used
if not fi.use_ffo:
    is_onehot = True
```

So when `fused_ops` is absent — or whenever tensors are on CPU, since
`augment_registry_backend` force-disables FFO for CPU tensors — `is_onehot` is
forced true, `interpolation_mode` becomes `None`, and the warp returns a
**one-hot `[1, C, H, W, D]` tensor** rather than `[1, 1, H, W, D]` labels. The
`RuntimeError` in `GridSampleDispatcher.__call__` is therefore unreachable
through this route: FireANTs degrades instead of failing.

`warp_atlases` must detect which shape it received and argmax when the tensor
is one-hot, rather than assuming labels. Returning a one-hot volume to
`ants.joint_label_fusion` would be a silent, well-formed disaster. The active
path is logged at INFO on every run so the mode is visible in job logs.

**Labels must be remapped to a contiguous range before warping.**
`utils/imageutils.integer_to_onehot` allocates `max_label + 1` channels:

```python
if max_label is None:
    max_label = image.max()
...
onehot = torch.zeros((num_labels, *image.shape), ...)
for i in range(max_label+1):
```

DKT31 label ids are sparse — 4, 11, 15, 30, then 1001–1035 and 2001–2035 — so
`max_label` is 2035 and the allocation is **2036 channels, not 97**. At 256³
float32 that is roughly 136 GB, which fails on any card. This applies to the
`genericlabel` path too, since it also one-hots internally.

`warp_atlases` therefore builds a contiguous mapping from the 97 distinct label
ids to 0…96 (0 reserved for background), passes it through FireANTs'
`seg_preprocessor` hook with `background_seg_label=-1` and
`max_seg_label=96`, and inverts the mapping after argmax. With
`background_seg_label=-1` the one-hot is built with `num_labels = max_label+1`
and no channel is skipped, so channel *c* corresponds to remapped label *c*
exactly — the inversion is a plain lookup. Using `background_seg_label=0`
instead would skip a channel and shift every index by one; it must not be used.

The mapping is derived from the union of label ids across all 20 atlas files,
computed once per run, and asserted to round-trip (`inverse[forward[x]] == x`)
before any warping.

With remapping in place the one-hot fallback costs 97 channels (~6.5 GB at
256³, workable on an 80 GB card and trivial on the small synthetic volumes tier
1 uses) instead of 2036.

`genericlabel` is generally the better choice, but it is a real source of Dice
difference against the reference derivative and must not be mistaken for a
registration difference when reading tier-3 results. `nearest` reproduces the
reference's method exactly, which is how the two effects get separated — see
section 9.

`--device cpu --label-interpolation genericlabel` is permitted and degrades to
the one-hot path described above, with a warning logged. It is not rejected,
because it works.

### Conversion boundary — file-based, never in-memory

ANTsPy holds images in memory; FireANTs loads through SimpleITK, whose numpy
axis order differs. Hand-rolling that conversion — pulling a numpy array out
of an ANTsImage and reattaching spacing, origin and a direction matrix on the
other side — is the highest-risk part of this change and the most likely home
for a silent orientation flip.

**So the design does not do it.** Every crossing goes through a NIfTI file on
`TMPDIR` (`/work` in the container). NIfTI carries the affine canonically, and
ANTsPy and SimpleITK are both well-tested readers and writers of it; neither
side reconstructs a header from loose parts. This is the pattern the repo
already uses for `antsBrainExtraction.sh` at `wrapper.py:769-795`
(`ants.image_write` → subprocess → `ants.image_read`), so it introduces no new
mechanism.

What actually crosses, per subject:

| Direction | Volumes | Mechanism |
|---|---|---|
| Atlas images and labels → FireANTs | 40 | none — read from `self.brains_dir` / `self.labels_dir` by path |
| Template, template labels → FireANTs | 2 | none — read from `self.template_dir` by path |
| `target`, `target_mask`, `brain_image` → FireANTs | 3 | `ants.image_write` to TMPDIR |
| Warped atlases and labels → ANTsPy | 40 | FireANTs `write_image` to TMPDIR, `ants.image_read` back |
| Warped template labels → ANTsPy (`quick`) | 1 | same |

Roughly 300 MB of scratch I/O per subject, seconds of wall-clock against the
twenty SyN registrations being eliminated. Note that 42 of the 86 volumes need
no conversion at all: the atlases and templates are already NIfTI files on
disk, so FireANTs reads them by path and ANTsPy never touches them.

`TemplateWarp` (above) is the one thing that does *not* cross this boundary —
it is a torch tensor produced and consumed entirely within FireANTs, which is
precisely why it never needs ANTs-format serialisation. Images cross as files;
the warp does not cross at all.

**Two concrete footguns, recorded so they are not rediscovered:**

- `fireants.io.Image.__init__` accepts an `orientation` argument that calls
  `sitk.DICOMOrient` and *rewrites the header*. It must be left at `None`.
  Passing it would reorient silently and every downstream volume would be
  wrong in a way that still looks like a valid image.
- `Image.__init__` also accepts `spacing`, `direction`, `origin` and `center`
  overrides. These must be left unset so the header comes from the file.

**Guard, in production and not only in tests.** A helper
`assert_geometry_match(a, b, tol)` compares shape, spacing, origin and
direction between an ANTsImage and what came back across the boundary, and
raises rather than proceeding. It runs on every crossing, not just under
pytest. A registration engine that silently returns a reoriented volume
produces a well-formed segmentation, a well-formed `nidm.ttl`, and exit 0 —
the failure mode this repo has already been bitten by.

**Why the fusion call stays on ANTsPy.** The container ships the full ANTs
2.5.4 binaries including `antsJointFusion`, so calling that CLI with the
warped file paths is possible. It saves no I/O: `ants.joint_label_fusion`
passes in-memory pointers (`myargs["g-MULTINAME-%i"] = atlas_list[k]`, then
`process_arguments` → `get_pointer_string()` → `get_lib_fn("antsJointFusion")`),
which is the same compiled code the CLI runs. Either way the 40 warped volumes
are written once by FireANTs and read once — by ANTsPy in one case, by the
binary's own ITK readers in the other.

The only difference is peak RAM: the CLI keeps 40 volumes (~2.7 GB) out of the
Python process. That is not a saving worth pursuing, because the current code
already holds all 40 simultaneously (`warped_images.append(...)` at
`wrapper.py:437`) — the file-based boundary matches today's memory profile
rather than regressing it.

Against that, switching would mean hand-rolling what ANTsPy already does
correctly: the `-MULTINAME-` argument scheme, `mktemp` allocation and
read-back for each probability image, and the flag mapping for `beta`, `rho`,
`r_search`, `no_zeroes` and `max_lab_plus_one`. The wrapper consumes all four
return values (`segmentation`, `intensity`, `probabilityimages`,
`segmentation_numbers`) and `_organize_bids_output` writes the probability
images out, so all of it would have to be reproduced. Errors there are silent:
a wrong flag yields a segmentation, just not the right one.

The fusion call therefore stays exactly as it is.

## 5. CLI

Added to `parse_arguments` in `src/run.py` and threaded through
`ANTsSegmentation`:

| Flag | Values | Default | Behaviour |
|---|---|---|---|
| `--device` | `cuda`, `cpu` | `cuda` | FireANTs compute device |
| `--fireants-batch-size` | int | `0` (all) | Atlases per registration batch |
| `--label-interpolation` | `genericlabel`, `nearest` | `genericlabel` | Atlas label warping. Without built `fused_ops`, or on CPU, `genericlabel` degrades to a one-hot return that `warp_atlases` argmaxes; a warning is logged |

No `--engine` flag: FireANTs is the only registration backend.

`--method fusion` with `--device cuda` and no visible GPU **fails before any
processing begins**, with a message naming the missing device. It does not
fall back to CPU: FireANTs on CPU is slower than the ANTs code it replaces,
and a silent fallback inside a BABS array turns a fast job into a many-hour
one while still exiting 0. `--device cpu` remains available as an explicit
opt-in for tests and small debugging runs.

Both `--method fusion` and `--method quick` now require a GPU, since the
template registration moves along with the fusion loop.

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
  and anisotropic spacing survives the file-based handoff — write, load in
  FireANTs, write back, read in ANTsPy — with voxel data, spacing, origin and
  direction intact
- `assert_geometry_match` raises on a deliberately reoriented volume, and does
  not raise on an identical one — the guard is tested, not assumed
- `fireants.io.Image` is constructed without `orientation`, `spacing`,
  `direction`, `origin` or `center` overrides (asserted on the call, since
  passing any of them rewrites the header)
- warped labels contain only values present in the input labels — no
  interpolated intermediates — under both `nearest` and `genericlabel`. On CPU
  `genericlabel` exercises the degraded one-hot path, which is precisely the
  branch most likely to be wrong
- given a one-hot `[1, C, H, W, D]` warp result, `warp_atlases` returns
  `[1, 1, H, W, D]` labels whose values are the original label ids, not
  channel indices
- `batch_size` chunking returns the same number of atlases in the same order
  as a single full batch
- **warp direction**: with a synthetic subject and template differing by a
  known translation, a marker placed in template space lands at the predicted
  subject-space voxel after `apply_warp`. This replaces
  `tests/test_wrapper.py:198-214`, which asserts
  `TemplateToSubjectTransforms == invtransforms`. That assertion is structural
  and cannot survive the representation change, but the bug it guards — the
  11.5 L brain volume from an inverted template↔subject mapping — is the most
  consequential regression in this repo's history. The replacement must be
  behavioural and must fail if the direction is flipped.

**Tier 2 — one GPU job on `pi_satra`, one real T1w.** Full pipeline end to
end for **both** `--method fusion` and `--method quick`, since the template
registration `quick` depends on is also being replaced. Outputs land in the
`sub-XX/anat`, `sub-XX/stats` layout with `nidm.ttl` and `ants_cde.ttl`
present.

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

The delivered derivative was produced with `--method fusion`, so no reference
exists for `--method quick`. It is covered by tier 2 (runs, produces a
plausible segmentation, BVOL within envelope) and by the tier-1 direction
test, but it has no Dice gate. This is a real coverage gap and is recorded as
such rather than papered over.

Failing the gate blocks the merge. With no ANTs registration path left in the
app, there is no half-landed state to fall back on: either the FireANTs output
meets the bar against the delivered derivative, or the branch does not merge.
Hyperparameters (section 4) and `--label-interpolation` are the tuning levers
before declaring failure.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Silent orientation flip crossing between ANTs and FireANTs | No in-memory conversion exists to get wrong: every crossing is a NIfTI file, the same mechanism already used for `antsBrainExtraction.sh`. Backed by `assert_geometry_match` on every crossing in production, plus tier-1 tests on the handoff and on the guard itself |
| Registration converges to a plausible-but-wrong warp; app exits 0 with a well-formed TTL | Jacobian gate, BVOL envelope, per-label Dice against a known-good reference — all three, not one |
| Template↔subject warp direction inverted, reproducing the 11.5 L brain-volume bug | Behavioural tier-1 direction test; `TemplateWarp` exposes a single direction rather than a list whose orientation depends on caller choice |
| GPU OOM at full 20-atlas batch on smaller cards | `--fireants-batch-size`; default full batch only because `pi_satra` has 80 GB cards |
| `Dockerfile` / `Singularity` drift | Version-parity test across both files |
| `fused_ops` build fails or is skipped | Silent: FireANTs degrades to one-hot rather than raising, changing the return shape and costing 97× memory at 256³. Caught at build time (the container asserts `import fireants_fused_ops` imports) and at runtime (the active path is logged, and `warp_atlases` handles both shapes explicitly) |
| Label-warping method differs (fused argmax vs nearestNeighbor), confounding tier-3 attribution | `--label-interpolation nearest` reproduces the reference's method exactly, isolating the registration change from the interpolation change without needing an ANTs engine |
| No in-app fallback if FireANTs regresses on some subject | The delivered 38-subject derivative and its container remain on disk; rollback is redeploying that image, and the tier-3 gate blocks merge rather than discovering this in production |

## 10. Deliverables

1. `src/registration/` implementing `warp_atlases`, `register_to_template`,
   `apply_warp`, the file-based handoff, and `assert_geometry_match`
2. `wrapper.py` with all five `ants.registration` / `ants.apply_transforms`
   sites deleted and `TemplateToSubjectTransforms` replaced by `TemplateWarp`
3. `--device`, `--fireants-batch-size`, `--label-interpolation` in `run.py`
4. Updated `Dockerfile`, `Singularity`, `requirements.txt`, `run_container.sh`
5. Tier-1 tests in `tests/test_registration.py`
6. Tier-2 and tier-3 harnesses as submittable `sbatch` scripts under `scripts/`
7. Tier-3 report: per-label Dice, volume deltas, Jacobian, BVOL for 3 subjects
