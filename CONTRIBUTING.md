# Contributing

How the code is organised, what each tool is responsible for, and how to test
a change. User-facing instructions are in the [README](README.md).

## Code layout

```
src/
├── run.py                  # CLI, BIDS layout, participant/session dispatch, NIDM call
├── antspy/
│   ├── wrapper.py          # ANTsSegmentation: preprocessing, both methods, outputs
│   └── label_vocabulary.py # Neuromorphometrics <-> FreeSurfer aseg bijection
├── registration/           # the FireANTs boundary
│   ├── fireants.py         # warp_atlases, register_to_template, apply_warp
│   ├── handoff.py          # file-based ANTsPy <-> FireANTs crossing, geometry guard
│   ├── labels.py           # sparse label ids <-> contiguous indices
│   └── device.py           # resolve_device: fail fast when no GPU is visible
└── ants_seg_to_nidm/       # git submodule: stats -> NIDM conversion
```

`registration/fireants.py` is the only module that imports `fireants` or
`torch`. The rest of `registration/` stays importable without them, so most of
the test suite runs in a CI image with no GPU stack.

## Which tool does what

FireANTs does registration and nothing else: it has no bias-field correction,
brain extraction, tissue classification or label-fusion voting. Those stay on
ANTs 2.5.4 and ANTsPy, and the container ships both.

| Step | Tool | Where |
|---|---|---|
| N4 bias correction | ANTsPy | `compute_cortical_thickness` |
| Brain extraction | `antsBrainExtraction.sh` (ANTs CLI) | `compute_cortical_thickness` |
| 20 atlas registrations (`fusion`) | FireANTs, GPU | `registration.warp_atlases` |
| Subject-to-template registration (`quick`) | FireANTs, GPU | `registration.register_to_template` |
| Label fusion (`fusion`) | `ants.joint_label_fusion` | `segment_image` |
| Stats and NIDM | pandas, `ants_seg_to_nidm` | `_organize_bids_output`, `run.nidm_conversion` |

Registration needs a visible CUDA device. With none, `resolve_device` raises
`DeviceUnavailable` before any image is read. There is no CPU fallback inside a
normal run: FireANTs on CPU is slower than the ANTs registration it replaced,
and a silent fallback in a job array turns every job into a many-hour one that
still exits 0. `--device cpu` exists for tests and small debugging runs.

## The ANTsPy / FireANTs boundary

ANTsPy and FireANTs (which reads through SimpleITK) order numpy axes
differently. Rebuilding an image from an array plus spacing, origin and
direction is where a silent orientation flip would come from, so the code
never does it: every image crosses as a NIfTI file under the scratch
directory, written by one library and read by the other.

- `handoff.read_nifti(path, expect=...)` compares shape, spacing, origin and
  direction with what was sent and raises `GeometryMismatch` on any
  difference. It runs in production, not only in tests.
- `fireants.io.Image` must be constructed without `orientation`, `spacing`,
  `direction`, `origin` or `center`. Each of these rewrites the header. If one
  is ever passed, the geometry check on the way back catches the result; a
  test loads a volume with `orientation="LPS"` and expects the mismatch.
- The template warp used by `quick` stays a torch object (`TemplateWarp`) and
  never crosses the boundary.

## Joint label fusion settings

`JLF_PARAMS` in `wrapper.py` holds the production settings, and the validation
scripts import it so they fuse the same way. Two of them are not tuning
choices:

- `max_lab_plus_one=True` gives the vote a background class. The brain mask is
  much larger than the union of labelled structures; without the flag every
  masked but unlabelled voxel is forced onto a real label, and the argmax of a
  near-zero probability stack returns the lowest id (4, the third ventricle).
- `usecor=True` makes the patch similarity scale-invariant. The target (an
  N4-corrected brain) and the OASIS-TRT-20 atlases are on different intensity
  scales, and the default mean-squared difference then shifts labels by tissue
  intensity: white-matter structures shrink and grey-matter and CSF
  structures grow.

## Label vocabularies

The two methods number subcortical structures differently, in the same
integer range, so nothing fails when they are mixed up.

| | `--method fusion` | `--method quick` |
|---|---|---|
| source | the 20 per-atlas DKT31_CMA label files | the pre-fused OASIS-30 template label map |
| subcortical ids | Neuromorphometrics / BrainCOLOR | FreeSurfer aseg |
| brain stem | 35 | 16 |
| hippocampi | 47 / 48 | 53 / 17 |
| cortical ids | DKT 1002-1035 / 2002-2035 | identical |

The segmentations are not renumbered. `run.LABEL_PROTOCOL_BY_METHOD` tells the
NIDM converter which table to name structures from (`--label-protocol`). A new
`--method` needs an entry there, and a test fails without one. With
`--skip-ants`, `--method` is required because existing derivatives do not
record which method made them.

`label_vocabulary.py` holds the 37-entry bijection between the two tables.
`scripts/diag_label_vocabulary_correspondence.py` re-derives it from the
atlases and exits non-zero if an entry stops reproducing.

## Scratch space

The image sets `TMPDIR=/work`. A subject needs 1-2 GB of registration scratch,
and `ants.joint_label_fusion` writes its probability images through `mktemp`.
At startup the app checks that `$TMPDIR` is writable with at least 4 GiB free;
otherwise (a read-only `/work`, or Apptainer's size-capped `--writable-tmpfs`
overlay as BABS runs it) it moves its scratch and the process temp directory to
`<output_dir>/tmp` and logs a warning. Binding a real directory over `/work`
avoids the fallback.

## Tests

```bash
pytest tests/                          # everything that can run here
pytest -m "not requires_fireants" tests/   # without fireants/torch installed
```

Markers (`pytest.ini`): `requires_fireants` needs `fireants` and `torch`
importable and runs on CPU; `ci_incompatible` needs a full ANTs and template
environment.

GPU checks, run as Slurm jobs against a built image:

- `scripts/tier2_smoke.sbatch <sif> <bids> <out> <participant>` runs one
  subject through both methods and checks the per-subject layout, `nidm.ttl`
  and `ants_cde.ttl`. `SRC_FROM_REPO=1` binds a copy of the working tree's
  `src/` over the image's, to test a change without rebuilding.
- `scripts/diag_jlf_leave_one_out.sbatch <sif> <out>` holds out each
  OASIS-TRT-20 atlas in turn, fuses the other 19 and scores the result against
  the held-out manual labels: the accuracy measure with a ground truth. Run
  `scripts/diag_jlf_leave_one_out_summary.py <out> --gate` on the result; it
  exits 1 unless every fold reaches `GATE_MIN_DICE` in each structure group.
  The floors sit 0.03 below the worst fold of a validated run; the fusion
  without `usecor` fails every group on every fold.

Checks on outputs, no GPU needed:

- `scripts/check_label_volume_plausibility.py <dir>` flags structures far
  outside the range seen in the 20 atlases, or missing from fusion output
  altogether. It exits non-zero on a finding.
- `scripts/freesurfer_agreement.py <app_out> --subject <id>` reports Dice and
  volume ratios against a FreeSurfer `aparc.DKTatlas+aseg`, and BVOL against
  BrainSegVol. FreeSurfer uses a different protocol, so this is a report, not
  a pass criterion.

Scripts that read cluster data take their paths from a gitignored `.env`
(`CALTECH_DERIVATIVES`, `SCRATCH`).

## Debugging the GPU path

- `DeviceUnavailable` at startup: no visible CUDA device. Check `--nv`
  (Apptainer) or `--gpus all` (Docker) and `--gres=gpu:1`; run `nvidia-smi`
  inside the container.
- The log line `FireANTs device=... fused_ops=...` appears once per
  `warp_atlases` call. `fused_ops=False` means the image was built without the
  compiled extension; label warping then goes through a one-hot volume of about
  100 channels per atlas. The extension needs `nvcc` at build time, which is
  why both container recipes start from a CUDA `devel` image.
- `registration folded for ...`: the warp has non-positive Jacobian
  determinants on more than 0.1% of voxels. This is a hard failure for both the
  atlas and the template registrations.

## NIDM converter

Changes to NIDM output go in the `src/ants_seg_to_nidm/` submodule. Running
its own test suite writes `GVol`/`WVol` entries into its tracked
`mapping_data/ants-cdes.json`; check `git status` in the submodule before
committing.
