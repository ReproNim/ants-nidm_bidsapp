# ANTs BIDS App

A BIDS App for ANTs-based brain segmentation with NIDM outputs.

## Overview

This BIDS App segments a T1-weighted MRI into about 100 labelled brain
structures (DKT31 cortical regions plus subcortical structures) with
multi-atlas joint label fusion, and writes per-structure volumes as CSV and as
NIDM (RDF/Turtle) so the results can be pooled and queried across studies.

For each subject (or subject and session) the pipeline:

1. picks the T1w image to process (see [Input images](#input-images));
2. corrects intensity bias with N4 and extracts the brain with
   `antsBrainExtraction.sh` (ANTs);
3. registers each of the 20 OASIS-TRT-20 atlases to the subject on the GPU
   ([FireANTs](https://github.com/rohitrango/FireANTs));
4. fuses the 20 warped label maps into one segmentation with ANTsPy
   `joint_label_fusion`;
5. writes the segmentation, per-label volumes, brain volume and NIDM.

Two methods are available:

| `--method` | What it does | Use it for |
|---|---|---|
| `fusion` (default) | 20 atlas registrations and a joint-fusion vote in subject space | results you will analyse |
| `quick` | one registration to the OASIS-30 template, then warps that template's pre-computed label map | fast checks; less accurate |

The two methods number subcortical structures differently (see
[Label numbering](#label-numbering)); do not mix their outputs.

## Features

- BIDS-compliant input/output, single- and multi-session datasets
- N4 bias field correction and brain extraction (ANTs)
- GPU-accelerated registration via FireANTs
- Joint label fusion voting (ANTsPy `joint_label_fusion`)
- NIDM outputs that extend an existing per-subject NIDM file
- Docker and Apptainer/Singularity images; a ready-made BABS configuration

## Requirements: a GPU is mandatory

**Breaking change.** Every registration in this app now runs on FireANTs, on
the GPU. Both `--method fusion` and `--method quick` need a visible CUDA
device -- `quick` too, because the subject-to-template registration it depends
on moved along with the fusion loop.

With no visible GPU the app **exits immediately** with `DeviceUnavailable`,
before reading any image. It does not fall back to CPU: FireANTs on CPU is
slower than the ANTs code it replaced, so a silent fallback inside a BABS
array would turn a fast job into a many-hour one while still exiting 0.
`--device cpu` remains available as an explicit opt-in for tests and small
debugging runs.

In practice this means:

- **Apptainer/Singularity:** pass `--nv`. `run_container.sh` already does.
- **Docker:** pass `--gpus all`. `run_container.sh` already does.
- **Slurm:** request a GPU, e.g. `--gres=gpu:1` on a GPU partition.
- **A writable `/work` bind.** The image sets `TMPDIR=/work` and the app puts
  its registration scratch (~1-2 GB per subject, deleted after use) there.
  `run_container.sh` mounts it.

```bash
# Slurm + Apptainer
sbatch --partition <gpu-partition> --gres=gpu:1 --cpus-per-task 8 --mem 48G \
  --wrap "apptainer run --nv \
    -B /path/to/bids:/bids:ro -B /path/to/out:/out -B \$TMPDIR:/work \
    ants-nidm_bidsapp.sif /bids /out participant \
      --participant-label 01 --method fusion --device cuda --num-threads 8"
```

**What did not change.** FireANTs does registration and nothing else. N4 bias
correction, `antsBrainExtraction.sh` and the `joint_label_fusion` voting still
run on ANTs 2.5.4 / ANTsPy, which the container still ships.

**Resources per subject.** One GPU with about 20 GB of free memory at the
default `--fireants-batch-size 5` (lower it on smaller cards), up to ~18 GB of
RAM for the fusion vote, and 1-2 GB of scratch.

## Installation

### Container Images

Pre-built images will be available once the app is published to Docker Hub. For now, please build from source (see below).

### From Source

```bash
git clone --recursive https://github.com/ReproNim/ants-nidm_bidsapp.git
cd ants-nidm_bidsapp
pip install -e src/ants_seg_to_nidm
pip install -r requirements.txt
pip install --upgrade 'rdflib>=7.0.0,<8'
pip install -e .
```

`--recursive` fetches the NIDM converter, which is a git submodule at
`src/ants_seg_to_nidm`; without it that directory is empty and the NIDM
conversion step fails. In an existing clone, run
`git submodule update --init --recursive`. The install order matches the
container recipes: the converter first, then `requirements.txt`, whose pins
take precedence, then rdflib 7 (pynidm 4.5.0 declares rdflib<6.4 but its
oxigraph store needs rdflib 7).

### Building Containers from Source

This BIDS App follows standard BIDS Apps practices with a Dockerfile as the primary container definition. For HPC environments without Docker, we also provide a native Singularity definition file.

#### Building with Docker (on systems with Docker installed)

```bash
# Using the setup.py helper script
python setup.py docker

# Or directly with Docker
docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ants-nidm_bidsapp:latest .

# Save for transfer to HPC (if needed)
docker save ants-nidm_bidsapp:latest -o ants-nidm_bidsapp.tar
```

#### Building with Singularity/Apptainer (for HPC environments)

```bash
# Direct build from Singularity definition file
# The --fakeroot flag is required on HPC systems without root access
apptainer build --fakeroot --build-arg GIT_COMMIT=$(git rev-parse HEAD) ants-nidm_bidsapp.sif Singularity

# Or using the setup.py helper
python setup.py singularity
```

#### Converting Docker to Singularity

If you have a Docker image (either built locally or from a tar file):

```bash
# From a saved Docker tar file
singularity build ants-nidm_bidsapp.sif docker-archive://ants-nidm_bidsapp.tar

# From local Docker daemon (requires Docker)
singularity build ants-nidm_bidsapp.sif docker-daemon://ants-nidm_bidsapp:latest
```

## Usage

### Quickstart (container, recommended)

Build the container once (see above), then run one subject with the bundled
wrapper — it sets up the binds and writable scratch the image expects:

```bash
./run_container.sh ants-nidm_bidsapp.sif /path/to/bids /path/to/output 01
```

The wrapper works with an Apptainer/Singularity `.sif` file or a Docker image
name, and passes any extra arguments straight to the app:

```bash
./run_container.sh ants-nidm_bidsapp.sif /path/to/bids /path/to/output 01 \
  --num-threads 8 --nidm-input-dir /path/to/nidm --verbose
```

The wrapper passes `--nv` (Apptainer) or `--gpus all` (Docker) for you and
mounts a writable scratch directory at `/work`.

RAM for joint label fusion is still up to ~18 GB -- the voting stage is
unchanged ANTsPy. Wall-clock is dominated by the twenty atlas registrations,
which now run on the GPU instead of on 8-16 CPU threads.

### Basic Usage

```bash
ants-nidm bids_dir output_dir participant --participant-label 01
```

### Advanced Options

```bash
# Full pipeline with all options
ants-nidm bids_dir output_dir participant \
  --participant-label 01 \
  --session-label pre \
  --modality T1w \
  --nidm-input-dir /path/to/nidm/inputs \
  --num-threads 4 \
  --verbose
```

### NIDM-Only Mode (Skip ANTs)

If you have already run ANTs segmentation and only want to generate NIDM outputs:

```bash
# Run only NIDM conversion using existing ANTs results
ants-nidm bids_dir output_dir participant \
  --participant-label 01 \
  --skip-ants \
  --ants-input /path/to/existing/ants-seg \
  --method fusion \
  --nidm-input-dir /path/to/nidm/inputs
```

`--method` is required here and must name the method that produced
`--ants-input`: `fusion` and `quick` number subcortical structures differently,
and the wrong one gives a valid NIDM file with wrong structure names.

### Running under BABS

[`examples/babs-ants-nidm-bids-study.yaml`](examples/babs-ants-nidm-bids-study.yaml)
is a complete BABS configuration. It uses the BIDS-study project layout from
[PennLINC/babs#369](https://github.com/PennLINC/babs/pull/369); delete its three
`*_path` settings to keep BABS's legacy layout. Two of its settings are required:

- `zip_foldernames: {${subid}: ...}`, so each job zips its own `sub-<id>/`
  folder. An app-name entry puts every subject's NIDM on the same path, and all
  but the first are lost when the zips are merged.
- `--nv` in `singularity_args` and `--gres=gpu:1` in the job resources, because
  registration needs a GPU.

BABS passes the positional level `participant` even for a project initialised
at the session level, and gives the session through `$SESSION_SELECTION_FLAG`.
When `--session-label` comes with `participant`, the app processes that session
only, as it would at the `session` level.

## Running it on your own dataset

A checklist for a new dataset.

1. **Validate the BIDS layout.** The app reads `sub-<id>/anat/` or
   `sub-<id>/ses-<id>/anat/`. It indexes the dataset with pybids, which
   validates file names unless you pass `--skip-bids-validation`.
2. **Check which T1w each subject will use.** See [Input images](#input-images).
   After a run, `processing_summary.json` records the file that was segmented
   and any that were not.
3. **Choose the analysis level.** A dataset without `ses-` folders runs at
   `participant` level, one job per subject. A dataset with `ses-` folders
   runs one job per session: `session --participant-label X --session-label Y`,
   or `participant` plus `--session-label` (what BABS passes).
4. **Point it at existing NIDM, if you have it.** With `--nidm-input-dir` the
   app adds its measures to each subject's NIDM file instead of starting a new
   one. Without the flag it looks in `<bids_dir>/../NIDM`. If nothing is found
   it writes a new `nidm.ttl`. See [NIDM input](#nidm-input).
5. **Run one subject first** with `run_container.sh` or
   `scripts/diag_babs_scratch.sbatch` (it checks the outputs for you), then
   the rest, under BABS or as a job array.
6. **Check the numbers, not just the exit code.** A run can exit 0 with a
   wrong segmentation. `scripts/check_label_volume_plausibility.py
   <output_dir>` compares every structure with its range in the 20 atlases and
   flags anything far outside it (add `--protocol freesurfer` for `quick`
   outputs). It counts voxels against the 1 mm atlases, so it is only
   meaningful for inputs near 1 mm. Look at a few `dseg` images over their T1w.

**Scope.** The atlases are the 20 adult OASIS-TRT-20 brains. The app has not
been validated on young children, on brains with large lesions or resections,
or on non-human data.

### Input images

A subject or session may contain several T1w images, for example
`sub-01_ses-1_run-1_T1w.nii.gz` next to `sub-01_ses-1_acq-hires_run-1_T1w.nii.gz`.
Exactly one is segmented, chosen in this order:

1. no `acq-` entity before any `acq-` entity (the standard acquisition before
   variants);
2. no other optional entity (`ce-`, `rec-`, ...) before some;
3. lowest `run-` number (no `run-` counts as run 0);
4. file name.

An image with any voxel dimension above `--max-voxel-size` (default 2 mm) is
skipped and the next one is used. If no image is fine enough, the subject is
not segmented and the run fails. A file's BIDS label says nothing about its
resolution: ABIDE II UCLA_Long's `acq-hires` T1w is 1.5 x 1.5 x 4 mm.

The choice and every skipped file are logged and recorded in
`processing_summary.json` (`input_file`, `inputs_not_used`,
`inputs_too_coarse` with voxel sizes, and `failure_reason` when nothing was
segmented) and in the segmentation's sidecar (`Sources`). If a dataset needs a different choice,
restructure the input or open an issue.

### NIDM input

`--nidm-input-dir` is searched most specific first:

1. `<dir>/sub-<id>/[ses-<ses>/]nidm.ttl`
2. `<dir>/sub-<id>/[ses-<ses>/]sub-<id>[_ses-<ses>].ttl`
3. `<dir>/sub-<id>[_ses-<ses>].ttl`
4. `<dir>/nidm.ttl`

The lookup runs before segmentation. If the directory exists but none of
these is present, the app logs a warning and writes a `nidm.ttl` without the
subject's demographics or input provenance. If the first one present is a
symlink with no content behind it (a datalad/git-annex clone before
`datalad get`), the run fails instead of treating it as a missing record.

### Command-line Arguments

**Required:**
- `bids_dir`: Path to the BIDS dataset
- `output_dir`: Path where outputs will be stored
- `analysis_level`: Level of the analysis (`participant` or `session`)

**Participant/Session Selection:**
- `--participant-label`, `--participant_label`: Label of the participant to analyze (with or without "sub-" prefix)
- `--session-label`, `--session_label`: Label of the session to analyze (with or without "ses-" prefix)

**Processing Options:**
- `--modality`: Imaging modality to process (default: T1w)
- `--max-voxel-size`: skip input images with any voxel dimension above this,
  in mm (default: 2.0); the subject fails if none is fine enough
- `--method`: Segmentation method - `quick` or `fusion` (default: fusion)
- `--num-threads`: Number of threads to use for processing (default: 1). Applies
  to the ANTs stages (N4, brain extraction, fusion voting) via
  `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`; registration runs on the GPU.

**Registration Options (FireANTs):**
- `--device`: `cuda` or `cpu` (default: `cuda`). `cuda` with no visible GPU
  raises `DeviceUnavailable` before any processing; there is no automatic
  fallback. `cpu` is an explicit opt-in for debugging and is much slower than
  the ANTs path it replaced.
- `--fireants-batch-size`: how many atlases to register per batch, for
  `--method fusion` (default: `5`; `0` means all twenty at once, which needs
  more than 80 GB of GPU memory). Lower it if GPU memory is tight; results are
  identical, only peak VRAM changes.
- `--label-interpolation`: `genericlabel` or `nearest` (default:
  `genericlabel`). `genericlabel` uses FireANTs' fused CUDA generic-label
  sampler, which returns argmax labels directly. That kernel exists only on
  CUDA, so with `--device cpu` (or an image built without the fused ops) the
  app requests a one-hot warp instead and argmaxes it itself -- same result,
  ~100x the channels, and a warning in the log. `nearest` warps the raw label
  intensities with nearest-neighbour sampling.

**Skip Options:**
- `--skip-nidm`: Skip NIDM conversion step (run ANTs only)
- `--skip-ants`: Skip ANTs segmentation step (run NIDM only, requires `--ants-input`)
- `--skip-bids-validation`: Skip BIDS validation step

**Input Options (for NIDM-only mode):**
- `--ants-input`: Path to existing ANTs segmentation derivatives (required if `--skip-ants`)
- `--method` is also required with `--skip-ants`, set to the method that produced `--ants-input`
- `--nidm-input-dir`: Directory containing existing NIDM files (optional; default `<bids_dir>/../NIDM`). Search order under [NIDM input](#nidm-input)

**Other:**
- `-v`, `--verbose`: Print detailed logs
- `--version`: Print version and exit

## Outputs

The app generates the following output structure. Everything produced for one
subject lives under `sub-XX/` -- that directory is the unit BABS zips, so two
subjects can never write to the same path:

```
output_dir/
├── dataset_description.json                    # Derivative-root metadata (not part of the per-subject zip)
├── sub-XX/                                     # Single-session datasets
│   ├── anat/
│   │   ├── sub-XX_space-orig_dseg.nii.gz       # Label image, in the T1w's space
│   │   ├── sub-XX_space-orig_dseg.json         # Sources: the T1w it came from
│   │   └── sub-XX_space-orig_label-<id>_probseg.nii.gz   # Per-label posteriors (fusion only)
│   ├── stats/
│   │   ├── sub-XX_antslabelstats.csv           # Label, VolumeInVoxels, Volume_mm3
│   │   └── sub-XX_antsbrainvols.csv            # BVOL, LABELVOL
│   ├── processing_summary.json                 # Success, input file used, versions
│   ├── nidm.ttl                                # Input NIDM augmented with ANTs metrics
│   └── ants_cde.ttl                            # Shared CDE vocabulary, shipped alongside
├── sub-YY/
│   └── ses-ZZ/                                 # Multi-session datasets nest a session level
│       ├── anat/, stats/, processing_summary.json
│       ├── nidm.ttl
│       └── ants_cde.ttl
├── tmp/                                        # Only if $TMPDIR was unusable; safe to delete
└── logs/                                       # Processing logs
```

**Volumes.** `BVOL` is the volume of the brain extraction mask. `LABELVOL` is
the sum of all labelled structures. The DKT31 protocol has no cerebral white
matter label, so `LABELVOL` is roughly half of `BVOL` and is not a brain
volume.

**NIDM layout:** the output is always named `nidm.ttl` and subject identity is
carried by the directory. There is deliberately no app-name wrapper directory
and no shared `nidm/` directory: a shared `nidm/nidm.ttl` made every subject's
NIDM collide on one path, and `unzip -n` at BABS merge time silently kept only
the first subject's copy. `ants_cde.ttl` is a static, byte-identical vocabulary
required to resolve the `ants_*` predicates; it is shipped next to `nidm.ttl`
rather than merged into it, matching the sibling FreeSurfer and FSL apps.

### Label numbering

Cortical regions use DKT ids (1002-1035 left, 2002-2035 right) under both
methods. Subcortical ids differ: `fusion` uses the Neuromorphometrics
(BrainCOLOR) numbering of the OASIS-TRT-20 label files, `quick` uses
FreeSurfer aseg numbering. The brain stem, for example, is 35 in one and 16 in
the other, and each id exists in both tables with different meanings. The
NIDM output names structures with the correct table for the method that
produced them; if you read the CSVs or `dseg` images directly, use the table
that matches the method. The mapping between the two is in
`src/antspy/label_vocabulary.py`.

## Validation harnesses

`scripts/` carries the GPU test tiers for the FireANTs registration engine:

- `scripts/diag_babs_scratch.sbatch <sif> <bids> <out> <participant> [<session>] [<nidm_dir>]`
  -- one subject or session with the container flags BABS uses
  (`--writable-tmpfs`, nothing bound over `/work`). Checks the outputs, the
  recorded input file and, with a NIDM dir, that the input NIDM was found.
- `scripts/check_label_volume_plausibility.py <output_dir>` -- flags any
  structure whose volume is far outside its range in the 20 atlases.
- `scripts/tier2_smoke.sbatch <sif> <bids> <out> <participant>` -- one subject
  through both methods on a GPU node, asserting the per-subject layout plus
  `nidm.ttl` and `ants_cde.ttl`.
- `scripts/diag_jlf_leave_one_out.sbatch <sif> <out>` -- the accuracy gate.
  Each of the 20 OASIS-TRT-20 atlases is held out in turn, the other 19 are
  registered to it and fused, and the result is scored against its manual
  labels. `scripts/diag_jlf_leave_one_out_summary.py <out> --gate` then
  requires every fold to reach a Dice floor in each structure group.
- `scripts/freesurfer_agreement.py <app_out> --subject <id>` -- report only:
  Dice and volume ratio of the delivered dseg against FreeSurfer's
  `aparc.DKTatlas+aseg`, and BVOL against FreeSurfer's BrainSegVol. FreeSurfer
  uses a different protocol, so this is a check on real scans, not a pass
  criterion.

## NIDM Outputs

`nidm.ttl` is the subject's input NIDM file (when one was found) with this
app's measures added: one volume per labelled structure, plus `BVOL` and
`LABELVOL`. Measure terms are defined in `ants_cde.ttl`, which ships next to it. The
conversion is done by the
[`ants_seg_to_nidm`](https://github.com/yibeichan/ants_seg_to_nidm) submodule.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
[CONTRIBUTING.md](CONTRIBUTING.md) covers the code layout, the
ANTsPy/FireANTs boundary and how to test a change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this BIDS App in your research, please cite:

```
ANTs NIDM BIDS App. ReproNim. https://github.com/ReproNim/ants-nidm_bidsapp
```
