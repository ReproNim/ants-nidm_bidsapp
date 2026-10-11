# ANTs BIDS App

A BIDS App for ANTs-based brain segmentation with NIDM outputs.

## Overview

This BIDS App provides a standardized way to run ANTs-based brain segmentation on BIDS-formatted datasets. It includes preprocessing steps, segmentation, and generates NIDM-compatible outputs for better reproducibility and sharing of results.

## Features

- BIDS-compliant input/output
- N4 bias field correction (ANTs)
- Brain extraction via `antsBrainExtraction.sh` (ANTs)
- GPU-accelerated registration via [FireANTs](https://github.com/rohitrango/FireANTs)
- Joint label fusion voting (ANTsPy `joint_label_fusion`)
- NIDM-compatible outputs
- Docker and Singularity support

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
- **Slurm:** request a GPU, e.g. `--partition pi_satra --gres=gpu:1`.
- **A writable `/work` bind.** The image sets `TMPDIR=/work` and the app puts
  its registration scratch (~1-2 GB per subject, deleted after use) there.
  `run_container.sh` mounts it.

```bash
# Slurm + Apptainer
sbatch --partition pi_satra --gres=gpu:1 --cpus-per-task 8 --mem 48G \
  --wrap "apptainer run --nv \
    -B /path/to/bids:/bids:ro -B /path/to/out:/out -B \$TMPDIR:/work \
    ants-nidm_bidsapp.sif /bids /out participant \
      --participant-label 01 --method fusion --device cuda --num-threads 8"
```

**What did not change.** FireANTs does registration and nothing else. N4 bias
correction, `antsBrainExtraction.sh`, Atropos and the `joint_label_fusion`
voting still run on ANTs 2.5.4 / ANTsPy, which the container still ships.

## Installation

### Container Images

Pre-built images will be available once the app is published to Docker Hub. For now, please build from source (see below).

### From Source

```bash
git clone https://github.com/ReproNim/ants-nidm_bidsapp.git
cd ants-nidm_bidsapp
pip install -e .
```

### Building Containers from Source

This BIDS App follows standard BIDS Apps practices with a Dockerfile as the primary container definition. For HPC environments without Docker, we also provide a native Singularity definition file.

#### Building with Docker (on systems with Docker installed)

```bash
# Using the setup.py helper script
python setup.py docker

# Or directly with Docker
docker build -t ants-nidm_bidsapp:latest .

# Save for transfer to HPC (if needed)
docker save ants-nidm_bidsapp:latest -o ants-nidm_bidsapp.tar
```

#### Building with Singularity/Apptainer (for HPC environments)

```bash
# Direct build from Singularity definition file
# The --fakeroot flag is required on HPC systems without root access
apptainer build --fakeroot ants-nidm_bidsapp.sif Singularity

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
  --prob-threshold 0.5 \
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
  --nidm-input-dir /path/to/nidm/inputs
```

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
- `--method`: Segmentation method - `quick` or `fusion` (default: fusion)
- `--prob-threshold`: Probability threshold for binary mask creation (default: 0.5)
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
- `--nidm-input-dir`: Directory containing existing NIDM files (optional). The app will search for files matching `sub-{id}/[ses-{session}/]sub-{id}[_ses-{session}].ttl` or fallback to `nidm.ttl`

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
│   │   ├── sub-XX_T1w_space-orig_dseg.nii.gz
│   │   ├── sub-XX_T1w_BrainSegmentation.nii.gz
│   │   └── sub-XX_T1w_BrainSegmentationPosteriors*.nii.gz
│   ├── stats/
│   │   ├── sub-XX_antslabelstats.csv
│   │   └── sub-XX_antsbrainvols.csv
│   ├── nidm.ttl                                # Input NIDM augmented with ANTs metrics
│   └── ants_cde.ttl                            # Shared CDE vocabulary, shipped alongside
├── sub-YY/
│   └── ses-ZZ/                                 # Multi-session datasets nest a session level
│       ├── anat/, stats/
│       ├── nidm.ttl
│       └── ants_cde.ttl
└── logs/                                       # Processing logs
```

**NIDM layout:** the output is always named `nidm.ttl` and subject identity is
carried by the directory. There is deliberately no app-name wrapper directory
and no shared `nidm/` directory: a shared `nidm/nidm.ttl` made every subject's
NIDM collide on one path, and `unzip -n` at BABS merge time silently kept only
the first subject's copy. `ants_cde.ttl` is a static, byte-identical vocabulary
required to resolve the `ants_*` predicates; it is shipped next to `nidm.ttl`
rather than merged into it, matching the sibling FreeSurfer and FSL apps.

Output files include:
- **Segmentation results** in BIDS-derivatives format
- **Probability maps** for each tissue class
- **Statistics files** (CSV) for downstream analysis
- **NIDM-compatible outputs** (Turtle RDF format) for reproducibility and data sharing

## Validation harnesses

`scripts/` carries the GPU test tiers from the FireANTs design
(`docs/superpowers/specs/2026-09-01-fireants-engine-design.md` section 7):

- `scripts/tier2_smoke.sbatch <sif> <bids> <out> <participant>` -- one subject
  through both methods on a GPU node, asserting the per-subject layout plus
  `nidm.ttl` and `ants_cde.ttl`.
- `scripts/tier3_agreement.sbatch <sif> <out>` -- three ABIDE Caltech subjects
  compared against the delivered ANTs derivative, with volume-stratified Dice
  and volume thresholds. The comparison runs inside the container, because it
  needs FireANTs' own `scripts/evalutils.py`.

## NIDM Outputs

The app generates NIDM-compatible outputs that can be used with NIDM tools for visualization and sharing of results. The NIDM outputs include:

- Segmentation statistics
- Brain volumes
- Tissue volumes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this BIDS App in your research, please cite:

```
ANTs NIDM BIDS App. ReproNim. https://github.com/ReproNim/ants-nidm_bidsapp
```
