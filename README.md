# ANTs BIDS App

A BIDS App for ANTs-based brain segmentation with NIDM outputs.

## Overview

This BIDS App provides a standardized way to run ANTs-based brain segmentation on BIDS-formatted datasets. It includes preprocessing steps, segmentation, and generates NIDM-compatible outputs for better reproducibility and sharing of results.

## Features

- BIDS-compliant input/output
- ANTs-based brain segmentation
- N4 bias field correction
- Brain extraction
- Tissue segmentation
- NIDM-compatible outputs
- Docker and Singularity support

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

Expect roughly 1–2 hours per subject at 8–16 threads and up to ~18 GB of RAM
for the default joint-label-fusion method.

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
- `--num-threads`: Number of threads to use for processing (default: 1)

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
