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
git clone --recursive git@github.com:ReproNim/ants-nidm_bidsapp.git
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

The app generates the following output structure:

```
output_dir/
├── ants-nidm_bidsapp/                          # Main BIDS App output directory
│   ├── ants-seg/                               # ANTs segmentation derivatives
│   │   ├── dataset_description.json
│   │   └── sub-XX/
│   │       ├── ses-YY/                         # For multi-session datasets
│   │       │   ├── anat/
│   │       │   │   ├── sub-XX_ses-YY_T1w_space-orig_dseg.nii.gz
│   │       │   │   ├── sub-XX_ses-YY_T1w_BrainSegmentation.nii.gz
│   │       │   │   └── sub-XX_ses-YY_T1w_BrainSegmentationPosteriors*.nii.gz
│   │       │   └── stats/
│   │       │       ├── sub-XX_ses-YY_antslabelstats.csv
│   │       │       └── sub-XX_ses-YY_antsbrainvols.csv
│   │       └── anat/, stats/                   # For single-session datasets (no ses-)
│   └── nidm/                                   # NIDM outputs (flat structure)
│       ├── dataset_description.json
│       ├── sub-01_ses-baseline.ttl            # NIDM files (Turtle format)
│       └── sub-02_ses-baseline.ttl
└── logs/                                       # Processing logs
```

**Note:** NIDM outputs use a **flat file structure** (all TTL files in one directory) rather than hierarchical subdirectories, which simplifies file management and discovery.

Output files include:
- **Segmentation results** in BIDS-derivatives format
- **Probability maps** for each tissue class
- **Statistics files** (CSV) for downstream analysis
- **NIDM-compatible outputs** (Turtle RDF format) for reproducibility and data sharing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this BIDS App in your research, please cite:

```
ANTs NIDM BIDS App. ReproNim. https://github.com/ReproNim/ants-nidm_bidsapp
```
