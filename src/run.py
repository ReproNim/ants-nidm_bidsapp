#!/usr/bin/env python3
# src/run.py

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import shutil
from datetime import datetime
import subprocess
from bids import BIDSLayout
import ants

# Import local modules
from src.antspy.wrapper import ANTsSegmentation

# Version of the BIDS *specification* these derivatives claim conformance to.
# Must never be derived from a library version: pkg_resources.get_distribution
# ('pybids').version returns pybids' package version (0.16.x), which is not a
# valid BIDSVersion value.
BIDS_VERSION = "1.8.0"

# App version. Kept in one place: initialize(), --version, and the per-subject
# processing_summary.json all read it.
APP_VERSION = "0.1.0"

def setup_logger(log_dir, verbose=False):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"ants-nidm-{timestamp}.log")
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('ants-nidm')

def subject_output_dir(output_dir, bids_subject, bids_session=None):
    """Per-subject output directory -- the unit BABS zips.

    Returns ``<output_dir>/sub-<id>[/ses-<session>]``. Everything produced for a
    subject (segmentation, stats, nidm.ttl, ants_cde.ttl) lives here, so the
    zip's top-level folder is the subject directory and two subjects can never
    write to the same path.

    Args:
        output_dir (str or Path): Derivative root
        bids_subject (str): Subject label without "sub-" prefix
        bids_session (str, optional): Session label without "ses-" prefix
    """
    subject_dir = Path(output_dir) / f"sub-{bids_subject}"
    if bids_session:
        subject_dir = subject_dir / f"ses-{bids_session}"
    return subject_dir


def find_nidm_input_file(nidm_input_dir, subject_id, session_id=None):
    """Search for NIDM input file in standard locations.

    Candidates are tried most-specific first. The per-subject ``nidm.ttl``
    layout comes first because that is what the shared NIDM datasets use
    (e.g. ``nidm_4.5.0/sub-0051456/nidm.ttl``); the ``sub-<id>.ttl`` and flat
    forms are kept for older/hand-built inputs.

    Args:
        nidm_input_dir (Path): Directory containing NIDM files
        subject_id (str): Subject ID without 'sub-' prefix
        session_id (str, optional): Session ID without 'ses-' prefix

    Returns:
        Path or None: Path to NIDM file if found, None otherwise
    """
    if nidm_input_dir is None:
        return None

    nidm_input_dir = Path(nidm_input_dir)

    if not nidm_input_dir.exists():
        return None

    subject_dirname = f"sub-{subject_id}"
    candidates = []

    if session_id:
        candidates += [
            nidm_input_dir / subject_dirname / f"ses-{session_id}" / "nidm.ttl",
            nidm_input_dir / f"{subject_dirname}_ses-{session_id}" / "nidm.ttl",
            nidm_input_dir / subject_dirname / f"ses-{session_id}" / f"{subject_dirname}_ses-{session_id}.ttl",
            nidm_input_dir / f"{subject_dirname}_ses-{session_id}.ttl",
        ]

    candidates += [
        nidm_input_dir / subject_dirname / "nidm.ttl",
        nidm_input_dir / subject_dirname / f"{subject_dirname}.ttl",
        nidm_input_dir / f"{subject_dirname}.ttl",
        # Dataset-level fallback: a single NIDM file covering all subjects.
        nidm_input_dir / "nidm.ttl",
    ]

    return next((c for c in candidates if c.exists()), None)


def get_version_info(app_version):
    """Version provenance for the delivered outputs.

    Mirrors the sibling freesurfer-nidm app's version_info block so the two
    derivatives can be compared. ANTs' version comes from ANTsPy rather than a
    base image, since that is what actually performs the segmentation.
    """
    try:
        ants_version = ants.__version__
    except Exception:
        ants_version = "unknown"

    return {
        "ants-nidm": {
            "version": app_version,
            "source": "setup.py",
            "timestamp": datetime.now().isoformat(),
        },
        "ants": {
            "version": ants_version,
            "source": "antspyx",
        },
        "python": {
            "version": sys.version,
            "packages": {},
        },
    }


def save_processing_summary(logger, subject_dir, bids_subject, bids_session,
                            app_version, succeeded, nidm_written):
    """Write processing_summary.json inside the subject directory.

    BABS zips only sub-<id>/, so anything written outside it is dropped from the
    delivered derivative -- which is why this goes in the subject dir and not the
    derivative root. Same file the freesurfer-nidm app ships per subject.
    """
    subject_label = f"sub-{bids_subject}"
    if bids_session:
        subject_label += f"_ses-{bids_session}"

    summary = {
        "total": 1,
        "success": 1 if succeeded else 0,
        "failure": 0 if succeeded else 1,
        "skipped": 0,
        "success_list": [subject_label] if succeeded else [],
        "failure_list": [] if succeeded else [subject_label],
        "skipped_list": [],
        "nidm_written": nidm_written,
        "version_info": get_version_info(app_version),
    }

    try:
        subject_dir = Path(subject_dir)
        subject_dir.mkdir(parents=True, exist_ok=True)
        output_path = subject_dir / "processing_summary.json"
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Processing summary saved to {output_path}")
        return output_path
    except OSError as e:
        # Provenance is not worth failing an otherwise-good subject over.
        logger.warning(f"Could not write processing summary: {e}")
        return None


def create_dataset_description(output_dir, app_version):
    """Create a dataset_description.json file in the output directory."""
    dataset_description = {
        "Name": "ANTs segmentation derivatives",
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "ANTs BIDS App",
                "Version": app_version,
                "CodeURL": "https://github.com/ReproNim/ants-nidm_bidsapp"
            }
        ],
        "HowToAcknowledge": "Please cite the ANTs segmentation tool and the NIDM standard in your publications."
    }
    
    with open(os.path.join(output_dir, 'dataset_description.json'), 'w') as f:
        json.dump(dataset_description, f, indent=4)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ANTs Segmentation BIDS App with NIDM outputs')
    
    # Required arguments
    parser.add_argument('bids_dir', help='The directory with the input dataset formatted according to the BIDS standard.')
    parser.add_argument('output_dir', help='The directory where the output files should be stored.')
    parser.add_argument('analysis_level', help='Level of the analysis that will be performed.',
                        choices=['participant', 'session'])
    
    # Optional arguments
    parser.add_argument('--participant-label', '--participant_label', dest='participant_label',
                        help='The label of the participant that should be analyzed. The label corresponds to sub-<participant_label> from the BIDS spec (with or without "sub-" prefix).')
    parser.add_argument('--session-label', '--session_label', dest='session_label',
                        help='The label of the session that should be analyzed. The label corresponds to ses-<session_label> from the BIDS spec (with or without "ses-" prefix).')
    parser.add_argument('--modality', help='Modality to process [default: T1w]',
                        default='T1w')
    parser.add_argument('--prob-threshold', help='Probability threshold for binary mask creation [default: 0.5]',
                        type=float, default=0.5)
    
    # Segmentation method
    parser.add_argument('--method', help='Segmentation method to use [default: fusion]',
                        choices=['quick', 'fusion'], default='fusion')
    
    parser.add_argument('--skip-bids-validation', help='Skip BIDS validation step',
                        action='store_true')
    parser.add_argument('--skip-nidm', help='Skip NIDM conversion step',
                        action='store_true')
    
    # Skip ANTs and process existing results
    parser.add_argument('--skip-ants', help='Skip ANTs segmentation and only run NIDM conversion. Requires --ants-input.',
                        action='store_true')
    parser.add_argument('--ants-input', help='Path to existing ANTs segmentation derivatives. Required if --skip-ants is set.',
                        type=str, default=None)
    parser.add_argument('--nidm-input-dir', help='Directory containing existing NIDM files. Files will be searched in standard BIDS structure: sub-{id}/[ses-{session}/]sub-{id}[_ses-{session}].ttl or fallback to nidm.ttl. Defaults to <bids_dir>/../NIDM for BABS workflows.',
                        type=str, default=None)
    
    parser.add_argument('--num-threads', help='Number of threads to use for processing [default: 1]',
                        type=int, default=1)
    parser.add_argument('-v', '--verbose', help='Verbose output',
                        action='store_true')
    parser.add_argument('--version', action='version',
                        version=f'ANTs BIDS App v{APP_VERSION}')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_ants and not args.ants_input:
        parser.error("--skip-ants requires --ants-input to specify existing ANTs derivatives")
    
    return args

def initialize(args):
    """Initialize the ANTs BIDS app.
    Args:
        args: Command line arguments
    Returns:
        tuple: (layout, segmenter, derivatives_dir, nidm_input_dir)
    """
    # Normalize incoming paths from argparse to Path objects
    args.bids_dir = Path(args.bids_dir)
    args.output_dir = Path(args.output_dir)

    # Initialize BIDS Layout
    layout = BIDSLayout(str(args.bids_dir), validate=not args.skip_bids_validation)

    # Handle NIDM input directory
    # Use CLI argument if provided, otherwise default to BABS location
    if args.nidm_input_dir is None:
        nidm_input_dir = args.bids_dir.parent / "NIDM"
    else:
        nidm_input_dir = Path(args.nidm_input_dir)

    if not nidm_input_dir.exists():
        nidm_input_dir = None

    # Per-subject output layout (study-wide standard): everything this app
    # produces for a subject lives under <output_dir>/sub-<id>/, so the BABS
    # zip's top-level folder is the subject directory and results land as
    # <derivative_name>/sub-<id>/... when unzipped. There is deliberately no
    # app-name wrapper directory and no shared nidm/ directory: a shared
    # nidm/nidm.ttl made every subject's NIDM collide on one path, and
    # `unzip -n` at merge time silently kept only the first subject's copy.
    derivatives_dir = args.output_dir
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    # dataset_description.json describes the derivative root, which is owned by
    # the surrounding dataset (BABS/DataLad) rather than by a per-subject job.
    # It is written for standalone runs; under BABS only sub-<id>/ is zipped, so
    # this file is not part of the delivered per-subject unit and cannot collide.
    create_dataset_description(derivatives_dir, APP_VERSION)

    # Initialize segmentation with appropriate parameters (only if not skipping ANTs)
    segmenter = None
    if not args.skip_ants:
        segmenter = ANTsSegmentation(
            bids_dir=str(args.bids_dir),
            output_dir=str(derivatives_dir),
            modality=args.modality,
            prob_threshold=args.prob_threshold,
            num_threads=args.num_threads,
            verbose=args.verbose
        )

    return layout, segmenter, derivatives_dir, nidm_input_dir

def nidm_conversion(logger, derivatives_dir, bids_subject, nidm_input_file=None,
                    bids_session=None, verbose=False, input_file=None):
    """Convert ANTs segmentation outputs to NIDM format.

    Writes ``nidm.ttl`` (and the shared ``ants_cde.ttl`` vocabulary, which
    ants_seg_to_nidm serializes next to the -o target) into the subject's own
    output directory, alongside the segmentation results. The output is always
    named ``nidm.ttl`` -- subject identity is carried by the directory, which is
    also what keeps concurrent subjects from colliding.

    Args:
        logger: Logger instance
        derivatives_dir (str or Path): Derivative root (output_dir)
        bids_subject (str): Subject label (without "sub-" prefix)
        nidm_input_file (Path or None): Optional existing NIDM TTL file to append to
        bids_session (str): Session label (without "ses-" prefix)
        verbose (bool): Enable verbose output
        input_file (str or Path): Path to the input T1w file
    Returns:
        bool: True if conversion succeeded, False otherwise
    """
    log_prefix = f"subject {bids_subject}" + (f", session {bids_session}" if bids_session else "")

    try:
        derivatives_dir = Path(derivatives_dir)
        subject_dir = subject_output_dir(derivatives_dir, bids_subject, bids_session)
        subject_dir.mkdir(parents=True, exist_ok=True)

        # Existing NIDM file to append to. Always the pristine input -- never a
        # nidm.ttl left in the output directory by an earlier attempt, which
        # would double-add this subject's measurements on a retry.
        existing_nidm_file = None
        if nidm_input_file and Path(nidm_input_file).exists():
            existing_nidm_file = Path(nidm_input_file)

        # Segmentation outputs live under sub-*/[ses-*/]{anat,stats}/
        seg_base = f"sub-{bids_subject}"
        if bids_session:
            seg_base += f"_ses-{bids_session}"

        anat_dir = subject_dir / "anat"
        stats_dir = subject_dir / "stats"

        seg_path = anat_dir / f"{seg_base}_space-orig_dseg.nii.gz"
        label_stats = stats_dir / f"{seg_base}_antslabelstats.csv"
        brain_vols = stats_dir / f"{seg_base}_antsbrainvols.csv"

        # Check if required files exist
        required_files = [seg_path, label_stats, brain_vols]
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required file not found: {file_path}")
                return False

        # The product is always <subject_dir>/nidm.ttl, whether or not there was
        # an input NIDM file to augment.
        nidm_output = subject_dir / "nidm.ttl"

        label_stats_str = str(label_stats.absolute())
        brain_vols_str = str(brain_vols.absolute())
        seg_path_str = str(seg_path.absolute())
        nidm_output_str = str(nidm_output.absolute())

        # Construct the command to run ants_seg_to_nidm.py.
        # NOTE: -subjid is the participant identifier used to match/attach the
        # subject inside the NIDM graph; ants_seg_to_nidm has no -session
        # option, so session identity is carried by the output directory.
        cmd = [
            "python", "-m",
            "ants_seg_to_nidm.ants_seg_to_nidm",
            "-f", f"{label_stats_str},{brain_vols_str},{seg_path_str}",
            "-subjid", f"sub-{bids_subject}",
            "-o", nidm_output_str
        ]

        # Add existing NIDM file if available
        if existing_nidm_file:
            cmd.extend(["--nidm", str(existing_nidm_file.absolute()), "--forcenidm"])

        logger.info(f"Converting segmentation to NIDM for {log_prefix}")
        if existing_nidm_file:
            logger.info(f"Adding data to existing NIDM file: {existing_nidm_file}")
        logger.info(f"NIDM will be written to: {nidm_output_str}")
        logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Error in NIDM conversion: {result.stderr}")
            return False

        if not nidm_output.exists():
            logger.error(f"NIDM conversion reported success but {nidm_output} was not written")
            return False

        logger.info(f"NIDM conversion complete for {log_prefix}")
        for produced in sorted(subject_dir.glob("*.ttl")):
            logger.info(f"  - {produced.name} ({produced.stat().st_size} bytes)")
        if verbose:
            logger.debug(f"NIDM conversion output: {result.stdout}")
        return True

    except Exception as e:
        logger.error(f"Error in NIDM conversion for {log_prefix}: {str(e)}")
        return False

def process_participant(args, logger):
    """Run participant analysis, including BABS session-wise invocations."""
    if args.session_label:
        logger.info(
            "Session label supplied with participant analysis level; "
            "processing the requested session"
        )
        return process_session(args, logger)

    logger.info("Starting participant level analysis (single-session dataset)")

    # Initialize app
    layout, segmenter, derivatives_dir, nidm_input_dir = initialize(args)
    
    # Get subject to process
    available_subjects = layout.get_subjects()
    participant_label = args.participant_label
    # Normalize participant label - strip "sub-" if present for consistency
    if participant_label.startswith('sub-'):
        bids_subject = participant_label[4:]
    else:
        bids_subject = participant_label
    participant_label = f"sub-{bids_subject}"  # Ensure full label for logging
    
    if bids_subject not in available_subjects:
        logger.error(f"Subject {participant_label} not found in dataset")
        return 1
    
    # For participant level: process single session (no session folders expected)
    bids_session = None
    
    logger.info(f"Processing single-session data for subject {participant_label}")
    
    success = True
    
    # Run ANTs segmentation (unless skipped)
    if args.skip_ants:
        logger.info("Skipping ANTs segmentation (--skip-ants flag set)")
        # Use existing ANTs derivatives if provided
        if args.ants_input:
            derivatives_dir = Path(args.ants_input)
            logger.info(f"Using existing ANTs derivatives from: {derivatives_dir}")
    else:
        # Run segmentation
        if segmenter.run_subject(participant_label, None, method=args.method):
            logger.info(f"Segmentation completed for subject {participant_label}")
        else:
            success = False
            logger.error(f"Segmentation failed for subject {participant_label}")
    
    # Convert segmentation to NIDM if requested and segmentation succeeded (or skipped)
    if success and not args.skip_nidm:
        # Find NIDM input file for this subject (no session for participant level)
        nidm_input_file = find_nidm_input_file(nidm_input_dir, bids_subject, session_id=None)
        if nidm_input_file:
            logger.info(f"Found NIDM input file: {nidm_input_file}")

        # Get input file path for NIDM conversion (single session)
        input_path = layout.get(subject=bids_subject, suffix=args.modality, extension='nii.gz')
        input_file = input_path[0].path if input_path else None

        success = nidm_conversion(
            logger=logger,
            derivatives_dir=derivatives_dir,
            bids_subject=bids_subject,
            nidm_input_file=nidm_input_file,
            bids_session=bids_session,
            verbose=args.verbose,
            input_file=input_file,
        )
    
    save_processing_summary(
        logger=logger,
        subject_dir=subject_output_dir(derivatives_dir, bids_subject, bids_session),
        bids_subject=bids_subject,
        bids_session=bids_session,
        app_version=APP_VERSION,
        succeeded=success,
        nidm_written=(not args.skip_nidm) and success,
    )

    logger.info(f"Participant level analysis complete. Processing {'succeeded' if success else 'failed'}")

    return 0 if success else 1

def process_session(args, logger):
    """Run the session level analysis for multi-session datasets.

    Note: BABS schedules each session as a separate task, so this processes
    ONE session per task execution.
    """
    logger.info("Starting session level analysis (multi-session dataset)")

    # Initialize app
    layout, segmenter, derivatives_dir, nidm_input_dir = initialize(args)
    
    # Get subject to process
    available_subjects = layout.get_subjects()
    participant_label = args.participant_label
    # Normalize participant label - strip "sub-" if present for consistency
    if participant_label.startswith('sub-'):
        bids_subject = participant_label[4:]
    else:
        bids_subject = participant_label
    participant_label = f"sub-{bids_subject}"  # Ensure full label for logging
    
    if bids_subject not in available_subjects:
        logger.error(f"Subject {participant_label} not found in dataset")
        return 1
    
    # For session level: --session-label is required by BABS
    if not args.session_label:
        logger.error("--session-label is required for session level analysis")
        return 1
    
    # Normalize session label
    session_label = args.session_label
    if session_label.startswith('ses-'):
        bids_session = session_label[4:]
    else:
        bids_session = session_label
    session_label = f"ses-{bids_session}"  # Ensure full label for logging
    
    # Validate session exists
    available_sessions = layout.get_sessions(subject=bids_subject)
    if bids_session not in available_sessions:
        logger.error(f"Session {session_label} not found for subject {participant_label}")
        return 1
    
    logger.info(f"Processing session {session_label} for subject {participant_label}")
    
    success = True
    
    # Run ANTs segmentation (unless skipped)
    if args.skip_ants:
        logger.info("Skipping ANTs segmentation (--skip-ants flag set)")
        # Use existing ANTs derivatives if provided
        if args.ants_input:
            derivatives_dir = Path(args.ants_input)
            logger.info(f"Using existing ANTs derivatives from: {derivatives_dir}")
    else:
        # Run segmentation
        if segmenter.run_subject(participant_label, session_label, method=args.method):
            logger.info(f"Segmentation completed for session {session_label}")
        else:
            success = False
            logger.error(f"Segmentation failed for session {session_label}")
    
    # Convert segmentation to NIDM if requested and segmentation succeeded (or skipped)
    if success and not args.skip_nidm:
        # Find NIDM input file for this subject and session
        nidm_input_file = find_nidm_input_file(nidm_input_dir, bids_subject, session_id=bids_session)
        if nidm_input_file:
            logger.info(f"Found NIDM input file: {nidm_input_file}")

        # Get input file path for NIDM conversion
        input_path = layout.get(subject=bids_subject, session=bids_session, suffix=args.modality, extension='nii.gz')
        input_file = input_path[0].path if input_path else None

        success = nidm_conversion(
            logger=logger,
            derivatives_dir=derivatives_dir,
            bids_subject=bids_subject,
            nidm_input_file=nidm_input_file,
            bids_session=bids_session,
            verbose=args.verbose,
            input_file=input_file,
        )
    
    save_processing_summary(
        logger=logger,
        subject_dir=subject_output_dir(derivatives_dir, bids_subject, bids_session),
        bids_subject=bids_subject,
        bids_session=bids_session,
        app_version=APP_VERSION,
        succeeded=success,
        nidm_written=(not args.skip_nidm) and success,
    )

    logger.info(f"Session level analysis complete. Processing {'succeeded' if success else 'failed'}")

    return 0 if success else 1


def main():
    """Main function to coordinate the workflow."""
    args = parse_arguments()
    
    # Set up logging
    log_dir = os.path.join(args.output_dir, 'logs')
    logger = setup_logger(log_dir, args.verbose)
    
    logger.info("Starting ANTs BIDS App")
    logger.info(f"BIDS directory: {args.bids_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Analysis level: {args.analysis_level}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.analysis_level == 'participant':
            return process_participant(args, logger)
        elif args.analysis_level == 'session':
            return process_session(args, logger)
        else:
            logger.error(f"Unsupported analysis level: {args.analysis_level}")
            return 1
    except Exception as e:
        logger.error(f"Error in {args.analysis_level} level analysis: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
