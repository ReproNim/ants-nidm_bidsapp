"""Choice of the anatomical image to segment.

A BIDS subject or session can hold several images of the requested suffix:
ABIDE II's longitudinal sites name every file ``run-1``, and UCLA_Long adds an
``acq-hires`` acquisition beside the standard one. Exactly one image is
segmented per subject/session, chosen by:

1. no ``acq-`` entity before any ``acq-`` entity (the site's standard
   acquisition over its variants),
2. then no other optional entities (``ce-``, ``rec-``, ...) before some,
3. then the lowest ``run-`` number (a file with no run counts as run 0),
4. then the file name.

Every caller that needs the input path uses this, so segmentation and the
recorded provenance name the same file. The choice and every skipped
alternative are logged.
"""
import re
from pathlib import Path


def _candidates(anat_dir, subject, session, suffix):
    prefix = f"sub-{subject}" + (f"_ses-{session}" if session else "")
    pattern = re.compile(
        rf"^{re.escape(prefix)}((?:_[a-zA-Z0-9]+-[a-zA-Z0-9]+)*)_{re.escape(suffix)}\.nii(\.gz)?$"
    )
    found = []
    for path in anat_dir.iterdir():
        m = pattern.match(path.name)
        if not m:
            continue
        entities = dict(e.split("-", 1) for e in m.group(1).split("_") if e)
        found.append((path, entities))
    return found


def _rank(item):
    path, entities = item
    others = set(entities) - {"acq", "run"}
    run = entities.get("run", "0")
    return ("acq" in entities, bool(others),
            int(run) if run.isdigit() else float("inf"), path.name)


def select_anat_input(bids_dir, subject, session=None, suffix="T1w", logger=None):
    """Return the image to segment for one subject/session, or None.

    ``subject`` and ``session`` are BIDS labels without their ``sub-``/``ses-``
    prefixes.
    """
    anat_dir = Path(bids_dir) / f"sub-{subject}"
    if session:
        anat_dir = anat_dir / f"ses-{session}"
    anat_dir = anat_dir / "anat"
    if not anat_dir.is_dir():
        return None

    ranked = sorted(_candidates(anat_dir, subject, session, suffix), key=_rank)
    if not ranked:
        return None
    chosen = ranked[0][0]
    if logger is not None:
        logger.info(f"{suffix} input: {chosen.name}")
        for path, _ in ranked[1:]:
            logger.info(f"  not used: {path.name}")
    return chosen
