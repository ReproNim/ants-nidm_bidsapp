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

Images coarser than ``max_voxel_size`` in any dimension are skipped and the
next one in that order is used; if none is fine enough, nothing is chosen and
the subject fails. ABIDE II UCLA_Long's ``acq-hires`` T1w, for example, is
1.5 x 1.5 x 4 mm, too coarse for a cortical parcellation. An image whose
header cannot be read ranks after every readable one, so it is used only when
nothing else is left (and then fails when loaded).

Every caller that needs the input path uses this, so segmentation and the
recorded provenance name the same file. The choice and every skipped
alternative are logged.
"""
import re
from collections import namedtuple
from pathlib import Path

DEFAULT_MAX_VOXEL_SIZE = 2.0  # mm, any dimension

# chosen: Path or None; not_used: fine-enough Paths ranked below the choice;
# too_coarse: (Path, voxel size tuple) pairs over the limit.
AnatChoice = namedtuple("AnatChoice", ["chosen", "not_used", "too_coarse"])


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


def rank_anat_inputs(bids_dir, subject, session=None, suffix="T1w"):
    """Every matching image for one subject/session, best first; [] if none.

    ``subject`` and ``session`` are BIDS labels without their ``sub-``/``ses-``
    prefixes.
    """
    anat_dir = Path(bids_dir) / f"sub-{subject}"
    if session:
        anat_dir = anat_dir / f"ses-{session}"
    anat_dir = anat_dir / "anat"
    if not anat_dir.is_dir():
        return []
    return [path for path, _ in sorted(_candidates(anat_dir, subject, session, suffix), key=_rank)]


def voxel_size(path):
    """Voxel size in mm of the first three dimensions, from the header only."""
    import nibabel as nib
    return tuple(round(float(z), 3) for z in nib.load(str(path)).header.get_zooms()[:3])


def choose_anat_input(bids_dir, subject, session=None, suffix="T1w",
                      max_voxel_size=None, logger=None):
    """Pick the image to segment; see the module docstring for the order."""
    ranked = rank_anat_inputs(bids_dir, subject, session, suffix)
    fine, unreadable, too_coarse = [], [], []
    for path in ranked:
        if max_voxel_size is not None:
            try:
                zooms = voxel_size(path)
            except Exception:
                unreadable.append(path)
                continue
            if max(zooms) > max_voxel_size:
                too_coarse.append((path, zooms))
                continue
        fine.append(path)
    fine += unreadable
    choice = AnatChoice(fine[0] if fine else None, fine[1:], too_coarse)
    if logger is not None:
        if choice.chosen is not None:
            logger.info(f"{suffix} input: {choice.chosen.name}")
        for path in choice.not_used:
            logger.info(f"  not used: {path.name}")
        for path in unreadable:
            logger.warning(f"  header unreadable, ranked last: {path.name}")
        for path, zooms in too_coarse:
            logger.warning(f"  not used, voxels {zooms} mm exceed {max_voxel_size} mm: {path.name}")
    return choice


def select_anat_input(bids_dir, subject, session=None, suffix="T1w", logger=None):
    """Return the best-ranked image for one subject/session, or None."""
    return choose_anat_input(bids_dir, subject, session, suffix, logger=logger).chosen


def bids_relative(path, bids_dir):
    """``path`` relative to the BIDS root as a POSIX string, else its name."""
    if bids_dir is None:
        return Path(path).name
    try:
        return Path(path).relative_to(Path(bids_dir)).as_posix()
    except ValueError:
        return Path(path).name


def recorded_anat_input(subject_dir, bids_dir):
    """The input a finished segmentation recorded, as an AnatChoice, or None.

    save_results writes the segmented image into the dseg sidecar as a BIDS
    ``Sources`` entry (``bids:raw:<path relative to the BIDS root>``). Reading
    it back lets a run that skips segmentation name the image that was
    actually segmented, instead of choosing again under the current rule.
    Returns None when there is no sidecar or it has no such entry, which is
    the case for derivatives written before the sidecar existed.
    """
    import json

    anat_dir = Path(subject_dir) / "anat"
    sidecars = sorted(anat_dir.glob("*_dseg.json")) if anat_dir.is_dir() else []
    for sidecar in sidecars:
        try:
            sources = json.loads(sidecar.read_text()).get("Sources", [])
        except (OSError, ValueError, AttributeError):
            continue
        for source in sources:
            if isinstance(source, str) and source.startswith("bids:raw:"):
                return AnatChoice(Path(bids_dir) / source[len("bids:raw:"):], [], [])
    return None
