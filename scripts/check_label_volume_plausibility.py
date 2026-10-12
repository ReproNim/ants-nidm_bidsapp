"""Flag structures whose measured volume is impossible for what they claim to be.

Why this exists
---------------
The earlier pilot gate, a separate script run on delivered BABS zips,
checks volumes against the field of view: background must be at least 25% of
the FOV, no single label may exceed 25% of it. Those catch a segmentation that
has escaped the brain mask entirely, which is the failure the August 2026 pilot
hit.

They do not catch a structure that is merely 70 times too large, because 70x
too large is still under 1% of a 256^3 FOV. That is exactly what shipped: every
one of the 38 delivered Caltech subjects reported label 4, the third ventricle,
at a median of 81355 voxels against an atlas range of 321 to 1087 (measured
2026-09-25). The joint-fusion vote has no background class, so the unlabelled
interior of the brain mask landed on the lowest label id. See
`scripts/diag_jlf_background_fallback.py` for the mechanism.

Worse, the old gate's own comments had already rationalised the symptom --
"BVOL == brain-mask volume under 'fusion' (ANTsPy jlf argmax labels every mask
voxel; there is no background class)" -- and its BVOL ceiling was raised to
accommodate it. The behaviour was seen and normalised rather than questioned.

So this check compares each label against *its own* plausible range, taken from
the twenty OASIS-TRT-20 atlases, rather than against the FOV.

The comparison is in voxels, and the atlases are 1 mm isotropic, so the
check is only meaningful for inputs near 1 mm. A 1.5 x 1.5 x 4 mm input
(9 mm^3 voxels) flags almost every label as undersize even when the volumes
in mm^3 are plausible; the app now refuses such inputs (--max-voxel-size).

Calibration
-----------
On the 38 delivered Caltech subjects, a threshold of 5x the atlas maximum flags
label 4 in 38 of 38 and nothing else. 10x and 20x flag exactly the same thing.
The default ships at 10x: generous enough that genuinely variable structures
never trip it (the lateral ventricles reach 28011 voxels in the atlases and
1.5x that in the delivered data), strict enough to catch a 71x fallback.

The undersize direction (below atlas minimum / 5, and only for labels present
in all twenty atlases) flags label 69, the optic chiasm, in 2 of 38 -- a real
miss of a very small structure, not noise.

This is a sanity gate, not a QC metric. It answers "is this number possible",
not "is this segmentation good".

Usage
-----
    python scripts/check_label_volume_plausibility.py <path> [<path> ...]

Each path may be a stats CSV, a subject directory, or a derivative root that
contains subject directories. Add ``--protocol freesurfer`` when checking
``--method quick`` output, whose ids are FreeSurfer aseg rather than the atlas
vocabulary. Exit status is 0 when everything is plausible, 1 otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.antspy.label_vocabulary import (  # noqa: E402
    ASEG_TO_NEUROMORPHOMETRICS, NEUROMORPHOMETRICS_TO_ASEG)

RANGES_FILE = Path(__file__).resolve().parent.parent / "src" / "antspy" / "atlas_label_volume_ranges.json"

#: Multiples of the atlas range beyond which a volume is treated as impossible.
#: See "Calibration" above; these are deliberately loose.
OVERSIZE_FACTOR = 10.0
UNDERSIZE_FACTOR = 5.0


def load_ranges(path: Path = RANGES_FILE) -> tuple[dict[int, dict], int]:
    data = json.loads(path.read_text())
    return {int(k): v for k, v in data["labels"].items()}, int(data["n_atlases"])


def read_label_stats(csv_path: Path) -> dict[int, float]:
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    out: dict[int, float] = {}
    for row in rows:
        if "Label" not in row or "VolumeInVoxels" not in row:
            raise ValueError(f"{csv_path} is not an ANTs label stats file")
        out[int(float(row["Label"]))] = float(row["VolumeInVoxels"])
    return out


def check_one(volumes: dict[int, float], ranges: dict[int, dict], n_atlases: int,
              protocol: str = "neuromorphometrics",
              oversize: float = OVERSIZE_FACTOR,
              undersize: float = UNDERSIZE_FACTOR,
              judge_missing: bool = False) -> list[dict]:
    """Return one finding per implausible label. Empty list means plausible.

    With ``judge_missing``, ``volumes`` is taken to be the whole stats file,
    and a structure present in every atlas but absent from it is reported as
    ``missing``: the worst case of undersize, which has no row to read.

    ``protocol`` says which numbering ``volumes`` uses. A ``quick`` output is
    FreeSurfer aseg and is translated into the atlas vocabulary first, because
    the ranges are expressed there.
    """
    findings = []
    for label, vol in sorted(volumes.items()):
        key = label
        if protocol == "freesurfer":
            # Cortical DKT ids are shared; subcortical ones need translating.
            key = ASEG_TO_NEUROMORPHOMETRICS.get(label, label)
        ref = ranges.get(key)
        if ref is None:
            continue  # no atlas evidence for this id; not this check's business
        if vol > ref["max"] * oversize:
            findings.append({
                "label": label, "atlas_label": key, "volume": vol,
                "atlas_min": ref["min"], "atlas_max": ref["max"],
                "kind": "oversize", "factor": vol / max(ref["max"], 1),
            })
        elif ref["n_atlases"] == n_atlases and vol < ref["min"] / undersize:
            # Only judge absence for structures every atlas agrees are present.
            findings.append({
                "label": label, "atlas_label": key, "volume": vol,
                "atlas_min": ref["min"], "atlas_max": ref["max"],
                "kind": "undersize", "factor": vol / max(ref["min"], 1),
            })
    if judge_missing:
        seen = {ASEG_TO_NEUROMORPHOMETRICS.get(L, L) if protocol == "freesurfer" else L
                for L in volumes}
        for key, ref in sorted(ranges.items()):
            if ref["n_atlases"] == n_atlases and key not in seen:
                label = NEUROMORPHOMETRICS_TO_ASEG.get(key, key) if protocol == "freesurfer" else key
                findings.append({
                    "label": label, "atlas_label": key, "volume": 0.0,
                    "atlas_min": ref["min"], "atlas_max": ref["max"],
                    "kind": "missing", "factor": 0.0,
                })
    return findings


def find_stats_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    hits = sorted(root.glob("**/*_antslabelstats.csv"))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path,
                    help="stats CSV, subject directory, or derivative root")
    ap.add_argument("--protocol", choices=("neuromorphometrics", "freesurfer"),
                    default="neuromorphometrics",
                    help="numbering of the stats being checked; use freesurfer "
                         "for --method quick output (default: neuromorphometrics)")
    ap.add_argument("--oversize-factor", type=float, default=OVERSIZE_FACTOR)
    ap.add_argument("--undersize-factor", type=float, default=UNDERSIZE_FACTOR)
    args = ap.parse_args()

    ranges, n_atlases = load_ranges()

    stats_files: list[Path] = []
    for p in args.paths:
        if not p.exists():
            print(f"no such path: {p}", file=sys.stderr)
            return 2
        stats_files.extend(find_stats_files(p))
    if not stats_files:
        print("no *_antslabelstats.csv found under the given paths", file=sys.stderr)
        return 2

    n_bad = 0
    for f in stats_files:
        try:
            volumes = read_label_stats(f)
        except (ValueError, OSError) as exc:
            print(f"FAIL  {f}: {exc}")
            n_bad += 1
            continue
        findings = check_one(volumes, ranges, n_atlases, args.protocol,
                             args.oversize_factor, args.undersize_factor,
                             # quick warps one pre-fused template map, which
                             # holds no right vessel or optic chiasm and one
                             # voxel of left vessel, so absence there says
                             # nothing about the subject.
                             judge_missing=args.protocol == "neuromorphometrics")
        name = f.parent.parent.name or f.name
        if not findings:
            print(f"ok    {name}: {len(volumes)} labels all within "
                  f"[{args.undersize_factor:g}x under, {args.oversize_factor:g}x over] atlas range")
            continue
        n_bad += 1
        print(f"FAIL  {name}: {len(findings)} implausible label(s)")
        for d in findings:
            shown = (f"{d['label']}" if d["label"] == d["atlas_label"]
                     else f"{d['label']} (atlas {d['atlas_label']})")
            print(f"        label {shown}: {d['volume']:,.0f} voxels, "
                  f"atlas range {d['atlas_min']:,}-{d['atlas_max']:,} "
                  f"({d['kind']}, {d['factor']:.1f}x)")

    print(f"\n{len(stats_files)} checked, {n_bad} with implausible volumes")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
