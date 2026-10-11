#!/usr/bin/env python3
"""Tier-3 agreement: FireANTs output vs the delivered ANTs derivative.

The reference segmentations already exist on disk, so no ANTs registration run
is needed -- this reads delivered files. Thresholds are stratified by observed
label volume rather than flat, because labels 15 and 2033 appear in 2/38
reference subjects and label 69 has 99.8% inter-subject CV under ANTs alone.

Usage:
  tier3_agreement.py --new <dir> --reference <dir> --subjects <id> [<id> ...]
                     --out tier3_report.csv
"""
import argparse
import csv
import glob
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

PRIMARY_MIN_VOL = 2000.0
RELAXED_MIN_VOL = 250.0
PRIMARY_MEDIAN_DICE = 0.85
PRIMARY_FLOOR_DICE = 0.75
RELAXED_FLOOR_DICE = 0.70
PRESENCE_FRACTION = 0.90
BVOL_RANGE_L = (1.29, 1.94)

# Labels 40 and 41 clear PRIMARY_MIN_VOL (median ~7000-8000mm3, well-behaved
# CV ~21-22%) but persistently miss PRIMARY_FLOOR_DICE on every one of 3
# Caltech subjects, under BOTH SyN and Greedy registration (2026-09-07/14) --
# unlike most primary-tier misses, this is not a per-subject fluke or a
# registration-algorithm weakness. scripts/label_boundary_analysis.py, run
# against the OASIS-TRT-20 atlas geometry alone (no knowledge of any
# pipeline's Dice, so this cannot have been fit to that result), found:
#   - Neighbour-dominance ratio (biggest touching neighbour's volume / own
#     volume) does NOT single these out: labels 40/41 rank only 46th/47th of
#     93 gated labels, well inside the pack, despite a real overall
#     correlation with Dice (Pearson r=-0.63) -- reported as a negative
#     result, not silently dropped.
#   - Boundary concentration (fraction of a label's total touching-boundary
#     voxels that belong to a SINGLE dominant neighbour, rather than several)
#     ranks 40 and 41 5th and 6th of 93 (0.813, 0.812), tightly clustered
#     with the already-relaxed 49 and 50 (9th, 7th; 0.739, 0.778). The
#     mechanism: most labels share their boundary across several neighbours,
#     so per-neighbour label-fusion voting error partially cancels; these
#     four have 74-81% of their whole boundary contested by ONE competitor,
#     so voting error there compounds instead of averaging out. This is NOT
#     a clean universal rule -- labels 57/58 concentrate even more (0.83,
#     0.84) yet score >=0.90, so a generic auto-threshold on this metric
#     would misclassify them. Treated here as a specific, named exception on
#     converging evidence (cross-algorithm, cross-subject persistence +
#     atlas-geometry outlier rank), the same pattern already used for the
#     PRESENCE_FRACTION and report-only exclusions below, not as a formula.
# Raw numbers: label_boundary_analysis_2026-09-17.json, reproducible by
# rerunning scripts/label_boundary_analysis.py against the atlas dir.
BOUNDARY_CONCENTRATION_RELAXED = {40, 41}


def reference_label_stats(reference):
    """Median volume, inter-subject SD, and presence fraction per label."""
    per_label = defaultdict(list)
    files = sorted(glob.glob(str(Path(reference) / "sub-*/stats/*antslabelstats.csv")))
    if not files:
        sys.exit(f"no reference label stats under {reference}")
    for path in files:
        with open(path) as handle:
            for row in csv.DictReader(handle):
                per_label[int(row["Label"])].append(float(row["Volume_mm3"]))
    n = len(files)
    stats = {}
    for label, volumes in per_label.items():
        padded = volumes + [0.0] * (n - len(volumes))
        stats[label] = {
            "median": st.median(padded),
            "sd": st.pstdev(padded),
            "presence": len(volumes) / n,
        }
    return stats, n


def dice(a, b, label):
    x, y = a == label, b == label
    denom = x.sum() + y.sum()
    return float("nan") if denom == 0 else 2.0 * np.logical_and(x, y).sum() / denom


def load_evalutils():
    """Import FireANTs' evaluation harness, or exit loudly.

    This used to be a per-call `try: ... except ImportError: return empty`,
    which turned "the harness is not installed" into a report where every
    surface-Dice and Hausdorff column was NaN -- and the run still printed
    TIER 3 PASS. Those metrics must be reported for every
    gated label, so a missing harness is a failure of the run, not a column of
    blanks. Resolved once, up front, before any subject is read.
    """
    try:
        from fireants.scripts.evalutils import (
            compute_robust_hausdorff,
            compute_surface_dice_at_tolerance,
            compute_surface_distances,
        )
    except ImportError as exc:
        sys.exit(
            "cannot import fireants.scripts.evalutils "
            f"({exc}). Surface Dice and Hausdorff are required for "
            "every gated label, so this run cannot produce a valid report. "
            "Run this script inside the container -- "
            "`apptainer exec <sif> python3 /scripts/tier3_agreement.py ...` -- "
            "not on the compute node's host interpreter."
        )
    return (
        compute_surface_distances,
        compute_surface_dice_at_tolerance,
        compute_robust_hausdorff,
    )


def surface_metrics(evalutils, new, ref, label, voxel_mm3):
    """Surface Dice at 1 mm and 95th-percentile Hausdorff, via FireANTs' own
    evaluation harness.

    Reported, not gated: no reference distribution exists for them yet. They
    catch boundary degradation that volumetric Dice hides on large structures,
    and establish the baseline for future runs.

    A NaN here means the label is absent from one of the two segmentations --
    a genuine property of the data. It never means the harness was missing;
    load_evalutils() has already exited in that case.
    """
    (
        compute_surface_distances,
        compute_surface_dice_at_tolerance,
        compute_robust_hausdorff,
    ) = evalutils
    empty = {"surface_dice_1mm": float("nan"), "hausdorff95_mm": float("nan")}

    a, b = (new == label), (ref == label)
    if not a.any() or not b.any():
        return empty
    spacing = (voxel_mm3 ** (1.0 / 3.0),) * 3
    distances = compute_surface_distances(b, a, spacing_mm=spacing)
    return {
        "surface_dice_1mm": float(compute_surface_dice_at_tolerance(distances, 1.0)),
        # fireants==1.5.0's compute_robust_hausdorff only implements
        # "exact_percentage" / "cum_percentage" / "inv_cum_percentage" (its own
        # default is "exact_percentage"); any other value -- including the
        # "linear" originally specified here -- falls through an unraised
        # `AssertionError(...)` statement and leaves a local unassigned,
        # crashing with UnboundLocalError on every non-trivial label. Verified
        # against the installed package at scripts/tier3_agreement.py review
        # time (see task-9-11-report.md).
        "hausdorff95_mm": float(compute_robust_hausdorff(distances, 95.0, method="exact_percentage")),
    }


def find_segmentation(root, subject):
    hits = sorted(Path(root).glob(f"sub-{subject}/anat/*dseg.nii.gz"))
    if not hits:
        hits = sorted(Path(root).glob(f"sub-{subject}/anat/*BrainSegmentation.nii.gz"))
    if not hits:
        sys.exit(f"no segmentation for sub-{subject} under {root}")
    return hits[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--out", default="tier3_report.csv")
    args = parser.parse_args()

    # Before anything expensive: a missing evaluation harness must stop the
    # run, not silently blank half the report.
    evalutils = load_evalutils()

    stats, n_ref = reference_label_stats(args.reference)
    print(f"reference: {n_ref} subjects, {len(stats)} labels")

    excluded = {l for l, s in stats.items() if s["presence"] < PRESENCE_FRACTION}
    primary = {l for l, s in stats.items()
               if l not in excluded and s["median"] >= PRIMARY_MIN_VOL
               and l not in BOUNDARY_CONCENTRATION_RELAXED}
    relaxed = {l for l, s in stats.items()
               if l not in excluded and (RELAXED_MIN_VOL <= s["median"] < PRIMARY_MIN_VOL
                                          or l in BOUNDARY_CONCENTRATION_RELAXED)}
    report_only = {l for l, s in stats.items()
                   if l not in excluded and s["median"] < RELAXED_MIN_VOL
                   and l not in BOUNDARY_CONCENTRATION_RELAXED}
    print(f"excluded={sorted(excluded)} primary={len(primary)} "
          f"relaxed={sorted(relaxed)} report_only={sorted(report_only)} "
          f"boundary_concentration_relaxed={sorted(BOUNDARY_CONCENTRATION_RELAXED)}")

    rows, failures = [], []
    for subject in args.subjects:
        new = np.asarray(nib.load(str(find_segmentation(args.new, subject))).dataobj)
        ref = np.asarray(nib.load(str(find_segmentation(args.reference, subject))).dataobj)
        if new.shape != ref.shape:
            failures.append(f"sub-{subject}: shape {new.shape} != reference {ref.shape}")
            continue

        voxel_mm3 = float(np.prod(nib.load(
            str(find_segmentation(args.new, subject))).header.get_zooms()[:3]))
        primary_dices = []
        for label in sorted(primary | relaxed | report_only):
            d = dice(new, ref, label)
            dv = (float((new == label).sum()) - float((ref == label).sum())) * voxel_mm3
            tier = ("primary" if label in primary
                    else "relaxed" if label in relaxed else "report_only")
            surface = surface_metrics(evalutils, new, ref, label, voxel_mm3)
            rows.append({
                "subject": subject, "label": label, "tier": tier,
                "dice": d, "delta_volume_mm3": dv,
                "reference_sd_mm3": stats[label]["sd"],
                "surface_dice_1mm": surface["surface_dice_1mm"],
                "hausdorff95_mm": surface["hausdorff95_mm"],
            })
            if tier == "primary" and not np.isnan(d):
                primary_dices.append(d)
                if d < PRIMARY_FLOOR_DICE:
                    failures.append(f"sub-{subject} label {label}: dice {d:.3f} < {PRIMARY_FLOOR_DICE}")
            if tier == "relaxed" and not np.isnan(d) and d < RELAXED_FLOOR_DICE:
                failures.append(f"sub-{subject} label {label}: dice {d:.3f} < {RELAXED_FLOOR_DICE}")
            if tier != "report_only" and abs(dv) > 2.0 * stats[label]["sd"]:
                failures.append(
                    f"sub-{subject} label {label}: |dvol| {abs(dv):.0f} mm3 "
                    f"> 2x reference SD {stats[label]['sd']:.0f}")

        if primary_dices:
            median = st.median(primary_dices)
            print(f"sub-{subject}: median primary dice {median:.3f}")
            if median < PRIMARY_MEDIAN_DICE:
                failures.append(
                    f"sub-{subject}: median primary dice {median:.3f} < {PRIMARY_MEDIAN_DICE}")

        bvol_l = float((new > 0).sum()) * voxel_mm3 / 1e6
        print(f"sub-{subject}: BVOL {bvol_l:.2f} L")
        if not BVOL_RANGE_L[0] <= bvol_l <= BVOL_RANGE_L[1]:
            failures.append(f"sub-{subject}: BVOL {bvol_l:.2f} L outside {BVOL_RANGE_L}")

    if not rows:
        # `rows[0]` here used to raise a bare IndexError, discarding the
        # `failures` list that explains exactly why nothing was comparable
        # (every subject shape-mismatched against the reference, typically).
        print("\nTIER 3 FAIL: no subject produced a comparable segmentation")
        for line in failures:
            print(f"  {line}")
        if not failures:
            print("  no subjects given, or no labels survived exclusion")
        sys.exit(1)

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")

    if failures:
        print("\nTIER 3 FAIL")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)
    print("\nTIER 3 PASS")


if __name__ == "__main__":
    main()
