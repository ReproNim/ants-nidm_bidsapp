"""Diagnose why --method quick's output never matches the app's own label
semantics, independent of registration accuracy.

2026-09-17/18 investigation: quick's dseg placed structures 45-68 voxels from
where fusion and the delivered reference agree they sit (real GPU tier-2
output, job 22717989). Two registration engines were tested end to end on the
exact same subject/template pair -- FireANTs (SyN and Greedy, masked and
unmasked) and ANTsPy's own mature `ants.registration(type_of_transform=
"SyNRA")`, completely independent of FireANTs -- and ALL FOUR produced the
same failure (centroid_dist ~16 and ~68 voxels for labels 4 and 11
respectively). Registration engine is therefore ruled out.

This script checks the actual root cause: `quick` warps
`OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_OASIS-30_v2.nii.gz` (the single
pre-fused template-space label map), but the rest of the app -- `fusion`'s
live per-atlas registration, the delivered reference derivative, and the
former tier-3 gate's label semantics (see label_boundary_analysis.py) -- all
derive from `OASIS-TRT-20_DKT31_CMA_labels_v2/` (the per-atlas label files).
These are DIFFERENT label vocabularies that happen to share a numeric range:
the pre-fused template file's ids look like standard FreeSurfer aseg codes
(5=Left-Inf-Lat-Vent, 6/7=Left-Cerebellum, 10=Left-Thalamus-Proper,
12/13=Left-Putamen/Pallidum, 17/18=Left-Hippocampus/Amygdala, ...), while the
per-atlas/reference vocabulary uses the Mindboggle/Klein-Tourville numbering
(e.g. 38="Left-Porg", 40="Right-Cerebral-Exterior" per
ants_seg_to_nidm/mapping_data/FreeSurferColorLUT.txt -- names from a
different numbering convention entirely). Even ids that are numerically
shared (4, 11, ...) do not denote the same structure: template label 4 is
~7700 voxels here, reference label 4 is ~77000 -- a 10x difference.

If this diagnosis is confirmed, --method quick cannot be fixed by registration
tuning: it needs either a corrected pre-fused template label file built from
the SAME per-atlas vocabulary as `fusion`, or a remapping table from the
template file's ids to the app's vocabulary. This predates the FireANTs
migration entirely -- template_labels_path and its content are unchanged by
the SyN/Greedy switch -- so it is not a regression from this week's work.

Run inside the container; no GPU needed (pure nibabel/numpy).
"""
import argparse
import json

import nibabel as nib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-labels", required=True,
                     help="OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_OASIS-30_v2.nii.gz")
    ap.add_argument("--reference-dseg", required=True,
                     help="A delivered reference sub-*/anat/*_space-orig_dseg.nii.gz")
    ap.add_argument("--out", default="quick_label_vocabulary.json")
    args = ap.parse_args()

    tmpl = nib.load(args.template_labels)
    ref = nib.load(args.reference_dseg)
    ta = np.asarray(tmpl.dataobj).astype(int)
    ra = np.asarray(ref.dataobj).astype(int)

    t_ids = sorted(set(np.unique(ta).tolist()) - {0})
    r_ids = sorted(set(np.unique(ra).tolist()) - {0})
    only_t = sorted(set(t_ids) - set(r_ids))
    only_r = sorted(set(r_ids) - set(t_ids))
    shared = sorted(set(t_ids) & set(r_ids))

    t_vol = {i: int((ta == i).sum()) for i in t_ids}
    r_vol = {i: int((ra == i).sum()) for i in r_ids}
    # For shared ids, how much do the volumes disagree? A real match would be
    # roughly the same order of magnitude (allowing for template-vs-subject
    # scale); a vocabulary collision looks like arbitrary, large ratios.
    shared_ratio = {i: round(t_vol[i] / max(1, r_vol[i]), 2) for i in shared}

    result = {
        "n_template_ids": len(t_ids), "n_reference_ids": len(r_ids),
        "n_shared_ids": len(shared),
        "template_only_ids": only_t, "reference_only_ids": only_r,
        "shared_ids_volume_ratio_template_over_reference": shared_ratio,
    }
    print(f"template: {len(t_ids)} ids, reference: {len(r_ids)} ids, shared: {len(shared)}")
    print(f"ids only in template (likely standard aseg): {only_t}")
    print(f"ids only in reference (likely Mindboggle/Klein-Tourville): {only_r}")
    extreme = {i: r for i, r in shared_ratio.items() if r < 0.3 or r > 3.0}
    print(f"shared ids with >3x volume disagreement (vocabulary collision candidates): {extreme}")

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
