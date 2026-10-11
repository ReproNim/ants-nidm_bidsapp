"""Folding diagnostic for the tier-2 Jacobian-gate failure.

Tier 2 (job 22186508, 2026-09-07) died because atlas 18 folded on 0.105% of
voxels against the spec's 0.1% gate, with three more atlases between 0.06%
and 0.09%. Before touching the threshold or the regularisation this script
answers two questions on one subject, using the production code path
(`warp_atlases` with the gate disabled and `folding_fraction` instrumented):

1. WHERE are the folded voxels -- inside the brain mask, where labels live,
   or in the zero background outside it?
2. How far does each sanctioned tuning lever (SyN learning
   rate, smoothing sigmas) move the per-atlas folding fraction, and what does
   it cost in a cheap accuracy proxy (majority-vote Dice against the
   delivered ANTs derivative)?

Runs inside the container on a GPU node; see scripts/diag_folding.sbatch.
"""
import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/opt")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("diag")

# Round 5 (job 22320466): GreedyRegistration with FireANTs' OASIS schedule
# lifted primary median Dice from 0.81 to 0.852 (NCC 0.54 -> 0.655, zero
# folding, 141 s for 20 atlases), leaving 4 primary labels (41, 40, 1021,
# 2019) below 0.75 and relaxed 49/50 below 0.70. Round 6 tunes around Greedy.
G = {"DEFORMABLE": "greedy", "LOSS_MASK_MODE": "max"}
CONFIGS = {
    "g": dict(G),
    "g_w0.5": dict(G, SMOOTH_WARP_SIGMA=0.5),
    "g_lr0.25": dict(G, GREEDY_LR=0.25),
    "g_lr1.0": dict(G, GREEDY_LR=1.0),
    "g_cc5": dict(G, GREEDY_CC_KERNEL_SIZE=5),
    "g_cc9": dict(G, GREEDY_CC_KERNEL_SIZE=9),
    "g_iters400": dict(G, GREEDY_ITERATIONS=[400, 300, 150]),
    "g_grad0.5": dict(G, SMOOTH_GRAD_SIGMA=0.5),
    "g_grad2": dict(G, SMOOTH_GRAD_SIGMA=2.0),
    "g_nearest": dict(G, LABEL_INTERP="nearest"),
}

def atlas_pairs(brains_dir, labels_dir):
    def by_number(paths):
        out = {}
        for p in paths:
            m = re.search(r"OASIS-TRT-20-(\d+)", p.name)
            assert m, p
            out[int(m.group(1))] = p
        return out
    b = by_number(Path(brains_dir).glob("*.nii.gz"))
    l = by_number(Path(labels_dir).glob("*.nii.gz"))
    assert set(b) == set(l) and len(b) == 20, (sorted(b), sorted(l))
    keys = sorted(b)
    return [b[k] for k in keys], [l[k] for k in keys]


def preprocess(t1w, out, threads):
    import ants
    from src.antspy.wrapper import ANTsSegmentation

    brain_p, mask_p = out / "masked_brain.nii.gz", out / "mask.nii.gz"
    if brain_p.exists() and mask_p.exists():
        log.info("reusing cached preprocessing in %s", out)
    else:
        # The constructor validates bids_dir even though only the preprocessing
        # method is used here; /bids is bound by diag_folding.sbatch.
        seg = ANTsSegmentation(bids_dir="/bids", output_dir=str(out / "seg_out"),
                               temp_dir=str(out / "tmp"), device="cuda", num_threads=threads)
        res = seg.compute_cortical_thickness(ants.image_read(str(t1w)))
        masked = res["BrainSegmentationN4"] * res["BrainExtractionMask"]
        ants.image_write(masked, str(brain_p))
        ants.image_write(res["BrainExtractionMask"], str(mask_p))
    return ants.image_read(str(brain_p)), ants.image_read(str(mask_p)), mask_p


def dice_table(vote, ref):
    ids = sorted(set(np.unique(ref)) - {0})
    d = {}
    for i in ids:
        a, b = vote == i, ref == i
        denom = a.sum() + b.sum()
        d[int(i)] = float(2 * np.logical_and(a, b).sum() / denom) if denom else float("nan")
    return d


def main():
    ap = argparse.ArgumentParser()
    # ANTsSegmentation sets ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS only when it is
    # constructed, i.e. only on the uncached preprocessing path; the container
    # default is 1, which made joint label fusion single-threaded here.
    import os
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(next((a.split("=")[-1] if "=" in a else sys.argv[i + 1] for i, a in enumerate(sys.argv) if a.startswith("--threads")), "8"))
    ap.add_argument("--t1w", required=True)
    ap.add_argument("--reference-dseg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = ap.parse_args()

    import torch
    from fireants.io import Image
    from fireants.utils.imageutils import jacobian
    from src.registration import fireants as fr

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target, mask, mask_p = preprocess(args.t1w, out, args.threads)
    T1s, labels = atlas_pairs("/opt/data/OASIS-TRT-20_brains",
                              "/opt/data/OASIS-TRT-20_DKT31_CMA_labels_v2")

    # Mask on FireANTs' own grid ordering (loaded the same way the fixed image
    # is), cropped like the Jacobian interior in folding_fraction.
    mask_t = Image.load_file(str(mask_p), device="cuda").array[0, 0] > 0.5
    mask_c = mask_t[1:-1, 1:-1, 1:-1]
    log.info("mask voxels %d of %d (%.1f%%)", int(mask_t.sum()), mask_t.numel(),
             100 * float(mask_t.sum()) / mask_t.numel())

    records = []

    def instrumented(registration, fixed_batch, moving_batch):
        warp = registration.get_warped_coordinates(fixed_batch, moving_batch)
        jac = jacobian(warp).permute(0, 2, 3, 4, 1, 5)[:, 1:-1, 1:-1, 1:-1, :]
        bad = torch.linalg.det(jac) <= 0
        fr_all = []
        for k in range(bad.shape[0]):
            b = bad[k]
            rec = dict(
                overall=float(b.float().mean()),
                inside_mask=float(b[mask_c].float().mean()),
                outside_mask=float(b[~mask_c].float().mean()),
                n_bad=int(b.sum()),
                n_bad_inside=int(b[mask_c].sum()),
            )
            records.append(rec)
            fr_all.append(rec["overall"])
        return fr_all

    fr.folding_fraction = instrumented
    fr.MAX_FOLDING_FRACTION = 1.0  # record, never raise

    ref = np.asarray(__import__("ants").image_read(args.reference_dseg).numpy()).astype(np.int32)
    defaults = {k: getattr(fr, k) for k in ("MOMENTS_INIT", "LOSS_MASKED", "LOSS_TYPE", "LOSS_MASK_MODE", "LOSS_MASK_MOVING", "DEFORMABLE", "GREEDY_ITERATIONS", "GREEDY_LR", "GREEDY_CC_KERNEL_SIZE", "AFFINE_LR", "AFFINE_SCALES", "SYN_SCALES", "AFFINE_ITERATIONS", "CC_KERNEL_SIZE",
                                             "SMOOTH_WARP_SIGMA", "SMOOTH_GRAD_SIGMA", "SYN_LR", "SYN_ITERATIONS")}
    import ants as _ants
    ref_img = _ants.image_read(args.reference_dseg)
    vox_mm3 = float(np.prod(ref_img.spacing))
    # Tier-3 volume groups, from the reference cohort's median label volumes.
    import csv, glob as _glob
    cohort = {}
    for f in _glob.glob(str(Path(args.reference_dseg).parents[2] / "sub-*/stats/*antslabelstats.csv")):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                try:
                    lab = int(float(row.get("Label") or row.get("label")))
                    v = float(row.get("VolumeInVoxels") or row.get("Volume") or row.get("volume") or 0)
                except (TypeError, ValueError):
                    continue
                cohort.setdefault(lab, []).append(v)
    med_mm3 = {k: float(np.median(v)) * vox_mm3 for k, v in cohort.items()}
    PRIMARY = sorted(k for k, v in med_mm3.items() if v >= 2000 and k not in (15, 2033))
    RELAXED = sorted(k for k, v in med_mm3.items() if 250 <= v < 2000)
    log.info("spec groups from %d reference subjects: %d primary, relaxed %s", len(next(iter(cohort.values()), [])), len(PRIMARY), RELAXED)

    def score(seg, ref_arr):
        d = {}
        for i in set(PRIMARY) | set(RELAXED):
            A, B = seg == i, ref_arr == i
            den = A.sum() + B.sum()
            d[i] = float(2 * np.logical_and(A, B).sum() / den) if den else float("nan")
        prim = np.array([d[i] for i in PRIMARY if not np.isnan(d[i])])
        return dict(primary_median=float(np.median(prim)), primary_below_0p75=int((prim < 0.75).sum()),
                    primary_min=float(prim.min()), relaxed={int(i): round(d[i], 3) for i in RELAXED},
                    relaxed_below_0p70=int(sum(d[i] < 0.70 for i in RELAXED if not np.isnan(d[i]))),
                    gate_pass=bool(np.median(prim) >= 0.85 and (prim >= 0.75).all() and all(d[i] >= 0.70 for i in RELAXED)),
                    per_label={int(k): round(v, 3) for k, v in d.items()})

    # Alignment after the affine stage alone, before SyN: separates "the affine
    # never gets there" from "SyN degrades it". Recorded by wrapping optimize().
    from fireants.registration.affine import AffineRegistration
    affine_cov = []
    _orig_affine_optimize = AffineRegistration.optimize

    def _optimize_and_record(self, *a, **kw):
        out = _orig_affine_optimize(self, *a, **kw)
        with torch.no_grad():
            moved = self.evaluate(self.fixed_images, self.moving_images)[:, 0]
            for k in range(moved.shape[0]):
                nz = moved[k] > 0
                affine_cov.append(float((nz & mask_t).sum() / mask_t.sum()))
        return out
    AffineRegistration.optimize = _optimize_and_record
    tgt_np = target.numpy().astype(np.float32)
    mask_np = mask.numpy() > 0
    summary = {}
    for name in args.configs:
        for k, v in defaults.items():
            setattr(fr, k, v)
        label_interp = CONFIGS[name].get("LABEL_INTERP", "genericlabel")
        for k, v in CONFIGS[name].items():
            if k != "LABEL_INTERP":
                setattr(fr, k, v)
        log.info("=== config %s: %s label_interp=%s ===", name, {k: getattr(fr, k) for k in defaults}, label_interp)
        records.clear()
        affine_cov.clear()
        t0 = time.time()
        import gc
        gc.collect()
        torch.cuda.empty_cache()  # the previous config's registration objects
        warped_atlases, warped_labels = fr.warp_atlases(
            target, mask, T1s, labels, temp_dir=out / "tmp" / name, device="cuda",
            batch_size=args.batch_size, label_interpolation=label_interp)
        reg_s = time.time() - t0
        for k, rec in enumerate(records):
            rec["atlas"] = T1s[k].name
        stack = torch.stack([torch.as_tensor(l.numpy().astype(np.int64)) for l in warped_labels]).cuda()
        vote = torch.mode(stack, dim=0).values.cpu().numpy().astype(np.int32)
        del stack
        torch.cuda.empty_cache()
        dice = dice_table(vote, ref) if vote.shape == ref.shape else {}
        # The real thing: joint label fusion with the wrapper's parameters
        # (src/antspy/wrapper.py), scored against the spec's gate groups.
        t1 = time.time()
        try:
            fusion = _ants.joint_label_fusion(
                target_image=target, target_image_mask=mask, atlas_list=warped_atlases,
                label_list=warped_labels, beta=4.0, rad=2, rho=0.01, usecor=False,
                r_search=3, nonnegative=False, no_zeroes=False, max_lab_plus_one=False, verbose=False)
            seg = fusion["segmentation"].numpy().astype(np.int32)
            jlf = score(seg, ref)
            jlf["seconds"] = time.time() - t1
            _ants.image_write(fusion["segmentation"], str(out / f"jlf_{name}_dseg.nii.gz"))
        except Exception as e:  # keep the sweep alive; report the failure
            log.exception("JLF failed for %s", name)
            jlf = {"error": f"{type(e).__name__}: {e}"}
        # Registration quality per atlas, independent of labels: how much of
        # the target brain the warped atlas covers, and the NCC inside the mask.
        align = []
        for wa in warped_atlases:
            a = wa.numpy().astype(np.float32)
            nz = a > 0
            align.append(dict(mask_covered=float((nz & mask_np).sum() / mask_np.sum()),
                              ncc_in_mask=float(np.corrcoef(tgt_np[mask_np], a[mask_np])[0, 1])))
        del warped_atlases, warped_labels
        overall = [r["overall"] for r in records]
        s = dict(
            params={k: getattr(fr, k) for k in defaults},
            registration_seconds=reg_s,
            max_folding=max(overall),
            n_atlases_over_0p1pct=sum(o > 0.001 for o in overall),
            n_atlases_over_0p05pct=sum(o > 0.0005 for o in overall),
            mean_inside_mask=float(np.mean([r["inside_mask"] for r in records])),
            mean_outside_mask=float(np.mean([r["outside_mask"] for r in records])),
            frac_of_bad_voxels_inside_mask=(sum(r["n_bad_inside"] for r in records) / max(1, sum(r["n_bad"] for r in records))),
            affine_stage_mean_covered=float(np.mean(affine_cov)) if affine_cov else None,
            affine_stage_min_covered=float(np.min(affine_cov)) if affine_cov else None,
            mean_mask_covered=float(np.mean([x["mask_covered"] for x in align])),
            min_mask_covered=float(np.min([x["mask_covered"] for x in align])),
            mean_ncc_in_mask=float(np.mean([x["ncc_in_mask"] for x in align])),
            per_atlas_alignment=align,
            jlf=jlf,
            majority_vote_mean_dice=float(np.nanmean(list(dice.values()))) if dice else None,
            majority_vote_dice=dice,
            per_atlas=records[:],
        )
        summary[name] = s
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        log.info("RESULT %-22s affine_cov=%.3f(min %.3f) -> covered=%.3f(min %.3f) ncc=%.3f | max_fold=%.4f%% over0.1=%d over0.05=%d "
                 "inside=%.4f%% outside=%.4f%% bad_inside_frac=%.2f | dice=%s | reg=%.0fs",
                 name, s["affine_stage_mean_covered"] or -1, s["affine_stage_min_covered"] or -1,
                 s["mean_mask_covered"], s["min_mask_covered"], s["mean_ncc_in_mask"],
                 100 * s["max_folding"], s["n_atlases_over_0p1pct"], s["n_atlases_over_0p05pct"],
                 100 * s["mean_inside_mask"], 100 * s["mean_outside_mask"],
                 s["frac_of_bad_voxels_inside_mask"],
                 f"{s['majority_vote_mean_dice']:.4f}" if s["majority_vote_mean_dice"] is not None else "n/a",
                 reg_s)
        if "error" in jlf:
            log.info("   JLF %-22s ERROR %s", name, jlf["error"])
        else:
            log.info("   JLF %-22s GATE %s | primary median=%.3f min=%.3f below0.75=%d | relaxed below0.70=%d %s | jlf=%.0fs",
                     name, "PASS" if jlf["gate_pass"] else "FAIL", jlf["primary_median"], jlf["primary_min"],
                     jlf["primary_below_0p75"], jlf["relaxed_below_0p70"], jlf["relaxed"], jlf["seconds"])
    log.info("DIAG DONE")


if __name__ == "__main__":
    main()
