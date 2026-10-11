"""Leave-one-out accuracy of the fusion against the OASIS-TRT-20 manual labels.

Holds out atlas k, registers the other 19 to it with production warp_atlases,
fuses their labels, and compares the result with atlas k's own manual labels.
The labels are hand-drawn and use the same protocol as the fused output, so
this is a ground-truth measure, unlike a comparison with FreeSurfer.

The target is built the way production builds it: N4 on the atlas brain inside
its mask, multiplied by the mask. Production runs N4 on the whole head before
masking; the atlases ship skull-stripped, so that step is approximated here.

Two fusions are written per fold:
  production  JLF_PARAMS from src/antspy/wrapper.py, as the app runs it
  msd         the same with usecor=False, the setting before Oct 10
The second one asks whether leave-one-out can see the intensity-scale bias at
all. The atlases share one intensity scale, so it may not; the target and
atlas in-mask medians are recorded so the answer can be read off.

Output per fold, under <out>/atlas-NN/:
  production_dseg.nii.gz, msd_dseg.nii.gz, scores.json

Must run inside the app container with a GPU; see diag_jlf_leave_one_out.sbatch.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt")
import ants  # noqa: E402
import numpy as np  # noqa: E402

from src.antspy.wrapper import JLF_PARAMS, ANTsSegmentation  # noqa: E402
from src.registration.fireants import warp_atlases  # noqa: E402


def score(seg, ref):
    """Per-label Dice and volume ratio of seg against the manual labels ref."""
    out = {}
    for L in np.unique(ref):
        if L == 0:
            continue
        s, r = seg == L, ref == L
        sv, rv = int(s.sum()), int(r.sum())
        out[str(int(L))] = dict(vox=sv, ref_vox=rv,
                                dice=float(2 * np.logical_and(s, r).sum() / (sv + rv)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold", type=int, required=True, help="held-out atlas, 1-based")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=5)
    a = ap.parse_args()

    out = Path(a.out) / f"atlas-{a.fold:02d}"
    out.mkdir(parents=True, exist_ok=True)
    seg = ANTsSegmentation(bids_dir=str(out), output_dir=str(out), num_threads=a.num_threads,
                           device="cuda", fireants_batch_size=a.batch_size, verbose=True)
    from natsort import natsorted
    T1s = natsorted(str(p) for p in seg.atlas_dir.glob("*.nii.gz"))
    labels = natsorted(str(p) for p in seg.labels_dir.glob("*.nii.gz"))
    assert len(T1s) == len(labels) == 20, (len(T1s), len(labels))
    k = a.fold - 1
    held_t1, held_lab = T1s[k], labels[k]
    rest_t1 = T1s[:k] + T1s[k + 1:]
    rest_lab = labels[:k] + labels[k + 1:]

    brain = ants.image_read(held_t1)
    mask = ants.get_mask(brain, low_thresh=1e-6, cleanup=0)
    target = ants.n4_bias_field_correction(brain, mask=mask) * mask
    ref = np.rint(ants.image_read(held_lab).numpy()).astype(np.int32)
    m = mask.numpy() > 0
    info = dict(fold=a.fold, held_out=Path(held_t1).name, labels=Path(held_lab).name,
                target_median=float(np.median(target.numpy()[m])),
                held_brain_median=float(np.median(brain.numpy()[m])))

    t0 = time.time()
    with seg._scratch("fireants_") as scratch:
        warped_images, warped_labels = warp_atlases(
            target, mask, rest_t1, rest_lab, temp_dir=scratch, device="cuda",
            batch_size=a.batch_size, label_interpolation=seg.label_interpolation,
            logger=seg.logger)
    info["warp_s"] = round(time.time() - t0)
    info["atlas_medians"] = [float(np.median(w.numpy()[m])) for w in warped_images]

    variants = {"production": dict(JLF_PARAMS), "msd": {**JLF_PARAMS, "usecor": False}}
    info["params"] = variants
    info["scores"] = {}
    for name, params in variants.items():
        t1 = time.time()
        # joint_label_fusion mutates its label images, so each fusion gets copies.
        fused = ants.joint_label_fusion(
            target_image=target, target_image_mask=mask, atlas_list=warped_images,
            label_list=[x.clone() for x in warped_labels], verbose=False, **params)
        dseg = fused["segmentation"]
        ants.image_write(dseg, str(out / f"{name}_dseg.nii.gz"))
        info["scores"][name] = score(np.rint(dseg.numpy()).astype(np.int32), ref)
        info[f"{name}_s"] = round(time.time() - t1)
        print(f"fold {a.fold} {name}: mean Dice "
              f"{np.mean([v['dice'] for v in info['scores'][name].values()]):.3f}", flush=True)

    (out / "scores.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    sys.exit(main())
