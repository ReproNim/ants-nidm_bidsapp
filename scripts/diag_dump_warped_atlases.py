"""Save one subject's 20 warped OASIS-TRT-20 atlases and labels for offline study.

Production never keeps the warped atlases: warp_atlases writes them to a scratch
directory that is deleted as soon as joint_label_fusion has them in memory. This
script runs the same preprocessing and the same warp_atlases call as
ANTsSegmentation.segment_image(method='fusion'), but writes everything to
--out instead:

  target_masked_brain.nii.gz   N4 * BrainExtractionMask, the fusion target
  brain_mask.nii.gz            BrainExtractionMask
  atlas-NN_brain.nii.gz        warped atlas intensity image, NN = 01..20
  atlas-NN_labels.nii.gz       warped atlas label image
  fusion_dseg.nii.gz           production joint_label_fusion on exactly these
  fusion_label-<id>_probseg.nii.gz
  manifest.json                atlas sources, versions, device, timings

The labels are written before joint_label_fusion runs, because that call
modifies its label images in place.

fusion_dseg is the check that the dump matches production: it should agree with
a normal fusion run on the same subject to within FireANTs' run-to-run
nondeterminism (~3% per label, Sep 25). diag_warped_atlas_qc.py does the rest.

Must run inside the app container with a GPU; see diag_dump_warped_atlases.sbatch.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt")

import ants  # noqa: E402
from natsort import natsorted  # noqa: E402

from src.antspy.wrapper import ANTsSegmentation  # noqa: E402
from src.registration.fireants import warp_atlases  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bids", required=True)
    ap.add_argument("--participant", required=True, help="without the sub- prefix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=5)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seg = ANTsSegmentation(bids_dir=a.bids, output_dir=str(out), num_threads=a.num_threads,
                           device="cuda", fireants_batch_size=a.batch_size, verbose=True)
    t1 = Path(a.bids) / f"sub-{a.participant}" / "anat" / f"sub-{a.participant}_T1w.nii.gz"
    manifest = {"participant": a.participant, "t1w": t1.name, "batch_size": a.batch_size}
    t0 = time.time()

    img = seg.load_image(str(t1))
    th = seg.compute_cortical_thickness(img)
    mask = th["BrainExtractionMask"]
    target = th["BrainSegmentationN4"] * mask
    ants.image_write(target, str(out / "target_masked_brain.nii.gz"))
    ants.image_write(mask, str(out / "brain_mask.nii.gz"))
    manifest["preprocess_s"] = round(time.time() - t0)

    T1s = natsorted(str(p) for p in seg.atlas_dir.glob("*.nii.gz"))
    labels = natsorted(str(p) for p in seg.labels_dir.glob("*.nii.gz"))
    manifest["atlases"] = [Path(p).name for p in T1s]
    manifest["labels"] = [Path(p).name for p in labels]

    t1_ = time.time()
    with seg._scratch("fireants_") as scratch:
        warped_images, warped_labels = warp_atlases(
            target, mask, T1s, labels, temp_dir=scratch, device="cuda",
            batch_size=a.batch_size, label_interpolation=seg.label_interpolation,
            logger=seg.logger)
    manifest["warp_s"] = round(time.time() - t1_)
    for i, (wi, wl) in enumerate(zip(warped_images, warped_labels), start=1):
        ants.image_write(wi, str(out / f"atlas-{i:02d}_brain.nii.gz"))
        ants.image_write(wl, str(out / f"atlas-{i:02d}_labels.nii.gz"))

    # Same call and parameters as the production fusion branch.
    t2 = time.time()
    fusion = ants.joint_label_fusion(
        target_image=target, target_image_mask=mask, atlas_list=warped_images,
        label_list=warped_labels, beta=4.0, rad=2, rho=0.01, usecor=False, r_search=3,
        nonnegative=False, no_zeroes=False, max_lab_plus_one=True, verbose=True)
    manifest["fusion_s"] = round(time.time() - t2)
    ants.image_write(fusion["segmentation"], str(out / "fusion_dseg.nii.gz"))
    for lab, p in zip(fusion["segmentation_numbers"], fusion["probabilityimages"]):
        ants.image_write(p, str(out / f"fusion_label-{int(lab)}_probseg.nii.gz"))

    import fireants
    manifest["versions"] = {"ants": ants.__version__,
                            "fireants": getattr(fireants, "__version__", "unknown")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("done", json.dumps(manifest))


if __name__ == "__main__":
    main()
