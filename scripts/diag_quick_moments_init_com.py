"""Centre-of-mass init translation of --method quick, old inputs vs new (#16).

Runs inside the app container on CPU. Repeats quick's preprocessing for one T1w
(N4 + antsBrainExtraction.sh, as ANTsSegmentation.compute_cortical_thickness
does), then calls src.registration.fireants._moments_init twice on the
resulting subject brain and T_template0_BrainCerebellum:

  old  the masked-loss batches (image + image>0 mask channel)
  new  the plain single-channel images

and prints the translation part of each affine init and their difference.
The brain extraction is rerun here rather than read from the GPU runs (quick
does not save the brain image), so the brain differs from those runs by the
extraction's run-to-run noise.

Usage:
    see scripts/diag_quick_moments_init_com.sbatch
"""
import json
import logging
import sys
from pathlib import Path

import ants

sys.path.insert(0, "/opt")
from src.antspy.wrapper import ANTsSegmentation  # noqa: E402
from src.registration import fireants as reg  # noqa: E402


def main():
    t1w, work, out_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    work.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    seg = ANTsSegmentation(bids_dir=str(t1w.parents[2]), output_dir=str(work),
                           temp_dir=str(work / "tmp"), device="cpu", num_threads=8)
    pre = seg.compute_cortical_thickness(ants.image_read(str(t1w)), register_template=False)
    brain = pre["BrainSegmentationN4"] * pre["BrainExtractionMask"]
    template = ants.image_read(str(seg.template_dir / "T_template0_BrainCerebellum.nii.gz"))
    subject_path = reg.write_nifti(brain, work / "subject_brain.nii.gz")
    template_path = reg.write_nifti(template, work / "template_brain.nii.gz")

    from fireants.io import BatchedImages, Image

    fixed = Image.load_file(str(subject_path), device="cpu")
    moving = Image.load_file(str(template_path), device="cpu")
    inits = {
        "old": reg._moments_init(BatchedImages([reg._with_mask_channel(fixed)]),
                                 BatchedImages([reg._with_mask_channel(moving)])),
        "new": reg._moments_init(BatchedImages([fixed]), BatchedImages([moving])),
    }
    result = {k: v.cpu().numpy().tolist() for k, v in inits.items()}
    told = inits["old"][0, :, -1]
    tnew = inits["new"][0, :, -1]
    result["translation_old"] = told.tolist()
    result["translation_new"] = tnew.tolist()
    result["translation_diff_norm"] = float((told - tnew).norm())
    out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
