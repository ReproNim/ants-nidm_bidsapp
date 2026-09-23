"""Why --method fusion reported 89924 voxels of third ventricle, and which
joint-fusion flag fixes it. CPU only, no GPU, no FireANTs, seconds to run.

The finding
-----------
`ants.joint_label_fusion` has no background class. The brain extraction mask
is much larger than the union of the labelled structures -- the atlases leave
CSF and everything between structures at 0 -- so every masked-but-unlabelled
voxel still has to be assigned some real label. An argmax over a probability
stack that is ~0 everywhere returns the first entry, which is the lowest label
id. For the OASIS-TRT-20 vocabulary that is 4, the third ventricle.

Measured on the delivered tier-2 output for sub-0051456 (2026-09-14):

  label 4: 89924 voxels, against ~350-700 in the atlases
           2725 disconnected components
           74% of its voxels within 3 voxels of the mask boundary
             (brain stem control: 13%; left lateral ventricle control: 0%)
           96% of its voxels at a max across-label probability below 0.05
             (every other structure sits at 0.37-0.72)

This reproduction builds the same geometry in miniature -- a mask far larger
than two small structures -- and runs the real `ants.joint_label_fusion` under
the three flag settings. It is the evidence behind `max_lab_plus_one=True` in
`src/antspy/wrapper.py`, and it re-runs in seconds if that choice is ever
questioned.

Expected output:

  current  (both False)   unlabelled band -> label 4, which inflates ~54x
  max_lab_plus_one=True   unlabelled band -> 0, label 4 exactly its true size
  no_zeroes=True          unlabelled band -> 0, but label 4 eroded by a third

`no_zeroes` constrains the solution to voxels non-zero in *every* atlas, so it
also empties the band but eats real structure at the boundaries. That is why
`max_lab_plus_one` is the one that ships.

Usage
-----
    python scripts/diag_jlf_background_fallback.py
"""
import numpy as np
import ants

rng = np.random.default_rng(0)
N = 40
zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
c = N // 2


def sphere(cx, cy, cz, r):
    return ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= r * r


mask = sphere(c, c, c, 15).astype("uint32")
# Two structures, well inside. Everything else in the mask is unlabelled in
# every atlas -- that band is what the real data dumps into label 4.
struct_a = sphere(c - 5, c, c, 4)   # will be label 4  (lowest id)
struct_b = sphere(c + 5, c, c, 4)   # will be label 35

print(f"mask: {mask.sum()} vox | labelled in atlases: {(struct_a|struct_b).sum()} vox "
      f"| unlabelled inside mask: {mask.sum() - (struct_a|struct_b).sum()} vox")

target_np = np.where(struct_a, 2.0, np.where(struct_b, 3.0, 1.0)) * mask
target = ants.from_numpy(target_np.astype("float32"))
mask_img = ants.from_numpy(mask.astype("float32"))

atlases, labels = [], []
for i in range(5):
    jx = int(rng.integers(-1, 2))
    a = sphere(c - 5 + jx, c, c, 4)
    b = sphere(c + 5 + jx, c, c, 4)
    lab = np.zeros((N, N, N), dtype="float32")
    lab[a] = 4
    lab[b] = 35
    img = (np.where(a, 2.0, np.where(b, 3.0, 1.0)) * mask).astype("float32")
    img += rng.normal(0, 0.02, img.shape).astype("float32")
    atlases.append(ants.from_numpy(img))
    labels.append(ants.from_numpy(lab))


def run(**kw):
    # label_list is mutated in place by max_lab_plus_one, so hand over copies
    ll = [l.clone() for l in labels]
    out = ants.joint_label_fusion(
        target_image=target, target_image_mask=mask_img,
        atlas_list=atlases, label_list=ll,
        beta=4.0, rad=2, rho=0.01, usecor=False, r_search=3, verbose=False, **kw
    )
    seg = out["segmentation"].numpy().astype(int)
    unlabelled = (mask > 0) & ~(struct_a | struct_b)
    vals, cnts = np.unique(seg[unlabelled], return_counts=True)
    got = dict(zip(vals.tolist(), cnts.tolist()))
    return got, out["segmentation_numbers"], int((seg == 4).sum())


TRUE_A = int(struct_a.sum())
results = {}
for name, kw in [
    ("current  (both False)", dict(nonnegative=False, no_zeroes=False, max_lab_plus_one=False)),
    ("max_lab_plus_one=True", dict(nonnegative=False, no_zeroes=False, max_lab_plus_one=True)),
    ("no_zeroes=True",        dict(nonnegative=False, no_zeroes=True,  max_lab_plus_one=False)),
]:
    got, nums, n4 = run(**kw)
    results[name] = (got, n4)
    print(f"\n--- {name} ---")
    print(f"  labels returned: {list(nums)}")
    print(f"  what the unlabelled band became: {got}")
    print(f"  total voxels called label 4: {n4}  (true structure A = {TRUE_A})")

# Assertions, so this doubles as a regression gate rather than a report to read.
failures = []
band_default, n4_default = results["current  (both False)"]
if band_default.get(4, 0) == 0:
    failures.append("the unlabelled band no longer falls onto label 4 by default; "
                    "the premise behind max_lab_plus_one=True may no longer hold")
if n4_default <= TRUE_A * 2:
    failures.append(f"default label 4 = {n4_default}, expected it to dwarf the true {TRUE_A}")

band_fix, n4_fix = results["max_lab_plus_one=True"]
if band_fix.get(0, 0) != int(((mask > 0) & ~(struct_a | struct_b)).sum()):
    failures.append("max_lab_plus_one=True did not send the whole unlabelled band to 0")
if n4_fix != TRUE_A:
    failures.append(f"max_lab_plus_one=True gave label 4 = {n4_fix}, expected exactly {TRUE_A}")

band_nz, n4_nz = results["no_zeroes=True"]
if n4_nz >= TRUE_A:
    failures.append(f"no_zeroes=True gave label 4 = {n4_nz}; it was expected to erode "
                    f"below the true {TRUE_A}, which is why it is not the chosen fix")

print()
if failures:
    for f in failures:
        print(f"FAIL: {f}")
    raise SystemExit(1)
print("OK: max_lab_plus_one=True recovers the true extent; the default inflates it; "
      "no_zeroes erodes it.")
