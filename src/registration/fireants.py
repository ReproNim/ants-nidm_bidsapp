"""FireANTs registration backend.

The only module in this package that imports `fireants` or `torch`. Everything
else stays importable without them so CI can run the geometry, label, and
device tests unconditionally.

Design constraints enforced here:
  * Images cross to and from ANTsPy as NIfTI files, never as arrays with a
    hand-rebuilt header. Registration outputs are tensors, so their header is
    copied wholesale from the fixed image via SimpleITK CopyInformation --
    copied, not reconstructed from spacing/origin/direction parts.
  * fireants.io.Image is never given orientation / spacing / direction /
    origin / center. `orientation` applies sitk.DICOMOrient and rewrites the
    header; the rest override geometry that should come from the file.
  * Labels are remapped to a contiguous range before one-hotting, because
    integer_to_onehot allocates max_label + 1 channels and DKT31 ids reach 2035.
"""
import logging
from pathlib import Path

import numpy as np

from src.registration.handoff import (
    assert_geometry_match,
    geometry_of,
    read_nifti,
    write_nifti,
)
from src.registration.labels import build_label_map

# Hyperparameters transcribed from FireANTs itself, not invented, but from two
# different places rather than one:
#   - SYN_SCALES, SYN_ITERATIONS, deformation_type='compositive', and
#     optimizer='adam' are the fixed constants in FireANTs' own
#     scripts/pairwise/oasis.py. Everything else in that file's SyN call
#     (learning rate, cc kernel size, smoothing sigmas) is a ray-tune search
#     value there (config['lr'], 1 + 2*config['cc_size'], etc.), not a
#     constant, so it is not a usable source for those.
#   - AFFINE_SCALES, AFFINE_ITERATIONS, the affine optimizer_lr=3e-3, and
#     CC_KERNEL_SIZE=5 come from tutorials/tutorial1_basic_usage.py.
#   - SYN_LR = 0.5 is that same tutorial's GreedyRegistration optimizer_lr.
#   - SMOOTH_GRAD_SIGMA = 1.0 is that same tutorial's smooth_grad_sigma=1.
#   - SMOOTH_WARP_SIGMA = 0.5 is SyNRegistration's own documented default.
# SyN rather than Greedy, as the closer analogue to the SyNRA being replaced.
AFFINE_SCALES = [4, 2, 1]
AFFINE_ITERATIONS = [200, 100, 50]
SYN_SCALES = [4, 2, 1]
SYN_ITERATIONS = [200, 100, 50]
SYN_LR = 0.5
CC_KERNEL_SIZE = 5
SMOOTH_GRAD_SIGMA = 1.0
# 1.0, not SyNRegistration's documented default of 0.5: on Caltech sub-0051456
# (diagnostic 22191183, 2026-09-07) 0.5 folded up to 0.54% of voxels on some
# atlases while 1.0 folded none, at identical coverage/NCC/Dice.
SMOOTH_WARP_SIGMA = 1.0

# Deformable algorithm for the atlas registrations: "greedy" (FireANTs'
# primary algorithm, tuned against its own OASIS hyperparameters) or "syn"
# (symmetric; the spec's original choice as the closer analogue to SyNRA).
# Round 5/6 tuning (jobs 22320466, 22321551, 2026-09-08) on Caltech
# sub-0051456 found SyN plateaus at primary-label median Dice 0.81 across ten
# configurations (iterations, masking, metric, smoothing, schedule depth) --
# the optimiser had converged, so more effort could not close the gap. Greedy
# with FireANTs' own OASIS schedule reached 0.852 with zero folding at less
# than half the registration time (141s vs ~250s for 20 atlases). Ten further
# Greedy variants (kernel 5/9, lr 0.25/1.0, warp/grad sigma, 400 iterations,
# nearest labels) did not improve on the baseline Greedy config below -- it
# had also converged. Four primary labels (41, 40, 1021, 2019) and two
# relaxed labels (49, 50) still miss the per-label floor on that one subject.
# Same API on both classes
# (optimize / evaluate / get_warp_parameters / .affine), so the folding gate
# and label warps are unchanged either way. GREEDY_* mirror FireANTs'
# scripts/pairwise/oasis.py schedule and GreedyRegistration defaults.
DEFORMABLE = "greedy"
GREEDY_SCALES = [4, 2, 1]
GREEDY_ITERATIONS = [250, 200, 100]
GREEDY_LR = 0.5
GREEDY_CC_KERNEL_SIZE = 7

# Folding gate. A folding warp is provably bad independently of
# Dice; the ANTs path being replaced had no equivalent check.
MAX_FOLDING_FRACTION = 0.001

# Centre-of-mass initialisation before the affine. The ANTs path being replaced
# (SyNRA) starts every registration from antsRegistration's default
# initial-moving-transform, which aligns centres of mass; FireANTs starts from
# identity unless told otherwise. That matters here: every OASIS-TRT-20 atlas
# brain sits 50-70 mm from a typical subject brain in physical space and the
# OASIS-30 template ~200 mm, and with both sides skull-stripped the CC loss
# saturates on the shared zero background, so the affine never closes that gap
# (diagnostic 22188503, 2026-09-07: warped atlases covered 26-50% of the target
# brain, in-mask NCC 0.09-0.22, label Dice ~0). MomentsRegistration with
# moments=1 and transl_mode="com" is FireANTs' equivalent of ANTs' default;
# scale 4 matches FireANTs' own oasis_deformable.yaml.
MOMENTS_INIT = True
MOMENTS_SCALE = 4
# FireANTs' AffineRegistration default. tutorial1's 3e-3 was the original
# value; with the masked loss below, 3e-2 gave affine-stage coverage 0.89 (min
# 0.82) vs 0.88 (min 0.79) and Dice 0.40 vs 0.37 (diagnostic 22191183). With
# the UNMASKED loss 3e-2 is harmful (one atlas fell to 6% coverage), so the two
# settings go together.
AFFINE_LR = 3e-2
# Restrict the CC metric to brain: FireANTs' "masked_cc" expects the mask as a
# trailing image channel on BOTH sides and multiplies the two (maskedutils).
# Both sides here are skull-stripped, so ~85% of the fixed FOV and ~87% of an
# atlas FOV are zero on both sides; the local CC of two all-zero patches is
# perfect agreement, which is why the unmasked affine reported -0.94 while the
# brains barely overlapped (diagnostic 22188503). Masking took affine-stage
# coverage from 0.66 (min 0.29) to 0.88 (min 0.79) and majority-vote Dice
# against the delivered derivative from 0.21 to 0.37 (diagnostic 22191183).
LOSS_MASKED = True
# Similarity metric for affine and SyN. "cc" is what FireANTs' tutorial and its
# OASIS config use; the ANTs SyNRA path being replaced used mattes (MI) for
# both stages (main:src/antspy/wrapper.py aff_metric/syn_metric='mattes').
LOSS_TYPE = "cc"
# How the two mask channels combine (fireants.losses.maskedutils): "mult" scores
# only voxels that are brain on BOTH sides, so subject brain the atlas fails to
# reach contributes nothing and nothing pulls the atlas outward; "max" scores
# the union. LOSS_MASK_MOVING selects the moving-side channel: "brain" (atlas >
# 0) or "all" (ones), the latter making the fixed mask alone define the metric
# domain, as antsRegistration's -x fixed_mask does.
# "max" (union of the two brain masks), not "mult" (intersection): "mult"
# gives zero loss gradient over subject brain the atlas has not yet reached,
# which was the single biggest lever in the Sep 7 sweep (round 4, job
# 22311946) -- affine-stage coverage 0.66->0.88, though Dice alone plateaued
# at 0.81 until Greedy (above) closed the rest of the gap.
LOSS_MASK_MODE = "max"
LOSS_MASK_MOVING = "brain"


def _with_mask_channel(image, mask_array=None):
    """Return `image` with a binary mask as a trailing channel (masked_cc).

    `mask_array` is a tensor on the image's grid; defaults to image > 0, which
    is the brain for a skull-stripped atlas.
    """
    import copy

    import torch

    mask_img = copy.copy(image)
    arr = image.array
    if mask_array is not None:
        mask_img.array = mask_array.to(arr.dtype)
    elif LOSS_MASK_MOVING == "all":
        mask_img.array = torch.ones_like(arr)
    else:
        mask_img.array = (arr > 0).to(arr.dtype)
    mask_img.channels = 1
    return image.concatenate(mask_img, optimize_memory=False)


def _loss_kwargs():
    kw = {"loss_type": ("masked_" if LOSS_MASKED else "") + LOSS_TYPE}
    if LOSS_MASKED:
        # A fresh dict per call: AbstractRegistration mutates loss_params.
        kw["loss_params"] = {"mask_mode": LOSS_MASK_MODE}
    return kw


def _moments_init(fixed_batch, moving_batch):
    """Centre-of-mass alignment as an affine init tensor, or None if disabled."""
    if not MOMENTS_INIT:
        return None
    from fireants.registration.moments import MomentsRegistration

    moments = MomentsRegistration(
        scale=MOMENTS_SCALE,
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        moments=1,
        transl_mode="com",
        cc_kernel_size=CC_KERNEL_SIZE,
    )
    moments.optimize()
    return moments.get_affine_init().detach()


def labels_from_warp_result(tensor, label_map):
    """Turn a FireANTs label-warp result into original-id labels.

    The result has one of two shapes depending on whether fused ops are active:
      [1, 1, H, W, D]  argmax indices, when interpolation_mode == 'genericlabel'
      [1, C, H, W, D]  one-hot, when fireants/io/image.py:145-147 forced
                       is_onehot=True because FFO was unavailable

    Both are remapped indices, never original ids.
    """
    array = tensor.detach().cpu().numpy()
    channels = array.shape[1]
    if channels == 1:
        indices = np.squeeze(array, axis=(0, 1)).round().astype(np.int64)
    else:
        expected = label_map.max_index + 1
        if channels != expected:
            raise ValueError(
                f"one-hot warp result has {channels} channels but the label map "
                f"has {expected}; the label map and the loaded image disagree"
            )
        indices = np.argmax(array[0], axis=0).astype(np.int64)
    return label_map.restore_array(indices)


def _force_onehot(device, ffo_available):
    """Whether to ask FireANTs for a one-hot label volume *explicitly*.

    `fireants.io.Image.__init__` forces `is_onehot=True` only when
    `fi.use_ffo` is false, and `use_ffo = USE_FFO and FFO_AVAILABLE` is a
    module-level global decided at import -- it is not per-tensor. So in the
    production container, where fused ops *are* built, `--device cpu` leaves
    `is_onehot=False` and `interpolation_mode='genericlabel'`; then at call
    time `GridSampleDispatcher.augment_registry_backend` flips `_use_ffo` off
    because the tensors are not on cuda, and `__call__` raises
    `RuntimeError("genericlabel interpolation requires fused ops")`. The
    combination the spec promises works would crash in the shipped image.

    Asking for one-hot whenever the fused generic-label sampler cannot be used
    -- no fused ops at all, or fused ops but not on cuda -- takes the degraded
    path deliberately instead.
    """
    return not ffo_available or device != "cuda"


def _seg_preprocessor(label_map):
    """Remap sparse ids to contiguous indices, on the torch tensor FireANTs
    hands us (fireants/io/image.py:151, applied before max_seg_label clamping).
    """
    import torch

    def preprocess(array):
        out = torch.zeros_like(array)
        for original, index in label_map.forward.items():
            out[array == original] = float(index)
        return out

    return preprocess


def _tensor_to_nifti(tensor, reference_itk, path):
    """Write a registration output tensor using the fixed image's header.

    CopyInformation copies the header object wholesale; it does not rebuild one
    from spacing/origin/direction values. The result is in fixed space by
    construction, so the fixed image's header is the correct one.
    """
    import SimpleITK as sitk

    array = np.asarray(tensor)
    while array.ndim > 3:
        if array.shape[0] != 1:
            raise ValueError(
                f"cannot drop leading axis of size {array.shape[0]} (expected "
                f"1) from array with shape {array.shape}; only singleton "
                "leading dims may be squeezed away"
            )
        array = array[0]
    out = sitk.GetImageFromArray(array)
    out.CopyInformation(reference_itk)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(path))
    return path


def _assert_inbound(expected, image, context):
    """Verify a `fireants.io.Image` loaded from an ANTsPy-written file kept its
    header.

    This is the geometry check that can actually fail. The *outbound* check --
    `read_nifti(..., expect=target_geom)` on a volume `_tensor_to_nifti` just
    stamped with `CopyInformation(reference_itk)` from that very same fixed
    image -- compares a header with itself; only shape can differ there, and
    only for non-cubic volumes. What is genuinely unverified is whether
    SimpleITK reproduced what ANTsPy wrote.
    """
    assert_geometry_match(expected, geometry_of(image.itk_image), context=context)


def _chunks(items, size):
    if not size or size <= 0 or size >= len(items):
        return [list(items)]
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def warp_atlases(
    target,
    target_mask,
    atlas_paths,
    label_paths,
    *,
    temp_dir,
    device="cuda",
    batch_size=0,
    label_interpolation="genericlabel",
    logger=None,
):
    """Register every atlas to `target` and return warped atlases and labels.

    `target_mask` restricts the similarity loss on the fixed side: with
    LOSS_MASKED it becomes the trailing mask channel of the fixed image, and
    each atlas gets its own nonzero mask. Both sides are skull-stripped, and an
    unmasked CC scores the shared zero background as perfect agreement (see
    LOSS_MASKED above).

    Returns (warped_atlas_images, warped_label_images) as ANTsImages.
    """
    import torch
    from fireants.interpolator import fireants_interpolator
    from fireants.io import BatchedImages, Image
    from fireants.registration.affine import AffineRegistration
    from fireants.registration.syn import SyNRegistration

    log = logger or logging.getLogger("ants-nidm.registration")
    atlas_paths = [Path(p) for p in atlas_paths]
    label_paths = [Path(p) for p in label_paths]
    if len(atlas_paths) != len(label_paths):
        raise ValueError(
            f"{len(atlas_paths)} atlases but {len(label_paths)} label volumes"
        )

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    target_geom = geometry_of(target)

    label_map = build_label_map(label_paths)
    log.info(
        "Label map: %d distinct ids -> 0..%d (one-hot would otherwise need %d "
        "channels)",
        len(label_map.forward),
        label_map.max_index,
        int(label_map.inverse.max()) + 1,
    )

    from fireants.interpolator import FFO_AVAILABLE

    log.info(
        "FireANTs device=%s fused_ops=%s label_interpolation=%s",
        device,
        FFO_AVAILABLE,
        label_interpolation,
    )
    onehot = _force_onehot(device, FFO_AVAILABLE)
    if label_interpolation == "genericlabel" and onehot:
        log.warning(
            "genericlabel requested but the fused generic-label sampler is "
            "unusable here (fused_ops=%s, device=%s): it is a CUDA kernel and "
            "is skipped for tensors that are not on cuda. Requesting "
            "is_onehot=True explicitly instead, costing %d channels. FireANTs "
            "only forces one-hot by itself when fused ops are missing "
            "entirely -- with fused ops built but device=cpu it would keep "
            "interpolation_mode='genericlabel' and raise at call time.",
            FFO_AVAILABLE,
            device,
            label_map.max_index + 1,
        )

    target_path = write_nifti(target, temp_dir / "target.nii.gz")
    fixed = Image.load_file(str(target_path), device=device)
    # Inbound guard -- the direction that can actually differ. Outbound tensors
    # get the fixed image's header stamped onto them wholesale by
    # _tensor_to_nifti's CopyInformation, so checking those against the same
    # fixed image compares a header with itself. What is genuinely unverified
    # is what SimpleITK made of the file ANTsPy just wrote.
    _assert_inbound(target_geom, fixed, "target crossing into FireANTs")
    reference_itk = fixed.itk_image
    # Moments (centre of mass) is computed on the plain image; the masked loss,
    # if enabled, sees image + brain-mask channels.
    moments_fixed_batch = BatchedImages([fixed])
    if LOSS_MASKED:
        mask_path = write_nifti(target_mask, temp_dir / "target_mask.nii.gz")
        fixed_mask = Image.load_file(str(mask_path), device=device)
        fixed_batch = BatchedImages([_with_mask_channel(fixed, fixed_mask.array > 0.5)])
    else:
        fixed_batch = moments_fixed_batch

    warped_atlases, warped_labels = [], []
    index = 0
    for chunk in _chunks(list(zip(atlas_paths, label_paths)), batch_size):
        moving_images = [Image.load_file(str(a), device=device) for a, _ in chunk]
        moments_moving_batch = BatchedImages(moving_images)
        moving_batch = (
            BatchedImages([_with_mask_channel(m) for m in moving_images])
            if LOSS_MASKED else moments_moving_batch
        )
        # One fixed image against len(chunk) moving ones. FireANTs' registration
        # accepts a fixed batch of 1 as "broadcastable" at construction, but
        # its losses compare fixed and moving shapes strictly, and
        # BatchedImages.__call__ only expands the batch to N after broadcast(N)
        # -- without it tier 2 died with "ground truth has differing shape
        # ([1,1,...]) from pred ([20,1,...])". This is what FireANTs' own
        # template builder does before each batch. Re-issued per chunk because
        # a trailing partial chunk is smaller than the ones before it.
        fixed_batch.broadcast(len(chunk))
        moments_fixed_batch.broadcast(len(chunk))
        log.info("Registering %d atlases (affine)", len(chunk))
        affine = AffineRegistration(
            AFFINE_SCALES,
            AFFINE_ITERATIONS,
            fixed_batch,
            moving_batch,
            optimizer="Adam",
            optimizer_lr=AFFINE_LR,
            cc_kernel_size=CC_KERNEL_SIZE,
            init_rigid=_moments_init(moments_fixed_batch, moments_moving_batch),
            **_loss_kwargs(),
        )
        affine.optimize()

        if DEFORMABLE == "greedy":
            from fireants.registration.greedy import GreedyRegistration

            log.info("Registering %d atlases (Greedy)", len(chunk))
            syn = GreedyRegistration(
                GREEDY_SCALES,
                GREEDY_ITERATIONS,
                fixed_batch,
                moving_batch,
                deformation_type="compositive",
                optimizer="Adam",
                optimizer_lr=GREEDY_LR,
                cc_kernel_size=GREEDY_CC_KERNEL_SIZE,
                smooth_grad_sigma=SMOOTH_GRAD_SIGMA,
                smooth_warp_sigma=SMOOTH_WARP_SIGMA,
                init_affine=affine.get_affine_matrix().detach(),
                **_loss_kwargs(),
            )
        elif DEFORMABLE == "syn":
            log.info("Registering %d atlases (SyN)", len(chunk))
            syn = SyNRegistration(
                SYN_SCALES,
                SYN_ITERATIONS,
                fixed_batch,
                moving_batch,
                deformation_type="compositive",
                optimizer="Adam",
                optimizer_lr=SYN_LR,
                cc_kernel_size=CC_KERNEL_SIZE,
                smooth_grad_sigma=SMOOTH_GRAD_SIGMA,
                smooth_warp_sigma=SMOOTH_WARP_SIGMA,
                init_affine=affine.get_affine_matrix().detach(),
                **_loss_kwargs(),
            )
        else:
            raise ValueError(f"DEFORMABLE must be 'syn' or 'greedy', got {DEFORMABLE!r}")
        syn.optimize()

        # [N, C, ...]; with the masked loss C == 2 and channel 1 is the mask.
        moved = syn.evaluate(fixed_batch, moving_batch)[:, :1]

        # Checked per atlas, not as a batch mean: one folding atlas among
        # twenty clean ones is invisible in the average; the gate applies to every
        # atlas registration.
        foldings = folding_fraction(syn, fixed_batch, moving_batch)
        for k, folding in enumerate(foldings):
            _refuse_folding(folding, f"atlas {index + k} ({chunk[k][0]})", log)

        if label_interpolation == "genericlabel":
            label_batch = BatchedImages(
                [
                    Image.load_file(
                        str(lp),
                        device=device,
                        is_segmentation=True,
                        background_seg_label=-1,
                        max_seg_label=label_map.max_index,
                        seg_preprocessor=_seg_preprocessor(label_map),
                        is_onehot=onehot,
                    )
                    for _, lp in chunk
                ]
            )
            moved_labels = syn.evaluate(fixed_batch, label_batch)
        elif label_interpolation == "nearest":
            label_batch = BatchedImages(
                [Image.load_file(str(lp), device=device) for _, lp in chunk]
            )
            coords = syn.get_warp_parameters(fixed_batch, label_batch)
            moved_labels = fireants_interpolator(
                label_batch(), **coords, mode="nearest", align_corners=True
            )
            del coords
        else:
            raise ValueError(
                f"unknown label_interpolation {label_interpolation!r}; "
                "expected 'genericlabel' or 'nearest'"
            )

        for k in range(len(chunk)):
            atlas_out = _tensor_to_nifti(
                moved[k].detach().cpu().numpy(),
                reference_itk,
                temp_dir / f"warped_atlas_{index:02d}.nii.gz",
            )
            if label_interpolation == "genericlabel":
                ids = labels_from_warp_result(moved_labels[k : k + 1], label_map)
            else:
                ids = (
                    moved_labels[k].detach().cpu().numpy().squeeze().round().astype(np.int64)
                )
            label_out = _tensor_to_nifti(
                ids.astype(np.float32),
                reference_itk,
                temp_dir / f"warped_labels_{index:02d}.nii.gz",
            )

            warped_atlases.append(
                read_nifti(atlas_out, expect=target_geom, context=f"warped atlas {index}")
            )
            warped_labels.append(
                read_nifti(label_out, expect=target_geom, context=f"warped labels {index}")
            )
            index += 1

        # Everything that holds this chunk's device tensors, so empty_cache can
        # release it before the next chunk loads. label_batch is the big one:
        # on the one-hot path it is ~105 channels per atlas.
        del affine, syn, moved, moved_labels, moving_batch, label_batch
        del moments_moving_batch, moving_images
        if device == "cuda":
            torch.cuda.empty_cache()

    log.info("Warped %d atlases", len(warped_atlases))
    return warped_atlases, warped_labels


def _refuse_folding(folding, what, log):
    """Log a warp's folding fraction and raise above MAX_FOLDING_FRACTION."""
    log.info("Non-positive Jacobian determinants, %s: %.4f%%", what, folding * 100.0)
    if folding > MAX_FOLDING_FRACTION:
        raise RuntimeError(
            f"registration folded for {what}: {folding * 100:.3f}% of voxels "
            f"have a non-positive Jacobian determinant (limit "
            f"{MAX_FOLDING_FRACTION * 100:.3f}%). The warp is not "
            f"diffeomorphic; the segmentation would be wrong in a way "
            f"Dice against a single reference may not reveal."
        )


def folding_fractions_from_det(det):
    """Per-batch-element fraction of non-positive Jacobian determinants.

    Reduces over the spatial axes only. Flattening the whole [N, ...] tensor
    and taking one mean dilutes a folding atlas by the N-1 clean ones: at the
    default batch_size=0 (N=20) an atlas folding on 2% of its voxels shows up
    as a 0.1% batch mean and slips under the gate. The gate is for every atlas
    registration, not for their average.
    """
    n = det.shape[0]
    bad = (det <= 0).reshape(n, -1)
    return [float(v) for v in bad.to(det.dtype).mean(dim=1)]


def folding_fraction(registration, fixed_batch, moving_batch):
    """Fraction of voxels where the warp's Jacobian determinant is <= 0, as one
    value per batch element in batch order.

    A folding warp is provably bad independently of Dice, and the ANTs path
    being replaced had no such check. MAX_FOLDING_FRACTION gates this at
    0.1% per registration. Mirrors the diffeomorphism check in FireANTs' own
    tutorials/tutorial1_basic_usage.py.
    """
    import torch
    from fireants.utils.imageutils import jacobian

    warp = registration.get_warped_coordinates(fixed_batch, moving_batch)
    jac = jacobian(warp).permute(0, 2, 3, 4, 1, 5)[:, 1:-1, 1:-1, 1:-1, :]
    det = torch.linalg.det(jac)
    # `warp` maps fixed torch coordinates to moving torch coordinates, so its
    # linear part is moving_phy2torch @ affine @ fixed_torch2phy (syn.py's
    # affine_map_init). When the two headers differ by a reflection -- an odd
    # axis permutation or a flip, e.g. an RAS subject against the ASL-stored
    # OASIS atlases -- that determinant is negative everywhere for a perfect
    # rigid transform, and a raw `det <= 0` test reports 100% folding. Folding
    # is a sign *change* relative to that global orientation, so normalise by
    # it. (LAS subjects happened to reach ASL by an even permutation, which is
    # why Caltech never tripped this.)
    # Linear parts only: translations do not enter the Jacobian, and
    # AffineRegistration stores a [N, D, D+1] matrix where SyN stores [N, D+1, D+1].
    d = det.ndim - 1
    lin = torch.matmul(
        moving_batch.get_phy2torch()[:, :d, :d].to(det.dtype),
        torch.matmul(registration.affine[:, :d, :d].to(det.dtype),
                     fixed_batch.get_torch2phy()[:, :d, :d].to(det.dtype)),
    )
    orientation_sign = torch.sign(torch.linalg.det(lin))
    if orientation_sign.shape[0] != det.shape[0]:
        orientation_sign = orientation_sign.expand(det.shape[0])
    return folding_fractions_from_det(det * orientation_sign.view(-1, 1, 1, 1))


class TemplateWarp:
    """A subject<->template registration, exposed in one direction only.

    Consumers pull template-space images into subject space -- the brain
    probability mask, and the DKT31 label atlas under --method quick. That is
    the only direction exposed. It replaces the previous
    `TemplateToSubjectTransforms`, a list of ANTs transform file paths whose
    orientation depended on whether the caller read fwdtransforms or
    invtransforms; reading the wrong one produced an 11.5 L brain volume
    (wrapper.py:839-851).

    Held as torch tensors. Never serialised to ANTs transform format, which is
    why the composition-order and inversion round trip through disk does not
    exist in this design.
    """

    def __init__(self, registration, subject_batch, template_batch, reference_itk, device):
        self._registration = registration
        self._subject_batch = subject_batch
        self._template_batch = template_batch
        self.reference_itk = reference_itk
        self.device = device

    def warp_parameters(self):
        """Coordinates that map template-space samples into subject space."""
        return self._registration.get_warp_parameters(
            self._subject_batch, self._template_batch
        )


def register_to_template(subject, template, *, temp_dir, device="cuda", logger=None):
    """Register `subject` to `template`, returning a template->subject warp.

    `subject` is the skull-stripped brain; `template` is
    T_template0_BrainCerebellum. Registering onto the whole-head T_template0
    biased the affine toward scaling the brain up to the head outline
    (wrapper.py:804-812) -- pass the brain-only template.
    """
    from fireants.io import BatchedImages, Image
    from fireants.registration.affine import AffineRegistration
    from fireants.registration.syn import SyNRegistration

    log = logger or logging.getLogger("ants-nidm.registration")
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    subject_geom = geometry_of(subject)
    template_geom = geometry_of(template)

    subject_path = write_nifti(subject, temp_dir / "tpl_subject.nii.gz")
    template_path = write_nifti(template, temp_dir / "tpl_template.nii.gz")

    # fixed = subject, moving = template. The registration is set up in the
    # direction we need to *apply*, so no inversion is ever required.
    fixed = Image.load_file(str(subject_path), device=device)
    moving = Image.load_file(str(template_path), device=device)
    # Inbound guards: verify FireANTs' SimpleITK read reproduced the header
    # ANTsPy wrote. This is the crossing whose geometry is not stamped on by
    # construction, so it is the one worth asserting.
    _assert_inbound(subject_geom, fixed, "subject crossing into FireANTs")
    _assert_inbound(template_geom, moving, "template crossing into FireANTs")
    if LOSS_MASKED:
        fixed_batch = BatchedImages([_with_mask_channel(fixed)])
        moving_batch = BatchedImages([_with_mask_channel(moving)])
    else:
        fixed_batch, moving_batch = BatchedImages([fixed]), BatchedImages([moving])

    # Both sides are skull-stripped brain-only images (the docstring above:
    # `template` is T_template0_BrainCerebellum, not the whole head), so this
    # is exactly the shared-zero-background scenario that made warp_atlases's
    # unmasked affine saturate on 2026-09-07 (diagnostic 22188503) -- and this
    # function had never received that fix. Confirmed on real data
    # 2026-09-17: the quick method's dseg (which calls this function, via
    # apply_warp) placed structures 45-50 voxels from where fusion and the
    # delivered reference both agree they sit, while quick's own
    # preprocessing (shared with fusion) matched the reference closely -- so
    # the defect was specifically here, not upstream. masked_cc + Greedy are
    # the same two levers that fixed warp_atlases.
    log.info("Template registration (affine)")
    affine = AffineRegistration(
        AFFINE_SCALES,
        AFFINE_ITERATIONS,
        fixed_batch,
        moving_batch,
        optimizer="Adam",
        optimizer_lr=AFFINE_LR,
        cc_kernel_size=CC_KERNEL_SIZE,
        init_rigid=_moments_init(fixed_batch, moving_batch),
        **_loss_kwargs(),
    )
    affine.optimize()

    if DEFORMABLE == "greedy":
        from fireants.registration.greedy import GreedyRegistration

        log.info("Template registration (Greedy)")
        syn = GreedyRegistration(
            GREEDY_SCALES,
            GREEDY_ITERATIONS,
            fixed_batch,
            moving_batch,
            deformation_type="compositive",
            optimizer="Adam",
            optimizer_lr=GREEDY_LR,
            cc_kernel_size=GREEDY_CC_KERNEL_SIZE,
            smooth_grad_sigma=SMOOTH_GRAD_SIGMA,
            smooth_warp_sigma=SMOOTH_WARP_SIGMA,
            init_affine=affine.get_affine_matrix().detach(),
            **_loss_kwargs(),
        )
    elif DEFORMABLE == "syn":
        log.info("Template registration (SyN)")
        syn = SyNRegistration(
            SYN_SCALES,
            SYN_ITERATIONS,
            fixed_batch,
            moving_batch,
            deformation_type="compositive",
            optimizer="Adam",
            optimizer_lr=SYN_LR,
            cc_kernel_size=CC_KERNEL_SIZE,
            smooth_grad_sigma=SMOOTH_GRAD_SIGMA,
            smooth_warp_sigma=SMOOTH_WARP_SIGMA,
            init_affine=affine.get_affine_matrix().detach(),
            **_loss_kwargs(),
        )
    else:
        raise ValueError(f"DEFORMABLE must be 'syn' or 'greedy', got {DEFORMABLE!r}")
    syn.optimize()
    # --method quick pulls the template labels through this warp, so the
    # atlas gate applies here too.
    _refuse_folding(folding_fraction(syn, fixed_batch, moving_batch)[0], "the template", log)

    # TemplateWarp only ever calls the deformable stage's own registration
    # methods (get_warp_parameters etc.) on fixed_batch/moving_batch, which
    # -- when LOSS_MASKED -- carry a trailing mask channel; that channel's
    # own probability content isn't itself warped anywhere, so this is safe.
    return TemplateWarp(syn, fixed_batch, moving_batch, fixed.itk_image, device)


def apply_warp(
    warp,
    moving,
    reference,
    *,
    temp_dir,
    interpolation="genericlabel",
    label_map=None,
    logger=None,
):
    """Pull a template-space image into subject space.

    `interpolation='nearest'` warps raw intensities; 'genericlabel' treats the
    input as a label volume and requires `label_map`.
    """
    from fireants.interpolator import FFO_AVAILABLE, fireants_interpolator
    from fireants.io import BatchedImages, Image

    log = logger or logging.getLogger("ants-nidm.registration")
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    reference_geom = geometry_of(reference)
    moving_geom = geometry_of(moving)

    moving_path = write_nifti(moving, temp_dir / "apply_moving.nii.gz")
    coords = warp.warp_parameters()

    if interpolation == "genericlabel":
        if label_map is None:
            raise ValueError("interpolation='genericlabel' requires label_map")
        onehot = _force_onehot(warp.device, FFO_AVAILABLE)
        if onehot:
            log.warning(
                "genericlabel requested but the fused generic-label sampler is "
                "unusable here (fused_ops=%s, device=%s); requesting "
                "is_onehot=True explicitly, costing %d channels",
                FFO_AVAILABLE,
                warp.device,
                label_map.max_index + 1,
            )
        img = Image.load_file(
            str(moving_path),
            device=warp.device,
            is_segmentation=True,
            background_seg_label=-1,
            max_seg_label=label_map.max_index,
            seg_preprocessor=_seg_preprocessor(label_map),
            is_onehot=onehot,
        )
        _assert_inbound(moving_geom, img, "moving crossing into FireANTs")
        result = fireants_interpolator(
            BatchedImages([img])(),
            **coords,
            mode=img.interpolation_mode or "bilinear",
            align_corners=True,
        )
        array = labels_from_warp_result(result, label_map).astype(np.float32)
    elif interpolation == "nearest":
        img = Image.load_file(str(moving_path), device=warp.device)
        _assert_inbound(moving_geom, img, "moving crossing into FireANTs")
        result = fireants_interpolator(
            BatchedImages([img])(), **coords, mode="nearest", align_corners=True
        )
        array = result.detach().cpu().numpy()
    else:
        raise ValueError(
            f"unknown interpolation {interpolation!r}; "
            "expected 'genericlabel' or 'nearest'"
        )

    out_path = _tensor_to_nifti(array, warp.reference_itk, temp_dir / "applied.nii.gz")
    log.info("Applied template->subject warp (%s)", interpolation)
    return read_nifti(out_path, expect=reference_geom, context="apply_warp result")
