"""FireANTs registration backend.

The only module in this package that imports `fireants` or `torch`. Everything
else stays importable without them so CI can run the geometry, label, and
device tests unconditionally.

Design constraints enforced here (spec section 4):
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
SMOOTH_WARP_SIGMA = 0.5

# Spec section 8 criterion 7. A folding warp is provably bad independently of
# Dice; the ANTs path being replaced had no equivalent check.
MAX_FOLDING_FRACTION = 0.001


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

    `target_mask` is accepted because ants.joint_label_fusion needs it
    downstream; it is deliberately not wired into the similarity loss. Both
    sides of the metric are already skull-stripped (`masked_brain` is
    N4 * BrainExtractionMask, and the OASIS-TRT-20 atlases are brain-only), so a
    mask would change nothing while adding a difference from the reference.

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
    if label_interpolation == "genericlabel" and (
        not FFO_AVAILABLE or device != "cuda"
    ):
        log.warning(
            "genericlabel requested but fused ops are unavailable (FFO=%s, "
            "device=%s); FireANTs will fall back to one-hot, costing %d channels",
            FFO_AVAILABLE,
            device,
            label_map.max_index + 1,
        )

    target_path = write_nifti(target, temp_dir / "target.nii.gz")
    fixed = Image.load_file(str(target_path), device=device)
    fixed_batch = BatchedImages([fixed])
    reference_itk = fixed.itk_image

    warped_atlases, warped_labels = [], []
    index = 0
    for chunk in _chunks(list(zip(atlas_paths, label_paths)), batch_size):
        moving_batch = BatchedImages(
            [Image.load_file(str(a), device=device) for a, _ in chunk]
        )
        log.info("Registering %d atlases (affine)", len(chunk))
        affine = AffineRegistration(
            AFFINE_SCALES,
            AFFINE_ITERATIONS,
            fixed_batch,
            moving_batch,
            optimizer="Adam",
            optimizer_lr=3e-3,
            cc_kernel_size=CC_KERNEL_SIZE,
        )
        affine.optimize()

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
        )
        syn.optimize()

        moved = syn.evaluate(fixed_batch, moving_batch)

        folding = folding_fraction(syn, fixed_batch, moving_batch)
        log.info("Non-positive Jacobian determinants: %.4f%%", folding * 100.0)
        if folding > MAX_FOLDING_FRACTION:
            raise RuntimeError(
                f"registration folded: {folding * 100:.3f}% of voxels have a "
                f"non-positive Jacobian determinant (limit "
                f"{MAX_FOLDING_FRACTION * 100:.3f}%). The warp is not "
                f"diffeomorphic; the segmentation would be wrong in a way Dice "
                f"against a single reference may not reveal."
            )

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

        del affine, syn, moved, moved_labels, moving_batch
        if device == "cuda":
            torch.cuda.empty_cache()

    log.info("Warped %d atlases", len(warped_atlases))
    return warped_atlases, warped_labels


def folding_fraction(registration, fixed_batch, moving_batch):
    """Fraction of voxels where the warp's Jacobian determinant is <= 0.

    A folding warp is provably bad independently of Dice, and the ANTs path
    being replaced had no such check. Spec section 8 criterion 7 gates this at
    0.1%. Mirrors the diffeomorphism check in FireANTs' own
    tutorials/tutorial1_basic_usage.py.
    """
    import torch
    from fireants.utils.imageutils import jacobian

    warp = registration.get_warped_coordinates(fixed_batch, moving_batch)
    jac = jacobian(warp).permute(0, 2, 3, 4, 1, 5)[:, 1:-1, 1:-1, 1:-1, :]
    det = torch.linalg.det(jac).reshape(-1)
    return float((det <= 0).float().mean().item())


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

    subject_path = write_nifti(subject, temp_dir / "tpl_subject.nii.gz")
    template_path = write_nifti(template, temp_dir / "tpl_template.nii.gz")

    # fixed = subject, moving = template. The registration is set up in the
    # direction we need to *apply*, so no inversion is ever required.
    fixed = Image.load_file(str(subject_path), device=device)
    moving = Image.load_file(str(template_path), device=device)
    fixed_batch, moving_batch = BatchedImages([fixed]), BatchedImages([moving])

    log.info("Template registration (affine)")
    affine = AffineRegistration(
        AFFINE_SCALES,
        AFFINE_ITERATIONS,
        fixed_batch,
        moving_batch,
        optimizer="Adam",
        optimizer_lr=3e-3,
        cc_kernel_size=CC_KERNEL_SIZE,
    )
    affine.optimize()

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
    )
    syn.optimize()

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
    from fireants.interpolator import fireants_interpolator
    from fireants.io import BatchedImages, Image

    log = logger or logging.getLogger("ants-nidm.registration")
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    reference_geom = geometry_of(reference)

    moving_path = write_nifti(moving, temp_dir / "apply_moving.nii.gz")
    coords = warp.warp_parameters()

    if interpolation == "genericlabel":
        if label_map is None:
            raise ValueError("interpolation='genericlabel' requires label_map")
        img = Image.load_file(
            str(moving_path),
            device=warp.device,
            is_segmentation=True,
            background_seg_label=-1,
            max_seg_label=label_map.max_index,
            seg_preprocessor=_seg_preprocessor(label_map),
        )
        result = fireants_interpolator(
            BatchedImages([img])(),
            **coords,
            mode=img.interpolation_mode or "bilinear",
            align_corners=True,
        )
        array = labels_from_warp_result(result, label_map).astype(np.float32)
    elif interpolation == "nearest":
        img = Image.load_file(str(moving_path), device=warp.device)
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
