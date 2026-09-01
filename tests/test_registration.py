"""FireANTs backend tests on tiny CPU volumes.

Registration quality is not under test here -- that is tier 2 and tier 3 on a
GPU. What is under test is the wiring most likely to be silently wrong: label
shape handling, remapping round trips, and the geometry of what comes back.
"""
import numpy as np
import pytest

fireants = pytest.importorskip("fireants")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.requires_fireants

from src.registration.fireants import labels_from_warp_result  # noqa: E402
from src.registration.labels import LabelMap  # noqa: E402


@pytest.fixture(autouse=True)
def _use_real_ants(restore_real_ants):
    """Every test in this file that does real ants I/O needs the module-level
    sys.modules['ants'] mock leaked by tests/test_run.py / tests/test_wrapper.py
    undone -- see tests/conftest.py's restore_real_ants for why that fixture is
    opt-in rather than global. Applying it file-wide here (rather than only on
    the two tests that call ants.image_read/image_write) keeps every test in
    this module consistent regardless of which future test starts doing ants
    I/O."""


def _label_map(ids):
    ordered = sorted(set(ids) | {0})
    return LabelMap(
        forward={v: i for i, v in enumerate(ordered)},
        inverse=np.array(ordered, dtype=np.int32),
    )


def test_argmax_shaped_result_is_restored_to_original_ids():
    """With FFO active the warp returns [1, 1, H, W, D] remapped indices."""
    lm = _label_map([0, 4, 2035])
    tensor = torch.tensor([[[[[0, 1, 2]]]]], dtype=torch.float32)
    out = labels_from_warp_result(tensor, lm)
    np.testing.assert_array_equal(out, np.array([[[0, 4, 2035]]]))


def test_onehot_shaped_result_is_argmaxed_then_restored():
    """Without FFO, fireants/io/image.py:145-147 forces is_onehot=True and the
    warp returns [1, C, H, W, D]. Handing that to joint_label_fusion unchanged
    would be a well-formed disaster."""
    lm = _label_map([0, 4, 2035])
    onehot = torch.zeros((1, 3, 1, 1, 3), dtype=torch.float32)
    onehot[0, 0, 0, 0, 0] = 1.0   # channel 0 -> id 0
    onehot[0, 2, 0, 0, 1] = 1.0   # channel 2 -> id 2035
    onehot[0, 1, 0, 0, 2] = 1.0   # channel 1 -> id 4
    out = labels_from_warp_result(onehot, lm)
    np.testing.assert_array_equal(out, np.array([[[0, 2035, 4]]]))


def test_result_contains_only_ids_present_in_the_map():
    lm = _label_map([0, 4, 11, 2035])
    onehot = torch.rand((1, 4, 2, 2, 2))
    out = labels_from_warp_result(onehot, lm)
    assert set(np.unique(out)).issubset({0, 4, 11, 2035})


def test_folding_fraction_is_zero_for_an_identity_warp():
    """A registration of an image to itself must not fold."""
    import nibabel as nib

    from fireants.io import BatchedImages, Image
    from fireants.registration.affine import AffineRegistration

    from src.registration.fireants import folding_fraction

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/blob.nii.gz"
        blob = np.zeros((16, 16, 16), dtype=np.float32)
        blob[6:10, 6:10, 6:10] = 1.0
        nib.save(nib.Nifti1Image(blob, np.eye(4)), path)
        batch = BatchedImages([Image.load_file(path, device="cpu")])
        reg = AffineRegistration([2, 1], [10, 5], batch, batch,
                                 optimizer="Adam", optimizer_lr=1e-3,
                                 cc_kernel_size=3)
        reg.optimize()
        assert folding_fraction(reg, batch, batch) < 0.001


def test_unexpected_channel_count_is_rejected():
    """A one-hot whose channel count does not match the map means the label
    map and the loaded image disagree -- fail loudly rather than argmax into
    the wrong ids."""
    lm = _label_map([0, 4, 2035])
    with pytest.raises(ValueError, match="channel"):
        labels_from_warp_result(torch.zeros((1, 7, 1, 1, 1)), lm)


def test_template_warp_moves_template_space_into_subject_space(tmp_path):
    """Behavioural direction guard, replacing the structural invtransforms
    assertion in tests/test_wrapper.py.

    A marker placed in template space must land at the predicted subject-space
    voxel. If the direction is inverted it lands on the opposite side and this
    fails. This is the regression guard for the 11.5 L whole-FOV bug.
    """
    import nibabel as nib

    from src.registration.fireants import apply_warp, register_to_template

    ants = pytest.importorskip("ants")

    # Deviation from the brief's literal (24, 24, 24): FireANTs 1.5.0's
    # SyNRegistration.get_warp_parameters() unconditionally inverts the
    # reverse warp via fireants.utils.warputils.compositive_warp_inverse,
    # which clamps its internal warp field to fireants.utils.globals.
    # MIN_IMG_SIZE (32) regardless of the true image size. Below 32 voxels on
    # any axis, that leaves the clamped field (32) larger than the true image
    # size, and InverseConsistencyOperator.forward's own scale check --
    # image_size / disp_size, required to be >= 1 -- fails every time
    # (reproduced deterministically at 16 and 24; confirmed passing at exactly
    # 32). This is a test-scale-only artifact: production T1w volumes are two
    # orders of magnitude larger than 32 voxels per axis, so it never binds in
    # real use. 32 is the smallest size this test can use while still
    # exercising the real register_to_template -> apply_warp path.
    shape = (32, 32, 32)
    shift = 4  # voxels along x

    subject = np.zeros(shape, dtype=np.float32)
    subject[8 + shift : 16 + shift, 8:16, 8:16] = 1.0
    template = np.zeros(shape, dtype=np.float32)
    template[8:16, 8:16, 8:16] = 1.0

    sub_path = tmp_path / "subject.nii.gz"
    tpl_path = tmp_path / "template.nii.gz"
    nib.save(nib.Nifti1Image(subject, np.eye(4)), str(sub_path))
    nib.save(nib.Nifti1Image(template, np.eye(4)), str(tpl_path))

    sub_img = ants.image_read(str(sub_path))
    tpl_img = ants.image_read(str(tpl_path))

    warp = register_to_template(
        sub_img, tpl_img, temp_dir=tmp_path / "work", device="cpu"
    )

    marker = np.zeros(shape, dtype=np.float32)
    marker[8:16, 8:16, 8:16] = 1.0
    marker_path = tmp_path / "marker.nii.gz"
    nib.save(nib.Nifti1Image(marker, np.eye(4)), str(marker_path))

    moved = apply_warp(
        warp,
        ants.image_read(str(marker_path)),
        sub_img,
        temp_dir=tmp_path / "work",
        interpolation="nearest",
    )
    out = moved.numpy()

    centre = float(np.mean(np.nonzero(out > 0.5)[0]))
    subject_centre = float(np.mean(np.nonzero(subject > 0.5)[0]))
    template_centre = float(np.mean(np.nonzero(template > 0.5)[0]))
    assert abs(centre - subject_centre) < abs(centre - template_centre), (
        f"template marker landed at x={centre:.1f}; subject box is at "
        f"x={subject_centre:.1f} and template box at x={template_centre:.1f}. "
        "Landing nearer the template centre means the warp direction is inverted "
        "-- this is the 11.5 L brain-volume bug."
    )


def test_apply_warp_output_shares_the_reference_geometry(tmp_path):
    import nibabel as nib

    from src.registration.fireants import apply_warp, register_to_template
    from src.registration.handoff import assert_geometry_match, geometry_of

    ants = pytest.importorskip("ants")

    # See the shape comment in test_template_warp_moves_template_space_into_
    # subject_space: FireANTs 1.5.0 requires every axis >= MIN_IMG_SIZE (32)
    # for SyNRegistration.get_warp_parameters() to succeed; 16 deterministically
    # fails with the same AssertionError there.
    shape = (32, 32, 32)
    affine = np.diag([1.1, 0.9, 1.3, 1.0])
    blob = np.zeros(shape, dtype=np.float32)
    blob[6:10, 6:10, 6:10] = 1.0

    paths = {}
    for name in ("subject", "template"):
        p = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(blob, affine), str(p))
        paths[name] = ants.image_read(str(p))

    warp = register_to_template(
        paths["subject"], paths["template"], temp_dir=tmp_path / "w", device="cpu"
    )
    moved = apply_warp(
        warp,
        paths["template"],
        paths["subject"],
        temp_dir=tmp_path / "w",
        interpolation="nearest",
    )
    assert_geometry_match(geometry_of(paths["subject"]), geometry_of(moved))
