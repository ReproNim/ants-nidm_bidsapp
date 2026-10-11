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
        # One value per batch element, not a batch mean -- see
        # tests/test_fireants_units.py for why the distinction matters.
        fractions = folding_fraction(reg, batch, batch)
        assert len(fractions) == 1
        assert max(fractions) < 0.001


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


def test_fireants_load_preserves_the_ants_header(tmp_path):
    """The inbound crossing guard, on the direction that can actually differ.

    The *outbound* check -- read_nifti(..., expect=target_geom) on a volume
    _tensor_to_nifti just stamped with CopyInformation(reference_itk), where
    reference_itk is that same fixed image -- compares a header with itself.
    Spacing, origin and direction cannot differ there by construction, so an
    assertion on it can never fail and proves nothing.

    What is genuinely unverified is whether FireANTs' SimpleITK read
    reproduced the header ANTsPy wrote. That is what warp_atlases,
    register_to_template and apply_warp now assert on every load, and what
    this test exercises: a deliberately nasty affine (anisotropic spacing,
    non-identity direction, non-zero origin, non-cubic shape) written by
    ANTsPy and read back by FireANTs.
    """
    import nibabel as nib

    from fireants.io import Image

    from src.registration.fireants import _assert_inbound
    from src.registration.handoff import geometry_of, write_nifti

    ants = pytest.importorskip("ants")

    # Non-cubic on purpose: an (n, n, n) volume hides every axis-order bug.
    data = np.arange(8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
    affine = np.array([
        [0.0, -1.2, 0.0, 10.0],
        [0.9, 0.0, 0.0, -20.0],
        [0.0, 0.0, 3.0, 5.5],
        [0.0, 0.0, 0.0, 1.0],
    ])
    seed = tmp_path / "seed.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(seed))
    source = ants.image_read(str(seed))

    written = write_nifti(source, tmp_path / "crossing.nii.gz")
    loaded = Image.load_file(str(written), device="cpu")

    _assert_inbound(geometry_of(source), loaded, "inbound test")


def test_inbound_guard_catches_a_reoriented_load(tmp_path):
    """Negative control: the guard has teeth.

    `fireants.io.Image.__init__` accepts an `orientation` argument that runs
    sitk.DICOMOrient and rewrites the header, a known hazard. Production never passes it -- but if it ever did, or if
    SimpleITK read the file differently for any other reason, the inbound
    guard must fail rather than let a silently reoriented volume through.
    """
    import nibabel as nib

    from fireants.io import Image

    from src.registration.fireants import _assert_inbound
    from src.registration.handoff import GeometryMismatch, geometry_of, write_nifti

    ants = pytest.importorskip("ants")

    data = np.arange(8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
    affine = np.diag([1.1, 0.9, 1.3, 1.0])
    affine[:3, 3] = [3.0, -4.0, 5.0]
    seed = tmp_path / "seed.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(seed))
    source = ants.image_read(str(seed))

    written = write_nifti(source, tmp_path / "crossing.nii.gz")
    reoriented = Image.load_file(str(written), device="cpu", orientation="LPS")

    with pytest.raises(GeometryMismatch) as exc:
        _assert_inbound(geometry_of(source), reoriented, "inbound test")
    # Not a shape difference: this reorientation keeps the shape and moves the
    # origin and direction, which is precisely the silent failure mode.
    assert "shape" not in str(exc.value)


def test_apply_warp_output_lands_on_the_reference_grid(tmp_path):
    """End-to-end shape agreement between apply_warp's result and the
    reference. Kept deliberately modest in what it claims: the header of the
    result is copied from the fixed image, so this can only catch a shape
    error -- the falsifiable geometry content is in the two inbound tests
    above.
    """
    import nibabel as nib

    from src.registration.fireants import apply_warp, register_to_template

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
    assert moved.shape == paths["subject"].shape


def _write_box(path, shape, box, value=1.0, dtype=np.float32):
    import nibabel as nib

    arr = np.zeros(shape, dtype=dtype)
    arr[box] = value
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))
    return path


@pytest.fixture
def fast_registration(monkeypatch):
    """Cut the multi-resolution schedule to a handful of iterations. These
    tests check batch plumbing, not registration quality."""
    from src.registration import fireants as mod

    monkeypatch.setattr(mod, "AFFINE_ITERATIONS", [2, 1, 1])
    monkeypatch.setattr(mod, "SYN_ITERATIONS", [2, 1, 1])


def _atlas_set(tmp_path, n, shape=(32, 32, 32)):
    """n atlases (shifted boxes) with sparse DKT-style label ids."""
    atlases, labels = [], []
    for k in range(n):
        s = 8 + k
        atlases.append(_write_box(tmp_path / f"atlas{k}.nii.gz", shape,
                                  (slice(s, s + 8), slice(8, 16), slice(8, 16))))
        labels.append(_write_box(tmp_path / f"labels{k}.nii.gz", shape,
                                 (slice(s, s + 8), slice(8, 16), slice(8, 16)),
                                 value=1002 + k, dtype=np.int32))
    return atlases, labels


@pytest.mark.parametrize("n_atlases,batch_size", [(2, 0), (3, 2)])
def test_warp_atlases_registers_a_batch_against_one_fixed_image(
    tmp_path, fast_registration, n_atlases, batch_size
):
    """Tier-2 regression (job 22184095, 2026-09-07): one fixed image against
    N batched atlases raised

        ground truth has differing shape (torch.Size([1, 1, ...])) from pred
        (torch.Size([20, 1, ...]))

    inside fireants' CC loss. FireANTs' registration accepts a fixed batch of 1
    as "broadcastable" at construction, but the loss compares shapes strictly
    and BatchedImages only expands to N once the caller invokes
    `.broadcast(N)` -- which is what fireants' own template builder does per
    batch. (3, 2) covers a trailing partial chunk, which must be broadcast to
    its own size, not the previous chunk's.
    """
    from src.registration.fireants import warp_atlases

    ants = pytest.importorskip("ants")

    shape = (32, 32, 32)
    target_path = _write_box(tmp_path / "target.nii.gz", shape,
                             (slice(10, 18), slice(8, 16), slice(8, 16)))
    mask_path = _write_box(tmp_path / "mask.nii.gz", shape,
                           (slice(4, 28), slice(4, 28), slice(4, 28)))
    atlases, labels = _atlas_set(tmp_path, n_atlases, shape)

    warped_atlases, warped_labels = warp_atlases(
        ants.image_read(str(target_path)),
        ants.image_read(str(mask_path)),
        atlases,
        labels,
        temp_dir=tmp_path / "work",
        device="cpu",
        batch_size=batch_size,
        label_interpolation="nearest",
    )

    assert len(warped_atlases) == n_atlases
    assert len(warped_labels) == n_atlases
    for k, lab in enumerate(warped_labels):
        ids = set(np.unique(lab.numpy()).astype(int)) - {0}
        assert ids == {1002 + k}, f"atlas {k} labels came back as {ids}"


@pytest.mark.parametrize("masked,moving_mask,mode", [
    (False, "brain", "mult"), (True, "brain", "mult"), (True, "all", "mult"), (True, "brain", "max"),
], ids=["cc", "masked_cc", "masked_fixed_only", "masked_max"])
def test_warp_atlases_recovers_a_large_physical_offset(tmp_path, fast_registration, monkeypatch, masked, moving_mask, mode):
    """Diagnostic 22188503 (2026-09-07): every OASIS-TRT-20 atlas brain sits
    50-70 mm from the subject brain in physical space (the OASIS-30 template,
    ~200 mm), and the affine started from identity, so the atlases never
    reached the subject: warped atlases covered 26-50% of the target brain,
    in-mask NCC 0.09-0.22, single-atlas label Dice ~0 against the delivered
    derivative. ANTs' SyNRA begins with a centre-of-mass initial transform;
    FireANTs offers the same via MomentsRegistration. This test puts the atlas
    box 40 mm away through its header alone -- same voxel content, shifted
    origin -- which identity initialisation cannot recover on a 32-voxel grid.
    """
    import nibabel as nib

    from src.registration import fireants as mod
    from src.registration.fireants import warp_atlases

    ants = pytest.importorskip("ants")
    monkeypatch.setattr(mod, "LOSS_MASKED", masked)
    monkeypatch.setattr(mod, "LOSS_MASK_MOVING", moving_mask)
    monkeypatch.setattr(mod, "LOSS_MASK_MODE", mode)

    shape = (32, 32, 32)
    box = (slice(10, 18), slice(10, 18), slice(10, 18))
    target_path = _write_box(tmp_path / "target.nii.gz", shape, box)
    mask_path = _write_box(tmp_path / "mask.nii.gz", shape,
                           (slice(4, 28), slice(4, 28), slice(4, 28)))

    shifted = np.eye(4)
    shifted[0, 3] = 40.0  # header origin only: the box is physically 40 mm away
    arr = np.zeros(shape, dtype=np.float32)
    arr[box] = 1.0
    lab = np.zeros(shape, dtype=np.int32)
    lab[box] = 1002
    nib.save(nib.Nifti1Image(arr, shifted), str(tmp_path / "atlas.nii.gz"))
    nib.save(nib.Nifti1Image(lab, shifted), str(tmp_path / "labels.nii.gz"))

    _, warped_labels = warp_atlases(
        ants.image_read(str(target_path)),
        ants.image_read(str(mask_path)),
        [tmp_path / "atlas.nii.gz"],
        [tmp_path / "labels.nii.gz"],
        temp_dir=tmp_path / "work",
        device="cpu",
        batch_size=0,
        label_interpolation="nearest",
    )

    got = warped_labels[0].numpy() == 1002
    want = np.zeros(shape, dtype=bool)
    want[box] = True
    dice = 2 * np.logical_and(got, want).sum() / max(1, got.sum() + want.sum())
    assert dice > 0.8, (
        f"warped atlas label Dice {dice:.3f} against the target box "
        f"({int(got.sum())} label voxels): the 40 mm header offset was not "
        "recovered -- registration needs a centre-of-mass initialisation"
    )


@pytest.mark.parametrize("deformable", ["syn", "greedy"])
def test_warp_atlases_handles_a_reoriented_atlas_header(tmp_path, fast_registration, monkeypatch, deformable):
    """The OASIS-TRT-20 atlases are stored ASL; subjects arrive LAS. The same
    physical box, stored under a permuted/flipped axis order, must register
    onto the target with no search at all -- in physical space it already
    sits there. If direction cosines were ignored anywhere on the FireANTs
    side, the box would come back rotated and Dice would collapse. The box is
    elongated (12x6x3) so a wrong axis order cannot pass by symmetry.
    """
    import nibabel as nib

    from src.registration import fireants as mod
    from src.registration.fireants import warp_atlases

    ants = pytest.importorskip("ants")
    monkeypatch.setattr(mod, "DEFORMABLE", deformable)
    monkeypatch.setattr(mod, "GREEDY_ITERATIONS", [2, 1, 1])

    shape = (32, 32, 32)
    box = (slice(10, 22), slice(13, 19), slice(15, 18))
    target_path = _write_box(tmp_path / "target.nii.gz", shape, box)
    mask_path = _write_box(tmp_path / "mask.nii.gz", shape,
                           (slice(4, 28), slice(4, 28), slice(4, 28)))

    arr = np.zeros(shape, dtype=np.float32)
    arr[box] = 1.0
    lab = np.zeros(shape, dtype=np.int32)
    lab[box] = 1002
    ras_to_asl = nib.orientations.ornt_transform(
        nib.io_orientation(np.eye(4)), nib.orientations.axcodes2ornt(("A", "S", "L")))
    atlas = nib.Nifti1Image(arr, np.eye(4)).as_reoriented(ras_to_asl)
    labels = nib.Nifti1Image(lab, np.eye(4)).as_reoriented(ras_to_asl)
    assert nib.aff2axcodes(atlas.affine) == ("A", "S", "L")
    nib.save(atlas, str(tmp_path / "atlas.nii.gz"))
    nib.save(labels, str(tmp_path / "labels.nii.gz"))

    _, warped_labels = warp_atlases(
        ants.image_read(str(target_path)),
        ants.image_read(str(mask_path)),
        [tmp_path / "atlas.nii.gz"],
        [tmp_path / "labels.nii.gz"],
        temp_dir=tmp_path / "work",
        device="cpu",
        batch_size=0,
        label_interpolation="nearest",
    )

    got = warped_labels[0].numpy() == 1002
    want = np.zeros(shape, dtype=bool)
    want[box] = True
    dice = 2 * np.logical_and(got, want).sum() / max(1, got.sum() + want.sum())
    assert dice > 0.8, (
        f"Dice {dice:.3f} for an ASL-stored atlas against a RAS target: "
        "direction cosines are not being honoured somewhere in the FireANTs path"
    )


def test_template_registration_refuses_a_folded_warp(tmp_path, monkeypatch):
    """--method quick pulls the template labels through this warp, so a
    folding warp must stop the subject, as it does for an atlas warp."""
    import nibabel as nib

    from src.registration import fireants as mod

    ants = pytest.importorskip("ants")
    shape = (32, 32, 32)
    vol = np.zeros(shape, dtype=np.float32)
    vol[8:16, 8:16, 8:16] = 1.0
    path = tmp_path / "box.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))
    img = ants.image_read(str(path))

    monkeypatch.setattr(mod, "folding_fraction", lambda *a, **k: [0.01])
    with pytest.raises(RuntimeError, match="folded"):
        mod.register_to_template(img, img, temp_dir=tmp_path / "work", device="cpu")
