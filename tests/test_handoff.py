"""Geometry guard and NIfTI boundary tests.

These use nibabel only -- no antspyx, no torch, no fireants -- so they run in
CI unconditionally. They validate the claim the whole design rests on: that a
NIfTI file carries the affine faithfully across libraries.
"""
import numpy as np
import nibabel as nib
import pytest

from src.registration.handoff import (
    Geometry,
    GeometryMismatch,
    assert_geometry_match,
    geometry_of,
)


def _img(affine, shape=(4, 5, 6)):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return nib.Nifti1Image(data, affine)


# A deliberately nasty affine: anisotropic spacing, non-identity direction,
# non-zero origin. An identity affine would hide exactly the bugs we care about.
OBLIQUE = np.array([
    [0.0, -1.2, 0.0, 10.0],
    [0.9, 0.0, 0.0, -20.0],
    [0.0, 0.0, 3.0, 5.5],
    [0.0, 0.0, 0.0, 1.0],
])


def test_geometry_of_extracts_spacing_from_oblique_affine():
    g = geometry_of(_img(OBLIQUE))
    assert g.shape == (4, 5, 6)
    np.testing.assert_allclose(g.spacing, (0.9, 1.2, 3.0), rtol=1e-6)


def test_geometry_survives_a_disk_round_trip(tmp_path):
    src = _img(OBLIQUE)
    path = tmp_path / "oblique.nii.gz"
    nib.save(src, str(path))
    assert_geometry_match(geometry_of(src), geometry_of(nib.load(str(path))))


def test_reoriented_image_is_rejected():
    a = geometry_of(_img(OBLIQUE))
    flipped = OBLIQUE.copy()
    flipped[:3, :3] = -flipped[:3, :3]
    with pytest.raises(GeometryMismatch, match="direction"):
        assert_geometry_match(a, geometry_of(_img(flipped)))


def test_changed_spacing_is_rejected():
    a = geometry_of(_img(OBLIQUE))
    rescaled = OBLIQUE.copy()
    rescaled[:3, :3] *= 2.0
    with pytest.raises(GeometryMismatch, match="spacing"):
        assert_geometry_match(a, geometry_of(_img(rescaled)))


def test_shifted_origin_is_rejected():
    a = geometry_of(_img(OBLIQUE))
    shifted = OBLIQUE.copy()
    shifted[0, 3] += 1.0
    with pytest.raises(GeometryMismatch, match="origin"):
        assert_geometry_match(a, geometry_of(_img(shifted)))


def test_differing_shape_is_rejected():
    with pytest.raises(GeometryMismatch, match="shape"):
        assert_geometry_match(
            geometry_of(_img(OBLIQUE)), geometry_of(_img(OBLIQUE, shape=(4, 5, 7)))
        )


def test_context_string_appears_in_the_error():
    a = geometry_of(_img(OBLIQUE))
    shifted = OBLIQUE.copy()
    shifted[0, 3] += 1.0
    with pytest.raises(GeometryMismatch, match="warped atlas 3"):
        assert_geometry_match(a, geometry_of(_img(shifted)), context="warped atlas 3")


def test_tolerance_admits_float_noise():
    a = geometry_of(_img(OBLIQUE))
    noisy = OBLIQUE.copy()
    noisy[0, 3] += 1e-9
    assert_geometry_match(a, geometry_of(_img(noisy)))


def test_geometry_of_accepts_itk_style_objects():
    """ANTsImage and SimpleITK images expose spacing/origin/direction directly.

    A stub stands in so this runs without antspyx installed.
    """

    class FakeANTsImage:
        shape = (4, 5, 6)
        spacing = (0.9, 1.2, 3.0)
        origin = (-10.0, -20.0, 5.5)
        direction = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    g = geometry_of(FakeANTsImage())
    assert g.shape == (4, 5, 6)
    np.testing.assert_allclose(g.spacing, (0.9, 1.2, 3.0), rtol=1e-6)
