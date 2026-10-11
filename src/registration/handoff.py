"""File-based handoff between ANTsPy and FireANTs.

Every image crossing between the two libraries goes through a NIfTI file on
disk. Nothing here rebuilds a header from loose parts: NIfTI carries the affine
and both ANTsPy and SimpleITK are well-tested readers of it. Pulling a numpy
array out of an ANTsImage and reattaching spacing/origin/direction on the other
side is the failure mode this module exists to make impossible.

"""
from collections import namedtuple
from pathlib import Path

import numpy as np

Geometry = namedtuple("Geometry", "shape spacing origin direction")

# ITK (ANTs, SimpleITK) uses LPS; nibabel/NIfTI affines are RAS. Negating the
# first two axes converts between them.
_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0])


class GeometryMismatch(RuntimeError):
    """Two images that should share a voxel grid do not."""


def _from_affine(affine, shape):
    affine = np.asarray(affine, dtype=float)
    lps = _RAS_TO_LPS @ affine[:3, :3]
    spacing = np.linalg.norm(lps, axis=0)
    spacing[spacing == 0.0] = 1.0
    direction = lps / spacing
    origin = _RAS_TO_LPS @ affine[:3, 3]
    return Geometry(
        tuple(int(s) for s in shape),
        tuple(float(s) for s in spacing),
        tuple(float(o) for o in origin),
        tuple(float(d) for d in np.asarray(direction).ravel()),
    )


def geometry_of(img):
    """Extract a comparable geometry from an ANTsImage, a SimpleITK image, or a
    nibabel image. Results are always in LPS so the three are comparable.
    """
    if hasattr(img, "affine"):  # nibabel
        return _from_affine(img.affine, img.shape)
    if hasattr(img, "GetSpacing"):  # SimpleITK
        # GetSize() is already index order (x, y, z) -- the same convention as
        # ANTsImage.shape and nibabel's .shape, and the one every other branch
        # here returns. It is *not* the numpy shape of GetArrayFromImage(),
        # which is reversed; reversing here made a SimpleITK image compare
        # unequal to the ANTsImage of the very same file whenever the volume
        # was not cubic. Nothing exercised this branch until the inbound
        # crossing guard in fireants.py started comparing an
        # `Image.itk_image` against the ANTsImage it was written from.
        return Geometry(
            tuple(int(s) for s in img.GetSize()),
            tuple(float(s) for s in img.GetSpacing()),
            tuple(float(o) for o in img.GetOrigin()),
            tuple(float(d) for d in img.GetDirection()),
        )
    # ANTsImage
    return Geometry(
        tuple(int(s) for s in img.shape),
        tuple(float(s) for s in img.spacing),
        tuple(float(o) for o in img.origin),
        tuple(float(d) for d in np.asarray(img.direction).ravel()),
    )


def assert_geometry_match(a, b, tol=1e-4, context=""):
    """Raise GeometryMismatch unless a and b describe the same voxel grid.

    Called on every crossing of the ANTs/FireANTs boundary in production, not
    only under pytest. A silently reoriented volume yields a well-formed
    segmentation, a well-formed nidm.ttl, and exit 0.
    """
    where = f" [{context}]" if context else ""
    if tuple(a.shape) != tuple(b.shape):
        raise GeometryMismatch(f"shape differs{where}: {a.shape} != {b.shape}")
    for field in ("spacing", "origin", "direction"):
        va, vb = np.asarray(getattr(a, field)), np.asarray(getattr(b, field))
        if not np.allclose(va, vb, atol=tol, rtol=0.0):
            raise GeometryMismatch(
                f"{field} differs{where} (atol={tol}): {tuple(va)} != {tuple(vb)}"
            )


def write_nifti(img, path):
    """Write an ANTsImage to `path` for FireANTs to read. Returns the path."""
    import ants

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(img, str(path))
    return path


def read_nifti(path, expect=None, context=""):
    """Read a NIfTI back as an ANTsImage, verifying geometry against `expect`.

    `expect` is a Geometry (typically of the fixed/target image). Passing it is
    the point of this function; omitting it makes the read unguarded.
    """
    import ants

    img = ants.image_read(str(path))
    if expect is not None:
        assert_geometry_match(expect, geometry_of(img), context=context or str(path))
    return img
