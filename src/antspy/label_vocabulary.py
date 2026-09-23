"""The two label vocabularies this app emits, and the map between them.

The app's two segmentation methods do not speak the same label vocabulary, and
until 2026-09-23 nothing in the codebase said so.

``--method fusion`` votes over ``OASIS-TRT-20_DKT31_CMA_labels_v2/`` (the 20
per-atlas label files). Their cortical ids are DKT (1002-1035 / 2002-2035), but
their **subcortical** ids follow the Neuromorphometrics / BrainCOLOR protocol
distributed with the MICCAI 2012 Multi-Atlas Labelling Challenge -- 35 is the
brain stem, 47/48 the hippocampi, 71/72/73 the cerebellar vermal lobules.

``--method quick`` warps
``OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_OASIS-30_v2.nii.gz``. Despite
the ``DKT31_CMA`` in its filename its subcortical ids are standard FreeSurfer
aseg -- 16 is the brain stem, 17/53 the hippocampi, 10/49 the thalami. The
cortical ids are the same DKT numbers in both.

Both numberings occupy the same small-integer range, so the mismatch is silent:
every id in one vocabulary also exists in the other, denoting a different
structure. That is what made ``--method quick`` look like a 45-68 voxel
registration failure. It was never a registration failure. Comparing label N of
one method against label N of the other compares, for instance, the right
thalamus against the right inferior lateral ventricle.

Why this matters beyond method agreement: ``src/ants_seg_to_nidm`` names every
structure through ``mapping_data/FreeSurferColorLUT.txt``. That LUT is correct
for ``quick``'s aseg ids and wrong for ``fusion``'s Neuromorphometrics ids, so
the delivered ``fusion`` NIDM output carries 13 anatomically impossible
structure names (``Left-Porg``, ``Right-Stellate``, ``Right-undetermined``,
``5th-Ventricle``, ...) and roughly twenty more that are silently wrong but
plausible-looking -- the right thalamus reported as ``Right-Substancia-Nigra``,
the cerebellar white matter as ``Right-Cerebral-White-Matter``, the lateral
ventricles as ``Right-Putamen`` / ``Right-Pallidum``. Those pass every
structural check on the TTL.

This module only states the correspondence. It deliberately does not decide
which vocabulary the app should emit; see
``scripts/diag_label_vocabulary_correspondence.py`` for the evidence behind
each entry and ``docs/superpowers/specs/analysis/`` for the recorded run.

Provenance
----------
Every entry was derived from data, not from a published protocol table:

* ``overlap`` -- confirmed by reciprocal maximum spatial overlap between the
  ``fusion`` and ``quick`` dsegs of the *same* subject in the *same* space
  (sub-0051456, tier-2 run of 2026-09-14). The recorded Dice is in
  ``EVIDENCE``.
* ``centroid`` -- the structure is small enough, or over-segmented enough, that
  maximum overlap picks a neighbour; confirmed instead by centroid proximity
  plus per-atlas volume being in the right range.
* ``protocol`` -- too small for either test to be decisive (a few dozen voxels);
  assigned from the Neuromorphometrics protocol and consistent with the
  left/right mirror pairing of its neighbours.
"""

from __future__ import annotations

import numpy as np

#: Neuromorphometrics / BrainCOLOR id -> FreeSurfer aseg id.
#:
#: Subcortical only. Cortical DKT ids (>= 1000) are identical in both
#: vocabularies and are deliberately absent here; see
#: :func:`neuromorphometrics_to_aseg`, which passes them through unchanged.
NEUROMORPHOMETRICS_TO_ASEG: dict[int, int] = {
    4: 14,    # 3rd Ventricle
    11: 15,   # 4th Ventricle
    23: 58,   # Right Accumbens area
    30: 26,   # Left Accumbens area
    31: 54,   # Right Amygdala
    32: 18,   # Left Amygdala
    35: 16,   # Brain Stem
    36: 50,   # Right Caudate
    37: 11,   # Left Caudate
    38: 45,   # Right Cerebellum Exterior
    39: 6,    # Left Cerebellum Exterior
    40: 46,   # Right Cerebellum White Matter
    41: 7,    # Left Cerebellum White Matter
    46: 24,   # CSF
    47: 53,   # Right Hippocampus
    48: 17,   # Left Hippocampus
    49: 44,   # Right Inferior Lateral Ventricle
    50: 5,    # Left Inferior Lateral Ventricle
    51: 43,   # Right Lateral Ventricle
    52: 4,    # Left Lateral Ventricle
    55: 52,   # Right Pallidum
    56: 13,   # Left Pallidum
    57: 51,   # Right Putamen
    58: 12,   # Left Putamen
    59: 49,   # Right Thalamus Proper
    60: 10,   # Left Thalamus Proper
    61: 60,   # Right Ventral DC
    62: 28,   # Left Ventral DC
    63: 62,   # Right vessel
    64: 30,   # Left vessel
    69: 85,   # Optic Chiasm
    71: 630,  # Cerebellar Vermal Lobules I-V
    72: 631,  # Cerebellar Vermal Lobules VI-VII
    73: 632,  # Cerebellar Vermal Lobules VIII-X
    75: 91,   # Left Basal Forebrain
    76: 92,   # Right Basal Forebrain
}

#: How each entry of :data:`NEUROMORPHOMETRICS_TO_ASEG` was established.
#:
#: ``(evidence_kind, dice)``. ``dice`` is the measured Dice on sub-0051456 for
#: ``overlap`` entries and ``None`` otherwise. It is recorded so a regression in
#: the derivation is visible rather than silently re-derived.
EVIDENCE: dict[int, tuple[str, float | None]] = {
    4: ("centroid", None),
    11: ("centroid", None),
    23: ("overlap", 0.42),
    30: ("overlap", 0.28),
    31: ("overlap", 0.37),
    32: ("overlap", 0.43),
    35: ("overlap", 0.69),
    36: ("overlap", 0.38),
    37: ("overlap", 0.36),
    38: ("overlap", 0.74),
    39: ("overlap", 0.72),
    40: ("overlap", 0.41),
    41: ("overlap", 0.33),
    46: ("overlap", 0.40),
    47: ("overlap", 0.53),
    48: ("overlap", 0.55),
    49: ("centroid", None),
    50: ("centroid", None),
    51: ("overlap", 0.43),
    52: ("overlap", 0.46),
    55: ("overlap", 0.46),
    56: ("overlap", 0.38),
    57: ("overlap", 0.60),
    58: ("overlap", 0.63),
    59: ("overlap", 0.64),
    60: ("overlap", 0.68),
    61: ("overlap", 0.55),
    62: ("overlap", 0.57),
    63: ("protocol", None),
    64: ("protocol", None),
    69: ("protocol", None),
    71: ("overlap", 0.64),
    72: ("overlap", 0.45),
    73: ("overlap", 0.47),
    75: ("centroid", None),
    76: ("centroid", None),
}

#: FreeSurfer aseg id -> Neuromorphometrics / BrainCOLOR id.
ASEG_TO_NEUROMORPHOMETRICS: dict[int, int] = {
    v: k for k, v in NEUROMORPHOMETRICS_TO_ASEG.items()
}

#: Aseg targets that ``FreeSurferColorLUT.txt`` does not name.
#:
#: The NIDM converter skips a stats row whose id is absent from the LUT, so
#: these five structures are dropped from the TTL rather than mis-named. The
#: cerebellar vermal lobules and basal forebrain have no standard aseg id; the
#: pre-fused template's own 630/631/632 and 91/92 are carried through so the
#: two methods at least agree.
ASEG_IDS_ABSENT_FROM_FREESURFER_LUT: frozenset[int] = frozenset({91, 92, 630, 631, 632})

#: Cortical ids at or above this value are DKT and identical in both
#: vocabularies.
CORTICAL_ID_FLOOR = 1000


def _check_bijection() -> None:
    if len(ASEG_TO_NEUROMORPHOMETRICS) != len(NEUROMORPHOMETRICS_TO_ASEG):
        raise AssertionError("label vocabulary map is not injective")


_check_bijection()


def remap_labels(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Renumber ``labels`` through ``mapping``, atomically.

    The two vocabularies overlap: 30 means the left accumbens area in one and
    the left vessel in the other, and 4, 11 and 52 collide the same way. An
    in-place, id-by-id rewrite would therefore clobber ids it has already
    written. This builds the output from the untouched input instead.

    Ids absent from ``mapping`` are passed through unchanged, which is what
    carries the cortical DKT range and the background across.

    Parameters
    ----------
    labels
        Integer label volume.
    mapping
        Source id -> target id.

    Returns
    -------
    numpy.ndarray
        A new array of the same shape and dtype.
    """
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"label volume must be integer-typed, got {labels.dtype}")
    if labels.size and labels.min() < 0:
        raise ValueError("label volume must not contain negative ids")

    out = labels.copy()
    if not mapping:
        return out

    # A lookup table is both atomic and O(n): every voxel is read from the
    # pristine input exactly once.
    highest = max(int(labels.max(initial=0)), max(mapping))
    table = np.arange(highest + 1, dtype=labels.dtype)
    for src, dst in mapping.items():
        table[src] = dst
    return table[labels]


def neuromorphometrics_to_aseg(labels: np.ndarray) -> np.ndarray:
    """Renumber a ``fusion`` label volume into FreeSurfer aseg ids.

    Cortical DKT ids pass through unchanged.
    """
    return remap_labels(labels, NEUROMORPHOMETRICS_TO_ASEG)


def aseg_to_neuromorphometrics(labels: np.ndarray) -> np.ndarray:
    """Renumber a ``quick`` label volume into Neuromorphometrics ids.

    Cortical DKT ids pass through unchanged.
    """
    return remap_labels(labels, ASEG_TO_NEUROMORPHOMETRICS)
