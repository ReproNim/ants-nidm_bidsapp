"""The fusion/quick label vocabularies collide, so the map between them has to
be a bijection and the remap has to be atomic.

These are the two properties that, if broken, reproduce the original bug in a
new form: a renumbering that silently merges or clobbers structures looks
exactly like a working segmentation.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from src.antspy.label_vocabulary import (
    ASEG_IDS_ABSENT_FROM_FREESURFER_LUT,
    ASEG_TO_NEUROMORPHOMETRICS,
    CORTICAL_ID_FLOOR,
    EVIDENCE,
    NEUROMORPHOMETRICS_TO_ASEG,
    aseg_to_neuromorphometrics,
    neuromorphometrics_to_aseg,
    remap_labels,
)


def test_map_is_a_bijection():
    targets = list(NEUROMORPHOMETRICS_TO_ASEG.values())
    assert len(set(targets)) == len(targets), "two source ids share a target id"
    assert len(ASEG_TO_NEUROMORPHOMETRICS) == len(NEUROMORPHOMETRICS_TO_ASEG)


def test_round_trip_restores_every_source_id():
    src = np.array(sorted(NEUROMORPHOMETRICS_TO_ASEG), dtype=np.int32)
    assert np.array_equal(aseg_to_neuromorphometrics(neuromorphometrics_to_aseg(src)), src)


def test_map_covers_only_subcortical_ids():
    """Neither side of the map may touch the cortical DKT range, because those
    ids are already identical in both vocabularies. Remapping one would break
    the 62 cortical labels that currently agree."""
    for src, dst in NEUROMORPHOMETRICS_TO_ASEG.items():
        assert src < CORTICAL_ID_FLOOR, f"source {src} is in the cortical range"
        assert dst < CORTICAL_ID_FLOOR, f"target {dst} is in the cortical range"


def test_cortical_ids_pass_through_unchanged():
    cortical = np.array([1002, 1035, 2002, 2035], dtype=np.int32)
    assert np.array_equal(neuromorphometrics_to_aseg(cortical), cortical)
    assert np.array_equal(aseg_to_neuromorphometrics(cortical), cortical)


def test_background_is_preserved():
    assert neuromorphometrics_to_aseg(np.zeros((2, 2), dtype=np.int32)).sum() == 0


def test_every_entry_records_its_evidence():
    assert set(EVIDENCE) == set(NEUROMORPHOMETRICS_TO_ASEG)
    for src, (kind, dice) in EVIDENCE.items():
        assert kind in {"overlap", "centroid", "protocol"}
        if kind == "overlap":
            assert dice is not None and 0.0 < dice <= 1.0, f"{src} has no usable Dice"
        else:
            assert dice is None


def test_colliding_ids_are_not_clobbered():
    """30 is the left accumbens in one vocabulary and the left vessel in the
    other; 4, 11 and 52 collide the same way. A sequential in-place rewrite
    would lose one of each pair."""
    colliding = sorted(set(NEUROMORPHOMETRICS_TO_ASEG) & set(NEUROMORPHOMETRICS_TO_ASEG.values()))
    assert colliding, "fixture assumes the vocabularies overlap"

    volume = np.array(sorted(NEUROMORPHOMETRICS_TO_ASEG), dtype=np.int32)
    out = neuromorphometrics_to_aseg(volume)
    # Distinct inputs must stay distinct.
    assert len(np.unique(out)) == len(np.unique(volume))
    for src, dst in NEUROMORPHOMETRICS_TO_ASEG.items():
        assert out[volume == src].tolist() == [dst]


def test_remap_rejects_float_volumes():
    with pytest.raises(TypeError):
        remap_labels(np.zeros((2, 2), dtype=np.float32), {1: 2})


def test_remap_rejects_negative_ids():
    with pytest.raises(ValueError):
        remap_labels(np.array([-1, 2], dtype=np.int32), {1: 2})


def test_remap_handles_ids_larger_than_any_mapping_key():
    volume = np.array([0, 35, 9999], dtype=np.int32)
    out = remap_labels(volume, NEUROMORPHOMETRICS_TO_ASEG)
    assert out.tolist() == [0, 16, 9999]


def test_empty_mapping_is_identity():
    volume = np.array([1, 2, 3], dtype=np.int32)
    assert np.array_equal(remap_labels(volume, {}), volume)


def test_unnameable_targets_are_declared():
    """These five aseg targets are absent from FreeSurferColorLUT.txt, so the
    NIDM converter drops them. Declaring them keeps that loss visible instead
    of it showing up as five missing rows in a TTL."""
    assert ASEG_IDS_ABSENT_FROM_FREESURFER_LUT <= set(NEUROMORPHOMETRICS_TO_ASEG.values())


def test_declared_unnameable_targets_match_the_shipped_lut():
    from pathlib import Path

    lut = Path(__file__).resolve().parent.parent / (
        "src/ants_seg_to_nidm/ants_seg_to_nidm/mapping_data/FreeSurferColorLUT.txt"
    )
    if not lut.exists():  # submodule not initialised
        pytest.skip("ants_seg_to_nidm submodule not checked out")
    named = set()
    for line in lut.read_text().splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            named.add(int(parts[0]))
    unnameable = {d for d in NEUROMORPHOMETRICS_TO_ASEG.values() if d not in named}
    assert unnameable == set(ASEG_IDS_ABSENT_FROM_FREESURFER_LUT)


def test_importing_this_module_does_not_pull_in_antspy():
    """``tests/test_wrapper.py`` mocks ANTs by assigning ``sys.modules['ants']``
    before importing the wrapper. That only works while nothing has already
    bound the real ANTsPy into ``src.antspy.wrapper``, so a submodule of
    ``src.antspy`` must not drag ANTsPy in. When it did, the wrapper tests
    failed deep inside ANTsPy with a ``ptrstr()`` type error, naming nothing
    that pointed at the import order.

    Run in a subprocess: by the time the suite reaches this test, other tests
    have legitimately imported ants into this interpreter.
    """
    root = Path(__file__).resolve().parent.parent
    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, %r)
        import src.antspy.label_vocabulary  # noqa: F401
        assert 'ants' not in sys.modules, 'importing label_vocabulary pulled in ANTsPy'
        print('ok')
        """
    ) % str(root)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_antspy_package_still_exports_the_segmentation_class():
    """The lazy ``__init__`` must keep ``from src.antspy import ANTsSegmentation``
    working; making it lazy is only safe if the public name survives."""
    root = Path(__file__).resolve().parent.parent
    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, %r)
        from src.antspy import ANTsSegmentation
        assert ANTsSegmentation.__name__ == 'ANTsSegmentation'
        print('ok')
        """
    ) % str(root)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
