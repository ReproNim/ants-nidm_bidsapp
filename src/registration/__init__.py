"""Registration for the ants-nidm BIDS App, backed by FireANTs.

The FireANTs backend is imported lazily so that `handoff`, `labels`, and
`device` stay importable without torch or fireants -- CI runs their tests
unconditionally.
"""
from src.registration.device import DeviceUnavailable, resolve_device
from src.registration.handoff import (
    Geometry,
    GeometryMismatch,
    assert_geometry_match,
    geometry_of,
)
from src.registration.labels import LabelMap, build_label_map

__all__ = [
    "DeviceUnavailable",
    "Geometry",
    "GeometryMismatch",
    "LabelMap",
    "TemplateWarp",
    "apply_warp",
    "assert_geometry_match",
    "build_label_map",
    "geometry_of",
    "register_to_template",
    "resolve_device",
    "warp_atlases",
]


def __getattr__(name):
    if name in ("warp_atlases", "register_to_template", "apply_warp", "TemplateWarp"):
        from src.registration import fireants

        return getattr(fireants, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
