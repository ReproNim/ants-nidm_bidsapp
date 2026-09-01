"""Device resolution tests.

The CUDA probe is injectable so these run without torch installed.
"""
import pytest

from src.registration.device import DeviceUnavailable, resolve_device


def test_cuda_resolves_when_a_gpu_is_present():
    assert resolve_device("cuda", cuda_available=lambda: True) == "cuda"


def test_cuda_without_a_gpu_raises_rather_than_falling_back():
    """Silent CPU fallback under a BABS array turns a fast job into a
    many-hour one while still exiting 0."""
    with pytest.raises(DeviceUnavailable) as exc:
        resolve_device("cuda", cuda_available=lambda: False)
    assert "--device" in str(exc.value)
    assert "cpu" in str(exc.value)


def test_cpu_is_honoured_and_never_probes_for_cuda():
    def explode():
        raise AssertionError("must not probe for CUDA when cpu was requested")

    assert resolve_device("cpu", cuda_available=explode) == "cpu"


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="mps"):
        resolve_device("mps", cuda_available=lambda: True)
