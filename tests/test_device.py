"""Device resolution tests.

The CUDA probe is injectable so these run without torch installed.
"""
import pytest

from src.registration.device import DeviceUnavailable, resolve_device


def test_cuda_resolves_when_a_gpu_is_present():
    assert resolve_device("cuda", cuda_probe=lambda: True) == "cuda"


def test_cuda_without_a_gpu_raises_rather_than_falling_back():
    """Silent CPU fallback under a BABS array turns a fast job into a
    many-hour one while still exiting 0."""
    with pytest.raises(DeviceUnavailable) as exc:
        resolve_device("cuda", cuda_probe=lambda: False)
    assert "--device" in str(exc.value)
    assert "cpu" in str(exc.value)


def test_cpu_is_honoured_and_never_probes_for_cuda():
    def explode():
        raise AssertionError("must not probe for CUDA when cpu was requested")

    assert resolve_device("cpu", cuda_probe=explode) == "cpu"


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="mps"):
        resolve_device("mps", cuda_probe=lambda: True)


def test_a_bare_false_probe_is_honoured_not_ignored():
    """The parameter used to be `cuda_available`, resolved as
    `cuda_available or _torch_cuda_available`. Passing the bool the name
    invites -- `cuda_available=False` -- fell straight through to the real
    torch probe, so the caller's answer was silently discarded and the result
    depended on the host's actual GPU."""
    with pytest.raises(DeviceUnavailable):
        resolve_device("cuda", cuda_probe=False)


def test_a_bare_true_probe_is_honoured_not_a_typeerror():
    """`True()` is a TypeError, which is what the old spelling produced."""
    assert resolve_device("cuda", cuda_probe=True) == "cuda"
