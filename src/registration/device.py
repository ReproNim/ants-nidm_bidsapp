"""Compute-device selection for the FireANTs registration path."""

VALID_DEVICES = ("cuda", "cpu")


class DeviceUnavailable(RuntimeError):
    """A GPU was requested and none is visible."""


def _torch_cuda_available():
    import torch

    return torch.cuda.is_available()


def resolve_device(requested, cuda_available=None):
    """Return the device to run FireANTs on, or raise.

    Deliberately does not fall back to CPU when CUDA is missing. FireANTs on CPU
    is slower than the ANTs code it replaces, so a silent fallback inside a BABS
    array produces hundreds of many-hour jobs that all exit 0. `--device cpu`
    remains available as an explicit opt-in for tests and small debugging runs.
    """
    if requested not in VALID_DEVICES:
        raise ValueError(
            f"unknown device {requested!r}; expected one of {VALID_DEVICES}"
        )
    if requested == "cpu":
        return "cpu"
    probe = cuda_available or _torch_cuda_available
    if not probe():
        raise DeviceUnavailable(
            "--device cuda was requested but torch.cuda.is_available() is False. "
            "Submit to a GPU partition (e.g. --partition pi_satra --gres=gpu:1) "
            "and run the container with `apptainer --nv`, or pass --device cpu "
            "for a small debugging run. Not falling back to CPU automatically: "
            "FireANTs on CPU is slower than the ANTs path it replaces."
        )
    return "cuda"
