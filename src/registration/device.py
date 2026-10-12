"""Compute-device selection for the FireANTs registration path."""

VALID_DEVICES = ("cuda", "cpu")


class DeviceUnavailable(RuntimeError):
    """A GPU was requested and none is visible."""


def _torch_cuda_available():
    import torch

    return torch.cuda.is_available()


def resolve_device(requested, cuda_probe=None):
    """Return the device to run FireANTs on, or raise.

    Deliberately does not fall back to CPU when CUDA is missing. FireANTs on CPU
    is slower than the ANTs code it replaces, so a silent fallback inside a BABS
    array produces hundreds of many-hour jobs that all exit 0. `--device cpu`
    remains available as an explicit opt-in for tests and small debugging runs.

    `cuda_probe` overrides the CUDA check so tests need not install torch. It
    is a zero-argument callable, but a plain bool is accepted too: the previous
    name (`cuda_available`) invited `cuda_available=False`, which silently fell
    through to the real probe -- `False or _torch_cuda_available` is the
    function -- while `cuda_available=True` raised TypeError on `True()`.
    """
    if requested not in VALID_DEVICES:
        raise ValueError(
            f"unknown device {requested!r}; expected one of {VALID_DEVICES}"
        )
    if requested == "cpu":
        return "cpu"
    if cuda_probe is None:
        cuda_probe = _torch_cuda_available
    available = cuda_probe() if callable(cuda_probe) else bool(cuda_probe)
    if not available:
        raise DeviceUnavailable(
            "--device cuda was requested but torch.cuda.is_available() is False. "
            "Request a GPU (e.g. sbatch --gres=gpu:1) and run the container with "
            "`apptainer --nv` or `docker --gpus all`, or pass --device cpu for a "
            "small debugging run. If a GPU was allocated and nvidia-smi lists it, "
            "the node's GPU or driver is likely faulty: resubmit excluding that "
            "node (sbatch --exclude=<node>). Not falling back to CPU automatically: "
            "FireANTs on CPU is slower than the ANTs path it replaces."
        )
    return "cuda"
