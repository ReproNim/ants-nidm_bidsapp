"""Dockerfile and Singularity are near-duplicates and have drifted before.

This pins the versions that must match across both, and asserts the container
verifies its own fused_ops build -- without it FireANTs silently degrades to a
one-hot label path costing 97x the memory.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text()
SINGULARITY = (ROOT / "Singularity").read_text()
REQUIREMENTS = (ROOT / "requirements.txt").read_text()

BUILD_FILES = [(DOCKERFILE, "Dockerfile"), (SINGULARITY, "Singularity")]
BUILD_IDS = ["Dockerfile", "Singularity"]

PINNED = {
    "ants": r"ANTs/releases/download/v([\d.]+)/",
    "cuda_base": r"nvidia/cuda:([\d.]+)-devel-ubuntu22\.04",
    "fireants": r"fireants==([\d.]+)",
}

# The commands themselves, not the prose about them. Both files discuss
# `setup.py build_ext` and `import fireants_fused_ops` in comments, so a plain
# substring search finds the comment first and any ordering check on it is
# meaningless.
BUILD_EXT_CMD = r"python3 setup\.py build_ext"
# `import torch` must come first: the extension links against torch/lib/libc10.so
# without an rpath, so a bare `import fireants_fused_ops` fails with
# "libc10.so: cannot open shared object file" even when the build is perfect
# (build job 22183323 on 2026-09-07 failed exactly there). FireANTs always has
# torch loaded before touching the extension, so torch-first is the real path.
FUSED_OPS_ASSERTION_CMD = r'python3 -c "import torch, fireants_fused_ops'


@pytest.mark.parametrize("name,pattern", sorted(PINNED.items()))
def test_versions_match_across_build_files(name, pattern):
    d = re.search(pattern, DOCKERFILE)
    s = re.search(pattern, SINGULARITY)
    assert d, f"{name} version not found in Dockerfile"
    assert s, f"{name} version not found in Singularity"
    assert d.group(1) == s.group(1), (
        f"{name} differs: Dockerfile {d.group(1)} vs Singularity {s.group(1)}"
    )


def test_fireants_pin_matches_requirements():
    req = re.search(r"fireants==([\d.]+)", REQUIREMENTS)
    doc = re.search(r"fireants==([\d.]+)", DOCKERFILE)
    assert req and doc and req.group(1) == doc.group(1)


@pytest.mark.parametrize("text,name", BUILD_FILES, ids=BUILD_IDS)
def test_build_verifies_fused_ops(text, name):
    """Without fused_ops the default label path silently degrades rather than
    failing, so the build must prove the extension imports -- and the assertion
    must be positioned so it can actually fail the build.

    Greping for the string alone is false confidence. In the Dockerfile the
    assertion has to stay inside the same `RUN` chain as `setup.py build_ext`,
    or it lands in a layer a broken build never reaches. In Singularity it sits
    ~40 commands from the end of one `%post` scriptlet whose exit status is its
    *last* command's -- see test_singularity_post_fails_fast.
    """
    build = re.search(BUILD_EXT_CMD, text)
    check = re.search(FUSED_OPS_ASSERTION_CMD, text)
    assert build, f"{name} must build fused_ops with `python3 setup.py build_ext`"
    assert check, (
        f'{name} must assert `python3 -c "import torch, fireants_fused_ops"` '
        "succeeds (torch first, or the import fails on libc10.so)"
    )
    assert check.start() > build.start(), (
        f"{name} must assert the extension imports *after* building it"
    )
    if name == "Dockerfile":
        # Every line between the two must be a `\`-continuation, i.e. both are
        # part of one RUN instruction.
        between = text[build.start():check.start()].splitlines()[:-1]
        assert between and all(line.rstrip().endswith("\\") for line in between), (
            "the fused_ops import assertion must live in the same RUN chain as "
            "`setup.py build_ext`, so a failed build fails that layer; found a "
            f"chain break in {between!r}"
        )


def test_singularity_post_fails_fast():
    """`%post` is one scriptlet and its exit status is its *last* command's.

    The `import fireants_fused_ops` assertion is followed by ~40 further
    commands (ANTs download, template downloads, pip installs), so without
    `set -e` a failed fused_ops build produces a container that builds fine and
    then silently returns one-hot label volumes at runtime -- exactly the
    failure that assertion exists to prevent.
    """
    # Anchor on the section header at the start of a line: "%post" also occurs
    # in prose inside the header comment.
    post = re.search(r"^%post\s*$", SINGULARITY, re.MULTILINE)
    assert post, "Singularity has no %post section"
    rest = SINGULARITY[post.end():]
    nxt = re.search(r"^%\w+\s*$", rest, re.MULTILINE)
    body = rest[: nxt.start()] if nxt else rest
    statements = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert statements, "%post is empty"
    assert statements[0] == "set -e", (
        "the first statement in %post must be `set -e`; found "
        f"{statements[0]!r}. Every check in %post -- including the fused_ops "
        "import assertion -- is inert without it."
    )


@pytest.mark.parametrize("text,name", BUILD_FILES, ids=BUILD_IDS)
def test_base_image_ships_the_cuda_toolkit(text, name):
    """fused_ops/setup.py uses cpp_extension.CUDAExtension, which calls
    _join_cuda_home() and raises `EnvironmentError: CUDA_HOME environment
    variable is not set` when nvcc is missing. A -runtime or plain-ubuntu base
    therefore cannot build the extension at all: the build dies at
    `setup.py build_ext`, before the import assertion is ever reached.
    """
    base = re.search(r"^\s*(?:FROM|From:)\s*nvidia/cuda:[\d.]+-(\w+)-ubuntu[\d.]+",
                     text, re.MULTILINE)
    assert base, f"{name} must be based on an nvidia/cuda image"
    assert base.group(1) == "devel", (
        f"{name} uses the '{base.group(1)}' CUDA image; only 'devel' ships nvcc"
    )


@pytest.mark.parametrize("text,name", BUILD_FILES, ids=BUILD_IDS)
def test_cuda_arch_list_names_the_target_gpus(text, name):
    """The build host has no GPU, so nvcc cannot autodetect an architecture.
    Deployment targets are A100 (sm_80) and H100 (sm_90).
    """
    match = re.search(r'TORCH_CUDA_ARCH_LIST="([^"]+)"', text)
    assert match, f"{name} must set TORCH_CUDA_ARCH_LIST for a GPU-less build host"
    arches = {a.strip() for a in match.group(1).replace(";", " ").split()}
    assert {"8.0", "9.0"} <= arches, (
        f"{name}: TORCH_CUDA_ARCH_LIST={match.group(1)!r} must cover A100 (8.0) "
        "and H100 (9.0)"
    )


@pytest.mark.parametrize("text,name", BUILD_FILES, ids=BUILD_IDS)
def test_build_installs_cuda_torch(text, name):
    assert "torch" in text, f"{name} must install torch"
    assert "cu12" in text or "cuda" in text.lower(), (
        f"{name} must install a CUDA torch build, not the CPU wheel"
    )


@pytest.mark.parametrize("text,name", BUILD_FILES, ids=BUILD_IDS)
def test_build_keeps_the_ants_payload(text, name):
    """FireANTs is registration-only: N4, antsBrainExtraction.sh and
    joint_label_fusion voting still run on ANTs 2.5.4 + antspyx. Dropping
    either from the image breaks the pipeline everywhere except registration.
    """
    assert "antsBrainExtraction.sh" in text, (
        f"{name} must install the ANTs binaries; ANTsPy does not ship the "
        "command-line scripts"
    )
    # antspyx reaches the Singularity image through requirements.txt rather
    # than a literal pip line, so assert it is still a declared dependency.
    assert re.search(r"^antspyx", REQUIREMENTS, re.MULTILINE), (
        "antspyx must stay a dependency: N4 and joint_label_fusion still run "
        "on ANTsPy"
    )


@pytest.mark.parametrize("text,name", BUILD_FILES, ids=BUILD_IDS)
def test_fused_ops_branch_matches_fireants_pin(text, name):
    """fused_ops is compiled from a `git clone --branch vX.Y.Z` checkout that is
    textually independent of the `fireants==X.Y.Z` pip pin a few lines above it.
    If a future version bump touches one and not the other, fused_ops gets
    built against a different FireANTs than the one actually on sys.path --
    and the failure is silent: a missing or mismatched extension makes
    FireANTs degrade to the one-hot label path rather than raise
    (fireants/io/image.py:145-147). This is the one drift the parity test most
    needs to catch.
    """
    branch = re.search(r"--branch v([\d.]+)", text)
    pin = re.search(r"fireants==([\d.]+)", text)
    assert branch, f"{name} must build fused_ops from a `--branch vX.Y.Z` tag"
    assert pin, f"{name} must pin fireants==X.Y.Z"
    assert branch.group(1) == pin.group(1), (
        f"{name}: fused_ops built from branch v{branch.group(1)} but the pip pin "
        f"is fireants=={pin.group(1)} -- these must match"
    )
