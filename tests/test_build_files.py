"""Dockerfile and Singularity are near-duplicates and have drifted before.

This pins the versions that must match across both, and asserts the container
verifies its own fused_ops build -- without it FireANTs silently degrades to a
one-hot label path costing 97x the memory.

Each `python3 -c` build check must be able to fail the build. The recipes are
read with bashlex (a shell parser, test-only dependency), and a check passes
only under these rules:

Singularity %post (run by sh with `set -e`):
- The check is a top-level simple command of %post: not inside an
  if/while/until/for/case compound, a `{ }` group, a `( )` subshell or a
  function body; not an operand of `&&`, `||`, `|` or `!`; not run with `&`;
  not in a heredoc body or a command substitution; and the command is the
  check alone (`[PYTHONPATH=/opt ]python3 -c "..."`, no `-O`).
- A top-level `set` turns errexit on before the check (`set -e` inside a
  compound does not count).
- Nothing before the check, at any depth (function bodies and command
  substitutions included), can end %post with status 0, override its status
  or turn errexit off: `exit` unless its status is a literal 1-255, `exec`
  with a command (`exec` with only redirections is fine), `return` outside a
  function body, `trap`, `eval`, `.`, `source`, `alias`, `command` or
  `builtin` wrapping any of these, a command name that is an expansion, a
  function named `python3`, and any `set` other than -e -u -x -v and
  -o errexit|nounset|xtrace|verbose|pipefail (so every spelling of `set +e`,
  and `set -n`, is rejected). `exit $?` is rejected too: inside
  `if true; then exit $?; fi` it exits 0.
- What follows the check does not matter: a failing check stops %post first.

Dockerfile RUN (run by sh -c, no errexit): the instruction is the check alone
or a `&&` chain of simple commands that includes it, with no heredoc, and the
commands before the check pass the same "nothing before" rule.

The rules fail closed: bashlex 0.18 does not parse `case` or `time` (and needs
quoted heredoc delimiters unquoted first, which the tests do), and a recipe it
cannot parse fails with bashlex's message. The rules were checked against
`bash --posix -e`; the image's /bin/sh is dash, which uses the same POSIX
errexit, exit and trap semantics.
"""
import re
from pathlib import Path

import bashlex
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


def _post_body(text=None):
    """The raw text of Singularity's %post section."""
    text = SINGULARITY if text is None else text
    # Anchor on the section header at the start of a line: "%post" also occurs
    # in prose inside the header comment.
    post = re.search(r"^%post\s*$", text, re.MULTILINE)
    assert post, "Singularity has no %post section"
    rest = text[post.end():]
    nxt = re.search(r"^%\w+\s*$", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def _post_statements():
    """Non-blank, non-comment lines of Singularity's %post, stripped, with
    `\\`-continuation lines joined into the line they continue."""
    return _join_continuations(_post_body())


def _join_continuations(text):
    """Non-blank, non-comment lines, stripped, a line ending in `\\` joined
    with the lines it continues: one entry per shell command line."""
    joined, current = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        current.append(s[:-1].rstrip() if s.endswith("\\") else s)
        if not s.endswith("\\"):
            joined.append(" ".join(current))
            current = []
    if current:
        joined.append(" ".join(current))
    return joined


def _commands(text):
    """Lines of a build file that are not comments, stripped."""
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


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
    statements = _post_statements()
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


def test_images_record_their_commit():
    """Both recipes stamp org.opencontainers.image.revision from GIT_COMMIT,
    defaulting to "unknown" so a build without --build-arg says so."""
    assert re.search(r"^%arguments\n\s+GIT_COMMIT=unknown$", SINGULARITY, re.M)
    assert re.search(r"^\s+org\.opencontainers\.image\.revision \{\{ GIT_COMMIT \}\}$",
                     SINGULARITY, re.M)
    assert re.search(r"^ARG GIT_COMMIT=unknown$", DOCKERFILE, re.M)
    assert "org.opencontainers.image.revision=$GIT_COMMIT" in DOCKERFILE


def test_docker_commit_label_comes_after_every_build_step():
    """An ARG used by a later RUN invalidates the cache from that point on every
    new commit, so the label must follow the last RUN."""
    assert DOCKERFILE.index("ARG GIT_COMMIT") > DOCKERFILE.rindex("\nRUN ")


# The command that fails the build when a file under the app's directories is
# not world-readable (or a directory not world-traversable).
UNREADABLE_CHECK = re.compile(
    r"find /opt(?: /app)? \\\( -type f ! -perm -o=r \\\) -o \\\( -type d ! -perm -o=rx \\\)"
)


def test_singularity_makes_atlas_data_world_readable():
    """The OSF atlas archives unpack as mode 600 and a SIF keeps the builder's
    uid, so without this chmod every other account fails at the first atlas
    read under --method fusion (reported in ReproNim/ants-nidm_bidsapp#7). It
    must follow the last unzip into /opt/data, or that archive keeps its 600s.
    """
    statements = _post_statements()
    unzips = [i for i, s in enumerate(statements)
              if s.startswith("unzip ") and "OASIS" in s]
    chmods = [i for i, s in enumerate(statements) if s == "chmod -R a+rX /opt/data"]
    assert unzips, "no OASIS unzip found in %post"
    assert chmods, "%post must run `chmod -R a+rX /opt/data`"
    assert chmods[-1] > unzips[-1], (
        "`chmod -R a+rX /opt/data` must come after the last OASIS unzip"
    )


@pytest.mark.parametrize("text,name", BUILD_FILES, ids=BUILD_IDS)
def test_build_fails_on_files_other_users_cannot_read(text, name):
    """A permission bug does not show up for the builder, who owns the files,
    so the build itself has to check -- after every step that writes files."""
    commands = _commands(text)
    checks = [i for i, c in enumerate(commands) if UNREADABLE_CHECK.search(c)]
    assert checks, f"{name} must fail the build on files other users cannot read"
    writers = [i for i, c in enumerate(commands)
               if re.search(r"\b(pip3? install|python3 -m pip install|unzip|wget|COPY)\b", c)]
    assert checks[-1] > writers[-1], (
        f"{name}: the readability check must come after the last step that "
        f"writes files; last writer is {commands[writers[-1]]!r}"
    )
    # It must be able to fail: in Singularity that relies on `set -e` plus an
    # explicit exit, in the Dockerfile on the RUN's exit status.
    tail = "\n".join(commands[checks[-1]:checks[-1] + 6])
    assert "exit 1" in tail, f"{name}: the readability check must exit non-zero"


# The converter is installed with --no-deps (requirements.txt alone decides the
# image's packages), so nothing at install time checks that its runtime
# imports are present; a missing one would surface only when the first
# subject's conversion runs, after segmentation. The build therefore imports
# the module the app runs, not only its PyNIDM shim.
CONVERTER_IMPORT_CHECK = (
    r'python3 -c "[^"\n]*from ants_seg_to_nidm\.ants_seg_to_nidm import main[^"\n]*"'
)


@pytest.mark.parametrize("text, name", BUILD_FILES, ids=BUILD_IDS)
def test_build_imports_the_converter_module_the_app_runs(text, name):
    install = re.search(r"pip3? install --no-deps -e (?:src/ants_seg_to_nidm|\.)", text)
    assert install, f"{name}: converter install not found"
    check = re.search(CONVERTER_IMPORT_CHECK, text)
    assert check, f"{name}: no build step imports ants_seg_to_nidm.ants_seg_to_nidm"
    assert check.start() > install.start(), f"{name}: the import check runs before the install"


def test_no_text_says_the_converter_declares_pynidm_4_5():
    """The converter's setup.py declares pynidm>=4.5.0,<6 since d72e752; the
    build files and docs once gave its pinning 4.5.0 as the reason for
    --no-deps."""
    for path in ("Dockerfile", "Singularity", "README.md", "requirements.txt"):
        text = " ".join((ROOT / path).read_text().split())
        for claim in (r"declares\s+pynidm==4\.5\.0", r"still\s+names?\s+(PyNIDM\s+)?4\.5\.0",
                      r"metadata still names"):
            assert not re.search(claim, text, re.IGNORECASE), (path, claim)


def _singularity_section(name):
    """Non-blank, non-comment lines of a Singularity section, stripped."""
    head = re.search(rf"^%{name}\s*$", SINGULARITY, re.MULTILINE)
    assert head, f"Singularity has no %{name} section"
    rest = SINGULARITY[head.end():]
    nxt = re.search(r"^%\w+\s*$", rest, re.MULTILINE)
    return _commands(rest[: nxt.start()] if nxt else rest)


CONSOLE_SCRIPT = 'exec ants-nidm "$@"'


def test_runscript_starts_the_console_script():
    """`apptainer run` (how BABS starts the app) must go through the console
    script. `python3 /opt/src/run.py` puts /opt/src first on sys.path, where
    the converter checkout shadows the installed package and the NIDM input
    check fails to import for every subject that has an input."""
    runscript = _singularity_section("runscript")
    assert runscript == [CONSOLE_SCRIPT], runscript
    assert not any(re.search(r"\brun\.py\b", line) for line in runscript)


def test_docker_entrypoint_starts_the_same_console_script():
    assert re.search(r"""echo '#!/bin/bash\\nexec ants-nidm "\$@"' > /entrypoint\.sh""",
                     DOCKERFILE)
    assert 'ENTRYPOINT ["/entrypoint.sh"]' in DOCKERFILE


def test_console_script_is_declared():
    setup = (ROOT / "setup.py").read_text()
    assert re.search(r"""["']ants-nidm\s*=\s*src\.run:main["']""", setup)


def test_build_checks_the_lookup_import_as_the_console_script_resolves_it():
    """The build must fail if the console script is missing or if, with its
    directory first on sys.path, the converter lookup does not import."""
    post = _post_statements()
    app_install = [i for i, s in enumerate(post) if s == "python3 -m pip install -e ."]
    assert app_install, "app install not found in %post"
    exists = [i for i, s in enumerate(post) if s == "command -v ants-nidm"]
    check = [i for i, s in enumerate(post)
             if "sys.path[0] = os.path.dirname(shutil.which('ants-nidm'))" in s
             and "from ants_seg_to_nidm.ants_seg_to_nidm import SubjectLookupError, find_subject" in s]
    assert exists and check, "the console-script checks are missing from %post"
    assert min(exists[0], check[0]) > app_install[-1]


def _pythonpath_after(assignment, host_value):
    """PYTHONPATH after `assignment` runs in sh, starting from host_value
    (None = unset). %environment is sourced by a POSIX shell."""
    import subprocess
    prelude = "unset PYTHONPATH" if host_value is None else f"export PYTHONPATH='{host_value}'"
    out = subprocess.run(["sh", "-c", f'{prelude}; {assignment}; printf %s "$PYTHONPATH"'],
                         capture_output=True, text=True, check=True)
    return out.stdout


@pytest.mark.parametrize("host_value", [None, "", "/host/lib"])
def test_environment_pythonpath_never_adds_the_working_directory(host_value):
    """An empty PYTHONPATH entry is the working directory to Python. With no
    host PYTHONPATH (a host that sets none, or a run with --cleanenv),
    `/opt:$PYTHONPATH` becomes `/opt:`, and a converter checkout in the
    working directory then shadows the installed converter."""
    lines = [s for s in _singularity_section("environment") if "PYTHONPATH" in s]
    assert lines == ["export PYTHONPATH=/opt${PYTHONPATH:+:$PYTHONPATH}"], lines
    entries = _pythonpath_after(lines[0], host_value).split(":")
    assert "" not in entries, entries
    assert entries[0] == "/opt"
    if host_value:
        assert entries[1:] == [host_value]


@pytest.mark.parametrize("assignment", [
    "export PYTHONPATH=/opt:$PYTHONPATH", "export PYTHONPATH=$PYTHONPATH:/opt"])
def test_pythonpath_check_rejects_a_trailing_empty_entry(assignment):
    """Positive control for the test above: the old form does leave an empty
    entry when the host has no PYTHONPATH."""
    assert "" in _pythonpath_after(assignment, None).split(":")


def test_dockerfile_does_not_extend_pythonpath():
    """Docker has no host PYTHONPATH, so `ENV PYTHONPATH=/opt:$PYTHONPATH`
    would also end in an empty entry. The image does not need one at all."""
    env = [c for c in _commands(DOCKERFILE) if re.match(r"ENV\s+PYTHONPATH\b", c)]
    assert all(re.fullmatch(r"ENV\s+PYTHONPATH=/opt", c) for c in env), env


CONVERTER_FILE = "/opt/src/ants_seg_to_nidm/ants_seg_to_nidm/ants_seg_to_nidm.py"
CONVERTER_PACKAGE_FILE = "/opt/src/ants_seg_to_nidm/ants_seg_to_nidm/__init__.py"



def _lookup_check():
    post = _post_statements()
    check = [s for s in post
             if "sys.path[0] = os.path.dirname(shutil.which('ants-nidm'))" in s]
    assert len(check) == 1, check
    return check[0]


def test_lookup_check_asserts_the_image_s_own_converter_file():
    """Importing the lookup proves only that some copy of the converter is
    importable; the check must name the file the image installed, and that
    the package is the real one, not a namespace package rooted at a checkout."""
    check = _lookup_check()
    assert f"m.__file__ != '{CONVERTER_FILE}'" in check, check
    assert f"p.__file__ != '{CONVERTER_PACKAGE_FILE}'" in check, check
    assert "sys.exit(" in check, check


def test_lookup_check_runs_with_the_runtime_pythonpath():
    """At run time sys.path is the console script's dir, then PYTHONPATH as
    %environment sets it with no host value, which is /opt."""
    assert _lookup_check().startswith('PYTHONPATH=/opt python3 -c "'), _lookup_check()




# Every python check a build relies on. Each must be able to fail the build.
BUILD_CHECKS = [
    ("Dockerfile", "fused_ops", FUSED_OPS_ASSERTION_CMD),
    ("Dockerfile", "converter import", CONVERTER_IMPORT_CHECK),
    ("Singularity", "fused_ops", FUSED_OPS_ASSERTION_CMD),
    ("Singularity", "converter import", CONVERTER_IMPORT_CHECK),
    ("Singularity", "lookup", re.escape("sys.path[0] = os.path.dirname(shutil.which('ants-nidm'))")),
]
PYTHON_C = r'python3 -c "(?:[^"\\]|\\.)*"'
# The whole command of a check: the python call, optionally with the runtime
# PYTHONPATH in front, and nothing else.
CHECK_COMMAND = re.compile(r'(PYTHONPATH=/opt )?python3 -c "(?:[^"\\]|\\.)*"')
CHECK_NAME = "python3"


def _check_unit(name, pattern):
    """The text of the instruction or %post command line holding a check,
    continuation lines joined."""
    units = _join_continuations(DOCKERFILE) if name == "Dockerfile" else _post_statements()
    found = [u for u in units if re.search(pattern, u)]
    assert len(found) == 1, (name, pattern, found)
    return found[0]


# bashlex 0.18 looks for a quoted heredoc delimiter (<<'EOF', <<"EOF", <<\EOF)
# with its quotes still on, so it never finds the end of the body. Quoting the
# delimiter only stops expansion inside the body, which bashlex does not parse
# either way, so the delimiter is unquoted before parsing. The replacement has
# the same length, so node positions still index the original text.
_QUOTED_HEREDOC = re.compile(r"""(?<![\\<])<<(-?)([ \t]*)(?:'(\w+)'|"(\w+)"|\\(\w+))""")


def _unquote_heredoc_delimiters(src):
    def same_length(m):
        plain = f"<<{m.group(1)}{m.group(2)} {m.group(3) or m.group(4) or m.group(5)}"
        return plain.ljust(len(m.group(0)))
    return _QUOTED_HEREDOC.sub(same_length, src)


def _parse_shell(src, where):
    """(bashlex's top-level nodes for src, None), or (None, why it failed).

    bashlex does not implement all of sh (`case` and `time` raise
    NotImplementedError), so any exception means the structure around a check
    cannot be verified, and the check is rejected."""
    try:
        return bashlex.parse(_unquote_heredoc_delimiters(src)), None
    except Exception as exc:  # ParsingError, NotImplementedError, AttributeError
        return None, (f"bashlex cannot parse {where} ({type(exc).__name__}: {exc}), so the "
                      "shell structure around the check cannot be verified")


def _children(node):
    for attr in ("parts", "list"):
        yield from getattr(node, attr, None) or []
    for attr in ("command", "heredoc", "output"):
        child = getattr(node, attr, None)
        if isinstance(child, bashlex.ast.node):
            yield child


def _span(node):
    """Source range of node and everything in it (a command's own `pos`
    leaves out its heredoc bodies)."""
    lo, hi = node.pos
    for child in _children(node):
        c_lo, c_hi = _span(child)
        lo, hi = min(lo, c_lo), max(hi, c_hi)
    return lo, hi


def _contains(node, span):
    lo, hi = _span(node)
    return lo <= span[0] and span[1] <= hi


def _statements(nodes):
    """The script's top-level statements as (parts, backgrounded): parts is
    one node, or the operands and `&&`/`||` operators of an and-or list."""
    out = []
    for node in nodes:
        if node.kind != "list":
            out.append(([node], False))
            continue
        seg = []
        for part in node.parts:
            if part.kind == "operator" and part.op in (";", "&", "\n"):
                if seg:
                    out.append((seg, part.op == "&"))
                seg = []
            else:
                seg.append(part)
        if seg:
            out.append((seg, False))
    return out


def _describe(node):
    """A compound, pipeline or function node, in words."""
    if node.kind == "pipeline":
        first = node.parts[0]
        return ("a `!` negation" if first.kind == "reservedword" and first.word == "!"
                else "a `|` pipeline")
    if node.kind == "function":
        return f"the body of function `{node.name.word}`, which runs only if called"
    if node.kind == "compound":
        first = node.list[0]
        word = first.word if first.kind == "reservedword" else first.parts[0].word
        return {"{": "a `{ ... }` group", "(": "a `( ... )` subshell"}.get(word, f"an `{word}` compound")
    return f"a bashlex {node.kind} node"


def _wrapper(node, span):
    """The heredoc body or command substitution around span inside node."""
    for child in _children(node):
        if _contains(child, span):
            if child.kind == "heredoc":
                return "a heredoc body"
            if child.kind in ("commandsubstitution", "processsubstitution"):
                return "a command substitution"
            found = _wrapper(child, span)
            if found:
                return found
    return None


def _check_command_problem(node, span, src):
    """Why the simple command `node`, which contains the check, is not the
    check alone, or None."""
    text = src[node.pos[0]:node.pos[1]]
    inner = _wrapper(node, span)
    if inner:
        return f"the check sits in {inner} of `{text[:60]}`, not in a command of its own"
    if re.search(r"python3\s+-O", text):
        return "the check runs under `python3 -O`"
    if not CHECK_COMMAND.fullmatch(text):
        return f"the check shares its command with other words: `{text[:80]}`"
    return None


def _placement_problem(parts, backgrounded, span, src):
    """Why the top-level statement (parts) holding the check does not let a
    failing check fail the script, or None."""
    if len(parts) > 1:
        ops = " ".join(sorted({f"`{p.op}`" for p in parts if p.kind == "operator"}))
        return f"the check is an operand of {ops}, which can skip it or ignore its status"
    node = parts[0]
    if backgrounded:
        return "the check runs in the background (`&`), so its status is never seen"
    if node.kind == "command":
        return _check_command_problem(node, span, src)
    return f"the check is inside {_describe(node)}"


_SET_FLAGS = re.compile(r"-[euxv]+")
_SET_LONG = {"errexit", "nounset", "xtrace", "verbose", "pipefail"}
_EXIT_STATUS = re.compile(r"[1-9]\d?|1\d\d|2[0-4]\d|25[0-5]")


def _literal(word):
    """A word with no expansion in it ($x, $(...), backticks)."""
    return word.kind == "word" and not getattr(word, "parts", None)


def _set_problem(args, text):
    i = 0
    while i < len(args):
        a = args[i]
        if _literal(a) and _SET_FLAGS.fullmatch(a.word):
            i += 1
        elif (a.word == "-o" and i + 1 < len(args)
              and _literal(args[i + 1]) and args[i + 1].word in _SET_LONG):
            i += 2
        else:
            return (f"`{text}` may turn errexit off (allowed: `set` with -e -u -x -v or "
                    "-o errexit|nounset|xtrace|verbose|pipefail)")
    return None


def _simple_command_problem(cmd, src, in_function):
    """Why the simple command cmd, run before a check, could end the script
    with status 0, override its status or turn errexit off; None if it cannot."""
    words = [p for p in cmd.parts if p.kind == "word"]  # drops assignments, redirections
    if not words:
        return None
    text = src[cmd.pos[0]:cmd.pos[1]]
    if not _literal(words[0]):
        return f"the command name in `{text}` is an expansion, which could be `exit` or `eval`"
    name, args = words[0].word, words[1:]
    while name in ("command", "builtin"):
        opts = []
        while args and _literal(args[0]) and args[0].word.startswith("-"):
            opts.append(args.pop(0).word)
        if name == "command" and any(set(o) & {"v", "V"} for o in opts):
            return None  # a lookup, runs nothing
        if not args:
            return None
        if not _literal(args[0]):
            return f"the command name in `{text}` is an expansion, which could be `exit` or `eval`"
        name, args = args[0].word, args[1:]
    if name == "exit":
        if len(args) == 1 and _literal(args[0]) and _EXIT_STATUS.fullmatch(args[0].word):
            return None
        return f"`{text}` can end the script with status 0 (only a literal `exit 1`..`exit 255` is allowed)"
    if name == "return":
        return None if in_function else f"`{text}` outside a function can end the script early"
    if name == "exec":
        return f"`{text}` replaces the shell, so nothing after it runs" if args else None
    if name in ("eval", ".", "source"):
        return f"`{text}` runs shell code the parser cannot see"
    if name == "trap":
        return f"`{text}` can replace the script's exit status"
    if name == "alias":
        return f"`{text}` can redefine the check's command"
    if name == "set":
        return _set_problem(args, text)
    return None


def _early_end(node, src, in_function=False):
    """The first construct in node (any depth, function bodies and command
    substitutions included) that _simple_command_problem rejects, or a
    function definition that shadows the check's command."""
    if node.kind == "function":
        if node.name.word == CHECK_NAME:
            return f"function `{CHECK_NAME}` shadows the check's command"
        in_function = True
    if node.kind == "command":
        problem = _simple_command_problem(node, src, in_function)
        if problem:
            return problem
    for child in _children(node):
        problem = _early_end(child, src, in_function)
        if problem:
            return problem
    return None


def _turns_errexit_on(parts):
    if len(parts) != 1 or parts[0].kind != "command":
        return False
    words = [p for p in parts[0].parts if p.kind == "word"]
    if not words or words[0].word != "set":
        return False
    args = [w.word for w in words[1:] if _literal(w)]
    return any(re.fullmatch(r"-[euxv]*e[euxv]*", a) for a in args) or any(
        a == "-o" and b == "errexit" for a, b in zip(args, args[1:]))


def _post_check_problem(body, pattern):
    """Why the check matching `pattern` in a %post body may not fail the
    build, or None. See the module docstring for the rule."""
    nodes, problem = _parse_shell(body, "%post")
    if problem:
        return problem
    statements = _statements(nodes)
    hits = [(m.span(), i) for m in re.finditer(pattern, body)
            for i, (parts, _) in enumerate(statements) if any(_contains(p, m.span()) for p in parts)]
    if len(hits) != 1:
        return f"the check is in {len(hits)} commands of %post, not exactly one"
    span, at = hits[0]
    problem = _placement_problem(*statements[at], span, body)
    if problem:
        return problem
    if not any(_turns_errexit_on(parts) for parts, _ in statements[:at]):
        return "errexit is not on at the check: %post needs a top-level `set -e` before it"
    for parts, _ in statements[:at]:
        for node in parts:
            problem = _early_end(node, body)
            if problem:
                return problem
    return None


def _run_check_problem(run, pattern):
    """Why the check matching `pattern` in a Dockerfile RUN instruction
    (continuations joined) may not fail that instruction, or None. The shell
    runs without errexit, so the instruction must be the check alone or a
    `&&` chain of simple commands that includes it, with nothing before it
    that _early_end rejects."""
    if not run.startswith("RUN "):
        return f"not a RUN instruction: {run[:40]!r}"
    shell = run[len("RUN "):]
    if shell.lstrip().startswith("["):
        return "an exec-form RUN has no shell to check"
    nodes, problem = _parse_shell(shell, "the RUN instruction")
    if problem:
        return problem
    if len(nodes) != 1:
        return "the RUN instruction holds more than one command line"
    node = nodes[0]
    if node.kind not in ("command", "list"):
        return f"the RUN instruction is {_describe(node)}, not a `&&` chain of commands"
    operands = [node] if node.kind == "command" else node.parts
    other_ops = sorted({p.op for p in operands if p.kind == "operator"} - {"&&"})
    if other_ops:
        return (f"the RUN instruction joins commands with {' '.join(f'`{o}`' for o in other_ops)}; "
                "only an `&&` chain fails when the check fails")
    commands = [p for p in operands if p.kind != "operator"]
    for c in commands:
        if c.kind != "command":
            return f"the RUN chain contains {_describe(c)}"
        if any(ch.kind == "heredoc" for ch in _walk(c)):
            return "the RUN instruction has a heredoc"
    hits = [(m.span(), i) for m in re.finditer(pattern, shell)
            for i, c in enumerate(commands) if _contains(c, m.span())]
    if len(hits) != 1:
        return f"the check is in {len(hits)} commands of the RUN instruction, not exactly one"
    span, at = hits[0]
    problem = _check_command_problem(commands[at], span, shell)
    if problem:
        return problem
    for c in commands[:at]:
        problem = _early_end(c, shell)
        if problem:
            return problem
    return None


def _walk(node):
    yield node
    for child in _children(node):
        yield from _walk(child)


def _why_cannot_fail(text, name, pattern):
    """Why the check matching `pattern` in build file `text` may not fail the
    build, or None when a failing check fails it."""
    if name == "Dockerfile":
        runs = [u for u in _join_continuations(text) if re.search(pattern, u)]
        if len(runs) != 1:
            return f"the check is in {len(runs)} Dockerfile instructions, not exactly one"
        return _run_check_problem(runs[0], pattern)
    return _post_check_problem(_post_body(text), pattern)


def test_build_files_parse_with_bashlex():
    """The checks below fail closed when bashlex cannot parse a recipe, so
    the recipes as they stand must parse."""
    assert _parse_shell(_post_body(), "%post")[1] is None
    for unit in _join_continuations(DOCKERFILE):
        if unit.startswith("RUN "):
            assert _parse_shell(unit[4:], unit[:40])[1] is None, unit


@pytest.mark.parametrize("name, check, pattern", BUILD_CHECKS,
                         ids=[f"{n}-{c}" for n, c, _ in BUILD_CHECKS])
def test_build_check_can_fail_the_build_as_written(name, check, pattern):
    text = DOCKERFILE if name == "Dockerfile" else SINGULARITY
    reason = _why_cannot_fail(text, name, pattern)
    assert reason is None, reason


# Text put around a check in the recipe: (prefix, suffix, %post reason, RUN
# reason), a reason being a regex the rejection must match, None meaning the
# check still fails the build. Under `set -e`, `check; true` stops %post at a
# failing check; a RUN shell has no errexit, so there `;` loses the status.
AFFIXES = [
    ("", "", None, None),
    ("", " || true", "`\\|\\|`", "`\\|\\|`"),
    ("", " || :", "`\\|\\|`", "`\\|\\|`"),
    ("", " || exit 0", "`\\|\\|`", "`\\|\\|`"),
    ("", "; true", None, "`;`"),
    ("", " || echo skipped", "`\\|\\|`", "`\\|\\|`"),
    ("", " &", "background", "`&`|cannot parse"),
    ("", " | cat", "pipeline", "pipeline"),
    ("", " \\\n    || true", "`\\|\\|`", "`\\|\\|`"),
    # after PYTHONPATH=/opt, `!` and `if` are plain words, not keywords
    ("! ", "", "`!` negation|shares its command", "`!` negation"),
    ("if ", "", "cannot parse|shares its command", "cannot parse"),
    ("true || ", "", "`\\|\\|`", "`\\|\\|`"),
]


@pytest.mark.parametrize("name, check, pattern", BUILD_CHECKS,
                         ids=[f"{n}-{c}" for n, c, _ in BUILD_CHECKS])
@pytest.mark.parametrize("prefix, suffix, post_reason, run_reason", AFFIXES,
                         ids=[repr(p + "CHECK" + s) for p, s, _, _ in AFFIXES])
def test_build_checks_can_fail_the_build(name, check, pattern, prefix, suffix,
                                         post_reason, run_reason):
    """Nothing around a python check may swallow its exit status. The affixes
    are applied to the recipe text and the whole recipe is re-parsed."""
    text = DOCKERFILE if name == "Dockerfile" else SINGULARITY
    code = list(re.finditer(PYTHON_C, _check_unit(name, pattern)))
    assert len(code) == 1, (name, check)
    code = code[0].group(0)
    assert text.count(code) == 1, (name, check)
    s = text.index(code)
    mutated = text[:s] + prefix + code + suffix + text[s + len(code):]
    reason = _why_cannot_fail(mutated, name, re.escape(code))
    expected = run_reason if name == "Dockerfile" else post_reason
    if expected is None:
        assert reason is None, reason
    else:
        assert reason is not None and re.search(expected, reason), reason


@pytest.mark.parametrize("name, check, pattern", BUILD_CHECKS,
                         ids=[f"{n}-{c}" for n, c, _ in BUILD_CHECKS])
def test_build_checks_do_not_rely_on_assert(name, check, pattern):
    """python3 -O and PYTHONOPTIMIZE drop asserts; a check exits explicitly."""
    code = re.search(PYTHON_C, _check_unit(name, pattern)).group(0)
    assert not re.search(r"\bassert\b", code), code


@pytest.mark.parametrize("name, check, pattern", BUILD_CHECKS,
                         ids=[f"{n}-{c}" for n, c, _ in BUILD_CHECKS])
def test_build_checks_reject_python_optimize(name, check, pattern):
    text = DOCKERFILE if name == "Dockerfile" else SINGULARITY
    code = re.search(PYTHON_C, _check_unit(name, pattern)).group(0)
    optimized = code.replace("python3 -c", "python3 -O -c", 1)
    reason = _why_cannot_fail(text.replace(code, optimized), name, re.escape(optimized))
    assert reason and "runs under `python3 -O`" in reason, reason


@pytest.mark.parametrize("off", ["set +e", "set +o errexit", "set +eu", "set -x +e",
                                 "set -o pipefail +o errexit", 'set "+e"', "set '+e'",
                                 "set +\\e", "set $opts", "set -n", "set --"])
def test_post_errexit_cannot_be_turned_off_later(off):
    """A later `set +e` would make every %post check after it a no-op, and
    `set -n` stops the shell running anything at all."""
    body = _post_body()
    mutated = body.replace("set -e\n", f"set -e\n    {off}\n", 1)
    assert mutated != body
    reason = _post_check_problem(mutated, CONVERTER_IMPORT_CHECK)
    assert reason and "may turn errexit off" in reason, reason


def test_post_needs_set_e_at_top_level_before_the_check():
    body = _post_body()
    for mutated in (body.replace("set -e\n", "\n", 1),
                    body.replace("set -e\n", "if true; then set -e; fi\n", 1)):
        assert mutated != body
        reason = _post_check_problem(mutated, CONVERTER_IMPORT_CHECK)
        assert reason and "errexit is not on" in reason, reason


PYNIDM_PIN = re.search(r"^pynidm==([\d.]+)\s*$", REQUIREMENTS, re.MULTILINE).group(1)


@pytest.mark.parametrize("text, name", BUILD_FILES, ids=BUILD_IDS)
def test_converter_import_check_asserts_the_pinned_pynidm(text, name):
    """The build fails unless the installed PyNIDM is requirements.txt's pin
    and the converter took its API for that major version."""
    check = re.search(CONVERTER_IMPORT_CHECK, text).group(0)
    major = PYNIDM_PIN.split(".")[0]
    assert (f"(PYNIDM_MAJOR != {major} or v != '{PYNIDM_PIN}') and sys.exit(" in check
            and f"not the pinned {PYNIDM_PIN}'" in check), check


def test_ci_installs_the_pinned_pynidm():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert set(re.findall(r"pynidm==([\d.]+)", ci)) == {PYNIDM_PIN}


def test_ci_installs_bashlex_as_pinned_for_tests():
    """These tests need bashlex; CI must install the same pin as
    requirements-test.txt, and the app itself must not depend on it."""
    pin = re.search(r"^bashlex==([\d.]+)\s*$", (ROOT / "requirements-test.txt").read_text(), re.M)
    assert pin, "requirements-test.txt must pin bashlex"
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert set(re.findall(r"bashlex==([\d.]+)", ci)) == {pin.group(1)}
    assert "bashlex" not in REQUIREMENTS
    assert "bashlex" not in (ROOT / "setup.py").read_text()


# Shell structure around a check, in synthetic %post bodies. The check line is
# unchanged in every case; what surrounds it decides whether a failing check
# fails the build.
CHECK = 'python3 -c "import sys; sys.exit(0)"'

# name: (lines placed where the check goes, CHECK being the check itself;
# regex the rejection reason must match). The rest of the body is `set -e`,
# an echo before and an echo after.
POST_BYPASSES = {
    # the check runs and its failure is ignored
    "if-condition": (["if", CHECK, "then :; fi"], "`if` compound"),
    "subshell-or-true": (["(", CHECK, ") || true"], "`\\|\\|`"),
    "group-into-tee": (["{", CHECK, "} 2>&1 | tee /tmp/post.log"], "`\\|` pipeline"),
    "child-shell-heredoc": (["sh <<'EOF'", CHECK, "true", "\nEOF"], "heredoc body"),
    "and-newline": (["false &&", CHECK], "`&&`"),
    "or-newline": (["true ||", CHECK], "`\\|\\|`"),
    "pipe-newline-after": ([CHECK + " |", "cat"], "`\\|` pipeline"),
    "negated": (["! " + CHECK], "`!` negation"),
    "backgrounded": ([CHECK + " &"], "background"),
    "in-command-substitution": (['export out="$(' + CHECK + ')"'], "command substitution"),
    "child-shell-c": (["sh -c '" + CHECK + "; true'"], "shares its command"),
    # the check never runs
    "heredoc-comment": ([": <<'EOF'", CHECK, "\nEOF"], "heredoc body"),
    "heredoc-comment-unquoted": ([": <<EOF", CHECK, "\nEOF"], "heredoc body"),
    "uncalled-function": (["check() {", CHECK, "}"], "function `check`"),
    "uncalled-function-keyword": (["function check {", CHECK, "}"], "function `check`"),
    "if-false": (["if false; then", CHECK, "fi"], "`if` compound"),
    "while-false": (["while false; do", CHECK, "done"], "`while` compound"),
    "until-true": (["until true; do", CHECK, "done"], "`until` compound"),
    "for-empty": (["for x in; do", CHECK, "done"], "`for` compound"),
    "multiline-dq-offset": (['if false; then', '    echo "', 'fi', '"', CHECK, 'echo "',
                             'if true; then', '"', 'fi'], "`if` compound"),
    "multiline-sq-offset": (["if false; then", "    echo '", "fi", "'", CHECK, "echo '",
                             "if true; then", "'", "fi"], "`if` compound"),
    "commented-out": (["# " + CHECK], "in 0 commands"),
    "exit-0-before": (["exit 0", CHECK], "`exit 0`"),
    "exit-bare-before": (["exit", CHECK], "`exit`"),
    "exit-256": (["if true; then exit 256; fi", CHECK], "`exit 256`"),
    "exit-dollar-q": (["if true; then exit $?; fi", CHECK], "`exit \\$\\?`"),
    "exit-quoted-zero": (['exit "0"', CHECK], "exit"),
    "exec-before": (["exec true", CHECK], "`exec true`"),
    "exec-redirect-and-command": (["exec >/tmp/post.log true", CHECK], "replaces the shell"),
    # POSIX leaves `return` outside a function unspecified; bash fails on it
    "return-before": (["return 0", CHECK], "outside a function"),
    "return-in-if": (["if true; then return 0; fi", CHECK], "outside a function"),
    "conditional-exit-before": (["true && exit 0", CHECK], "`exit 0`"),
    "exit-0-in-if-before": (["if true; then exit 0; fi", CHECK], "`exit 0`"),
    "exit-0-in-group-before": (["{ exit 0; }", CHECK], "`exit 0`"),
    "exit-in-called-function": (["f() { exit 0; }", "f", CHECK], "`exit 0`"),
    "command-exit": (["command exit 0", CHECK], "`command exit 0`"),
    "command-p-exit": (["command -p exit 0", CHECK], "exit 0"),
    "builtin-exit": (["builtin exit 0", CHECK], "`builtin exit 0`"),
    "eval-exit": (["eval 'exit 0'", CHECK], "`eval"),
    "redirect-prefixed-exit": ([">/dev/null exit 0", CHECK], "exit 0"),
    "assignment-prefixed-exit": (["X=1 exit 0", CHECK], "exit 0"),
    "backslash-exit": (["\\exit 0", CHECK], "exit 0"),
    "quoted-exit": (["'exit' 0", CHECK], "exit' 0"),
    "expansion-command-name": (["$EXIT 0", CHECK], "is an expansion"),
    "trap-exit": (["trap 'exit 0' EXIT", CHECK], "`trap"),
    "source": (["source /tmp/x.sh", CHECK], "`source"),
    "dot": ([". /tmp/x.sh", CHECK], "`\\. "),
    "alias": (["alias python3=true", CHECK], "`alias"),
    "function-shadows-python3": (["python3() { :; }", CHECK], "shadows"),
    "comment-ends-in-backslash": (["echo hi # trailing \\", "exit 0", CHECK], "`exit 0`"),
    "backslash-space": (["echo hi \\ ", "exit 0", CHECK], "`exit 0`"),
    # bashlex cannot read these, so the check is rejected rather than trusted
    "case-no-match": (["case x in", "y)", CHECK, ";;", "esac"], "cannot parse"),
    "unclosed-if": (["if true; then", CHECK], "cannot parse"),
    "unclosed-heredoc": ([": <<'EOF'", CHECK], "cannot parse"),
    "indented-heredoc-end": ([": <<'EOF'", "x", "    EOF", CHECK], "cannot parse"),
    "stray-fi": (["fi", CHECK], "cannot parse"),
}

# Structure that does not stop a failing check from failing the build.
POST_HARMLESS = {
    "plain": [CHECK],
    "single-line-if-before": ["if [ -z \"$X\" ]; then echo none; fi", CHECK],
    "multi-line-if-before": ["if [ -n \"$X\" ]; then", "    echo set", "fi", CHECK],
    "if-with-error-exit-before": ["if [ -z \"$X\" ]; then", "    echo no >&2", "    exit 1", "fi", CHECK],
    "exit-255-before": ["if [ -z \"$X\" ]; then exit 255; fi", CHECK],
    "cd-or-exit-1": ["cd /opt || exit 1", CHECK],
    "for-loop-before": ["for x in a b; do echo $x; done", CHECK],
    "while-loop-before": ["while false; do echo x; done", CHECK],
    "group-before": ["{ echo a; echo b; } > /dev/null", "(cd /tmp && ls) > /dev/null", CHECK],
    "command-substitution-before": ['u="$(find /opt \\( -type f \\) -o \\( -type d \\))"', CHECK],
    "multi-line-command-substitution": ['x="$(', "  echo hi", ')"', CHECK],
    "heredoc-before": ["cat > /tmp/f <<EOF", "if (", "\nEOF", CHECK],
    "quoted-heredoc-with-set-plus-e": ["cat > /usr/local/bin/x.sh <<'EOF'", "set +e", "exit 0", "\nEOF", CHECK],
    "set-plus-e-in-quotes": ['echo "do not set +e here"', CHECK],
    "exit-in-quotes": ["echo 'exit 0'", CHECK],
    "message-line-starting-for": ['echo "Warning: the', 'for the best result"', CHECK],
    "message-line-starting-if": ['echo "Note:', 'if this fails, rebuild"', CHECK],
    "function-defined-and-called-before": ["f() { echo f; }", "f", CHECK],
    "helper-with-return": ["fetch() {", "    for i in 1 2 3; do", '        wget -q "$1" && return 0',
                           "        sleep 1", "    done", "    return 1", "}", "fetch https://example.org/x",
                           CHECK],
    "exec-redirect-only": ["exec >/tmp/post.log 2>&1", CHECK],
    "exec-fd-only": ["exec 3>&1", CHECK],
    "more-set-options": ["set -o pipefail", "set -eux", CHECK],
    "command-v": ["command -v python3", CHECK],
    "command-v-exit": ["command -v exit", CHECK],
    "background-before": ["sleep 0 &", CHECK],
    "semicolon-after": [CHECK + "; true"],
    "compound-after": [CHECK, "if true; then", "    exit 1", "fi"],
    "exit-after": [CHECK, "exit 0"],
    "trap-after": [CHECK, "trap 'exit 0' EXIT"],
}

# Safe forms the rule rejects anyway. Each is pinned so that a change to the
# rule that starts accepting it is a deliberate one.
POST_FALSE_FAILURES = {
    # bashlex 0.18 does not implement `case` or `time`
    "case-before": (["case $X in", "a) echo a ;;", "*) echo other ;;", "esac", CHECK], "cannot parse"),
    "time-before": (["time true", CHECK], "cannot parse"),
    # the status is a variable, so the rule cannot see that it is non-zero
    "exit-variable": (["rc=1", 'if [ "$rc" -ne 0 ]; then exit "$rc"; fi', CHECK], "exit"),
    # a command name that is an expansion could be `exit 0` (`retry exit 0`)
    "helper-running-its-arguments": (["retry() {", "    for i in 1 2 3; do", '        "$@" && return 0',
                                      "        sleep 1", "    done", "    return 1", "}", "retry true", CHECK],
                                     "is an expansion"),
    # the check must occur in exactly one command, even when a second copy
    # sits where it never runs
    "check-twice": ([CHECK, "if false; then", CHECK, "fi"], "in 2 commands"),
    # only the listed `set` options are allowed
    "set-f": (["set -f", CHECK], "may turn errexit off"),
    # errexit applies inside a group, and a pipeline's status is its last
    # command's, but the rule accepts only a top-level simple command
    "group-wrapped": (["{ " + CHECK + "; }"], "`\\{ \\.\\.\\. \\}` group"),
    "redirected-group": (["{ " + CHECK + "; } > /tmp/check.log"], "group"),
    "pipe-into-check": (["true |", CHECK], "`\\|` pipeline"),
    # a bare assignment's status is its command substitution's
    "assignment-from-check": (['out="$(' + CHECK + ')"'], "command substitution"),
}


def _post_body_with(lines):
    return "\n    ".join(["", "set -e", "echo before"] + lines + ["echo after", ""])


@pytest.mark.parametrize("lines, reason", POST_BYPASSES.values(), ids=POST_BYPASSES.keys())
def test_post_check_inside_other_shell_structure_cannot_pass(lines, reason):
    """Each form leaves the check line as it is and either ignores the
    check's failure or never runs it; the rejection names the construct."""
    got = _post_check_problem(_post_body_with(lines), re.escape(CHECK))
    assert got is not None and re.search(reason, got), got


@pytest.mark.parametrize("lines", POST_HARMLESS.values(), ids=POST_HARMLESS.keys())
def test_post_check_after_unrelated_shell_structure_passes(lines):
    reason = _post_check_problem(_post_body_with(lines), re.escape(CHECK))
    assert reason is None, reason


@pytest.mark.parametrize("lines, reason", POST_FALSE_FAILURES.values(), ids=POST_FALSE_FAILURES.keys())
def test_post_known_false_failures_are_still_rejected(lines, reason):
    got = _post_check_problem(_post_body_with(lines), re.escape(CHECK))
    assert got is not None and re.search(reason, got), got


DOCKER_BYPASSES = {
    "exit-0-before": ("RUN exit 0 && " + CHECK, "`exit 0`"),
    "exec-true-before": ("RUN exec true && " + CHECK, "`exec true`"),
    "exit-later-in-chain": ("RUN echo a && exit 0 && echo b && " + CHECK, "`exit 0`"),
    "exec-mid-chain": ("RUN echo a && exec sh -c true && " + CHECK, "replaces the shell"),
    "trap-exit": ("RUN trap 'exit 0' EXIT && " + CHECK, "`trap"),
    "eval-exit": ("RUN eval 'exit 0' && " + CHECK, "`eval"),
    "command-exit": ("RUN command exit 0 && " + CHECK, "exit 0"),
    "exit-256": ("RUN exit 256 && " + CHECK, "`exit 256`"),
    "set-n": ("RUN set -n && " + CHECK, "may turn errexit off"),
    "semicolon-after": ("RUN " + CHECK + "; true", "`;`"),
    "or": ("RUN true || " + CHECK, "`\\|\\|`"),
    "or-after": ("RUN " + CHECK + " || true", "`\\|\\|`"),
    "pipe": ("RUN " + CHECK + " | cat", "pipeline"),
    "negated": ("RUN ! " + CHECK, "`!` negation"),
    "background": ("RUN " + CHECK + " &", "`&`"),
    "if-false": ("RUN if false; then " + CHECK + "; fi", "`if` compound"),
    "child-shell": ("RUN sh -c '" + CHECK + "; true'", "shares its command"),
}
DOCKER_HARMLESS = {
    "plain": "RUN " + CHECK,
    "after-commands": "RUN echo a && echo b && " + CHECK,
    "before-commands": "RUN echo a && " + CHECK + " && chmod -R 755 /app",
    "word-in-quotes": "RUN echo 'exit 0' && " + CHECK,
    "find-exec": "RUN find /opt/src -type f -exec chmod a+r {} + && " + CHECK,
    "exit-option": "RUN pytest --exitfirst --exec-timeout 1 && " + CHECK,
    "exit-after-check": "RUN " + CHECK + " && exit 0",
}


# Safe forms the RUN rule rejects anyway: the check's status is the
# instruction's, but the rule accepts only an `&&` chain of simple commands.
DOCKER_FALSE_FAILURES = {
    "semicolon-before-last-check": ("RUN true; " + CHECK, "`;`"),
    "subshell": ("RUN (" + CHECK + ")", "subshell"),
    "if-true": ("RUN if true; then " + CHECK + "; fi", "`if` compound"),
    "group-in-chain": ("RUN echo a && { " + CHECK + "; }", "group"),
}


@pytest.mark.parametrize("run, reason", list(DOCKER_BYPASSES.values()) + list(DOCKER_FALSE_FAILURES.values()),
                         ids=list(DOCKER_BYPASSES) + list(DOCKER_FALSE_FAILURES))
def test_dockerfile_run_that_can_skip_the_check_cannot_pass(run, reason):
    got = _run_check_problem(run, re.escape(CHECK))
    assert got is not None and re.search(reason, got), got


@pytest.mark.parametrize("run", DOCKER_HARMLESS.values(), ids=DOCKER_HARMLESS.keys())
def test_dockerfile_run_without_early_exit_passes(run):
    reason = _run_check_problem(run, re.escape(CHECK))
    assert reason is None, reason
