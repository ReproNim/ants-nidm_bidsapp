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


def _shell_outside_quotes(line):
    """The line with double-quoted strings removed: the shell part only."""
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)


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


def _check_unit(name, pattern):
    """The unit whose exit status decides whether the check fails the build:
    the whole RUN instruction in the Dockerfile, the %post command line in
    Singularity (which runs under `set -e`), continuation lines joined."""
    units = _join_continuations(DOCKERFILE) if name == "Dockerfile" else _post_statements()
    found = [u for u in units if re.search(pattern, u)]
    assert len(found) == 1, (name, pattern, found)
    return found[0]


def _shell_tokens(line):
    """Split one logical shell line into (kind, text) tokens, kind being
    'word', 'op' (`; ;; & && | || ( )`) or 'heredoc' (text is the delimiter,
    prefixed `-` for `<<-`). Quotes, `$( )` and backticks stay inside their
    word, so parentheses in them do not count as operators. Not a full shell
    grammar: enough to follow the POSIX compound commands."""
    tokens, cur, i, n = [], [], 0, len(line)

    def flush():
        if cur:
            tokens.append(("word", "".join(cur)))
            cur.clear()

    def skip_balanced(i):
        """Index after the `)` that closes the `(` at i, honouring quotes."""
        depth = 0
        while i < n:
            c = line[i]
            if c == "\\":
                i += 2
                continue
            if c in "'\"":
                j = line.find(c, i + 1)
                i = n if j < 0 else j + 1
                continue
            depth += (c == "(") - (c == ")")
            i += 1
            if depth == 0:
                break
        return i

    while i < n:
        c = line[i]
        if c in " \t":
            flush()
            i += 1
        elif c == "\\":
            cur.append(line[i:i + 2])
            i += 2
        elif c == "'":
            j = line.find("'", i + 1)
            j = n if j < 0 else j + 1
            cur.append(line[i:j])
            i = j
        elif c == '"':
            j = i + 1
            while j < n and line[j] != '"':
                if line[j] == "\\":
                    j += 2
                elif line.startswith("$(", j):
                    j = skip_balanced(j + 1)
                else:
                    j += 1
            cur.append(line[i:j + 1])
            i = j + 1
        elif c == "`":
            j = line.find("`", i + 1)
            j = n if j < 0 else j + 1
            cur.append(line[i:j])
            i = j
        elif c == "$" and line.startswith("$(", i):
            j = skip_balanced(i + 1)
            cur.append(line[i:j])
            i = j
        elif c == "#" and not cur:
            break
        elif c in ";&|()":
            flush()
            two = line[i:i + 2]
            if two in (";;", "&&", "||"):
                tokens.append(("op", two))
                i += 2
            else:
                tokens.append(("op", c))
                i += 1
        elif c in "<>":
            if line.startswith("<<", i) and not line.startswith("<<<", i):
                flush()
                i += 2
                dash = line.startswith("-", i)
                i += dash
                while i < n and line[i] in " \t":
                    i += 1
                j = i
                while j < n and line[j] not in " \t;&|()<>":
                    if line[j] in "'\"":
                        k = line.find(line[j], j + 1)
                        j = n if k < 0 else k
                    j += 1
                delim = re.sub(r"""['"\\]""", "", line[i:j])
                tokens.append(("heredoc", ("-" if dash else "") + delim))
                i = j
            else:
                # A redirection (`>`, `2>&1`, `>|`): its `&` is not a control
                # operator, so it stays in the word.
                cur.append(c)
                i += 1
                while i < n and line[i] in "<>&|":
                    cur.append(line[i])
                    i += 1
        else:
            cur.append(c)
            i += 1
    flush()
    return tokens


class _PostLine:
    """One `%post` command line: its text (as _join_continuations gives it),
    the compound-command depth it starts at, and every exit/exec/return in it
    as (word, depth, next word)."""

    def __init__(self, text, depth):
        self.text, self.depth, self.exits = text, depth, []


_OPENERS = {"if": "fi", "while": "done", "until": "done", "for": "done", "case": "esac"}
_CLOSERS = {"fi", "done", "esac"}
_LEADERS = {"then", "do", "else", "elif", "!", "time"}


def _scan_post(body):
    """Follow POSIX compound-command nesting over the command lines of a
    %post body: `if`/`fi`, `for`/`while`/`until` ... `done`, `case`/`esac`,
    `{`/`}`, `(`/`)` and function bodies (a function's `{` or `(` is a
    compound like any other). Heredoc bodies are skipped up to their
    delimiter, so a check inside one is never seen as a command line.
    Returns (lines, errors); errors lists unbalanced constructs and heredocs
    that never end, after which the scan cannot be trusted."""
    raw = body.splitlines()
    lines, errors, stack = [], [], []
    i = 0
    while i < len(raw):
        stripped = raw[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        parts = []
        while True:
            s = raw[i].strip()
            cont = s.endswith("\\")
            parts.append(s[:-1].rstrip() if cont else s)
            i += 1
            if not cont or i >= len(raw):
                break
        text = " ".join(parts)
        line = _PostLine(text, len(stack))
        lines.append(line)
        cmd_pos, fn_name, heredocs = True, False, []
        tokens = _shell_tokens(text)
        for idx, (kind, tok) in enumerate(tokens):
            if kind == "heredoc":
                heredocs.append(tok)
            elif kind == "op":
                if tok == "(":
                    stack.append("(")
                elif tok == ")":
                    if stack and stack[-1] == "(":
                        stack.pop()
                    elif not stack or stack[-1] != "case":
                        errors.append(f"unmatched `)` in {text!r}")
                cmd_pos = True
            elif fn_name:
                fn_name, cmd_pos = False, True
            elif cmd_pos and re.fullmatch(r"[A-Za-z_]\w*=.*", tok):
                pass
            elif cmd_pos and tok in _OPENERS:
                stack.append(tok)
                cmd_pos = tok in ("if", "while", "until")
            elif cmd_pos and tok in _LEADERS:
                pass
            elif cmd_pos and tok == "function":
                fn_name = True
            elif cmd_pos and tok == "{":
                stack.append("{")
            elif cmd_pos and tok == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
                else:
                    errors.append(f"unmatched `}}` in {text!r}")
                cmd_pos = False
            elif cmd_pos and tok in _CLOSERS:
                if stack and _OPENERS.get(stack[-1]) == tok:
                    stack.pop()
                else:
                    errors.append(f"unmatched `{tok}` in {text!r}")
                cmd_pos = False
            elif cmd_pos and tok in ("exit", "exec", "return"):
                nxt = tokens[idx + 1][1] if idx + 1 < len(tokens) else ""
                line.exits.append((tok, len(stack), nxt))
                cmd_pos = False
            else:
                cmd_pos = False
            if kind == "op" and tok in ("(", ";", ";;", "&", "&&", "|", "||", ")"):
                cmd_pos = True
        for spec in heredocs:
            delim = spec.lstrip("-")
            while i < len(raw):
                end = raw[i].lstrip("\t") if spec.startswith("-") else raw[i]
                i += 1
                if end == delim:
                    break
            else:
                errors.append(f"heredoc {delim!r} is never closed")
    if stack:
        errors.append(f"%post ends inside {stack[-1]!r}")
    return lines, errors


def _post_structure_problem(body, unit):
    """Why `unit` (a %post command line) may not run, or None. It must be a
    command line of its own at nesting depth 0 (not inside a group, `if`,
    function or heredoc body), and nothing before it may leave %post early: no
    `exit`, `exec` or `return` at depth 0, and none inside a compound unless it
    is `exit` with a non-zero literal status (the recipe's own error exits)."""
    lines, errors = _scan_post(body)
    if errors:
        return errors[0]
    at = [i for i, line in enumerate(lines) if line.text == unit]
    if len(at) != 1:
        return f"{unit[:40]!r} is not one command line of %post (inside a heredoc?)"
    if lines[at[0]].depth:
        return f"{unit[:40]!r} sits inside a compound command or function body"
    for line in lines[:at[0]]:
        for word, depth, arg in line.exits:
            if depth == 0 or not (word == "exit" and re.fullmatch(r"[1-9]\d*", arg)):
                return f"`{word}` in {line.text!r} can end %post before the check"
    return None


def _post_keeps_errexit(statements):
    """`set -e` first, and no later `set` turns it off (`set +e`, `set -x +e`,
    `set +o errexit`)."""
    return statements[:1] == ["set -e"] and not any(
        re.search(r"\bset\b[^;&|]*?\s(\+[A-Za-z]*e|\+o\s+errexit\b)", s) for s in statements)


def _can_fail_the_build(unit, name, post=None, body=None):
    """Whether a failing check fails the build.

    Singularity: the %post command line is the check alone, optionally with
    PYTHONPATH=/opt in front, and %post keeps `set -e`. Anything else on the
    line (`|| true`, `!`, `if ...`, a continuation) can swallow its status.
    Dockerfile: a plain RUN instruction (no heredoc, which runs without
    `set -e`) whose commands are joined by `&&` only, with no `!`, and the
    check right after `RUN` or an `&&`.
    Either way, no `python3 -O`; and nothing before the check may leave early:
    in the Dockerfile no `exit` or `exec` earlier in the RUN, in Singularity
    the check is a depth-0 command line with no early exit before it (see
    _post_structure_problem). `body` is the %post text, default the recipe's."""
    shell = _shell_outside_quotes(unit)
    if re.search(r"python3\s+-O", shell):
        return False
    if name == "Dockerfile":
        at = re.search(r'(^RUN |&& )python3 -c ""', shell)
        return (shell.startswith("RUN ") and "<<" not in shell
                and at is not None
                and not re.search(r";|\||(?<!&)&(?!&)|(^|\s)!(\s|$)", shell)
                and not re.search(r"\b(exit|exec)\b",
                                  re.sub(r"'[^']*'", "''", shell[:at.start()])))
    return (re.fullmatch(r'(PYTHONPATH=/opt )?python3 -c ""', shell) is not None
            and _post_keeps_errexit(_post_statements() if post is None else post)
            and _post_structure_problem(_post_body() if body is None else body, unit) is None)


SWALLOWING_SUFFIXES = [" || true", " || :", " || exit 0", "; true", " || echo skipped", " &", " | cat",
                       " \\\n    || true"]
SWALLOWING_PREFIXES = ["! ", "if ", "true || "]


@pytest.mark.parametrize("name, check, pattern", BUILD_CHECKS,
                         ids=[f"{n}-{c}" for n, c, _ in BUILD_CHECKS])
@pytest.mark.parametrize("prefix, suffix", [("", "")] + [("", s) for s in SWALLOWING_SUFFIXES]
                         + [(p, "") for p in SWALLOWING_PREFIXES])
def test_build_checks_can_fail_the_build(name, check, pattern, prefix, suffix):
    """Nothing around a python check may swallow its exit status. The
    prefixes and suffixes are mutants of the recipe text, re-parsed the way
    the tests parse the recipe (a suffix with a backslash-newline puts
    `|| true` on a continuation line); only the unchanged text passes."""
    text = DOCKERFILE if name == "Dockerfile" else SINGULARITY
    unit = _check_unit(name, pattern)
    code = list(re.finditer(PYTHON_C, unit))
    assert len(code) == 1, (name, check, unit)
    assert text.count(code[0].group(0)) == 1, (name, check)
    s = text.index(code[0].group(0))
    e = s + len(code[0].group(0))
    mutated = text[:s] + prefix + text[s:e] + suffix + text[e:]
    if name == "Dockerfile":
        units = _join_continuations(mutated)
    else:
        post = re.search(r"^%post\s*$", mutated, re.MULTILINE)
        rest = mutated[post.end():]
        nxt = re.search(r"^%\w+\s*$", rest, re.MULTILINE)
        body = rest[: nxt.start()] if nxt else rest
        units = _join_continuations(body)
    found = [u for u in units if code[0].group(0) in u]
    assert len(found) == 1, (name, check, prefix, suffix, found)
    ok = _can_fail_the_build(found[0], name, post=None if name == "Dockerfile" else units,
                             body=None if name == "Dockerfile" else body)
    assert ok == (prefix == suffix == ""), (name, check, prefix, suffix, found[0])


@pytest.mark.parametrize("name, check, pattern", BUILD_CHECKS,
                         ids=[f"{n}-{c}" for n, c, _ in BUILD_CHECKS])
def test_build_checks_do_not_rely_on_assert(name, check, pattern):
    """python3 -O and PYTHONOPTIMIZE drop asserts; a check exits explicitly."""
    code = re.search(PYTHON_C, _check_unit(name, pattern)).group(0)
    assert not re.search(r"\bassert\b", code), code


@pytest.mark.parametrize("name, check, pattern", BUILD_CHECKS,
                         ids=[f"{n}-{c}" for n, c, _ in BUILD_CHECKS])
def test_build_checks_reject_python_optimize(name, check, pattern):
    unit = _check_unit(name, pattern)
    assert _can_fail_the_build(unit, name)
    assert not _can_fail_the_build(unit.replace("python3 -c", "python3 -O -c", 1), name)


@pytest.mark.parametrize("off", ["set +e", "set +o errexit", "set +eu", "set -x +e", "set -o pipefail +o errexit"])
def test_post_errexit_cannot_be_turned_off_later(off):
    """A later `set +e` would make every %post check after it a no-op."""
    post = _post_statements()
    assert _post_keeps_errexit(post)
    assert not _post_keeps_errexit(post[:2] + [off] + post[2:])
    unit = _check_unit("Singularity", CONVERTER_IMPORT_CHECK)
    assert not _can_fail_the_build(unit, "Singularity", post=post[:2] + [off] + post[2:])


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


# Shell structure around a check. Every line-level rule above reads one command
# line, so these build %post bodies and RUN instructions whose check line is
# unchanged and whose surroundings stop it from failing the build.
CHECK = 'python3 -c "import sys; sys.exit(0)"'

# Each entry: the lines placed where the check goes, CHECK being the check line
# itself; the rest of the body is `set -e`, an echo before and an echo after.
POST_BYPASSES = {
    # the check runs and its failure is ignored
    "if-condition": ["if", CHECK, "then :; fi"],
    "subshell-or-true": ["(", CHECK, ") || true"],
    "group-into-tee": ["{", CHECK, "} 2>&1 | tee /tmp/post.log"],
    "child-shell-heredoc": ["sh <<'EOF'", CHECK, "true", "\nEOF"],
    # the check never runs
    "heredoc-comment": [": <<'EOF'", CHECK, "\nEOF"],
    "uncalled-function": ["check() {", CHECK, "}"],
    "uncalled-function-keyword": ["function check {", CHECK, "}"],
    "if-false": ["if false; then", CHECK, "fi"],
    "while-false": ["while false; do", CHECK, "done"],
    "for-empty": ["for x in; do", CHECK, "done"],
    "case-no-match": ["case x in", "y)", CHECK, ";;", "esac"],
    "exit-0-before": ["exit 0", CHECK],
    "exit-bare-before": ["exit", CHECK],
    "exec-before": ["exec true", CHECK],
    "return-before": ["return 0", CHECK],
    "conditional-exit-before": ["true && exit 0", CHECK],
    "exit-0-in-if-before": ["if true; then exit 0; fi", CHECK],
    "exit-0-in-group-before": ["{ exit 0; }", CHECK],
    # structure the scanner cannot close is not trusted either
    "unclosed-if": ["if true; then", CHECK],
    "unclosed-heredoc": [": <<'EOF'", CHECK],
    "indented-heredoc-end": [": <<'EOF'", "x", "    EOF", CHECK],
    "stray-fi": ["fi", CHECK],
}

# Structure that does not touch the check.
POST_HARMLESS = {
    "plain": [CHECK],
    "single-line-if-before": ["if [ -z \"$X\" ]; then echo none; fi", CHECK],
    "multi-line-if-before": ["if [ -n \"$X\" ]; then", "    echo set", "fi", CHECK],
    "if-with-error-exit-before": ["if [ -z \"$X\" ]; then", "    echo no >&2", "    exit 1", "fi", CHECK],
    "for-loop-before": ["for x in a b; do echo $x; done", CHECK],
    "case-before": ["case $X in", "a) echo a ;;", "*) echo other ;;", "esac", CHECK],
    "group-before": ["{ echo a; echo b; } > /dev/null", "(cd /tmp && ls) > /dev/null", CHECK],
    "command-substitution-before": ['u="$(find /opt \\( -type f \\) -o \\( -type d \\))"', CHECK],
    "heredoc-before": ["cat > /tmp/f <<EOF", "if (", "\nEOF", CHECK],
    "function-defined-and-called-before": ["f() { echo f; }", "f", CHECK],
    "compound-after": [CHECK, "if true; then", "    exit 1", "fi"],
    "exit-after": [CHECK, "exit 0"],
}


def _post_body_with(lines):
    return "\n    ".join(["", "set -e", "echo before"] + lines + ["echo after", ""])


@pytest.mark.parametrize("lines", POST_BYPASSES.values(), ids=POST_BYPASSES.keys())
def test_post_check_inside_other_shell_structure_cannot_pass(lines):
    """Each form leaves the check line as it is, passes the line-level rules,
    and either ignores the check's failure or never runs it."""
    body = _post_body_with(lines)
    assert not _can_fail_the_build(CHECK, "Singularity", post=_join_continuations(body), body=body)


@pytest.mark.parametrize("lines", POST_HARMLESS.values(), ids=POST_HARMLESS.keys())
def test_post_check_after_unrelated_shell_structure_passes(lines):
    body = _post_body_with(lines)
    assert _can_fail_the_build(CHECK, "Singularity", post=_join_continuations(body), body=body)


def test_post_structure_scan_accepts_the_recipe():
    lines, errors = _scan_post(_post_body())
    assert errors == []
    assert lines
    for pattern in (FUSED_OPS_ASSERTION_CMD, CONVERTER_IMPORT_CHECK):
        assert _post_structure_problem(_post_body(), _check_unit("Singularity", pattern)) is None


DOCKER_BYPASSES = {
    "exit-0-before": "RUN exit 0 && " + CHECK,
    "exec-true-before": "RUN exec true && " + CHECK,
    "exit-later-in-chain": "RUN echo a && exit 0 && echo b && " + CHECK,
    "exec-mid-chain": "RUN echo a && exec sh -c true && " + CHECK,
}
DOCKER_HARMLESS = {
    "plain": "RUN " + CHECK,
    "after-commands": "RUN echo a && echo b && " + CHECK,
    "word-in-quotes": "RUN echo 'exit 0' && " + CHECK,
}


@pytest.mark.parametrize("run", DOCKER_BYPASSES.values(), ids=DOCKER_BYPASSES.keys())
def test_dockerfile_run_that_exits_before_the_check_cannot_pass(run):
    assert not _can_fail_the_build(run, "Dockerfile")


@pytest.mark.parametrize("run", DOCKER_HARMLESS.values(), ids=DOCKER_HARMLESS.keys())
def test_dockerfile_run_without_early_exit_passes(run):
    assert _can_fail_the_build(run, "Dockerfile")
