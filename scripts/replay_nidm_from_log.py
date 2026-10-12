#!/usr/bin/env python3
"""
Replay a recorded NIDM conversion command from an ANTs BABS log.

This helper parses the command logged by ``src/run.py`` and re-executes the
conversion against the referenced ANTs outputs using the locally checked-in
``ants_seg_to_nidm`` package.
use  `micromamba activate nidm-test`
Example usage:
    python scripts/replay_nidm_from_log.py <babs_project>/analysis/logs/ant.e<job>_<task> --forcenidm

"""

import argparse
import importlib
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the JSON-LD serializer is registered with rdflib if available.
try:  # pragma: no cover - optional dependency
    import rdflib
    from rdflib.plugin import Serializer, register
    import rdflib_jsonld  # noqa: F401

    register(
        name="jsonld",
        kind=Serializer,
        module_path="rdflib_jsonld.serializer",
        class_name="JsonLDSerializer",
    )
except ImportError:
    rdflib = None  # optional dependency missing


def _parse_log(log_path: Path) -> List[Dict[str, Optional[str]]]:
    """Collect each ``python -m ants_seg_to_nidm`` command block from the log."""
    entries: List[Dict[str, Optional[str]]] = []
    current: Optional[Dict[str, Optional[str]]] = None

    for raw_line in log_path.read_text().splitlines():
        line = raw_line.strip()
        if "Running command:" in line and "ants_seg_to_nidm" in line:
            if current:
                entries.append(current)
            current = {"command": line.split("Running command:", 1)[1].strip(), "nidm": None}
        elif current and "Adding data to existing NIDM file:" in line:
            marker = "Adding data to existing NIDM file:"
            current["nidm"] = line.split(marker, 1)[1].strip()

    if current:
        entries.append(current)

    return entries


def _split_command(command: str) -> Tuple[str, List[str]]:
    """Split the recorded command into module path and arguments."""
    tokens = shlex.split(command)
    try:
        idx = tokens.index("-m")
    except ValueError as exc:
        raise ValueError("Command does not include a '-m' module invocation") from exc

    try:
        module = tokens[idx + 1]
    except IndexError as exc:
        raise ValueError("Module name missing after '-m'") from exc

    return module, tokens[idx + 2 :]


def _sanitize_args(args: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Drop unsupported flags (currently -t1/--t1 and -j/--jsonld to default to TTL) and return cleaned args."""
    cleaned: List[str] = []
    removed: Dict[str, str] = {}
    skip_next = False

    for idx, token in enumerate(args):
        if skip_next:
            skip_next = False
            continue

        if token in {"-t1", "--t1"}:
            removed[token] = args[idx + 1] if idx + 1 < len(args) else ""
            skip_next = True
            continue

        # Remove -j/--jsonld to default to TTL output
        if token in {"-j", "--jsonld"}:
            removed[token] = ""
            continue

        cleaned.append(token)

    return cleaned, removed


def _ensure_pythonpath(script_path: Path) -> None:
    """Include the repo's ``src`` tree on sys.path so imports succeed."""
    repo_root = script_path.resolve().parents[1]
    src_root = repo_root / "src"

    for candidate in (src_root, repo_root):
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _import_module(module: str):
    """Import the module containing ``main`` for execution."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Unable to import '{module}'. Ensure the repo is on PYTHONPATH or the "
            "package is installed (pip install -e .)."
        ) from exc


def _run_module(module_name: str, argv: List[str]) -> int:
    """Invoke ``main`` from the targeted module, returning its exit code."""
    module = _import_module(module_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"Module '{module_name}' does not expose a 'main' callable")

    saved_argv = sys.argv
    sys.argv = [module_name.split(".")[-1]] + argv
    try:
        module.main()  # type: ignore[attr-defined]
    except ValueError as exc:
        # Catch missing label errors and report them for debugging
        error_msg = str(exc)
        if "not found in ANTS data elements file" in error_msg:
            print(f"\n{'='*70}")
            print("MISSING LABEL DETECTED:")
            print(f"{'='*70}")
            print(error_msg)
            print(f"{'='*70}\n")
            # Extract label info if possible
            if "ANTSDKT(structure=" in error_msg:
                import re
                match = re.search(r"ANTSDKT\(([^)]+)\)", error_msg)
                if match:
                    print(f"Label details: {match.group(0)}")
            return 1
        else:
            raise
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code
    finally:
        sys.argv = saved_argv

    return 0


def _summarize_inputs(args: List[str]) -> Dict[str, List[Path]]:
    """Collect referenced filesystem inputs for a quick existence check."""
    summary: Dict[str, List[Path]] = {}
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == "-f" and idx + 1 < len(args):
            summary["stats"] = [Path(p) for p in args[idx + 1].split(",")]
            idx += 1
        elif token in {"-o", "--nidm", "-n"} and idx + 1 < len(args):
            key = "output" if token == "-o" else "nidm"
            summary.setdefault(key, []).append(Path(args[idx + 1]))
            idx += 1
        idx += 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay a logged ants_seg_to_nidm invocation against existing ANTs outputs.",
    )
    parser.add_argument("log_path", type=Path, help="Path to the ANTs log file")
    parser.add_argument(
        "--occurrence",
        type=int,
        default=1,
        help="Which occurrence of the command to run (1 = first)",
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        help="Override the module path to execute instead of the one in the log",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the reconstructed command without running it",
    )
    parser.add_argument(
        "--forcenidm",
        action="store_true",
        help="Append --forcenidm to the replayed command if it is not already present",
    )
    parser.add_argument(
        "--collect-missing",
        action="store_true",
        help="Append --collect-missing to the replayed command to report all missing labels",
    )

    args = parser.parse_args()
    entries = _parse_log(args.log_path)
    if not entries:
        raise SystemExit("No ants_seg_to_nidm commands found in the provided log.")

    if args.occurrence < 1 or args.occurrence > len(entries):
        raise SystemExit(
            f"Occurrence {args.occurrence} is out of range; found {len(entries)} commands."
        )

    entry = entries[args.occurrence - 1]
    
    # Print the parsed log entry
    print(f"\n{'='*70}")
    print(f"PARSED LOG ENTRY (occurrence {args.occurrence} of {len(entries)}):")
    print(f"{'='*70}")
    print(f"Command: {entry['command']}")
    if entry['nidm']:
        print(f"NIDM file: {entry['nidm']}")
    print(f"{'='*70}\n")
    
    module_name, module_args = _split_command(entry["command"])
    replay_module = args.module or module_name

    sanitized_args, removed = _sanitize_args(module_args)
    if entry["nidm"]:
        if "--nidm" in sanitized_args:
            idx = sanitized_args.index("--nidm")
            needs_value = idx + 1 >= len(sanitized_args) or sanitized_args[idx + 1].startswith("-")
            if needs_value:
                sanitized_args.insert(idx + 1, entry["nidm"])
        elif "-n" in sanitized_args:
            idx = sanitized_args.index("-n")
            needs_value = idx + 1 >= len(sanitized_args) or sanitized_args[idx + 1].startswith("-")
            if needs_value:
                sanitized_args.insert(idx + 1, entry["nidm"])
        else:
            sanitized_args.extend(["--nidm", entry["nidm"]])
        
        # Fix output path: when adding to existing NIDM file, output should be nidm.ttl
        # not the subject-specific JSON-LD file
        if "-o" in sanitized_args:
            o_idx = sanitized_args.index("-o")
            if o_idx + 1 < len(sanitized_args):
                old_output = Path(sanitized_args[o_idx + 1])
                # If output was subject-specific JSON-LD, change to nidm.ttl
                if old_output.name.endswith("_dseg.json-ld"):
                    new_output = old_output.parent / "nidm.ttl"
                    print(f"Correcting output path:")
                    print(f"  Old: {old_output}")
                    print(f"  New: {new_output}")
                    sanitized_args[o_idx + 1] = str(new_output)

    if args.forcenidm and "--forcenidm" not in sanitized_args and "-forcenidm" not in sanitized_args:
        sanitized_args.append("--forcenidm")
    
    if args.collect_missing and "--collect-missing" not in sanitized_args:
        sanitized_args.append("--collect-missing")

    command_preview = " ".join(shlex.quote(part) for part in sanitized_args)
    print(f"Resolved module: {replay_module}")
    print(f"Arguments: {command_preview}")
    if removed:
        print("Removed unsupported options:")
        for flag, value in removed.items():
            if value:
                print(f"  {flag} {value}")
            else:
                print(f"  {flag}")

    summary = _summarize_inputs(sanitized_args)
    for label, paths in summary.items():
        for path in paths:
            status = "OK" if path.exists() else "MISSING"
            print(f"{label.upper()}: {path} [{status}]")

    if args.dry_run:
        return 0

    _ensure_pythonpath(Path(__file__))
    try:
        exit_code = _run_module(replay_module, sanitized_args)
    except (ImportError, AttributeError) as exc:
        print(str(exc))
        return 1
    if exit_code == 0:
        print("NIDM conversion finished successfully.")
    else:
        print(f"NIDM conversion exited with code {exit_code}.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
