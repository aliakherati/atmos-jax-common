"""Build Fortran + run the canonical matrix + commit reference outputs (chunk C0.9).

Drives the full goldens-generation workflow:

1. Load ``data/canonical_runs/manifest.json``.
2. Build ``box.exe`` from ``third_party/som-tomas-fortran/src`` (unless
   ``--skip-build`` is passed).
3. For each run, render the box.f stdin, execute via the
   ``fortran_runner`` wrapper, and copy every output file produced
   under that run's name into ``data/canonical_runs/expected/<run_id>/``.
4. Write a per-run ``metadata.json`` recording the input SHA-256, the
   per-source-file SHA-256s of the Fortran build, the submodule commit,
   and a UTC generation timestamp. The metadata is what the
   mechanism-drift CI guard (C0.10) compares against.

Modes
-----

``--generate`` (default)
    Writes outputs + metadata into ``data/canonical_runs/expected/``.
``--check``
    Generates into a scratch dir and diffs every committed output file
    against the freshly-produced one. Exits non-zero on any byte
    difference. ``metadata.json`` is excluded from the diff (its
    ``generated_at_utc`` field changes every run by design).
``--only RUN_ID [RUN_ID ...]``
    Restrict to a subset of run IDs (faster local iteration; CI uses the
    full matrix).
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from atmos_jax_common.canonical_runs import (
    CanonicalMatrix,
    CanonicalRun,
    SharedParams,
    default_manifest_path,
    load_manifest,
    render_input,
)
from atmos_jax_common.fortran_runner import build, run

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SRC_DIR = _REPO_ROOT / "third_party" / "som-tomas-fortran" / "src"
_DEFAULT_EXPECTED_DIR = _REPO_ROOT / "data" / "canonical_runs" / "expected"


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_of_file(path: Path) -> str:
    return _sha256_of_bytes(path.read_bytes())


def _git_rev(path: Path) -> str | None:
    """Return the git HEAD SHA for the repository containing ``path``,
    or ``None`` if it isn't a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _build_metadata(
    run: CanonicalRun,
    input_text: str,
    src_dir: Path,
) -> dict:
    fortran_files = {path.name: _sha256_of_file(path) for path in sorted(src_dir.glob("*.f"))}
    return {
        "run_id": run.run_id,
        "summary": run.summary,
        "input_sha256": _sha256_of_bytes(input_text.encode("utf-8")),
        "submodule_commit": _git_rev(src_dir),
        "fortran_source_sha256": fortran_files,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _generate_one(
    run: CanonicalRun,
    shared: SharedParams,
    src_dir: Path,
    target_dir: Path,
) -> None:
    """Run ``box.exe`` once for ``run`` and write everything into ``target_dir``."""
    input_text = render_input(run, shared)
    result = run_box(src_dir, input_text)

    target_dir.mkdir(parents=True, exist_ok=True)
    # Wipe stale outputs so removed files (e.g., after a Fortran change)
    # don't linger in the committed tree.
    for old in target_dir.glob(f"{run.run_id}*"):
        old.unlink()
    for src_file in result.outputs_dir.glob(f"{run.run_id}*"):
        shutil.copy2(src_file, target_dir / src_file.name)
    # Also keep the resolved input next to the outputs for ease of audit.
    (target_dir / f"{run.run_id}.input").write_text(input_text, encoding="utf-8")

    metadata = _build_metadata(run, input_text, src_dir)
    (target_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def run_box(src_dir: Path, input_text: str):
    """Indirection so unit tests can monkey-patch the actual box.exe call."""
    result = run(src_dir, input_text)
    if result.returncode != 0:
        raise RuntimeError(
            f"box.exe failed (exit={result.returncode}); stderr tail:\n{result.stderr[-1000:]}"
        )
    return result


def _diff_dirs(committed: Path, fresh: Path) -> list[str]:
    """Return a list of human-readable drift descriptions.

    ``metadata.json`` is excluded from the diff (its ``generated_at_utc``
    field intentionally changes every run; the auditable parts of
    metadata are the SHAs, which are checked separately by C0.10).
    """
    diffs: list[str] = []
    committed_files = {p.name for p in committed.iterdir() if p.is_file()}
    fresh_files = {p.name for p in fresh.iterdir() if p.is_file()}

    only_committed = committed_files - fresh_files - {"metadata.json"}
    only_fresh = fresh_files - committed_files - {"metadata.json"}

    if only_committed:
        diffs.append(f"  files removed: {sorted(only_committed)}")
    if only_fresh:
        diffs.append(f"  files added:   {sorted(only_fresh)}")

    common = (committed_files & fresh_files) - {"metadata.json"}
    for name in sorted(common):
        if not filecmp.cmp(committed / name, fresh / name, shallow=False):
            diffs.append(f"  bytes differ: {name}")

    # Compare metadata.json by SHAs only (not timestamp).
    if (committed / "metadata.json").exists() and (fresh / "metadata.json").exists():
        a = json.loads((committed / "metadata.json").read_text())
        b = json.loads((fresh / "metadata.json").read_text())
        for key in ("input_sha256", "fortran_source_sha256"):
            if a.get(key) != b.get(key):
                diffs.append(f"  metadata.{key} differs")
    return diffs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--generate",
        action="store_true",
        help="Default mode: regenerate goldens into data/canonical_runs/expected/.",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Generate into a scratch dir and diff against the committed outputs. "
        "Exits non-zero on any drift.",
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=_DEFAULT_SRC_DIR,
        help="Path to the Fortran source dir (default: third_party submodule).",
    )
    parser.add_argument(
        "--expected-dir",
        type=Path,
        default=_DEFAULT_EXPECTED_DIR,
        help="Path to write/check goldens against.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest_path(),
        help="Path to manifest.json.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="RUN_ID",
        help="Restrict to a subset of run IDs.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Don't run `make` first; assume box.exe is already built.",
    )
    args = parser.parse_args(argv)

    if not args.check:
        args.generate = True  # default

    matrix: CanonicalMatrix = load_manifest(args.manifest)
    if args.only is not None:
        wanted = set(args.only)
        runs = [r for r in matrix.runs if r.run_id in wanted]
        missing = wanted - {r.run_id for r in runs}
        if missing:
            print(f"unknown run_id(s): {sorted(missing)}", file=sys.stderr)
            return 2
    else:
        runs = list(matrix.runs)

    src_dir = args.src_dir.resolve()
    if not args.skip_build:
        print(f"Building Fortran in {src_dir}...")
        build(src_dir)

    if args.check:
        return _run_check(matrix, runs, src_dir, args.expected_dir)
    return _run_generate(matrix, runs, src_dir, args.expected_dir)


def _run_generate(
    matrix: CanonicalMatrix,
    runs: list[CanonicalRun],
    src_dir: Path,
    out_dir: Path,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating goldens into {out_dir.relative_to(_REPO_ROOT)} ({len(runs)} runs)")
    for crun in runs:
        print(f"  -> {crun.run_id}")
        _generate_one(crun, matrix.shared, src_dir, out_dir / crun.run_id)
    print("done")
    return 0


def _run_check(
    matrix: CanonicalMatrix,
    runs: list[CanonicalRun],
    src_dir: Path,
    committed_dir: Path,
) -> int:
    if not committed_dir.is_dir():
        print(f"committed goldens dir missing: {committed_dir}", file=sys.stderr)
        return 1
    drift_total = 0
    with tempfile.TemporaryDirectory(prefix="atmos_goldens_check_") as tmp:
        scratch = Path(tmp)
        for crun in runs:
            print(f"checking {crun.run_id}...")
            _generate_one(crun, matrix.shared, src_dir, scratch / crun.run_id)
            committed_run = committed_dir / crun.run_id
            if not committed_run.is_dir():
                print(f"  drift: no committed dir for {crun.run_id}", file=sys.stderr)
                drift_total += 1
                continue
            diffs = _diff_dirs(committed_run, scratch / crun.run_id)
            if diffs:
                print(f"  drift in {crun.run_id}:", file=sys.stderr)
                for d in diffs:
                    print(d, file=sys.stderr)
                drift_total += len(diffs)
    if drift_total:
        print(
            f"\n{drift_total} drift entr(y/ies) detected. "
            f"Either the Fortran source changed (rerun with --generate to refresh) "
            f"or the committed goldens are stale.",
            file=sys.stderr,
        )
        return 1
    print(f"All {len(runs)} canonical runs match the committed goldens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
