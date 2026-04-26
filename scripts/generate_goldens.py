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
    against the freshly-produced one. Numeric ``.dat`` files are
    compared with relative tolerance ``rtol=1e-4`` (float32 ULP-level
    noise from the REAL*4 chemistry callback re-orders across
    gfortran versions and CPU architectures and is intentionally
    accepted); text files (``.input``, ``_spec.dat``) keep byte
    equality. Real Fortran source changes shift cascade species by
    ≥0.1% — well above tolerance — and are caught. Exits non-zero on
    any drift. ``metadata.json``'s ``generated_at_utc`` is excluded;
    its SHA fields are cross-checked.
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

import numpy as np

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


# Tolerance for cross-machine numerical drift on the committed Fortran
# outputs. The Fortran chemistry callback DIFUN runs in REAL*4
# (per the integr2.f bridge), and the resulting RKZ*C*C products carry
# float32 ULP noise (~1e-7 relative). Different gfortran versions and
# CPU architectures reorder those products slightly, so two cleanly-
# built copies of the same source can land at ~1e-5 relative apart on
# cascade species. We accept that as "no drift" and only flag genuine
# Fortran source changes (which produce ≥0.1% shifts on at least one
# species). Tolerance is per-element via np.allclose, atol-floored so
# zero-reference cells (initial-condition columns) compare cleanly.
_DRIFT_RTOL = 1e-4
_DRIFT_ATOL = 1e-30

# Chemistry-relevant output files that the drift guard compares
# numerically. Outputs not in this set are ignored — specifically the
# aerosol-side files (``_aemass.dat``, ``_noconc.dat``, ``<run>.dat``,
# ``_tau.dat``, ``_vl.dat``, ``_kw.dat``) come out of TOMAS, which has
# its own cross-machine reproducibility issues (see upstream
# som-tomas-fortran#1 / U1) and isn't the signal we want to gate on
# for SAPRC mechanism drift. When tomas-jax lands and TOMAS
# regression coverage matters, we'll add a separate aerosol-drift
# guard with appropriate per-file atols.
_CHEMISTRY_DRIFT_FILES_SUFFIXES = ("_saprcgc.dat", "_gc.dat")
_BYTE_EQUAL_SUFFIXES = (".input", "_spec.dat")


def _is_chemistry_drift_file(name: str) -> bool:
    return any(name.endswith(s) for s in _CHEMISTRY_DRIFT_FILES_SUFFIXES)


def _is_byte_equal_file(name: str) -> bool:
    return any(name.endswith(s) for s in _BYTE_EQUAL_SUFFIXES)


def _try_loadtxt(path: Path) -> np.ndarray | None:
    """Return ``path`` parsed as a 2-D float matrix, or ``None`` if it
    isn't pure-numeric (e.g., ``_spec.dat`` carries species names)."""
    try:
        arr = np.loadtxt(path, ndmin=2)
        return arr
    except (ValueError, OSError):
        return None


def _diff_dirs(committed: Path, fresh: Path) -> list[str]:
    """Return a list of human-readable drift descriptions.

    Three classes of files:

    - **Chemistry numeric** (``_saprcgc.dat``, ``_gc.dat``): compared
      with :data:`_DRIFT_RTOL` relative tolerance via
      :func:`np.allclose`. Byte equality is too strict because REAL*4
      chemistry products carry float32 ULP noise that re-orders across
      gfortran versions and CPU architectures.
    - **Byte-equal text** (``.input``, ``_spec.dat``): byte equality
      preserved (these are fully deterministic).
    - **Aerosol / solver metadata** (everything else): ignored. The
      aerosol pipeline (``_aemass.dat``, ``_noconc.dat``, etc.) has
      cross-machine reproducibility issues unrelated to SAPRC
      chemistry; tomas-jax regression coverage will gate those
      separately.

    ``metadata.json`` is excluded from byte/numeric comparison (its
    ``generated_at_utc`` changes every run by design); we cross-check
    its SHA fields, which catches mechanism / input drift directly.
    """
    diffs: list[str] = []
    committed_files = {p.name for p in committed.iterdir() if p.is_file()}
    fresh_files = {p.name for p in fresh.iterdir() if p.is_file()}

    # Adding/removing chemistry-relevant or byte-equal files counts as
    # drift; aerosol-side files don't.
    def _gate(names: set[str]) -> set[str]:
        return {n for n in names if _is_chemistry_drift_file(n) or _is_byte_equal_file(n)}

    only_committed = _gate(committed_files - fresh_files - {"metadata.json"})
    only_fresh = _gate(fresh_files - committed_files - {"metadata.json"})

    if only_committed:
        diffs.append(f"  files removed: {sorted(only_committed)}")
    if only_fresh:
        diffs.append(f"  files added:   {sorted(only_fresh)}")

    common = (committed_files & fresh_files) - {"metadata.json"}
    for name in sorted(common):
        if not (_is_chemistry_drift_file(name) or _is_byte_equal_file(name)):
            continue  # aerosol / solver metadata — not a drift signal here
        a_path = committed / name
        b_path = fresh / name
        if filecmp.cmp(a_path, b_path, shallow=False):
            continue
        if _is_chemistry_drift_file(name):
            a_arr = _try_loadtxt(a_path)
            b_arr = _try_loadtxt(b_path)
            if a_arr is not None and b_arr is not None:
                if a_arr.shape != b_arr.shape:
                    diffs.append(f"  shape differs: {name} ({a_arr.shape} vs {b_arr.shape})")
                    continue
                if np.allclose(a_arr, b_arr, rtol=_DRIFT_RTOL, atol=_DRIFT_ATOL, equal_nan=False):
                    continue  # within tolerance → not drift
                denom = np.maximum(np.abs(a_arr), _DRIFT_ATOL)
                with np.errstate(invalid="ignore"):
                    worst = float(np.nanmax(np.abs(a_arr - b_arr) / denom))
                diffs.append(
                    f"  numeric drift: {name} (max rel diff {worst:.2e} > rtol {_DRIFT_RTOL:.0e})"
                )
                continue
            diffs.append(f"  numeric load failed: {name}")
            continue
        # Byte-equal text file that differs → drift.
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
