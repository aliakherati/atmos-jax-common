"""Build and run the Fortran SOM-TOMAS reference.

This module wraps the command-line workflow described in the ``som-tomas-app``
repo (``make -f simple_makfile`` to build, ``./box.exe < input`` to run) into a
typed Python API. The wrapper handles three annoyances of the Fortran layout:

1. **Relative output paths.** ``box.f`` writes outputs to ``../outputs/`` relative
   to its current working directory. Running from a user's source tree would
   splatter outputs into ``som-tomas-app/outputs/``. Each :func:`run` call
   instead creates a scratch directory with a mirror of the source (as
   symlinks — zero-copy) plus a fresh ``outputs/`` directory, so the relative
   paths resolve there.

2. **``STATUS='new'`` open semantics.** Fortran's ``OPEN(..., STATUS='new')``
   raises if the target file already exists. By always using a fresh scratch
   directory per run, we never hit this.

3. **Build state.** ``make`` handles incremental compilation, so :func:`build`
   is idempotent-cheap. Call it once per test session; :func:`run` does not
   implicitly rebuild.

The heavy lifting (reading the output files, normalising units, comparing to
JAX outputs) lives in :mod:`atmos_jax_common.goldens` and
:mod:`atmos_jax_common.compare` (C0.5–C0.7).

Example
-------
::

    from atmos_jax_common.fortran_runner import build, run

    src_dir = Path("som-tomas-app/src")
    build(src_dir)                                # builds box.exe in-place

    input_text = Path("my_canonical_input").read_text()
    result = run(src_dir, input_text)

    print(result.stdout[-500:])                   # tail of Fortran stdout
    for p in result.output_files:
        print(p.relative_to(result.outputs_dir))  # e.g. myrun_gc.dat
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "FortranBuildError",
    "FortranRunError",
    "RunOutputs",
    "build",
    "run",
]


class FortranBuildError(RuntimeError):
    """Raised when ``make`` fails to produce ``box.exe``."""


class FortranRunError(RuntimeError):
    """Raised when ``./box.exe`` exits non-zero or produces no output."""


@dataclass(frozen=True)
class RunOutputs:
    """Captured results of a single ``./box.exe`` invocation.

    Attributes
    ----------
    run_name
        The run identifier parsed from the first line of the input text;
        also the prefix the Fortran uses for every output-file basename.
    stdout
        Combined stdout+stderr from the Fortran process.
    returncode
        Exit code from the Fortran process (always 0 if :func:`run`
        returns — a non-zero exit raises ``FortranRunError`` instead).
    scratch_dir
        Root of the scratch tree. Contains ``src/`` (symlinks into the
        user's Fortran source) and ``outputs/`` (the files below).
        Callers who create their own ``scratch_dir`` are responsible for
        cleanup; runs with the default ``tempfile.mkdtemp()`` scratch are
        left on disk so the caller can inspect outputs, and can be
        cleaned up via :func:`shutil.rmtree`.
    outputs_dir
        Directory containing the per-run output files. Equivalent to
        ``scratch_dir / "outputs"``.
    output_files
        Tuple of every file produced under ``outputs_dir``, in sorted
        order. Typically 9 entries: one main ``.dat`` and eight
        suffix-qualified files (``_gc.dat``, ``_noconc.dat``,
        ``_aemass.dat``, ``_spec.dat``, ``_saprcgc.dat``, ``_vl.dat``,
        ``_tau.dat``, ``_kw.dat``).
    """

    run_name: str
    stdout: str
    returncode: int
    scratch_dir: Path
    outputs_dir: Path
    output_files: tuple[Path, ...]


def build(
    src_dir: Path | str,
    *,
    makefile: str = "simple_makfile",
    skip_clean: bool = False,
) -> Path:
    """Build ``box.exe`` in place and return its path.

    Always runs ``make clean`` before the build by default. The ``som-tomas-app``
    source tree ships with committed ``.o`` files from a Linux build in 2022;
    without a clean they defeat ``make``'s incremental logic on a different
    architecture and the link step fails with ``symbol(s) not found``. Set
    ``skip_clean=True`` only if you are sure the current ``.o`` files are
    consistent with the target toolchain.

    Parameters
    ----------
    src_dir
        Directory containing the Fortran source tree (e.g.
        ``som-tomas-app/src``).
    makefile
        Name of the Makefile to use. The repo's default
        ``"simple_makfile"`` targets ``gfortran`` and has been verified to
        produce a working executable; the Intel-oriented ``"Makefile"``
        works only when ``ifort`` is installed.
    skip_clean
        If ``True``, skip ``make clean``. Faster on repeated invocations
        but only safe when the existing ``.o`` artifacts match the active
        toolchain.

    Returns
    -------
    Path to the built ``box.exe``.

    Raises
    ------
    FileNotFoundError
        If ``src_dir`` or the chosen makefile does not exist.
    FortranBuildError
        If ``make`` exits non-zero or if ``box.exe`` is missing after
        a successful-looking invocation.
    """
    src_path = Path(src_dir).resolve()
    if not src_path.is_dir():
        raise FileNotFoundError(f"src_dir {src_path} is not a directory")
    if not (src_path / makefile).is_file():
        raise FileNotFoundError(f"{makefile} not found in {src_path}")

    if not skip_clean:
        _make(src_path, makefile, ["clean"])
    _make(src_path, makefile)

    box_exe = src_path / "box.exe"
    if not box_exe.is_file():
        raise FortranBuildError(
            f"make completed but {box_exe} does not exist — "
            "check the Makefile or the Fortran source for silent errors"
        )
    return box_exe


def _make(src_path: Path, makefile: str, extra_args: list[str] | None = None) -> None:
    cmd = ["make", "-f", makefile, *(extra_args or [])]
    proc = subprocess.run(
        cmd,
        cwd=src_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise FortranBuildError(
            f"'{' '.join(cmd)}' failed with exit code {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )


def run(
    src_dir: Path | str,
    input_text: str,
    *,
    scratch_dir: Path | str | None = None,
    timeout_seconds: float = 600.0,
) -> RunOutputs:
    """Run ``./box.exe`` with ``input_text`` on stdin and collect outputs.

    Parameters
    ----------
    src_dir
        Directory containing the built Fortran source tree. Must contain
        ``box.exe``; call :func:`build` first if it's missing.
    input_text
        Text to feed ``box.exe`` on stdin. Its first line is the run
        identifier that the Fortran uses as a prefix for every output
        filename. See ``som-tomas-app/src/runme.py`` for the canonical
        input format.
    scratch_dir
        Directory to use as the scratch root. If ``None``, a fresh
        ``tempfile.mkdtemp()`` directory is created and left on disk
        after return (so callers can read the output files). If the
        caller provides a ``scratch_dir``, it must either be empty or
        not yet exist — existing ``outputs/`` content would make the
        Fortran's ``STATUS='new'`` ``OPEN`` fail.
    timeout_seconds
        Hard ceiling on the ``box.exe`` call. The Fortran is synchronous
        and can run for minutes on long canonical inputs; raise this
        value if you hit ``subprocess.TimeoutExpired``.

    Returns
    -------
    :class:`RunOutputs` with the run name, captured stdout, and the list
    of output files produced under ``scratch_dir / "outputs"``.

    Raises
    ------
    FileNotFoundError
        If ``src_dir`` or ``src_dir/box.exe`` is missing.
    FortranRunError
        If ``box.exe`` exits non-zero, or if it exits zero but produces
        no output files (a silent failure in the Fortran I/O layer).
    subprocess.TimeoutExpired
        If the Fortran exceeds ``timeout_seconds``.
    """
    src_path = Path(src_dir).resolve()
    box_exe = src_path / "box.exe"
    if not box_exe.is_file():
        raise FileNotFoundError(f"{box_exe} not found — run build(src_dir) first")

    scratch_root = (
        Path(tempfile.mkdtemp(prefix="atmos_jax_common_"))
        if scratch_dir is None
        else Path(scratch_dir)
    )
    mirror_src, mirror_out = _prepare_scratch(scratch_root, src_path)

    # First line of the input text is the run name (Fortran reads it as
    # RUNNAME via read(*,'(A120)')). We mirror that so RunOutputs can
    # report it without re-parsing the Fortran input format.
    run_name = input_text.split("\n", 1)[0].strip()

    proc = subprocess.run(
        ["./box.exe"],
        cwd=mirror_src,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )

    if proc.returncode != 0:
        raise FortranRunError(
            f"box.exe exited with code {proc.returncode}\n"
            f"run_name={run_name!r}\n"
            f"scratch_dir={scratch_root}\n"
            f"--- stdout tail ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-2000:]}"
        )

    output_files = tuple(sorted(mirror_out.iterdir()))
    if not output_files:
        raise FortranRunError(
            f"box.exe succeeded but wrote no files to {mirror_out}. "
            "The Fortran may have failed silently at file open; check the "
            "stdout for 'STATUS=new' or permission errors."
        )

    # Prefer combined stdout+stderr for the captured text — diffrax
    # parity work may need to inspect warnings/prints. stderr is usually
    # empty from gfortran builds, but included for completeness.
    combined = proc.stdout
    if proc.stderr:
        combined += "\n--- stderr ---\n" + proc.stderr

    return RunOutputs(
        run_name=run_name,
        stdout=combined,
        returncode=proc.returncode,
        scratch_dir=scratch_root,
        outputs_dir=mirror_out,
        output_files=output_files,
    )


def _prepare_scratch(scratch_root: Path, src_path: Path) -> tuple[Path, Path]:
    """Create the ``scratch_root/{src,outputs}`` layout.

    Uses symlinks (not copies) so large mechanism files and static data
    stay out of the scratch tree. ``scratch_root`` may or may not exist
    already; either way, ``src`` and ``outputs`` under it are created
    fresh here.
    """
    scratch_root.mkdir(parents=True, exist_ok=True)
    mirror_src = scratch_root / "src"
    mirror_out = scratch_root / "outputs"

    if mirror_src.exists():
        shutil.rmtree(mirror_src)
    mirror_src.mkdir()

    if mirror_out.exists():
        # Fresh outputs dir is essential for STATUS='new' semantics.
        shutil.rmtree(mirror_out)
    mirror_out.mkdir()

    for entry in src_path.iterdir():
        # Skip the original outputs link or anything Fortran-generated.
        if entry.name in {"outputs", "scratch"}:
            continue
        (mirror_src / entry.name).symlink_to(entry.resolve())

    return mirror_src, mirror_out
