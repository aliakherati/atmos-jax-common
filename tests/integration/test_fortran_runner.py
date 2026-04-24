"""Integration tests for the Fortran runner.

Gated by ``gfortran`` being on PATH and by the Fortran source being
available — either via the vendored ``third_party/som-tomas-fortran``
submodule (default) or the ``SOM_TOMAS_FORTRAN_SRC`` environment
variable override. See ``tests/integration/conftest.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from atmos_jax_common.fortran_runner import (
    FortranRunError,
    RunOutputs,
    build,
    run,
)

pytestmark = pytest.mark.fortran

# A runname that the Fortran is happy to accept and that we can
# cheaply identify in the outputs directory.
_RUN_NAME = "integration_test_run"

# Minimal input that drives box.exe end-to-end. Fields are the stdin
# format read by box.f: the first line is the run name, then the scalars
# in the order box.f reads them. Sourced from the sample ``input`` file
# in som-tomas-app/src, abbreviated for CI speed.
#
# Switches chosen for a short, well-defined run:
# - COAG=0, VWL=0, PWL=0: aerosol microphysics essentially off
# - GENVOC + OH only; OH=1.5e6; endtime=0.1 h (6 min) for fast tests
_SHORT_INPUT = f"""{_RUN_NAME}
0
0
0
0.000000
0.000000
1.500000E+06
0.000
0.000
7000000.0
0.10
0.00100
1.00e-10
0.000000
0.02500
101325.000
298.00
0.20000
1.00000
0.10000
1.800
1
1
GENVOC
6
0.050000
0
01
GENSOMG
1.83
"""


@pytest.fixture(scope="session")
def built_box(som_tomas_src: Path) -> Path:
    """Build box.exe once per session."""
    return build(som_tomas_src)


def test_build_produces_box_exe(built_box: Path) -> None:
    assert built_box.is_file()
    assert built_box.name == "box.exe"


def test_run_returns_runoutputs_with_expected_files(som_tomas_src: Path, built_box: Path) -> None:
    assert built_box.is_file()  # consume the fixture
    outputs = run(som_tomas_src, _SHORT_INPUT)

    assert isinstance(outputs, RunOutputs)
    assert outputs.returncode == 0
    assert outputs.run_name == _RUN_NAME
    assert outputs.outputs_dir.is_dir()

    # The Fortran produces nine files per run. Confirm all of them exist
    # and begin with the run name.
    basenames = {p.name for p in outputs.output_files}
    expected_suffixes = {
        ".dat",
        "_gc.dat",
        "_noconc.dat",
        "_aemass.dat",
        "_spec.dat",
        "_saprcgc.dat",
        "_vl.dat",
        "_tau.dat",
        "_kw.dat",
    }
    for suffix in expected_suffixes:
        assert f"{_RUN_NAME}{suffix}" in basenames, f"expected {_RUN_NAME}{suffix} in {basenames}"

    # Clean up after ourselves; we're the ones who allocated the tempdir.
    shutil.rmtree(outputs.scratch_dir, ignore_errors=True)


def test_run_writes_nonempty_gc_file(som_tomas_src: Path, built_box: Path) -> None:
    assert built_box.is_file()
    outputs = run(som_tomas_src, _SHORT_INPUT)
    try:
        gc_file = outputs.outputs_dir / f"{_RUN_NAME}_gc.dat"
        assert gc_file.is_file()
        assert gc_file.stat().st_size > 0
    finally:
        shutil.rmtree(outputs.scratch_dir, ignore_errors=True)


def test_two_sequential_runs_with_same_runname_in_different_scratches(
    som_tomas_src: Path, built_box: Path
) -> None:
    """STATUS='new' doesn't bite across independent scratch dirs."""
    assert built_box.is_file()
    first = run(som_tomas_src, _SHORT_INPUT)
    second = run(som_tomas_src, _SHORT_INPUT)
    try:
        assert first.scratch_dir != second.scratch_dir
        assert first.returncode == 0
        assert second.returncode == 0
    finally:
        shutil.rmtree(first.scratch_dir, ignore_errors=True)
        shutil.rmtree(second.scratch_dir, ignore_errors=True)


# --- error paths --------------------------------------------------------


def test_build_raises_for_missing_src_dir(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="is not a directory"):
        build(missing)


def test_build_raises_for_missing_makefile(tmp_path: Path) -> None:
    empty = tmp_path / "empty_src"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="simple_makfile"):
        build(empty)


def test_run_raises_when_box_exe_is_missing(tmp_path: Path) -> None:
    empty = tmp_path / "empty_src"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=r"box\.exe"):
        run(empty, "irrelevant\n")


def test_run_raises_fortran_run_error_on_garbage_input(
    som_tomas_src: Path, built_box: Path
) -> None:
    """Feeding box.exe a deliberately malformed input should surface as
    a FortranRunError rather than a silent success."""
    assert built_box.is_file()
    # Only the first line; every subsequent read() will hit EOF.
    garbage = "bad_run_name_\n"
    with pytest.raises(FortranRunError):
        run(som_tomas_src, garbage)
