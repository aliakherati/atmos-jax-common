"""Unit tests for :mod:`atmos_jax_common.goldens`.

Uses fixtures committed under ``tests/fixtures/sample_run/`` — actual
output files from a short ``box.exe`` run with ``runme.py`` defaults
(``GENVOC + OH``, ``OH=1.5e6``, ``endtime=0.10 h``). These are tiny
(~80 KB total) and stable across rebuilds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atmos_jax_common.goldens import (
    FortranOutputBlock,
    GoldenRun,
    _load_aemass,
    load_block_file,
    load_golden_run,
    load_spec_file,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "sample_run"
_RUN_NAME = "sample_for_loader_dev"


# --- block parser -------------------------------------------------------


def test_load_gc_block_returns_2d_values_with_time_axis() -> None:
    block = load_block_file(_FIXTURE_DIR / f"{_RUN_NAME}_gc.dat")
    assert isinstance(block, FortranOutputBlock)
    assert block.time_hours.ndim == 1
    assert block.values.ndim == 2
    assert block.time_hours.shape[0] == block.values.shape[0]
    # First row of the sample is t=0.
    assert float(block.time_hours[0]) == pytest.approx(0.0, abs=1e-12)


def test_load_gc_block_dtypes_are_float64() -> None:
    block = load_block_file(_FIXTURE_DIR / f"{_RUN_NAME}_gc.dat")
    assert block.time_hours.dtype == np.float64
    assert block.values.dtype == np.float64


def test_load_block_promotes_single_row_to_2d() -> None:
    """Sanity: even a single-timestep file must come back as
    ``(1, n_cols)`` so callers don't need to special-case the 1-D path."""
    block = load_block_file(_FIXTURE_DIR / f"{_RUN_NAME}_kw.dat")
    assert block.values.ndim == 2


def test_load_block_rejects_too_narrow_files(tmp_path: Path) -> None:
    p = tmp_path / "single_col.dat"
    p.write_text("0.0\n0.5\n1.0\n")
    with pytest.raises(ValueError, match="at least 2 columns"):
        load_block_file(p)


# --- spec parser --------------------------------------------------------


def test_spec_parses_active_gas_species_list() -> None:
    spec = load_spec_file(_FIXTURE_DIR / f"{_RUN_NAME}_spec.dat")
    # Sanity: the SAPRC active gas species list is large and includes
    # well-known names.
    assert "O3" in spec.active_gas_species
    assert "NO2" in spec.active_gas_species
    assert "GENVOC" in spec.active_gas_species


def test_spec_parses_40_som_species() -> None:
    spec = load_spec_file(_FIXTURE_DIR / f"{_RUN_NAME}_spec.dat")
    assert len(spec.som_species) == 40
    assert spec.som_species[0] == "GENSOMG_01_01"
    assert spec.som_species[-1] == "GENSOMG_07_07"


def test_spec_carbon_oxygen_arrays_have_correct_dtypes_and_lengths() -> None:
    spec = load_spec_file(_FIXTURE_DIR / f"{_RUN_NAME}_spec.dat")
    assert spec.som_carbon.dtype == np.int32
    assert spec.som_oxygen.dtype == np.int32
    assert spec.som_carbon.shape == (40,)
    assert spec.som_oxygen.shape == (40,)
    # Carbon ranges 1..7, oxygen ranges 1..7.
    assert int(spec.som_carbon.min()) >= 1
    assert int(spec.som_carbon.max()) <= 7
    assert int(spec.som_oxygen.min()) >= 1
    assert int(spec.som_oxygen.max()) <= 7


def test_spec_carbon_oxygen_match_species_names() -> None:
    """Cross-check: each SOM name encodes its (C, O); they must agree
    with the per-species C/O arrays the Fortran wrote alongside."""
    spec = load_spec_file(_FIXTURE_DIR / f"{_RUN_NAME}_spec.dat")
    for i, name in enumerate(spec.som_species):
        # Names are GENSOMG_CC_OO.
        _, cc, oo = name.split("_")
        assert int(cc) == int(spec.som_carbon[i])
        assert int(oo) == int(spec.som_oxygen[i])


def test_spec_mw_arrays_align_with_carbon_oxygen() -> None:
    spec = load_spec_file(_FIXTURE_DIR / f"{_RUN_NAME}_spec.dat")
    assert spec.som_mwt.shape == (40,)
    # The first GENSOMG_01_01 species: MW from the .mod file is 31.0.
    assert float(spec.som_mwt[0]) == pytest.approx(31.0, abs=1e-3)
    # GENSOMG_07_07: 205.0.
    assert float(spec.som_mwt[-1]) == pytest.approx(205.0, abs=1e-3)


def test_spec_cstar_and_hvap_are_finite_and_positive() -> None:
    spec = load_spec_file(_FIXTURE_DIR / f"{_RUN_NAME}_spec.dat")
    assert spec.som_cstar.shape == (40,)
    assert np.all(np.isfinite(spec.som_cstar))
    assert np.all(spec.som_cstar > 0)
    assert spec.som_hvap.shape == (40,)
    assert np.all(np.isfinite(spec.som_hvap))
    assert np.all(spec.som_hvap > 0)


def test_spec_rejects_files_missing_expected_labels(tmp_path: Path) -> None:
    p = tmp_path / "partial.dat"
    p.write_text("ACTIVE GAS SP: O3 NO\n")
    with pytest.raises(ValueError, match="missing expected label"):
        load_spec_file(p)


# --- top-level GoldenRun loader -----------------------------------------


@pytest.fixture(scope="module")
def golden() -> GoldenRun:
    return load_golden_run(_FIXTURE_DIR, _RUN_NAME)


def test_golden_run_has_run_name_and_dir(golden: GoldenRun) -> None:
    assert golden.run_name == _RUN_NAME
    assert golden.outputs_dir == _FIXTURE_DIR


def test_golden_blocks_share_a_common_time_axis(golden: GoldenRun) -> None:
    """All *time-major* files (gc, noconc, aemass) record the same save
    points. ``_saprcgc.dat`` has no time column and is excluded; its
    rows align with ``gc.time_hours`` by Fortran-write convention."""
    np.testing.assert_allclose(golden.gc.time_hours, golden.noconc.time_hours, rtol=0, atol=1e-6)
    np.testing.assert_allclose(golden.gc.time_hours, golden.aemass_time_hours, rtol=0, atol=1e-6)


def test_saprcgc_row_count_matches_gc_time_axis(golden: GoldenRun) -> None:
    """saprcgc has no explicit time column, but each row corresponds
    to one timestep written synchronously with ``_gc.dat``."""
    assert golden.saprcgc_ppm.shape[0] == golden.gc.time_hours.shape[0]
    assert golden.saprcgc_ppm.shape[1] == len(golden.spec.active_gas_species)


def test_golden_aemass_shape(golden: GoldenRun) -> None:
    """aemass reshapes to (n_timesteps, 43, 26): 43 ICOMP species, 26
    IBINS bins."""
    n_t = golden.n_timesteps()
    assert golden.aemass_kg.shape == (n_t, 43, 26)


def test_golden_aemass_values_are_finite_and_nonnegative(golden: GoldenRun) -> None:
    a = golden.aemass_kg
    assert np.all(np.isfinite(a))
    # Aerosol masses are >= 0.
    assert np.all(a >= 0)


def test_gc_columns_align_with_som_species(golden: GoldenRun) -> None:
    """The first columns of the gc block correspond to SOM species in
    spec.som_species order (the Fortran writes Gc(2..iorg+1) which are
    the 40 SOM organics; column 0 of gc.values aligns with
    spec.som_species[0])."""
    n_som = len(golden.spec.som_species)
    assert golden.gc.values.shape[1] >= n_som


def test_gc_for_species_returns_named_column(golden: GoldenRun) -> None:
    target = "GENSOMG_07_04"
    series = golden.gc_for_species(target)
    assert series.shape == (golden.n_timesteps(),)
    # GENSOMG_07_04 is the highest-yield first-gen product; non-trivially
    # populated even after a short run.
    assert np.all(np.isfinite(series))


def test_gc_for_unknown_species_raises(golden: GoldenRun) -> None:
    with pytest.raises(ValueError):
        golden.gc_for_species("NOT_A_SPECIES")


def test_load_golden_run_raises_on_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_golden_run(tmp_path, "nonexistent_run")


# --- aemass reshape edge case -------------------------------------------


def test_aemass_loader_rejects_inconsistent_row_count(tmp_path: Path) -> None:
    """If row count is not a multiple of n_aerosol_species, fail fast
    rather than silently mis-reshape."""
    # Write a tiny synthetic file with 5 rows + 4 columns. n_species=2
    # would divide cleanly; n_species=3 would not.
    p = tmp_path / "fake_aemass.dat"
    rows = [
        "0.0 1.0 2.0 3.0",
        "0.0 4.0 5.0 6.0",
        "0.0 7.0 8.0 9.0",
        "1.0 1.0 2.0 3.0",
        "1.0 4.0 5.0 6.0",
    ]
    p.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="not divisible by n_aerosol_species"):
        _load_aemass(p, n_aerosol_species=3)
