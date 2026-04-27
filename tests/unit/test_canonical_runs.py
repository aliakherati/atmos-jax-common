"""Unit tests for ``atmos_jax_common.canonical_runs`` (chunk C0.8).

Verifies the manifest loader, the typed dataclass tree, and the box.exe
input-file renderer. The goldens-side smoke test (running every input
through the actual Fortran reference) lives in ``tests/integration``
since it requires a built ``box.exe``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atmos_jax_common.canonical_runs import (
    CanonicalMatrix,
    CanonicalRun,
    RunOverrides,
    SharedParams,
    default_expected_dir,
    default_inputs_dir,
    default_manifest_path,
    load_manifest,
    render_input,
    write_input,
)


@pytest.fixture(scope="module")
def matrix() -> CanonicalMatrix:
    return load_manifest(default_manifest_path())


# --- manifest schema -----------------------------------------------------


def test_manifest_loads_with_expected_count(matrix: CanonicalMatrix) -> None:
    # Matrix v0.1 commits 10 runs. Bumping this number is a matrix-design
    # decision and should be a deliberate test update.
    assert len(matrix.runs) == 10


def test_run_ids_are_unique(matrix: CanonicalMatrix) -> None:
    ids = [r.run_id for r in matrix.runs]
    assert len(ids) == len(set(ids))


def test_short_baseline_matches_s1_10_fixture_inputs(matrix: CanonicalMatrix) -> None:
    """`short_baseline` is the smoke-test entry; same conditions as the
    som-jax S1.10 fixture so it round-trips through the existing
    sample_run loader without surprises."""
    short = matrix.by_id("short_baseline")
    assert short.params.OH_molec_per_cm3 == pytest.approx(1.5e6)
    assert short.params.ippmprec_ppm == pytest.approx(0.05)
    assert short.params.temp_K == pytest.approx(298.0)


def test_loader_rejects_unknown_schema_version(tmp_path: Path) -> None:
    bogus = {
        "$schema_version": 999,
        "description": "x",
        "shared_params": {},
        "runs": [],
    }
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(bogus))
    with pytest.raises(ValueError, match="schema version"):
        load_manifest(f)


def test_loader_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    base = json.loads(default_manifest_path().read_text())
    # Force duplicate run_ids
    base["runs"][1] = dict(base["runs"][0])
    f = tmp_path / "dup.json"
    f.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(f)


def test_by_id_raises_on_unknown_run(matrix: CanonicalMatrix) -> None:
    with pytest.raises(KeyError):
        matrix.by_id("does_not_exist")


# --- input rendering -----------------------------------------------------


def test_render_input_is_deterministic(matrix: CanonicalMatrix) -> None:
    long = matrix.by_id("long_baseline")
    a = render_input(long, matrix.shared)
    b = render_input(long, matrix.shared)
    assert a == b


def test_render_input_first_line_is_run_id(matrix: CanonicalMatrix) -> None:
    """``box.f`` reads the run name first; downstream output filenames
    depend on this."""
    for run in matrix.runs:
        text = render_input(run, matrix.shared)
        first = text.splitlines()[0]
        assert first == run.run_id


def test_render_input_carries_varying_axis_values(matrix: CanonicalMatrix) -> None:
    """Sanity: each per-run override should appear in the rendered text
    in some recognisable form. We check by formatting the same way the
    renderer does for each axis."""
    for run in matrix.runs:
        text = render_input(run, matrix.shared)
        p = run.params
        # OH is rendered in scientific notation
        assert f"{p.OH_molec_per_cm3:e}" in text, (
            f"run {run.run_id}: OH not found in rendered input"
        )
        # endtime / ippmprec / temp use fixed-point formats
        assert f"{p.endtime_h:8.5f}" in text, (
            f"run {run.run_id}: endtime not found in rendered input"
        )
        assert f"{p.ippmprec_ppm:f}" in text, (
            f"run {run.run_id}: ippmprec not found in rendered input"
        )
        assert f"{p.temp_K:8.5f}" in text, f"run {run.run_id}: temp not found in rendered input"


def test_render_input_uses_precursor_block_when_emissions_on(
    matrix: CanonicalMatrix,
) -> None:
    """Matrix v0.1 always has precspemiss=1, so the precursor block
    (NINPPRECSP / GENVOC / preclenname / IPPMPREC) must always appear."""
    for run in matrix.runs:
        text = render_input(run, matrix.shared)
        assert "GENVOC" in text


def test_write_input_roundtrips(tmp_path: Path, matrix: CanonicalMatrix) -> None:
    short = matrix.by_id("short_baseline")
    target = tmp_path / "short_baseline.txt"
    write_input(short, matrix.shared, target)
    assert target.exists()
    assert target.read_text(encoding="utf-8") == render_input(short, matrix.shared)


# --- committed inputs match the manifest ---------------------------------


def test_committed_inputs_match_manifest(matrix: CanonicalMatrix) -> None:
    """Every committed file under ``data/canonical_runs/inputs/`` must be
    byte-identical to what ``render_input`` produces. CI also enforces
    this via ``scripts/regenerate_canonical_inputs.py --check``; the
    pytest version surfaces it as a unit failure during local dev."""
    inputs_dir = default_inputs_dir()
    assert inputs_dir.is_dir(), f"inputs dir missing: {inputs_dir}"
    committed = sorted(p.stem for p in inputs_dir.glob("*.txt"))
    expected = sorted(r.run_id for r in matrix.runs)
    assert committed == expected
    for run in matrix.runs:
        committed_path = inputs_dir / f"{run.run_id}.txt"
        assert committed_path.read_text(encoding="utf-8") == render_input(run, matrix.shared), (
            f"committed input {committed_path} drifted from manifest; "
            f"run scripts/regenerate_canonical_inputs.py to refresh."
        )


def test_default_expected_dir_resolves_to_committed_goldens(
    matrix: CanonicalMatrix,
) -> None:
    """Every run in the matrix has a corresponding subdirectory under
    ``default_expected_dir()`` containing Fortran-generated goldens
    plus a ``metadata.json``. Verifies the
    ``[tool.hatch.build.targets.wheel.force-include]`` packaging keeps
    the data discoverable in both editable and wheel installs."""
    expected_dir = default_expected_dir()
    assert expected_dir.is_dir(), f"expected dir missing: {expected_dir}"
    for run in matrix.runs:
        run_dir = expected_dir / run.run_id
        assert run_dir.is_dir(), f"missing golden dir: {run_dir}"
        assert (run_dir / "metadata.json").is_file(), f"missing metadata.json in {run_dir}"
        assert (run_dir / f"{run.run_id}_saprcgc.dat").is_file(), (
            f"missing _saprcgc.dat in {run_dir}"
        )


# --- shared params sanity ------------------------------------------------


def test_shared_params_have_emissions_on(matrix: CanonicalMatrix) -> None:
    """All matrix v0.1 runs use precursor emissions (GENVOC). The
    renderer's precursor branch is therefore always exercised."""
    assert matrix.shared.emissions.precspemiss == 1
    assert matrix.shared.emissions.precname == "GENVOC"


def test_shared_params_disable_aerosol_coupling(matrix: CanonicalMatrix) -> None:
    """Gas-phase regression coverage: VWL/PWL/COAG off so soacond and
    multicoag don't contaminate the chemistry trajectories. Aerosol
    coverage belongs in the tomas-jax matrix later."""
    assert matrix.shared.switches.coag == 0
    assert matrix.shared.switches.vwl == 0
    assert matrix.shared.switches.pwl == 0


# --- typing ---------------------------------------------------------------


def test_dataclasses_are_frozen() -> None:
    """The matrix tree should be immutable so it's safe to share between
    tests / fixtures without accidental mutation."""
    matrix = load_manifest(default_manifest_path())
    # frozen dataclasses raise FrozenInstanceError (a subclass of AttributeError)
    # on any attribute mutation attempt.
    with pytest.raises(AttributeError):
        matrix.runs[0].params.endtime_h = 999.0  # type: ignore[misc]


def test_run_overrides_construction_validation() -> None:
    """Direct construction with the documented fields should work."""
    overrides = RunOverrides(
        endtime_h=4.0,
        OH_molec_per_cm3=1.5e6,
        ippmprec_ppm=0.05,
        temp_K=298.0,
    )
    assert overrides.endtime_h == 4.0


def test_shared_params_construction_documented_via_loader(matrix: CanonicalMatrix) -> None:
    """The dataclass tree is exposed; constructing manually requires
    knowing the nested layout. Loader is the supported path."""
    assert isinstance(matrix.shared, SharedParams)
    assert isinstance(matrix.runs[0], CanonicalRun)
