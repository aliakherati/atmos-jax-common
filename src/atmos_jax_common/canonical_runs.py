"""Canonical-run matrix for the SOM-TOMAS Fortran reference (chunk C0.8).

Defines the typed view over ``data/canonical_runs/manifest.json`` plus the
glue needed to deterministically render each entry into a ``box.exe``
stdin file. The matrix is the contract between the goldens generator
(``scripts/generate_goldens.py``) and the mechanism-drift CI guard.

The Fortran ``box.f`` ``read`` calls expect specific FORMAT specifiers
per line (see ``box.f`` lines 173-323); the writer in this module
mirrors those formats so the produced text round-trips through
``box.exe`` without parser drift. Every committed input under
``data/canonical_runs/inputs/<run_id>.txt`` is regenerable from the
manifest via ``write_input``; CI verifies they stay in sync.

Versioning
----------
``manifest.json`` carries a ``$schema_version`` field. Bumps to that
field are breaking changes to the matrix; loaders should refuse
unfamiliar versions rather than silently misinterpret. The matrix
itself is at v0.1 — open to revision when master plan Q5 is answered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _SwitchesParams:
    coag: int
    vwl: int
    pwl: int


@dataclass(frozen=True)
class _RatesParams:
    ke: float
    kw0: float


@dataclass(frozen=True)
class _BoxParams:
    nucrate_per_cm3_per_s: float
    dilution_rate: float
    boxvol_cm3: float
    pres_Pa: float
    rh: float


@dataclass(frozen=True)
class _ParticlesParams:
    alpha: float
    Dbk_m2_per_s: float
    kc_per_s: float
    storg_N_per_m: float
    No_per_cm3: float
    Dpm_um: float
    sigma: float


@dataclass(frozen=True)
class _EmissionsParams:
    precspemiss: int
    ninpprec: int
    precname: str
    preclenname: int
    oxyspemiss: int


@dataclass(frozen=True)
class _SomParams:
    nsomprec: int
    somprecname: str
    dlvp: float


@dataclass(frozen=True)
class SharedParams:
    """Run-invariant configuration. The varying axes are in :class:`RunOverrides`."""

    switches: _SwitchesParams
    rates: _RatesParams
    box: _BoxParams
    particles: _ParticlesParams
    emissions: _EmissionsParams
    som: _SomParams


@dataclass(frozen=True)
class RunOverrides:
    """Per-run varying parameters."""

    endtime_h: float
    OH_molec_per_cm3: float
    ippmprec_ppm: float
    temp_K: float


@dataclass(frozen=True)
class CanonicalRun:
    """A single run in the matrix."""

    run_id: str
    summary: str
    params: RunOverrides


@dataclass(frozen=True)
class CanonicalMatrix:
    """The full canonical-run matrix as parsed from ``manifest.json``."""

    description: str
    shared: SharedParams
    runs: tuple[CanonicalRun, ...]

    def by_id(self, run_id: str) -> CanonicalRun:
        for run in self.runs:
            if run.run_id == run_id:
                return run
        raise KeyError(f"unknown run_id {run_id!r}; known: {[r.run_id for r in self.runs]}")


# --- loading --------------------------------------------------------------


def load_manifest(path: Path | str) -> CanonicalMatrix:
    """Load ``manifest.json`` into a typed :class:`CanonicalMatrix`.

    Raises
    ------
    ValueError
        If the schema version is unfamiliar or required keys are missing.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = raw.get("$schema_version")
    if schema != _MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unfamiliar manifest schema version {schema!r}; this loader supports "
            f"{_MANIFEST_SCHEMA_VERSION!r}. Bump the loader before running new manifests."
        )
    s = raw["shared_params"]
    # JSON uses uppercase keys for COAG/VWL/PWL to match Fortran symbol
    # names; the Python dataclass fields are lowercase.
    sw = s["switches"]
    shared = SharedParams(
        switches=_SwitchesParams(coag=sw["COAG"], vwl=sw["VWL"], pwl=sw["PWL"]),
        rates=_RatesParams(**s["rates"]),
        box=_BoxParams(**s["box"]),
        particles=_ParticlesParams(**s["particles"]),
        emissions=_EmissionsParams(**s["emissions"]),
        som=_SomParams(**s["som"]),
    )
    runs = tuple(
        CanonicalRun(
            run_id=r["run_id"],
            summary=r["summary"],
            params=RunOverrides(**r["params"]),
        )
        for r in raw["runs"]
    )
    if len({r.run_id for r in runs}) != len(runs):
        dups = sorted({r.run_id for r in runs if sum(1 for x in runs if x.run_id == r.run_id) > 1})
        raise ValueError(f"duplicate run_id(s) in manifest: {dups}")
    return CanonicalMatrix(description=raw["description"], shared=shared, runs=runs)


# --- input file rendering -------------------------------------------------


def render_input(run: CanonicalRun, shared: SharedParams) -> str:
    """Render a ``CanonicalRun`` to a ``box.exe`` stdin string.

    Mirrors the FORMAT specifiers in ``box.f`` lines 173-323. A trailing
    newline is included so the file ends cleanly.
    """
    p = run.params
    s = shared
    lines = [
        run.run_id,
        f"{s.switches.coag:1d}",
        f"{s.switches.vwl:1d}",
        f"{s.switches.pwl:1d}",
        f"{s.rates.ke:f}",
        f"{s.rates.kw0:f}",
        f"{p.OH_molec_per_cm3:e}",
        f"{s.box.nucrate_per_cm3_per_s:7.1f}",
        f"{s.box.dilution_rate:7.4f}",
        f"{s.box.boxvol_cm3:15.1f}",
        f"{p.endtime_h:8.5f}",
        f"{s.particles.alpha:8.5f}",
        f"{s.particles.Dbk_m2_per_s:.2e}",
        f"{s.particles.kc_per_s:f}",
        f"{s.particles.storg_N_per_m:8.5f}",
        f"{s.box.pres_Pa:12.3f}",
        f"{p.temp_K:8.5f}",
        f"{s.box.rh:8.5f}",
        f"{s.particles.No_per_cm3:8.5f}",
        f"{s.particles.Dpm_um:8.5f}",
        f"{s.particles.sigma:8.5f}",
        f"{s.emissions.precspemiss:1d}",
    ]
    if s.emissions.precspemiss == 1:
        lines.extend(
            [
                f"{s.emissions.ninpprec:3d}",
                s.emissions.precname,
                f"{s.emissions.preclenname}",
                f"{p.ippmprec_ppm:f}",
            ]
        )
    lines.append(f"{s.emissions.oxyspemiss:1d}")
    # If oxyspemiss == 1 the input would carry the oxygenated-species block
    # here. The matrix v0.1 only uses precursor-only emissions, so we omit
    # that branch.
    lines.extend(
        [
            f"{s.som.nsomprec:02d}",
            s.som.somprecname,
            f"{s.som.dlvp:6.3f}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_input(run: CanonicalRun, shared: SharedParams, target: Path | str) -> None:
    """Write ``render_input(...)`` to ``target``."""
    Path(target).write_text(render_input(run, shared), encoding="utf-8")


# --- defaults --------------------------------------------------------------
#
# The canonical-run matrix data ships in two layouts depending on how the
# package was installed:
#
# - **Editable / source checkout**: data lives at repo root under
#   ``data/canonical_runs/``. The path resolves as
#   ``__file__/../../../data/canonical_runs/`` (parents[2] from
#   ``src/atmos_jax_common/canonical_runs.py``).
# - **Wheel install** (``pip install atmos-jax-common``): the data is
#   force-included into the package as ``atmos_jax_common/_data/
#   canonical_runs/`` (per the ``[tool.hatch.build.targets.wheel.force-include]``
#   section of ``pyproject.toml``). The path resolves as
#   ``__file__/../_data/canonical_runs/``.
#
# Helpers below probe both locations and return the first that exists.
# Consumers (``som-jax``, ``saprc-jax``, ``tomas-jax``) use these paths
# unchanged so the same code works in both layouts.


def _canonical_runs_data_root() -> Path:
    """Resolve the on-disk root for the canonical-run data, regardless
    of editable-vs-wheel install mode."""
    pkg_dir = Path(__file__).resolve().parent  # src/atmos_jax_common/
    candidates = [
        pkg_dir / "_data" / "canonical_runs",  # wheel install
        pkg_dir.parents[1] / "data" / "canonical_runs",  # editable / source
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        "Cannot locate the canonical-run data on disk. Looked at:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def default_manifest_path() -> Path:
    """Return the on-disk path to the committed manifest."""
    return _canonical_runs_data_root() / "manifest.json"


def default_inputs_dir() -> Path:
    """Return the on-disk directory holding committed input files."""
    return _canonical_runs_data_root() / "inputs"


def default_expected_dir() -> Path:
    """Return the on-disk directory holding committed reference outputs.

    Each canonical run has a subdirectory ``<run_id>/`` containing the
    Fortran-generated ``.dat`` files plus a ``metadata.json`` recording
    the input + Fortran-source SHAs at generation time. Use this together
    with :func:`atmos_jax_common.goldens.load_golden_run` to load any
    canonical run's reference trajectory.
    """
    return _canonical_runs_data_root() / "expected"
