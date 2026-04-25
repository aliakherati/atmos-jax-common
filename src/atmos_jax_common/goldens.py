"""Parse Fortran SOM-TOMAS output files into typed numpy arrays.

The Fortran reference (``som-tomas-fortran``) writes nine output files per
run, all under ``../outputs/`` relative to ``box.exe``'s working directory.
This module parses the subset that downstream regression tests care about
into a single :class:`GoldenRun` dataclass.

File layouts (all whitespace-separated scientific-notation text):

- ``<run>_gc.dat`` — SOM gas-phase concentrations (ppm). Each row is a
  timestep; column 0 is time (hours), columns 1.. are species
  concentrations in the order written by ``box.f``. Real*4 precision.
- ``<run>_saprcgc.dat`` — full SAPRC active-gas-species concentrations
  (ppm). Same row layout, hundreds of columns. Real*4.
- ``<run>_noconc.dat`` — number concentration per size bin
  (particles cm⁻³). Column 0 time, columns 1..ibins per bin.
- ``<run>_aemass.dat`` — aerosol mass per (species, bin) per timestep.
  Stored as a tall block: each timestep contributes ``icomp`` consecutive
  rows (one per aerosol species) of ``1 + ibins`` columns. Reshape into
  ``(n_timesteps, icomp, ibins)`` for ergonomic indexing.
- ``<run>_spec.dat`` — labeled metadata for the active gas species,
  the SOM family species, and per-SOM-species C/O/MW/c*/HVAP/PSTAR.
- ``<run>_vl.dat``, ``<run>_tau.dat``, ``<run>_kw.dat`` — auxiliary
  diagnostics. Same time-major shape as ``_gc.dat``; not parsed by
  default but available via :func:`load_block_file`.

References
----------
- ``som-tomas-fortran/src/box.f`` — write loops at lines 1037+ (gc),
  1090+ (saprcgc), 1180+ (noconc), and similar.
- :mod:`atmos_jax_common.species` — `parse_species_name` for splitting
  the SOM names returned in :class:`SpecMetadata` back into ``(family,
  carbon, oxygen)`` triples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FortranOutputBlock",
    "GoldenRun",
    "SpecMetadata",
    "load_block_file",
    "load_golden_run",
    "load_saprcgc_file",
    "load_spec_file",
]


# --- low-level block parser ---------------------------------------------


@dataclass(frozen=True)
class FortranOutputBlock:
    """A single time-major Fortran output file's contents.

    Attributes
    ----------
    path
        Source file path.
    time_hours
        ``(n_timesteps,)`` first-column values (time in hours).
    values
        ``(n_timesteps, n_columns)`` remaining columns. Order matches
        the file's column order; species-name lookup belongs to higher-
        level structures (see :class:`GoldenRun`).
    """

    path: Path
    time_hours: NDArray[np.float64]
    values: NDArray[np.float64]


def load_block_file(path: Path | str) -> FortranOutputBlock:
    """Parse a time-major Fortran output file (e.g. ``_gc.dat``).

    Each row is a timestep. Column 0 is time in hours; columns 1.. are
    the per-row data values. The Fortran writes these in real*4 with
    free-format scientific notation (``"   0.123456E+00   ..."``).

    Use :func:`load_saprcgc_file` for ``_saprcgc.dat``, which omits the
    time column and stores species concentrations directly.
    """
    p = Path(path)
    # ndmin=2 keeps single-row and single-column files honest; without it
    # np.loadtxt collapses both to 1-D and we can't tell them apart.
    raw = np.loadtxt(p, dtype=np.float64, ndmin=2)
    if raw.shape[1] < 2:
        raise ValueError(f"{p}: expected at least 2 columns (time + 1 value); got {raw.shape[1]}")
    return FortranOutputBlock(path=p, time_hours=raw[:, 0], values=raw[:, 1:])


def load_saprcgc_file(
    path: Path | str, *, n_active_species: int | None = None
) -> NDArray[np.float64]:
    """Parse ``<run>_saprcgc.dat`` — full SAPRC active-species gas state.

    Unlike ``_gc.dat`` this file has **no time column**: each row is a
    timestep and every column is a species concentration in ppm. The
    column ordering matches :attr:`SpecMetadata.active_gas_species` from
    the matching ``_spec.dat``.

    Parameters
    ----------
    path
        Path to the ``_saprcgc.dat`` file.
    n_active_species
        If provided, raises :class:`ValueError` when the parsed column
        count does not match. ``None`` (default) skips the check.

    Returns
    -------
    A ``(n_timesteps, n_active_species)`` ``float64`` array.
    """
    p = Path(path)
    raw = np.loadtxt(p, dtype=np.float64, ndmin=2)
    if n_active_species is not None and raw.shape[1] != n_active_species:
        raise ValueError(
            f"{p}: got {raw.shape[1]} columns; expected {n_active_species} "
            "matching ACTIVE GAS SP in the spec file."
        )
    return raw


# --- spec.dat parser ----------------------------------------------------


@dataclass(frozen=True)
class SpecMetadata:
    """Parsed contents of ``<run>_spec.dat``.

    Holds species lists and per-SOM-species physical properties as written
    by the Fortran. Species ordering in the trailing arrays matches
    :attr:`som_species`.
    """

    active_gas_species: tuple[str, ...]
    som_species: tuple[str, ...]
    som_carbon: NDArray[np.int32]
    som_oxygen: NDArray[np.int32]
    som_mwt: NDArray[np.float64]
    som_mworg: NDArray[np.float64]
    som_mworg_hc: NDArray[np.float64]
    som_cstar: NDArray[np.float64]
    som_hvap: NDArray[np.float64]
    som_pstar: NDArray[np.float64]


_SPEC_LABEL_RE = re.compile(r"^\s*(?P<label>[^:]+?)\s*:\s*(?P<rest>.*)$")
_SPEC_LABELS = {
    "active gas sp": "active_gas_species",
    "som sp": "som_species",
    "som carbon no": "som_carbon",
    "som oxygen no": "som_oxygen",
    "mwt of som": "som_mwt",
    "mworg of som": "som_mworg",
    "mworg_hc of som": "som_mworg_hc",
    "cstar": "som_cstar",
    "hvap": "som_hvap",
    "pstar": "som_pstar",
}


def load_spec_file(path: Path | str) -> SpecMetadata:
    """Parse the labeled metadata in ``<run>_spec.dat``.

    Each line has the form ``"   <label>:    <whitespace-separated values>"``
    where ``<label>`` is one of the known headings (``ACTIVE GAS SP``,
    ``SOM SP``, ``SOM CARBON NO``, ...). String labels yield species
    names; numeric labels yield ``int32`` (counts) or ``float64`` arrays.
    """
    p = Path(path)
    fields: dict[str, list[str]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        m = _SPEC_LABEL_RE.match(line)
        if m is None:
            continue
        label = m.group("label").strip().lower()
        key = _SPEC_LABELS.get(label)
        if key is None:
            continue
        fields[key] = m.group("rest").split()

    missing = set(_SPEC_LABELS.values()) - fields.keys()
    if missing:
        raise ValueError(
            f"{p}: missing expected label(s): {sorted(missing)}. "
            "The Fortran reference may have omitted them; check box.f's "
            "spec-file write loop."
        )

    return SpecMetadata(
        active_gas_species=tuple(fields["active_gas_species"]),
        som_species=tuple(fields["som_species"]),
        som_carbon=np.asarray(fields["som_carbon"], dtype=np.int32),
        som_oxygen=np.asarray(fields["som_oxygen"], dtype=np.int32),
        som_mwt=np.asarray(fields["som_mwt"], dtype=np.float64),
        som_mworg=np.asarray(fields["som_mworg"], dtype=np.float64),
        som_mworg_hc=np.asarray(fields["som_mworg_hc"], dtype=np.float64),
        som_cstar=np.asarray(fields["som_cstar"], dtype=np.float64),
        som_hvap=np.asarray(fields["som_hvap"], dtype=np.float64),
        som_pstar=np.asarray(fields["som_pstar"], dtype=np.float64),
    )


# --- aemass parser (3-D reshape) ----------------------------------------


def _load_aemass(
    path: Path, n_aerosol_species: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Parse ``_aemass.dat`` as a ``(n_timesteps, n_species, n_bins)`` array.

    Fortran writes one row per (timestep, aerosol-species), so the file
    has ``n_timesteps * n_species`` rows. Each row's first column is the
    timestep's time in hours (repeated across all species in that
    timestep), and the remaining columns are the per-bin masses.

    Returns ``(time_hours, mass_3d)`` where ``mass_3d`` has shape
    ``(n_timesteps, n_species, n_bins)``.
    """
    raw = np.loadtxt(path, dtype=np.float64, ndmin=2)
    n_rows, n_cols = raw.shape
    if n_rows % n_aerosol_species != 0:
        raise ValueError(
            f"{path}: row count {n_rows} not divisible by n_aerosol_species "
            f"{n_aerosol_species}; cannot reshape to "
            "(n_timesteps, n_species, n_bins)."
        )
    n_timesteps = n_rows // n_aerosol_species
    n_bins = n_cols - 1

    # Verify the time column is consistent within each species block.
    times_repeated = raw[:, 0].reshape(n_timesteps, n_aerosol_species)
    if not np.allclose(times_repeated, times_repeated[:, [0]], atol=0):
        raise ValueError(
            f"{path}: time column is not constant across species rows within "
            "the same timestep; aemass file layout has changed."
        )
    time_hours = times_repeated[:, 0]
    mass_3d = raw[:, 1:].reshape(n_timesteps, n_aerosol_species, n_bins)
    return time_hours, mass_3d


# --- top-level loader ---------------------------------------------------


@dataclass(frozen=True)
class GoldenRun:
    """All Fortran outputs for a single canonical run.

    Attributes
    ----------
    run_name
        Run identifier; matches the prefix of every output file's basename.
    outputs_dir
        Directory containing the parsed files.
    spec
        Parsed ``_spec.dat`` metadata.
    gc
        ``_gc.dat`` block — SOM gas-phase concentrations (ppm). Columns
        of ``gc.values`` are species in the order written by the Fortran;
        in practice that aligns with :attr:`SpecMetadata.som_species`
        for the SOM family columns.
    saprcgc_ppm
        ``(n_timesteps, n_active_species)`` SAPRC active-species gas
        concentrations from ``_saprcgc.dat``. Columns align with
        :attr:`SpecMetadata.active_gas_species`. Time axis is implicit;
        ``saprcgc_ppm[i]`` corresponds to ``gc.time_hours[i]`` because
        the Fortran writes both files synchronously.
    noconc
        ``_noconc.dat`` block — particle number concentration per bin
        (particles cm⁻³).
    aemass_kg
        ``(n_timesteps, n_aerosol_species, n_bins)`` aerosol mass (kg).
        Built from ``_aemass.dat`` by reshaping the time-and-species-
        major file into 3-D.
    aemass_time_hours
        ``(n_timesteps,)`` time axis for ``aemass_kg``. Equal to
        ``gc.time_hours`` and ``noconc.time_hours`` to within float
        precision in well-behaved runs; not deduplicated automatically
        because callers may want to verify the Fortran's per-block
        consistency themselves.
    """

    run_name: str
    outputs_dir: Path
    spec: SpecMetadata
    gc: FortranOutputBlock
    saprcgc_ppm: NDArray[np.float64]
    noconc: FortranOutputBlock
    aemass_kg: NDArray[np.float64]
    aemass_time_hours: NDArray[np.float64]

    def n_timesteps(self) -> int:
        return int(self.gc.time_hours.size)

    def gc_for_species(self, name: str) -> NDArray[np.float64]:
        """Return ``(n_timesteps,)`` GC ppm trajectory for one SOM species.

        Looks the species up in :attr:`SpecMetadata.som_species`; the
        column index in ``gc.values`` is the same as the species's
        position in that tuple.
        """
        idx = self.spec.som_species.index(name)
        return self.gc.values[:, idx]


def load_golden_run(
    outputs_dir: Path | str,
    run_name: str,
    *,
    n_aerosol_species: int = 43,
) -> GoldenRun:
    """Load all relevant Fortran output files for a single run.

    Parameters
    ----------
    outputs_dir
        Directory containing the per-run files (typically the
        ``outputs/`` subdir of a runner scratch tree).
    run_name
        Per-run prefix written by ``box.f`` (the first stdin line).
    n_aerosol_species
        Number of aerosol species per timestep written into
        ``_aemass.dat``. Defaults to ``43`` to match
        ``ICOMP=43`` in ``sizecode.COM``. Override only if a future
        Fortran build changes the size.

    Raises
    ------
    FileNotFoundError
        If any of ``_gc.dat``, ``_saprcgc.dat``, ``_noconc.dat``,
        ``_aemass.dat``, or ``_spec.dat`` is missing.
    """
    out = Path(outputs_dir)

    def _path(suffix: str) -> Path:
        p = out / f"{run_name}{suffix}"
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found; expected output of run {run_name!r}")
        return p

    spec = load_spec_file(_path("_spec.dat"))
    gc = load_block_file(_path("_gc.dat"))
    saprcgc_ppm = load_saprcgc_file(
        _path("_saprcgc.dat"), n_active_species=len(spec.active_gas_species)
    )
    noconc = load_block_file(_path("_noconc.dat"))
    aemass_time, aemass_kg = _load_aemass(_path("_aemass.dat"), n_aerosol_species)

    return GoldenRun(
        run_name=run_name,
        outputs_dir=out,
        spec=spec,
        gc=gc,
        saprcgc_ppm=saprcgc_ppm,
        noconc=noconc,
        aemass_kg=aemass_kg,
        aemass_time_hours=aemass_time,
    )
