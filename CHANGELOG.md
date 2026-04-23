# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial package scaffolding: `pyproject.toml`, `src/` layout, MIT license.
- CI workflow: pytest + ruff + mypy on Python 3.11 and 3.12 (macOS + Linux, CPU only).
- README and CHANGELOG with accurate project status.
- **C0.2 — units module**: `atmos_jax_common.units` ports the exact ppm ↔ molec/cm³ ↔ kg/bag conversions used by `som-tomas-app/src/box.f` (lines 416, 592, 604–605, 805–806). Exposes physical constants (`R = 8.314`, `N_A = 6.022e23`, `MW_AIR_KG_PER_MOL = 0.0289`) pinned to the Fortran values — not CODATA — so round-trip tolerances stay in single-ULP territory. Also `box_air_mass_kg(boxvol, T, P)` as a sanity-check helper. NumPy-array inputs supported via `np.asarray`. 19 unit tests include line-by-line parity checks against the Fortran formulas.
- Ruff config adds `RUF002` / `RUF003` to the ignore list — unicode markup (`×`, `↔`, `²`, `³`, `⁻¹`, `—`) is deliberate in docstrings and not an ambiguity hazard.
- **C0.3 — species-name parser**: `atmos_jax_common.species` provides `parse_species_name`, `format_species_name`, `is_som_species`, and a frozen `SpeciesIdentifier` dataclass. Recognises `FAMILY_CC_OO` strings (uppercase family starting with a letter; two-digit zero-padded C and O). 39 unit tests cover parsing, formatting, round-trips, rejection of invalid inputs, hashability, and the full 40-species GENSOMG and 47-species AR1SOMG grids from the Fortran reference. Total: 60 tests pass.

## [0.1.0] — planned

First usable release. Gate: all scientific model repos (`som-jax`, `saprc-jax`, `tomas-jax`) can load golden outputs from this package and run regression tests.
