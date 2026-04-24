# atmos-jax-common

Shared infrastructure for the SOM-TOMAS Python/JAX port. Provides Fortran-reference plumbing (build + run wrappers, golden-output loaders, unit conversions, tolerance-aware comparators) used by the sibling model packages:

- [`som-jax`](https://github.com/aliakherati/som-jax) — Statistical Oxidation Model
- `saprc-jax` *(not yet created)* — explicit gas-phase chemistry (SAPRC)
- `tomas-jax` *(not yet created)* — Two-Moment Aerosol Sectional microphysics

This package has **no scientific logic of its own**. It exists so the three model repos don't each reimplement the same Fortran-runner + goldens-loader plumbing.

## Status

**Early alpha — scaffolding only.** Current state:

- Repository created, package skeleton installable (`pip install -e .`).
- CI configured (pytest + ruff + mypy on CPU).
- No public API yet. Modules land incrementally; see [project status](#project-status) below.

## Install

```bash
git clone https://github.com/aliakherati/atmos-jax-common.git
cd atmos-jax-common
pip install -e ".[dev]"
```

Requires Python ≥ 3.11.

## Project status

| Module | Purpose | Status |
|---|---|---|
| `atmos_jax_common.units` | ppm ↔ molec/cm³ ↔ kg/bag conversions | alpha (C0.2) |
| `atmos_jax_common.species` | SOM species-name parser (e.g., `GENSOMG_04_03` → `(C=4, O=3)`) | alpha (C0.3) |
| `atmos_jax_common.real4` | `float64` → `float32` downcast for faithful-mode comparison | not started |
| `atmos_jax_common.fortran_runner` | Subprocess wrapper: build `box.exe`, run with input, capture outputs | alpha (C0.4) |
| `atmos_jax_common.goldens` | Parser/loader for `_gc.dat`, `_noconc.dat`, `_aemass.dat`, etc. | not started |
| `atmos_jax_common.compare` | Tolerance-aware diff primitives (`relative_l2`, correlation, carbon balance) | not started |

Tracked in the master plan at `~/.claude/plans/enchanted-exploring-dewdrop.md` (owner's local) as chunks `C0.0` … `C0.10`.

## Relationship to the Fortran reference

The Fortran box model lives in a separate repository (not yet public). This package drives it via subprocess to generate golden outputs for regression testing in the three model packages. You need a working `gfortran` (or `ifort`) to regenerate goldens; the committed golden datasets in `data/canonical_runs/*/expected/` can be consumed without the Fortran toolchain.

### Running the Fortran reference from Python

Once `gfortran` is installed and the Fortran source is available, run:

```python
from pathlib import Path
from atmos_jax_common.fortran_runner import build, run

src = Path("som-tomas-app/src")
build(src)  # compiles box.exe in place (runs `make clean` first by default)

input_text = Path("som-tomas-app/src/input").read_text()
result = run(src, input_text)          # scratch dir allocated automatically

print(result.stdout[-500:])             # Fortran stdout tail
print(result.outputs_dir)               # <scratch>/outputs/
for p in result.output_files:
    print(p.relative_to(result.outputs_dir))
```

Each `run()` gets its own scratch directory (symlinked source + fresh `outputs/`) so the Fortran's ``STATUS='new'`` ``OPEN`` never collides with prior outputs. See `tests/integration/test_fortran_runner.py` for the canonical input shape. Tests are gated by the ``SOM_TOMAS_APP_SRC`` environment variable and skip cleanly when unset.

## License

MIT — see [LICENSE](LICENSE).

## Citation

This repo itself is infrastructure; scientific citations live with the model packages (`som-jax`, `saprc-jax`, `tomas-jax`).
