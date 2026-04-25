r"""Generate diagnostics for the goldens loader (C0.6).

One panel under ``docs/figures/c0.6/``:

1. ``loaded_run_overview.png`` — a 4-panel render of the committed
   sample fixture (``tests/fixtures/sample_run/``). Shows that the
   loader's typed structure exposes the right pieces of a Fortran
   run: gas-phase ppm, size-distribution number conc, total aerosol
   mass per timestep, and the SOM (C, O) grid coloured by c\*.

The fixture is short (only two save points: t=0 and t~=0.167 h) so
"trajectories" are stem plots rather than smooth curves. The point
of the figure is structural — to show what fields a ``GoldenRun``
actually carries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from atmos_jax_common.goldens import load_golden_run  # noqa: E402

_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "sample_run"
_RUN_NAME = "sample_for_loader_dev"
_OUT = _REPO_ROOT / "docs" / "figures" / "c0.6"


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(_REPO_ROOT)}  ({size_kb:.1f} KB)")


def main() -> int:
    g = load_golden_run(_FIXTURE_DIR, _RUN_NAME)
    spec = g.spec

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    fig.suptitle(
        f"GoldenRun overview — fixture '{_RUN_NAME}'  "
        f"({g.n_timesteps()} timesteps; gc={g.gc.values.shape[1]} cols, "
        f"saprcgc={g.saprcgc_ppm.shape[1]} cols, "
        f"aemass={g.aemass_kg.shape})",
        fontsize=11,
    )

    # --- Panel A: GENVOC + first-gen GENSOMG_07_x products from gc -----
    ax = axes[0, 0]
    # gc.values columns map to the SOM species in spec.som_species order
    # for the SOM block. The first columns (0..n_som-1) align with the
    # SOM species; we plot a few first-gen products at the t=t_final point
    # to show non-zero values (t=0 is by construction zero for products).
    targets = ["GENSOMG_07_01", "GENSOMG_07_02", "GENSOMG_07_03", "GENSOMG_07_04"]
    times_h = g.gc.time_hours
    for name in targets:
        series = g.gc_for_species(name)
        ax.plot(
            times_h,
            np.maximum(series, 1e-30),
            marker="o",
            linewidth=1.2,
            label=name,
        )
    ax.set_yscale("log")
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("gas concentration  (ppm, log scale)")
    ax.set_title("(a) gc.values: first-gen GENSOMG products")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3, linewidth=0.4)

    # --- Panel B: Number concentration per bin at the final timestep ----
    ax = axes[0, 1]
    bins = np.arange(g.noconc.values.shape[1])
    final_idx = g.n_timesteps() - 1
    ax.bar(
        bins,
        g.noconc.values[final_idx, :],
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel("size-bin index")
    ax.set_ylabel("number concentration  (cm$^{-3}$)")
    ax.set_title(
        f"(b) noconc.values at t = {float(times_h[final_idx]):.3f} h "
        f"(IBINS = {g.noconc.values.shape[1]})"
    )

    # --- Panel C: total aerosol mass over time --------------------------
    ax = axes[1, 0]
    # aemass_kg is (n_t, n_aerosol_species, n_bins). Sum across species
    # and bins to get total aerosol mass per timestep.
    total_mass = g.aemass_kg.sum(axis=(1, 2))
    ax.plot(g.aemass_time_hours, total_mass, marker="o", color="#2ca02c", linewidth=1.5)
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("total aerosol mass  (kg in bag)")
    ax.set_title("(c) aemass_kg.sum(axis=(species, bin)) — total aerosol mass")
    ax.grid(True, alpha=0.3, linewidth=0.4)

    # --- Panel D: SOM grid coloured by c* -------------------------------
    ax = axes[1, 1]
    c_max = int(spec.som_carbon.max())
    o_max = int(spec.som_oxygen.max())
    grid = np.full((o_max + 1, c_max + 1), np.nan)
    for i in range(len(spec.som_species)):
        c = int(spec.som_carbon[i])
        o = int(spec.som_oxygen[i])
        grid[o, c] = float(spec.som_cstar[i])
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("#eeeeee")  # unpopulated cells in light gray
    im = ax.imshow(
        grid,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(
            vmin=max(1e-3, np.nanmin(grid)),
            vmax=np.nanmax(grid),
        ),
    )
    ax.set_xticks(np.arange(c_max + 1))
    ax.set_yticks(np.arange(o_max + 1))
    ax.set_xlabel("carbon")
    ax.set_ylabel("oxygen")
    ax.set_title("(d) spec.som_cstar over the (C, O) grid (log colour)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"c$^\ast$  ($\mu$g m$^{-3}$)")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "loaded_run_overview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
