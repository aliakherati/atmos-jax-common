"""Render visual diagnostics for the C0.8 canonical-run matrix.

Two-panel figure under ``docs/figures/c0.8/``:

1. ``matrix_coverage.png``
   - Top: scatter of all 10 runs in the ``(endtime_h, OH)`` plane,
     marker size encoding GENVOC magnitude and colour encoding
     temperature. Annotated with the run_id.
   - Bottom: GENVOC trajectory over time for every run, read straight
     out of ``data/canonical_runs/expected/<run_id>/<run_id>_saprcgc.dat``.
     Logs both axes to keep the very_long curve readable next to
     short_baseline.

The figure is what reviewers look at before approving a matrix design
change. When master plan Q5 lands and the matrix is revised, this
script just re-runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from atmos_jax_common.canonical_runs import default_manifest_path, load_manifest  # noqa: E402
from atmos_jax_common.goldens import load_golden_run  # noqa: E402

_EXPECTED_DIR = _REPO_ROOT / "data" / "canonical_runs" / "expected"
_OUT = _REPO_ROOT / "docs" / "figures" / "c0.8"


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(_REPO_ROOT)}  ({size_kb:.1f} KB)")


def main() -> int:
    matrix = load_manifest(default_manifest_path())

    fig, (ax_cov, ax_traj) = plt.subplots(2, 1, figsize=(11.0, 9.5))

    # --- Top: coverage scatter --------------------------------------------
    endtimes = np.array([r.params.endtime_h for r in matrix.runs])
    ohs = np.array([r.params.OH_molec_per_cm3 for r in matrix.runs])
    vocs = np.array([r.params.ippmprec_ppm for r in matrix.runs])
    temps = np.array([r.params.temp_K for r in matrix.runs])

    sizes = 80 + 1500 * (vocs - vocs.min()) / (vocs.max() - vocs.min() + 1e-12)
    sc = ax_cov.scatter(
        endtimes,
        ohs,
        s=sizes,
        c=temps,
        cmap="coolwarm",
        edgecolor="black",
        linewidth=0.6,
        vmin=temps.min() - 5,
        vmax=temps.max() + 5,
    )
    cbar = fig.colorbar(sc, ax=ax_cov, label="T (K)")
    cbar.ax.tick_params(labelsize=8)

    for r, x, y in zip(matrix.runs, endtimes, ohs, strict=False):
        ax_cov.annotate(
            r.run_id,
            xy=(x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax_cov.set_xscale("log")
    ax_cov.set_yscale("log")
    ax_cov.set_xlabel("endtime (h, log)")
    ax_cov.set_ylabel("OH (molec cm⁻³, log)")
    ax_cov.set_title("C0.8 canonical-run matrix coverage — marker size ∝ GENVOC initial conc")
    ax_cov.grid(True, which="both", alpha=0.3, linewidth=0.4)

    # Manual legend for marker size scale.
    voc_lo, voc_hi = vocs.min(), vocs.max()
    legend_handles = [
        plt.scatter(
            [],
            [],
            s=80,
            color="lightgray",
            edgecolor="black",
            linewidth=0.6,
            label=f"GENVOC = {voc_lo:.3f} ppm",
        ),
        plt.scatter(
            [],
            [],
            s=80 + 1500,
            color="lightgray",
            edgecolor="black",
            linewidth=0.6,
            label=f"GENVOC = {voc_hi:.3f} ppm",
        ),
    ]
    ax_cov.legend(handles=legend_handles, loc="lower right", fontsize=8, scatterpoints=1)

    # --- Bottom: GENVOC trajectory per run --------------------------------
    cmap = plt.get_cmap("viridis")
    n_runs = len(matrix.runs)
    for i, run in enumerate(matrix.runs):
        run_dir = _EXPECTED_DIR / run.run_id
        if not run_dir.is_dir():
            ax_traj.text(
                0.05,
                0.5,
                "goldens not generated yet — run scripts/generate_goldens.py",
                transform=ax_traj.transAxes,
                fontsize=11,
            )
            break
        golden = load_golden_run(run_dir, run.run_id)
        genvoc_idx = golden.spec.active_gas_species.index("GENVOC")
        t = np.asarray(golden.gc.time_hours)
        y = np.asarray(golden.saprcgc_ppm[:, genvoc_idx])
        # Skip t=0 to keep the log axis happy.
        mask = t > 0
        ax_traj.plot(
            t[mask],
            y[mask],
            label=run.run_id,
            color=cmap(i / max(n_runs - 1, 1)),
            linewidth=1.2,
        )

    ax_traj.set_xscale("log")
    ax_traj.set_yscale("log")
    ax_traj.set_xlabel("time (h, log)")
    ax_traj.set_ylabel("GENVOC (ppm, log)")
    ax_traj.set_title("GENVOC trajectories per run (Fortran ``_saprcgc.dat``)")
    ax_traj.grid(True, which="both", alpha=0.3, linewidth=0.4)
    ax_traj.legend(loc="lower left", fontsize=7, ncol=2)

    fig.tight_layout()
    _save(fig, "matrix_coverage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
