"""Generate a DiffReport demo figure for the compare primitives (C0.7).

One panel under ``docs/figures/c0.7/``:

1. ``diff_report_demo.png`` — synthetic Fortran-vs-JAX scenario:
   take the committed sample fixture's ``saprcgc_ppm`` as the "Fortran
   reference", build a "JAX candidate" by perturbing it with a fixed
   per-species multiplicative bias, then render the resulting
   ``DiffReport``. Top: per-species relative L2 with the master-plan
   tolerance line. Bottom: candidate-vs-reference overlay for the
   species with the worst L2.

The figure is the natural rendering of a DiffReport — what users see
when an S1.10-style regression fails. Demonstrates that the API
correctly identifies the offending species.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from atmos_jax_common.compare import compare_trajectories  # noqa: E402
from atmos_jax_common.goldens import load_golden_run  # noqa: E402

_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "sample_run"
_RUN_NAME = "sample_for_loader_dev"
_OUT = _REPO_ROOT / "docs" / "figures" / "c0.7"


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(_REPO_ROOT)}  ({size_kb:.1f} KB)")


def main() -> int:
    g = load_golden_run(_FIXTURE_DIR, _RUN_NAME)

    # Use the SOM block of saprcgc as the reference. It's the rightmost
    # 40 columns by Fortran-write convention. The 2-timestep fixture is
    # short but enough to render a representative bar chart.
    n_som = len(g.spec.som_species)
    ref = g.saprcgc_ppm[:, -n_som:].copy()
    species_names = list(g.spec.som_species)

    # Build a synthetic "JAX candidate" with a per-species bias seeded
    # so that one species is markedly worse than the rest. This mimics
    # the kind of failure mode S1.10 will eventually catch (a single
    # incorrect rate constant or stoichiometry coefficient).
    rng = np.random.default_rng(0)
    bias = 1.0 + 0.001 * rng.standard_normal(n_som)
    bias[5] = 1.05  # 5% bias on a single species
    candidate = ref * bias[np.newaxis, :]

    report = compare_trajectories(candidate, ref, species_names)

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.5),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.55},
    )

    # --- top: per-species relative L2 bar chart ------------------------
    x = np.arange(n_som)
    l2 = np.asarray(report.relative_l2_errors)
    bars = ax_top.bar(x, l2, color="#1f77b4", edgecolor="black", linewidth=0.3)
    worst_idx = int(np.argmax(l2))
    bars[worst_idx].set_color("#d62728")  # highlight the worst species
    tol = 1e-3
    ax_top.axhline(
        tol,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label=f"master-plan tolerance ({tol:g})",
    )
    ax_top.set_yscale("log")
    ax_top.set_ylim(max(1e-12, l2[l2 > 0].min() / 5), l2.max() * 3)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(species_names, rotation=90, fontsize=6)
    ax_top.set_ylabel("relative L2 error  (log scale)")
    ax_top.set_title(
        "DiffReport: per-species relative L2 between candidate and reference  "
        f"(verdict: {'PASS' if report.passes() else 'FAIL'})"
    )
    ax_top.legend(loc="upper right", fontsize=9)
    ax_top.grid(True, which="both", linewidth=0.3, alpha=0.4)

    # Annotate the worst species.
    ax_top.annotate(
        f"worst: {report.worst_species()} (L2={l2[worst_idx]:.2e})",
        xy=(worst_idx, l2[worst_idx]),
        xytext=(0.02, 0.95),
        textcoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        color="#d62728",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d62728", lw=0.6),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
    )

    # --- bottom: candidate vs reference for the worst species ----------
    times = g.gc.time_hours
    name = species_names[worst_idx]
    ax_bot.plot(times, ref[:, worst_idx], "o-", color="#d62728", linewidth=2.0, label="reference")
    ax_bot.plot(
        times,
        candidate[:, worst_idx],
        "s--",
        color="#1f77b4",
        linewidth=1.5,
        label="candidate",
    )
    ax_bot.set_xlabel("time (hours)")
    ax_bot.set_ylabel(f"[{name}]  (ppm)")
    ax_bot.set_title(f"Worst species: {name} — candidate vs reference")
    ax_bot.legend(loc="best", fontsize=9)
    ax_bot.grid(True, alpha=0.3, linewidth=0.4)

    _save(fig, "diff_report_demo.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
