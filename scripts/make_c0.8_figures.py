"""Render visual diagnostics for the C0.8 canonical-run matrix.

The matrix v0.1 is structured as a *baseline* (``long_baseline``: 4 h,
OH=1.5e6, GENVOC=0.05, T=298 K) plus four families of runs that each
sweep ONE axis while holding the others at the baseline. The figure
shows that structure with one panel per axis, so each varying axis is
visually distinct rather than collapsing onto a single (x, y) scatter
point.

Two-panel figure under ``docs/figures/c0.8/``:

1. ``matrix_coverage.png``
   - Top (4 small panels): one per varying axis. Within each panel the
     runs that vary that axis are shown along a 1-D line, with the
     baseline run highlighted. Each panel makes its own axis tick
     scaling explicit so "what does this matrix cover?" is unambiguous.
   - Bottom: GENVOC(t) trajectory per run, drawn straight from the
     committed ``_saprcgc.dat``. Curves are colour-coded by which
     family they belong to (time, OH, VOC, T) so the per-axis effect is
     easy to read.
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

# Group runs by the axis they vary. Family colours feed both the
# small-multiples panels and the trajectory legend so a reader can
# trace a run from its family bar to its trajectory.
_FAMILIES: dict[str, dict] = {
    "time": {
        "axis_label": "endtime (h, log)",
        "axis_attr": "endtime_h",
        "log": True,
        "members": ("short_baseline", "medium_baseline", "long_baseline", "very_long"),
        "colour": "#1f77b4",
    },
    "OH": {
        "axis_label": "OH (molec cm⁻³, log)",
        "axis_attr": "OH_molec_per_cm3",
        "log": True,
        "members": ("low_oh", "long_baseline", "high_oh"),
        "colour": "#d62728",
    },
    "VOC": {
        "axis_label": "GENVOC (ppm, log)",
        "axis_attr": "ippmprec_ppm",
        "log": True,
        "members": ("low_voc", "long_baseline", "high_voc"),
        "colour": "#2ca02c",
    },
    "T": {
        "axis_label": "temperature (K)",
        "axis_attr": "temp_K",
        "log": False,
        "members": ("cold", "long_baseline", "hot"),
        "colour": "#9467bd",
    },
}
_BASELINE_RUN_ID = "long_baseline"


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(_REPO_ROOT)}  ({size_kb:.1f} KB)")


def _draw_axis_panel(ax: plt.Axes, family_key: str, runs_by_id: dict) -> None:
    fam = _FAMILIES[family_key]
    members = fam["members"]
    values = np.asarray([getattr(runs_by_id[rid].params, fam["axis_attr"]) for rid in members])
    y = np.zeros_like(values)
    ax.plot(values, y, color=fam["colour"], linewidth=1.5, alpha=0.6, zorder=1)
    # Alternate label vertical offsets to avoid overlap when values
    # are close on a log axis (e.g., the 4-member time family).
    label_offsets = [40, 14] if len(members) >= 4 else [14] * len(members)
    for idx, (rid, v) in enumerate(zip(members, values, strict=False)):
        is_baseline = rid == _BASELINE_RUN_ID
        ax.scatter(
            v,
            0.0,
            s=240 if is_baseline else 130,
            facecolor="white" if is_baseline else fam["colour"],
            edgecolor=fam["colour"],
            linewidth=2.0 if is_baseline else 1.0,
            zorder=2,
        )
        label = rid + ("\n(baseline)" if is_baseline else "")
        offset_y = label_offsets[idx % len(label_offsets)]
        ax.annotate(
            label,
            xy=(v, 0.0),
            xytext=(0, offset_y),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        if fam["axis_attr"] == "OH_molec_per_cm3":
            value_str = f"{v:.1e}"
        elif fam["axis_attr"] == "ippmprec_ppm":
            value_str = f"{v:.3g} ppm"
        elif fam["axis_attr"] == "temp_K":
            value_str = f"{v:.0f} K"
        else:
            value_str = f"{v:g} h"
        ax.annotate(
            value_str,
            xy=(v, 0.0),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7,
            color="dimgray",
        )

    if fam["log"]:
        ax.set_xscale("log")
    ax.set_xlabel(fam["axis_label"], fontsize=9)
    ax.set_yticks([])
    # Extra headroom when we stagger labels for 4+-member families.
    ax.set_ylim(-0.5, 0.9 if len(members) >= 4 else 0.5)
    if fam["log"]:
        ax.set_xlim(values.min() * 0.5, values.max() * 2.0)
    else:
        ax.set_xlim(values.min() - 10, values.max() + 10)
    ax.set_title(f"{family_key} family", fontsize=10, color=fam["colour"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.4)


def _family_for_run(run_id: str) -> str:
    if run_id == _BASELINE_RUN_ID:
        return "baseline"
    for fkey, fam in _FAMILIES.items():
        if run_id in fam["members"]:
            return fkey
    return "other"


def main() -> int:
    matrix = load_manifest(default_manifest_path())
    runs_by_id = {r.run_id: r for r in matrix.runs}

    fig = plt.figure(figsize=(12.0, 9.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.5], hspace=0.45, wspace=0.4)
    for i, key in enumerate(_FAMILIES):
        ax = fig.add_subplot(gs[0, i])
        _draw_axis_panel(ax, key, runs_by_id)

    fig.suptitle(
        "C0.8 canonical-run matrix v0.1 — 10 runs, one varying axis at a time",
        fontsize=12,
        y=0.985,
    )

    ax_traj = fig.add_subplot(gs[1, :])

    plotted_any = False
    family_styles = {
        "time": ("#1f77b4", "-"),
        "OH": ("#d62728", "-"),
        "VOC": ("#2ca02c", "-"),
        "T": ("#9467bd", "-"),
        "baseline": ("black", "--"),
    }
    for run in matrix.runs:
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
        fam = _family_for_run(run.run_id)
        colour, ls = family_styles.get(fam, ("#7f7f7f", "-"))
        golden = load_golden_run(run_dir, run.run_id)
        genvoc_idx = golden.spec.active_gas_species.index("GENVOC")
        t = np.asarray(golden.gc.time_hours)
        y = np.asarray(golden.saprcgc_ppm[:, genvoc_idx])
        mask = t > 0
        ax_traj.plot(
            t[mask],
            y[mask],
            color=colour,
            linestyle=ls,
            linewidth=2.0 if run.run_id == _BASELINE_RUN_ID else 1.0,
            alpha=0.95 if run.run_id == _BASELINE_RUN_ID else 0.8,
            label=run.run_id,
        )
        plotted_any = True

    if plotted_any:
        ax_traj.set_xscale("log")
        ax_traj.set_yscale("log")
        ax_traj.set_xlabel("time (h, log)")
        ax_traj.set_ylabel("GENVOC (ppm, log)")
        ax_traj.set_title(
            "GENVOC(t) per run — colour = family (blue=time, red=OH, green=VOC, "
            "purple=T), dashed black = long_baseline",
            fontsize=10,
        )
        ax_traj.grid(True, which="both", alpha=0.3, linewidth=0.4)
        ax_traj.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.95)

    _save(fig, "matrix_coverage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
