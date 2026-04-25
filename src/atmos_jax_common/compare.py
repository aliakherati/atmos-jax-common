"""Tolerance-aware comparison primitives for trajectory regression.

Used by ``som-jax`` / ``saprc-jax`` / ``tomas-jax`` test suites to compare
JAX-side simulation outputs against the Fortran goldens loaded by
:mod:`atmos_jax_common.goldens`. The primitives are deliberately
unit-agnostic: caller is responsible for choosing trajectories with
matching axes (time, species), aligning real*4 precision via
:mod:`atmos_jax_common.real4`, and converting unit systems with
:mod:`atmos_jax_common.units`.

Conventions
-----------
- Trajectories are 2-D ``(n_timesteps, n_species)`` ``float64`` arrays.
- ``a`` is the candidate (e.g. JAX simulation), ``b`` is the reference
  (Fortran golden). Relative errors are normalised by the reference.
- ``atol`` parameters guard against division by zero when reference
  columns hold all zeros (a common case for species the run never
  produces).

References
----------
- Master plan §6.2 sets the scientific-faithfulness bar: "≤0.1% relative
  L2 per species; correlation ≥ 0.999". The :class:`DiffReport.passes`
  defaults track that target.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DiffReport",
    "carbon_balance",
    "carbon_balance_drift",
    "compare_trajectories",
    "final_time_error",
    "per_species_correlation",
    "relative_l2",
]


# --- per-species primitives ---------------------------------------------


def relative_l2(
    a: ArrayLike,
    b: ArrayLike,
    *,
    atol: float = 1e-30,
) -> NDArray[np.float64]:
    r"""Per-column relative L2 error between two trajectory arrays.

    Defined as ``sqrt(sum((a - b)**2)) / sqrt(sum(b**2) + atol)`` per
    column. ``atol`` is added inside the denominator's square-root to
    keep zero-reference columns at a small but well-defined error
    rather than producing ``inf`` or ``nan``.

    Parameters
    ----------
    a, b
        ``(n_timesteps, n_species)`` arrays. ``a`` is the candidate and
        ``b`` is the reference.
    atol
        Floor on the reference norm. Default ``1e-30`` is well below
        any meaningful concentration scale; raise it if both arrays can
        legitimately reach exact zero.

    Returns
    -------
    ``(n_species,)`` array of relative L2 errors.

    Raises
    ------
    ValueError
        If shapes do not match or inputs are not 2-D.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    _check_2d_match(a_arr, b_arr)
    diff_sq = np.sum((a_arr - b_arr) ** 2, axis=0)
    ref_sq = np.sum(b_arr**2, axis=0)
    out = np.sqrt(diff_sq) / np.sqrt(ref_sq + atol)
    return cast(NDArray[np.float64], out)


def per_species_correlation(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Pearson correlation per column.

    Returns ``1.0`` for columns where either side is constant (no
    variance to correlate against) — an explicit choice so a
    "everything stayed at zero" species doesn't get reported as ``nan``.
    Use :func:`relative_l2` to detect those cases instead.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    _check_2d_match(a_arr, b_arr)

    a_centered = a_arr - a_arr.mean(axis=0, keepdims=True)
    b_centered = b_arr - b_arr.mean(axis=0, keepdims=True)
    a_var = (a_centered**2).sum(axis=0)
    b_var = (b_centered**2).sum(axis=0)
    cov = (a_centered * b_centered).sum(axis=0)
    denom = np.sqrt(a_var * b_var)

    out = np.ones_like(cov)  # default for the "no variance" branch
    nonzero = denom > 0
    out[nonzero] = cov[nonzero] / denom[nonzero]
    # Clamp to [-1, 1] — floating-point can drift epsilon-outside.
    return cast(NDArray[np.float64], np.clip(out, -1.0, 1.0))


def final_time_error(
    a: ArrayLike,
    b: ArrayLike,
    *,
    atol: float = 1e-30,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-species absolute and relative error at the last timestep.

    Returns ``(abs_error, rel_error)`` each of shape ``(n_species,)``.
    Useful as a "did it converge to the right place?" check independent
    of trajectory shape.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    _check_2d_match(a_arr, b_arr)
    abs_err = np.abs(a_arr[-1] - b_arr[-1])
    rel_err = abs_err / (np.abs(b_arr[-1]) + atol)
    return abs_err, rel_err


# --- carbon balance -----------------------------------------------------


def carbon_balance(
    concentrations: ArrayLike,
    carbon_per_species: ArrayLike,
) -> NDArray[np.float64]:
    """Sum of carbon mass across species per timestep.

    For a SOM gas-phase trajectory of shape ``(n_t, n_species)`` and
    ``carbon_per_species`` of shape ``(n_species,)``, returns an
    ``(n_t,)`` array of ``Σ_s carbon_count[s] * conc[t, s]``.

    Useful for sanity-checking conservation: in a system without
    fragmentation losses, total carbon should be flat. With Fortran
    SOM's mass-loss pathways (see ``box.f`` reactions S3.1, S4.1, etc.
    where products have zero yield), it drifts downward.
    """
    c = np.asarray(concentrations, dtype=np.float64)
    counts = np.asarray(carbon_per_species, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError(f"concentrations must be 2-D; got shape {c.shape}")
    if counts.ndim != 1 or counts.shape[0] != c.shape[1]:
        raise ValueError(
            f"carbon_per_species must be 1-D of length {c.shape[1]}; got shape {counts.shape}"
        )
    return c @ counts


def carbon_balance_drift(
    concentrations: ArrayLike,
    carbon_per_species: ArrayLike,
    *,
    atol: float = 1e-30,
) -> float:
    """Maximum relative drift of total carbon away from its t=0 value.

    Returns ``max(|C(t) - C(0)| / (|C(0)| + atol))`` over ``t``. A
    perfectly conserved system returns 0.0; a system that loses or
    gains carbon over time returns a positive number.
    """
    total = carbon_balance(concentrations, carbon_per_species)
    if total.size == 0:
        return 0.0
    initial = float(total[0])
    return float(np.max(np.abs(total - initial)) / (abs(initial) + atol))


# --- structured report --------------------------------------------------


@dataclass(frozen=True)
class DiffReport:
    """Structured per-species comparison summary.

    Attributes
    ----------
    species_names
        Column labels in the same order as :attr:`relative_l2_errors`,
        :attr:`correlations`, and the final-time arrays.
    relative_l2_errors
        ``(n_species,)`` per-species relative L2.
    correlations
        ``(n_species,)`` Pearson correlations.
    final_abs_errors, final_rel_errors
        Absolute and relative errors at the last timestep.
    """

    species_names: tuple[str, ...]
    relative_l2_errors: NDArray[np.float64]
    correlations: NDArray[np.float64]
    final_abs_errors: NDArray[np.float64]
    final_rel_errors: NDArray[np.float64]

    def passes(
        self,
        *,
        l2_tol: float = 1e-3,
        correlation_min: float = 0.999,
        final_rel_tol: float = 1e-3,
    ) -> bool:
        """Whether all three criteria pass with the given thresholds.

        Defaults track the master plan's §6.2 scientific-faithfulness
        bar (≤0.1% relative L2, correlation ≥ 0.999).
        """
        return bool(
            np.all(self.relative_l2_errors <= l2_tol)
            and np.all(self.correlations >= correlation_min)
            and np.all(self.final_rel_errors <= final_rel_tol)
        )

    def worst_species(self, *, by: str = "relative_l2") -> str:
        """Return the species name with the worst score.

        ``by`` is one of ``"relative_l2"``, ``"correlation"``,
        ``"final_rel"``. For ``"correlation"`` "worst" means smallest
        (furthest from 1); the others are largest.
        """
        if by == "relative_l2":
            idx = int(np.argmax(self.relative_l2_errors))
        elif by == "correlation":
            idx = int(np.argmin(self.correlations))
        elif by == "final_rel":
            idx = int(np.argmax(self.final_rel_errors))
        else:
            raise ValueError(f"by={by!r}; expected 'relative_l2', 'correlation', or 'final_rel'")
        return self.species_names[idx]

    def summary(
        self,
        *,
        l2_tol: float = 1e-3,
        correlation_min: float = 0.999,
        final_rel_tol: float = 1e-3,
        worst_n: int = 5,
    ) -> str:
        """Human-readable summary string with the worst few species per
        metric. Useful as the body of an assertion error in test output.
        """
        verdict = (
            "PASS"
            if self.passes(
                l2_tol=l2_tol,
                correlation_min=correlation_min,
                final_rel_tol=final_rel_tol,
            )
            else "FAIL"
        )
        l2_max = float(self.relative_l2_errors.max())
        corr_min_obs = float(self.correlations.min())
        final_max = float(self.final_rel_errors.max())
        worst_l2 = np.argsort(-self.relative_l2_errors)[:worst_n]
        worst_lines = [
            f"  {self.species_names[i]}: L2={self.relative_l2_errors[i]:.2e}, "
            f"corr={self.correlations[i]:.6f}, "
            f"final_rel={self.final_rel_errors[i]:.2e}"
            for i in worst_l2
        ]
        return (
            f"DiffReport({verdict}): "
            f"max(L2)={l2_max:.2e} (tol {l2_tol:.0e}), "
            f"min(corr)={corr_min_obs:.6f} (tol {correlation_min:.6f}), "
            f"max(final_rel)={final_max:.2e} (tol {final_rel_tol:.0e})\n"
            f"top {min(worst_n, len(self.species_names))} species by L2:\n" + "\n".join(worst_lines)
        )


def compare_trajectories(
    a: ArrayLike,
    b: ArrayLike,
    species_names: Sequence[str],
    *,
    atol: float = 1e-30,
) -> DiffReport:
    """Build a :class:`DiffReport` from two trajectory arrays.

    Parameters
    ----------
    a, b
        ``(n_timesteps, n_species)`` arrays — candidate and reference.
    species_names
        Column labels. Length must match ``a.shape[1]``.
    atol
        Tolerance floor passed through to :func:`relative_l2` and
        :func:`final_time_error`.

    Raises
    ------
    ValueError
        If shapes don't match, or if ``species_names`` length doesn't
        match the number of columns.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    _check_2d_match(a_arr, b_arr)
    if len(species_names) != a_arr.shape[1]:
        raise ValueError(
            f"species_names has {len(species_names)} entries; "
            f"trajectory has {a_arr.shape[1]} species columns"
        )

    abs_err, rel_err = final_time_error(a_arr, b_arr, atol=atol)
    return DiffReport(
        species_names=tuple(species_names),
        relative_l2_errors=relative_l2(a_arr, b_arr, atol=atol),
        correlations=per_species_correlation(a_arr, b_arr),
        final_abs_errors=abs_err,
        final_rel_errors=rel_err,
    )


# --- helpers ------------------------------------------------------------


def _check_2d_match(a: NDArray[np.float64], b: NDArray[np.float64]) -> None:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"compare primitives expect 2-D (n_timesteps, n_species) arrays; "
            f"got shapes {a.shape} and {b.shape}"
        )
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a={a.shape}, b={b.shape}")
