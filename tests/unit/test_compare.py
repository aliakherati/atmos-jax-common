"""Unit tests for :mod:`atmos_jax_common.compare`."""

from __future__ import annotations

import numpy as np
import pytest

from atmos_jax_common.compare import (
    carbon_balance,
    carbon_balance_drift,
    compare_trajectories,
    final_time_error,
    per_species_correlation,
    relative_l2,
)

# --- relative_l2 --------------------------------------------------------


def test_relative_l2_identical_inputs_are_zero() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_array_equal(relative_l2(a, a), [0.0, 0.0])


def test_relative_l2_doubled_inputs_give_unit_relative_error() -> None:
    """If a = 2*b then |a - b| = |b|, so relative L2 = ||b||/||b|| = 1."""
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    a = 2 * b
    np.testing.assert_allclose(relative_l2(a, b), [1.0, 1.0], rtol=1e-12)


def test_relative_l2_zero_reference_does_not_explode_to_inf() -> None:
    """A column of zero in the reference must not yield inf/nan.
    With atol=1e-30 floor, a zero-vs-zero comparison returns 0; a
    nonzero-vs-zero returns a finite (though huge) number."""
    a_zero = np.zeros((3, 1))
    b_zero = np.zeros((3, 1))
    np.testing.assert_array_equal(relative_l2(a_zero, b_zero), [0.0])

    a_nonzero = np.array([[1.0], [1.0], [1.0]])
    out = relative_l2(a_nonzero, b_zero)
    assert np.isfinite(out[0])
    assert out[0] > 0


def test_relative_l2_rejects_shape_mismatch() -> None:
    a = np.zeros((3, 2))
    b = np.zeros((3, 3))
    with pytest.raises(ValueError, match="shape mismatch"):
        relative_l2(a, b)


def test_relative_l2_rejects_non_2d_inputs() -> None:
    with pytest.raises(ValueError, match="2-D"):
        relative_l2(np.zeros(5), np.zeros(5))


# --- per_species_correlation -------------------------------------------


def test_correlation_identical_columns_are_1() -> None:
    a = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 2.0]])
    np.testing.assert_allclose(per_species_correlation(a, a), [1.0, 1.0], rtol=1e-12)


def test_correlation_perfectly_anticorrelated_columns_are_minus_1() -> None:
    a = np.array([[1.0], [2.0], [3.0], [4.0]])
    b = -a
    np.testing.assert_allclose(per_species_correlation(a, b), [-1.0], rtol=1e-12)


def test_correlation_with_constant_column_returns_one_by_convention() -> None:
    """Constant columns have zero variance, so Pearson correlation is
    undefined. We return 1.0 by convention (so a "stayed at zero"
    species doesn't pollute a passes() check via NaN propagation)."""
    a = np.zeros((5, 2))
    b = np.zeros((5, 2))
    np.testing.assert_array_equal(per_species_correlation(a, b), [1.0, 1.0])


def test_correlation_clamps_to_unit_interval() -> None:
    """Floating-point arithmetic can produce 1 + epsilon; clamping
    keeps downstream comparisons honest."""
    a = np.array([[1.0], [2.0], [3.0], [4.0]])
    out = per_species_correlation(a, a + 1e-15)
    assert -1.0 <= float(out[0]) <= 1.0


# --- final_time_error --------------------------------------------------


def test_final_time_error_picks_last_row() -> None:
    a = np.array([[10.0, 0.0], [0.0, 0.0], [3.0, 7.0]])
    b = np.array([[10.0, 0.0], [0.0, 0.0], [3.5, 7.5]])
    abs_err, rel_err = final_time_error(a, b)
    np.testing.assert_allclose(abs_err, [0.5, 0.5], rtol=1e-12)
    np.testing.assert_allclose(rel_err, [0.5 / 3.5, 0.5 / 7.5], rtol=1e-12)


def test_final_time_error_zero_reference_does_not_explode() -> None:
    a = np.array([[0.0, 0.0], [1e-9, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 0.0]])
    abs_err, rel_err = final_time_error(a, b)
    assert np.all(np.isfinite(abs_err))
    assert np.all(np.isfinite(rel_err))


# --- carbon_balance ----------------------------------------------------


def test_carbon_balance_sums_carbon_mass() -> None:
    """Total C at each timestep = Σ_s carbon[s] * conc[t, s]."""
    conc = np.array([[1.0, 2.0, 0.5], [0.5, 1.0, 1.0]])
    carbon = np.array([1.0, 2.0, 4.0])  # C counts
    expected = np.array([1.0 * 1 + 2.0 * 2 + 0.5 * 4, 0.5 * 1 + 1.0 * 2 + 1.0 * 4])
    np.testing.assert_allclose(carbon_balance(conc, carbon), expected, rtol=1e-12)


def test_carbon_balance_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        carbon_balance(np.zeros((2, 3)), np.zeros((4,)))


def test_carbon_balance_rejects_non_2d_concentrations() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        carbon_balance(np.zeros(3), np.zeros(3))


def test_carbon_balance_drift_zero_for_conserving_system() -> None:
    """If total C is constant, drift is exactly 0."""
    # Two species each with C=1; whatever the parent loses, the product
    # gains. Total C = sum is constant at 1 across all timesteps.
    conc_conserving = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.25, 0.75],
        ]
    )
    carbon = np.array([1.0, 1.0])
    drift = carbon_balance_drift(conc_conserving, carbon)
    assert drift == pytest.approx(0.0, abs=1e-15)


def test_carbon_balance_drift_nonzero_for_lossy_system() -> None:
    """An aggregate that loses carbon shows a positive drift."""
    # Start at C=2, decay to C=1 — 50% loss.
    conc = np.array([[2.0], [1.5], [1.0]])
    carbon = np.array([1.0])
    drift = carbon_balance_drift(conc, carbon)
    assert drift == pytest.approx(0.5, rel=1e-12)


def test_carbon_balance_drift_handles_empty_input() -> None:
    drift = carbon_balance_drift(np.zeros((0, 3)), np.array([1.0, 2.0, 3.0]))
    assert drift == 0.0


# --- compare_trajectories + DiffReport ---------------------------------


def test_compare_trajectories_identical_inputs_pass() -> None:
    a = np.linspace(0.0, 1.0, 10).reshape(5, 2)
    names = ("X", "Y")
    report = compare_trajectories(a, a, names)
    assert report.passes()
    assert np.all(report.relative_l2_errors == 0)
    np.testing.assert_allclose(report.correlations, [1.0, 1.0])


def test_compare_trajectories_perturbed_input_fails_default_threshold() -> None:
    rng = np.random.default_rng(42)
    base = rng.uniform(1.0, 10.0, size=(20, 5))
    perturbed = base * (1 + 0.1 * rng.standard_normal(base.shape))  # 10% noise
    names = tuple(f"sp{i}" for i in range(5))
    report = compare_trajectories(perturbed, base, names)
    assert not report.passes()  # 10% noise is well above 0.1% L2 threshold


def test_diff_report_passes_threshold_overrides() -> None:
    """A 1% deviation passes a 5% threshold but fails a 0.1% threshold."""
    base = np.full((10, 1), 1.0)
    candidate = base * 1.01
    report = compare_trajectories(candidate, base, ("X",))
    assert report.passes(l2_tol=0.1, correlation_min=0.0, final_rel_tol=0.1)
    assert not report.passes(l2_tol=1e-4)


def test_diff_report_worst_species_by_l2() -> None:
    base = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    candidate = np.array([[1.0, 1.0, 1.0], [2.0, 2.5, 2.0]])  # only species 1 (Y) drifts
    names = ("X", "Y", "Z")
    report = compare_trajectories(candidate, base, names)
    assert report.worst_species(by="relative_l2") == "Y"


def test_diff_report_summary_includes_metric_lines(capsys: pytest.CaptureFixture) -> None:
    base = np.full((5, 2), 1.0)
    candidate = base.copy()
    candidate[-1, 1] = 1.05  # 5% bump on the last row of species 1
    names = ("X", "Y")
    report = compare_trajectories(candidate, base, names)
    text = report.summary()
    assert "DiffReport" in text
    assert "max(L2)" in text
    assert "min(corr)" in text
    # Y is the offending species.
    assert "Y" in text


def test_compare_trajectories_rejects_wrong_species_count() -> None:
    a = np.zeros((3, 4))
    with pytest.raises(ValueError, match="species_names has"):
        compare_trajectories(a, a, ["X", "Y"])
