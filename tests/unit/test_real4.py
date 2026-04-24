"""Unit tests for :mod:`atmos_jax_common.real4`."""

from __future__ import annotations

import numpy as np
import pytest

from atmos_jax_common.real4 import (
    downcast_to_real4,
    real4_epsilon,
    with_real4_precision,
)

# --- basic round-trip ---------------------------------------------------


def test_exact_floats_survive_round_trip() -> None:
    # Powers of two representable in both float32 and float64 are unchanged.
    x = np.array([0.0, 1.0, 0.5, -0.25, 1024.0, 1 / 16])
    np.testing.assert_array_equal(downcast_to_real4(x), x)


def test_integer_input_is_coerced_to_float64_array() -> None:
    result = downcast_to_real4([1, 2, 3])
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_shape_preserved() -> None:
    x = np.random.default_rng(0).normal(size=(3, 4, 5))
    assert downcast_to_real4(x).shape == x.shape


def test_dtype_always_float64_even_though_precision_is_float32() -> None:
    x = np.array([1.0, 2.0, 3.0])
    result = downcast_to_real4(x)
    assert result.dtype == np.float64


# --- precision-loss behaviour ------------------------------------------


def test_ulp_level_float64_difference_is_lost() -> None:
    # Two numbers differing by less than float32 resolution at 1.0 collapse
    # to the same value after downcast.
    a = 1.0
    b = 1.0 + 1e-9  # well below float32 eps (~1.2e-7)
    lossy_a = float(downcast_to_real4(a))
    lossy_b = float(downcast_to_real4(b))
    assert lossy_a == lossy_b


def test_inexact_decimal_loses_precision_at_the_float32_level() -> None:
    """0.1 is not exactly representable in either float32 or float64.
    Downcasting to float32 loses ~23 bits of mantissa: the round-trip
    error should be ~float32-eps."""
    x = 0.1
    lossy = float(downcast_to_real4(x))
    abs_err = abs(lossy - x)
    # Error should be within an ULP of float32's mantissa at 0.1 (~6e-9).
    assert 0 < abs_err < 1e-7
    # And well above float64's eps (~2e-16).
    assert abs_err > 1e-12


def test_large_magnitude_precision_scales_with_value() -> None:
    """Absolute round-trip error scales roughly linearly with magnitude
    (relative error is roughly constant at ~float32-eps)."""
    for magnitude in (1e-5, 1e0, 1e5, 1e10):
        x = 1.123456789 * magnitude
        lossy = float(downcast_to_real4(x))
        rel_err = abs(lossy - x) / x
        assert rel_err < real4_epsilon() * 5  # 5x eps is generous slack


# --- special values ----------------------------------------------------


def test_nan_passes_through() -> None:
    result = downcast_to_real4(np.array([np.nan, 1.0, np.nan]))
    assert np.isnan(result[0])
    assert not np.isnan(result[1])
    assert np.isnan(result[2])


def test_positive_infinity_passes_through() -> None:
    result = downcast_to_real4(np.array([np.inf, 1.0]))
    assert np.isposinf(result[0])


def test_negative_infinity_passes_through() -> None:
    result = downcast_to_real4(np.array([-np.inf, 1.0]))
    assert np.isneginf(result[0])


def test_values_well_below_float32_min_underflow_to_zero() -> None:
    # Float32 subnormal minimum is ~1.4e-45. 1e-50 is far below that and
    # rounds to 0 on the float32 cast.
    x = 1e-50
    lossy = float(downcast_to_real4(x))
    assert lossy == 0.0


def test_values_in_float32_subnormal_range_lose_precision_but_survive() -> None:
    # 1e-40 is a float32 subnormal — representable but with degraded
    # precision. The round-trip preserves order of magnitude.
    x = 1e-40
    lossy = float(downcast_to_real4(x))
    assert 0 < lossy < 1e-39  # same order of magnitude, not zero


def test_values_above_float32_overflow_become_inf() -> None:
    x = 1e40  # float32 max is ~3.4e38
    # numpy emits an overflow warning; we suppress it at the test level
    # because the behaviour under test *is* the overflow.
    with np.errstate(over="ignore"):
        lossy = float(downcast_to_real4(x))
    assert np.isposinf(lossy)


# --- with_real4_precision ----------------------------------------------


def test_with_real4_precision_returns_two_downcasts() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.1, 2.1, 3.1])
    la, lb = with_real4_precision(a, b)
    np.testing.assert_array_equal(la, downcast_to_real4(a))
    np.testing.assert_array_equal(lb, downcast_to_real4(b))


def test_with_real4_precision_aligns_sub_ulp_differences() -> None:
    """Two float64 values differing by less than float32 eps compare
    equal after the shared round-trip."""
    a = 1.2345678
    b = a + 1e-9  # below float32 resolution near 1.0
    la, lb = with_real4_precision(a, b)
    assert float(la) == float(lb)


# --- real4_epsilon -----------------------------------------------------


def test_real4_epsilon_is_positive_and_well_below_1() -> None:
    eps = real4_epsilon()
    assert 0 < eps < 1e-5


def test_real4_epsilon_matches_numpy_finfo() -> None:
    assert real4_epsilon() == pytest.approx(float(np.finfo(np.float32).eps))
