"""Mimic Fortran ``REAL*4`` precision loss for faithful-mode comparisons.

Several Fortran output files in ``som-tomas-app`` are written in
single-precision (``REAL*4``) even when the internal computation was
double-precision. Notably ``SAPRCGC`` (written to ``_gc.dat``) and the
scratch arrays in ``box.f`` are real*4, while ``GC`` and most internal
arrays are ``DOUBLE PRECISION``.

When comparing a JAX (float64) trajectory against a Fortran golden file
read as ppm values, the golden has already lost ~23 bits of mantissa to
the single-precision round-trip. To do a faithful comparison — i.e. to
test how close the JAX port is to what the Fortran user actually *sees*
— downcast the JAX output through float32 and back to float64 before
taking the diff.

This module provides the round-trip helper plus a convenience
``compare_with_real4_precision`` wrapper that aligns two arrays to the
same real*4-lossy precision.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["downcast_to_real4", "real4_epsilon", "with_real4_precision"]


def downcast_to_real4(x: ArrayLike) -> NDArray[np.float64]:
    """Round-trip ``x`` through float32 and back to float64.

    The resulting float64 array carries only the ~7 decimal digits of
    mantissa that survive a ``REAL*4`` write + read cycle. Useful for
    comparing against Fortran output files written in single precision.

    Parameters
    ----------
    x
        Array-like to downcast. Non-float inputs are first cast to
        float64 for consistency.

    Returns
    -------
    float64 array of the same shape as ``x``, but containing only
    single-precision-representable values.
    """
    arr = np.asarray(x, dtype=np.float64)
    return arr.astype(np.float32).astype(np.float64)


def with_real4_precision(
    a: ArrayLike, b: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Downcast both ``a`` and ``b`` through real*4 so they compare
    apples-to-apples with a Fortran golden.

    Returns the two arrays as float64 but with single-precision content.
    Shorthand for ``(downcast_to_real4(a), downcast_to_real4(b))``.
    """
    return downcast_to_real4(a), downcast_to_real4(b)


def real4_epsilon() -> float:
    """Machine epsilon of float32, exposed as a float64 value.

    Useful as a lower bound on meaningful relative-error tolerances when
    comparing against Fortran real*4 output. Any tolerance below
    :func:`real4_epsilon` is not distinguishable from rounding noise in
    the Fortran write.
    """
    return float(np.finfo(np.float32).eps)
