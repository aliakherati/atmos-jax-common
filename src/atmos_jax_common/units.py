"""Unit conversions for the Fortran SOM-TOMAS reference.

The Fortran box model in ``som-tomas-app/src`` expresses gas-phase
concentrations in three units depending on where you are in the code:

- **ppm** (parts per million by mole) — what the SAPRC integrator sees.
- **molec/cm³** — input/output units for OH, sulfate, etc.
- **kg/bag** — internal TOMAS units after converting from ppm at the
  chemistry/microphysics boundary.

This module ports the exact conversion formulas used in the Fortran so
regression comparisons are apples-to-apples. Every formula is cross-
referenced to a specific line in ``box.f``; run the unit tests to verify
round-trip correctness.

Sign conventions and assumptions
--------------------------------
- Temperature ``T_K`` in Kelvin.
- Pressure ``P_Pa`` in Pascal.
- ``boxvol_cm3`` is the box/chamber volume in cm³ (the Fortran input unit;
  ``runme.py`` passes e.g. ``7e6`` for a 7 L Teflon bag).
- ``mw_g_per_mol`` is species molecular weight in g/mol (matches
  ``MWT`` in the Fortran mechanism, and :attr:`Species.molecular_weight`
  in ``som-jax``'s parsed mechanism).

References
----------
- ``box.f:416``   ``boxmass = 0.0289 * pres * boxvol * 1e-6 / R / temp`` — total
  mass of air in the bag, in kg.
- ``box.f:592``   OH conversion ``molec/cm³ → ppm``.
- ``box.f:604``   ``ppm → kg/bag`` (``SAPRCGC → GC``).
- ``box.f:806``   ``kg/bag → ppm`` (``GC → SAPRCGC``).

All formulas use the Fortran gas constant ``R = 8.314 J mol⁻¹ K⁻¹`` so that
any future drift in a more precise CODATA value does not break Fortran
parity. If bit-exact reproduction is ever needed, the constants here are
the authoritative values.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

# --- physical constants --------------------------------------------------

R: float = 8.314
"""Gas constant, J mol⁻¹ K⁻¹. Matches ``R`` in ``box.f:161``."""

N_A: float = 6.022e23
"""Avogadro's number, molec mol⁻¹. Matches ``6.022E23`` in ``box.f:592``."""

MW_AIR_KG_PER_MOL: float = 0.0289
"""Molecular weight of air, kg mol⁻¹. Matches ``0.0289`` in ``box.f:416``."""

ScalarOrArray: TypeAlias = float | ArrayLike


# --- molec/cm³ ↔ ppm ----------------------------------------------------


def molec_cm3_to_ppm(
    molec_per_cm3: ScalarOrArray,
    T_K: ScalarOrArray,
    P_Pa: ScalarOrArray,
) -> NDArray[np.float64]:
    """Convert number density (molec cm⁻³) to mole fraction in ppm.

    Mirrors ``box.f:592`` exactly::

        CONST(6) = OH_conc * 1.0E6 / 6.022E23 / PRES * R * TEMP * 1.0E6

    Rearranged: ``ppm = N_molec * R * T * 1e12 / (N_A * P)``
    (the two ``1e6`` factors compose into ``1e12``; the ``1e6`` before
    ``/N_A`` is the cm³→m³ conversion, the trailing ``1e6`` is the
    mole-fraction→ppm conversion).

    Sanity check: at ``T=298 K, P=101325 Pa`` this gives ``1 ppm ≈
    2.463e13 molec/cm³`` (the standard atmospheric value).
    """
    molec = np.asarray(molec_per_cm3, dtype=np.float64)
    T = np.asarray(T_K, dtype=np.float64)
    P = np.asarray(P_Pa, dtype=np.float64)
    return molec * R * T * 1e12 / (N_A * P)


def ppm_to_molec_cm3(
    ppm: ScalarOrArray,
    T_K: ScalarOrArray,
    P_Pa: ScalarOrArray,
) -> NDArray[np.float64]:
    """Inverse of :func:`molec_cm3_to_ppm`.

    ``molec/cm³ = ppm * N_A * P / (R * T * 1e12)``
    """
    ppm_ = np.asarray(ppm, dtype=np.float64)
    T = np.asarray(T_K, dtype=np.float64)
    P = np.asarray(P_Pa, dtype=np.float64)
    return ppm_ * N_A * P / (R * T * 1e12)


# --- ppm ↔ kg/bag -------------------------------------------------------


def ppm_to_kg_per_bag(
    ppm: ScalarOrArray,
    T_K: ScalarOrArray,
    P_Pa: ScalarOrArray,
    boxvol_cm3: ScalarOrArray,
    mw_g_per_mol: ScalarOrArray,
) -> NDArray[np.float64]:
    """Convert ppm to species mass in the bag, in kg.

    Mirrors ``box.f:604–605`` exactly::

        SAPRCGC(I) * 1.0E-6 * PRES/R/TEMP * MWT(I+OFFSET)
            * 1.0E-3 * BOXVOL * 1.0e-6

    Step-by-step units check:
    1. ``ppm × 1e-6`` → mole fraction.
    2. ``× P/(R T)`` → ``mol/m³`` (total air density times mole fraction is
       species concentration).
    3. ``× MW × 1e-3`` → ``kg/m³`` (MW is g/mol, 1e-3 converts to kg/mol).
    4. ``× boxvol_cm3 × 1e-6`` → ``kg`` (cm³→m³ conversion).
    """
    ppm_ = np.asarray(ppm, dtype=np.float64)
    T = np.asarray(T_K, dtype=np.float64)
    P = np.asarray(P_Pa, dtype=np.float64)
    V = np.asarray(boxvol_cm3, dtype=np.float64)
    mw = np.asarray(mw_g_per_mol, dtype=np.float64)
    return ppm_ * 1e-6 * P / (R * T) * mw * 1e-3 * V * 1e-6


def kg_per_bag_to_ppm(
    mass_kg: ScalarOrArray,
    T_K: ScalarOrArray,
    P_Pa: ScalarOrArray,
    boxvol_cm3: ScalarOrArray,
    mw_g_per_mol: ScalarOrArray,
) -> NDArray[np.float64]:
    """Inverse of :func:`ppm_to_kg_per_bag`.

    Mirrors ``box.f:805–806`` exactly::

        GC(I) * 1.0E6 / PRES * R * TEMP / MWT(I+OFFSET)
            * 1.0E3 / BOXVOL * 1.0e6

    which simplifies to ``ppm = mass_kg * R * T * 1e12
    / (P * mw * V_cm3 * 1e-6)``.
    """
    mass = np.asarray(mass_kg, dtype=np.float64)
    T = np.asarray(T_K, dtype=np.float64)
    P = np.asarray(P_Pa, dtype=np.float64)
    V = np.asarray(boxvol_cm3, dtype=np.float64)
    mw = np.asarray(mw_g_per_mol, dtype=np.float64)
    return mass * 1e6 / P * R * T / mw * 1e3 / V * 1e6


# --- derived helper ------------------------------------------------------


def box_air_mass_kg(
    boxvol_cm3: ScalarOrArray,
    T_K: ScalarOrArray,
    P_Pa: ScalarOrArray,
) -> NDArray[np.float64]:
    """Total air mass inside the bag, in kg.

    Exact port of ``box.f:416``::

        boxmass = 0.0289 * pres * boxvol * 1e-6 / R / temp

    Useful as a sanity check when converting between mole fraction and
    mass: ``ppm_to_kg_per_bag(ppm=1e6, ..., mw=MW_AIR_KG_PER_MOL*1e3)``
    should equal this value (a pure-air "tracer" at 100% mole fraction
    has the total-air mass).
    """
    V = np.asarray(boxvol_cm3, dtype=np.float64)
    T = np.asarray(T_K, dtype=np.float64)
    P = np.asarray(P_Pa, dtype=np.float64)
    return MW_AIR_KG_PER_MOL * P * V * 1e-6 / R / T


__all__ = [
    "MW_AIR_KG_PER_MOL",
    "N_A",
    "R",
    "box_air_mass_kg",
    "kg_per_bag_to_ppm",
    "molec_cm3_to_ppm",
    "ppm_to_kg_per_bag",
    "ppm_to_molec_cm3",
]
