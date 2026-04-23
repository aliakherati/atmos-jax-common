"""Unit tests for :mod:`atmos_jax_common.units`.

Verifies physical constants, round-trip identities, known textbook
values at STP, and — critically — line-by-line parity with the Fortran
formulas in ``som-tomas-app/src/box.f`` lines 416, 592, 604–605, 805–806.
"""

from __future__ import annotations

import numpy as np
import pytest

from atmos_jax_common.units import (
    MW_AIR_KG_PER_MOL,
    N_A,
    R,
    box_air_mass_kg,
    kg_per_bag_to_ppm,
    molec_cm3_to_ppm,
    ppm_to_kg_per_bag,
    ppm_to_molec_cm3,
)

# Canonical test conditions: room temperature, standard atmosphere,
# the 7 L Teflon bag from runme.py (runme.py writes boxvol=7e6 cm³).
T_K = 298.0
P_PA = 101325.0
BOXVOL_CM3 = 7.0e6
MW_SPECIES_G_PER_MOL = 92.14  # GENVOC in saprc14_rev1.mod


# --- physical constants --------------------------------------------------


def test_gas_constant_matches_fortran() -> None:
    assert R == 8.314


def test_avogadro_matches_fortran() -> None:
    # box.f:592 uses 6.022E23 (not the CODATA 6.02214076e23). We honour the
    # Fortran value so round-trip tolerances stay in single-ULP territory.
    assert N_A == 6.022e23


def test_mw_air_matches_fortran() -> None:
    assert MW_AIR_KG_PER_MOL == 0.0289


# --- molec/cm³ ↔ ppm ----------------------------------------------------


def test_one_ppm_at_stp_is_approximately_2p46e13_molec_per_cm3() -> None:
    # Standard textbook value; at T=298 K, P=101325 Pa:
    # n_air / V = P / (R*T) = 4.089e-5 mol/cm³ × N_A = 2.463e13 molec/cm³.
    result = float(ppm_to_molec_cm3(1.0, T_K, P_PA))
    assert result == pytest.approx(2.463e13, rel=1e-3)


def test_molec_ppm_round_trip() -> None:
    ppm_in = 50.0  # typical ambient NO2
    molec = ppm_to_molec_cm3(ppm_in, T_K, P_PA)
    ppm_out = molec_cm3_to_ppm(molec, T_K, P_PA)
    assert float(ppm_out) == pytest.approx(ppm_in, rel=1e-12)


def test_ppm_scales_linearly_with_molec() -> None:
    base = molec_cm3_to_ppm(1.0e13, T_K, P_PA)
    doubled = molec_cm3_to_ppm(2.0e13, T_K, P_PA)
    assert float(doubled) == pytest.approx(2.0 * float(base), rel=1e-12)


def test_oh_conversion_matches_fortran_line_592() -> None:
    """box.f:592 reads::

        CONST(6) = OH_conc * 1.0E6 / 6.022E23 / PRES * R * TEMP * 1.0E6

    Re-derive it here explicitly and compare.
    """
    oh_molec_per_cm3 = 1.5e6
    fortran = oh_molec_per_cm3 * 1.0e6 / 6.022e23 / P_PA * R * T_K * 1.0e6
    ours = float(molec_cm3_to_ppm(oh_molec_per_cm3, T_K, P_PA))
    assert ours == pytest.approx(fortran, rel=1e-15, abs=0)


# --- ppm ↔ kg/bag -------------------------------------------------------


def test_ppm_kg_per_bag_round_trip() -> None:
    ppm_in = 0.05  # 50 ppb
    kg = ppm_to_kg_per_bag(ppm_in, T_K, P_PA, BOXVOL_CM3, MW_SPECIES_G_PER_MOL)
    ppm_out = kg_per_bag_to_ppm(kg, T_K, P_PA, BOXVOL_CM3, MW_SPECIES_G_PER_MOL)
    assert float(ppm_out) == pytest.approx(ppm_in, rel=1e-12)


def test_ppm_to_kg_matches_fortran_lines_604_and_605() -> None:
    """box.f:604–605 reads::

    SAPRCGC(I) * 1.0E-6 * PRES/R/TEMP * MWT(I+OFFSET)
        * 1.0E-3 * BOXVOL * 1.0e-6
    """
    ppm = 0.05
    fortran = ppm * 1.0e-6 * P_PA / R / T_K * MW_SPECIES_G_PER_MOL * 1.0e-3 * BOXVOL_CM3 * 1.0e-6
    ours = float(ppm_to_kg_per_bag(ppm, T_K, P_PA, BOXVOL_CM3, MW_SPECIES_G_PER_MOL))
    assert ours == pytest.approx(fortran, rel=1e-15, abs=0)


def test_kg_to_ppm_matches_fortran_lines_805_and_806() -> None:
    """box.f:805–806 reads::

    GC(I) * 1.0E6 / PRES * R * TEMP / MWT(I+OFFSET)
        * 1.0E3 / BOXVOL * 1.0e6
    """
    mass_kg = 1.5e-6  # ~1.5 mg of organic in the bag
    fortran = mass_kg * 1.0e6 / P_PA * R * T_K / MW_SPECIES_G_PER_MOL * 1.0e3 / BOXVOL_CM3 * 1.0e6
    ours = float(kg_per_bag_to_ppm(mass_kg, T_K, P_PA, BOXVOL_CM3, MW_SPECIES_G_PER_MOL))
    assert ours == pytest.approx(fortran, rel=1e-15, abs=0)


def test_ppm_to_kg_scales_linearly_with_ppm() -> None:
    a = ppm_to_kg_per_bag(1.0, T_K, P_PA, BOXVOL_CM3, MW_SPECIES_G_PER_MOL)
    b = ppm_to_kg_per_bag(3.0, T_K, P_PA, BOXVOL_CM3, MW_SPECIES_G_PER_MOL)
    assert float(b) == pytest.approx(3.0 * float(a), rel=1e-12)


def test_ppm_to_kg_scales_linearly_with_boxvol() -> None:
    a = ppm_to_kg_per_bag(1.0, T_K, P_PA, 1e6, MW_SPECIES_G_PER_MOL)
    b = ppm_to_kg_per_bag(1.0, T_K, P_PA, 2e6, MW_SPECIES_G_PER_MOL)
    assert float(b) == pytest.approx(2.0 * float(a), rel=1e-12)


# --- box_air_mass_kg ----------------------------------------------------


def test_box_air_mass_kg_matches_fortran_line_416() -> None:
    fortran = 0.0289 * P_PA * BOXVOL_CM3 * 1.0e-6 / R / T_K
    ours = float(box_air_mass_kg(BOXVOL_CM3, T_K, P_PA))
    assert ours == pytest.approx(fortran, rel=1e-15, abs=0)


def test_full_mole_fraction_of_air_equals_air_mass() -> None:
    """Pure-air "tracer" at 100% mole fraction (1e6 ppm) with MW=MW_AIR
    must have the total air mass in the bag — by definition of mole
    fraction. Key cross-check that the ppm-to-kg chain is dimensionally
    consistent.
    """
    # MW_AIR_KG_PER_MOL is in kg/mol; ppm_to_kg_per_bag takes g/mol.
    mw_air_g = MW_AIR_KG_PER_MOL * 1e3
    kg_all_air = ppm_to_kg_per_bag(1.0e6, T_K, P_PA, BOXVOL_CM3, mw_air_g)
    air_mass = box_air_mass_kg(BOXVOL_CM3, T_K, P_PA)
    assert float(kg_all_air) == pytest.approx(float(air_mass), rel=1e-12)


# --- vectorised (numpy array) input -------------------------------------


def test_vectorised_ppm_to_molec() -> None:
    ppm_arr = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    molec = ppm_to_molec_cm3(ppm_arr, T_K, P_PA)
    assert molec.shape == ppm_arr.shape
    # Linearity: each entry scales with its ppm value.
    one_ppm_value = float(ppm_to_molec_cm3(1.0, T_K, P_PA))
    np.testing.assert_allclose(molec, ppm_arr * one_ppm_value, rtol=1e-12)


def test_vectorised_round_trip_kg_to_ppm() -> None:
    ppm_arr = np.array([0.001, 0.01, 0.1, 1.0])
    kg = ppm_to_kg_per_bag(ppm_arr, T_K, P_PA, BOXVOL_CM3, MW_SPECIES_G_PER_MOL)
    recovered = kg_per_bag_to_ppm(kg, T_K, P_PA, BOXVOL_CM3, MW_SPECIES_G_PER_MOL)
    np.testing.assert_allclose(recovered, ppm_arr, rtol=1e-12)


# --- edge cases --------------------------------------------------------


def test_zero_ppm_is_zero_molec() -> None:
    assert float(ppm_to_molec_cm3(0.0, T_K, P_PA)) == 0.0


def test_zero_molec_is_zero_ppm() -> None:
    assert float(molec_cm3_to_ppm(0.0, T_K, P_PA)) == 0.0


def test_temperature_scaling_matches_ideal_gas() -> None:
    """At fixed P, doubling T halves the number density (and therefore
    doubles the ppm required to produce a given molec/cm³)."""
    molec_cold = float(ppm_to_molec_cm3(1.0, 150.0, P_PA))
    molec_hot = float(ppm_to_molec_cm3(1.0, 300.0, P_PA))
    assert molec_cold == pytest.approx(2.0 * molec_hot, rel=1e-12)
