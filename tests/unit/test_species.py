"""Unit tests for :mod:`atmos_jax_common.species`.

Covers: parsing, formatting, classification, round-trips, edge cases, and
the two real SOM species grids (47 AR1SOMG + 40 GENSOMG species from the
Fortran reference).
"""

from __future__ import annotations

import pytest

from atmos_jax_common.species import (
    SpeciesIdentifier,
    format_species_name,
    is_som_species,
    parse_species_name,
)

# --- parsing basics -----------------------------------------------------


def test_parse_gensomg_species() -> None:
    sp = parse_species_name("GENSOMG_04_03")
    assert sp.family == "GENSOMG"
    assert sp.carbon == 4
    assert sp.oxygen == 3
    assert sp.raw == "GENSOMG_04_03"


def test_parse_ar1somg_species() -> None:
    sp = parse_species_name("AR1SOMG_07_07")
    assert sp.family == "AR1SOMG"
    assert sp.carbon == 7
    assert sp.oxygen == 7


def test_parse_accepts_numeric_in_family_after_first_letter() -> None:
    sp = parse_species_name("AR2SOMG_05_03")
    assert sp.family == "AR2SOMG"


@pytest.mark.parametrize(
    "name",
    [
        "GENVOC",  # precursor, not SOM-structured
        "OH",  # radical
        "gensomg_04_03",  # lowercase rejected
        "GENSOMG_4_3",  # not two-digit
        "GENSOMG_04_03_extra",  # trailing tokens
        "GENSOMG__04_03",  # double underscore
        "GENSOMG_04_ab",  # non-numeric O
        "1GENSOMG_04_03",  # family starts with digit
        "",  # empty
        " GENSOMG_04_03",  # leading whitespace
    ],
)
def test_parse_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError, match="does not match SOM pattern"):
        parse_species_name(name)


# --- formatting --------------------------------------------------------


def test_format_matches_expected() -> None:
    assert format_species_name("GENSOMG", 4, 3) == "GENSOMG_04_03"
    assert format_species_name("AR1SOMG", 7, 7) == "AR1SOMG_07_07"


def test_format_zero_pads_single_digits() -> None:
    assert format_species_name("X", 0, 0) == "X_00_00"
    assert format_species_name("X", 1, 9) == "X_01_09"


def test_format_rejects_negative() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 99\\]"):
        format_species_name("GENSOMG", -1, 3)


def test_format_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 99\\]"):
        format_species_name("GENSOMG", 4, 100)


def test_format_rejects_lowercase_family() -> None:
    with pytest.raises(ValueError, match="uppercase"):
        format_species_name("gensomg", 4, 3)


def test_format_rejects_family_starting_with_digit() -> None:
    with pytest.raises(ValueError, match="uppercase"):
        format_species_name("1GENSOMG", 4, 3)


# --- round trips -------------------------------------------------------


@pytest.mark.parametrize(
    "family,c,o",
    [
        ("GENSOMG", 1, 1),
        ("GENSOMG", 4, 3),
        ("GENSOMG", 7, 7),
        ("AR1SOMG", 1, 0),
        ("AR1SOMG", 5, 4),
        ("AR1SOMG", 7, 7),
        ("AR2SOMG", 10, 10),  # hypothetical larger grid
    ],
)
def test_parse_format_round_trip(family: str, c: int, o: int) -> None:
    raw = format_species_name(family, c, o)
    sp = parse_species_name(raw)
    assert sp.family == family
    assert sp.carbon == c
    assert sp.oxygen == o
    assert sp.formatted() == raw


def test_identifier_formatted_preserves_content() -> None:
    sp = SpeciesIdentifier(family="GENSOMG", carbon=6, oxygen=4, raw="GENSOMG_06_04")
    assert sp.formatted() == "GENSOMG_06_04"


# --- is_som_species classifier -----------------------------------------


def test_classifier_true_for_som_names() -> None:
    assert is_som_species("GENSOMG_04_03")
    assert is_som_species("AR1SOMG_07_07")


@pytest.mark.parametrize(
    "name",
    [
        "GENVOC",
        "OH",
        "H2O",
        "",
        "genvoc_04_03",
        "GENSOMG_4_3",
    ],
)
def test_classifier_false_for_non_som_names(name: str) -> None:
    assert not is_som_species(name)


# --- full real grids ---------------------------------------------------


def _gensomg_species_names() -> list[str]:
    """Enumerate the 40 GENSOMG species that actually appear in
    saprc14_rev1.mod (the parser treats these as the canonical grid)."""
    pairs = [
        *((1, o) for o in (1, 2)),
        *((2, o) for o in (1, 2, 3, 4)),
        *((3, o) for o in (1, 2, 3, 4, 5, 6)),
        *((4, o) for o in (1, 2, 3, 4, 5, 6, 7)),
        *((5, o) for o in (1, 2, 3, 4, 5, 6, 7)),
        *((6, o) for o in (1, 2, 3, 4, 5, 6, 7)),
        *((7, o) for o in (1, 2, 3, 4, 5, 6, 7)),
    ]
    return [format_species_name("GENSOMG", c, o) for c, o in pairs]


def _ar1somg_species_names() -> list[str]:
    """Enumerate the 47 AR1SOMG species listed in saprc14_rev1.som."""
    pairs = [
        *((1, o) for o in (0, 1, 2)),
        *((2, o) for o in (0, 1, 2, 3, 4)),
        *((3, o) for o in (0, 1, 2, 3, 4, 5, 6)),
        *((4, o) for o in (0, 1, 2, 3, 4, 5, 6, 7)),
        *((5, o) for o in (0, 1, 2, 3, 4, 5, 6, 7)),
        *((6, o) for o in (0, 1, 2, 3, 4, 5, 6, 7)),
        *((7, o) for o in (0, 1, 2, 3, 4, 5, 6, 7)),
    ]
    return [format_species_name("AR1SOMG", c, o) for c, o in pairs]


def test_all_40_gensomg_species_parse() -> None:
    names = _gensomg_species_names()
    assert len(names) == 40
    for name in names:
        sp = parse_species_name(name)
        assert sp.family == "GENSOMG"
        assert 1 <= sp.carbon <= 7
        assert 1 <= sp.oxygen <= 7


def test_all_47_ar1somg_species_parse() -> None:
    names = _ar1somg_species_names()
    assert len(names) == 47
    for name in names:
        sp = parse_species_name(name)
        assert sp.family == "AR1SOMG"
        assert 1 <= sp.carbon <= 7
        assert 0 <= sp.oxygen <= 7


def test_gensomg_and_ar1somg_do_not_overlap() -> None:
    """Two families with the same (C, O) resolve to distinct species."""
    g = parse_species_name("GENSOMG_04_03")
    a = parse_species_name("AR1SOMG_04_03")
    assert g.family != a.family
    assert (g.carbon, g.oxygen) == (a.carbon, a.oxygen)


# --- hashability / frozen semantics ------------------------------------


def test_species_identifier_is_hashable() -> None:
    sp = parse_species_name("GENSOMG_04_03")
    s = {sp, sp}  # exercise __hash__
    assert len(s) == 1


def test_species_identifier_is_immutable() -> None:
    sp = parse_species_name("GENSOMG_04_03")
    with pytest.raises((AttributeError, TypeError)):
        sp.carbon = 99  # type: ignore[misc]
