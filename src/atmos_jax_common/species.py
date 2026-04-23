"""Parse SOM species names.

SOM (Statistical Oxidation Model) species in the Fortran reference and in
the sibling ``som-jax`` package are encoded as strings of the form
``<FAMILY>_<CC>_<OO>`` where ``CC`` and ``OO`` are two-digit zero-padded
integer counts of carbon and oxygen atoms. Examples::

    GENSOMG_04_03    -> family GENSOMG, 4 carbons, 3 oxygens
    AR1SOMG_07_07    -> family AR1SOMG, 7 carbons, 7 oxygens

Non-SOM species (precursors like ``GENVOC``, structural species like ``OH``)
do not match this pattern; callers must handle them out-of-band.

This module provides a typed :class:`SpeciesIdentifier`, a parser, a
formatter, and a classifier. It's deliberately pure plumbing — no JAX,
no numpy — so it can be called from data-loading paths without
triggering array-library imports.

References
----------
- ``som-tomas-app/src/saprc14_rev1.mod`` lines 111-150 — GENSOMG cards.
- ``som-tomas-app/src/saprc14_rev1.som`` — AR1SOMG cards (same shape).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Family is letters/digits starting with a letter (e.g. GENSOMG, AR1SOMG,
# AR2SOMG, ISOPSOMG, …); CC and OO are two-digit zero-padded counts.
_SOM_NAME_RE = re.compile(r"^(?P<family>[A-Z][A-Z0-9]*)_(?P<c>\d{2})_(?P<o>\d{2})$")


@dataclass(frozen=True, slots=True)
class SpeciesIdentifier:
    """The parsed decomposition of a SOM species name.

    Attributes
    ----------
    family
        Family identifier (``"GENSOMG"``, ``"AR1SOMG"``, …).
    carbon
        Number of carbon atoms.
    oxygen
        Number of oxygen atoms.
    raw
        The original input string, preserved so callers can round-trip
        through parse/format without losing source case.
    """

    family: str
    carbon: int
    oxygen: int
    raw: str

    def formatted(self) -> str:
        """Re-serialise to the canonical ``FAMILY_CC_OO`` form."""
        return format_species_name(self.family, self.carbon, self.oxygen)


def is_som_species(name: str) -> bool:
    """Return ``True`` iff ``name`` matches the ``FAMILY_CC_OO`` SOM pattern.

    This is a cheap classifier — useful for filtering a mechanism's full
    species list down to just the SOM subset. Does **not** try to
    validate scientific sanity (no limits on C, O, or family name); use
    :func:`parse_species_name` and catch ``ValueError`` for that.
    """
    return _SOM_NAME_RE.match(name) is not None


def parse_species_name(name: str) -> SpeciesIdentifier:
    """Parse ``"GENSOMG_04_03"`` into a :class:`SpeciesIdentifier`.

    Raises
    ------
    ValueError
        If ``name`` does not match the SOM pattern. The error message
        includes the offending input for debugging.
    """
    match = _SOM_NAME_RE.match(name)
    if match is None:
        raise ValueError(
            f"species name {name!r} does not match SOM pattern FAMILY_CC_OO "
            "(uppercase family starting with a letter, two-digit C and O)"
        )
    return SpeciesIdentifier(
        family=match.group("family"),
        carbon=int(match.group("c")),
        oxygen=int(match.group("o")),
        raw=name,
    )


def format_species_name(family: str, carbon: int, oxygen: int) -> str:
    """Build ``"GENSOMG_04_03"`` from ``("GENSOMG", 4, 3)``.

    Enforces the formatting rules used by the Fortran reference:
    - ``family`` must be ASCII uppercase/digit starting with a letter.
    - ``carbon`` and ``oxygen`` must be non-negative and fit in two digits.

    Raises
    ------
    ValueError
        For family names that violate the pattern, or for C/O out of
        ``[0, 99]``.
    """
    if not re.fullmatch(r"[A-Z][A-Z0-9]*", family):
        raise ValueError(f"family {family!r} must be uppercase ASCII starting with a letter")
    if not (0 <= carbon <= 99):
        raise ValueError(f"carbon {carbon} must be in [0, 99]")
    if not (0 <= oxygen <= 99):
        raise ValueError(f"oxygen {oxygen} must be in [0, 99]")
    return f"{family}_{carbon:02d}_{oxygen:02d}"


__all__ = [
    "SpeciesIdentifier",
    "format_species_name",
    "is_som_species",
    "parse_species_name",
]
