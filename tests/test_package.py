"""Smoke tests — package is importable and exposes a version."""

import re

import atmos_jax_common


def test_version_is_exposed() -> None:
    assert isinstance(atmos_jax_common.__version__, str)
    assert atmos_jax_common.__version__


def test_version_matches_semver() -> None:
    # Allow "0.0.0+unknown" fallback during local dev when package is not installed.
    semver = re.compile(r"^\d+\.\d+\.\d+(?:[-+].+)?$")
    assert semver.match(atmos_jax_common.__version__)
