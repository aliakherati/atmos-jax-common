"""Fixtures for Fortran-integration tests.

Integration tests in this directory actually build and run the Fortran
SOM-TOMAS reference, which needs:

1. A Fortran compiler (``gfortran`` on macOS / Linux).
2. The Fortran source tree. By default the fixture uses the vendored
   submodule at ``third_party/som-tomas-fortran/src/`` (populated via
   ``git submodule update --init``). Override with the
   ``SOM_TOMAS_FORTRAN_SRC`` environment variable to point at a
   different checkout — useful when iterating locally on the upstream
   repo without waiting for the submodule pin to move.

If neither the submodule nor the env var resolves, and ``gfortran`` is
not on PATH, the tests skip — so CI without the Fortran toolchain just
reports skips, not failures.

Legacy env-var name ``SOM_TOMAS_APP_SRC`` is still honoured (warns once
when used) so local shells / CI jobs that pre-date the rename keep
working.
"""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUBMODULE_SRC = _REPO_ROOT / "third_party" / "som-tomas-fortran" / "src"


def _resolve_src_override() -> Path | None:
    """Return an env-var override for the Fortran source dir, or None."""
    canonical = os.environ.get("SOM_TOMAS_FORTRAN_SRC")
    if canonical:
        return Path(canonical).resolve()
    legacy = os.environ.get("SOM_TOMAS_APP_SRC")
    if legacy:
        warnings.warn(
            "SOM_TOMAS_APP_SRC is deprecated; use SOM_TOMAS_FORTRAN_SRC instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Path(legacy).resolve()
    return None


@pytest.fixture(scope="session")
def som_tomas_src() -> Path:
    """Return the Fortran source directory or skip.

    Resolution order:
    1. ``SOM_TOMAS_FORTRAN_SRC`` environment variable (or the legacy
       ``SOM_TOMAS_APP_SRC`` with a deprecation warning).
    2. The ``third_party/som-tomas-fortran/src`` submodule checkout.

    The resulting directory must contain ``simple_makfile`` and
    ``box.f``. ``gfortran`` must be on PATH.
    """
    path = _resolve_src_override()
    if path is None:
        if _SUBMODULE_SRC.is_dir() and (_SUBMODULE_SRC / "simple_makfile").is_file():
            path = _SUBMODULE_SRC
        else:
            pytest.skip(
                "Fortran source not found. Either set SOM_TOMAS_FORTRAN_SRC or "
                "initialise the submodule: git submodule update --init"
            )

    if not path.is_dir():
        pytest.skip(f"Fortran source path {path} is not a directory")
    if not (path / "simple_makfile").is_file():
        pytest.skip(f"{path}/simple_makfile missing")
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not on PATH")
    return path
