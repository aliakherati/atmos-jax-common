"""Fixtures for Fortran-integration tests.

Integration tests in this directory actually build and run the Fortran
SOM-TOMAS reference, which needs:

1. A Fortran compiler (``gfortran`` on macOS / Linux).
2. The Fortran source tree, whose location is supplied via the
   ``SOM_TOMAS_APP_SRC`` environment variable.

If either is missing, the tests skip — so CI without gfortran / without
the Fortran repo checked out just reports skips, not failures.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def som_tomas_src() -> Path:
    """Return the Fortran source directory or skip.

    Controlled by ``SOM_TOMAS_APP_SRC``. The path must point at the
    directory containing ``simple_makfile`` and ``box.f`` (usually
    ``som-tomas-app/src``).
    """
    env = os.environ.get("SOM_TOMAS_APP_SRC")
    if not env:
        pytest.skip("SOM_TOMAS_APP_SRC not set — Fortran integration skipped")
    path = Path(env).resolve()
    if not path.is_dir():
        pytest.skip(f"SOM_TOMAS_APP_SRC={path} is not a directory")
    if not (path / "simple_makfile").is_file():
        pytest.skip(f"{path}/simple_makfile missing")
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not on PATH")
    return path
