"""Shared infrastructure for the SOM-TOMAS Python/JAX port.

This package provides Fortran-reference plumbing used by the sibling model
repos (`som-jax`, `saprc-jax`, `tomas-jax`). No scientific logic lives here.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("atmos-jax-common")
except PackageNotFoundError:  # pragma: no cover - only during uninstalled dev
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
