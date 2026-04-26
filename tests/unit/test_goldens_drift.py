"""Unit tests for the tolerance-aware drift checker in
``scripts/generate_goldens.py`` (chunk C0.10).

The drift check accepts cross-machine REAL*4 noise (~1e-7 to 1e-5
relative on cascade species) but flags real Fortran source changes
(~0.1% or more on at least one species). These tests pin that
contract so future tolerance changes are deliberate.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GG_PATH = _REPO_ROOT / "scripts" / "generate_goldens.py"


@pytest.fixture(scope="module")
def gg():
    """Load ``scripts/generate_goldens.py`` as a module for direct
    testing of internal helpers without invoking the CLI."""
    spec = importlib.util.spec_from_file_location("generate_goldens", _GG_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_dat(path: Path, arr: np.ndarray) -> None:
    np.savetxt(path, arr, fmt="%40.20e")


def test_identical_numeric_dat_files_have_no_drift(gg, tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    arr = np.linspace(0, 1, 12).reshape(3, 4)
    _write_dat(a / "run_saprcgc.dat", arr)
    _write_dat(b / "run_saprcgc.dat", arr)
    assert gg._diff_dirs(a, b) == []


def test_sub_tolerance_drift_passes(gg, tmp_path: Path) -> None:
    """REAL*4 ULP-level noise (1e-6 relative) should not trigger drift."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    arr = np.linspace(0.01, 1.0, 12).reshape(3, 4)
    _write_dat(a / "run_saprcgc.dat", arr)
    _write_dat(b / "run_saprcgc.dat", arr * (1 + 1e-6))
    assert gg._diff_dirs(a, b) == []


def test_above_tolerance_drift_is_caught(gg, tmp_path: Path) -> None:
    """A 0.1% shift — the magnitude we'd expect from a real Fortran
    source change — must be flagged as drift."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    arr = np.linspace(0.01, 1.0, 12).reshape(3, 4)
    _write_dat(a / "run_saprcgc.dat", arr)
    _write_dat(b / "run_saprcgc.dat", arr * 1.001)
    diffs = gg._diff_dirs(a, b)
    assert len(diffs) == 1
    assert "numeric drift" in diffs[0]
    assert "run_saprcgc.dat" in diffs[0]


def test_shape_mismatch_is_caught(gg, tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_dat(a / "run_saprcgc.dat", np.zeros((3, 4)))
    _write_dat(b / "run_saprcgc.dat", np.zeros((3, 5)))
    diffs = gg._diff_dirs(a, b)
    assert len(diffs) == 1
    assert "shape differs" in diffs[0]


def test_non_numeric_dat_falls_back_to_byte_diff(gg, tmp_path: Path) -> None:
    """``_spec.dat`` carries species-name strings, not floats. The
    diff helper should fall back to byte-equality there."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "run_spec.dat").write_text("ACTIVE GAS SP: O3 NO NO2\n", encoding="utf-8")
    (b / "run_spec.dat").write_text("ACTIVE GAS SP: O3 NO HONO\n", encoding="utf-8")  # changed
    diffs = gg._diff_dirs(a, b)
    assert len(diffs) == 1
    assert "bytes differ" in diffs[0]


def test_input_text_files_use_byte_equality(gg, tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "run.input").write_text("foo\n", encoding="utf-8")
    (b / "run.input").write_text("foo\n", encoding="utf-8")
    assert gg._diff_dirs(a, b) == []
    # Now mutate
    (b / "run.input").write_text("bar\n", encoding="utf-8")
    diffs = gg._diff_dirs(a, b)
    assert len(diffs) == 1
    assert "bytes differ: run.input" in diffs[0]


def test_metadata_timestamp_is_excluded(gg, tmp_path: Path) -> None:
    """``generated_at_utc`` changes every run by design and must not
    trigger drift on its own."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    base = {
        "input_sha256": "abc",
        "fortran_source_sha256": {"box.f": "def"},
    }
    (a / "metadata.json").write_text(json.dumps({**base, "generated_at_utc": "T1"}))
    (b / "metadata.json").write_text(json.dumps({**base, "generated_at_utc": "T2"}))
    assert gg._diff_dirs(a, b) == []


def test_metadata_sha_changes_are_caught(gg, tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "metadata.json").write_text(
        json.dumps(
            {
                "input_sha256": "abc",
                "fortran_source_sha256": {"box.f": "def"},
                "generated_at_utc": "T",
            }
        )
    )
    (b / "metadata.json").write_text(
        json.dumps(
            {
                "input_sha256": "abc",
                "fortran_source_sha256": {"box.f": "DIFFERENT"},
                "generated_at_utc": "T",
            }
        )
    )
    diffs = gg._diff_dirs(a, b)
    assert any("fortran_source_sha256" in d for d in diffs)


def test_added_or_removed_files_are_caught(gg, tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_dat(a / "run_saprcgc.dat", np.zeros((1, 1)))
    _write_dat(a / "run_gc.dat", np.zeros((1, 1)))
    _write_dat(b / "run_saprcgc.dat", np.zeros((1, 1)))
    # b is missing run_gc.dat
    diffs = gg._diff_dirs(a, b)
    assert any("files removed" in d for d in diffs)
