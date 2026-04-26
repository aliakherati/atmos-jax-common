"""Unit tests for the SHA-based contract checker in
``scripts/generate_goldens.py`` (chunk C0.10).

The drift guard verifies that committed ``metadata.json`` SHAs still
match the live manifest + Fortran source — catching "Fortran source
changed without regenerating goldens" or "manifest changed without
refreshing inputs". Cross-machine numeric reproducibility of the
actual ``.dat`` outputs is intentionally NOT a regression signal here
(REAL*4 chemistry re-orders across gfortran versions producing
0.4-4% drift on cascade species — noise, not Fortran source change).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GG_PATH = _REPO_ROOT / "scripts" / "generate_goldens.py"


@pytest.fixture(scope="module")
def gg():
    """Load ``scripts/generate_goldens.py`` as a module so we can call
    its private helpers directly without invoking the CLI."""
    spec = importlib.util.spec_from_file_location("generate_goldens", _GG_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class _StubRun:
    """Minimal duck-type for CanonicalRun + RunOverrides used by
    ``render_input``. We bypass the real loader so the test isn't
    coupled to schema bumps in the manifest."""

    run_id: str
    summary: str
    params: object


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _make_run_with_metadata(
    gg,
    tmp_path: Path,
    *,
    src_files: dict[str, str],
    rendered_input: str,
    metadata_overrides: dict | None = None,
) -> tuple[Path, Path, _StubRun, object]:
    """Set up an isolated (src_dir, expected_dir, run, shared) so a
    test can call ``_check_run_contract`` cleanly."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    for name, contents in src_files.items():
        (src_dir / name).write_text(contents, encoding="utf-8")

    expected_dir = tmp_path / "expected"
    run_dir = expected_dir / "run42"
    run_dir.mkdir(parents=True)

    metadata = {
        "run_id": "run42",
        "summary": "test",
        "input_sha256": _sha256_text(rendered_input),
        "fortran_source_sha256": {
            name: hashlib.sha256(c.encode("utf-8")).hexdigest() for name, c in src_files.items()
        },
        "submodule_commit": "deadbeef",
        "generated_at_utc": "2026-04-26T00:00:00+00:00",
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Patch render_input on the module so it returns our test text
    # regardless of inputs. We swap it back if other tests need it,
    # but pytest fixture isolation keeps each test independent.
    original_render = gg.render_input

    def fake_render(_run, _shared):
        return rendered_input

    gg.render_input = fake_render

    run = _StubRun(run_id="run42", summary="test", params=object())
    shared = object()

    yield_value = (src_dir, expected_dir, run, shared)
    return yield_value, original_render


@pytest.fixture
def restore_render(gg):
    """Save and restore ``gg.render_input`` so tests that monkey-patch
    it don't leak state to siblings."""
    original = gg.render_input
    yield
    gg.render_input = original


# --- happy path ---------------------------------------------------------


def test_unchanged_source_and_manifest_have_no_drift(gg, tmp_path: Path, restore_render) -> None:
    rendered = "manifest input text v1\n"
    src_files = {"box.f": "C box.f stub\n", "integr2.f": "C integr2.f stub\n"}
    env, _ = _make_run_with_metadata(gg, tmp_path, src_files=src_files, rendered_input=rendered)
    src_dir, expected_dir, run, shared = env
    assert gg._check_run_contract(run, shared, src_dir, expected_dir) == []


# --- input-side drift ----------------------------------------------------


def test_input_sha_change_is_caught(gg, tmp_path: Path, restore_render) -> None:
    rendered = "manifest input text v1\n"
    src_files = {"box.f": "x\n"}
    env, _ = _make_run_with_metadata(gg, tmp_path, src_files=src_files, rendered_input=rendered)
    src_dir, expected_dir, run, shared = env
    # Now pretend the manifest renders something different.
    gg.render_input = lambda _r, _s: "manifest input text v2\n"
    diffs = gg._check_run_contract(run, shared, src_dir, expected_dir)
    assert any("input_sha256 mismatch" in d for d in diffs)


# --- source-side drift ---------------------------------------------------


def test_source_file_modification_is_caught(gg, tmp_path: Path, restore_render) -> None:
    rendered = "input\n"
    src_files = {"box.f": "C original\n", "rhs.f": "C rhs\n"}
    env, _ = _make_run_with_metadata(gg, tmp_path, src_files=src_files, rendered_input=rendered)
    src_dir, expected_dir, run, shared = env
    # Mutate one Fortran source file.
    (src_dir / "box.f").write_text("C MUTATED\n", encoding="utf-8")
    diffs = gg._check_run_contract(run, shared, src_dir, expected_dir)
    assert any("fortran source changed" in d and "box.f" in d for d in diffs)


def test_source_file_removed_is_caught(gg, tmp_path: Path, restore_render) -> None:
    rendered = "input\n"
    src_files = {"box.f": "C box\n", "rhs.f": "C rhs\n"}
    env, _ = _make_run_with_metadata(gg, tmp_path, src_files=src_files, rendered_input=rendered)
    src_dir, expected_dir, run, shared = env
    (src_dir / "rhs.f").unlink()
    diffs = gg._check_run_contract(run, shared, src_dir, expected_dir)
    assert any("fortran source removed" in d and "rhs.f" in d for d in diffs)


def test_source_file_added_is_caught(gg, tmp_path: Path, restore_render) -> None:
    rendered = "input\n"
    src_files = {"box.f": "C box\n"}
    env, _ = _make_run_with_metadata(gg, tmp_path, src_files=src_files, rendered_input=rendered)
    src_dir, expected_dir, run, shared = env
    (src_dir / "newfile.f").write_text("C new mechanism file\n", encoding="utf-8")
    diffs = gg._check_run_contract(run, shared, src_dir, expected_dir)
    assert any("fortran source added" in d and "newfile.f" in d for d in diffs)


# --- missing artefacts ---------------------------------------------------


def test_missing_metadata_is_caught(gg, tmp_path: Path, restore_render) -> None:
    rendered = "input\n"
    src_files = {"box.f": "x\n"}
    env, _ = _make_run_with_metadata(gg, tmp_path, src_files=src_files, rendered_input=rendered)
    src_dir, expected_dir, run, shared = env
    (expected_dir / run.run_id / "metadata.json").unlink()
    diffs = gg._check_run_contract(run, shared, src_dir, expected_dir)
    assert any("no metadata.json" in d for d in diffs)


# --- timestamp irrelevance ------------------------------------------------


def test_timestamp_change_alone_does_not_drift(gg, tmp_path: Path, restore_render) -> None:
    """``generated_at_utc`` is intentionally not part of the contract;
    rerunning ``--generate`` updates it but should not trigger drift."""
    rendered = "input\n"
    src_files = {"box.f": "x\n"}
    env, _ = _make_run_with_metadata(
        gg,
        tmp_path,
        src_files=src_files,
        rendered_input=rendered,
        metadata_overrides={"generated_at_utc": "2099-01-01T00:00:00+00:00"},
    )
    src_dir, expected_dir, run, shared = env
    assert gg._check_run_contract(run, shared, src_dir, expected_dir) == []
