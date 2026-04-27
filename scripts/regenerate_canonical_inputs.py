"""Regenerate ``data/canonical_runs/inputs/<run_id>.txt`` from ``manifest.json``.

The input files are committed for transparency (a reviewer can ``cat`` them
without needing to execute Python), but the manifest is the source of
truth. CI runs this script and fails if the committed files drift from
what the manifest produces.

Usage::

    python scripts/regenerate_canonical_inputs.py            # writes files
    python scripts/regenerate_canonical_inputs.py --check    # exits non-zero on drift
"""

from __future__ import annotations

import argparse
import sys

from atmos_jax_common.canonical_runs import (
    default_inputs_dir,
    default_manifest_path,
    load_manifest,
    render_input,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify committed files match what the manifest produces; exit non-zero on drift.",
    )
    args = parser.parse_args(argv)

    matrix = load_manifest(default_manifest_path())
    out_dir = default_inputs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    drift: list[str] = []
    for run in matrix.runs:
        rendered = render_input(run, matrix.shared)
        target = out_dir / f"{run.run_id}.txt"
        if args.check:
            if not target.exists():
                drift.append(f"missing: {target.name}")
                continue
            existing = target.read_text(encoding="utf-8")
            if existing != rendered:
                drift.append(f"drift:   {target.name}")
        else:
            target.write_text(rendered, encoding="utf-8")
            print(f"wrote {target.relative_to(default_manifest_path().parents[2])}")

    if args.check:
        if drift:
            print("Canonical input drift detected:", file=sys.stderr)
            for d in drift:
                print(f"  {d}", file=sys.stderr)
            print(
                "\nRegenerate with: python scripts/regenerate_canonical_inputs.py",
                file=sys.stderr,
            )
            return 1
        print(f"All {len(matrix.runs)} canonical inputs match the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
