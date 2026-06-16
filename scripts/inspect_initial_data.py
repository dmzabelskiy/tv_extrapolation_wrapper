#!/usr/bin/env python3
"""Print a compact metadata table for the initial MTZ/PDB bundle."""

from __future__ import annotations

from pathlib import Path
import sys

from mtz_metadata import read_mtz_metadata


def main() -> int:
    root = Path("initial")
    if not root.exists():
        print("initial/ not found", file=sys.stderr)
        return 1

    print("path,d_min,n_reflections,columns")
    for mtz in sorted(root.glob("*/*.mtz")):
        meta = read_mtz_metadata(mtz)
        d_min = f"{meta.d_min:.3f}" if meta.d_min else ""
        print(
            f"{mtz},{d_min},{meta.n_reflections or ''},"
            f"{' '.join(meta.columns)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
