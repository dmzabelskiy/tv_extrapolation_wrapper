#!/usr/bin/env python3
"""Create Phenix-friendly MTZs from extrapolated map-coefficient MTZ files."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import reciprocalspaceship as rs


def _condition_seed(condition: str, base_seed: int) -> int:
    digest = hashlib.sha256(condition.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def _latest_extrapolated_mtz(condition_dir: Path, condition: str) -> Path:
    candidates = sorted(condition_dir.glob(f"{condition}_it_tv_extrapolated_xtr*.mtz"))
    if not candidates:
        raise FileNotFoundError(f"No extrapolated MTZ found for {condition_dir}")

    def factor(path: Path) -> float:
        match = re.search(r"_xtr([0-9]+(?:\.[0-9]+)?)\\.mtz$", path.name)
        return float(match.group(1)) if match else -1.0

    return max(candidates, key=factor)


def _make_free_flags(n: int, fraction: float, seed: int) -> np.ndarray:
    if not 0.0 < fraction < 1.0:
        raise ValueError("--free-fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    flags = np.zeros(n, dtype=np.int32)
    n_free = max(1, int(round(n * fraction)))
    flags[rng.choice(n, size=n_free, replace=False)] = 1
    return flags


def _convert_one(condition_dir: Path, condition: str, free_fraction: float, base_seed: int) -> Path:
    source = _latest_extrapolated_mtz(condition_dir, condition)
    ds = rs.read_mtz(str(source))
    required = {"F", "SIGF", "PHI"}
    missing = required.difference(ds.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(sorted(missing))}")

    out = rs.DataSet(index=ds.index.copy())
    out.cell = ds.cell
    out.spacegroup = ds.spacegroup

    out["FEXTRA"] = rs.DataSeries(
        ds["F"].to_numpy(dtype=float),
        index=out.index,
        dtype=rs.StructureFactorAmplitudeDtype(),
    )
    out["SIGFEXTRA"] = rs.DataSeries(
        ds["SIGF"].to_numpy(dtype=float),
        index=out.index,
        dtype=rs.StandardDeviationDtype(),
    )
    out["FreeR_flag"] = rs.DataSeries(
        _make_free_flags(len(out), free_fraction, _condition_seed(condition, base_seed)),
        index=out.index,
        dtype=rs.MTZIntDtype(),
    )

    # Conventional map-coefficient aliases for Phenix/Coot. These coefficients
    # are the extrapolated map coefficients, labeled as a 2mFo-DFc-style map.
    out["2FOFCWT"] = rs.DataSeries(
        ds["F"].to_numpy(dtype=float),
        index=out.index,
        dtype=rs.StructureFactorAmplitudeDtype(),
    )
    out["PH2FOFCWT"] = rs.DataSeries(
        ds["PHI"].to_numpy(dtype=float),
        index=out.index,
        dtype=rs.PhaseDtype(),
    )
    out["FWT"] = rs.DataSeries(
        ds["F"].to_numpy(dtype=float),
        index=out.index,
        dtype=rs.StructureFactorAmplitudeDtype(),
    )
    out["PHWT"] = rs.DataSeries(
        ds["PHI"].to_numpy(dtype=float),
        index=out.index,
        dtype=rs.PhaseDtype(),
    )

    target_dir = condition_dir / "phenix_ready"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{condition}_it_tv_extrapolated_phenix_ready.mtz"
    out.write_mtz(str(target))
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/it_tv_ocp"))
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--free-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260520)
    args = parser.parse_args()

    if args.conditions is None:
        conditions = sorted(path.name for path in args.root.iterdir() if path.is_dir())
    else:
        conditions = args.conditions

    for condition in conditions:
        target = _convert_one(args.root / condition, condition, args.free_fraction, args.seed)
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
