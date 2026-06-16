#!/usr/bin/env python3
"""Prepare common-basis MTZ pairs for TV extrapolation analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
import reciprocalspaceship as rs
import yaml

from mtz_metadata import read_mtz_metadata


DEFAULT_CONDITIONS = "esrf_5ms,esrf_75ms,trapping_1,trapping_2"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _condition_files(condition_dir: Path) -> tuple[Path, Path, Path]:
    mtzs = sorted(condition_dir.glob("*.mtz"))
    pdbs = sorted(condition_dir.glob("*.pdb"))
    if len(mtzs) != 2:
        raise ValueError(f"Expected exactly 2 MTZ files in {condition_dir}, found {len(mtzs)}")
    if len(pdbs) != 1:
        raise ValueError(f"Expected exactly 1 PDB file in {condition_dir}, found {len(pdbs)}")
    ground = [path for path in mtzs if "ground" in path.name.lower()]
    if not ground:
        raise ValueError(f"Could not identify ground/dark MTZ in {condition_dir}")
    if len(ground) > 1:
        raise ValueError(f"Multiple ground/dark MTZ candidates in {condition_dir}: {ground}")
    dark = ground[0]
    triggered = [path for path in mtzs if path != dark][0]
    return dark, triggered, pdbs[0]


def _cut_resolution(ds: rs.DataSet, d_min: float) -> rs.DataSet:
    with_dhkl = ds.compute_dHKL()
    return ds.loc[with_dhkl["dHKL"] >= d_min].copy()


def _finite_common_indices(
    dark: rs.DataSet,
    triggered: rs.DataSet,
    columns: tuple[str, ...],
) -> object:
    common = dark.index.intersection(triggered.index)
    dark_common = dark.loc[common]
    triggered_common = triggered.loc[common]
    finite = np.ones(len(common), dtype=bool)
    for column in columns:
        finite &= np.isfinite(np.asarray(dark_common[column], dtype=float))
        finite &= np.isfinite(np.asarray(triggered_common[column], dtype=float))
    return common[finite]


def _cell_delta_percent(
    dark_cell: tuple[float, float, float, float, float, float],
    triggered_cell: tuple[float, float, float, float, float, float],
) -> tuple[float, float]:
    deltas = [
        abs(triggered_cell[i] - dark_cell[i]) / dark_cell[i] * 100.0
        for i in range(3)
    ]
    rms = float(np.sqrt(np.mean(np.square(deltas))))
    return max(deltas), rms


def _write_config(
    condition: str,
    dark_common: Path,
    triggered_common: Path,
    pdb: Path,
    config_path: Path,
    result_root: Path,
    high_resolution_limit: float,
) -> None:
    result_dir = result_root / condition
    config = {
        "general": {
            "name_machine": condition,
            "name_human": condition.replace("_", " "),
            "output_folder": str(result_dir / "tmp"),
            "plot_folder": str(result_dir / "plots"),
            "high_resolution_limit": round(high_resolution_limit, 3),
            "comparison_type": "triggered",
        },
        "input_files": {
            "map_dark": str(dark_common),
            "map_triggered": str(triggered_common),
            "pdb_dark": str(pdb),
            "impose_dark_phases": True,
            "columns_are_ints": False,
            "columns_dark": {
                "amplitude_column": "F",
                "phase_column": "MODEL",
                "uncertainty_column": "SIGF",
            },
            "columns_triggered": {
                "amplitude_column": "F",
                "phase_column": "MODEL",
                "uncertainty_column": "SIGF",
            },
        },
        "map_processing": {
            "diffmap_type": "vanilla",
            "dark_mean_correction": True,
            "simple_dark_correction": True,
            "calculate_diffmap_before_f000": False,
        },
        "plot": {
            "show_plot": False,
            "save_to_file": True,
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, default=Path("initial"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/tv_isomorphous"))
    parser.add_argument("--configs-out", type=Path, default=Path("configs/xtr_tv_isomorphous_moderate"))
    parser.add_argument("--results", type=Path, default=Path("results/xtr_tv_isomorphous_moderate"))
    parser.add_argument(
        "--conditions",
        default=DEFAULT_CONDITIONS,
        help="Comma-separated condition names to prepare.",
    )
    args = parser.parse_args()

    requested = [_slug(value.strip()) for value in args.conditions.split(",") if value.strip()]
    rows = []
    for condition in requested:
        condition_dir = args.initial / condition
        if not condition_dir.exists():
            raise FileNotFoundError(f"Condition directory not found: {condition_dir}")
        dark_mtz, triggered_mtz, pdb = _condition_files(condition_dir)
        dark_meta = read_mtz_metadata(dark_mtz)
        triggered_meta = read_mtz_metadata(triggered_mtz)
        high_resolution_limit = max(
            value for value in (dark_meta.d_min, triggered_meta.d_min) if value
        )

        dark = _cut_resolution(rs.read_mtz(str(dark_mtz)), high_resolution_limit)
        triggered = _cut_resolution(rs.read_mtz(str(triggered_mtz)), high_resolution_limit)
        if not {"F", "SIGF"}.issubset(dark.columns) or not {"F", "SIGF"}.issubset(
            triggered.columns
        ):
            raise ValueError(f"{condition}: expected F/SIGF columns in both MTZs")

        common = _finite_common_indices(dark, triggered, ("F", "SIGF"))
        dark_common = dark.loc[common].copy()
        triggered_common = triggered.loc[common].copy()
        actual_high_resolution_limit = float(dark_common.compute_dHKL()["dHKL"].min())
        dark_common.cell = dark.cell
        triggered_common.cell = dark.cell
        dark_common.spacegroup = dark.spacegroup
        triggered_common.spacegroup = dark.spacegroup

        out_dir = args.out / condition
        out_dir.mkdir(parents=True, exist_ok=True)
        dark_out = out_dir / "dark_common.mtz"
        triggered_out = out_dir / "triggered_common.mtz"
        dark_common.write_mtz(str(dark_out))
        triggered_common.write_mtz(str(triggered_out))

        max_cell_delta, rms_cell_delta = _cell_delta_percent(dark_meta.cell, triggered_meta.cell)
        metadata = {
            "condition": condition,
            "dark_mtz": str(dark_mtz),
            "triggered_mtz": str(triggered_mtz),
            "pdb_dark": str(pdb),
            "dark_cell": dark_meta.cell,
            "triggered_cell": triggered_meta.cell,
            "dark_d_min": dark_meta.d_min,
            "triggered_d_min": triggered_meta.d_min,
            "common_high_resolution_limit": high_resolution_limit,
            "finite_common_high_resolution_limit": actual_high_resolution_limit,
            "common_hkl_count": len(common),
            "dark_reflection_count": dark_meta.n_reflections,
            "triggered_reflection_count": triggered_meta.n_reflections,
            "max_cell_delta_percent": max_cell_delta,
            "rms_cell_delta_percent": rms_cell_delta,
            "dark_common_mtz": str(dark_out),
            "triggered_common_mtz": str(triggered_out),
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        config_path = args.configs_out / f"{condition}.yaml"
        _write_config(
            condition=condition,
            dark_common=dark_out,
            triggered_common=triggered_out,
            pdb=pdb,
            config_path=config_path,
            result_root=args.results,
            high_resolution_limit=actual_high_resolution_limit,
        )
        row = {
            "condition": condition,
            "common_hkl_count": len(common),
            "dark_reflection_count": dark_meta.n_reflections,
            "triggered_reflection_count": triggered_meta.n_reflections,
            "common_high_resolution_limit": f"{actual_high_resolution_limit:.3f}",
            "max_cell_delta_percent": f"{max_cell_delta:.3f}",
            "rms_cell_delta_percent": f"{rms_cell_delta:.3f}",
            "dark_common_mtz": str(dark_out),
            "triggered_common_mtz": str(triggered_out),
            "config": str(config_path),
        }
        rows.append(row)
        print(f"{condition}: common_hkl={len(common)} cell_delta={max_cell_delta:.3f}%")

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {manifest}")
    print(f"Wrote configs in {args.configs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
