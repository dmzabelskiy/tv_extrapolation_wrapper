#!/usr/bin/env python3
"""Prepare finite common-reflection firstprocessing inputs for it-TV runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gemmi
import numpy as np
import reciprocalspaceship as rs
import yaml

from xtr_estimator.processing import gemmi_structure_to_calculated_map


def _intensities_to_amplitudes(ds: rs.DataSet, i_col: str, sigi_col: str) -> tuple[rs.DataSet, dict[str, int]]:
    ds = ds.hkl_to_asu()
    intensities = ds[i_col].to_numpy(dtype=float)
    sigmas = ds[sigi_col].to_numpy(dtype=float)
    mask = np.isfinite(intensities) & np.isfinite(sigmas) & (intensities != 0.0) & (sigmas > 0.0)

    clean = ds.loc[mask].copy()
    i_clean = clean[i_col].to_numpy(dtype=float)
    sigi_clean = clean[sigi_col].to_numpy(dtype=float)
    abs_i = np.sqrt(np.abs(i_clean))

    clean["F"] = rs.DataSeries(
        np.sign(i_clean) * abs_i,
        index=clean.index,
        dtype=rs.StructureFactorAmplitudeDtype(),
    )
    clean["SIGF"] = rs.DataSeries(
        sigi_clean / (2.0 * abs_i),
        index=clean.index,
        dtype=rs.StandardDeviationDtype(),
    )

    return clean, {
        "rows_source": len(ds),
        "rows_finite_amplitude": len(clean),
        "rows_removed": int((~mask).sum()),
    }


def _prepare_one(condition: str, config_dir: Path, out_dir: Path) -> dict[str, str | int]:
    config_path = config_dir / f"{condition}.yaml"
    config = yaml.safe_load(config_path.read_text())
    condition_dir = out_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)

    dark_cols = config["input_files"]["columns_dark_ints"]
    triggered_cols = config["input_files"]["columns_triggered_ints"]
    dark_ds, dark_stats = _intensities_to_amplitudes(
        rs.read_mtz(config["input_files"]["map_dark"]),
        dark_cols["ints_column"],
        dark_cols["int_uncertainty_column"],
    )
    triggered_ds, triggered_stats = _intensities_to_amplitudes(
        rs.read_mtz(config["input_files"]["map_triggered"]),
        triggered_cols["ints_column"],
        triggered_cols["int_uncertainty_column"],
    )

    model_map = gemmi_structure_to_calculated_map(
        gemmi.read_pdb(config["input_files"]["pdb_dark"]),
        high_resolution_limit=config["general"]["high_resolution_limit"],
    )
    common = dark_ds.index.intersection(triggered_ds.index).intersection(model_map.index)
    dark_common = dark_ds.loc[common].copy()
    triggered_common = triggered_ds.loc[common].copy()
    triggered_common.cell = dark_common.cell
    triggered_common.spacegroup = dark_common.spacegroup

    dark_out = condition_dir / f"{condition}_dark_finite_amplitudes.mtz"
    triggered_out = condition_dir / f"{condition}_triggered_finite_amplitudes.mtz"
    dark_common[["F", "SIGF"]].write_mtz(str(dark_out))
    triggered_common[["F", "SIGF"]].write_mtz(str(triggered_out))

    config["input_files"]["map_dark"] = str(dark_out)
    config["input_files"]["map_triggered"] = str(triggered_out)
    config["input_files"]["columns_are_ints"] = False
    config["input_files"].pop("columns_dark_ints", None)
    config["input_files"].pop("columns_triggered_ints", None)
    columns = {
        "amplitude_column": "F",
        "phase_column": "MODEL",
        "uncertainty_column": "SIGF",
    }
    config["input_files"]["columns_dark"] = dict(columns)
    config["input_files"]["columns_triggered"] = dict(columns)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    return {
        "condition": condition,
        "dark_rows_source": dark_stats["rows_source"],
        "dark_rows_finite_amplitude": dark_stats["rows_finite_amplitude"],
        "dark_rows_removed": dark_stats["rows_removed"],
        "triggered_rows_source": triggered_stats["rows_source"],
        "triggered_rows_finite_amplitude": triggered_stats["rows_finite_amplitude"],
        "triggered_rows_removed": triggered_stats["rows_removed"],
        "common_cleaned_reflections": len(common),
        "dark_cleaned": str(dark_out),
        "triggered_cleaned": str(triggered_out),
        "config": str(config_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs/it_tv_firstprocessing"))
    parser.add_argument("--out", type=Path, default=Path("results/it_tv_firstprocessing"))
    parser.add_argument("--conditions", nargs="+", default=["30min", "1h"])
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/it_tv_firstprocessing/input_cleaning_summary.csv"),
    )
    args = parser.parse_args()

    rows = [_prepare_one(condition, args.config_dir, args.out) for condition in args.conditions]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['condition']}: common={row['common_cleaned_reflections']}, "
            f"dark_removed={row['dark_rows_removed']}, "
            f"triggered_removed={row['triggered_rows_removed']}"
        )
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
