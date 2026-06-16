#!/usr/bin/env python3
"""Prepare finite common-reflection inputs/configs for initial_ocp_can it-TV runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gemmi
import numpy as np
import reciprocalspaceship as rs
import yaml

from xtr_estimator.processing import gemmi_structure_to_calculated_map


@dataclass(frozen=True)
class CanDataset:
    condition: str
    triggered_mtz: Path
    high_resolution_limit: float
    dark_mtz: Optional[Path] = None
    triggered_is_intensity: bool = False


ROOT = Path("initial_ocp_can")
DARK_MTZ = ROOT / "can_initial/dark/dimple_processing/can-dark_dimple.mtz"
DARK_PDB = ROOT / "M4_CAN_dark_prefin.pdb"


DATASETS = (
    CanDataset(
        "can_laser14",
        ROOT / "can_initial/laser1/dimple_processing/can-laser14_dimple.mtz",
        1.90,
    ),
    CanDataset(
        "can_laser26",
        ROOT / "can_initial/laser2/dimple_processing/can-laser26_dimple.mtz",
        1.85,
    ),
)

FILTERED_DARK_MTZ = ROOT / "can_initial_filtered/dark/dimple_processing/can-dark_dimple.mtz"
FILTERED_DATASETS = (
    CanDataset(
        "can_laser14_filtered",
        ROOT / "can_initial_filtered/laser1/can-laser14.mtz",
        1.93,
        dark_mtz=FILTERED_DARK_MTZ,
        triggered_is_intensity=True,
    ),
    CanDataset(
        "can_laser26_filtered",
        ROOT / "can_initial_filtered/laser2/dimple_processing/can-laser26_dimple.mtz",
        1.80,
        dark_mtz=FILTERED_DARK_MTZ,
    ),
)


def _finite_filter(ds: rs.DataSet, columns: tuple[str, ...]) -> np.ndarray:
    mask = np.ones(len(ds), dtype=bool)
    for column in columns:
        mask &= np.isfinite(ds[column].to_numpy(dtype=float))
    return mask


def _clean_amplitudes(path: Path) -> tuple[rs.DataSet, dict[str, int | str]]:
    ds = rs.read_mtz(str(path)).hkl_to_asu()
    mask = _finite_filter(ds, ("F", "SIGF"))
    clean = ds.loc[mask].copy()
    return clean, {
        "source": str(path),
        "rows_source": len(ds),
        "rows_finite": len(clean),
        "rows_removed_nonfinite": int((~mask).sum()),
    }


def _intensities_to_amplitudes(path: Path) -> tuple[rs.DataSet, dict[str, int | str]]:
    ds = rs.read_mtz(str(path)).hkl_to_asu()
    intensities = ds["I"].to_numpy(dtype=float)
    sigmas = ds["SIGI"].to_numpy(dtype=float)
    mask = np.isfinite(intensities) & np.isfinite(sigmas) & (intensities > 0.0) & (sigmas > 0.0)
    clean = ds.loc[mask].copy()
    i_clean = clean["I"].to_numpy(dtype=float)
    sigi_clean = clean["SIGI"].to_numpy(dtype=float)
    abs_i = np.sqrt(np.abs(i_clean))
    clean["F"] = rs.DataSeries(abs_i, index=clean.index, dtype=rs.StructureFactorAmplitudeDtype())
    clean["SIGF"] = rs.DataSeries(
        sigi_clean / (2.0 * abs_i),
        index=clean.index,
        dtype=rs.StandardDeviationDtype(),
    )
    return clean, {
        "source": str(path),
        "rows_source": len(ds),
        "rows_finite": len(clean),
        "rows_removed_nonfinite": int((~mask).sum()),
    }


def _cryst1_line(cell: gemmi.UnitCell, spacegroup: gemmi.SpaceGroup) -> str:
    return (
        f"CRYST1{cell.a:9.3f}{cell.b:9.3f}{cell.c:9.3f}"
        f"{cell.alpha:7.2f}{cell.beta:7.2f}{cell.gamma:7.2f} "
        f"{spacegroup.hm:>11s}\n"
    )


def _write_pdb_with_cell(source: Path, target: Path, cell: gemmi.UnitCell, spacegroup: gemmi.SpaceGroup) -> None:
    lines = source.read_text().splitlines(keepends=True)
    cryst1 = _cryst1_line(cell, spacegroup)
    for i, line in enumerate(lines):
        if line.startswith("CRYST1"):
            lines[i] = cryst1
            break
    else:
        lines.insert(0, cryst1)
    target.write_text("".join(lines))


def _prepare_dataset(dataset: CanDataset, config_dir: Path, out_dir: Path) -> dict[str, str | int]:
    condition_dir = out_dir / dataset.condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    dark_source = dataset.dark_mtz if dataset.dark_mtz is not None else DARK_MTZ
    dark_ds, dark_stats = _clean_amplitudes(dark_source)
    if dataset.triggered_is_intensity:
        triggered_ds, triggered_stats = _intensities_to_amplitudes(dataset.triggered_mtz)
    else:
        triggered_ds, triggered_stats = _clean_amplitudes(dataset.triggered_mtz)

    pdb_out = condition_dir / f"{dataset.condition}_dark_cell.pdb"
    _write_pdb_with_cell(DARK_PDB, pdb_out, dark_ds.cell, dark_ds.spacegroup)
    model_map = gemmi_structure_to_calculated_map(
        gemmi.read_pdb(str(pdb_out)),
        high_resolution_limit=dataset.high_resolution_limit,
    )
    common = dark_ds.index.intersection(triggered_ds.index).intersection(model_map.index)
    dark_common = dark_ds.loc[common].copy()
    triggered_common = triggered_ds.loc[common].copy()
    triggered_common.cell = dark_common.cell
    triggered_common.spacegroup = dark_common.spacegroup

    dark_out = condition_dir / f"{dataset.condition}_dark_finite.mtz"
    triggered_out = condition_dir / f"{dataset.condition}_triggered_finite.mtz"
    dark_common[["F", "SIGF"]].write_mtz(str(dark_out))
    triggered_common[["F", "SIGF"]].write_mtz(str(triggered_out))

    config = {
        "general": {
            "name_machine": dataset.condition,
            "name_human": dataset.condition,
            "output_folder": str(out_dir),
            "plot_folder": str(condition_dir),
            "high_resolution_limit": dataset.high_resolution_limit,
            "comparison_type": "triggered",
        },
        "input_files": {
            "map_dark": str(dark_out),
            "map_triggered": str(triggered_out),
            "pdb_dark": str(pdb_out),
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
            "diffmap_type": "it_tv",
            "dark_mean_correction": True,
            "simple_dark_correction": True,
            "calculate_diffmap_before_f000": False,
        },
        "plot": {
            "show_plot": False,
            "save_to_file": True,
        },
    }
    config_path = config_dir / f"{dataset.condition}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    return {
        "condition": dataset.condition,
        "input_kind": "intensity_to_amplitude" if dataset.triggered_is_intensity else "amplitude",
        "dark_source": str(dark_source),
        "dark_rows_source": dark_stats["rows_source"],
        "dark_rows_finite": dark_stats["rows_finite"],
        "dark_rows_removed": dark_stats["rows_removed_nonfinite"],
        "triggered_source": str(dataset.triggered_mtz),
        "triggered_rows_source": triggered_stats["rows_source"],
        "triggered_rows_finite": triggered_stats["rows_finite"],
        "triggered_rows_removed": triggered_stats["rows_removed_nonfinite"],
        "common_cleaned_reflections": len(common),
        "dark_cleaned": str(dark_out),
        "triggered_cleaned": str(triggered_out),
        "pdb_dark_cell": str(pdb_out),
        "config": str(config_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs/it_tv_initial_ocp_can"))
    parser.add_argument("--out", type=Path, default=Path("results/it_tv_ocp_can"))
    parser.add_argument("--filtered", action="store_true")
    parser.add_argument("--conditions", nargs="+", default=[d.condition for d in DATASETS])
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/it_tv_ocp_can/input_cleaning_summary.csv"),
    )
    args = parser.parse_args()

    datasets = FILTERED_DATASETS if args.filtered else DATASETS
    if args.conditions == [d.condition for d in DATASETS] and args.filtered:
        args.conditions = [d.condition for d in FILTERED_DATASETS]
    dataset_by_condition = {dataset.condition: dataset for dataset in datasets}
    rows = [
        _prepare_dataset(dataset_by_condition[condition], args.config_dir, args.out)
        for condition in args.conditions
    ]

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
