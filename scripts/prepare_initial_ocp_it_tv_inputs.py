#!/usr/bin/env python3
"""Prepare finite common-reflection inputs/configs for current initial_ocp it-TV runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np
import reciprocalspaceship as rs
import yaml

from xtr_estimator.processing import gemmi_structure_to_calculated_map


@dataclass(frozen=True)
class OcpDataset:
    condition: str
    triggered_mtz: Path
    high_resolution_limit: float


ROOT = Path("initial_ocp")
DARK_MTZ = ROOT / "ech_dark/dimple_processing/ech-dark_dimple.mtz"
DARK_PDB = ROOT / "OCP_ECH_dark.pdb"


DATASETS = (
    OcpDataset(
        "ech_laser_30min",
        ROOT / "ech_laser_30min/dimple_processing/ech-laser_30min_dimple.mtz",
        1.8,
    ),
    OcpDataset(
        "ech_laser_1h_d1",
        ROOT / "ech_laser_1h_1/dimple_processing/ech-laser_1h_d1_dimple.mtz",
        1.83,
    ),
    OcpDataset(
        "ech_laser_1h_d2",
        ROOT / "ech_laser_1h_2/dimple_processing/ech-laser_1h_d2_dimple.mtz",
        1.83,
    ),
    OcpDataset(
        "ech_laser_2h",
        ROOT / "ech_laser_2h/dimple_processing/ech-laser_2h_dimple.mtz",
        1.8,
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


def _prepare_dataset(dataset: OcpDataset, config_dir: Path, out_dir: Path) -> dict[str, str | int]:
    condition_dir = out_dir / dataset.condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    dark_ds, dark_stats = _clean_amplitudes(DARK_MTZ)
    triggered_ds, triggered_stats = _clean_amplitudes(dataset.triggered_mtz)

    pdb_out = condition_dir / f"{dataset.condition}_dark_cell.pdb"
    _write_pdb_with_cell(
        DARK_PDB,
        pdb_out,
        dark_ds.cell,
        dark_ds.spacegroup,
    )
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
        "input_kind": "amplitude",
        "dark_source": str(DARK_MTZ),
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
    parser.add_argument("--config-dir", type=Path, default=Path("configs/it_tv_initial_ocp"))
    parser.add_argument("--out", type=Path, default=Path("results/it_tv_ocp"))
    parser.add_argument("--conditions", nargs="+", default=[d.condition for d in DATASETS])
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/it_tv_ocp/input_cleaning_summary.csv"),
    )
    args = parser.parse_args()

    dataset_by_condition = {dataset.condition: dataset for dataset in DATASETS}
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
