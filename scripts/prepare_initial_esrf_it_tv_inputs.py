#!/usr/bin/env python3
"""Prepare finite common-reflection inputs/configs for initial_esrf it-TV runs."""

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
class EsrfDataset:
    condition: str
    high_resolution_limit: float


DATASETS = (
    EsrfDataset("5ms_0-37p5ms", 2.2),
    EsrfDataset("5ms_0-75ms", 2.2),
    EsrfDataset("5ms_37p5-75ms", 2.2),
    EsrfDataset("75ms_0-37p5ms", 2.3),
    EsrfDataset("75ms_0-75ms", 2.45),
    EsrfDataset("75ms_37p5-75ms", 2.58),
)


def _finite_filter(ds: rs.DataSet, columns: tuple[str, ...]) -> np.ndarray:
    mask = np.ones(len(ds), dtype=bool)
    for column in columns:
        mask &= np.isfinite(ds[column].to_numpy(dtype=float))
    return mask


def _clean(path: Path) -> tuple[rs.DataSet, dict[str, int | str]]:
    ds = rs.read_mtz(str(path)).hkl_to_asu()
    mask = _finite_filter(ds, ("F", "SIGF"))
    clean = ds.loc[mask].copy()
    return clean, {
        "source": str(path),
        "rows_source": len(ds),
        "rows_finite": len(clean),
        "rows_removed_nonfinite": int((~mask).sum()),
    }


def _prepare_dataset(dataset: EsrfDataset, config_dir: Path, out_dir: Path) -> dict[str, str | int | float]:
    source_dir = Path("initial_esrf") / dataset.condition
    condition_dir = out_dir / dataset.condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    dark_source = source_dir / "ground.mtz"
    triggered_source = source_dir / f"{dataset.condition}.mtz"
    pdb_source = source_dir / "olpvr1_id30a3_dark_refine_031.pdb"

    dark_ds, dark_stats = _clean(dark_source)
    triggered_ds, triggered_stats = _clean(triggered_source)
    model_map = gemmi_structure_to_calculated_map(
        gemmi.read_pdb(str(pdb_source)),
        high_resolution_limit=dataset.high_resolution_limit,
    )
    common = dark_ds.index.intersection(triggered_ds.index).intersection(model_map.index)
    dark_common = dark_ds.loc[common].copy()
    triggered_common = triggered_ds.loc[common].copy()
    triggered_common.cell = dark_common.cell
    triggered_common.spacegroup = dark_common.spacegroup

    dark_out = condition_dir / f"{dataset.condition}_dark_finite.mtz"
    triggered_out = condition_dir / f"{dataset.condition}_triggered_finite.mtz"
    dark_common.write_mtz(str(dark_out))
    triggered_common.write_mtz(str(triggered_out))

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
            "pdb_dark": str(pdb_source),
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
        "high_resolution_limit": dataset.high_resolution_limit,
        "dark_source": str(dark_source),
        "dark_rows_source": dark_stats["rows_source"],
        "dark_rows_finite": dark_stats["rows_finite"],
        "dark_rows_removed_nonfinite": dark_stats["rows_removed_nonfinite"],
        "triggered_source": str(triggered_source),
        "triggered_rows_source": triggered_stats["rows_source"],
        "triggered_rows_finite": triggered_stats["rows_finite"],
        "triggered_rows_removed_nonfinite": triggered_stats["rows_removed_nonfinite"],
        "common_cleaned_reflections": len(common),
        "dark_cleaned": str(dark_out),
        "triggered_cleaned": str(triggered_out),
        "config": str(config_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs/it_tv_initial_esrf"))
    parser.add_argument("--out", type=Path, default=Path("results/it_tv_esrf"))
    parser.add_argument("--conditions", nargs="+", default=[dataset.condition for dataset in DATASETS])
    parser.add_argument("--summary", type=Path, default=Path("results/it_tv_esrf/input_cleaning_summary.csv"))
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
            f"{row['condition']}: finite dark {row['dark_rows_finite']}/{row['dark_rows_source']}, "
            f"triggered {row['triggered_rows_finite']}/{row['triggered_rows_source']}, "
            f"common={row['common_cleaned_reflections']}"
        )
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
