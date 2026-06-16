#!/usr/bin/env python3
"""Generate iterative-TV xtr-estimator configs for the project datasets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from mtz_metadata import choose_column_mode, read_mtz_metadata


IT_TV_CONDITIONS = (
    "10ns",
    "5us",
    "10ms",
    "esrf_5ms",
    "esrf_5ms_2",
    "esrf_75ms",
    "low_ph",
    "trapping_1",
    "trapping_2",
)


def _pdb_for(condition_dir: Path) -> Path:
    pdbs = sorted(condition_dir.glob("*.pdb"))
    if len(pdbs) != 1:
        raise ValueError(f"Expected exactly one PDB in {condition_dir}, found {len(pdbs)}")
    return pdbs[0]


def _condition_mtz_pair(condition_dir: Path) -> tuple[Path, Path]:
    dark = condition_dir / "ground.mtz"
    if not dark.exists():
        dark_candidates = sorted(condition_dir.glob("*ground*.mtz"))
        if len(dark_candidates) != 1:
            raise ValueError(
                f"Expected ground.mtz or exactly one *ground*.mtz in {condition_dir}, "
                f"found {dark_candidates}"
            )
        dark = dark_candidates[0]
    triggered = condition_dir / f"{condition_dir.name}.mtz"
    if not triggered.exists():
        candidates = sorted(
            mtz
            for mtz in condition_dir.glob("*.mtz")
            if mtz != dark and not mtz.name.startswith("extrapolated_")
        )
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one triggered MTZ in {condition_dir}, found {candidates}"
            )
        triggered = candidates[0]
    return dark, triggered


def _config(condition: str, dark_mtz: Path, triggered_mtz: Path, pdb: Path, results: Path) -> dict:
    dark_meta = read_mtz_metadata(dark_mtz)
    triggered_meta = read_mtz_metadata(triggered_mtz)
    dark_mode = choose_column_mode(dark_meta)
    triggered_mode = choose_column_mode(triggered_meta)
    if dark_mode != triggered_mode:
        raise ValueError(
            f"Column mode mismatch for {condition}: dark={dark_mode}, triggered={triggered_mode}"
        )

    dmins = [value for value in (dark_meta.d_min, triggered_meta.d_min) if value]
    high_resolution_limit = round(max(dmins), 3) if dmins else 2.5
    result_dir = results / condition

    config = {
        "general": {
            "name_machine": condition,
            "name_human": condition,
            "output_folder": str(results),
            "plot_folder": str(result_dir),
            "high_resolution_limit": high_resolution_limit,
            "comparison_type": "triggered",
        },
        "input_files": {
            "map_dark": str(dark_mtz),
            "map_triggered": str(triggered_mtz),
            "pdb_dark": str(pdb),
            "impose_dark_phases": True,
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

    if dark_mode == "intensity":
        config["input_files"].update(
            {
                "columns_are_ints": True,
                "columns_dark_ints": {
                    "ints_column": "I",
                    "int_uncertainty_column": "SIGI",
                },
                "columns_triggered_ints": {
                    "ints_column": "I",
                    "int_uncertainty_column": "SIGI",
                },
            }
        )
    else:
        columns = {
            "amplitude_column": "F",
            "phase_column": "MODEL",
            "uncertainty_column": "SIGF",
        }
        config["input_files"].update(
            {
                "columns_are_ints": False,
                "columns_dark": dict(columns),
                "columns_triggered": dict(columns),
            }
        )
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, default=Path("initial"))
    parser.add_argument("--out", type=Path, default=Path("configs/xtr_it_tv_xfel"))
    parser.add_argument("--results", type=Path, default=Path("results/it_tv_extrapolated_maps"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/xtr_it_tv_xfel/manifest.csv"))
    parser.add_argument("--conditions", nargs="+", default=list(IT_TV_CONDITIONS))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for condition in args.conditions:
        condition_dir = args.initial / condition
        dark_mtz, triggered_mtz = _condition_mtz_pair(condition_dir)
        pdb = _pdb_for(condition_dir)
        config = _config(condition, dark_mtz, triggered_mtz, pdb, args.results)
        config_path = args.out / f"{condition}.yaml"
        with config_path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        rows.append(
            {
                "condition": condition,
                "dark_mtz": str(dark_mtz),
                "triggered_mtz": str(triggered_mtz),
                "pdb_dark": str(pdb),
                "high_resolution_limit": config["general"]["high_resolution_limit"],
                "config": str(config_path),
            }
        )
        print(config_path)

    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote manifest: {args.manifest}")
    print(f"Generated {len(rows)} configs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
