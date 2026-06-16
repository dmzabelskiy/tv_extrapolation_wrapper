#!/usr/bin/env python3
"""Generate xtr-estimator YAML configs from the initial data layout."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys

import yaml

from mtz_metadata import choose_column_mode, read_mtz_metadata


DEFAULT_MAP_PROCESSING_MODE = "vanilla"


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


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _baseline_config(
    condition: str,
    dark_mtz: Path,
    triggered_mtz: Path,
    pdb: Path,
    output_root: Path,
    mode: str,
    dark_mean_correction: bool,
    simple_dark_correction: bool,
    calculate_diffmap_before_f000: bool,
    prefer_intensity_columns: bool,
) -> dict:
    dark_meta = read_mtz_metadata(dark_mtz)
    triggered_meta = read_mtz_metadata(triggered_mtz)
    dark_mode = choose_column_mode(dark_meta)
    triggered_mode = choose_column_mode(triggered_meta)
    if prefer_intensity_columns:
        if {"IMEAN", "SIGIMEAN"}.issubset(dark_meta.columns) and {
            "IMEAN",
            "SIGIMEAN",
        }.issubset(triggered_meta.columns):
            dark_mode = "mean_intensity"
            triggered_mode = "mean_intensity"
        elif {"I", "SIGI"}.issubset(dark_meta.columns) and {
            "I",
            "SIGI",
        }.issubset(triggered_meta.columns):
            dark_mode = "intensity"
            triggered_mode = "intensity"
    if dark_mode != triggered_mode:
        raise ValueError(
            f"Column mode mismatch for {condition}: dark={dark_mode}, triggered={triggered_mode}"
        )

    dmins = [value for value in (dark_meta.d_min, triggered_meta.d_min) if value]
    high_resolution_limit = round(max(dmins), 3) if dmins else 2.5

    result_dir = output_root / condition
    config = {
        "general": {
            "name_machine": condition,
            "name_human": condition.replace("_", " "),
            "output_folder": str(result_dir / "tmp"),
            "plot_folder": str(result_dir / "plots"),
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
            "diffmap_type": mode,
            "dark_mean_correction": dark_mean_correction,
            "simple_dark_correction": simple_dark_correction,
            "calculate_diffmap_before_f000": calculate_diffmap_before_f000,
        },
        "plot": {
            "show_plot": False,
            "save_to_file": True,
        },
    }

    if dark_mode in {"intensity", "mean_intensity"}:
        ints_column = "IMEAN" if dark_mode == "mean_intensity" else "I"
        sigi_column = "SIGIMEAN" if dark_mode == "mean_intensity" else "SIGI"
        config["input_files"].update(
            {
                "columns_are_ints": True,
                "columns_dark_ints": {
                    "ints_column": ints_column,
                    "int_uncertainty_column": sigi_column,
                },
                "columns_triggered_ints": {
                    "ints_column": ints_column,
                    "int_uncertainty_column": sigi_column,
                },
            }
        )
    else:
        column_config = {
            "amplitude_column": "F",
            "phase_column": "MODEL",
            "uncertainty_column": "SIGF",
        }
        config["input_files"].update(
            {
                "columns_are_ints": False,
                "columns_dark": dict(column_config),
                "columns_triggered": dict(column_config),
            }
        )

    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, default=Path("initial"))
    parser.add_argument("--out", type=Path, default=Path("configs/xtr"))
    parser.add_argument("--results", type=Path, default=Path("results/xtr"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/xtr/manifest.csv"))
    parser.add_argument(
        "--mode",
        choices=["vanilla", "kweighted", "tv", "it_tv"],
        default=DEFAULT_MAP_PROCESSING_MODE,
        help="map_processing.diffmap_type to write into generated configs",
    )
    parser.add_argument(
        "--no-dark-mean-correction",
        action="store_true",
        help="Disable the xtr-estimator dark/triggered F000 correction step.",
    )
    parser.add_argument(
        "--no-simple-dark-correction",
        action="store_true",
        help="Use xtr-estimator's newer absolute-density correction path.",
    )
    parser.add_argument(
        "--calculate-diffmap-before-f000",
        action="store_true",
        help="Calculate the difference map before F000 correction.",
    )
    parser.add_argument(
        "--prefer-intensity-columns",
        action="store_true",
        help="Use I/SIGI or IMEAN/SIGIMEAN columns even when F/SIGF columns exist.",
    )
    args = parser.parse_args()

    if not args.initial.exists():
        print(f"{args.initial} not found", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    generated = []
    manifest_rows = []
    for condition_dir in sorted(path for path in args.initial.iterdir() if path.is_dir()):
        condition = _slug(condition_dir.name)
        dark_mtz, triggered_mtz, pdb = _condition_files(condition_dir)
        dark_meta = read_mtz_metadata(dark_mtz)
        triggered_meta = read_mtz_metadata(triggered_mtz)
        column_mode = choose_column_mode(dark_meta)
        if args.prefer_intensity_columns and {"IMEAN", "SIGIMEAN"}.issubset(
            dark_meta.columns
        ):
            column_mode = "mean_intensity"
        config = _baseline_config(
            condition=condition,
            dark_mtz=dark_mtz,
            triggered_mtz=triggered_mtz,
            pdb=pdb,
            output_root=args.results,
            mode=args.mode,
            dark_mean_correction=not args.no_dark_mean_correction,
            simple_dark_correction=not args.no_simple_dark_correction,
            calculate_diffmap_before_f000=args.calculate_diffmap_before_f000,
            prefer_intensity_columns=args.prefer_intensity_columns,
        )
        output_path = args.out / f"{condition}.yaml"
        with output_path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        generated.append(output_path)
        manifest_rows.append(
            {
                "condition": condition,
                "column_mode": column_mode,
                "dark_mtz": str(dark_mtz),
                "triggered_mtz": str(triggered_mtz),
                "pdb_dark": str(pdb),
                "dark_d_min": f"{dark_meta.d_min:.3f}" if dark_meta.d_min else "",
                "triggered_d_min": f"{triggered_meta.d_min:.3f}" if triggered_meta.d_min else "",
                "high_resolution_limit": config["general"]["high_resolution_limit"],
                "config": str(output_path),
            }
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    for path in generated:
        print(path)
    print(f"Wrote manifest: {args.manifest}")
    print(f"Generated {len(generated)} configs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
