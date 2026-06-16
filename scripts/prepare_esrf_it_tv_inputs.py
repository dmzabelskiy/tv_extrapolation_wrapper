#!/usr/bin/env python3
"""Prepare finite-only ESRF MTZ inputs for iterative-TV runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import gemmi
import reciprocalspaceship as rs
import yaml

from xtr_estimator.processing import gemmi_structure_to_calculated_map


DEFAULT_CONDITIONS = ("esrf_5ms", "esrf_5ms_2", "esrf_75ms")


def _finite_filter(ds: rs.DataSet, columns: tuple[str, ...]) -> np.ndarray:
    mask = np.ones(len(ds), dtype=bool)
    for column in columns:
        mask &= np.isfinite(ds[column].to_numpy())
    return mask


def _clean_dataset(path: Path, columns: tuple[str, ...]) -> tuple[rs.DataSet, dict[str, int | str]]:
    ds = rs.read_mtz(str(path))
    mask = _finite_filter(ds, columns)
    clean = ds.loc[mask].copy()
    return clean, {
        "source": str(path),
        "rows_source": len(ds),
        "rows_finite": len(clean),
        "rows_removed_nonfinite": int((~mask).sum()),
    }


def _manifest_sources(config_dir: Path) -> dict[str, dict[str, str]]:
    manifest = config_dir / "manifest.csv"
    if not manifest.exists():
        return {}
    with manifest.open() as handle:
        return {row["condition"]: row for row in csv.DictReader(handle)}


def _source_path(
    config: dict,
    manifest_row: dict[str, str] | None,
    key: str,
    manifest_key: str,
    prefer_manifest: bool,
) -> Path:
    if prefer_manifest and manifest_row is not None:
        manifest_path = Path(manifest_row[manifest_key])
        if manifest_path.exists():
            return manifest_path
    configured = Path(config["input_files"][key])
    if configured.exists():
        return configured
    if manifest_row is not None:
        manifest_path = Path(manifest_row[manifest_key])
        if manifest_path.exists():
            return manifest_path
    return configured


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs/xtr_it_tv_xfel"))
    parser.add_argument("--out", type=Path, default=Path("results/it_tv_extrapolated_maps"))
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/it_tv_extrapolated_maps/esrf_input_cleaning_summary.csv"),
    )
    parser.add_argument("--prefer-manifest", action="store_true")
    args = parser.parse_args()

    rows = []
    manifest = _manifest_sources(args.config_dir)
    for condition in args.conditions:
        config_path = args.config_dir / f"{condition}.yaml"
        config = yaml.safe_load(config_path.read_text())
        manifest_row = manifest.get(condition)
        condition_dir = args.out / condition
        dark_out = condition_dir / f"{condition}_dark_finite.mtz"
        triggered_out = condition_dir / f"{condition}_triggered_finite.mtz"

        dark_ds, dark = _clean_dataset(
            _source_path(config, manifest_row, "map_dark", "dark_mtz", args.prefer_manifest),
            ("F", "SIGF"),
        )
        triggered_ds, triggered = _clean_dataset(
            _source_path(
                config,
                manifest_row,
                "map_triggered",
                "triggered_mtz",
                args.prefer_manifest,
            ),
            ("F", "SIGF"),
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
        dark_out.parent.mkdir(parents=True, exist_ok=True)
        dark_common.write_mtz(str(dark_out))
        triggered_common.write_mtz(str(triggered_out))

        config["input_files"]["map_dark"] = str(dark_out)
        config["input_files"]["map_triggered"] = str(triggered_out)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))

        rows.append(
            {
                "condition": condition,
                "dark_source": dark["source"],
                "dark_rows_source": dark["rows_source"],
                "dark_rows_finite": dark["rows_finite"],
                "dark_rows_removed_nonfinite": dark["rows_removed_nonfinite"],
                "dark_cleaned_common": str(dark_out),
                "triggered_source": triggered["source"],
                "triggered_rows_source": triggered["rows_source"],
                "triggered_rows_finite": triggered["rows_finite"],
                "triggered_rows_removed_nonfinite": triggered["rows_removed_nonfinite"],
                "triggered_cleaned_common": str(triggered_out),
                "common_cleaned_reflections": len(common),
                "config": str(config_path),
            }
        )
        print(
            f"{condition}: finite dark {dark['rows_finite']}/{dark['rows_source']}, "
            f"triggered {triggered['rows_finite']}/{triggered['rows_source']}, "
            f"common={len(common)}"
        )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
