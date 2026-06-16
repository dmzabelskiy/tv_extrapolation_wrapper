#!/usr/bin/env python3
"""Generate reference-style TV-extrapolated complex structure factors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import reciprocalspaceship as rs

from meteor.scale import scale_maps as meteor_scale_maps
from meteor.rsmap import Map
from xtr_estimator.configuration import dump_config
from xtr_estimator.main import parse_settings
from xtr_estimator.processing import cut_resolution, get_maps, prepare_maps
from xtr_estimator.xtr_maps import save_extrapolated_map

import xtr_estimator.processing as processing


def _as_xtr_folder(path: Path) -> str:
    return str(path) + "/"


def _install_reference_highres_check() -> None:
    def check_highres_limit_reference(map_dark: Map, map_triggered: Map, general_config: dict):
        dmin_dark = float(map_dark.compute_dHKL().min())
        dmin_triggered = float(map_triggered.compute_dHKL().min())
        requested = float(general_config["high_resolution_limit"])
        effective = max(dmin_dark, dmin_triggered)
        if effective - requested > 0.01:
            general_config["high_resolution_limit"] = effective
        return map_dark, map_triggered

    processing.check_highres_limit = check_highres_limit_reference


def _estimate_for(condition: str, summary_paths: tuple[Path, ...]) -> float:
    for summary in reversed(summary_paths):
        if not summary.exists():
            continue
        with summary.open() as handle:
            for row in csv.DictReader(handle):
                if row.get("condition") == condition and row.get("estimate"):
                    return float(row["estimate"])
    raise KeyError(f"No extrapolation estimate available for {condition}")


def _diffmap_path(condition: str, out_dir: Path, chi: float) -> Path:
    factor_named = out_dir / condition / f"{condition}_it_tv_diffmap_chi_{chi:.6f}.mtz"
    if factor_named.exists():
        return factor_named
    factor_candidates = sorted((out_dir / condition).glob(f"{condition}_it_tv_diffmap_chi_*.mtz"))
    if len(factor_candidates) == 1:
        return factor_candidates[0]
    candidates = sorted(
        (out_dir / condition).glob("diffmap_*_it_tv_*.mtz")
    )
    if len(candidates) != 1:
        raise ValueError(f"Expected one it_tv MTZ for {condition}, found {candidates}")
    return candidates[0]


def _write_ccp4(map_coefficients: Map, path: Path, sampling: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ccp4 = map_coefficients.to_ccp4_map(map_sampling=sampling)
    ccp4.write_ccp4_map(str(path))


def _minimum_resolution(map_coefficients: Map) -> float:
    return float(map_coefficients.compute_dHKL().min())


def _align_reference_style_inputs(
    unscaled_dark: Map,
    unscaled_triggered: Map,
    config: dict,
) -> tuple[Map, Map]:
    unscaled_triggered.cell = unscaled_dark.cell
    unscaled_triggered.spacegroup = unscaled_dark.spacegroup
    unscaled_dark = cut_resolution(
        unscaled_dark,
        high_resolution_limit=config["general"]["high_resolution_limit"],
    )
    unscaled_triggered = cut_resolution(
        unscaled_triggered,
        high_resolution_limit=config["general"]["high_resolution_limit"],
    )
    common_index = unscaled_dark.index.intersection(unscaled_triggered.index)
    unscaled_dark = unscaled_dark.loc[common_index].copy()
    unscaled_triggered = unscaled_triggered.loc[common_index].copy()
    for map_coefficients in (unscaled_dark, unscaled_triggered):
        map_coefficients.cell = unscaled_dark.cell
        map_coefficients.spacegroup = unscaled_dark.spacegroup

    effective_limit = max(
        config["general"]["high_resolution_limit"],
        _minimum_resolution(unscaled_dark),
        _minimum_resolution(unscaled_triggered),
    )
    if effective_limit - config["general"]["high_resolution_limit"] > 0.01:
        config["general"]["high_resolution_limit"] = effective_limit
    return unscaled_dark, unscaled_triggered


def _prepare_reference_style_dark_map(config: dict) -> Map:
    unscaled_dark, unscaled_triggered = get_maps(config)
    unscaled_dark, unscaled_triggered = _align_reference_style_inputs(
        unscaled_dark,
        unscaled_triggered,
        config,
    )
    _diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)
    return map_dark


def _write_extrapolated_mtz(
    condition: str,
    chi: float,
    out_dir: Path,
    config_dir: Path,
) -> dict[str, str | float | int]:
    condition_dir = out_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    config = dump_config(parse_settings(config_dir / f"{condition}.yaml", extra_overrides={}))
    config["general"]["output_folder"] = _as_xtr_folder(out_dir)
    config["general"]["plot_folder"] = _as_xtr_folder(condition_dir)
    raw_dark = rs.read_mtz(str(config["input_files"]["map_dark"]))
    dark_map = _prepare_reference_style_dark_map(config)
    executed_config = out_dir / "executed_config.yaml"
    if executed_config.exists():
        executed_config.replace(condition_dir / "extrapolation_executed_config.yaml")
    diffmap_path = _diffmap_path(condition, out_dir, chi)
    diff_map = Map.read_mtz_file(diffmap_path, amplitude_column="F", phase_column="PHI")

    factor = 1.0 / chi
    raw_mtz_path = Path(
        save_extrapolated_map(
            factor,
            dark_map,
            diff_map,
            dark_map_file_loc=config["input_files"]["map_dark"],
            folder=condition_dir,
            name_prefix=f"{condition}_it_tv_extrapolated",
        )
    )
    mtz_path = condition_dir / f"{condition}_it_tv_extrapolated_xtr{factor:.2f}.mtz"
    if raw_mtz_path != mtz_path:
        raw_mtz_path.replace(mtz_path)

    extrapolated_map = Map.read_mtz_file(mtz_path, amplitude_column="F", phase_column="PHI")
    ccp4_path = condition_dir / f"{condition}_it_tv_extrapolated_xtr{factor:.2f}.ccp4"
    _write_ccp4(extrapolated_map, ccp4_path)

    dark_sf = dark_map.to_structurefactor()
    diff_sf = diff_map.to_structurefactor()
    common_index = raw_dark.index.intersection(dark_sf.index).intersection(diff_sf.index)

    return {
        "condition": condition,
        "chi": chi,
        "extrapolation_factor": factor,
        "n_reflections_dark": len(raw_dark),
        "n_reflections_common": len(common_index),
        "n_reflections_missing": len(raw_dark) - len(common_index),
        "dark_mtz": config["input_files"]["map_dark"],
        "diffmap_mtz": str(diffmap_path),
        "extrapolated_mtz": str(mtz_path),
        "extrapolated_ccp4": str(ccp4_path),
        "model": config["input_files"]["pdb_dark"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/it_tv_extrapolated_maps"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs/xtr_it_tv_xfel"))
    parser.add_argument("--conditions", nargs="+", default=["5us", "10ms", "10ns"])
    parser.add_argument("--linear-scaling", action="store_true")
    parser.add_argument(
        "--estimate-summary",
        type=Path,
        action="append",
        default=[
            Path("results/it_tv_extrapolated_maps/meteor_summary_esrf.csv"),
            Path("results/it_tv_extrapolated_maps/meteor_summary.csv"),
        ],
    )
    args = parser.parse_args()
    _install_reference_highres_check()

    if args.linear_scaling:
        def scale_maps_linear(*, reference_map, map_to_scale, **kwargs):
            kwargs["least_squares_loss"] = "linear"
            return meteor_scale_maps(
                reference_map=reference_map,
                map_to_scale=map_to_scale,
                **kwargs,
            )

        processing.scale_maps = scale_maps_linear

    rows = []
    for condition in args.conditions:
        row = _write_extrapolated_mtz(
            condition,
            _estimate_for(condition, tuple(args.estimate_summary)),
            args.out,
            args.config_dir,
        )
        rows.append(row)
        print(f"{condition}: wrote {row['extrapolated_mtz']}")

    summary = args.out / "summary.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
