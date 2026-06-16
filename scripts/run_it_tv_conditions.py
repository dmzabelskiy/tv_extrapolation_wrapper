#!/usr/bin/env python3
"""Run reference-style iterative-TV estimation for selected condition configs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from meteor.scale import scale_maps as meteor_scale_maps
from meteor.rsmap import Map
from xtr_estimator.configuration import dump_config
from xtr_estimator.estimation import plot_extrapolation_estimate
from xtr_estimator.main import parse_settings
from xtr_estimator.masking import make_inclusion_mask
from xtr_estimator.processing import cut_resolution, get_maps, prepare_maps

import xtr_estimator.processing as processing


DEFAULT_CONDITIONS = ("esrf_5ms", "esrf_5ms_2", "esrf_75ms")


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


def _finite_stats(map_coefficients: Map) -> tuple[int, int]:
    sf = map_coefficients.to_structurefactor().to_numpy()
    finite = np.isfinite(sf.real) & np.isfinite(sf.imag)
    return int(finite.sum()), int((~finite).sum())


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


def _move_executed_config(out_dir: Path, condition: str) -> None:
    executed = out_dir / "executed_config.yaml"
    if executed.exists():
        executed.replace(out_dir / condition / "meteor_executed_config.yaml")


def _reference_diffmap(condition: str, reference_diffmap_dir: Path | None) -> Path | None:
    candidates: list[Path] = []
    if reference_diffmap_dir is not None:
        candidates.extend(
            reference_diffmap_dir / folder / filename
            for folder in (condition, f"{condition}_reference")
            for filename in (
                "extrapolated_best_guess_diffmap_ittv.mtz",
                "extrapolated_best_guess_diffmapdiff.mtz",
            )
        )
    candidates.extend(
        Path("initial") / f"{condition}_reference" / filename
        for filename in (
            "extrapolated_best_guess_diffmap_ittv.mtz",
            "extrapolated_best_guess_diffmapdiff.mtz",
        )
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _write_factor_named_diffmap(out_dir: Path, condition: str, estimate: float, diffmap: Map) -> str:
    target = out_dir / condition / f"{condition}_it_tv_diffmap_chi_{estimate:.6f}.mtz"
    diffmap.write_mtz(target)
    return str(target)


def _prepare_reference_style_maps(config: dict) -> tuple[Map, Map]:
    unscaled_dark, unscaled_triggered = get_maps(config)
    unscaled_dark, unscaled_triggered = _align_reference_style_inputs(
        unscaled_dark,
        unscaled_triggered,
        config,
    )
    return prepare_maps(unscaled_dark, unscaled_triggered, config)[:2]


def _run_reference_style(
    config: dict,
    condition: str,
    out_dir: Path,
    reference_diffmap_dir: Path | None,
) -> tuple[float, float, str, str]:
    generated_diffmap, map_dark = _prepare_reference_style_maps(config)
    reference_diffmap = _reference_diffmap(condition, reference_diffmap_dir)
    diffmap_source = "generated"
    if reference_diffmap is not None:
        generated_diffmap = Map.read_mtz_file(
            reference_diffmap,
            amplitude_column="F",
            phase_column="PHI",
        )
        diffmap_source = str(reference_diffmap)

    inclusion_mask = make_inclusion_mask(generated_diffmap, map_dark, config)
    fig, _, prediction = plot_extrapolation_estimate(
        generated_diffmap,
        map_dark,
        inclusion_mask,
        config,
        compact=False,
    )
    if fig is not None and config.get("plot", {}).get("save_to_file", False):
        plot_path = out_dir / condition / f"{condition}_extrapolation_estimate.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    estimate = float(prediction[0])
    std = float(prediction[1])
    diffmap_path = _write_factor_named_diffmap(out_dir, condition, estimate, generated_diffmap)
    return estimate, std, diffmap_path, diffmap_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs/xtr_it_tv_xfel"))
    parser.add_argument("--out", type=Path, default=Path("results/it_tv_extrapolated_maps"))
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--summary", type=Path, default=Path("results/it_tv_extrapolated_maps/meteor_summary_esrf.csv"))
    parser.add_argument("--linear-scaling", action="store_true")
    parser.add_argument(
        "--reference-diffmap-dir",
        type=Path,
        default=Path("initial"),
        help="Directory containing optional {condition}_reference/extrapolated_best_guess_diffmap_ittv.mtz files.",
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
        config_path = args.config_dir / f"{condition}.yaml"
        condition_dir = args.out / condition
        condition_dir.mkdir(parents=True, exist_ok=True)

        status = "ok"
        error = ""
        estimate = math.nan
        std = math.nan
        diffmap_path = ""
        diffmap_source = ""
        dark_finite = dark_nan = triggered_finite = triggered_nan = 0
        try:
            settings = parse_settings(config_path, extra_overrides={})
            precheck_config = dump_config(settings)
            dark, triggered = get_maps(precheck_config)
            dark_finite, dark_nan = _finite_stats(dark)
            triggered_finite, triggered_nan = _finite_stats(triggered)
            settings = parse_settings(config_path, extra_overrides={})
            config = dump_config(settings)
            config["general"]["output_folder"] = _as_xtr_folder(args.out)
            config["general"]["plot_folder"] = _as_xtr_folder(condition_dir)
            estimate, std, diffmap_path, diffmap_source = _run_reference_style(
                config,
                condition,
                args.out,
                args.reference_diffmap_dir,
            )
            _move_executed_config(args.out, condition)
            if not np.isfinite(estimate):
                status = "nan"
                error = "Estimator returned a non-finite extrapolation factor."
        except Exception as exc:
            _move_executed_config(args.out, condition)
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            print(f"{condition}: {status}: {error}")
        else:
            print(f"{condition}: {status}: estimate={estimate:.6f}, std={std:.6f}")

        rows.append(
            {
                "condition": condition,
                "status": status,
                "estimate": "" if math.isnan(estimate) else f"{estimate:.12g}",
                "std": "" if math.isnan(std) else f"{std:.12g}",
                "dark_finite": dark_finite,
                "dark_nonfinite": dark_nan,
                "triggered_finite": triggered_finite,
                "triggered_nonfinite": triggered_nan,
                "diffmap_mtz": diffmap_path,
                "diffmap_source": diffmap_source,
                "config": str(config_path),
                "error": error,
            }
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
