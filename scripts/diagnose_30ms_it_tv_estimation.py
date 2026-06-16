#!/usr/bin/env python3
"""Inspect 30ms xtr-estimator prediction thresholds using the cached it_tv map."""

from __future__ import annotations

import numpy as np

from xtr_estimator.configuration import dump_config
from xtr_estimator.estimation import _calculate_statistics, cummean_and_errors
from xtr_estimator.main import parse_settings
from xtr_estimator.masking import make_inclusion_mask
from xtr_estimator.processing import get_maps, prepare_maps


def _enable_safe_processing() -> None:
    import xtr_estimator.processing as processing
    from meteor.scale import scale_maps as meteor_scale_maps

    def safe_scale_maps(*, reference_map, map_to_scale, **kwargs):
        kwargs["least_squares_loss"] = "linear"
        return meteor_scale_maps(
            reference_map=reference_map,
            map_to_scale=map_to_scale,
            **kwargs,
        )

    def safe_check_highres_limit(map_dark, map_triggered, general_config):
        dmin_dark = map_dark.compute_dHKL().min()
        dmin_triggered = map_triggered.compute_dHKL().min()
        general_config["high_resolution_limit"] = float(max(dmin_dark, dmin_triggered))
        return map_dark, map_triggered

    processing.scale_maps = safe_scale_maps
    processing.check_highres_limit = safe_check_highres_limit


def _prediction(cummean_dict: dict, plot_config: dict) -> tuple[float, float, int]:
    std_cutoff = plot_config["std_cutoff"]
    thresh_line = cummean_dict["thresh_line"]
    average_distance_mask = (
        cummean_dict["pseudo_sort"] + std_cutoff * cummean_dict["pseudo_std"]
        > thresh_line
    )
    bottom_index = 0
    if np.any(average_distance_mask):
        bottom_index = int(np.where(average_distance_mask)[0][0])
    if bottom_index <= 10:
        return np.nan, np.nan, bottom_index
    middle_diff = (
        cummean_dict["diff_sigma"][bottom_index] + cummean_dict["diff_sigma"][0]
    ) / 2
    middle_index = int(np.where(middle_diff > cummean_dict["diff_sigma"])[0][0])
    middle_index = max(middle_index, 10)
    return (
        float(cummean_dict["pseudo_sort"][middle_index]),
        float(cummean_dict["pseudo_std"][middle_index]),
        bottom_index,
    )


def main() -> int:
    _enable_safe_processing()
    settings = parse_settings("configs/xtr_it_tv_xfel/30ms.yaml", extra_overrides={})
    config = dump_config(settings)
    dark, triggered = get_maps(config)
    diffmap, map_dark, _ = prepare_maps(dark, triggered, config)
    inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)

    general_config = config["general"]
    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=general_config["map_sampling"])
    map_dark_np = map_dark.to_3d_numpy_map(map_sampling=general_config["map_sampling"])
    stats = _calculate_statistics(diffmap_np, map_dark_np, inclusion_mask)

    print("solvent_density,std_cutoff,bottom_index,estimate,std")
    for solvent_density in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40):
        for std_cutoff in (1.0, 2.0, 3.0, 4.0):
            plot_config = dict(config["plot"])
            plot_config["solvent_density"] = solvent_density
            plot_config["std_cutoff"] = std_cutoff
            cummean = cummean_and_errors(stats, number_sym_ops=1, plot_config=plot_config)
            estimate, std, bottom_index = _prediction(cummean, plot_config)
            print(f"{solvent_density},{std_cutoff},{bottom_index},{estimate},{std}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
