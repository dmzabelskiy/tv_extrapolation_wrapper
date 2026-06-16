#!/usr/bin/env python3
"""Rebuild the 5us benchmark-style extrapolated map from the saved reference diffmap."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import reciprocalspaceship as rs

from meteor.rsmap import Map
from xtr_estimator.configuration import dump_config
from xtr_estimator.estimation import plot_extrapolation_estimate
from xtr_estimator.main import parse_settings
from xtr_estimator.masking import make_inclusion_mask
from xtr_estimator.processing import get_maps, prepare_maps
from xtr_estimator.xtr_maps import save_extrapolated_map


CONFIG = Path("configs/xtr_it_tv_xfel/5us.yaml")
REFERENCE_DIFFMAP = Path("initial/5us_reference/extrapolated_best_guess_diffmap_ittv.mtz")
REFERENCE_XTR = Path("initial/5us_reference/extrapolated_best_guess_xtr6.88.mtz")
OUT_DIR = Path("results/it_tv_extrapolated_maps/5us_test")


def _complex_mtz(path: Path) -> rs.DataSeries:
    data = rs.read_mtz(path)
    return data["F"].to_numpy() * np.exp(1j * np.deg2rad(data["PHI"].to_numpy()))


def _compare(label: str, left: Path, right: Path) -> dict[str, float | int | str]:
    left_data = rs.read_mtz(str(left))
    right_data = rs.read_mtz(str(right))
    common = left_data.index.intersection(right_data.index)
    left_sf = left_data.loc[common, "F"].to_numpy() * np.exp(
        1j * np.deg2rad(left_data.loc[common, "PHI"].to_numpy())
    )
    right_sf = right_data.loc[common, "F"].to_numpy() * np.exp(
        1j * np.deg2rad(right_data.loc[common, "PHI"].to_numpy())
    )
    finite = np.isfinite(left_sf) & np.isfinite(right_sf)
    delta = left_sf[finite] - right_sf[finite]
    return {
        "comparison": label,
        "rows_left": len(left_data),
        "rows_right": len(right_data),
        "common": len(common),
        "finite_common": int(finite.sum()),
        "complex_rms_delta": float(np.sqrt(np.mean(np.abs(delta) ** 2))),
        "complex_mean_abs_delta": float(np.mean(np.abs(delta))),
        "amplitude_corr": float(
            np.corrcoef(
                left_data.loc[common, "F"].to_numpy()[finite],
                right_data.loc[common, "F"].to_numpy()[finite],
            )[0, 1]
        ),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = parse_settings(CONFIG, extra_overrides={})
    config_dict = dump_config(config)
    executed_config = Path(config_dict["general"]["output_folder"]) / "executed_config.yaml"
    if executed_config.exists():
        executed_config.replace(OUT_DIR / "rebuild_executed_config.yaml")
    unscaled_dark, unscaled_triggered = get_maps(config)
    _current_diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)

    reference_diffmap = Map.read_mtz_file(
        REFERENCE_DIFFMAP,
        amplitude_column="F",
        phase_column="PHI",
    )
    inclusion_mask = make_inclusion_mask(reference_diffmap, map_dark, config)
    _, _, prediction_tuple = plot_extrapolation_estimate(
        reference_diffmap,
        map_dark,
        inclusion_mask,
        config,
        compact=False,
    )
    chi = float(prediction_tuple[0])
    factor = 1.0 / chi

    diffmap_out = OUT_DIR / f"5us_test_it_tv_diffmap_chi_{chi:.6f}.mtz"
    reference_diffmap.write_mtz(diffmap_out)
    mtz_out = Path(
        save_extrapolated_map(
            factor,
            map_dark,
            reference_diffmap,
            dark_map_file_loc=config_dict["input_files"]["map_dark"],
            folder=OUT_DIR,
            name_prefix="5us_test_it_tv_extrapolated",
        )
    )
    stable_mtz_out = OUT_DIR / f"5us_test_it_tv_extrapolated_chi_{chi:.6f}_xtr{factor:.2f}.mtz"
    if mtz_out != stable_mtz_out:
        mtz_out.replace(stable_mtz_out)
        mtz_out = stable_mtz_out

    xtr_map = Map.read_mtz_file(mtz_out, amplitude_column="F", phase_column="PHI")
    ccp4_out = OUT_DIR / f"5us_test_it_tv_extrapolated_chi_{chi:.6f}_xtr{factor:.2f}.ccp4"
    xtr_map.to_ccp4_map(map_sampling=3).write_ccp4_map(str(ccp4_out))

    rows = [
        {
            "condition": "5us_test",
            "chi": f"{chi:.12f}",
            "extrapolation_factor": f"{factor:.12f}",
            "diffmap_mtz": str(diffmap_out),
            "extrapolated_mtz": str(mtz_out),
            "extrapolated_ccp4": str(ccp4_out),
            "reference_diffmap": str(REFERENCE_DIFFMAP),
            "reference_xtr": str(REFERENCE_XTR),
        }
    ]
    with (OUT_DIR / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    comparison = _compare("rebuilt_vs_reference_xtr", mtz_out, REFERENCE_XTR)
    with (OUT_DIR / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison))
        writer.writeheader()
        writer.writerow(comparison)

    print(f"chi={chi:.12f}")
    print(f"factor={factor:.12f}")
    print(f"diffmap={diffmap_out}")
    print(f"mtz={mtz_out}")
    print(f"ccp4={ccp4_out}")
    print(comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
