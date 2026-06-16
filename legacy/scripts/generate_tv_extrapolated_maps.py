#!/usr/bin/env python3
"""Generate TV-extrapolated MTZs and 2mFo-DFc maps for numeric TV estimates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np


DEFAULT_PHENIX_MAPS = Path("/home/dmitrii/phenix-2.0-5936/phenix_bin/phenix.maps")


def _enable_safe_xtr_paths() -> None:
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
        high_res_limit = float(max(dmin_dark, dmin_triggered))
        if not np.isclose(high_res_limit, general_config["high_resolution_limit"]):
            general_config["high_resolution_limit"] = high_res_limit
        return map_dark, map_triggered

    processing.scale_maps = safe_scale_maps
    processing.check_highres_limit = safe_check_highres_limit


def _read_numeric_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("estimate")
        and row.get("estimate", "").lower() != "nan"
    ]


def _load_cases(strict_summary: Path, moderate_summary: Path) -> list[dict[str, str]]:
    cases = []
    for row in _read_numeric_rows(strict_summary):
        row = dict(row)
        row["source"] = "strict_original"
        cases.append(row)
    for row in _read_numeric_rows(moderate_summary):
        row = dict(row)
        row["source"] = "moderate_common_basis"
        cases.append(row)
    return cases


def _phenix_maps_params(
    *,
    pdb_path: Path,
    xtr_mtz: Path,
    out_dir: Path,
    condition: str,
) -> str:
    ccp4_path = (out_dir / f"{condition}_tv_2mFo-DFc.ccp4").resolve()
    return f"""maps {{
  input {{
    pdb_file_name = {pdb_path.resolve()}
    reflection_data {{
      file_name = {xtr_mtz.resolve()}
      labels = F
      r_free_flags {{
        required = False
        ignore_r_free_flags = True
      }}
    }}
  }}
  output {{
    directory = {out_dir.resolve()}
    prefix = {condition}_tv_2mFo-DFc
    include_r_free_flags = False
  }}
  map_coefficients {{
    map_type = 2mFo-DFc
    format = mtz
    mtz_label_amplitudes = TV2FOFCWT
    mtz_label_phases = PHTV2FOFCWT
    fill_missing_f_obs = False
    exclude_free_r_reflections = False
  }}
  map {{
    map_type = 2mFo-DFc
    format = ccp4
    file_name = {ccp4_path}
    fill_missing_f_obs = False
    region = cell
    exclude_free_r_reflections = False
  }}
}}
"""


def _run_phenix_maps(
    phenix_maps: Path,
    params_path: Path,
    log_path: Path,
) -> int:
    result = subprocess.run(
        [str(phenix_maps.resolve()), str(params_path.resolve())],
        text=True,
        capture_output=True,
    )
    log_path.write_text(
        "$ " + " ".join([str(phenix_maps.resolve()), str(params_path.resolve())])
        + "\n\nSTDOUT:\n"
        + result.stdout
        + "\nSTDERR:\n"
        + result.stderr
    )
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-summary", type=Path, default=Path("results/xtr/summary.csv"))
    parser.add_argument(
        "--moderate-summary",
        type=Path,
        default=Path("results/xtr_tv_isomorphous_moderate/summary.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/tv_extrapolated_maps"),
    )
    parser.add_argument("--phenix-maps", type=Path, default=DEFAULT_PHENIX_MAPS)
    args = parser.parse_args()

    if not args.phenix_maps.exists():
        print(f"phenix.maps not found: {args.phenix_maps}", file=sys.stderr)
        return 1

    _enable_safe_xtr_paths()

    from xtr_estimator.configuration import dump_config
    from xtr_estimator.main import parse_settings
    from xtr_estimator.processing import get_maps, prepare_maps
    from xtr_estimator.xtr_maps import save_extrapolated_map

    cases = _load_cases(args.strict_summary, args.moderate_summary)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in cases:
        condition = row["condition"]
        config_path = Path(row["config"])
        xtr_factor = float(row["estimate"])
        case_out = args.out / condition
        case_out.mkdir(parents=True, exist_ok=True)

        status = "ok"
        error = ""
        xtr_mtz = ""
        coeff_mtz = ""
        ccp4_map = ""
        params_path = case_out / f"{condition}_phenix_maps.params"
        log_path = case_out / "phenix_maps.log"
        try:
            settings = parse_settings(data_yaml=config_path, extra_overrides={})
            config = dump_config(settings)
            unscaled_dark, unscaled_triggered = get_maps(config)
            diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)
            xtr_mtz = save_extrapolated_map(
                xtr_factor=xtr_factor,
                map_dark=map_dark,
                diffmap=diffmap,
                dark_map_file_loc=config["input_files"]["map_dark"],
                folder=case_out,
                name_prefix=f"{condition}_tv",
                file_loc_diff="_tv",
            )
            params_path.write_text(
                _phenix_maps_params(
                    pdb_path=Path(config["input_files"]["pdb_dark"]),
                    xtr_mtz=Path(xtr_mtz),
                    out_dir=case_out,
                    condition=condition,
                )
            )
            returncode = _run_phenix_maps(args.phenix_maps, params_path, log_path)
            if returncode != 0:
                status = "phenix_maps_error"
                error = f"phenix.maps returned {returncode}; see {log_path}"
            coeff_candidates = sorted(case_out.glob(f"{condition}_tv_2mFo-DFc*.mtz"))
            ccp4_candidates = sorted(case_out.glob(f"{condition}_tv_2mFo-DFc*.ccp4"))
            coeff_mtz = str(coeff_candidates[-1]) if coeff_candidates else ""
            ccp4_map = str(ccp4_candidates[-1]) if ccp4_candidates else ""
            if status == "ok" and not coeff_mtz:
                status = "missing_coeff_mtz"
                error = "phenix.maps completed but no coefficient MTZ was found"
            if status == "ok" and not ccp4_map:
                status = "missing_ccp4"
                error = "phenix.maps completed but no CCP4 map was found"
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "condition": condition,
                "source": row["source"],
                "tv_estimate": f"{xtr_factor:.8f}",
                "tv_std": row.get("std", ""),
                "status": status,
                "xtr_mtz": str(xtr_mtz),
                "diffmap_mtz": str(case_out / f"{condition}_tv_diffmap_tv.mtz"),
                "two_mfo_dfc_coeff_mtz": coeff_mtz,
                "two_mfo_dfc_ccp4": ccp4_map,
                "params": str(params_path),
                "log": str(log_path),
                "error": error,
            }
        )
        print(f"{condition}: {status} xtr={xtr_mtz}")

    summary = args.out / "summary.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    failures = sum(row["status"] != "ok" for row in rows)
    print(f"Wrote {summary}; {len(rows) - failures} ok, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
