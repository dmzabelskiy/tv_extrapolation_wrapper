#!/usr/bin/env python3
"""Run generated xtr-estimator configs and write a summary CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import traceback


def _load_xtr():
    try:
        from xtr_estimator.configuration import dump_config
        from xtr_estimator.main import execute_as_main, parse_settings
    except ImportError as exc:
        raise SystemExit(
            "Could not import xtr_estimator. Create/activate the Python 3.12 "
            "environment from environment.yml first."
        ) from exc
    return parse_settings, dump_config, execute_as_main


def _enable_safe_scale() -> None:
    """Patch fragile xtr-estimator/Meteor preprocessing paths for commonized MTZs."""

    import numpy as np
    import xtr_estimator.processing as processing
    from meteor.scale import scale_maps as meteor_scale_maps

    def safe_scale_maps(*, reference_map, map_to_scale, **kwargs):
        kwargs["least_squares_loss"] = "linear"
        return meteor_scale_maps(
            reference_map=reference_map,
            map_to_scale=map_to_scale,
            **kwargs,
        )

    processing.scale_maps = safe_scale_maps

    def safe_check_highres_limit(map_dark, map_triggered, general_config):
        dmin_dark = map_dark.compute_dHKL().min()
        dmin_triggered = map_triggered.compute_dHKL().min()
        high_res_limit = float(max(dmin_dark, dmin_triggered))
        if not np.isclose(high_res_limit, general_config["high_resolution_limit"]):
            general_config["high_resolution_limit"] = high_res_limit
        return map_dark, map_triggered

    processing.check_highres_limit = safe_check_highres_limit


def _prediction_fields(prediction) -> tuple[str, str]:
    if prediction is None:
        return "", ""
    try:
        estimate, std = prediction
    except (TypeError, ValueError):
        return str(prediction), ""
    return str(estimate), str(std)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, default=Path("configs/xtr"))
    parser.add_argument("--summary", type=Path, default=Path("results/xtr/summary.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--safe-scale",
        action="store_true",
        help="Patch xtr-estimator to call Meteor scaling with linear least-squares loss.",
    )
    args = parser.parse_args()

    config_paths = sorted(args.configs.glob("*.yaml"))
    if args.limit is not None:
        config_paths = config_paths[: args.limit]
    if not config_paths:
        print(f"No YAML configs found in {args.configs}", file=sys.stderr)
        return 1

    parse_settings, dump_config, execute_as_main = _load_xtr()
    if args.safe_scale:
        _enable_safe_scale()
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for config_path in config_paths:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": str(config_path),
            "condition": config_path.stem,
            "status": "ok",
            "estimate": "",
            "std": "",
            "error": "",
        }
        try:
            settings = parse_settings(data_yaml=config_path, extra_overrides={})
            config = dump_config(settings)
            prediction = execute_as_main(config, show=False)
            estimate, std = _prediction_fields(prediction)
            row["estimate"] = estimate
            row["std"] = std
            if estimate.lower() == "nan" or std.lower() == "nan":
                row["status"] = "nan"
                row["error"] = "Estimator returned nan; inspect plot and mask statistics."
        except Exception as exc:  # noqa: BLE001 - batch runner must continue.
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            log_path = args.summary.parent / f"{config_path.stem}_error.log"
            log_path.write_text(traceback.format_exc())
        rows.append(row)
        print(f"{row['condition']}: {row['status']} {row['estimate']} {row['std']}")

    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    failures = sum(row["status"] != "ok" for row in rows)
    print(f"Wrote {args.summary}; {len(rows) - failures} ok, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
