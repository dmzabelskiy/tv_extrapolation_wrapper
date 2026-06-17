from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import DatasetConfig
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tv-extrapolate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the it_tv pipeline for one or more dataset configs")
    run_parser.add_argument("configs", nargs="+", type=Path)
    run_parser.add_argument("--summary", type=Path, default=Path("results/summary.csv"))

    args = parser.parse_args(argv)

    if args.command == "run":
        rows = []
        for config_path in args.configs:
            config = DatasetConfig.from_yaml(config_path)
            result = run(config)
            rows.append(result.as_row())
            print(f"{result.condition}: {result.status}: chi={result.chi}, factor={result.extrapolation_factor}")

        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.summary}")
        return 0

    return 1
