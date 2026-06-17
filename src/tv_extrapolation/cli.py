from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal

from .config import DatasetConfig
from .pipeline import run


def _is_direct_mode(paths: list[Path]) -> bool:
    """True when the caller passed (dark.mtz, triggered.mtz, structure.pdb)."""
    return (
        len(paths) == 3
        and paths[0].suffix.lower() == ".mtz"
        and paths[1].suffix.lower() == ".mtz"
        and paths[2].suffix.lower() == ".pdb"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tv-extrapolate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the it_tv pipeline.  Pass YAML config files, or a single triplet "
             "dark.mtz triggered.mtz structure.pdb for a quick one-shot run.",
    )
    run_parser.add_argument(
        "configs",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="Dataset YAML files — or — dark.mtz triggered.mtz structure.pdb",
    )
    run_parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/summary.csv"),
        help="Output summary CSV path (default: results/summary.csv)",
    )

    # Direct-mode overrides (ignored in YAML mode)
    dm = run_parser.add_argument_group("direct-mode options (dark.mtz triggered.mtz pdb)")
    dm.add_argument(
        "--name",
        type=str,
        default=None,
        help="Dataset name (default: triggered MTZ filename stem)",
    )
    dm.add_argument(
        "--resolution",
        type=float,
        default=None,
        metavar="Å",
        help="High-resolution cutoff in Å (default: coarser of the two MTZ files)",
    )
    dm.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: results/)",
    )
    dm.add_argument(
        "--scaling-loss",
        choices=["huber", "linear", "huber_safe"],
        default="huber",
        dest="scaling_loss",
        help="Scaling loss function (default: huber)",
    )
    dm.add_argument(
        "--finite-filter",
        action="store_true",
        default=False,
        dest="finite_filter",
        help="Drop non-finite reflections before scaling",
    )
    dm.add_argument(
        "--rewrite-pdb-cell",
        action="store_true",
        default=False,
        dest="rewrite_pdb_cell",
        help="Rewrite the PDB CRYST1 record to match the dark MTZ unit cell",
    )
    dm.add_argument(
        "--phenix-refine-cell",
        action="store_true",
        default=False,
        dest="phenix_refine_cell",
        help="After rewriting the PDB cell, run phenix.refine rigid-body to "
             "re-seat the model (implies --rewrite-pdb-cell; requires Phenix in PATH)",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        if _is_direct_mode(args.configs):
            dark_mtz, triggered_mtz, pdb = args.configs
            print(f"Direct mode: auto-detecting columns and resolution from MTZ files")
            config = DatasetConfig.from_files(
                dark_mtz,
                triggered_mtz,
                pdb,
                name=args.name,
                resolution_limit=args.resolution,
                output_dir=args.output or Path("results"),
                scaling_loss=args.scaling_loss,
                finite_filter=args.finite_filter,
                rewrite_pdb_cell=args.rewrite_pdb_cell or args.phenix_refine_cell,
                phenix_refine_cell=args.phenix_refine_cell,
            )
            print(
                f"  name={config.name!r}  resolution={config.resolution_limit} Å  "
                f"dark={config.columns['dark'].kind}/{config.columns['dark'].amplitude_or_intensity}  "
                f"triggered={config.columns['triggered'].kind}/{config.columns['triggered'].amplitude_or_intensity}"
            )
            result = run(config)
            rows = [result.as_row()]
            print(f"{result.condition}: {result.status}: chi={result.chi}, factor={result.extrapolation_factor}")
        else:
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
