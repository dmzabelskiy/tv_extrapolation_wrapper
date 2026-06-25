from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal

from .config import DatasetConfig
from .pipeline import run
from .occupancy_scan import run_phenix_adp_refine, run_scan


def _collect_configs(paths: list[Path]) -> list[Path]:
    result = []
    for p in paths:
        if p.is_dir():
            result.extend(sorted(p.rglob("*.yaml")))
        else:
            result.append(p)
    return result


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

    # --- refine-extrap subcommand ---
    re_parser = subparsers.add_parser(
        "refine-extrap",
        help="Refine dark PDB against extrapolated MTZ to produce an excited-state model.",
    )
    re_parser.add_argument("dark_pdb", type=Path, help="Dark-state PDB (ground)")
    re_parser.add_argument("extrap_mtz", type=Path, help="Phenix-ready extrapolated MTZ")
    re_parser.add_argument("--out-dir", type=Path, required=True, dest="out_dir",
                           help="Output directory for refinement files")
    re_parser.add_argument("--cif", type=Path, action="append", default=[], dest="cif_files",
                           metavar="FILE", help="Ligand CIF restraint file (repeatable)")
    re_parser.add_argument("--cpus", type=int, default=2)
    re_parser.add_argument("--phenix-bin", default="phenix.refine", dest="phenix_bin",
                           help="Path to phenix.refine binary")
    re_parser.add_argument("--strategy", default="individual_sites+individual_adp",
                           help="Phenix refinement strategy (default: individual_sites+individual_adp)")

    # --- scan subcommand ---
    sc_parser = subparsers.add_parser(
        "scan",
        help="Scan occupancy x-grid: refine mixed models vs triggered MTZ, plot Rfree vs x.",
    )
    sc_parser.add_argument("ground_pdb", type=Path, help="Ground-state (dark) PDB")
    sc_parser.add_argument("extrap_pdb", type=Path, help="Excited-state PDB (from refine-extrap)")
    sc_parser.add_argument("triggered_mtz", type=Path, help="Raw triggered MTZ")
    sc_parser.add_argument("--out-dir", type=Path, required=True, dest="out_dir",
                           help="Output directory")
    sc_parser.add_argument("--x-grid", nargs="+", type=float, dest="x_grid",
                           default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
                           help="Occupancy x values to scan")
    sc_parser.add_argument("--cif", type=Path, action="append", default=[], dest="cif_files",
                           metavar="FILE")
    sc_parser.add_argument("--cpus", type=int, default=2)
    sc_parser.add_argument("--phenix-bin", default="phenix.refine", dest="phenix_bin")
    sc_parser.add_argument("--mode", choices=["occupancy", "coords"], default="occupancy")
    sc_parser.add_argument("--strategy", default="individual_adp",
                           help="Phenix refinement strategy (default: individual_adp)")
    sc_parser.add_argument("--cycles", type=int, default=1,
                           help="Number of phenix.refine macrocycles (default: 1)")

    args = parser.parse_args(argv)

    if args.command == "run":
        config_paths = _collect_configs(args.configs)
        if _is_direct_mode(config_paths):
            dark_mtz, triggered_mtz, pdb = config_paths
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
            for config_path in config_paths:
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

    elif args.command == "refine-extrap":
        log_path, ok = run_phenix_adp_refine(
            args.dark_pdb, args.extrap_mtz, args.out_dir,
            cif_files=args.cif_files,
            cpus=args.cpus,
            phenix_bin=args.phenix_bin,
            strategy=args.strategy,
        )
        status = "ok" if ok else "FAILED"
        print(f"refine-extrap {status}: log at {log_path}")
        return 0 if ok else 1

    elif args.command == "scan":
        result = run_scan(
            args.ground_pdb, args.extrap_pdb, args.triggered_mtz,
            out_dir=args.out_dir,
            x_grid=args.x_grid,
            cif_files=args.cif_files,
            cpus=args.cpus,
            phenix_bin=args.phenix_bin,
            mode=args.mode,
            strategy=args.strategy,
            cycles=args.cycles,
        )
        if result.best is not None:
            print(f"Best x={result.best.x:.3f}  Rfree={result.best.rfree}  Rwork={result.best.rwork}")
        if result.plot_path:
            print(f"Plot: {result.plot_path}")
        print(f"CSV:  {args.out_dir / 'scan_results.csv'}")
        return 0

    return 1
