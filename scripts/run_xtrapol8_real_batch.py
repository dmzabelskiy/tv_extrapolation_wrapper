#!/usr/bin/env python3
"""Run the standalone Xtrapol8 Fextr.py implementation for every dataset."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys


DEFAULT_PHENIX_PYTHON = Path("/home/dmitrii/phenix-2.0-5936/phenix_bin/phenix.python")
DEFAULT_XTRAPOL8 = Path("external/Xtrapol8_py3/Fextr.py")
DEFAULT_REFLECTION_CONVERTER = Path(
    "/home/dmitrii/phenix-2.0-5936/phenix_bin/phenix.reflection_file_converter"
)
DEFAULT_OCCUPANCIES = "0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35"


def _condition_files(condition_dir: Path) -> tuple[Path, Path, Path]:
    mtzs = sorted(condition_dir.glob("*.mtz"))
    pdbs = sorted(condition_dir.glob("*.pdb"))
    if len(mtzs) != 2:
        raise ValueError(f"Expected exactly 2 MTZ files in {condition_dir}, found {len(mtzs)}")
    if len(pdbs) != 1:
        raise ValueError(f"Expected exactly 1 PDB file in {condition_dir}, found {len(pdbs)}")
    ground = [path for path in mtzs if "ground" in path.name.lower()]
    if not ground:
        raise ValueError(f"Could not identify reference/ground MTZ in {condition_dir}")
    if len(ground) > 1:
        raise ValueError(f"Multiple reference/ground MTZ candidates in {condition_dir}: {ground}")
    reference_mtz = ground[0]
    triggered_mtz = [path for path in mtzs if path != reference_mtz][0]
    return pdbs[0], reference_mtz, triggered_mtz


def _discover_conditions(initial: Path) -> list[tuple[str, Path, Path, Path]]:
    conditions = []
    for condition_dir in sorted(path for path in initial.iterdir() if path.is_dir()):
        model, reference_mtz, triggered_mtz = _condition_files(condition_dir)
        conditions.append((condition_dir.name, model, reference_mtz, triggered_mtz))
    return conditions


def _mtz_columns(path: Path) -> set[str]:
    data = path.read_bytes()[-65536:]
    return {
        match.group(1).decode("latin1")
        for match in re.finditer(rb"COLUMN\s+(\S+)\s+[A-Z]\s+", data)
    }


def _prepare_fobs_mtz(
    source: Path,
    target: Path,
    reflection_converter: Path,
    force: bool,
) -> tuple[Path, str]:
    if target.exists() and not force:
        return target, "existing"

    columns = _mtz_columns(source)
    if {"F", "SIGF"}.issubset(columns):
        command = [
            str(reflection_converter),
            str(source),
            "--label",
            "F",
            "--mtz",
            str(target),
            "--mtz_root_label",
            "FOBS",
        ]
        mode = "amplitude"
    elif {"I", "SIGI"}.issubset(columns):
        command = [
            str(reflection_converter),
            str(source),
            "--label",
            "I",
            "--mtz",
            str(target),
            "--write_mtz_amplitudes",
            "--massage_intensities",
            "--mtz_root_label",
            "FOBS",
        ]
        mode = "intensity_to_amplitude"
    elif {"IMEAN", "SIGIMEAN"}.issubset(columns):
        command = [
            str(reflection_converter),
            str(source),
            "--label",
            "IMEAN",
            "--mtz",
            str(target),
            "--write_mtz_amplitudes",
            "--massage_intensities",
            "--mtz_root_label",
            "FOBS",
        ]
        mode = "mean_intensity_to_amplitude"
    else:
        raise ValueError(f"Cannot identify F/SIGF or I/SIGI columns in {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"reflection_file_converter failed for {source}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return target, mode


def _stream_process(
    command: list[str],
    cwd: Path,
    log_path: Path,
    extra_path: Path | None = None,
) -> tuple[int, str]:
    output = []
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-xtrapol8")
    if extra_path is not None:
        env["PATH"] = f"{extra_path}{os.pathsep}{env.get('PATH', '')}"
    with log_path.open("w") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output.append(line)
            log.write(line)
        return process.wait(), "".join(output)


def _extract_occupancy(log_text: str) -> str:
    matches = re.findall(r"Optimal occupancy of triggered state\s+([0-9]+(?:\.[0-9]+)?)", log_text)
    return matches[-1] if matches else ""


def _actual_output_dir(requested: Path) -> Path:
    if requested.exists() and any(requested.rglob("Xtrapol8_out.phil")):
        return requested
    siblings = sorted(
        requested.parent.glob(f"{requested.name}_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for sibling in siblings:
        if sibling.is_dir() and any(sibling.rglob("Xtrapol8_out.phil")):
            return sibling
    return requested


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, default=Path("initial"))
    parser.add_argument("--out", type=Path, default=Path("results/xtrapol8_real_refined_occ015_035_step002"))
    parser.add_argument("--inputs-out", type=Path, default=Path("results/xtrapol8_real_inputs"))
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/xtrapol8_real_refined_occ015_035_step002/summary.csv"),
    )
    parser.add_argument("--phenix-python", type=Path, default=DEFAULT_PHENIX_PYTHON)
    parser.add_argument("--xtrapol8", type=Path, default=DEFAULT_XTRAPOL8)
    parser.add_argument("--reflection-converter", type=Path, default=DEFAULT_REFLECTION_CONVERTER)
    parser.add_argument("--occupancies", default=DEFAULT_OCCUPANCIES)
    parser.add_argument("--map-types", default="qfextr")
    parser.add_argument(
        "--additional-files",
        default="",
        help="Comma-separated restraint/support files passed to Xtrapol8 input.additional_files.",
    )
    parser.add_argument("--refinement-cycles", type=int, default=1)
    parser.add_argument("--real-space-refinement-cycles", type=int, default=1)
    parser.add_argument("--no-refinement", action="store_true")
    parser.add_argument(
        "--conditions",
        default="",
        help="Comma-separated condition names to run. Empty means all discovered conditions.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for executable in (args.phenix_python, args.xtrapol8, args.reflection_converter):
        if not executable.exists():
            print(f"Required path not found: {executable}", file=sys.stderr)
            return 1

    conditions = _discover_conditions(args.initial)
    if args.conditions:
        requested_conditions = {condition.strip() for condition in args.conditions.split(",") if condition.strip()}
        conditions = [
            condition
            for condition in conditions
            if condition[0] in requested_conditions
        ]
        missing_conditions = requested_conditions - {condition[0] for condition in conditions}
        if missing_conditions:
            print(
                "Requested condition(s) not found: " + ", ".join(sorted(missing_conditions)),
                file=sys.stderr,
            )
            return 1
    if args.limit is not None:
        conditions = conditions[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition, model, reference_mtz, triggered_mtz in conditions:
        x8_outdir = args.out / condition
        log_dir = args.out / "_logs" / condition
        log_path = log_dir / "xtrapol8_real.log"
        prepared_dir = args.inputs_out / condition
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "condition": condition,
            "status": "pending",
            "returncode": "",
            "occupancy": "",
            "occupancies_tested": args.occupancies,
            "refinement": "False" if args.no_refinement else "True",
            "refinement_cycles": "" if args.no_refinement else args.refinement_cycles,
            "real_space_refinement_cycles": "" if args.no_refinement else args.real_space_refinement_cycles,
            "workdir": str(x8_outdir),
            "log": str(log_path),
            "reference_conversion": "",
            "triggered_conversion": "",
            "mtz_files": "",
            "ccp4_files": "",
            "error": "",
        }
        try:
            prepared_reference, reference_mode = _prepare_fobs_mtz(
                reference_mtz.resolve(),
                prepared_dir / "reference_fobs.mtz",
                args.reflection_converter,
                force=args.force,
            )
            prepared_triggered, triggered_mode = _prepare_fobs_mtz(
                triggered_mtz.resolve(),
                prepared_dir / "triggered_fobs.mtz",
                args.reflection_converter,
                force=args.force,
            )
        except Exception as exc:  # noqa: BLE001
            row["status"] = "prepare_error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(f"{condition}: prepare_error {row['error']}")
            continue

        row["reference_conversion"] = reference_mode
        row["triggered_conversion"] = triggered_mode

        if log_path.exists() and not args.force:
            row["status"] = "skipped_existing_log"
            rows.append(row)
            print(f"{condition}: skipped, log already exists")
            continue

        log_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.phenix_python.resolve()),
            str(args.xtrapol8.resolve()),
            f"input.reference_mtz={prepared_reference.resolve()}",
            f"input.triggered_mtz={prepared_triggered.resolve()}",
            f"input.reference_pdb={model.resolve()}",
            f"occupancies.list_occ={args.occupancies}",
            f"f_and_maps.f_extrapolated_and_maps={args.map_types}",
            "f_and_maps.negative_and_missing=keep_no_fill",
            f"refinement.run_refinement={'False' if args.no_refinement else 'True'}",
            f"refinement.phenix_keywords.main.cycles={args.refinement_cycles}",
            f"refinement.phenix_keywords.real_space_refine.cycles={args.real_space_refinement_cycles}",
            "output.open_coot=False",
            f"output.outdir={x8_outdir.resolve()}",
            f"output.outname={condition}",
        ]
        if args.additional_files:
            command.append(f"input.additional_files={args.additional_files}")
        print(f"\n===== Real Xtrapol8: {condition} =====")
        returncode, log_text = _stream_process(
            command,
            Path.cwd(),
            log_path,
            extra_path=args.phenix_python.resolve().parent,
        )
        actual_outdir = _actual_output_dir(x8_outdir)
        row["workdir"] = str(actual_outdir)
        row["returncode"] = returncode
        row["occupancy"] = _extract_occupancy(log_text)
        row["mtz_files"] = sum(1 for path in actual_outdir.rglob("*.mtz") if path.is_file())
        row["ccp4_files"] = sum(1 for path in actual_outdir.rglob("*.ccp4") if path.is_file())
        if returncode == 0 and row["occupancy"]:
            row["status"] = "ok"
        elif returncode == 0:
            row["status"] = "ok_no_occupancy"
        else:
            row["status"] = "error"
            traceback = log_text.strip().splitlines()[-20:]
            row["error"] = " | ".join(traceback)
        rows.append(row)
        print(
            f"{condition}: {row['status']} occupancy={row['occupancy'] or '-'} "
            f"mtz={row['mtz_files']} ccp4={row['ccp4_files']}"
        )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.summary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    failures = sum(row["status"] in {"error", "prepare_error"} for row in rows)
    print(f"Wrote {args.summary}; {len(rows) - failures} non-error, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
