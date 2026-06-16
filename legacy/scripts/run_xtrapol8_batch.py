#!/usr/bin/env python3
"""Run Phenix Xtrapol8 for every dataset in the initial data bundle."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


DEFAULT_PHENIX_PYTHON = Path("/home/dmitrii/phenix-2.0-5936/phenix_bin/phenix.python")
DEFAULT_REFLECTION_CONVERTER = Path(
    "/home/dmitrii/phenix-2.0-5936/phenix_bin/phenix.reflection_file_converter"
)
DEFAULT_XTRAPOL8_LAUNCHER = Path(__file__).with_name("phenix_xtrapol8_label_patch.py")


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
    import re

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


def _stream_process(command: list[str], cwd: Path, log_path: Path) -> int:
    with log_path.open("w") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, default=Path("initial"))
    parser.add_argument("--out", type=Path, default=Path("results/xtrapol8"))
    parser.add_argument("--inputs-out", type=Path, default=Path("results/xtrapol8_inputs"))
    parser.add_argument("--summary", type=Path, default=Path("results/xtrapol8/summary.csv"))
    parser.add_argument("--phenix-python", type=Path, default=DEFAULT_PHENIX_PYTHON)
    parser.add_argument("--xtrapol8-launcher", type=Path, default=DEFAULT_XTRAPOL8_LAUNCHER)
    parser.add_argument("--reflection-converter", type=Path, default=DEFAULT_REFLECTION_CONVERTER)
    parser.add_argument("--n-alpha", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when a prior per-condition log exists.",
    )
    args = parser.parse_args()

    if not args.phenix_python.exists():
        print(f"Phenix Python executable not found: {args.phenix_python}", file=sys.stderr)
        return 1
    if not args.xtrapol8_launcher.exists():
        print(f"Patched Xtrapol8 launcher not found: {args.xtrapol8_launcher}", file=sys.stderr)
        return 1
    if not args.reflection_converter.exists():
        print(
            f"Phenix reflection converter not found: {args.reflection_converter}",
            file=sys.stderr,
        )
        return 1

    conditions = _discover_conditions(args.initial)
    if args.limit is not None:
        conditions = conditions[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition, model, reference_mtz, triggered_mtz in conditions:
        workdir = args.out / condition
        workdir.mkdir(parents=True, exist_ok=True)
        log_path = workdir / "xtrapol8.log"
        prepared_dir = args.inputs_out / condition
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "condition": condition,
            "status": "pending",
            "returncode": "",
            "workdir": str(workdir),
            "log": str(log_path),
            "model": str(model),
            "reference_mtz": str(reference_mtz),
            "triggered_mtz": str(triggered_mtz),
            "prepared_reference_mtz": "",
            "prepared_triggered_mtz": "",
            "reference_conversion": "",
            "triggered_conversion": "",
            "n_alpha": args.n_alpha,
            "output_file_count": "",
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

        row["prepared_reference_mtz"] = str(prepared_reference)
        row["prepared_triggered_mtz"] = str(prepared_triggered)
        row["reference_conversion"] = reference_mode
        row["triggered_conversion"] = triggered_mode

        command = [
            str(args.phenix_python),
            str(args.xtrapol8_launcher.resolve()),
            str(model.resolve()),
            str(prepared_reference.resolve()),
            str(prepared_triggered.resolve()),
            "--overwrite",
            f"output.prefix={condition}",
            f"n_alpha={args.n_alpha}",
        ]
        if args.dry_run:
            command.append("--dry-run")

        if log_path.exists() and not args.force:
            row["status"] = "skipped_existing_log"
            rows.append(row)
            print(f"{condition}: skipped, log already exists")
            continue

        print(f"\n===== Xtrapol8: {condition} =====")
        returncode = _stream_process(command, workdir, log_path)
        row["returncode"] = returncode
        outputs = [
            path for path in workdir.iterdir()
            if path.is_file() and path.name != log_path.name
        ]
        row["output_file_count"] = len(outputs)
        if returncode == 0 and outputs:
            row["status"] = "ok"
        elif returncode == 0:
            row["status"] = "ok_no_outputs"
        else:
            row["status"] = "error"
        rows.append(row)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    failures = sum(row["status"] == "error" for row in rows)
    print(f"Wrote {args.summary}; {len(rows) - failures} non-error, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
