#!/usr/bin/env python3
"""Lightweight MTZ header inspection without crystallography dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import re


NCOL_RE = re.compile(rb"NCOL\s+(\d+)\s+(\d+)\s+(\d+)")
CELL_RE = re.compile(
    rb"CELL\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+"
    rb"([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)"
)
RESO_RE = re.compile(rb"RESO\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)")
COLUMN_RE = re.compile(rb"COLUMN\s+(\S+)\s+([A-Z])\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)")


@dataclass(frozen=True)
class MtzMetadata:
    path: Path
    columns: tuple[str, ...]
    column_types: dict[str, str]
    n_columns: int | None
    n_reflections: int | None
    d_min: float | None
    cell: tuple[float, float, float, float, float, float] | None


def _tail_header(path: Path, tail_bytes: int = 65536) -> bytes:
    data = path.read_bytes()
    return data[-tail_bytes:]


def read_mtz_metadata(path: str | Path) -> MtzMetadata:
    path = Path(path)
    header = _tail_header(path)

    ncol_match = NCOL_RE.search(header)
    cell_match = CELL_RE.search(header)
    reso_match = RESO_RE.search(header)
    column_matches = list(COLUMN_RE.finditer(header))

    n_columns = int(ncol_match.group(1)) if ncol_match else None
    n_reflections = int(ncol_match.group(2)) if ncol_match else None
    cell = (
        tuple(float(v) for v in cell_match.groups()) if cell_match else None
    )
    d_min = None
    if reso_match:
        high_res_inverse_square = float(reso_match.group(2))
        if high_res_inverse_square > 0:
            d_min = 1.0 / sqrt(high_res_inverse_square)

    column_types = {
        match.group(1).decode("latin1"): match.group(2).decode("ascii")
        for match in column_matches
    }
    return MtzMetadata(
        path=path,
        columns=tuple(column_types),
        column_types=column_types,
        n_columns=n_columns,
        n_reflections=n_reflections,
        d_min=d_min,
        cell=cell,
    )


def choose_column_mode(metadata: MtzMetadata) -> str:
    columns = set(metadata.columns)
    if {"F", "SIGF"}.issubset(columns):
        return "amplitude"
    if {"I", "SIGI"}.issubset(columns):
        return "intensity"
    raise ValueError(
        f"Could not infer supported column mode for {metadata.path}: "
        f"{', '.join(metadata.columns)}"
    )
