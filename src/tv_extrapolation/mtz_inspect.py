from __future__ import annotations

from pathlib import Path

import gemmi

from .config import ColumnSpec

# MTZ column type letters relevant to us
_INTENSITY_TYPE = "J"
_AMPLITUDE_TYPE = "F"
_SIGMA_TYPE = "Q"

# Column name prefixes that identify model/map coefficients rather than
# experimental amplitudes in a dimple/refmac output MTZ.
_CALCULATED_AMPLITUDE_PREFIXES = ("FC", "FWT", "DELFWT")


def _columns_by_type(mtz_path: Path) -> dict[str, list[str]]:
    mtz = gemmi.read_mtz_file(str(mtz_path))
    result: dict[str, list[str]] = {}
    for col in mtz.columns:
        result.setdefault(col.type, []).append(col.label)
    return result


def _filter_experimental_amplitudes(amplitude_cols: list[str]) -> list[str]:
    """Remove calculated/map-coefficient columns; keep experimental amplitudes.

    Dimple/refmac output MTZs contain both experimental (F, FP) and model
    columns (FC, FWT, DELFWT, FC_ALL, …).  We discard any column whose name
    starts with a known calculated prefix so the detector settles on just the
    experimental amplitude.
    """
    experimental = [
        c for c in amplitude_cols
        if not any(c.upper().startswith(p) for p in _CALCULATED_AMPLITUDE_PREFIXES)
    ]
    # Fall back to full list if everything was filtered out (shouldn't happen)
    return experimental if experimental else amplitude_cols


def detect_column_spec(mtz_path: Path) -> ColumnSpec:
    """Auto-detect data and sigma columns from an MTZ file.

    When both intensity (J) and amplitude (F) columns are present, amplitude
    is preferred — that is the common case for ESRF/synchrotron data that has
    been truncated to structure-factor amplitudes.

    Raises ValueError with a descriptive message if the choice is ambiguous.
    """
    by_type = _columns_by_type(mtz_path)
    intensity_cols = by_type.get(_INTENSITY_TYPE, [])
    amplitude_cols = by_type.get(_AMPLITUDE_TYPE, [])
    sigma_cols = by_type.get(_SIGMA_TYPE, [])

    if amplitude_cols:
        kind = "amplitude"
        data_cols = _filter_experimental_amplitudes(amplitude_cols)
    elif intensity_cols:
        kind = "intensity"
        data_cols = intensity_cols
    else:
        raise ValueError(
            f"{mtz_path}: no intensity (J) or amplitude (F) columns found. "
            f"Available column types: {sorted(by_type)}"
        )

    if len(data_cols) != 1:
        raise ValueError(
            f"{mtz_path}: cannot unambiguously pick a single {kind} column from {data_cols}. "
            "Specify columns explicitly via a dataset YAML."
        )
    data_col = data_cols[0]

    if not sigma_cols:
        raise ValueError(f"{mtz_path}: no sigma (Q) column found.")

    sigma_col = _match_sigma(data_col, sigma_cols, mtz_path)
    return ColumnSpec(kind=kind, amplitude_or_intensity=data_col, sigma=sigma_col)


def _match_sigma(data_col: str, sigma_cols: list[str], mtz_path: Path) -> str:
    """Return the sigma column whose name is SIG + data_col (case-insensitive)."""
    if len(sigma_cols) == 1:
        return sigma_cols[0]

    target = f"SIG{data_col.upper()}"
    matches = [s for s in sigma_cols if s.upper() == target]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"{mtz_path}: multiple sigma columns {sigma_cols} and none matches "
        f"'SIG{data_col}'. Specify columns explicitly via a dataset YAML."
    )


def detect_resolution_limit(dark_mtz: Path, triggered_mtz: Path) -> float:
    """Return the coarser high-resolution limit (Å) across both MTZ files.

    Uses the minimum resolution present in each file (i.e. the largest minimum
    d-spacing), so the analysis is restricted to data that both datasets share.
    """
    dark = gemmi.read_mtz_file(str(dark_mtz))
    triggered = gemmi.read_mtz_file(str(triggered_mtz))
    return round(max(dark.resolution_high(), triggered.resolution_high()), 2)
