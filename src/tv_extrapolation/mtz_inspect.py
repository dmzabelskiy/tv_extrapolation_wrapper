from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np

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
    if not experimental:
        raise ValueError(
            f"All F-type columns {amplitude_cols} look calculated "
            f"(prefixes {_CALCULATED_AMPLITUDE_PREFIXES}). "
            "Specify columns explicitly via a dataset YAML."
        )
    return experimental


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


# Measured (data, sigma) column pairs, in preference order.  Amplitudes (Fobs)
# first — the resolution limit is the extent of measured structure-factor
# amplitudes — then intensities for datasets that ship only merged intensities.
# Only known experimental labels are matched, so calculated/map-coefficient
# columns (FC, FWT, F-model, 2FOFCWT, …) and free-R flags are never selected.
_MEASURED_AMPLITUDE_CANDIDATES = (
    ("F", "SIGF"),
    ("FP", "SIGFP"),
    ("F-obs", "SIGF-obs"),
    ("FOBS", "SIGFOBS"),
)
_MEASURED_INTENSITY_CANDIDATES = (("IMEAN", "SIGIMEAN"), ("I", "SIGI"))


def _select_measured_columns(mtz: gemmi.Mtz, mtz_path: Path) -> tuple[str, str]:
    """Pick the (data, sigma) columns of the measured experimental data."""
    labels = {c.label for c in mtz.columns}
    for data, sigma in _MEASURED_AMPLITUDE_CANDIDATES + _MEASURED_INTENSITY_CANDIDATES:
        if data in labels and sigma in labels:
            return data, sigma
    raise ValueError(
        f"{mtz_path}: no measured data column pair "
        "(F/SIGF, FP/SIGFP, F-obs/SIGF-obs, IMEAN/SIGIMEAN, I/SIGI) found."
    )


def resolution_cutoff_by_measured_extent(mtz_path: Path) -> float:
    """Highest-resolution limit (Å) with measured data in `mtz_path`.

    Returns the smallest d-spacing among reflections that carry a finite
    measured amplitude (or intensity) with σ > 0.  Free-R-flag rows and other
    padded reflections that lack a measured value are excluded, so a complete
    free-R set generated to the detector edge does not inflate the resolution.
    """
    mtz_path = Path(mtz_path)
    mtz = gemmi.read_mtz_file(str(mtz_path))
    data_col, sigma_col = _select_measured_columns(mtz, mtz_path)
    labels = [c.label for c in mtz.columns]
    table = np.array(mtz, copy=False)
    d = mtz.make_d_array()
    values = table[:, labels.index(data_col)]
    sigmas = table[:, labels.index(sigma_col)]
    measured = np.isfinite(values) & np.isfinite(sigmas) & (sigmas > 0)
    if not measured.any():
        raise ValueError(
            f"{mtz_path}: no finite {data_col}/{sigma_col} reflections."
        )
    return round(float(d[measured].min()), 2)


def detect_resolution_limit(dark_mtz: Path, triggered_mtz: Path) -> float:
    """Return the coarser high-resolution limit (Å) across both MTZ files.

    Each file's limit is the extent of its measured data (see
    resolution_cutoff_by_measured_extent).  The coarser of the two — the larger
    d — is returned so the analysis is restricted to data both datasets support.
    """
    return round(
        max(
            resolution_cutoff_by_measured_extent(Path(dark_mtz)),
            resolution_cutoff_by_measured_extent(Path(triggered_mtz)),
        ),
        2,
    )
