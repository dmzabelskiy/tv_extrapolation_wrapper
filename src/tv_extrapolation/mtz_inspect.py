from __future__ import annotations

import warnings
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


# Column pairs used for I/σ analysis, in preference order.  Intensities first:
# the I/σI >= 2 style criterion is defined on intensities.
_ISIGMA_INTENSITY_CANDIDATES = (("IMEAN", "SIGIMEAN"), ("I", "SIGI"))
_ISIGMA_AMPLITUDE_CANDIDATES = (("F", "SIGF"), ("FP", "SIGFP"))


def _select_isigma_columns(mtz: gemmi.Mtz, mtz_path: Path) -> tuple[str, str, bool]:
    """Pick (data, sigma, is_intensity) columns for an I/σ cutoff.

    Intensities are preferred over amplitudes.  Returns is_intensity=False when
    only amplitudes are available so the caller can double the threshold.
    """
    labels = {c.label for c in mtz.columns}
    for data, sigma in _ISIGMA_INTENSITY_CANDIDATES:
        if data in labels and sigma in labels:
            return data, sigma, True
    for data, sigma in _ISIGMA_AMPLITUDE_CANDIDATES:
        if data in labels and sigma in labels:
            return data, sigma, False
    raise ValueError(
        f"{mtz_path}: no (IMEAN/SIGIMEAN, I/SIGI, F/SIGF, FP/SIGFP) column pair "
        "for I/σ analysis."
    )


def _merge_sparse_shells(shells: list[dict], min_shell_n: int) -> list[dict]:
    """Merge shells with fewer than min_shell_n reflections into their
    lower-resolution neighbour so every emitted shell has reliable statistics.

    A sparse shell folds backward into the already-accepted lower-resolution
    entry.  A sparse leading shell (no lower-resolution neighbour) instead
    absorbs its higher-resolution neighbour.  Shells are ordered low->high
    resolution (decreasing dmin).
    """
    merged: list[dict] = []
    for sh in shells:
        if merged and (sh["n"] < min_shell_n or merged[-1]["n"] < min_shell_n):
            prev = merged[-1]
            prev["n"] += sh["n"]
            prev["sum"] += sh["sum"]
            prev["dmin"] = min(prev["dmin"], sh["dmin"])
        else:
            merged.append(dict(sh))
    return merged


def _isigma_shells(
    d: np.ndarray, isig: np.ndarray, n_shells: int, min_shell_n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bin reflections into equal-volume reciprocal shells (low→high res).

    Returns (dmin, mean): dmin[i] is the high-resolution edge (smallest d) of
    shell i and mean[i] is ⟨I/σ⟩ in that shell.  Shells with fewer than
    min_shell_n reflections are merged into their lower-resolution neighbour.
    """
    x = (1.0 / d) ** 3  # equal-volume coordinate; increases with resolution
    edges = np.linspace(x.min(), x.max(), n_shells + 1)
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, n_shells - 1)
    shells: list[dict] = []
    for s in range(n_shells):
        sel = idx == s
        if not sel.any():
            continue
        shells.append(
            {
                "n": int(sel.sum()),
                "sum": float(isig[sel].sum()),
                "dmin": float(d[sel].min()),
            }
        )
    merged = _merge_sparse_shells(shells, min_shell_n)
    dmin = np.array([m["dmin"] for m in merged])
    mean = np.array([m["sum"] / m["n"] for m in merged])
    return dmin, mean


def resolution_cutoff_by_isigma(
    mtz_path: Path,
    threshold: float = 1.0,
    n_shells: int = 20,
    min_shell_n: int = 50,
) -> float:
    """High-resolution cutoff (Å) where ⟨I/σ⟩ drops below `threshold`.

    Only reflections with finite intensity and σ > 0 are used, so NaN-padded HKL
    rows are ignored.  Intensities are preferred; when only amplitudes are
    present the threshold is doubled (I/σI ≈ 0.5 · F/σF).
    """
    mtz_path = Path(mtz_path)
    mtz = gemmi.read_mtz_file(str(mtz_path))
    data_col, sigma_col, is_intensity = _select_isigma_columns(mtz, mtz_path)
    labels = [c.label for c in mtz.columns]
    table = np.array(mtz, copy=False)
    d = mtz.make_d_array()
    values = table[:, labels.index(data_col)]
    sigmas = table[:, labels.index(sigma_col)]
    finite = np.isfinite(values) & np.isfinite(sigmas) & (sigmas > 0)
    d = d[finite]
    isig = values[finite] / sigmas[finite]
    if d.size == 0:
        raise ValueError(
            f"{mtz_path}: no finite {data_col}/{sigma_col} reflections."
        )
    effective_threshold = threshold if is_intensity else 2.0 * threshold
    dmin, mean = _isigma_shells(d, isig, n_shells, min_shell_n)
    last_pass: int | None = None
    for i in range(len(mean)):
        if mean[i] >= effective_threshold:
            last_pass = i
        elif i + 1 >= len(mean) or mean[i + 1] < effective_threshold:
            break
    if last_pass is None:
        warnings.warn(
            f"{mtz_path}: ⟨I/σ⟩ never reaches {effective_threshold}; "
            "returning coarsest measured resolution.",
            stacklevel=2,
        )
        return round(float(d.max()), 2)
    return round(float(dmin[last_pass]), 2)


def detect_resolution_limit(dark_mtz: Path, triggered_mtz: Path) -> float:
    """Return the coarser high-resolution limit (Å) across both MTZ files.

    Uses the minimum resolution present in each file (i.e. the largest minimum
    d-spacing), so the analysis is restricted to data that both datasets share.
    """
    dark = gemmi.read_mtz_file(str(dark_mtz))
    triggered = gemmi.read_mtz_file(str(triggered_mtz))
    return round(max(dark.resolution_high(), triggered.resolution_high()), 2)
