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


def _measured_mask_and_d(mtz_path: Path) -> tuple[np.ndarray, np.ndarray, gemmi.Mtz]:
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
    return measured, d, mtz


def resolution_cutoff_by_measured_extent(
    mtz_path: Path, *, min_completeness: float = 0.9, n_shells: int = 40
) -> float:
    """Highest-resolution limit (Å) with well-measured data in `mtz_path`.

    Base rule: the smallest d-spacing among reflections carrying a finite
    measured amplitude (or intensity) with σ > 0, so free-R rows padded to the
    detector edge never inflate the limit.

    Junk guard: a legitimate footprint decays monotonically to its extent —
    smooth or sparse anisotropic tails keep the extent behavior.  When shell
    completeness collapses to ~0 and then *resurrects* deeper (garbage values
    written past the real data, e.g. by a broken truncate step), the limit is
    pulled back to the last shell with completeness >= `min_completeness`
    before the collapse, and a warning names the discarded reflections.
    """
    mtz_path = Path(mtz_path)
    measured, d, _ = _measured_mask_and_d(mtz_path)
    extent = float(d[measured].min())

    if measured.all():
        return round(extent, 2)

    # Shells of equal reciprocal volume over the region above the raw extent.
    # Rows below the extent (free-R padding past the data) are never measured
    # and must not count against shell completeness.
    region = d >= extent - 1e-9
    d = d[region]
    measured = measured[region]
    s3 = 1.0 / d**3
    edges = np.linspace(s3.min(), s3.max(), n_shells + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    shell_of = np.digitize(s3, edges) - 1

    shells = []
    for shell in range(n_shells):
        sel = shell_of == shell
        if sel.any():
            shells.append((shell, float(measured[sel].mean())))

    collapse_floor = 0.05
    collapsed_at = None
    resurrected = False
    for shell, completeness in shells:
        if collapsed_at is None:
            if completeness < collapse_floor:
                collapsed_at = shell
        elif completeness >= 2 * collapse_floor:
            resurrected = True
            break
    if not resurrected:
        return round(extent, 2)

    first_bad = next(
        (s for s, c in shells if s < collapsed_at and c < min_completeness),
        collapsed_at,
    )
    passing = measured & (shell_of < first_bad)
    if not passing.any():
        passing = measured & (shell_of < collapsed_at)
    if not passing.any():
        raise ValueError(
            f"{mtz_path}: measured data collapses and resurrects with no "
            "trustworthy region before the collapse."
        )
    cutoff = float(d[passing].min())
    beyond = measured & (shell_of >= first_bad)
    if beyond.any():
        import warnings
        warnings.warn(
            f"{mtz_path}: {int(beyond.sum())} measured reflections beyond "
            f"{cutoff:.2f} Å ignored (completeness collapses and resurrects — "
            "corrupt high-resolution values)",
            stacklevel=2,
        )
    return round(cutoff, 2)


def truncate_mtz(
    in_path: Path,
    out_path: Path,
    *,
    dmin: float | None = None,
    dmax: float | None = None,
    drop_unmeasured: bool = True,
) -> dict:
    """Write a copy of `in_path` keeping only reflections inside [dmin, dmax].

    With `drop_unmeasured` (default), rows whose measured data/sigma pair is
    non-finite or has σ ≤ 0 are removed as well — this strips free-R padding
    and corrupt placeholder values.  All columns are carried over unchanged.
    Returns a stats dict (kept, total, dmin, dmax of the output).
    """
    in_path, out_path = Path(in_path), Path(out_path)
    measured, d, mtz = _measured_mask_and_d(in_path)
    table = np.array(mtz, copy=True)

    keep = np.ones(len(table), dtype=bool)
    if drop_unmeasured:
        keep &= measured
    if dmin is not None:
        keep &= d >= dmin
    if dmax is not None:
        keep &= d <= dmax
    if not keep.any():
        raise ValueError(f"{in_path}: no reflections left after truncation.")

    out = gemmi.Mtz(with_base=True)
    out.spacegroup = mtz.spacegroup
    out.cell = mtz.cell
    out.title = f"{in_path.name} truncated (dmin={dmin}, dmax={dmax})"
    out.add_dataset("truncated")
    for col in mtz.columns:
        if col.label not in ("H", "K", "L"):
            out.add_column(col.label, col.type)
    out.set_data(table[keep])
    out.write_to_file(str(out_path))
    return {
        "kept": int(keep.sum()),
        "total": int(len(table)),
        "dmin": round(float(d[keep].min()), 3),
        "dmax": round(float(d[keep].max()), 1),
    }


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
