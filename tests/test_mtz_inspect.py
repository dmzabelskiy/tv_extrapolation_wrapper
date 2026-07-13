"""Tests for MTZ column/resolution auto-detection.

These tests use real MTZ files from initial/; they are skipped when the data
is not present (CI / fresh clone).
"""
from __future__ import annotations

import math
from pathlib import Path

import gemmi
import numpy as np
import pytest

from tv_extrapolation.config import DatasetConfig
from tv_extrapolation.mtz_inspect import (
    detect_column_spec,
    detect_resolution_limit,
    resolution_cutoff_by_isigma,
)

# ---------------------------------------------------------------------------
# Paths to test data
# ---------------------------------------------------------------------------

XFEL_DARK = Path("initial/olpvr1/5us/ground.mtz")
XFEL_TRIGGERED = Path("initial/olpvr1/5us/5us.mtz")
XFEL_PDB = Path("initial/olpvr1/5us/olpvr1_xfel_dark_refine_007.pdb")

ESRF_DARK = Path("initial/olpvr1_esrf/esrf_5ms/ground.mtz")
ESRF_TRIGGERED = Path("initial/olpvr1_esrf/esrf_5ms/5ms_0-37p5ms.mtz")

have_xfel = pytest.mark.skipif(
    not (XFEL_DARK.exists() and XFEL_TRIGGERED.exists()),
    reason="XFEL test data not available",
)
have_esrf = pytest.mark.skipif(
    not (ESRF_DARK.exists() and ESRF_TRIGGERED.exists()),
    reason="ESRF test data not available",
)
have_xfel_pdb = pytest.mark.skipif(
    not (XFEL_DARK.exists() and XFEL_TRIGGERED.exists() and XFEL_PDB.exists()),
    reason="XFEL test data not available",
)


# ---------------------------------------------------------------------------
# resolution_cutoff_by_isigma (synthetic MTZ)
# ---------------------------------------------------------------------------


def _write_synthetic_mtz(
    path,
    *,
    finite_dmin,
    drop_below=None,
    drop_value=0.2,
    pad_dmin=None,
    high=10.0,
    cell_a=50.0,
    hkl_max=25,
    amplitude=False,
):
    """Write a cubic P1 MTZ with a controllable ⟨I/σ⟩ vs resolution profile.

    - Reflections with d >= finite_dmin get value `high` (σ = 1 → I/σ = high).
    - If `drop_below` is set, reflections with finite_dmin <= d < drop_below get
      `drop_value` instead (weak shell).
    - If `pad_dmin` is set, reflections with pad_dmin <= d < finite_dmin get a
      NaN value (unmeasured / padded HKL rows).
    """
    cell = gemmi.UnitCell(cell_a, cell_a, cell_a, 90, 90, 90)
    rows = []
    for h in range(-hkl_max, hkl_max + 1):
        for k in range(-hkl_max, hkl_max + 1):
            for l in range(0, hkl_max + 1):
                s = h * h + k * k + l * l
                if s == 0:
                    continue
                d = cell_a / math.sqrt(s)
                if d > 60.0:
                    continue
                if d >= finite_dmin:
                    value = high
                    if drop_below is not None and d < drop_below:
                        value = drop_value
                elif pad_dmin is not None and d >= pad_dmin:
                    value = float("nan")
                else:
                    continue
                rows.append((h, k, l, value, 1.0))
    arr = np.array(rows, dtype=float)
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.SpaceGroup("P 1")
    mtz.set_cell_for_all(cell)
    mtz.add_dataset("syn")
    if amplitude:
        mtz.add_column("F", "F")
        mtz.add_column("SIGF", "Q")
    else:
        mtz.add_column("IMEAN", "J")
        mtz.add_column("SIGIMEAN", "Q")
    mtz.set_data(arr)
    mtz.write_to_file(str(path))
    return path


def test_isigma_step_cutoff(tmp_path):
    """Strong data above 2.5 Å, weak below → cutoff near 2.5 Å."""
    p = _write_synthetic_mtz(tmp_path / "step.mtz", finite_dmin=2.0, drop_below=2.5)
    cutoff = resolution_cutoff_by_isigma(p, threshold=1.0)
    assert 2.2 < cutoff < 2.8


# ---------------------------------------------------------------------------
# detect_column_spec
# ---------------------------------------------------------------------------


@have_xfel
def test_detect_column_spec_xfel_dark():
    spec = detect_column_spec(XFEL_DARK)
    assert spec.kind == "intensity"
    assert spec.amplitude_or_intensity == "I"
    assert spec.sigma == "SIGI"


@have_xfel
def test_detect_column_spec_xfel_triggered():
    spec = detect_column_spec(XFEL_TRIGGERED)
    assert spec.kind == "intensity"
    assert spec.amplitude_or_intensity == "I"
    assert spec.sigma == "SIGI"


@have_esrf
def test_detect_column_spec_esrf_prefers_amplitude():
    # ESRF ground.mtz has both IMEAN/SIGIMEAN and F/SIGF; amplitude must win.
    spec = detect_column_spec(ESRF_DARK)
    assert spec.kind == "amplitude"
    assert spec.amplitude_or_intensity == "F"
    assert spec.sigma == "SIGF"


@have_esrf
def test_detect_column_spec_esrf_triggered():
    spec = detect_column_spec(ESRF_TRIGGERED)
    assert spec.kind == "amplitude"
    assert spec.amplitude_or_intensity == "F"


# ---------------------------------------------------------------------------
# detect_resolution_limit
# ---------------------------------------------------------------------------


@have_xfel
def test_detect_resolution_limit_xfel():
    limit = detect_resolution_limit(XFEL_DARK, XFEL_TRIGGERED)
    # dark goes to ~1.62 Å, triggered to ~1.80 Å → limit should be ~1.80
    assert 1.5 < limit < 3.0
    assert limit == round(limit, 2)


@have_xfel
def test_detect_resolution_limit_is_coarser():
    """The limit must be >= both individual resolutions (coarser = larger d).

    We compare within 0.01 Å to tolerate the 2-decimal rounding applied to the
    returned value.
    """
    import gemmi
    dark = gemmi.read_mtz_file(str(XFEL_DARK))
    triggered = gemmi.read_mtz_file(str(XFEL_TRIGGERED))
    limit = detect_resolution_limit(XFEL_DARK, XFEL_TRIGGERED)
    assert limit >= dark.resolution_high() - 0.01
    assert limit >= triggered.resolution_high() - 0.01


# ---------------------------------------------------------------------------
# DatasetConfig.from_files
# ---------------------------------------------------------------------------


@have_xfel_pdb
def test_from_files_xfel_defaults():
    config = DatasetConfig.from_files(XFEL_DARK, XFEL_TRIGGERED, XFEL_PDB)
    assert config.name == XFEL_TRIGGERED.stem
    assert config.columns["dark"].kind == "intensity"
    assert config.columns["triggered"].kind == "intensity"
    assert 1.0 < config.resolution_limit < 5.0
    assert config.output_dir == Path("results")
    assert config.scaling_loss == "huber"
    assert not config.finite_filter
    assert not config.rewrite_pdb_cell
    assert not config.phenix_refine_cell


@have_xfel_pdb
def test_from_files_overrides():
    config = DatasetConfig.from_files(
        XFEL_DARK,
        XFEL_TRIGGERED,
        XFEL_PDB,
        name="custom",
        resolution_limit=2.5,
        output_dir=Path("/tmp/out"),
        scaling_loss="linear",
        finite_filter=True,
        rewrite_pdb_cell=True,
    )
    assert config.name == "custom"
    assert config.resolution_limit == 2.5
    assert config.output_dir == Path("/tmp/out")
    assert config.scaling_loss == "linear"
    assert config.finite_filter
    assert config.rewrite_pdb_cell


def test_isigma_ignores_nan_padding(tmp_path):
    """Strong data to 2.5 Å, NaN-padded rows to 1.2 Å → cutoff ≈ 2.5, not ~1.2."""
    p = _write_synthetic_mtz(
        tmp_path / "padded.mtz", finite_dmin=2.5, pad_dmin=1.2
    )
    # sanity: the file's reflection-list extent really is ~1.2 Å
    assert gemmi.read_mtz_file(str(p)).resolution_high() < 1.4
    cutoff = resolution_cutoff_by_isigma(p, threshold=1.0)
    assert cutoff > 2.0
    assert cutoff < 2.9


def test_isigma_strong_data_returns_finest(tmp_path):
    """Strong I/σ in every shell → finest measured d_min (~2.0 Å)."""
    p = _write_synthetic_mtz(tmp_path / "strong.mtz", finite_dmin=2.0)
    cutoff = resolution_cutoff_by_isigma(p, threshold=1.0)
    assert 1.9 < cutoff < 2.2


def test_isigma_all_noise_warns(tmp_path):
    """No shell reaches threshold → warns and returns a coarse limit."""
    p = _write_synthetic_mtz(
        tmp_path / "noise.mtz", finite_dmin=2.0, drop_below=100.0, drop_value=0.2
    )
    with pytest.warns(UserWarning):
        cutoff = resolution_cutoff_by_isigma(p, threshold=1.0)
    assert cutoff > 5.0  # coarsest measured resolution


def test_isigma_amplitude_fallback(tmp_path):
    """Amplitude-only MTZ uses a 2× effective threshold.

    F/σF = 10 (>2) passes above 2.5 Å; F/σF = 1.5 (<2) fails below → cut ≈ 2.5.
    """
    p = _write_synthetic_mtz(
        tmp_path / "amp.mtz",
        finite_dmin=2.0,
        drop_below=2.5,
        drop_value=1.5,
        amplitude=True,
    )
    cutoff = resolution_cutoff_by_isigma(p, threshold=1.0)
    assert 2.2 < cutoff < 2.8
