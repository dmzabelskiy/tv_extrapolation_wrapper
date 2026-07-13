"""Tests for MTZ column/resolution auto-detection.

Some tests use real MTZ files under data/; they are skipped when the data is
not present (CI / fresh clone).
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
    resolution_cutoff_by_measured_extent,
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
# resolution_cutoff_by_measured_extent (synthetic MTZ)
# ---------------------------------------------------------------------------


def _write_synthetic_mtz(
    path,
    *,
    finite_dmin,
    pad_dmin=None,
    cell_a=50.0,
    hkl_max=25,
    amplitude=False,
):
    """Write a cubic P1 MTZ with a controllable measured-data extent.

    - Reflections with d >= finite_dmin carry a measured value (10.0, σ = 1).
    - If `pad_dmin` is set, reflections with pad_dmin <= d < finite_dmin carry a
      NaN value (unmeasured rows — mimicking a free-R set padded past the data).
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
                    value = 10.0
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


def test_extent_ignores_nan_padded_rows(tmp_path):
    """Measured data to 2.5 Å but the HKL list is padded to 1.2 Å (free-R set).

    The extent must follow the measured data, not the padded rows.
    """
    p = _write_synthetic_mtz(tmp_path / "padded.mtz", finite_dmin=2.5, pad_dmin=1.2)
    # sanity: the reflection-list extent really is ~1.2 Å
    assert gemmi.read_mtz_file(str(p)).resolution_high() < 1.4
    assert resolution_cutoff_by_measured_extent(p) == 2.5


def test_extent_amplitude_file(tmp_path):
    """Amplitude-only (F/SIGF) MTZ: extent is the finest measured F."""
    p = _write_synthetic_mtz(tmp_path / "amp.mtz", finite_dmin=2.0, amplitude=True)
    assert resolution_cutoff_by_measured_extent(p) == 2.0


def test_extent_requires_measured_columns(tmp_path):
    """A file with only a free-R flag column has no measured data → ValueError."""
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.SpaceGroup("P 1")
    mtz.set_cell_for_all(gemmi.UnitCell(50, 50, 50, 90, 90, 90))
    mtz.add_dataset("syn")
    mtz.add_column("FREE", "I")
    mtz.set_data(np.array([[0, 0, 1, 1.0], [0, 1, 0, 0.0]], dtype=float))
    p = tmp_path / "onlyfree.mtz"
    mtz.write_to_file(str(p))
    with pytest.raises(ValueError):
        resolution_cutoff_by_measured_extent(p)


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


# ---------------------------------------------------------------------------
# Real-data regression: measured extent vs. free-R-padded MTZ extent
# ---------------------------------------------------------------------------

# Real data present in this checkout under data/ (not initial/).
DATA_9_39_DARK = Path("data/olpvr1_esrf/9-39ms/ground.mtz")
DATA_9_39_TRIG = Path("data/olpvr1_esrf/9-39ms/9-39.mtz")
DATA_TEST_GROUND = Path("data/test/ground.mtz")

have_9_39 = pytest.mark.skipif(
    not (DATA_9_39_DARK.exists() and DATA_9_39_TRIG.exists()),
    reason="9-39ms data not available",
)
have_test_ground = pytest.mark.skipif(
    not DATA_TEST_GROUND.exists(), reason="test ground.mtz not available"
)


@have_9_39
def test_extent_real_triggered_not_padded():
    """9-39.mtz HKL list extends to 1.13 Å (free-R set); Fobs to ~2.36 Å."""
    ext = resolution_cutoff_by_measured_extent(DATA_9_39_TRIG)
    assert ext > 1.5  # not the 1.13 Å padded extent
    assert 2.25 < ext < 2.45


@have_9_39
def test_detect_resolution_limit_extent_coarser():
    """detect_resolution_limit returns the coarser of dark(~1.70)/triggered(~2.36)."""
    limit = detect_resolution_limit(DATA_9_39_DARK, DATA_9_39_TRIG)
    assert limit > 1.5  # regression: old code returned 1.13
    assert 2.25 < limit < 2.45
    dark = resolution_cutoff_by_measured_extent(DATA_9_39_DARK)
    trig = resolution_cutoff_by_measured_extent(DATA_9_39_TRIG)
    assert limit == round(max(dark, trig), 2)


@have_test_ground
def test_extent_test_ground_not_padded():
    """data/test/ground.mtz padded to 1.13 Å (free-R); Fobs to ~1.70 Å."""
    ext = resolution_cutoff_by_measured_extent(DATA_TEST_GROUND)
    assert ext > 1.5  # not 1.13
    assert 1.6 < ext < 1.8


@have_9_39
def test_config_resolves_resolution_via_extent():
    """A config with no resolution_limit auto-detects via measured extent."""
    dark_spec = detect_column_spec(DATA_9_39_DARK)
    trig_spec = detect_column_spec(DATA_9_39_TRIG)
    config = DatasetConfig(
        name="9-39-auto",
        dark_mtz=DATA_9_39_DARK,
        triggered_mtz=DATA_9_39_TRIG,
        pdb_dark=DATA_9_39_DARK,  # any path; not read during resolution resolve
        columns={"dark": dark_spec, "triggered": trig_spec},
        output_dir=Path("results"),
    )
    assert config.resolution_limit is not None
    assert config.resolution_limit > 1.5  # not 1.13
    assert 2.25 < config.resolution_limit < 2.45


@have_9_39
def test_config_explicit_resolution_wins():
    """An explicit resolution_limit is never overridden by the detector."""
    dark_spec = detect_column_spec(DATA_9_39_DARK)
    trig_spec = detect_column_spec(DATA_9_39_TRIG)
    config = DatasetConfig(
        name="9-39-explicit",
        dark_mtz=DATA_9_39_DARK,
        triggered_mtz=DATA_9_39_TRIG,
        pdb_dark=DATA_9_39_DARK,
        resolution_limit=2.58,
        columns={"dark": dark_spec, "triggered": trig_spec},
        output_dir=Path("results"),
    )
    assert config.resolution_limit == 2.58
