"""Tests for MTZ column/resolution auto-detection.

These tests use real MTZ files from initial/; they are skipped when the data
is not present (CI / fresh clone).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tv_extrapolation.config import DatasetConfig
from tv_extrapolation.mtz_inspect import detect_column_spec, detect_resolution_limit

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
