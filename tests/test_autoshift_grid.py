"""Regression test: xtr_estimator's autoshift must not crash when the observed and the
model-calculated dark maps get different FFT grids.

meteor grids each map with gemmi ``transform_f_phi_to_map(sample_rate=...)``, which sizes the
grid from that map's own resolution. The observed dark data usually stop slightly short of the
high-resolution limit, while the model map is calculated to exactly that limit, so the two grids
can differ by one FFT size (e.g. 432 vs 450). ``autoshift_rsmap_old`` then indexed one grid with
the other's mask (IndexError). The bundled ``trapping_1`` example hits this on
meteor-maps 0.5.1 / xtr_estimator 403b16bc.
"""

from pathlib import Path

import pytest

from tv_extrapolation.config import DatasetConfig
from tv_extrapolation.pipeline import run

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
TRAPPING_1 = EXAMPLES / "configs" / "trapping_1.yaml"

pytestmark = pytest.mark.skipif(
    not (EXAMPLES / "init" / "trapping_1").is_dir(), reason="bundled trapping_1 example not present"
)


def test_trapping_1_runs_without_grid_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(EXAMPLES.parent)
    config = DatasetConfig.from_yaml(TRAPPING_1)
    config = config.model_copy(update={"output_dir": tmp_path / "trapping_1"})
    result = run(config, fresh=True)
    assert result.status == "ok", result.error
    assert result.chi is not None and 0.0 < result.chi < 1.0
