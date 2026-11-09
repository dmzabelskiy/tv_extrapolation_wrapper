"""Optional fixed extrapolation factors (``extra_extrapolation_factors``).

For crystals that are fully in one state (e.g. grown at their own pH), the useful extrapolated map is the one at
N = 1, whatever χ the estimator returns. The extra maps must come from the same code path as the main one:
E(N) = dark + N·ΔF, so E(1) − E(1/χ) = (1 − 1/χ)·ΔF.
"""

from pathlib import Path

import numpy as np
import pytest
import reciprocalspaceship as rs

from tv_extrapolation.config import DatasetConfig
from tv_extrapolation.pipeline import run

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
TRAPPING_2 = EXAMPLES / "configs" / "trapping_2.yaml"

pytestmark = pytest.mark.skipif(
    not (EXAMPLES / "init" / "trapping_2").is_dir(), reason="bundled trapping_2 example not present"
)


def _complex(path: Path) -> "pd.Series":
    ds = rs.read_mtz(str(path))
    f = [c for c in ds.columns if isinstance(ds[c].dtype, rs.StructureFactorAmplitudeDtype)][0]
    p = [c for c in ds.columns if isinstance(ds[c].dtype, rs.PhaseDtype)][0]
    return ds[f].to_numpy(float) * np.exp(1j * np.deg2rad(ds[p].to_numpy(float))), ds.index


def test_extra_factor_is_linear(tmp_path, monkeypatch):
    monkeypatch.chdir(EXAMPLES.parent)
    config = DatasetConfig.from_yaml(TRAPPING_2)
    config = config.model_copy(
        update={"output_dir": tmp_path / "trapping_2", "extra_extrapolation_factors": [1.0]}
    )
    result = run(config, fresh=True)
    assert result.status == "ok", result.error

    cond = Path(result.extrapolated_mtz).parent
    fixed = cond / "trapping_2_it_tv_extrapolated_fixed1.00.mtz"
    ready = cond / "phenix_ready" / "trapping_2_fixed1.00_it_tv_extrapolated_phenix_ready.mtz"
    assert fixed.is_file() and ready.is_file()
    assert {"FEXTRA", "FreeR_flag"} <= set(rs.read_mtz(str(ready)).columns)
    assert str(ready) in result.extra_phenix_ready_mtz

    e1, i1 = _complex(fixed)
    ef, i_f = _complex(Path(result.extrapolated_mtz))
    dF, i_d = _complex(Path(result.diffmap_mtz))
    common = i1.intersection(i_f).intersection(i_d)
    a = dict(zip(i1, e1)); b = dict(zip(i_f, ef)); d = dict(zip(i_d, dF))
    lhs = np.array([a[h] - b[h] for h in common])
    rhs = np.array([(1.0 - result.extrapolation_factor) * d[h] for h in common])
    rel = np.linalg.norm(lhs - rhs) / np.linalg.norm(np.array([d[h] for h in common]))
    assert len(common) > 1000
    assert rel < 1e-3, rel
