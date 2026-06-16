import json
from pathlib import Path

import pytest

from tv_extrapolation.config import DatasetConfig
from tv_extrapolation.pipeline import run

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_5us_matches_baseline():
    baseline = json.loads((REPO_ROOT / "tests/baseline_5us_10ns.json").read_text())["5us"]
    config = DatasetConfig.from_yaml(REPO_ROOT / "datasets/5us.yaml")
    result = run(config)

    assert result.status == "ok"
    assert result.chi == pytest.approx(baseline["chi"], rel=1e-6)
    assert result.extrapolation_factor == pytest.approx(baseline["extrapolation_factor"], rel=1e-6)
