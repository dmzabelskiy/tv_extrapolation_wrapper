"""Unit tests for pipeline internals: scaling fallback, diffmap-cache sync,
and the scaling context managers. No crystallographic data required."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

import tv_extrapolation.pipeline as pipeline
from tv_extrapolation.config import DatasetConfig
from tv_extrapolation.pipeline import (
    EstimationResult,
    _huber_safe_scaling_ctx,
    _linear_scaling_ctx,
    _scaling_retry_reason,
    _sync_diffmap_cache,
    run,
)


def _config(tmp_path, **overrides) -> DatasetConfig:
    payload = {
        "name": "unit",
        "dark_mtz": "a.mtz",
        "triggered_mtz": "b.mtz",
        "pdb_dark": "c.pdb",
        "resolution_limit": 2.0,
        "columns": {
            "dark": {"kind": "amplitude", "amplitude_or_intensity": "F", "sigma": "SIGF"},
            "triggered": {"kind": "amplitude", "amplitude_or_intensity": "F", "sigma": "SIGF"},
        },
        "output_dir": str(tmp_path / "results" / "unit"),
    }
    payload.update(overrides)
    return DatasetConfig(**payload)


def _result(status, scaling_loss="huber_safe", error="") -> EstimationResult:
    return EstimationResult(
        condition="unit", status=status, chi=None, std=None,
        extrapolation_factor=None, dark_finite=0, dark_nonfinite=0,
        triggered_finite=0, triggered_nonfinite=0, diffmap_mtz="",
        extrapolated_mtz="", extrapolated_ccp4="",
        scaling_loss=scaling_loss, error=error,
    )


# --- _scaling_retry_reason -------------------------------------------------


def test_retry_reason_on_nan_and_huber_crash():
    assert _scaling_retry_reason(_result("nan")) is not None
    assert _scaling_retry_reason(
        _result("error", error="IndexError: boolean index did not match")
    ) is not None
    assert _scaling_retry_reason(_result("ok")) is None
    assert _scaling_retry_reason(
        _result("error", error="FileNotFoundError: a.mtz")
    ) is None


# --- run() fallback --------------------------------------------------------


def test_run_retries_with_linear_on_nan(tmp_path, monkeypatch):
    config = _config(tmp_path)
    calls = []

    def fake_run_once(cfg, fresh=False):
        calls.append(cfg.scaling_loss)
        if len(calls) == 1:
            return _result("nan", scaling_loss=cfg.scaling_loss)
        return _result("ok", scaling_loss=cfg.scaling_loss)

    monkeypatch.setattr(pipeline, "_run_once", fake_run_once)
    result = run(config)
    assert calls == ["huber_safe", "linear"]
    assert result.status == "ok"
    assert result.scaling_loss == "linear"


def test_run_does_not_retry_when_already_linear(tmp_path, monkeypatch):
    config = _config(tmp_path, scaling_loss="linear")
    calls = []

    def fake_run_once(cfg, fresh=False):
        calls.append(cfg.scaling_loss)
        return _result("nan", scaling_loss=cfg.scaling_loss)

    monkeypatch.setattr(pipeline, "_run_once", fake_run_once)
    result = run(config)
    assert calls == ["linear"]
    assert result.status == "nan"


def test_run_does_not_retry_on_unrelated_error(tmp_path, monkeypatch):
    config = _config(tmp_path)
    calls = []

    def fake_run_once(cfg, fresh=False):
        calls.append(cfg.scaling_loss)
        return _result("error", scaling_loss=cfg.scaling_loss,
                       error="FileNotFoundError: a.mtz")

    monkeypatch.setattr(pipeline, "_run_once", fake_run_once)
    result = run(config)
    assert calls == ["huber_safe"]
    assert result.status == "error"


# --- _sync_diffmap_cache ---------------------------------------------------


def _make_cache(condition_dir: Path) -> Path:
    condition_dir.mkdir(parents=True, exist_ok=True)
    cache = condition_dir / "diffmap_20.0_it_tv_010.mtz"
    cache.write_bytes(b"cache")
    old = time.time() - 10 * 24 * 3600
    os.utime(cache, (old, old))
    return cache


def test_cache_without_sidecar_is_adopted_and_touched(tmp_path):
    config = _config(tmp_path)
    condition_dir = config.condition_dir
    cache = _make_cache(condition_dir)

    _sync_diffmap_cache(condition_dir, config, fresh=False)

    assert cache.exists()
    assert time.time() - cache.stat().st_mtime < 60
    sidecar = json.loads((condition_dir / "diffmap_provenance.json").read_text())
    assert sidecar["scaling_loss"] == "huber_safe"


def test_cache_with_mismatched_sidecar_is_deleted(tmp_path):
    config = _config(tmp_path)
    condition_dir = config.condition_dir
    cache = _make_cache(condition_dir)
    _sync_diffmap_cache(condition_dir, config, fresh=False)

    changed = config.model_copy(update={"scaling_loss": "linear"})
    _sync_diffmap_cache(condition_dir, changed, fresh=False)

    assert not cache.exists()
    sidecar = json.loads((condition_dir / "diffmap_provenance.json").read_text())
    assert sidecar["scaling_loss"] == "linear"


def test_cache_fresh_flag_deletes_matching_cache(tmp_path):
    config = _config(tmp_path)
    condition_dir = config.condition_dir
    cache = _make_cache(condition_dir)
    _sync_diffmap_cache(condition_dir, config, fresh=False)
    assert cache.exists()

    _sync_diffmap_cache(condition_dir, config, fresh=True)
    assert not cache.exists()


# --- scaling context managers ----------------------------------------------


def test_huber_safe_ctx_pads_shrinking_residuals():
    import scipy.optimize as scipy_opt

    original = scipy_opt.least_squares
    calls = {"n": 0}

    def shrinking(x):
        calls["n"] += 1
        n = 5 if calls["n"] == 1 else 3
        return np.full(n, x[0] - 2.0)

    with _huber_safe_scaling_ctx():
        result = scipy_opt.least_squares(shrinking, [0.0], loss="huber")
    assert result.success
    assert scipy_opt.least_squares is original


def test_linear_ctx_swaps_and_restores_scale_maps():
    import xtr_estimator.processing as processing

    original = processing.scale_maps
    with _linear_scaling_ctx():
        assert processing.scale_maps is not original
    assert processing.scale_maps is original
