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


# --- phenix-ready R-free flags ----------------------------------------------


def _write_mtz(path, hkls, columns):
    import gemmi
    import reciprocalspaceship as rs

    ds = rs.DataSet(
        {"H": [h for h, _, _ in hkls], "K": [k for _, k, _ in hkls], "L": [l for _, _, l in hkls]},
        cell=gemmi.UnitCell(83.4, 83.4, 90.2, 90, 90, 120),
        spacegroup=gemmi.SpaceGroup("P 32 2 1"),
    ).astype({"H": "HKL", "K": "HKL", "L": "HKL"}).set_index(["H", "K", "L"])
    for name, (values, dtype) in columns.items():
        ds[name] = rs.DataSeries(values, index=ds.index, dtype=dtype)
    ds.write_mtz(str(path))


_HKLS = [(1, 0, 1), (2, 0, 1), (3, 0, 1), (1, 1, 1), (2, 1, 1), (3, 1, 2)]


def _extrapolated(tmp_path):
    import reciprocalspaceship as rs

    path = tmp_path / "extrap.mtz"
    n = len(_HKLS)
    _write_mtz(path, _HKLS, {
        "F": (np.linspace(10, 20, n), rs.StructureFactorAmplitudeDtype()),
        "PHI": (np.linspace(0, 90, n), rs.PhaseDtype()),
        "SIGF": (np.ones(n), rs.StandardDeviationDtype()),
    })
    return path


def _read_flags(path):
    import reciprocalspaceship as rs

    ds = rs.read_mtz(str(path)).hkl_to_asu()
    return {tuple(int(v) for v in idx): int(f) for idx, f in ds["FreeR_flag"].items()}


def test_phenix_ready_carries_phenix_style_dark_flags(tmp_path):
    import reciprocalspaceship as rs

    dark = tmp_path / "dark.mtz"
    flags = [0, 1, 0, 0, 1, 0]  # phenix convention: 1 = test set
    _write_mtz(dark, _HKLS, {"R-free-flags": (flags, rs.MTZIntDtype())})

    out = pipeline._make_phenix_ready(_extrapolated(tmp_path), "unit", tmp_path, dark)

    assert _read_flags(out) == dict(zip(_HKLS, flags))


def test_phenix_ready_converts_ccp4_style_dark_flags(tmp_path):
    import reciprocalspaceship as rs

    dark = tmp_path / "dark.mtz"
    flags = [0, 3, 7, 0, 12, 19]  # CCP4 convention: 0 = test set
    _write_mtz(dark, _HKLS, {"FreeR_flag": (flags, rs.MTZIntDtype())})

    out = pipeline._make_phenix_ready(_extrapolated(tmp_path), "unit", tmp_path, dark)

    assert _read_flags(out) == {hkl: int(f == 0) for hkl, f in zip(_HKLS, flags)}


# --- it-TV difference map ----------------------------------------------------


def _fake_it_tv(monkeypatch):
    import xtr_estimator.processing as processing

    calls = []

    class FakeDenoiser:
        def __init__(self, *, tv_weights_to_scan, max_iterations, verbose):
            calls.append(("init", tv_weights_to_scan, max_iterations))

        def __call__(self, *, derivative, native):
            calls.append(("denoise", derivative, native))
            return "denoised", []

    def fake_kweight(*, mapset, kweight_mode, kweight_parameter):
        calls.append(("kweight", mapset.native, mapset.derivative, kweight_mode))
        return "kweighted-diffmap", None

    monkeypatch.setattr(processing, "IterativeTvDenoiser", FakeDenoiser)
    monkeypatch.setattr(processing, "kweight_diffmap_according_to_mode", fake_kweight)
    return processing, calls


def test_it_tv_override_denoises_then_kweights(monkeypatch):
    processing, calls = _fake_it_tv(monkeypatch)
    monkeypatch.setattr(pipeline, "_IT_TV_WEIGHTS_TO_SCAN", None)

    # xtr_estimator's own grid default must not leak through
    result = processing.calculate_diffmaps(
        "dark", "triggered", "calc", diffmap_mode="it_tv",
        parameters={"ittv_weights": [0.001, 0.01, 0.1]},
    )

    assert result == "kweighted-diffmap"
    assert calls[0] == ("init", None, 20)  # None: per-cycle golden-section TV weight
    assert calls[1] == ("denoise", "triggered", "dark")
    assert calls[2][:3] == ("kweight", "dark", "denoised")
    assert calls[2][3] == processing.WeightMode.optimize


def test_it_tv_override_uses_configured_weight_grid(monkeypatch):
    processing, calls = _fake_it_tv(monkeypatch)
    monkeypatch.setattr(pipeline, "_IT_TV_WEIGHTS_TO_SCAN", [0.001, 0.01, 0.1])

    processing.calculate_diffmaps("dark", "triggered", "calc", diffmap_mode="it_tv")

    assert calls[0] == ("init", [0.001, 0.01, 0.1], 20)
