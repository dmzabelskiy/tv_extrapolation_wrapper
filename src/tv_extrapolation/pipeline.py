from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from meteor.rsmap import Map
from xtr_estimator.configuration import Settings, dump_config
from xtr_estimator.estimation import plot_extrapolation_estimate
from xtr_estimator.masking import make_inclusion_mask
from xtr_estimator.processing import cut_resolution, get_maps, prepare_maps
from xtr_estimator.xtr_maps import save_extrapolated_map
import xtr_estimator.processing as processing

from .config import DatasetConfig


def _install_resolution_check() -> None:
    """Match run_it_tv_conditions.py's reference-style behavior: only
    widen high_resolution_limit if the data is coarser than requested,
    never narrow/round it. xtr_estimator's own check_highres_limit always
    overrides to the rounded data resolution, which is not what today's
    XFEL configs rely on.
    """

    def check_highres_limit_reference(map_dark, map_triggered, general_config):
        dmin_dark = float(map_dark.compute_dHKL().min())
        dmin_triggered = float(map_triggered.compute_dHKL().min())
        requested = float(general_config["high_resolution_limit"])
        effective = max(dmin_dark, dmin_triggered)
        if effective - requested > 0.01:
            general_config["high_resolution_limit"] = effective
        return map_dark, map_triggered

    processing.check_highres_limit = check_highres_limit_reference


_install_resolution_check()


def _finite_stats(map_coefficients: Map) -> tuple[int, int]:
    sf = map_coefficients.to_structurefactor().to_numpy()
    finite = np.isfinite(sf.real) & np.isfinite(sf.imag)
    return int(finite.sum()), int((~finite).sum())


def _minimum_resolution(map_coefficients: Map) -> float:
    return float(map_coefficients.compute_dHKL().min())


def _align_reference_style_inputs(
    unscaled_dark: Map,
    unscaled_triggered: Map,
    config: dict,
) -> tuple[Map, Map]:
    """Reproduce run_it_tv_conditions.py's reference-style pre-processing:
    align cell/spacegroup, cut to the requested resolution, intersect the
    reflection indices of dark and triggered maps, and re-widen
    high_resolution_limit if the intersected data is coarser than requested.
    This runs before get_maps's output is handed to prepare_maps, and is
    distinct from (and in addition to) the check_highres_limit monkeypatch.
    """
    unscaled_triggered.cell = unscaled_dark.cell
    unscaled_triggered.spacegroup = unscaled_dark.spacegroup
    unscaled_dark = cut_resolution(
        unscaled_dark,
        high_resolution_limit=config["general"]["high_resolution_limit"],
    )
    unscaled_triggered = cut_resolution(
        unscaled_triggered,
        high_resolution_limit=config["general"]["high_resolution_limit"],
    )
    common_index = unscaled_dark.index.intersection(unscaled_triggered.index)
    unscaled_dark = unscaled_dark.loc[common_index].copy()
    unscaled_triggered = unscaled_triggered.loc[common_index].copy()
    for map_coefficients in (unscaled_dark, unscaled_triggered):
        map_coefficients.cell = unscaled_dark.cell
        map_coefficients.spacegroup = unscaled_dark.spacegroup

    effective_limit = max(
        config["general"]["high_resolution_limit"],
        _minimum_resolution(unscaled_dark),
        _minimum_resolution(unscaled_triggered),
    )
    if effective_limit - config["general"]["high_resolution_limit"] > 0.01:
        config["general"]["high_resolution_limit"] = effective_limit
    return unscaled_dark, unscaled_triggered


@dataclass
class EstimationResult:
    condition: str
    status: str
    chi: float | None
    std: float | None
    extrapolation_factor: float | None
    dark_finite: int
    dark_nonfinite: int
    triggered_finite: int
    triggered_nonfinite: int
    diffmap_mtz: str
    extrapolated_mtz: str
    extrapolated_ccp4: str
    error: str = ""

    def as_row(self) -> dict:
        def _fmt(value: float | None) -> str:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return ""
            return f"{value:.12g}"

        return {
            "condition": self.condition,
            "status": self.status,
            "chi": _fmt(self.chi),
            "std": _fmt(self.std),
            "extrapolation_factor": _fmt(self.extrapolation_factor),
            "dark_finite": self.dark_finite,
            "dark_nonfinite": self.dark_nonfinite,
            "triggered_finite": self.triggered_finite,
            "triggered_nonfinite": self.triggered_nonfinite,
            "diffmap_mtz": self.diffmap_mtz,
            "extrapolated_mtz": self.extrapolated_mtz,
            "extrapolated_ccp4": self.extrapolated_ccp4,
            "error": self.error,
        }


def run(config: DatasetConfig) -> EstimationResult:
    condition_dir = config.output_dir / config.name
    condition_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(**config.to_xtr_estimator_settings_dict())
    resolved = dump_config(settings)

    try:
        unscaled_dark, unscaled_triggered = get_maps(resolved)
        dark_finite, dark_nonfinite = _finite_stats(unscaled_dark)
        triggered_finite, triggered_nonfinite = _finite_stats(unscaled_triggered)
        unscaled_dark, unscaled_triggered = _align_reference_style_inputs(
            unscaled_dark, unscaled_triggered, resolved
        )
        diffmap, map_dark, _map_triggered = prepare_maps(
            unscaled_dark, unscaled_triggered, resolved
        )
        inclusion_mask = make_inclusion_mask(diffmap, map_dark, resolved)
        _fig, _ax, prediction = plot_extrapolation_estimate(
            diffmap, map_dark, inclusion_mask, resolved, compact=False
        )
        chi = float(prediction[0])
        std = float(prediction[1])
    except Exception as exc:
        return EstimationResult(
            condition=config.name,
            status="error",
            chi=None,
            std=None,
            extrapolation_factor=None,
            dark_finite=0,
            dark_nonfinite=0,
            triggered_finite=0,
            triggered_nonfinite=0,
            diffmap_mtz="",
            extrapolated_mtz="",
            extrapolated_ccp4="",
            error=f"{type(exc).__name__}: {exc}",
        )

    if not np.isfinite(chi):
        diffmap_path = condition_dir / f"{config.name}_it_tv_diffmap_chi_nan.mtz"
        diffmap.write_mtz(diffmap_path)
        return EstimationResult(
            condition=config.name,
            status="nan",
            chi=chi,
            std=std,
            extrapolation_factor=None,
            dark_finite=dark_finite,
            dark_nonfinite=dark_nonfinite,
            triggered_finite=triggered_finite,
            triggered_nonfinite=triggered_nonfinite,
            diffmap_mtz=str(diffmap_path),
            extrapolated_mtz="",
            extrapolated_ccp4="",
            error="Estimator returned a non-finite extrapolation factor.",
        )

    diffmap_path = condition_dir / f"{config.name}_it_tv_diffmap_chi_{chi:.6f}.mtz"
    diffmap.write_mtz(diffmap_path)

    factor = 1.0 / chi
    raw_mtz_path = Path(
        save_extrapolated_map(
            factor,
            map_dark,
            diffmap,
            dark_map_file_loc=str(config.dark_mtz),
            folder=condition_dir,
            name_prefix=f"{config.name}_it_tv_extrapolated",
        )
    )
    mtz_path = condition_dir / f"{config.name}_it_tv_extrapolated_xtr{factor:.2f}.mtz"
    if raw_mtz_path != mtz_path:
        raw_mtz_path.replace(mtz_path)

    extrapolated_map = Map.read_mtz_file(mtz_path, amplitude_column="F", phase_column="PHI")
    ccp4_path = condition_dir / f"{config.name}_it_tv_extrapolated_xtr{factor:.2f}.ccp4"
    extrapolated_map.to_ccp4_map(
        map_sampling=resolved["general"]["map_sampling"]
    ).write_ccp4_map(str(ccp4_path))

    return EstimationResult(
        condition=config.name,
        status="ok",
        chi=chi,
        std=std,
        extrapolation_factor=factor,
        dark_finite=dark_finite,
        dark_nonfinite=dark_nonfinite,
        triggered_finite=triggered_finite,
        triggered_nonfinite=triggered_nonfinite,
        diffmap_mtz=str(diffmap_path),
        extrapolated_mtz=str(mtz_path),
        extrapolated_ccp4=str(ccp4_path),
    )
