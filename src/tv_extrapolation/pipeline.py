from __future__ import annotations

import contextlib
import hashlib
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import gemmi
import matplotlib.pyplot as plt
import numpy as np
import reciprocalspaceship as rs
from meteor import scale as meteor_scale
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


@contextlib.contextmanager
def _linear_scaling_ctx():
    original = processing.scale_maps

    def _linear(*, reference_map, map_to_scale, **kwargs):
        kwargs["least_squares_loss"] = "linear"
        return meteor_scale.scale_maps(reference_map=reference_map, map_to_scale=map_to_scale, **kwargs)

    processing.scale_maps = _linear
    try:
        yield
    finally:
        processing.scale_maps = original


@contextlib.contextmanager
def _huber_safe_scaling_ctx():
    """Patch scipy.optimize.least_squares to fix the variable-length residual crash in
    meteor.scale.scale_maps.  That closure filters NaN residuals (removing them) so the
    returned vector shrinks mid-optimization; scipy's Huber loss pre-allocates rho based
    on the first call's size and then crashes on the size mismatch.

    Fix: capture the initial residual size on the first call and pad all subsequent calls
    with zeros to that size.  A zero residual contributes no gradient, which is equivalent
    to treating the missing reflection as perfectly scaled — matching the intent of the
    original NaN comment while keeping vector length fixed."""
    import scipy.optimize as _scipy_opt
    original_ls = _scipy_opt.least_squares

    def _safe_ls(fun, x0, **kwargs):
        initial_size: list[int | None] = [None]

        def _fixed_size_fun(x):
            r = fun(x)
            if initial_size[0] is None:
                initial_size[0] = len(r)
            m = initial_size[0]
            if len(r) < m:
                padded = np.zeros(m, dtype=np.float64)
                padded[:len(r)] = r
                return padded
            return r[:m]

        return original_ls(_fixed_size_fun, x0, **kwargs)

    _scipy_opt.least_squares = _safe_ls
    try:
        yield
    finally:
        _scipy_opt.least_squares = original_ls


def _finite_stats(map_coefficients: Map) -> tuple[int, int]:
    sf = map_coefficients.to_structurefactor().to_numpy()
    finite = np.isfinite(sf.real) & np.isfinite(sf.imag)
    return int(finite.sum()), int((~finite).sum())


def _minimum_resolution(map_coefficients: Map) -> float:
    return float(map_coefficients.compute_dHKL().min())


def _filter_nonfinite(map_coefficients: Map) -> Map:
    sf = map_coefficients.to_structurefactor().to_numpy()
    finite_mask = np.isfinite(sf.real) & np.isfinite(sf.imag)
    return map_coefficients[finite_mask].copy()


def _condition_seed(condition: str, base_seed: int = 20260520) -> int:
    digest = hashlib.sha256(condition.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def _rewrite_pdb_cell(pdb_path: Path, dark_mtz: Path, condition_dir: Path) -> Path:
    """Copy the unit cell from the dark MTZ into the PDB CRYST1 record."""
    mtz = gemmi.read_mtz_file(str(dark_mtz))
    structure = gemmi.read_structure(str(pdb_path))
    structure.cell = mtz.cell
    if mtz.spacegroup:
        structure.spacegroup_hm = mtz.spacegroup.hm
    out = condition_dir / f"{pdb_path.stem}_cell_corrected.pdb"
    structure.write_pdb(str(out))
    return out


def _run_phenix_rigid_body(pdb_path: Path, dark_mtz: Path, condition_dir: Path) -> Path:
    """Run phenix.refine rigid-body to re-seat the model in the corrected cell."""
    if not shutil.which("phenix.refine"):
        raise RuntimeError(
            "phenix.refine not found in PATH; cannot use --phenix-refine-cell"
        )
    prefix = condition_dir / "cell_refine"
    cmd = [
        "phenix.refine",
        str(dark_mtz),
        str(pdb_path),
        "refinement.main.strategy=rigid_body",
        f"output.prefix={prefix}",
        "refinement.main.number_of_macro_cycles=3",
        "refinement.main.nproc=auto",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(condition_dir))
    if result.returncode != 0:
        raise RuntimeError(f"phenix.refine failed:\n{result.stderr[-2000:]}")
    pdbs = sorted(condition_dir.glob("cell_refine_*.pdb"))
    if not pdbs:
        raise RuntimeError(
            f"phenix.refine completed but no output PDB found under {condition_dir}"
        )
    return pdbs[-1]


def _make_phenix_ready(extrapolated_mtz: Path, condition: str, condition_dir: Path) -> Path:
    ds = rs.read_mtz(str(extrapolated_mtz))
    rng = np.random.default_rng(_condition_seed(condition))
    n = len(ds)
    flags = np.zeros(n, dtype=np.int32)
    n_free = max(1, int(round(n * 0.05)))
    flags[rng.choice(n, size=n_free, replace=False)] = 1

    out = rs.DataSet(index=ds.index.copy())
    out.cell = ds.cell
    out.spacegroup = ds.spacegroup
    f_vals = ds["F"].to_numpy(dtype=float)
    phi_vals = ds["PHI"].to_numpy(dtype=float)
    sigf_vals = ds["SIGF"].to_numpy(dtype=float)
    out["FEXTRA"] = rs.DataSeries(f_vals, index=out.index, dtype=rs.StructureFactorAmplitudeDtype())
    out["SIGFEXTRA"] = rs.DataSeries(sigf_vals, index=out.index, dtype=rs.StandardDeviationDtype())
    out["FreeR_flag"] = rs.DataSeries(flags, index=out.index, dtype=rs.MTZIntDtype())
    out["2FOFCWT"] = rs.DataSeries(f_vals, index=out.index, dtype=rs.StructureFactorAmplitudeDtype())
    out["PH2FOFCWT"] = rs.DataSeries(phi_vals, index=out.index, dtype=rs.PhaseDtype())
    out["FWT"] = rs.DataSeries(f_vals, index=out.index, dtype=rs.StructureFactorAmplitudeDtype())
    out["PHWT"] = rs.DataSeries(phi_vals, index=out.index, dtype=rs.PhaseDtype())

    phenix_dir = condition_dir / "phenix_ready"
    phenix_dir.mkdir(parents=True, exist_ok=True)
    target = phenix_dir / f"{condition}_it_tv_extrapolated_phenix_ready.mtz"
    out.write_mtz(str(target))
    return target


def _run_occupancy_scan(
    config: DatasetConfig,
    phenix_ready_mtz: Path,
    condition_dir: Path,
) -> tuple[float | None, float | None, str, str]:
    """Run refine-extrap then occupancy scan. Returns (best_x, best_rfree, csv_path, plot_path)."""
    from .occupancy_scan import run_phenix_adp_refine, run_scan

    scan_cfg = config.occupancy_scan
    assert scan_cfg is not None

    refine_dir = condition_dir / "extrap_refine"
    refine_dir.mkdir(parents=True, exist_ok=True)
    _log, ok = run_phenix_adp_refine(
        config.pdb_dark,
        phenix_ready_mtz,
        refine_dir,
        cif_files=scan_cfg.cif_files,
        cpus=scan_cfg.cpus,
        phenix_bin=str(scan_cfg.phenix_bin),
        strategy="individual_sites+individual_adp",
        cycles=scan_cfg.cycles,
    )
    if not ok:
        print(f"WARNING: occupancy scan for {config.name}: refine-extrap failed")
        return None, None, "", ""

    extrap_pdbs = sorted(refine_dir.glob("*_refine_*.pdb"))
    if not extrap_pdbs:
        print(f"WARNING: occupancy scan for {config.name}: no refined PDB found in {refine_dir}")
        return None, None, "", ""
    extrap_pdb = extrap_pdbs[-1]

    scan_dir = condition_dir / "occupancy_scan"
    result = run_scan(
        config.pdb_dark,
        extrap_pdb,
        config.triggered_mtz,
        out_dir=scan_dir,
        x_grid=scan_cfg.x_grid,
        cif_files=scan_cfg.cif_files,
        cpus=scan_cfg.cpus,
        phenix_bin=str(scan_cfg.phenix_bin),
        strategy=scan_cfg.strategy,
        cycles=scan_cfg.cycles,
    )
    best_x = result.best.x if result.best else None
    best_rfree = result.best.rfree if result.best else None
    csv_path = str(scan_dir / "scan_results.csv") if (scan_dir / "scan_results.csv").exists() else ""
    plot_path = str(result.plot_path) if result.plot_path else ""
    return best_x, best_rfree, csv_path, plot_path


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
    phenix_ready_mtz: str = ""
    scan_best_x: float | None = None
    scan_best_rfree: float | None = None
    scan_csv: str = ""
    scan_plot: str = ""
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
            "scan_best_x": _fmt(self.scan_best_x),
            "scan_best_rfree": _fmt(self.scan_best_rfree),
            "diffmap_mtz": self.diffmap_mtz,
            "extrapolated_mtz": self.extrapolated_mtz,
            "extrapolated_ccp4": self.extrapolated_ccp4,
            "phenix_ready_mtz": self.phenix_ready_mtz,
            "scan_csv": self.scan_csv,
            "scan_plot": self.scan_plot,
            "error": self.error,
        }


def run(config: DatasetConfig) -> EstimationResult:
    condition_dir = config.output_dir / config.name
    condition_dir.mkdir(parents=True, exist_ok=True)

    if config.rewrite_pdb_cell or config.phenix_refine_cell:
        corrected_pdb = _rewrite_pdb_cell(config.pdb_dark, config.dark_mtz, condition_dir)
        if config.phenix_refine_cell:
            corrected_pdb = _run_phenix_rigid_body(corrected_pdb, config.dark_mtz, condition_dir)
        config = config.model_copy(update={"pdb_dark": corrected_pdb})

    settings = Settings(**config.to_xtr_estimator_settings_dict())
    resolved = dump_config(settings)

    if config.scaling_loss == "linear":
        scaling_ctx = _linear_scaling_ctx()
    elif config.scaling_loss == "huber_safe":
        scaling_ctx = _huber_safe_scaling_ctx()
    else:
        scaling_ctx = contextlib.nullcontext()
    with scaling_ctx:
        try:
            unscaled_dark, unscaled_triggered = get_maps(resolved)
            dark_finite, dark_nonfinite = _finite_stats(unscaled_dark)
            triggered_finite, triggered_nonfinite = _finite_stats(unscaled_triggered)
            if config.finite_filter:
                unscaled_dark = _filter_nonfinite(unscaled_dark)
                unscaled_triggered = _filter_nonfinite(unscaled_triggered)
            unscaled_dark, unscaled_triggered = _align_reference_style_inputs(
                unscaled_dark, unscaled_triggered, resolved
            )
            diffmap, map_dark, _map_triggered = prepare_maps(
                unscaled_dark, unscaled_triggered, resolved
            )
            inclusion_mask = make_inclusion_mask(diffmap, map_dark, resolved)
            fig, _ax, prediction = plot_extrapolation_estimate(
                diffmap, map_dark, inclusion_mask, resolved, compact=False
            )
            chi = float(prediction[0])
            std = float(prediction[1])
            plot_path = condition_dir / f"{config.name}_extrapolation_estimate.png"
            fig.savefig(plot_path)
            plt.close(fig)
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

    factor = 1.0 / chi if np.isfinite(chi) else None

    if not np.isfinite(chi):
        diffmap_path = condition_dir / f"{config.name}_it_tv_diffmap_xtr_nan.mtz"
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
        )

    assert factor is not None
    diffmap_path = condition_dir / f"{config.name}_it_tv_diffmap_xtr{factor:.2f}.mtz"
    diffmap.write_mtz(diffmap_path)

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

    phenix_path = _make_phenix_ready(mtz_path, config.name, condition_dir)

    scan_best_x = scan_best_rfree = None
    scan_csv = scan_plot = ""
    if config.occupancy_scan is not None:
        try:
            scan_best_x, scan_best_rfree, scan_csv, scan_plot = _run_occupancy_scan(
                config, Path(phenix_path), condition_dir
            )
        except Exception as exc:
            scan_plot = f"scan error: {type(exc).__name__}: {exc}"

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
        phenix_ready_mtz=str(phenix_path),
        scan_best_x=scan_best_x,
        scan_best_rfree=scan_best_rfree,
        scan_csv=scan_csv,
        scan_plot=scan_plot,
    )
