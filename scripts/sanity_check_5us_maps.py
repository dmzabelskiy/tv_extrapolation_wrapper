#!/usr/bin/env python3
"""Numerically compare 5us dark, itTV difference, and extrapolated MTZs."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from meteor.rsmap import Map
from xtr_estimator.configuration import dump_config
from xtr_estimator.main import parse_settings
from xtr_estimator.processing import get_maps


PATHS = {
    "external_ittv": "initial/5us/extrapolated_best_guess_diffmap_ittv.mtz",
    "external_xtr6.88": "initial/5us/extrapolated_best_guess_xtr6.88.mtz",
    "our": "results/it_tv_extrapolated_maps/5us/5us_it_tv_extrapolated_chi_0.133572.mtz",
    "it_tv_delta": "results/it_tv_extrapolated_maps/5us/diffmap_18.0_it_tv_010.mtz",
    "phenix_refined": "results/it_tv_extrapolated_maps/5us/5us_it_tv_dry_adp_001.mtz",
}


def _complex(path: str, amp: str = "F", phase: str = "PHI"):
    return Map.read_mtz_file(path, amplitude_column=amp, phase_column=phase).to_structurefactor()


def _stats(a, b, label: str) -> None:
    idx = a.index.intersection(b.index)
    aa = a.loc[idx].to_numpy()
    bb = b.loc[idx].to_numpy()
    amp_a = np.abs(aa)
    amp_b = np.abs(bb)
    diff = aa - bb
    amp_corr = np.corrcoef(amp_a, amp_b)[0, 1]
    complex_corr = np.vdot(aa, bb) / np.sqrt(np.vdot(aa, aa) * np.vdot(bb, bb))
    rel_rms = np.sqrt(np.mean(np.abs(diff) ** 2)) / np.sqrt(np.mean(np.abs(bb) ** 2))
    scale = np.vdot(bb, aa).real / np.vdot(bb, bb).real
    print(f"\n{label}")
    print(f"  common reflections: {len(idx)}")
    print(f"  amplitude corr:     {amp_corr:.6f}")
    print(f"  complex corr abs:   {abs(complex_corr):.6f}")
    print(f"  complex corr phase: {np.angle(complex_corr, deg=True):.3f} deg")
    print(f"  LS scale a~b:       {scale:.6f}")
    print(f"  relative complex RMS difference: {rel_rms:.6f}")
    print(f"  mean |a| / |b|:     {np.mean(amp_a) / np.mean(amp_b):.6f}")
    print(f"  median |a| / |b|:   {np.median(amp_a) / np.median(amp_b):.6f}")


def main() -> int:
    cfg = dump_config(parse_settings("configs/xtr_it_tv_xfel/5us.yaml", extra_overrides={}))
    dark, triggered = get_maps(cfg)
    executed_config = Path("results/it_tv_extrapolated_maps/executed_config.yaml")
    if executed_config.exists():
        executed_config.replace("results/it_tv_extrapolated_maps/5us/sanity_check_executed_config.yaml")
    dark_sf = dark.to_structurefactor()
    triggered_sf = triggered.to_structurefactor()
    delta = _complex(PATHS["it_tv_delta"])
    external_ittv = _complex(PATHS["external_ittv"])
    external_xtr = _complex(PATHS["external_xtr6.88"])
    ours = _complex(PATHS["our"])
    phenix_2fofc = _complex(PATHS["phenix_refined"], amp="2FOFCWT", phase="PH2FOFCWT")

    chi = 0.13357187572278476
    idx = dark_sf.index.intersection(delta.index)
    recomputed = dark_sf.loc[idx] + delta.loc[idx] / chi

    print("Reference files:")
    for name, path in PATHS.items():
        print(f"  {name}: {path}")
    print(f"\nchi used for our 5us extrapolation: {chi:.12f}")

    _stats(ours, recomputed, "our MTZ vs recomputed F_dark + Delta_itTV / chi")
    _stats(external_ittv, delta, "external extrapolated_best_guess_diffmap_ittv vs cached itTV Delta_F")
    _stats(external_ittv, ours, "external extrapolated_best_guess_diffmap_ittv vs our extrapolated MTZ")
    _stats(external_xtr, ours, "external extrapolated_best_guess_xtr6.88 vs our extrapolated MTZ")
    _stats(ours, dark_sf, "our extrapolated MTZ vs dark F0")
    _stats(external_ittv, dark_sf, "external extrapolated_best_guess_diffmap_ittv vs dark F0")
    _stats(external_xtr, dark_sf, "external extrapolated_best_guess_xtr6.88 vs dark F0")
    _stats(triggered_sf, dark_sf, "triggered 5us F vs dark F0")
    _stats(phenix_2fofc, dark_sf, "Phenix refined 2FOFCWT map coefficients vs dark F0")
    _stats(phenix_2fofc, ours, "Phenix refined 2FOFCWT map coefficients vs our extrapolated MTZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
