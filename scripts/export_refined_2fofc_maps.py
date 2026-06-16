#!/usr/bin/env python3
"""Export Phenix refined 2mFo-DFc maps and compare their coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from meteor.rsmap import Map


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RefinedMap:
    name: str
    mtz: Path
    ccp4: Path


MAPS = [
    RefinedMap(
        "our_5us",
        ROOT / "results/it_tv_extrapolated_maps/5us/5us_it_tv_dry_adp_001.mtz",
        ROOT / "results/it_tv_extrapolated_maps/5us/5us_it_tv_dry_adp_2mFo-DFc.ccp4",
    ),
    RefinedMap(
        "our_10ms",
        ROOT / "results/it_tv_extrapolated_maps/10ms/10ms_it_tv_dry_adp_001.mtz",
        ROOT / "results/it_tv_extrapolated_maps/10ms/10ms_it_tv_dry_adp_2mFo-DFc.ccp4",
    ),
    RefinedMap(
        "our_10ns",
        ROOT / "results/it_tv_extrapolated_maps/10ns/10ns_it_tv_dry_adp_001.mtz",
        ROOT / "results/it_tv_extrapolated_maps/10ns/10ns_it_tv_dry_adp_2mFo-DFc.ccp4",
    ),
    RefinedMap(
        "our_esrf_5ms_2",
        ROOT / "results/it_tv_extrapolated_maps/esrf_5ms_2/esrf_5ms_2_it_tv_dry_adp_001.mtz",
        ROOT
        / "results/it_tv_extrapolated_maps/esrf_5ms_2/esrf_5ms_2_it_tv_dry_adp_2mFo-DFc.ccp4",
    ),
    RefinedMap(
        "our_low_ph",
        ROOT / "results/it_tv_extrapolated_maps/low_ph/low_ph_it_tv_dry_adp_001.mtz",
        ROOT / "results/it_tv_extrapolated_maps/low_ph/low_ph_it_tv_dry_adp_2mFo-DFc.ccp4",
    ),
    RefinedMap(
        "our_trapping_1",
        ROOT / "results/it_tv_extrapolated_maps/trapping_1/trapping_1_it_tv_dry_adp_001.mtz",
        ROOT
        / "results/it_tv_extrapolated_maps/trapping_1/trapping_1_it_tv_dry_adp_2mFo-DFc.ccp4",
    ),
    RefinedMap(
        "our_trapping_2",
        ROOT / "results/it_tv_extrapolated_maps/trapping_2/trapping_2_it_tv_dry_adp_001.mtz",
        ROOT
        / "results/it_tv_extrapolated_maps/trapping_2/trapping_2_it_tv_dry_adp_2mFo-DFc.ccp4",
    ),
    RefinedMap(
        "benchmark_5us_diffmap_ittv",
        ROOT
        / "results/benchmark_input_extrapolated_maps/5us_diffmap_ittv/refine_adp/5us_benchmark_diffmap_ittv_dry_adp_001.mtz",
        ROOT
        / "results/benchmark_input_extrapolated_maps/5us_diffmap_ittv/refine_adp/5us_benchmark_diffmap_ittv_dry_adp_2mFo-DFc.ccp4",
    ),
    RefinedMap(
        "benchmark_5us_xtr6.88",
        ROOT
        / "results/benchmark_input_extrapolated_maps/5us_xtr6.88/refine_adp/5us_benchmark_xtr6p88_dry_adp_001.mtz",
        ROOT
        / "results/benchmark_input_extrapolated_maps/5us_xtr6.88/refine_adp/5us_benchmark_xtr6p88_dry_adp_2mFo-DFc.ccp4",
    ),
]


PAIRS = [
    ("our_5us", "benchmark_5us_diffmap_ittv"),
    ("our_5us", "benchmark_5us_xtr6.88"),
    ("benchmark_5us_diffmap_ittv", "benchmark_5us_xtr6.88"),
    ("our_5us", "our_10ms"),
    ("our_5us", "our_10ns"),
    ("our_10ms", "our_10ns"),
]


def read_2fofc(mtz: Path):
    return Map.read_mtz_file(
        mtz,
        amplitude_column="2FOFCWT",
        phase_column="PH2FOFCWT",
        uncertainty_column=None,
    )


def compare(a, b) -> dict[str, float]:
    a_sf = a.to_structurefactor()
    b_sf = b.to_structurefactor()
    idx = a_sf.index.intersection(b_sf.index)
    aa = a_sf.loc[idx].to_numpy()
    bb = b_sf.loc[idx].to_numpy()
    complex_corr = np.vdot(aa, bb) / np.sqrt(np.vdot(aa, aa) * np.vdot(bb, bb))
    diff = aa - bb
    return {
        "n_common": float(len(idx)),
        "amp_corr": float(np.corrcoef(np.abs(aa), np.abs(bb))[0, 1]),
        "complex_corr_abs": float(abs(complex_corr)),
        "complex_corr_phase_deg": float(np.angle(complex_corr, deg=True)),
        "rel_complex_rms": float(
            np.sqrt(np.mean(np.abs(diff) ** 2)) / np.sqrt(np.mean(np.abs(bb) ** 2))
        ),
        "mean_amp_ratio": float(np.mean(np.abs(aa)) / np.mean(np.abs(bb))),
    }


def main() -> int:
    maps = {}
    print("Exported model-weighted 2mFo-DFc maps:")
    for item in MAPS:
        item.ccp4.parent.mkdir(parents=True, exist_ok=True)
        mtz_map = read_2fofc(item.mtz)
        mtz_map.to_ccp4_map(map_sampling=3).write_ccp4_map(str(item.ccp4))
        maps[item.name] = mtz_map
        print(f"  {item.name}: {item.ccp4.relative_to(ROOT)}")

    print("\nCoefficient comparisons using 2FOFCWT/PH2FOFCWT:")
    for left, right in PAIRS:
        stats = compare(maps[left], maps[right])
        print(
            f"  {left} vs {right}: "
            f"n={stats['n_common']:.0f}, "
            f"amp_corr={stats['amp_corr']:.6f}, "
            f"complex_corr_abs={stats['complex_corr_abs']:.6f}, "
            f"phase={stats['complex_corr_phase_deg']:.3f} deg, "
            f"rel_rms={stats['rel_complex_rms']:.6f}, "
            f"mean_amp_ratio={stats['mean_amp_ratio']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
