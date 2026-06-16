# Refined Xtrapol8 Results

Run directory: `results/xtrapol8_real_refined_occ015_035_step002`

This run used the standalone upstream Xtrapol8 implementation with:

- Occupancy grid: `0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35`
- Refinement enabled: `refinement.run_refinement=True`
- Reciprocal-space refinement cycles: `refinement.phenix_keywords.main.cycles=1`
- Real-space refinement cycles: `refinement.phenix_keywords.real_space_refine.cycles=1`
- Map type: `qfextr`
- Negative/missing handling: `keep_no_fill`
- Coot disabled: `output.open_coot=False`

CCP4 command-line tools were still unavailable in the shell, so Xtrapol8 continued
without CCP4 `scaleit`/`pointless`/`truncate` scaling. Phenix refinement was
available and was run.

## Occupancy Summary

| Dataset | Status | Final occupancy | Difference-map max | Pearson CC | MTZ files | CCP4 maps |
|---|---:|---:|---:|---:|---:|---:|
| `10ms` | ok | 0.350 | 0.350 | 0.350 | 47 | 78 |
| `10ns` | ok | 0.270 | 0.270 | 0.270 | 47 | 78 |
| `30ms` | ok | 0.310 | 0.310 | 0.310 | 47 | 78 |
| `5us` | ok | 0.350 | 0.350 | 0.330 | 47 | 78 |
| `esrf_5ms` | ok | 0.270 | 0.270 | 0.290 | 47 | 78 |
| `esrf_5ms_2` | ok | 0.150 | 0.150 | 0.190 | 47 | 78 |
| `esrf_75ms` | ok | 0.290 | 0.290 | 0.330 | 37 | 58 |
| `low_ph` | ok | 0.330 | 0.330 | 0.350 | 41 | 66 |
| `trapping_1` | ok | 0.350 | 0.350 | 0.350 | 46 | 76 |
| `trapping_2` | ok | 0.330 | 0.330 | 0.330 | 47 | 78 |

Machine-readable outputs:

- `results/xtrapol8_real_refined_occ015_035_step002/summary_merged.csv`
- `results/xtrapol8_real_refined_occ015_035_step002/occupancy_analysis.csv`

## Compatibility Fixes Needed

The refined path exercised Xtrapol8 code that the map-only pass did not reach.
Two Python-library compatibility fixes were required:

- `external/Xtrapol8_py3/ddm.py`: replaced removed pandas `DataFrame.append`
  usage with row accumulation.
- `external/Xtrapol8_py3/distance_analysis.py`: updated SciPy `mode()` handling
  for scalar results.

