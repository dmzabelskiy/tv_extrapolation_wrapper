# XTR Estimation Plan

## Objective

Estimate extrapolation factors for every condition in `initial/` using the
`xtr-estimator` triggered-map workflow:

1. Load dark/ground MTZ and triggered/light MTZ.
2. Derive phases from the dark PDB when MTZ phases are absent.
3. Prepare and scale maps.
4. Generate a difference map with a selected map-processing mode.
5. Build a negative-difference-density inclusion mask.
6. Estimate `chi^-1 = -Delta rho / rho_dark` from masked voxels.
7. Save plots, executed configs, logs, and summary CSV rows.

## Data Handling

The source data in `initial/` should remain unchanged. Generated configs and
outputs go into `configs/xtr/` and `results/xtr/`.

Current data falls into two input-column classes:

- Intensity-only MTZs: `I` and `SIGI`; run with `columns_are_ints: true`.
- Amplitude MTZs: `F` and `SIGF`; run with `phase_column: MODEL` so
  `xtr-estimator` calculates phases from the dark PDB.

## Baseline Settings

The first batch should use:

- `comparison_type: triggered`
- `map_processing.diffmap_type: vanilla`
- `plot.show_plot: false`
- `plot.save_to_file: true`
- `masking` defaults from `xtr-estimator`
- pairwise high-resolution limit equal to the lower-resolution member of each
  dark/triggered MTZ pair

After the baseline succeeds, compare `vanilla`, `kweighted`, and `tv` modes on
representative conditions before committing to a final batch.

## QC Checks

Review every condition for:

- Successful numeric prediction, not `nan`.
- Generated extrapolation plot.
- Non-empty inclusion mask and enough included datapoints.
- Warnings about resolution mismatch, mask exhaustion, or large estimate
  variation.
- Stability under masking-sigma sweeps.

## Sensitivity Work

After the baseline batch:

1. Sweep `masking.sigma` across `2.5`, `3.0`, `3.5`, and `4.0`.
2. Sweep `masking.min_blob_size` if masks are unstable or empty.
3. Test `exclude_large_occupancy_outliers` for noisy cases.
4. Compare results across related conditions and replicate-like pairs.
