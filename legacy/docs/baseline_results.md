# Baseline Results

Generated on 2026-05-05 using `xtr-estimator` commit
`a6a29c06f48d34c1ce5675f36612f28b012b5c87` with `meteor-maps==0.4.1`.

## Baseline: `configs/xtr`

Uses `vanilla` difference maps and the default dark/F000 correction path.

| Condition | Status | Estimate | Std | Notes |
|---|---:|---:|---:|---|
| `10ms` | ok | 0.140517 | 0.013218 | Good first-pass result. |
| `10ns` | ok | 0.143155 | 0.040412 | Larger relative uncertainty. |
| `5us` | ok | 0.116940 | 0.018691 | Good first-pass result. |
| `30ms` | error | | | Autoshift grid mismatch. |
| `esrf_5ms` | error | | | Not isomorphous enough for Meteor diffmap. |
| `esrf_5ms_2` | error | | | Not isomorphous enough for Meteor diffmap. |
| `esrf_75ms` | error | | | Scaling residual shape mismatch. |
| `low_ph` | error | | | Scaling residual shape mismatch. |
| `trapping_1` | error | | | Scaling residual shape mismatch. |
| `trapping_2` | error | | | Scaling residual shape mismatch. |

Summary CSV: `results/xtr/summary.csv`.

## No Dark Mean Correction: `configs/xtr_no_darkcorr`

This variant disables the dark/triggered F000 correction step.

| Condition | Status | Estimate | Std | Notes |
|---|---:|---:|---:|---|
| `10ms` | ok | 0.190158 | 0.013175 | Different from corrected baseline; correction matters. |
| `5us` | ok | 0.164063 | 0.036185 | Different from corrected baseline; correction matters. |
| `10ns` | nan | nan | nan | Mask/estimation range insufficient. |
| `30ms` | nan | nan | nan | Crash avoided, but no usable estimate. |
| ESRF/high-res/trapping | error | | | Still fail before usable estimation. |

Summary CSV: `results/xtr_no_darkcorr/summary.csv`.

## Prefer Intensity Columns: `configs/xtr_intensity_pref`

This variant uses `IMEAN/SIGIMEAN` when present instead of `F/SIGF`.

The successful conditions are unchanged for `10ms`, `10ns`, and `5us`.
Amplitude-rich conditions still fail, which suggests the blocker is not simply
the choice of F/SIGF versus IMEAN/SIGIMEAN. The remaining work is preprocessing
the MTZ pairs so they have aligned cells, reflection sets, and map grids before
passing them into `xtr-estimator`.

Summary CSV: `results/xtr_intensity_pref/summary.csv`.

## Next Technical Step

Implement a preprocessing step for failed conditions:

1. Read each dark/triggered pair with `gemmi`/`reciprocalspaceship`.
2. Put both datasets onto a shared HKL set and common resolution cutoff.
3. Normalize or transplant compatible unit-cell metadata only when
   scientifically justified.
4. Write preprocessed MTZ pairs into `results/preprocessed/`.
5. Regenerate configs against preprocessed MTZs and rerun the baseline.

The two ESRF split-window cases have visibly different unit cells, so they
should be treated carefully rather than forced through the current pipeline.
