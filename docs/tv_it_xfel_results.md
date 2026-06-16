# Iterative-TV XFEL Results

Run date: 2026-05-14

Meteor version: `meteor-maps 0.4.2` from `rs-station/meteor@v0.4.2`

Mode: `xtr-estimator` triggered workflow with `map_processing.diffmap_type: it_tv`

Primary output directory: `results/it_tv_extrapolated_maps`

| Condition | Status | Estimate | Std | Notes |
|---|---:|---:|---:|---|
| `10ms` | ok | 0.232559 | 0.031819 | Standard iterative-TV estimator run. |
| `10ns` | ok | 0.216338 | 0.030437 | Standard iterative-TV estimator run. |
| `5us` | ok | 0.133572 | 0.031895 | Standard iterative-TV estimator run. |
| `esrf_5ms` | nan | | | Cleaned to finite common reflections, but estimator returned a non-finite factor. |
| `esrf_5ms_2` | ok | 0.170360 | 0.014290 | Cleaned to finite common reflections; linear scaling loss used to avoid Huber shape-mismatch failure. |
| `esrf_75ms` | nan | | | Cleaned to finite common reflections, but estimator returned a non-finite factor. |
| `low_ph` | ok | 0.667000 | 0.138325 | Cleaned to finite common reflections; effective high-resolution limit reduced to about 1.4 A. |
| `trapping_1` | ok | 0.358517 | 0.080168 | Cleaned to finite common reflections; run at 1.4 A to avoid grid mismatch. |
| `trapping_2` | ok | 0.196814 | 0.036342 | Cleaned to finite common reflections; run at 1.4 A to avoid grid mismatch. |

Files:

- `results/it_tv_extrapolated_maps/meteor_summary.csv` - primary four-condition run
- `results/it_tv_extrapolated_maps/meteor_summary_esrf.csv` - ESRF run status and NaN checks
- `results/it_tv_extrapolated_maps/esrf_input_cleaning_summary.csv` - finite common-reflection input cleanup details
- `results/it_tv_extrapolated_maps/meteor_summary_nonxfel.csv` - low-pH and first trapping attempt
- `results/it_tv_extrapolated_maps/meteor_summary_trapping.csv` - successful trapping runs at 1.4 A

The condition folders now contain the relevant Meteor and extrapolation products
directly, for example `results/it_tv_extrapolated_maps/5us/diffmap_18.0_it_tv_010.mtz`
and `results/it_tv_extrapolated_maps/5us/5us_it_tv_extrapolated_chi_0.133572.mtz`.
