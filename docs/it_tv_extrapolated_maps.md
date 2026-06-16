# Iterative-TV Extrapolated Maps

Generated on 2026-05-14 for datasets with finite TV-estimator values:
`5us`, `10ms`, `10ns`, `esrf_5ms_2`, `low_ph`, `trapping_1`, and
`trapping_2`. `30ms` is intentionally excluded.
The ESRF `esrf_5ms` and `esrf_75ms` datasets produced denoised diffmaps but
non-finite extrapolation factors, so extrapolated maps were not generated for
those two cases.

Formula:

`F_extrapolated = F_dark + Delta_F_itTV / chi`

where `Delta_F_itTV` is the complex structure factor from the cached iterative-TV
denoised difference map, not the simple difference map.

| Condition | chi | Common reflections | Extrapolated MTZ | Dry refine |
|---|---:|---:|---|---|
| `5us` | 0.133572 | 28715 | `results/it_tv_extrapolated_maps/5us/5us_it_tv_extrapolated_chi_0.133572.mtz` | ok, final R/Rfree 0.4196/0.4647 |
| `10ms` | 0.232559 | 39965 | `results/it_tv_extrapolated_maps/10ms/10ms_it_tv_extrapolated_chi_0.232559.mtz` | ok, final R/Rfree 0.4192/0.4531 |
| `10ns` | 0.216338 | 34584 | `results/it_tv_extrapolated_maps/10ns/10ns_it_tv_extrapolated_chi_0.216338.mtz` | ok, final R/Rfree 0.4098/0.4478 |
| `esrf_5ms_2` | 0.170360 | 12282 | `results/it_tv_extrapolated_maps/esrf_5ms_2/esrf_5ms_2_it_tv_extrapolated_chi_0.170360.mtz` | ok, final R/Rfree 0.4168/0.4630 |
| `low_ph` | 0.667000 | 35993 | `results/it_tv_extrapolated_maps/low_ph/low_ph_it_tv_extrapolated_chi_0.667000.mtz` | ok, final R/Rfree 0.4299/0.4534 |
| `trapping_1` | 0.358517 | 46436 | `results/it_tv_extrapolated_maps/trapping_1/trapping_1_it_tv_extrapolated_chi_0.358517.mtz` | ok, final R/Rfree 0.4548/0.4653 |
| `trapping_2` | 0.196814 | 46107 | `results/it_tv_extrapolated_maps/trapping_2/trapping_2_it_tv_extrapolated_chi_0.196814.mtz` | ok, final R/Rfree 0.4629/0.4913 |

Each extrapolated MTZ contains `F`, `PHI`, and propagated `SIGF` columns. A CCP4
map made directly from those complex coefficients is written next to each MTZ.
The corresponding Meteor iterative-TV diffmap MTZ, Meteor plot, extrapolated
map files, and ADP-refinement sanity-check products are kept in the same
condition folder.

The Phenix sanity checks used one generated R-free set and one macrocycle of
ADP-only refinement:

`phenix.refine dark_model extrapolated.mtz xray_data.labels=F,SIGF xray_data.r_free_flags.generate=True strategy=individual_adp main.number_of_macro_cycles=1 ordered_solvent=false`

Summary CSV:

- `results/it_tv_extrapolated_maps/summary.csv`
- `results/it_tv_extrapolated_maps/refinement_summary.csv`
