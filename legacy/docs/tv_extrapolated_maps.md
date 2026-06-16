# TV Extrapolated Maps

Generated for every dataset with a numeric TV estimate:

- strict/original TV: `10ms`, `10ns`, `5us`
- moderate common-basis TV: `esrf_5ms`, `esrf_75ms`, `trapping_1`, `trapping_2`

Borderline datasets and failed-TV datasets were not generated.

For each case, outputs are in:

```text
results/tv_extrapolated_maps/<condition>/
```

Each condition directory contains:

- `<condition>_tv_xtr*.mtz`: TV-extrapolated amplitudes, computed as dark map
  plus `TV estimate * difference map`.
- `<condition>_tv_diffmap_tv.mtz`: the TV/vanilla difference map used for the
  extrapolation.
- `<condition>_tv_2mFo-DFc_map_coeffs.mtz`: Phenix `2mFo-DFc` map coefficients
  calculated from the TV-extrapolated MTZ and reference model.
- `<condition>_tv_2mFo-DFc.ccp4`: real-space CCP4 map.
- `<condition>_phenix_maps.params`: Phenix map-generation parameters.
- `phenix_maps.log`: Phenix log.

Machine-readable manifest:

```text
results/tv_extrapolated_maps/summary.csv
```

Note: the requested `2mFc-mFo` map name is interpreted here as the standard
Phenix-supported `2mFo-DFc` map type.

