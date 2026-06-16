# Strategy for TV-Compatible Isomorphous Inputs

TV extrapolation is stricter than Xtrapol8 in two places:

1. Meteor/xtr-estimator requires dark and triggered `Map` objects to have close
   unit cells and the same space group before difference-map calculation.
2. xtr-estimator scales observed maps against a calculated dark-model map before
   estimating the density ratio. The observed and calculated map reflection sets
   can differ, and finite-value filtering inside scaling can trigger array-shape
   errors unless the inputs are commonized first.

## Current Pair Diagnostics

| Dataset | Common HKL | Dark dmin | Triggered dmin | Max cell delta | Current TV failure |
|---|---:|---:|---:|---:|---|
| `10ms` | 39970 | 1.900 | 2.000 | 0.000% | ok |
| `10ns` | 34612 | 1.900 | 2.100 | 0.000% | ok |
| `30ms` | 22057 | 1.620 | 1.970 | 0.000% | F000/grid shape mismatch |
| `5us` | 28730 | 1.620 | 1.800 | 0.000% | ok |
| `esrf_5ms` | 15921 | 2.000 | 2.200 | 0.611% | cell NotIsomorphousError |
| `esrf_5ms_2` | 15778 | 2.000 | 2.200 | 1.333% | cell NotIsomorphousError |
| `esrf_75ms` | 14082 | 2.000 | 2.300 | 0.389% | scaling shape mismatch |
| `low_ph` | 79243 | 1.129 | 1.130 | 0.702% | scaling shape mismatch |
| `trapping_1` | 79243 | 1.129 | 1.132 | 0.312% | scaling shape mismatch |
| `trapping_2` | 79244 | 1.128 | 1.132 | 0.484% | scaling shape mismatch |

All pairs have the triggered reflections almost entirely contained in the dark
reflection set. The main missing step is to make that common set explicit and
force all derived maps to use the same crystallographic basis before TV analysis.

## Recommended Pipeline

### 1. Create normalized TV input pairs

Write a preprocessing script, e.g. `scripts/prepare_tv_isomorphous_inputs.py`, that
creates:

```text
data/processed/tv_isomorphous/<condition>/dark_common.mtz
data/processed/tv_isomorphous/<condition>/triggered_common.mtz
data/processed/tv_isomorphous/<condition>/metadata.json
```

For each condition:

- Read dark and triggered MTZs with `reciprocalspaceship`/`gemmi`.
- Cut both to the lower common resolution, normally the triggered dmin.
- Keep only the finite common HKL set.
- Preserve the original observation columns:
  - intensity datasets: `I`, `SIGI`
  - amplitude datasets: `F`, `SIGF`
- Copy the dark FreeR column onto the common set when present.
- Stamp both output MTZs with the same space group and unit cell, using the dark
  cell as the TV reference cell.
- Record original cells, resolution limits, common HKL count, and cell deltas in
  `metadata.json`.

This is not a claim that a non-isomorphous dataset has become physically
isomorphous; it is a controlled projection onto a shared reciprocal-space basis
for TV map analysis. The metadata must preserve the original differences.

### 2. Patch the TV preparation step to commonize the calculated map

Preprocessing the two observed MTZs is necessary but not sufficient, because
xtr-estimator also generates a calculated dark-model map internally. Before each
call to `meteor.scale.scale_maps`, use only the finite common indices between:

- calculated dark model map
- observed dark map
- observed triggered map

The robust fix is a small local adapter around xtr-estimator preparation that:

- builds `map_dark_comp`
- intersects `map_dark_comp`, dark, and triggered indices
- drops rows with non-finite amplitude/sigma/phase values
- then calls scaling/difference-map code

This avoids the current SciPy/shape errors in `meteor.scale.scale_maps`, where
the residual vector can change length during robust least-squares optimization.

### 3. Classify cell compatibility before trusting estimates

Use three labels in the metadata and result table:

- `strict`: same cell within roughly 0.1%; safe baseline group.
- `moderate`: cell delta 0.1-0.7%; usable with normalized-basis TV, but compare
  carefully.
- `borderline`: cell delta above ~0.7%; run TV only as sensitivity analysis unless
  the data can be reprocessed from integration/scaling with a common reference.

Current classification:

- `strict`: `10ms`, `10ns`, `30ms`, `5us`
- `moderate`: `esrf_5ms`, `esrf_75ms`, `trapping_1`, `trapping_2`
- `borderline`: `esrf_5ms_2`, `low_ph`

### 4. Run validation before TV estimates

For each normalized pair, compute and store:

- common HKL count and completeness loss
- unit-cell deltas before normalization
- amplitude/intensity correlation on common HKLs
- scale factor and B-factor estimates
- Riso or mean fractional difference
- whether the TV estimate changes when dark/F000 correction is disabled

Datasets with unstable TV estimates across preprocessing variants should be
reported as sensitivity cases, not as final occupancies.

## Implementation Order

1. Build the normalized-pair preparation script and metadata table.
2. Run a smoke test on `30ms`, because it has identical cells and currently fails
   only from the map/F000 grid path.
3. Add the calculated-map commonization adapter if `30ms` still fails.
4. Run the strict group: `10ms`, `10ns`, `30ms`, `5us`.
5. Run the moderate group with explicit caveats.
6. Run the borderline group only after checking intensity correlation/Riso and
   report it separately if the metrics are poor.

