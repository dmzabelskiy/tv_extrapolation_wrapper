# TV Extrapolation on Moderate Isomorphous Set

Borderline datasets were intentionally excluded. The processed moderate set is:

- `esrf_5ms`
- `esrf_75ms`
- `trapping_1`
- `trapping_2`

Inputs were normalized into a common reciprocal-space basis under:

```text
data/processed/tv_isomorphous/<condition>/dark_common.mtz
data/processed/tv_isomorphous/<condition>/triggered_common.mtz
```

The preprocessing used the finite common `F/SIGF` HKL set, cut to common
resolution, and stamped both output MTZs onto the dark/reference cell. Original
cell deltas are preserved in `metadata.json` per condition and in:

```text
data/processed/tv_isomorphous/manifest.csv
```

The TV runner used `--safe-scale`, which patches two fragile upstream behaviors:

- Meteor scaling uses linear least-squares loss to keep residual lengths stable.
- xtr-estimator keeps the actual common-set high-resolution limit instead of
  rounding it downward and creating mismatched map grids.

## Results

| Dataset | TV estimate | Std | Common HKL | Resolution | Max cell delta |
|---|---:|---:|---:|---:|---:|
| `esrf_5ms` | 0.1156 | 0.0068 | 14412 | 2.206 A | 0.611% |
| `esrf_75ms` | 0.1348 | 0.0118 | 10294 | 2.307 A | 0.389% |
| `trapping_1` | 0.1599 | 0.0153 | 46989 | 1.335 A | 0.312% |
| `trapping_2` | 0.1225 | 0.0264 | 46599 | 1.339 A | 0.484% |

Machine-readable summary:

```text
results/xtr_tv_isomorphous_moderate/summary.csv
```

These estimates should be treated as moderate-isomorphism TV estimates, not as
strict-isomorphism equivalents. The cell differences are small enough to analyze
with caveats, but not zero.

