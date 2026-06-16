# TV vs Refined Xtrapol8 Comparison

TV values are chi^-1 / occupancy-like estimates. Xtrapol8 values are refined final occupancies from the 0.15-0.35, step 0.02 scan with one refinement cycle.

| Dataset | Class | TV source | TV estimate | TV std | Xtrapol8 occ | X8-TV | X8/TV | X8 map max | X8 Pearson | Common HKL | Cell delta |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10ms | strict | strict/original | 0.1405 | 0.0132 | 0.3500 | 0.2095 | 2.49 | 0.3500 | 0.3500 |  |  |
| 10ns | strict | strict/original | 0.1432 | 0.0404 | 0.2700 | 0.1268 | 1.89 | 0.2700 | 0.2700 |  |  |
| 30ms | strict | strict/original failed |  |  | 0.3100 |  |  | 0.3100 | 0.3100 |  |  |
| 5us | strict | strict/original | 0.1169 | 0.0187 | 0.3500 | 0.2331 | 2.99 | 0.3500 | 0.3300 |  |  |
| esrf_5ms | moderate | moderate common-basis | 0.1156 | 0.0068 | 0.2700 | 0.1544 | 2.34 | 0.2700 | 0.2900 | 14412 | 0.611 |
| esrf_75ms | moderate | moderate common-basis | 0.1348 | 0.0118 | 0.2900 | 0.1552 | 2.15 | 0.2900 | 0.3300 | 10294 | 0.389 |
| trapping_1 | moderate | moderate common-basis | 0.1599 | 0.0153 | 0.3500 | 0.1901 | 2.19 | 0.3500 | 0.3500 | 46989 | 0.312 |
| trapping_2 | moderate | moderate common-basis | 0.1225 | 0.0264 | 0.3300 | 0.2075 | 2.69 | 0.3300 | 0.3300 | 46599 | 0.484 |
| esrf_5ms_2 | borderline/excluded | borderline not rerun |  |  | 0.1500 |  |  | 0.1500 | 0.1900 |  |  |
| low_ph | borderline/excluded | borderline not rerun |  |  | 0.3300 |  |  | 0.3300 | 0.3500 |  |  |

Borderline datasets (`esrf_5ms_2`, `low_ph`) are shown only with Xtrapol8 values because TV common-basis reruns were intentionally skipped for now.
