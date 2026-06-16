# Xtrapol8 vs TV Extrapolation Comparison

Date: 2026-05-05

Primary TV extrapolation source: `results/xtr/summary.csv`.

Real Xtrapol8 source: `results/xtrapol8_real/summary.csv`.

## Run Setup

The real Xtrapol8 run uses the standalone upstream implementation from
`external/Xtrapol8_py3/Fextr.py`, converted locally for Phenix's Python 3.9.
The earlier `phenix.development.xtrapol8` attempt is not used for this
comparison because it is a development stub in the installed Phenix tree.

Xtrapol8 was run with:

```bash
python3 scripts/run_xtrapol8_real_batch.py --force
```

Parameters:

- Occupancy grid: `0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50`
- Map type: `qfextr`
- Negative/missing ESFA handling: `keep_no_fill`
- Refinement disabled: `refinement.run_refinement=False`
- Prepared `FOBS,SIGFOBS` MTZs were used for all datasets.

CCP4 executables are not available in this shell, so Xtrapol8 reports that
`scaleit`, `pointless`, and `truncate` are unavailable. The run therefore uses
the Phenix/cctbx path, unscaled reference/triggered amplitudes, and the
`keep_no_fill` negative-reflection strategy.

## Comparison Table

| Dataset | TV status | TV occupancy | TV std | Xtrapol8 status | Xtrapol8 occupancy | Delta X8-TV | Xtrapol8 MTZ | Xtrapol8 CCP4 | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 10ms | ok | 0.1405 | 0.0132 | ok | 0.450 | 0.3095 | 33 | 41 | Both methods completed; Xtrapol8 selects a much higher occupancy on this grid. |
| 10ns | ok | 0.1432 | 0.0404 | ok | 0.300 | 0.1568 | 33 | 41 | Both methods completed; Xtrapol8 higher. |
| 30ms | error | - | - | ok | 0.300 | - | 33 | 41 | TV failed with an indexing mismatch; Xtrapol8 completed. |
| 5us | ok | 0.1169 | 0.0187 | ok | 0.400 | 0.2831 | 33 | 41 | Both methods completed; Xtrapol8 higher. |
| esrf_5ms | error | - | - | ok | 0.250 | - | 32 | 39 | TV failed on non-isomorphism; Xtrapol8 completed without CCP4 scaling. |
| esrf_5ms_2 | error | - | - | ok | 0.100 | - | 33 | 41 | TV failed on non-isomorphism; Xtrapol8 completed without CCP4 scaling. |
| esrf_75ms | error | - | - | ok | 0.500 | - | 27 | 29 | TV failed with an indexing mismatch; Xtrapol8 optimum is at the top of the tested grid. |
| low_ph | error | - | - | ok | 0.500 | - | 28 | 31 | TV failed with an indexing mismatch; Xtrapol8 optimum is at the top of the tested grid. |
| trapping_1 | error | - | - | ok | 0.500 | - | 30 | 35 | TV failed with an indexing mismatch; Xtrapol8 optimum is at the top of the tested grid. |
| trapping_2 | error | - | - | ok | 0.350 | - | 32 | 39 | TV failed with an indexing mismatch; Xtrapol8 completed. |

## Takeaways

Xtrapol8 now has real outputs for all 10 datasets, including occupancy
estimates and generated MTZ/CCP4 maps.

The direct numeric comparison is currently possible only for `10ms`, `10ns`,
and `5us`, where TV extrapolation produced baseline estimates. Xtrapol8 selects
higher occupancies for all three.

For several datasets, Xtrapol8 selected the upper end of the tested grid
(`0.500`). Those should be rerun with an extended grid before treating the
value as a bounded optimum.

Because CCP4 is not available, these Xtrapol8 values should be considered
Phenix/cctbx-only first-pass results rather than the full recommended Xtrapol8
pipeline with `pointless`/`scaleit` support.
