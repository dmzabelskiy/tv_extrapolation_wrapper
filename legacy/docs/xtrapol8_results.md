# Xtrapol8 Batch Results

Date: 2026-05-05

## Corrected Run

The real Xtrapol8 batch was run with the standalone upstream implementation:

```bash
python3 scripts/run_xtrapol8_real_batch.py --force
```

Summary CSV: `results/xtrapol8_real/summary.csv`.

Output directories: `results/xtrapol8_real/<condition>/`.

## Results

| Condition | Status | Occupancy | MTZ files | CCP4 files |
| --- | --- | ---: | ---: | ---: |
| 10ms | ok | 0.450 | 33 | 41 |
| 10ns | ok | 0.300 | 33 | 41 |
| 30ms | ok | 0.300 | 33 | 41 |
| 5us | ok | 0.400 | 33 | 41 |
| esrf_5ms | ok | 0.250 | 32 | 39 |
| esrf_5ms_2 | ok | 0.100 | 33 | 41 |
| esrf_75ms | ok | 0.500 | 27 | 29 |
| low_ph | ok | 0.500 | 28 | 31 |
| trapping_1 | ok | 0.500 | 30 | 35 |
| trapping_2 | ok | 0.350 | 32 | 39 |

## Important Caveats

The previous `phenix.development.xtrapol8` run is not a real Xtrapol8 result.
That dispatcher points to a skeletal Phenix development implementation in this
installation and produced no maps. It should be ignored for result comparison.

The corrected run uses `external/Xtrapol8_py3/Fextr.py`, converted locally from
the upstream Python 2 source so it runs under Phenix 2.0's Python 3.9.

CCP4 is not available in this shell. Xtrapol8 therefore reports missing
`pointless`, `scaleit`, and `truncate`. To keep the run valid without CCP4, the
batch uses prepared `FOBS,SIGFOBS` amplitude MTZs and
`f_and_maps.negative_and_missing=keep_no_fill`, with refinement disabled.
