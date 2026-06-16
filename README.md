# TV Extrapolation

Workspace for time-resolved crystallography extrapolation analysis on the
initial OLPVR1 PDB/MTZ data.

The active state is intentionally clean: only the latest refined Xtrapol8 batch
and its prepared inputs are kept in `results/`. Earlier TV extrapolation
attempts, dry runs, stale notes, and archived plots live under `legacy/`.

## Layout

- `initial/` - source PDB and MTZ files, kept read-only by convention
- `results/xtrapol8_real_inputs/` - prepared `FOBS,SIGFOBS` MTZ inputs for the
  refined Xtrapol8 run
- `results/xtrapol8_real_refined_occ015_035_step002/` - latest refined Xtrapol8
  outputs using occupancies `0.15..0.35` in `0.02` steps and one refinement round
- `docs/xtrapol8_refined_results.md` - active Xtrapol8 summary
- `scripts/run_xtrapol8_real_batch.py` - batch runner for the refined Xtrapol8
  workflow
- `legacy/` - archived TV attempts, old Xtrapol8 attempts, plots, and stale
  writeups

## Quick Start

Create or activate the analysis environment:

```bash
conda env create -f environment.yml
conda activate tv-extrapolation
pip install --no-deps git+https://github.com/cvazz/xtr-estimator.git@main
```

Run the current Xtrapol8 batch:

```bash
python scripts/run_xtrapol8_real_batch.py
```

The runner uses local Phenix tools to prepare `FOBS,SIGFOBS` inputs and runs the
local Python 3-compatible Xtrapol8 implementation in `external/Xtrapol8_py3`.
The latest retained batch writes per-condition outputs under
`results/xtrapol8_real_refined_occ015_035_step002/`; its merged summary is
`results/xtrapol8_real_refined_occ015_035_step002/summary_merged.csv`.

Note: Meteor `v0.4.2` is installed from the upstream GitHub tag because PyPI can
lag the latest `meteor-maps` release.

## Unified it_tv pipeline (in progress)

`5us` and `10ns` now run through the installable `tv_extrapolation`
package instead of `scripts/run_it_tv_conditions.py` +
`scripts/generate_it_tv_extrapolated_maps.py`:

```bash
tv-extrapolate run datasets/5us.yaml datasets/10ns.yaml
```

This single command cleans inputs, computes the diffmap, estimates the
extrapolation factor, and writes the extrapolated map in one pass — see
`docs/superpowers/specs/2026-06-16-unified-it-tv-pipeline-design.md` for
the design and `docs/superpowers/plans/2026-06-16-unified-it-tv-pipeline.md`
for migration status.

The other XFEL/OLPVR1 conditions (`10ms`, `30ms`, `esrf_5ms`,
`esrf_5ms_2`, `esrf_75ms`, `trapping_1`, `trapping_2`, `low_ph`) and all
OCP/OCP-CAN/ESRF/firstprocessing datasets still go through
`scripts/run_it_tv_conditions.py` + `scripts/generate_it_tv_extrapolated_maps.py`
and the various `scripts/prepare_*_it_tv_inputs.py` scripts, pending their
own migration to the `datasets/*.yaml` schema.

Note: `run_it_tv_conditions.py` silently substitutes a cached reference
diffmap (`initial/{condition}_reference/extrapolated_best_guess_diffmap_ittv.mtz`)
for a freshly computed one when such a file exists — this affected `5us`
historically (`initial/5us_reference/`) and produced a stale occupancy
estimate (chi=0.145327 vs. chi=0.126056 from a genuine fresh computation,
confirmed against the independent reference script
`initial/5us_reference/dmitrii_5us.py`). The unified pipeline never does
this substitution; if you see a similar discrepancy when migrating another
condition, check for a `{condition}_reference/` directory under `initial/`
before assuming the new pipeline is wrong.
