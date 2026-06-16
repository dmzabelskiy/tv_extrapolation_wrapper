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
