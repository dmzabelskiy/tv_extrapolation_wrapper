# Unified it_tv Pipeline — Design

Date: 2026-06-16
Status: Approved (pending implementation)

## Problem

The `tv_extrapolation` repo computes time-resolved-crystallography occupancy
estimates by running iterative-TV (`it_tv`) difference maps through
`xtr_estimator`, and optionally cross-checking against Xtrapol8. Today this
is implemented as a sprawl of one-off scripts:

- `scripts/prepare_initial_ocp_it_tv_inputs.py`
- `scripts/prepare_initial_esrf_it_tv_inputs.py`
- `scripts/prepare_esrf_it_tv_inputs.py`
- `scripts/prepare_firstprocessing_it_tv_inputs.py`
- `scripts/prepare_initial_ocp_can_it_tv_inputs.py`
- `scripts/run_it_tv_conditions.py`
- `scripts/generate_it_tv_extrapolated_maps.py`

Each `prepare_*` script reimplements the same finite-value filtering,
intensity-to-amplitude conversion, and PDB cell-rewriting logic with minor
copy-paste variations. There is no shared library, no installable package,
no tests, and `environment.yml` is missing dependencies (`gemmi`,
`reciprocalspaceship`) that the scripts actually import. The repository also
has no working git history (`.git/` exists but is empty).

## Goal

Replace the per-dataset-family scripts with one installable package and CLI,
driven by a single YAML config per dataset, covering the `it_tv` /
`xtr_estimator` occupancy-estimation flow end-to-end (clean inputs → diffmap
→ occupancy estimate → extrapolated map). Prove it out on the two cleanest
datasets (`5us`, `10ns` from the XFEL/OLPVR1 family) via numeric regression
against existing results, then generalize to OCP/OCP-CAN/ESRF/firstprocessing
in follow-up work.

## Explicitly out of scope for this pass

- Migrating OCP/CAN/ESRF/firstprocessing datasets to the new config (planned
  follow-up once 5us/10ns prove the pattern).
- The Xtrapol8 comparison pipeline (`scripts/run_xtrapol8_real_batch.py`) —
  stays untouched.
- Pruning the existing 25GB of `results/` data.
- Fixing the `can_laser26` chi=nan estimator-threshold issue. The new config
  schema supports per-dataset `solvent_density`/`std_cutoff` overrides so
  this can be addressed later without code changes.

## Architecture

New package, installed editable into the existing `tv-extrapolation` conda
env:

```
src/tv_extrapolation/
  __init__.py
  io.py        # MTZ reading/cleaning: finite-filtering, intensity->amplitude conversion
  pdb.py       # CRYST1 cell rewriting helper
  config.py    # DatasetConfig schema (pydantic), loads/validates one YAML per dataset
  pipeline.py  # run(config) -> clean inputs, call xtr_estimator, write diffmap + extrapolated map, return summary row
  cli.py       # `tv-extrapolate run <config.yaml> [...]` entry point
pyproject.toml
```

`io.py`/`pdb.py` absorb and de-duplicate the logic currently copy-pasted
across the four `prepare_*_it_tv_inputs.py` scripts. `pipeline.py` folds
together what `run_it_tv_conditions.py` and `generate_it_tv_extrapolated_maps.py`
do today (estimate, then write the extrapolated map), into one call.

`environment.yml` gains `gemmi` and `reciprocalspaceship` as explicit
dependencies, and the new package itself via `pip install -e .`.

## Dataset config schema

One YAML file per dataset is both the "what to clean" and "how to estimate"
description:

```yaml
name: 5us
dark_mtz: initial/5us/ground.mtz
triggered_mtz: initial/5us/5us.mtz
pdb_dark: initial/5us/olpvr1_xfel_dark_refine_007.pdb
resolution_limit: null        # null = auto-detect from data (matches current check_highres_limit behavior)
columns:
  dark: {kind: intensity, amplitude_or_intensity: I, sigma: SIGI}
  triggered: {kind: intensity, amplitude_or_intensity: I, sigma: SIGI}
rewrite_pdb_cell: false        # true for OCP/CAN datasets where PDB cell doesn't match MTZ
estimation:
  solvent_density: 0.3         # per-dataset override; falls back to xtr_estimator default if omitted
  std_cutoff: null
output_dir: results/5us
```

`kind: intensity|amplitude` covers today's ad-hoc intensity-to-amplitude
branch. `rewrite_pdb_cell` covers the OCP/CAN-only CRYST1 patching. The same
schema is meant to work across all dataset families without per-family code
branches, even though only `5us`/`10ns` configs are written in this pass.

## Run flow

`pipeline.run(config: DatasetConfig) -> SummaryRow`:

1. Read dark/triggered MTZ via `reciprocalspaceship`; apply finite-value
   filtering and, if `kind: intensity`, intensity-to-amplitude conversion.
2. If `rewrite_pdb_cell`, patch the PDB's `CRYST1` line to match the MTZ
   cell.
3. Compute the calculated model map from the PDB; intersect
   dark/triggered/model-map indices to get common reflections.
4. Hand off to `xtr_estimator`: `get_maps` → `prepare_maps` →
   `make_inclusion_mask` → `plot_extrapolation_estimate`, using
   `estimation.solvent_density`/`std_cutoff` from config (xtr_estimator
   defaults if unset).
5. Write diffmap MTZ, extrapolated MTZ + CCP4, and the estimate plot PNG
   into `output_dir`.
6. Return one row (condition, chi, std, extrapolation_factor, status,
   reflection counts, output paths) — same shape as today's summary CSVs.

`cli.py run` takes one or more config paths, calls `pipeline.run` per
config, catches exceptions per-dataset (status: ok/nan/error, matching
today's pattern in `run_it_tv_conditions.py`), and writes a merged
`summary.csv`.

## Validation

Numeric regression check: run the new pipeline on `5us` and `10ns`, compare
the resulting `chi`/`extrapolation_factor` against the current values in
`results/it_tv_extrapolated_maps/summary.csv` / `meteor_summary*.csv`. Same
algorithm and same inputs should produce a near-exact match, not an
approximate one — any deviation beyond floating-point noise means the
migration introduced a behavior change and must be investigated before
proceeding.

## Migration steps

1. `git init` + baseline commit of current repo state, so every later step
   is a reviewable diff.
2. Scaffold the package (`pyproject.toml`, `src/tv_extrapolation/`
   skeleton, updated `environment.yml`). Delegated to `codex` — mechanical,
   easy to verify against this spec.
3. Implement `io.py` and `pdb.py` by hand, reconciling the four divergent
   `prepare_*_it_tv_inputs.py` implementations.
4. Implement `config.py` and `pipeline.run` by hand — the
   correctness-sensitive core, wired to `xtr_estimator`/`meteor` exactly as
   today's scripts are.
5. Write `datasets/5us.yaml` and `datasets/10ns.yaml`, translating the
   existing `configs/xtr_it_tv_xfel/5us.yaml` / `10ns.yaml`. Delegated to
   `codex` — mechanical schema translation, diff-checked afterward.
6. Numeric regression check (see Validation). Done by hand — this is the
   gate for the rest of the plan.
7. Add unit tests for `io.py`/`pdb.py`/`config.py`. Delegated to `codex` —
   standard test scaffolding against already-verified code.
8. Delete the superseded 5us/10ns code paths in `run_it_tv_conditions.py`
   and the corresponding `prepare_*` scripts once step 6 passes. Leave the
   OCP/CAN/ESRF/firstprocessing code paths in place since those datasets
   aren't migrated yet.
9. Update `README.md` to describe the new pipeline and config format.
