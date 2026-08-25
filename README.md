# tv_extrapolation

A wrapper pipeline around the **it_tv** (iterative total-variation) method implemented in
[rs-station/meteor](https://github.com/rs-station/meteor). Provides a unified CLI and config-driven
interface for running it_tv extrapolation across multiple datasets, with optional occupancy scanning
via Phenix refinement.

## Quick Start

```bash
conda activate tv-extrapolation

# Run the three bundled example datasets (data ships in examples/init/)
tv-extrapolate run examples/configs/ --summary examples/results/summary.csv

# Run a single dataset
tv-extrapolate run examples/configs/trapping_1.yaml

# Direct mode: no config file needed
tv-extrapolate run examples/init/trapping_1/ground.mtz \
                   examples/init/trapping_1/6_CD364A_473nm_RT_5sec_5mWt.mtz \
                   examples/init/trapping_1/olpvr1_high_res_refine_72.pdb \
                   --name trapping_1 --output examples/results/trapping_1 \
                   --scaling-loss huber_safe
```

Project datasets follow the same pattern with configs under `configs/datasets/<family>/`
and data under `data/<family>/<dataset>/` (both private, not in the public repo):

```bash
tv-extrapolate run configs/datasets/ --summary results/summary.csv   # everything
tv-extrapolate run configs/datasets/olpvr1_esrf/                     # one family
```

## Installation

```bash
conda env create -f environment.yml
conda activate tv-extrapolation

# xtr_estimator is git-only and must be installed WITHOUT its pinned deps
# (they conflict with this env). meteor (meteor-maps) and everything else
# are already handled by environment.yml above.
# Pinned to the commit this wrapper was validated against — pipeline.py
# monkeypatches xtr_estimator internals, so upstream main may break it.
pip install --no-deps "git+https://github.com/cvazz/xtr-estimator.git@a6a29c06f48d34c1ce5675f36612f28b012b5c87"

pip install -e .
```

Meteor is published on PyPI as **`meteor-maps`** (it imports as `meteor`) and
is pinned in `environment.yml`; it is also declared in `pyproject.toml`, so a
plain `pip install -e .` in any environment pulls it automatically. Only
`xtr_estimator` needs the manual git step above.

## CLI Reference

### `tv-extrapolate run`

Runs the it_tv pipeline on one or more datasets.

```
tv-extrapolate run FILE_OR_DIR [FILE_OR_DIR ...]
                   [--summary PATH]    # output CSV (default: results/summary.csv)
```

**YAML mode:** pass one or more `*.yaml` config files or directories containing them (searched
recursively; `legacy/` subtrees and pipeline-emitted `executed_config.yaml` files are skipped).
A config that fails to load is recorded as an `error` row in the summary and the batch continues.
**Direct mode:** pass exactly `dark.mtz triggered.mtz structure.pdb` — one dataset, no config file.

Direct-mode options: `--name`, `--resolution Å`, `--output DIR`,
`--scaling-loss {huber,linear,huber_safe}`, `--finite-filter`, `--rewrite-pdb-cell`,
`--phenix-refine-cell`, `--phenix-bin PATH`.

The TV diffmap is cached per dataset (`diffmap_*_it_tv_*.mtz` + a
`diffmap_provenance.json` sidecar recording what produced it): re-running with
an unchanged config reuses it deterministically; a config change or `--fresh`
recomputes it. If scaling crashes in the Huber loss or the chi estimate is nan,
the run is automatically retried once with `linear` scaling — the summary's
`scaling_loss` column records what actually produced each row.

### `tv-extrapolate truncate`

Writes a cleaned copy of an MTZ: cut by resolution and (by default) drop rows
whose measured data/sigma pair is non-finite or has σ ≤ 0 — free-R padding and
corrupt placeholder values.

```
tv-extrapolate truncate IN.mtz OUT.mtz [--dmin Å] [--dmax Å] [--keep-unmeasured]
```

### `tv-extrapolate refine-extrap`

Refines the dark model against the extrapolated MTZ to produce an excited-state model.

```
tv-extrapolate refine-extrap DARK_PDB EXTRAP_MTZ --out-dir DIR
    [--cif FILE] [--cpus N] [--phenix-bin PATH]
    [--strategy individual_sites+individual_adp] [--cycles N]
```

### `tv-extrapolate scan`

Scans occupancy by building mixed dark/excited models and refining against the triggered MTZ.

```
tv-extrapolate scan GROUND_PDB EXTRAP_PDB TRIGGERED_MTZ --out-dir DIR
    [--x-grid X [X ...]] [--cif FILE] [--cpus N]
    [--mode {occupancy,coords}] [--strategy individual_adp] [--cycles N]
```

`--mode occupancy` (default) builds two-conformer models with occupancies `1-x`/`x`;
`--mode coords` linearly interpolates coordinates instead. `--cycles` defaults to 3 —
1-cycle ADP refinement does not converge. The CLI's default `--x-grid` is
`0.0 … 0.5`; the YAML `occupancy_scan.x_grid` default is `0.05 … 0.30`.

## Dataset Config Format

A minimal dataset YAML. Only `columns` must be specified. When `resolution_limit` is omitted it
is auto-detected as the coarser of the two files' **measured-data extent** — the finest d-spacing
carrying a real Fobs/intensity with σ > 0 — so free-R-flag reflections padded to the detector edge
never inflate it. A junk guard additionally checks per-shell completeness: if the footprint
collapses to ~0 and then resurrects at higher resolution (corrupt values written past the real
data), the limit is pulled back to the last well-measured shell with a warning:

```yaml
name: my_dataset
dark_mtz: data/my_protein/dark/dark_dimple.mtz
triggered_mtz: data/my_protein/laser/laser_dimple.mtz
pdb_dark: data/my_protein/dark_model.pdb
output_dir: results/my_protein/my_dataset
columns:
  dark:
    kind: amplitude        # or intensity
    amplitude_or_intensity: F-obs
    sigma: SIGF-obs
  triggered:
    kind: amplitude
    amplitude_or_intensity: F
    sigma: SIGF
```

All available fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | required | Dataset identifier |
| `dark_mtz` | path | required | Ground-state MTZ |
| `triggered_mtz` | path | required | Laser/triggered MTZ |
| `pdb_dark` | path | required | Ground-state refined model |
| `output_dir` | path | required | Where to write results |
| `resolution_limit` | float | auto | High-resolution cutoff (Å); when omitted, the coarser of the two files' measured-data extent (finite Fobs/intensity, σ > 0; ignores free-R padding) |
| `columns` | dict | required | Column spec (see below) |
| `scaling_loss` | str | `huber_safe` | `huber`, `linear`, or `huber_safe` (huber_safe = huber plus a fix for the shrinking-residual crash; on a scaling crash or nan chi the run auto-retries with `linear`) |
| `finite_filter` | bool | false | Drop non-finite reflections before scaling |
| `rewrite_pdb_cell` | bool | false | Copy MTZ cell into PDB CRYST1 |
| `phenix_refine_cell` | bool | false | Run phenix rigid-body after cell rewrite |
| `phenix_bin` | str | `phenix.refine` | phenix.refine binary used by `phenix_refine_cell` |
| `occupancy_scan` | dict | null | If present, run occupancy scan after extrapolation |
| `estimation` | dict | `{}` | Extra keys merged into xtr_estimator's `plot:` section (e.g. `solvent_density`) |
| `masking` | dict | `{}` | Passed through as xtr_estimator's `masking:` section |

`columns` block — specify `kind: amplitude` or `kind: intensity` to match your MTZ:

```yaml
columns:
  dark:
    kind: amplitude        # or intensity
    amplitude_or_intensity: F-obs
    sigma: SIGF-obs
  triggered:
    kind: amplitude
    amplitude_or_intensity: F
    sigma: SIGF
```

`occupancy_scan` block — triggers automated refine-extrap + scan after extrapolation:

```yaml
occupancy_scan:
  cif_files: [data/my_protein/ligand.cif]   # omit if no CIF needed
  x_grid: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
  phenix_bin: /path/to/phenix-2.0/bin/phenix.refine
  cpus: 4
  cycles: 3
  strategy: individual_adp
```

## Adding a New Dataset

1. Put your MTZ and PDB files under `data/<family>/<dataset>/`
2. Create `configs/datasets/<family>/<name>.yaml` with the required fields
3. Run: `tv-extrapolate run configs/datasets/<family>/<name>.yaml`

Results land in `output_dir` when it already ends with the dataset name
(the `configs/datasets/` convention), otherwise in `output_dir/<name>/`.

## Output Structure

Each dataset produces:

```
results/<family>/<dataset>/
├── executed_config.yaml                          # full resolved config for reproducibility
├── <dataset>_extrapolation_estimate.png          # chi estimation plot
├── <dataset>_it_tv_diffmap_xtr<N>.mtz            # it-tv difference map
├── <dataset>_it_tv_extrapolated_xtr<N>.mtz       # extrapolated structure factors
├── <dataset>_it_tv_extrapolated_xtr<N>.ccp4      # extrapolated electron density map
└── phenix_ready/
    └── <dataset>_it_tv_extrapolated_phenix_ready.mtz
```

If `occupancy_scan:` is configured, two more directories appear:

```
├── extrap_refine/     # phenix.refine of dark PDB against extrapolated MTZ
└── occupancy_scan/
    ├── scan_results.csv
    └── occupancy_scan.png
```

`results/summary.csv` accumulates across runs: rows are merged by condition name,
so re-running a dataset replaces its row and leaves the others in place.

## Data Layout

The three bundled example datasets are from:
> Zabelskii et al. (2025) *Nature Structural & Molecular Biology* — https://www.nature.com/articles/s41594-025-01488-7

```
examples/init/
├── trapping_1/     # OLPVR1, EMBL
├── trapping_2/     # OLPVR1, EMBL
└── low_ph/         # OLPVR1, ESRF
```

## Known Issues

- Phenix not in PATH: pass `--phenix-bin /path/to/phenix-2.0/bin/phenix.refine` (or set
  `phenix_bin:` / `occupancy_scan.phenix_bin:` in the dataset YAML)
- A few datasets need `scaling_loss: linear` (e.g. `ech_405_cryo`); the automatic linear
  retry handles this, but pinning it in the YAML skips the wasted first attempt
- Column names vary by dataset — verify MTZ headers match `columns:` in your config (`F`/`SIGF` vs `F-obs`/`SIGF-obs`)
