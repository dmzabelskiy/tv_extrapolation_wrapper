# tv_extrapolation

A wrapper pipeline around the **it_tv** (iterative total-variation) method implemented in
[rs-station/meteor](https://github.com/rs-station/meteor). Provides a unified CLI and config-driven
interface for running it_tv extrapolation across multiple datasets, with optional occupancy scanning
via Phenix refinement.

## Quick Start

```bash
conda activate tv-extrapolation

# Run all datasets in one command
tv-extrapolate run configs/datasets/ --summary results/summary.csv

# Run a single family
tv-extrapolate run configs/datasets/olpvr1_esrf/

# Run a single dataset
tv-extrapolate run configs/datasets/olpvr1_esrf/trapping_1.yaml

# Direct mode: no config file needed
tv-extrapolate run data/olpvr1_esrf/trapping_1/ground.mtz \
                   data/olpvr1_esrf/trapping_1/6_CD364A_473nm_RT_5sec_5mWt.mtz \
                   data/olpvr1_esrf/trapping_1/olpvr1_high_res_refine_72.pdb \
                   --name trapping_1 --output results/olpvr1_esrf/trapping_1 \
                   --scaling-loss huber_safe
```

## Installation

```bash
conda env create -f environment.yml
conda activate tv-extrapolation
pip install -e .
```

## CLI Reference

### `tv-extrapolate run`

Runs the it_tv pipeline on one or more datasets.

```
tv-extrapolate run FILE_OR_DIR [FILE_OR_DIR ...]
                   [--summary PATH]    # output CSV (default: results/summary.csv)
```

**YAML mode:** pass one or more `*.yaml` config files or directories containing them (searched recursively).
**Direct mode:** pass exactly `dark.mtz triggered.mtz structure.pdb` — one dataset, no config file.

Direct-mode options: `--name`, `--resolution Å`, `--output DIR`,
`--scaling-loss {huber,linear,huber_safe}`, `--finite-filter`, `--rewrite-pdb-cell`, `--phenix-refine-cell`.

### `tv-extrapolate refine-extrap`

Refines the dark model against the extrapolated MTZ to produce an excited-state model.

```
tv-extrapolate refine-extrap DARK_PDB EXTRAP_MTZ --out-dir DIR
    [--cif FILE] [--cpus N] [--phenix-bin PATH]
    [--strategy individual_sites+individual_adp]
```

### `tv-extrapolate scan`

Scans occupancy by building mixed dark/excited models and refining against the triggered MTZ.

```
tv-extrapolate scan GROUND_PDB EXTRAP_PDB TRIGGERED_MTZ --out-dir DIR
    [--x-grid X [X ...]] [--cif FILE] [--cpus N]
    [--strategy individual_adp] [--cycles N]
```

## Dataset Config Format

A minimal dataset YAML. Note that `resolution_limit` and `columns` are required in YAML mode
(auto-detection is only available in direct mode via the CLI):

```yaml
name: my_dataset
dark_mtz: data/my_protein/dark/dark_dimple.mtz
triggered_mtz: data/my_protein/laser/laser_dimple.mtz
pdb_dark: data/my_protein/dark_model.pdb
output_dir: results/my_protein/my_dataset
resolution_limit: 2.0
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
| `resolution_limit` | float | required | High-resolution cutoff (Å) |
| `columns` | dict | required | Column spec (see below) |
| `scaling_loss` | str | `huber` | `huber`, `linear`, or `huber_safe` |
| `finite_filter` | bool | false | Drop non-finite reflections before scaling |
| `rewrite_pdb_cell` | bool | false | Copy MTZ cell into PDB CRYST1 |
| `phenix_refine_cell` | bool | false | Run phenix rigid-body after cell rewrite |
| `occupancy_scan` | dict | null | If present, run occupancy scan after extrapolation |

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

Results land in `output_dir/<name>/`.

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

`results/summary.csv` accumulates results from each run.

## Data Layout

```
data/
├── olpvr1_xfel/                 # OLPVR1 XFEL datasets
│   ├── 10ms/, 10ns/, 30ms/, 5us/
└── olpvr1_esrf/                 # OLPVR1 synchrotron datasets
    ├── 5ms_0-37p5ms/, 5ms_0-75ms/, 5ms_37p5-75ms/   # ESRF
    ├── 75ms_0-37p5ms/, 75ms_0-75ms/, 75ms_37p5-75ms/ # ESRF
    ├── esrf_5ms/, esrf_5ms_2/, esrf_75ms/             # ESRF
    ├── low_ph/                                         # ESRF
    └── trapping_1/, trapping_2/                        # EMBL
```

## Known Issues

- Phenix not in PATH: always pass `--phenix-bin /path/to/phenix-2.0/bin/phenix.refine`
- Some datasets require `scaling_loss: huber_safe` when standard Huber fails (e.g. `trapping_1`, `trapping_2`)
- Column names vary by dataset — verify MTZ headers match `columns:` in your config (`F`/`SIGF` vs `F-obs`/`SIGF-obs`)
