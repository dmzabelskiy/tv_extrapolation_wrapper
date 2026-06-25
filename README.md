# tv_extrapolation

Time-resolved crystallography pipeline implementing the **it_tv** (iterative total-variation)
occupancy-extrapolation method. Compares against Xtrapol8 across multiple protein/dataset families.

## Quick Start

```bash
conda activate tv-extrapolation

# Run all datasets in one command
tv-extrapolate run configs/datasets/ --summary results/summary.csv

# Run a single family
tv-extrapolate run configs/datasets/ocp_ech/

# Run a single dataset
tv-extrapolate run configs/datasets/ocp_ech/ech_405_cryo.yaml

# Direct mode: no config file needed
tv-extrapolate run data/ocp_can/can_dark_nonfilt/dimple_processing/can-dark_dimple.mtz \
                   data/ocp_can/can_laser14_nonfilt/dimple_processing/can-laser14_dimple.mtz \
                   data/ocp_can/M4_CAN_dark_nonfilt_refine_002.pdb \
                   --name can_laser14 --output results/ocp_can/can_laser14
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
  cif_files: [data/ocp_ech/ech_405nm_cryo/ECH.cif]
  x_grid: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
  phenix_bin: /home/dmitrii/phenix-2.0-5936/bin/phenix.refine
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
    └── occupancy_scan_combined.png
```

`results/summary.csv` accumulates results from each run.

## Data Layout

```
data/
├── ocp_ech/                     # OCP ECH cryo datasets
│   ├── dark/                    # shared ECH dark dataset
│   ├── ech_405nm_cryo/          # 405 nm cryo illumination
│   ├── ech_laser_1h_d1/         # 1h laser, replicate 1
│   ├── ech_laser_1h_d2/         # 1h laser, replicate 2
│   ├── ech_laser_2h/            # 2h laser
│   ├── ech_laser_30min/         # 30 min laser
│   ├── ocp_filtered/            # legacy filtered dataset
│   ├── ocp_non_filtered/        # legacy non-filtered dataset
│   ├── ocp_oldflip_maxiv/       # MAX IV flip variant
│   ├── ocp_part_filt/           # partially filtered
│   └── ocp_firstprocessing/     # first-pass processing
├── ocp_can/                     # OCP CAN datasets
│   ├── can_dark_nonfilt/        # non-filtered dark
│   ├── can_dark_filt/           # filtered dark
│   ├── can_laser14_nonfilt/     # laser 14 Hz, non-filtered
│   ├── can_laser14_filt/        # laser 14 Hz, filtered
│   ├── can_laser26_nonfilt/     # laser 26 Hz, non-filtered
│   ├── can_laser26_filt/        # laser 26 Hz, filtered
│   └── M4_CAN_dark_nonfilt_refine_002.pdb
├── olpvr1_xfel/                 # OLPVR1 XFEL datasets
│   ├── 10ms/, 10ns/, 30ms/, 5us/
└── olpvr1_esrf/                 # OLPVR1 ESRF datasets
    ├── 5ms_0-37p5ms/, 5ms_0-75ms/, 5ms_37p5-75ms/
    ├── 75ms_0-37p5ms/, 75ms_0-75ms/, 75ms_37p5-75ms/
    ├── esrf_5ms/, esrf_5ms_2/, esrf_75ms/
    └── low_ph/, trapping_1/, trapping_2/
```

## Known Issues

- `can_laser26` (both filtered and non-filtered): chi=nan — likely scaling failure, unresolved
- `ocp_firstprocessing_1h`: raw intensity-only MTZ, requires `scaling_loss: huber_safe`
- Phenix not in PATH: always pass `--phenix-bin /home/dmitrii/phenix-2.0-5936/bin/phenix.refine`
- `ech_laser_1h_d1` / `ech_laser_1h_d2`: column detection may need explicit `columns:` block
