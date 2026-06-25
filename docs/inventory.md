# Dataset Inventory

All 30 datasets in `configs/datasets/`. Run all with:

```bash
tv-extrapolate run configs/datasets/ --summary results/summary.csv
```

## OCP ECH (`configs/datasets/ocp_ech/`, 10 datasets)

| Dataset | Resolution | Dark cols | Flags | Last chi | Last factor |
|---------|-----------|-----------|-------|----------|-------------|
| ech_405_cryo           | 1.68 | amp/F-obs  | scaling_loss=linear     | —    | — |
| ech_laser_2h           | 1.80 | amp/F-obs  | —                       | —    | — |
| ech_laser_1h_d1        | 1.83 | amp/F      | —                       | —    | — |
| ech_laser_1h_d2        | 1.83 | amp/F      | —                       | —    | — |
| ech_laser_30min        | 1.80 | amp/F      | —                       | —    | — |
| ocp_filtered           | 1.80 | amp/F      | —                       | —    | — |
| ocp_non_filtered       | 1.80 | amp/F      | —                       | —    | — |
| ocp_oldflip_maxiv      | 1.66 | amp/F      | —                       | —    | — |
| ocp_part_filt          | 1.80 | amp/F      | —                       | —    | — |
| ocp_firstprocessing_1h | 1.24 | int/IMEAN  | scaling_loss=huber_safe | —    | — |

## OCP CAN (`configs/datasets/ocp_can/`, 4 datasets)

| Dataset | Resolution | Dark cols | Flags | Last chi | Last factor |
|---------|-----------|-----------|-------|----------|-------------|
| can_laser14           | 1.90 | amp/F | —                       | — | — |
| can_laser14_filtered  | 1.93 | int/I | scaling_loss=huber_safe | — | — |
| can_laser26           | 1.85 | amp/F | —                       | nan | — |
| can_laser26_filtered  | 1.80 | int/I | scaling_loss=huber_safe | nan | — |

> `can_laser26` variants produce chi=nan — unresolved scaling failure.

## OLPVR1 XFEL (`configs/datasets/olpvr1_xfel/`, 4 datasets)

| Dataset | Resolution | Dark cols | Flags | Last chi | Last factor |
|---------|-----------|-----------|-------|----------|-------------|
| 10ms | 2.00 | int/I | — | — | — |
| 10ns | 2.10 | int/I | — | — | — |
| 30ms | 1.97 | int/I | — | — | — |
| 5us  | 1.80 | int/I | — | — | — |

## OLPVR1 ESRF (`configs/datasets/olpvr1_esrf/`, 12 datasets)

| Dataset | Resolution | Dark cols | Flags | Last chi | Last factor |
|---------|-----------|-----------|-------|----------|-------------|
| 5ms_0-37p5ms    | 2.20 | amp/F | — | — | — |
| 5ms_0-75ms      | 2.20 | amp/F | — | — | — |
| 5ms_37p5-75ms   | 2.20 | amp/F | — | — | — |
| 75ms_0-37p5ms   | 2.30 | amp/F | — | — | — |
| 75ms_0-75ms     | 2.45 | amp/F | — | — | — |
| 75ms_37p5-75ms  | 2.58 | amp/F | — | — | — |
| esrf_5ms        | 2.20 | amp/F | — | — | — |
| esrf_5ms_2      | 2.20 | amp/F | — | — | — |
| esrf_75ms       | 2.30 | amp/F | — | — | — |
| low_ph          | 1.13 | amp/F | — | — | — |
| trapping_1      | 1.13 | amp/F | — | — | — |
| trapping_2      | 1.13 | amp/F | — | — | — |

_Update this table after each batch run by reading `results/summary.csv`._
