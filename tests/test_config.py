from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tv_extrapolation.config import DatasetConfig, OccupancyScanConfig


def _write_yaml(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_intensity_dataset_translates_to_settings_dict(tmp_path):
    config = DatasetConfig.from_yaml(
        _write_yaml(
            tmp_path,
            {
                "name": "5us",
                "dark_mtz": "initial/5us/ground.mtz",
                "triggered_mtz": "initial/5us/5us.mtz",
                "pdb_dark": "initial/5us/olpvr1_xfel_dark_refine_007.pdb",
                "resolution_limit": 1.8,
                "columns": {
                    "dark": {"kind": "intensity", "amplitude_or_intensity": "I", "sigma": "SIGI"},
                    "triggered": {"kind": "intensity", "amplitude_or_intensity": "I", "sigma": "SIGI"},
                },
                "estimation": {"solvent_density": 0.3},
                "masking": {
                    "sigma": 3.0,
                    "min_blob_size": 0.1,
                    "blocking_radius": 0.1,
                    "blocking_percentile": 0.1,
                    "exclude_solvent": True,
                    "dark_size_threshold": 0.1,
                    "exclude_positive_diffmap": True,
                    "exclude_large_occupancy_outliers": False,
                },
                "output_dir": "results/it_tv_pipeline",
            },
        )
    )

    payload = config.to_xtr_estimator_settings_dict()

    assert payload["general"]["name_machine"] == "5us"
    assert payload["general"]["high_resolution_limit"] == 1.8
    assert payload["input_files"]["map_dark"] == "initial/5us/ground.mtz"
    assert payload["input_files"]["columns_are_ints"] is True
    assert payload["input_files"]["columns_dark_ints"] == {
        "ints_column": "I",
        "int_uncertainty_column": "SIGI",
    }
    assert payload["plot"]["solvent_density"] == 0.3
    assert payload["masking"]["min_blob_size"] == 0.1


def test_amplitude_dataset_omits_masking_when_unset(tmp_path):
    config = DatasetConfig.from_yaml(
        _write_yaml(
            tmp_path,
            {
                "name": "demo",
                "dark_mtz": "a.mtz",
                "triggered_mtz": "b.mtz",
                "pdb_dark": "c.pdb",
                "resolution_limit": 2.0,
                "columns": {
                    "dark": {"kind": "amplitude", "amplitude_or_intensity": "F", "sigma": "SIGF"},
                    "triggered": {"kind": "amplitude", "amplitude_or_intensity": "F", "sigma": "SIGF"},
                },
                "output_dir": "results/demo",
            },
        )
    )

    payload = config.to_xtr_estimator_settings_dict()

    assert payload["input_files"]["columns_are_ints"] is False
    assert payload["input_files"]["columns_dark"] == {
        "amplitude_column": "F",
        "phase_column": "MODEL",
        "uncertainty_column": "SIGF",
    }
    assert "masking" not in payload


def test_dataset_config_missing_required_name_raises_validation_error(tmp_path):
    path = _write_yaml(
        tmp_path,
        {
            "dark_mtz": "a.mtz",
            "triggered_mtz": "b.mtz",
            "pdb_dark": "c.pdb",
            "resolution_limit": 2.0,
            "columns": {
                "dark": {"kind": "amplitude", "amplitude_or_intensity": "F", "sigma": "SIGF"},
                "triggered": {"kind": "amplitude", "amplitude_or_intensity": "F", "sigma": "SIGF"},
            },
            "output_dir": "results/demo",
        },
    )

    with pytest.raises(ValidationError):
        DatasetConfig.from_yaml(path)


def test_mixed_column_kinds_follow_dark_kind_for_translated_column_paths(tmp_path):
    config = DatasetConfig.from_yaml(
        _write_yaml(
            tmp_path,
            {
                "name": "mixed",
                "dark_mtz": "a.mtz",
                "triggered_mtz": "b.mtz",
                "pdb_dark": "c.pdb",
                "resolution_limit": 2.0,
                "columns": {
                    "dark": {"kind": "amplitude", "amplitude_or_intensity": "F_DARK", "sigma": "SIGF_DARK"},
                    "triggered": {
                        "kind": "intensity",
                        "amplitude_or_intensity": "I_TRIG",
                        "sigma": "SIGI_TRIG",
                    },
                },
                "output_dir": "results/mixed",
            },
        )
    )

    payload = config.to_xtr_estimator_settings_dict()

    assert payload["input_files"]["columns_are_ints"] is False
    assert payload["input_files"]["columns_dark"] == {
        "amplitude_column": "F_DARK",
        "phase_column": "MODEL",
        "uncertainty_column": "SIGF_DARK",
    }
    assert payload["input_files"]["columns_triggered"] == {
        "amplitude_column": "I_TRIG",
        "phase_column": "MODEL",
        "uncertainty_column": "SIGI_TRIG",
    }
    assert "columns_dark_ints" not in payload["input_files"]
    assert "columns_triggered_ints" not in payload["input_files"]


def test_dataset_config_omitted_estimation_and_masking_default_to_empty_dicts(tmp_path):
    config = DatasetConfig.from_yaml(
        _write_yaml(
            tmp_path,
            {
                "name": "demo",
                "dark_mtz": "a.mtz",
                "triggered_mtz": "b.mtz",
                "pdb_dark": "c.pdb",
                "resolution_limit": 2.0,
                "columns": {
                    "dark": {"kind": "amplitude", "amplitude_or_intensity": "F", "sigma": "SIGF"},
                    "triggered": {"kind": "amplitude", "amplitude_or_intensity": "F", "sigma": "SIGF"},
                },
                "output_dir": "results/demo",
            },
        )
    )

    assert config.estimation == {}
    assert config.masking == {}


def test_occupancy_scan_config_round_trip(tmp_path):
    yaml_text = """
name: test_ds
dark_mtz: /dev/null
triggered_mtz: /dev/null
pdb_dark: /dev/null
output_dir: /tmp
resolution_limit: 2.0
columns:
  dark:
    kind: amplitude
    amplitude_or_intensity: F
    sigma: SIGF
  triggered:
    kind: amplitude
    amplitude_or_intensity: F
    sigma: SIGF
occupancy_scan:
  x_grid: [0.1, 0.2, 0.3]
  phenix_bin: /usr/bin/phenix.refine
  cpus: 2
  cycles: 3
  strategy: individual_adp
"""
    p = tmp_path / "ds.yaml"
    p.write_text(yaml_text)
    cfg = DatasetConfig.from_yaml(p)
    assert cfg.occupancy_scan is not None
    assert cfg.occupancy_scan.x_grid == [0.1, 0.2, 0.3]
    assert cfg.occupancy_scan.cycles == 3


def test_occupancy_scan_config_absent_by_default(tmp_path):
    yaml_text = """
name: test_ds
dark_mtz: /dev/null
triggered_mtz: /dev/null
pdb_dark: /dev/null
output_dir: /tmp
resolution_limit: 2.0
columns:
  dark:
    kind: amplitude
    amplitude_or_intensity: F
    sigma: SIGF
  triggered:
    kind: amplitude
    amplitude_or_intensity: F
    sigma: SIGF
"""
    p = tmp_path / "ds.yaml"
    p.write_text(yaml_text)
    cfg = DatasetConfig.from_yaml(p)
    assert cfg.occupancy_scan is None
