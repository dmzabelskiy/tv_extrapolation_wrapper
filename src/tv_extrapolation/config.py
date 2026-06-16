from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ColumnSpec(BaseModel):
    kind: Literal["intensity", "amplitude"]
    amplitude_or_intensity: str
    sigma: str


class DatasetConfig(BaseModel):
    name: str
    dark_mtz: Path
    triggered_mtz: Path
    pdb_dark: Path
    resolution_limit: float
    columns: dict[str, ColumnSpec]
    rewrite_pdb_cell: bool = False
    estimation: dict = Field(default_factory=dict)
    masking: dict = Field(default_factory=dict)
    output_dir: Path

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DatasetConfig":
        with open(path) as handle:
            payload = yaml.safe_load(handle)
        return cls(**payload)

    def to_xtr_estimator_settings_dict(self) -> dict:
        dark_col = self.columns["dark"]
        triggered_col = self.columns["triggered"]
        columns_are_ints = dark_col.kind == "intensity"

        payload: dict = {
            "general": {
                "name_machine": self.name,
                "name_human": self.name,
                "output_folder": str(self.output_dir),
                "plot_folder": str(self.output_dir / self.name),
                "high_resolution_limit": self.resolution_limit,
                "comparison_type": "triggered",
            },
            "input_files": {
                "map_dark": str(self.dark_mtz),
                "map_triggered": str(self.triggered_mtz),
                "pdb_dark": str(self.pdb_dark),
                "impose_dark_phases": True,
                "columns_are_ints": columns_are_ints,
            },
            "map_processing": {
                "diffmap_type": "it_tv",
                "dark_mean_correction": True,
                "simple_dark_correction": True,
                "calculate_diffmap_before_f000": False,
            },
            "plot": {
                "show_plot": False,
                "save_to_file": True,
                **self.estimation,
            },
        }

        if self.masking:
            payload["masking"] = dict(self.masking)

        if columns_are_ints:
            payload["input_files"]["columns_dark_ints"] = {
                "ints_column": dark_col.amplitude_or_intensity,
                "int_uncertainty_column": dark_col.sigma,
            }
            payload["input_files"]["columns_triggered_ints"] = {
                "ints_column": triggered_col.amplitude_or_intensity,
                "int_uncertainty_column": triggered_col.sigma,
            }
        else:
            payload["input_files"]["columns_dark"] = {
                "amplitude_column": dark_col.amplitude_or_intensity,
                "phase_column": "MODEL",
                "uncertainty_column": dark_col.sigma,
            }
            payload["input_files"]["columns_triggered"] = {
                "amplitude_column": triggered_col.amplitude_or_intensity,
                "phase_column": "MODEL",
                "uncertainty_column": triggered_col.sigma,
            }

        return payload
