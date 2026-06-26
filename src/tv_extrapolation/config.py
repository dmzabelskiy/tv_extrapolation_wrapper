from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ColumnSpec(BaseModel):
    kind: Literal["intensity", "amplitude"]
    amplitude_or_intensity: str
    sigma: str


class OccupancyScanConfig(BaseModel):
    cif_files: list[Path] = Field(default_factory=list)
    x_grid: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    phenix_bin: Path = Path("phenix.refine")
    cpus: int = 2
    cycles: int = 3
    strategy: str = "individual_adp"


class DatasetConfig(BaseModel):
    name: str
    dark_mtz: Path
    triggered_mtz: Path
    pdb_dark: Path
    resolution_limit: float | None = None
    columns: dict[str, ColumnSpec]
    rewrite_pdb_cell: bool = False
    phenix_refine_cell: bool = False
    finite_filter: bool = False
    scaling_loss: Literal["huber", "linear", "huber_safe"] = "huber"
    occupancy_scan: OccupancyScanConfig | None = None
    estimation: dict = Field(default_factory=dict)
    masking: dict = Field(default_factory=dict)
    output_dir: Path

    @model_validator(mode="after")
    def _resolve_resolution(self) -> "DatasetConfig":
        if self.resolution_limit is None:
            from .mtz_inspect import detect_resolution_limit
            self.resolution_limit = detect_resolution_limit(self.dark_mtz, self.triggered_mtz)
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DatasetConfig":
        with open(path) as handle:
            payload = yaml.safe_load(handle)
        return cls(**payload)

    @classmethod
    def from_files(
        cls,
        dark_mtz: Path | str,
        triggered_mtz: Path | str,
        pdb_dark: Path | str,
        *,
        name: str | None = None,
        resolution_limit: float | None = None,
        output_dir: Path | str = Path("results"),
        scaling_loss: Literal["huber", "linear", "huber_safe"] = "huber",
        finite_filter: bool = False,
        rewrite_pdb_cell: bool = False,
        phenix_refine_cell: bool = False,
    ) -> "DatasetConfig":
        """Build a config by auto-detecting columns and resolution from the MTZ files.

        Column types (intensity vs amplitude) are inferred from MTZ column-type
        letters; amplitudes are preferred when both are present.  Resolution
        defaults to the coarser of the two datasets.  All keyword arguments
        can be used to override the auto-detected or default values.
        """
        from .mtz_inspect import detect_column_spec

        dark_mtz = Path(dark_mtz)
        triggered_mtz = Path(triggered_mtz)
        pdb_dark = Path(pdb_dark)

        dark_col = detect_column_spec(dark_mtz)
        triggered_col = detect_column_spec(triggered_mtz)

        if name is None:
            name = triggered_mtz.stem

        return cls(
            name=name,
            dark_mtz=dark_mtz,
            triggered_mtz=triggered_mtz,
            pdb_dark=pdb_dark,
            resolution_limit=resolution_limit,
            columns={"dark": dark_col, "triggered": triggered_col},
            output_dir=Path(output_dir),
            scaling_loss=scaling_loss,
            finite_filter=finite_filter,
            rewrite_pdb_cell=rewrite_pdb_cell,
            phenix_refine_cell=phenix_refine_cell,
        )

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
