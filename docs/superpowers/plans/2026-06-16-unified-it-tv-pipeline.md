# Unified it_tv Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ad hoc `run_it_tv_conditions.py` + `generate_it_tv_extrapolated_maps.py` two-script hand-off with one installable package (`tv_extrapolation`) and CLI (`tv-extrapolate run <config.yaml>...`) that goes from raw dark/triggered MTZ + ground-state PDB straight to an occupancy estimate and extrapolated map in a single pass, validated on the `5us` and `10ns` datasets.

**Architecture:** `src/tv_extrapolation/config.py` defines a `DatasetConfig` pydantic model (one YAML per dataset) and translates it into the dict shape `xtr_estimator.configuration.Settings` expects. `src/tv_extrapolation/pipeline.py` calls `xtr_estimator.processing.get_maps/prepare_maps`, `xtr_estimator.masking.make_inclusion_mask`, and `xtr_estimator.estimation.plot_extrapolation_estimate` directly — no intermediate CSV hand-off between estimation and extrapolated-map-writing, which is what made the `can_laser26` nan case silently produce no extrapolated map under today's two-script design. `src/tv_extrapolation/cli.py` is a thin argparse entry point.

**Tech Stack:** Python 3.12 (conda env `tv-extrapolation`), `xtr_estimator` 0.4.0, `meteor` 0.4.2, `reciprocalspaceship`, `gemmi`, `pydantic` 2.13, `pyyaml`, `pytest` (new dev dependency).

**Important correction from the approved spec:** the spec's migration step 8 said to delete "superseded 5us/10ns code paths" in `run_it_tv_conditions.py`/`prepare_*` scripts. Investigation during planning found these scripts are **not** per-dataset — they're generic and config-driven across the whole `configs/xtr_it_tv_xfel/` family (5us, 10ns, 10ms, 30ms, esrf_5ms, esrf_5ms_2, esrf_75ms, trapping_1, trapping_2, low_ph). Deleting them now would break the 8 conditions that are *not* being migrated in this pass. Task 10 below replaces "delete" with "stop using for 5us/10ns, document the split, leave the scripts in place."

---

### Task 1: Baseline git commit

The repo's `.git` was initialized for the spec commit only (one file staged). This task commits everything else as a starting snapshot so every later change is a reviewable diff.

**Files:**
- Modify: none (staging existing untracked files)

- [ ] **Step 1: Check what `.gitignore` already excludes**

Run: `cat .gitignore`
Expected output:
```
__pycache__/
*.py[cod]
.ipynb_checkpoints/
results/
```
This already excludes `results/` (25GB) and bytecode, so a full `git add -A` is safe and won't try to commit the data directories.

- [ ] **Step 2: Stage and commit everything else**

```bash
git add -A
git commit -m "$(cat <<'EOF'
Baseline snapshot before unified it_tv pipeline migration

Captures scripts/, configs/, docs/, environment.yml, and other
tracked-but-not-yet-committed project files as a starting point,
so the upcoming pipeline migration is a reviewable diff.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify**

Run: `git log --oneline` and `git status`
Expected: two commits total (spec doc, then this baseline), and `git status` reports `nothing to commit, working tree clean`.

---

### Task 2: Capture a fresh ground-truth baseline for 5us and 10ns

`results/it_tv_extrapolated_maps/5us/` and `.../10ns/` currently contain **multiple** diffmap/extrapolated-map files from different historical runs (e.g. `5us_it_tv_diffmap_chi_0.126056.mtz` from 2026-05-19 20:13 and `5us_it_tv_diffmap_chi_0.145327.mtz` from 2026-05-19 21:07 — the latter is newer and paired with the `xtr6.88` extrapolated map). Rather than trust old files of ambiguous provenance, regenerate fresh output now with the existing scripts at default settings, and record the resulting numbers as the regression target.

**Files:**
- Create: `tests/baseline_5us_10ns.json`

- [ ] **Step 1: Run the existing estimation script for both conditions**

```bash
cd /home/dmitrii/projects/tv_extrapolation
/home/dmitrii/miniforge3/envs/tv-extrapolation/bin/python scripts/run_it_tv_conditions.py \
  --config-dir configs/xtr_it_tv_xfel \
  --conditions 5us 10ns \
  --out results/baseline_check \
  --summary results/baseline_check/meteor_summary.csv
```
Expected: prints two lines like `5us: ok: estimate=0.NNNNNN, std=0.NNNNNN` and `10ns: ok: estimate=0.NNNNNN, std=0.NNNNNN`, then `Wrote results/baseline_check/meteor_summary.csv`.

- [ ] **Step 2: Run the existing extrapolated-map script for both conditions**

```bash
/home/dmitrii/miniforge3/envs/tv-extrapolation/bin/python scripts/generate_it_tv_extrapolated_maps.py \
  --config-dir configs/xtr_it_tv_xfel \
  --conditions 5us 10ns \
  --out results/baseline_check \
  --estimate-summary results/baseline_check/meteor_summary.csv
```
Expected: prints `5us: wrote results/baseline_check/5us/5us_it_tv_extrapolated_xtr{F}.mtz` and the same for `10ns`, then `Wrote results/baseline_check/summary.csv`.

- [ ] **Step 3: Read the two summary CSVs and write `tests/baseline_5us_10ns.json`**

```bash
mkdir -p tests
/home/dmitrii/miniforge3/envs/tv-extrapolation/bin/python - <<'EOF'
import csv, json
from pathlib import Path

estimates = {}
with open("results/baseline_check/meteor_summary.csv") as fh:
    for row in csv.DictReader(fh):
        estimates[row["condition"]] = {"chi": float(row["estimate"]), "std": float(row["std"])}

with open("results/baseline_check/summary.csv") as fh:
    for row in csv.DictReader(fh):
        estimates[row["condition"]]["extrapolation_factor"] = float(row["extrapolation_factor"])

Path("tests/baseline_5us_10ns.json").write_text(json.dumps(estimates, indent=2, sort_keys=True))
print(json.dumps(estimates, indent=2, sort_keys=True))
EOF
```
Expected: prints a JSON object with `5us` and `10ns` keys, each containing `chi`, `std`, `extrapolation_factor`. Note the printed numbers — they are this plan's regression target for Task 8.

- [ ] **Step 4: Commit the baseline file**

```bash
git add tests/baseline_5us_10ns.json
git commit -m "$(cat <<'EOF'
Capture fresh 5us/10ns baseline before pipeline migration

Old results/it_tv_extrapolated_maps/{5us,10ns} directories contain
multiple historical runs with ambiguous provenance (no log of which
flags produced which file). Regenerated once with the existing
scripts at default settings to get an unambiguous regression target
for the new unified pipeline.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```
`results/baseline_check/` itself is already excluded by `.gitignore` (`results/`), so only the small JSON file is committed.

---

### Task 3: Scaffold the package (delegated to codex)

**Files:**
- Create: `pyproject.toml`
- Create: `src/tv_extrapolation/__init__.py`
- Modify: `environment.yml`

- [ ] **Step 1: Write the codex prompt and dispatch it**

```bash
cat > /tmp/codex_task3_prompt.txt <<'EOF'
Working directory: /home/dmitrii/projects/tv_extrapolation

Create an installable Python package skeleton:

1. Create src/tv_extrapolation/__init__.py containing only:
   __version__ = "0.1.0"

2. Create pyproject.toml at the repo root with:
   - [build-system]: requires = ["setuptools>=68", "wheel"], build-backend = "setuptools.build_meta"
   - [project]: name = "tv_extrapolation", version = "0.1.0", requires-python = ">=3.12"
   - dependencies: pyyaml, pydantic>=2, numpy, gemmi, reciprocalspaceship
   - [project.optional-dependencies]: dev = ["pytest"]
   - [project.scripts]: tv-extrapolate = "tv_extrapolation.cli:main"
   - [tool.setuptools.packages.find]: where = ["src"]

3. Modify environment.yml: add `gemmi` and `reciprocalspaceship` to the
   pip: list (they are currently imported by scripts/ but missing from
   this file), and add a comment-free entry installing this package
   itself via `pip install -e .` as the last pip: list entry. Do not
   change any other existing line in environment.yml.

Do not create src/tv_extrapolation/cli.py, config.py, or pipeline.py —
those are implemented in a later step. Do not run pip install or touch
any other files.
EOF
codex exec --full-auto "$(cat /tmp/codex_task3_prompt.txt)"
```
Expected: codex reports creating `pyproject.toml` and `src/tv_extrapolation/__init__.py`, and editing `environment.yml`.

- [ ] **Step 2: Review codex's diff**

```bash
git diff --stat
git diff environment.yml
cat pyproject.toml
```
Verify: `environment.yml`'s existing lines (python=3.12, pandas, matplotlib, the `meteor` git pip install, etc.) are untouched and only `gemmi`, `reciprocalspaceship`, and `-e .` were added; `pyproject.toml` matches the spec above; no extra files were created.

- [ ] **Step 3: Install the package editable in the existing env**

```bash
/home/dmitrii/miniforge3/envs/tv-extrapolation/bin/python -m pip install -e .
/home/dmitrii/miniforge3/envs/tv-extrapolation/bin/python -m pip install pytest
/home/dmitrii/miniforge3/envs/tv-extrapolation/bin/python -c "import tv_extrapolation; print(tv_extrapolation.__version__)"
```
Expected: prints `0.1.0`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/tv_extrapolation/__init__.py environment.yml
git commit -m "$(cat <<'EOF'
Scaffold tv_extrapolation installable package

Adds pyproject.toml + src layout for the unified it_tv pipeline, and
fills in gemmi/reciprocalspaceship in environment.yml, which were
imported by scripts/ but never listed as dependencies.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Implement `config.py`

**Files:**
- Create: `src/tv_extrapolation/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path

import yaml

from tv_extrapolation.config import DatasetConfig


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/home/dmitrii/miniforge3/envs/tv-extrapolation/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tv_extrapolation.config'`.

- [ ] **Step 3: Implement `config.py`**

```python
# src/tv_extrapolation/config.py
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/home/dmitrii/miniforge3/envs/tv-extrapolation/bin/python -m pytest tests/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tv_extrapolation/config.py tests/test_config.py
git commit -m "$(cat <<'EOF'
Add DatasetConfig with translation to xtr_estimator settings

One YAML per dataset (paths, column kind, per-dataset estimation/
masking overrides) translates into the dict shape
xtr_estimator.configuration.Settings expects, replacing the
hand-written configs/xtr_it_tv_xfel/*.yaml format for migrated
datasets.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Implement `pipeline.py`

This is the correctness-sensitive core: clean inputs → diffmap → occupancy estimate → extrapolated map, in one call, with no intermediate CSV hand-off.

**Files:**
- Create: `src/tv_extrapolation/pipeline.py`
- Test: `tests/test_pipeline_5us.py`

- [ ] **Step 1: Implement `pipeline.py`**

```python
# src/tv_extrapolation/pipeline.py
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from meteor.rsmap import Map
from xtr_estimator.configuration import Settings, dump_config
from xtr_estimator.estimation import plot_extrapolation_estimate
from xtr_estimator.masking import make_inclusion_mask
from xtr_estimator.processing import get_maps, prepare_maps
from xtr_estimator.xtr_maps import save_extrapolated_map
import xtr_estimator.processing as processing

from .config import DatasetConfig


def _install_resolution_check() -> None:
    """Match run_it_tv_conditions.py's reference-style behavior: only
    widen high_resolution_limit if the data is coarser than requested,
    never narrow/round it. xtr_estimator's own check_highres_limit always
    overrides to the rounded data resolution, which is not what today's
    XFEL configs rely on.
    """

    def check_highres_limit_reference(map_dark, map_triggered, general_config):
        dmin_dark = float(map_dark.compute_dHKL().min())
        dmin_triggered = float(map_triggered.compute_dHKL().min())
        requested = float(general_config["high_resolution_limit"])
        effective = max(dmin_dark, dmin_triggered)
        if effective - requested > 0.01:
            general_config["high_resolution_limit"] = effective
        return map_dark, map_triggered

    processing.check_highres_limit = check_highres_limit_reference


_install_resolution_check()


def _finite_stats(map_coefficients: Map) -> tuple[int, int]:
    sf = map_coefficients.to_structurefactor().to_numpy()
    finite = np.isfinite(sf.real) & np.isfinite(sf.imag)
    return int(finite.sum()), int((~finite).sum())


@dataclass
class EstimationResult:
    condition: str
    status: str
    chi: float | None
    std: float | None
    extrapolation_factor: float | None
    dark_finite: int
    dark_nonfinite: int
    triggered_finite: int
    triggered_nonfinite: int
    diffmap_mtz: str
    extrapolated_mtz: str
    extrapolated_ccp4: str
    error: str = ""

    def as_row(self) -> dict:
        def _fmt(value: float | None) -> str:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return ""
            return f"{value:.12g}"

        return {
            "condition": self.condition,
            "status": self.status,
            "chi": _fmt(self.chi),
            "std": _fmt(self.std),
            "extrapolation_factor": _fmt(self.extrapolation_factor),
            "dark_finite": self.dark_finite,
            "dark_nonfinite": self.dark_nonfinite,
            "triggered_finite": self.triggered_finite,
            "triggered_nonfinite": self.triggered_nonfinite,
            "diffmap_mtz": self.diffmap_mtz,
            "extrapolated_mtz": self.extrapolated_mtz,
            "extrapolated_ccp4": self.extrapolated_ccp4,
            "error": self.error,
        }


def run(config: DatasetConfig) -> EstimationResult:
    condition_dir = config.output_dir / config.name
    condition_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(**config.to_xtr_estimator_settings_dict())
    resolved = dump_config(settings)

    try:
        unscaled_dark, unscaled_triggered = get_maps(resolved)
        dark_finite, dark_nonfinite = _finite_stats(unscaled_dark)
        triggered_finite, triggered_nonfinite = _finite_stats(unscaled_triggered)
        diffmap, map_dark, _map_triggered = prepare_maps(
            unscaled_dark, unscaled_triggered, resolved
        )
        inclusion_mask = make_inclusion_mask(diffmap, map_dark, resolved)
        _fig, _ax, prediction = plot_extrapolation_estimate(
            diffmap, map_dark, inclusion_mask, resolved, compact=False
        )
        chi = float(prediction[0])
        std = float(prediction[1])
    except Exception as exc:
        return EstimationResult(
            condition=config.name,
            status="error",
            chi=None,
            std=None,
            extrapolation_factor=None,
            dark_finite=0,
            dark_nonfinite=0,
            triggered_finite=0,
            triggered_nonfinite=0,
            diffmap_mtz="",
            extrapolated_mtz="",
            extrapolated_ccp4="",
            error=f"{type(exc).__name__}: {exc}",
        )

    if not np.isfinite(chi):
        diffmap_path = condition_dir / f"{config.name}_it_tv_diffmap_chi_nan.mtz"
        diffmap.write_mtz(diffmap_path)
        return EstimationResult(
            condition=config.name,
            status="nan",
            chi=chi,
            std=std,
            extrapolation_factor=None,
            dark_finite=dark_finite,
            dark_nonfinite=dark_nonfinite,
            triggered_finite=triggered_finite,
            triggered_nonfinite=triggered_nonfinite,
            diffmap_mtz=str(diffmap_path),
            extrapolated_mtz="",
            extrapolated_ccp4="",
            error="Estimator returned a non-finite extrapolation factor.",
        )

    diffmap_path = condition_dir / f"{config.name}_it_tv_diffmap_chi_{chi:.6f}.mtz"
    diffmap.write_mtz(diffmap_path)

    factor = 1.0 / chi
    raw_mtz_path = Path(
        save_extrapolated_map(
            factor,
            map_dark,
            diffmap,
            dark_map_file_loc=str(config.dark_mtz),
            folder=condition_dir,
            name_prefix=f"{config.name}_it_tv_extrapolated",
        )
    )
    mtz_path = condition_dir / f"{config.name}_it_tv_extrapolated_xtr{factor:.2f}.mtz"
    if raw_mtz_path != mtz_path:
        raw_mtz_path.replace(mtz_path)

    extrapolated_map = Map.read_mtz_file(mtz_path, amplitude_column="F", phase_column="PHI")
    ccp4_path = condition_dir / f"{config.name}_it_tv_extrapolated_xtr{factor:.2f}.ccp4"
    extrapolated_map.to_ccp4_map(
        map_sampling=resolved["general"]["map_sampling"]
    ).write_ccp4_map(str(ccp4_path))

    return EstimationResult(
        condition=config.name,
        status="ok",
        chi=chi,
        std=std,
        extrapolation_factor=factor,
        dark_finite=dark_finite,
        dark_nonfinite=dark_nonfinite,
        triggered_finite=triggered_finite,
        triggered_nonfinite=triggered_nonfinite,
        diffmap_mtz=str(diffmap_path),
        extrapolated_mtz=str(mtz_path),
        extrapolated_ccp4=str(ccp4_path),
    )
```

- [ ] **Step 2: Write `datasets/5us.yaml` by hand (needed to test the pipeline before Task 7 generates the full pair)**

```bash
mkdir -p datasets
cat > datasets/5us.yaml <<'EOF'
name: 5us
dark_mtz: initial/5us/ground.mtz
triggered_mtz: initial/5us/5us.mtz
pdb_dark: initial/5us/olpvr1_xfel_dark_refine_007.pdb
resolution_limit: 1.8
columns:
  dark: {kind: intensity, amplitude_or_intensity: I, sigma: SIGI}
  triggered: {kind: intensity, amplitude_or_intensity: I, sigma: SIGI}
rewrite_pdb_cell: false
estimation:
  solvent_density: 0.3
masking:
  sigma: 3.0
  min_blob_size: 0.1
  blocking_radius: 0.1
  blocking_percentile: 0.1
  exclude_solvent: true
  dark_size_threshold: 0.1
  exclude_positive_diffmap: true
  exclude_large_occupancy_outliers: false
output_dir: results/it_tv_pipeline
EOF
```

- [ ] **Step 3: Write an integration test against the Task 2 baseline**

```python
# tests/test_pipeline_5us.py
import json
from pathlib import Path

import pytest

from tv_extrapolation.config import DatasetConfig
from tv_extrapolation.pipeline import run

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_5us_matches_baseline():
    baseline = json.loads((REPO_ROOT / "tests/baseline_5us_10ns.json").read_text())["5us"]
    config = DatasetConfig.from_yaml(REPO_ROOT / "datasets/5us.yaml")
    result = run(config)

    assert result.status == "ok"
    assert result.chi == pytest.approx(baseline["chi"], rel=1e-6)
    assert result.extrapolation_factor == pytest.approx(baseline["extrapolation_factor"], rel=1e-6)
```

- [ ] **Step 4: Run it from the repo root (paths in `datasets/5us.yaml` are relative to cwd)**

Run: `cd /home/dmitrii/projects/tv_extrapolation && /home/dmitrii/miniforge3/envs/tv-extrapolation/bin/python -m pytest tests/test_pipeline_5us.py -v`
Expected: 1 passed. If it fails on the `chi`/`extrapolation_factor` comparison, do not loosen the tolerance — instead diff `config.to_xtr_estimator_settings_dict()` against the resolved dict that `scripts/run_it_tv_conditions.py` produced for `5us` in Task 2, field by field, to find the discrepancy (most likely a masking or plot-section field that didn't carry over).

- [ ] **Step 5: Commit**

```bash
git add src/tv_extrapolation/pipeline.py datasets/5us.yaml tests/test_pipeline_5us.py
git commit -m "$(cat <<'EOF'
Add unified pipeline.run(): config -> diffmap -> estimate -> extrapolated map

Single function replaces the run_it_tv_conditions.py +
generate_it_tv_extrapolated_maps.py two-script hand-off (the second
script read the first script's CSV output to find the chi value,
which is why a nan estimate silently produced no extrapolated map).
Verified against a fresh 5us baseline.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Implement `cli.py`

**Files:**
- Create: `src/tv_extrapolation/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Implement `cli.py`**

```python
# src/tv_extrapolation/cli.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import DatasetConfig
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tv-extrapolate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the it_tv pipeline for one or more dataset configs")
    run_parser.add_argument("configs", nargs="+", type=Path)
    run_parser.add_argument("--summary", type=Path, default=Path("results/it_tv_pipeline/summary.csv"))

    args = parser.parse_args(argv)

    if args.command == "run":
        rows = []
        for config_path in args.configs:
            config = DatasetConfig.from_yaml(config_path)
            result = run(config)
            rows.append(result.as_row())
            print(f"{result.condition}: {result.status}: chi={result.chi}, factor={result.extrapolation_factor}")

        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.summary}")
        return 0

    return 1
```

- [ ] **Step 2: Write a test that exercises the CLI end to end on 5us**

```python
# tests/test_cli.py
import csv
from pathlib import Path

from tv_extrapolation.cli import main

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_run_writes_summary_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    summary_path = tmp_path / "summary.csv"

    exit_code = main(["run", "datasets/5us.yaml", "--summary", str(summary_path)])

    assert exit_code == 0
    with summary_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["condition"] == "5us"
    assert rows[0]["status"] == "ok"
```

- [ ] **Step 3: Run it**

Run: `/home/dmitrii/miniforge3/envs/tv-extrapolation/bin/python -m pytest tests/test_cli.py -v`
Expected: 1 passed.

- [ ] **Step 4: Verify the console script entry point also works**

Run: `cd /home/dmitrii/projects/tv_extrapolation && /home/dmitrii/miniforge3/envs/tv-extrapolation/bin/tv-extrapolate run datasets/5us.yaml`
Expected: prints `5us: ok: chi=..., factor=...` then `Wrote results/it_tv_pipeline/summary.csv`.

- [ ] **Step 5: Commit**

```bash
git add src/tv_extrapolation/cli.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
Add tv-extrapolate CLI entry point

`tv-extrapolate run datasets/5us.yaml datasets/10ns.yaml` runs the
unified pipeline for one or more datasets and writes one merged
summary.csv, replacing the two separate script invocations.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Write `datasets/10ns.yaml` (delegated to codex)

`datasets/5us.yaml` already exists from Task 5. This task adds the second dataset.

- [ ] **Step 1: Dispatch the codex prompt**

```bash
cat > /tmp/codex_task7_prompt.txt <<'EOF'
Working directory: /home/dmitrii/projects/tv_extrapolation

Read configs/xtr_it_tv_xfel/10ns.yaml and datasets/5us.yaml (the new
schema). Create datasets/10ns.yaml in the same schema as
datasets/5us.yaml, translating the values from
configs/xtr_it_tv_xfel/10ns.yaml:

- name: 10ns
- dark_mtz / triggered_mtz / pdb_dark: copy the paths from
  configs/xtr_it_tv_xfel/10ns.yaml's input_files section
  (map_dark, map_triggered, pdb_dark) verbatim
- resolution_limit: copy from general.high_resolution_limit (2.1)
- columns.dark and columns.triggered: kind: intensity,
  amplitude_or_intensity: I, sigma: SIGI (matching
  columns_dark_ints/columns_triggered_ints in the source file)
- rewrite_pdb_cell: false
- estimation: {} (the source file's plot section has no
  solvent_density/std_cutoff override, so leave this empty)
- masking: {} (the source file has no masking section, so leave
  this empty)
- output_dir: results/it_tv_pipeline

Do not modify datasets/5us.yaml or any other file.
EOF
codex exec --full-auto "$(cat /tmp/codex_task7_prompt.txt)"
```
Expected: codex reports creating `datasets/10ns.yaml`.

- [ ] **Step 2: Review codex's output against the expected content**

```bash
cat datasets/10ns.yaml
```
Expected to match exactly:
```yaml
name: 10ns
dark_mtz: initial/10ns/ground.mtz
triggered_mtz: initial/10ns/10ns.mtz
pdb_dark: initial/10ns/OLPVR1_eufel_newsg_dark_refine_011.pdb
resolution_limit: 2.1
columns:
  dark: {kind: intensity, amplitude_or_intensity: I, sigma: SIGI}
  triggered: {kind: intensity, amplitude_or_intensity: I, sigma: SIGI}
rewrite_pdb_cell: false
estimation: {}
masking: {}
output_dir: results/it_tv_pipeline
```
If codex used a different (but equivalent) YAML flow style, that's fine — what matters is the parsed values match. Verify with:
```bash
/home/dmitrii/miniforge3/envs/tv-extrapolation/bin/python -c "
from tv_extrapolation.config import DatasetConfig
c = DatasetConfig.from_yaml('datasets/10ns.yaml')
assert str(c.dark_mtz) == 'initial/10ns/ground.mtz'
assert str(c.triggered_mtz) == 'initial/10ns/10ns.mtz'
assert str(c.pdb_dark) == 'initial/10ns/OLPVR1_eufel_newsg_dark_refine_011.pdb'
assert c.resolution_limit == 2.1
assert c.columns['dark'].kind == 'intensity'
assert c.estimation == {}
assert c.masking == {}
print('ok')
"
```
Expected: `ok`. If it doesn't load or any assertion fails, fix `datasets/10ns.yaml` by hand to match.

- [ ] **Step 3: Commit**

```bash
git add datasets/10ns.yaml
git commit -m "$(cat <<'EOF'
Add datasets/10ns.yaml in the unified pipeline schema

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Numeric regression check — the gate

**Files:**
- Create: `tests/test_pipeline_10ns.py`

- [ ] **Step 1: Write the 10ns regression test, mirroring `tests/test_pipeline_5us.py`**

```python
# tests/test_pipeline_10ns.py
import json
from pathlib import Path

import pytest

from tv_extrapolation.config import DatasetConfig
from tv_extrapolation.pipeline import run

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_10ns_matches_baseline():
    baseline = json.loads((REPO_ROOT / "tests/baseline_5us_10ns.json").read_text())["10ns"]
    config = DatasetConfig.from_yaml(REPO_ROOT / "datasets/10ns.yaml")
    result = run(config)

    assert result.status == "ok"
    assert result.chi == pytest.approx(baseline["chi"], rel=1e-6)
    assert result.extrapolation_factor == pytest.approx(baseline["extrapolation_factor"], rel=1e-6)
```

- [ ] **Step 2: Run the full test suite from the repo root**

Run: `cd /home/dmitrii/projects/tv_extrapolation && /home/dmitrii/miniforge3/envs/tv-extrapolation/bin/python -m pytest tests/ -v`
Expected: all tests pass, including both `test_pipeline_5us.py::test_5us_matches_baseline` and `test_pipeline_10ns.py::test_10ns_matches_baseline`.

This is the gate: if either regression test fails, stop and debug before proceeding to Task 9 or Task 10 — do not relax the tolerance or skip the test to move forward.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_10ns.py
git commit -m "$(cat <<'EOF'
Add 10ns regression test against fresh baseline

Both migrated datasets (5us, 10ns) now have a passing regression
test comparing the unified pipeline's chi/extrapolation_factor
against scripts/run_it_tv_conditions.py + generate_it_tv_extrapolated_maps.py
output captured in Task 2.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Additional config edge-case tests (delegated to codex)

**Files:**
- Modify: `tests/test_config.py`

- [ ] **Step 1: Dispatch the codex prompt**

```bash
cat > /tmp/codex_task9_prompt.txt <<'EOF'
Working directory: /home/dmitrii/projects/tv_extrapolation

Read src/tv_extrapolation/config.py and tests/test_config.py (the
existing tests already passing). Add new test functions to
tests/test_config.py (do not modify the existing two test functions)
covering these cases, using pytest.raises for the validation cases:

1. A DatasetConfig YAML missing the required "name" field raises a
   pydantic ValidationError when DatasetConfig.from_yaml() is called.
2. A DatasetConfig YAML where columns.dark.kind is "amplitude" but
   columns.triggered.kind is "intensity" (mixed kinds) is currently
   *not* rejected by DatasetConfig — write a test that documents this
   by asserting to_xtr_estimator_settings_dict() uses columns_dark
   (amplitude path) for input_files.columns_dark based on dark_col.kind,
   and columns_triggered_ints for triggered based on triggered_col.kind,
   i.e. confirm the function branches independently per dataset config's
   `dark_col.kind` only for the columns_are_ints flag (which is set from
   dark_col.kind alone) -- read the actual to_xtr_estimator_settings_dict
   implementation carefully and write a test that asserts its *actual*
   current behavior for this mixed-kind input, not desired behavior.
3. estimation and masking default to empty dicts when omitted from the
   YAML entirely (use the minimal valid payload from the existing
   test_amplitude_dataset_omits_masking_when_unset test as a base, but
   assert on config.estimation and config.masking directly, not just
   the translated payload).

Run pytest tests/test_config.py -v after writing the tests and show me
the output. All tests, including the existing two, must pass.
EOF
codex exec --full-auto "$(cat /tmp/codex_task9_prompt.txt)"
```

- [ ] **Step 2: Review codex's diff and run the tests myself**

```bash
git diff tests/test_config.py
/home/dmitrii/miniforge3/envs/tv-extrapolation/bin/python -m pytest tests/test_config.py -v
```
Verify: the two original test functions are unchanged (`git diff` should show only additions), all tests pass, and the new tests document real current behavior rather than asserting some desired-but-unimplemented validation. If codex added validation logic to `config.py` itself to make a test pass (e.g. rejecting mixed `kind`s), revert that — this task is test-only; any actual validation change belongs in a separate, deliberate change to `config.py` with its own commit.

- [ ] **Step 3: Commit**

```bash
git add tests/test_config.py
git commit -m "$(cat <<'EOF'
Add DatasetConfig edge-case tests

Covers missing required fields, mixed intensity/amplitude column
kinds, and default-empty estimation/masking overrides.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Document the split and update README

No script deletion in this pass (see the correction at the top of this plan) — `scripts/run_it_tv_conditions.py` and `scripts/generate_it_tv_extrapolated_maps.py` are still required for `10ms`, `30ms`, `esrf_5ms`, `esrf_5ms_2`, `esrf_75ms`, `trapping_1`, `trapping_2`, and `low_ph`, none of which are migrated yet.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a section to `README.md`**

Insert after the existing "Quick Start" section:

```markdown
## Unified it_tv pipeline (in progress)

`5us` and `10ns` now run through the installable `tv_extrapolation`
package instead of `scripts/run_it_tv_conditions.py` +
`scripts/generate_it_tv_extrapolated_maps.py`:

```bash
tv-extrapolate run datasets/5us.yaml datasets/10ns.yaml
```

This single command cleans inputs, computes the diffmap, estimates the
extrapolation factor, and writes the extrapolated map in one pass —
see `docs/superpowers/specs/2026-06-16-unified-it-tv-pipeline-design.md`
for the design and `docs/superpowers/plans/2026-06-16-unified-it-tv-pipeline.md`
for migration status.

The other XFEL/OLPVR1 conditions (`10ms`, `30ms`, `esrf_5ms`,
`esrf_5ms_2`, `esrf_75ms`, `trapping_1`, `trapping_2`, `low_ph`) and all
OCP/OCP-CAN/ESRF/firstprocessing datasets still go through
`scripts/run_it_tv_conditions.py` + `scripts/generate_it_tv_extrapolated_maps.py`
and the various `scripts/prepare_*_it_tv_inputs.py` scripts, pending
their own migration to the `datasets/*.yaml` schema.
```

- [ ] **Step 2: Verify the README renders sensibly**

Run: `cat README.md` and read through it — confirm the new section doesn't contradict the existing "Quick Start" / "Layout" sections (which describe the Xtrapol8 batch, untouched by this work).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
Document unified it_tv pipeline for 5us/10ns in README

Clarifies that only 5us/10ns are migrated so far; the other 8 XFEL
conditions and OCP/CAN/ESRF/firstprocessing datasets still depend on
the existing scripts/ pipeline.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## After this plan

Follow-up work, each its own future plan: migrate the remaining 8 `configs/xtr_it_tv_xfel/*.yaml` conditions to `datasets/*.yaml` (mechanical, same pattern as Task 7, every one delegable to codex with this plan's Task 7 as the template); migrate OCP/OCP-CAN/ESRF/firstprocessing (these need the `rewrite_pdb_cell` and finite-filtering logic this plan's schema reserved fields for but did not implement, since 5us/10ns didn't need them); only once every `configs/xtr_it_tv_xfel/*.yaml` condition is migrated, retire `scripts/run_it_tv_conditions.py` and `scripts/generate_it_tv_extrapolated_maps.py`; revisit the `can_laser26` chi=nan investigation using the new pipeline's per-dataset `estimation` overrides.
