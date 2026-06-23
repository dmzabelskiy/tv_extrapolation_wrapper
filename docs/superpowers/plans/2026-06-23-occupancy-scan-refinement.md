# Occupancy Scan via Phenix Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tv-extrapolate refine-extrap` and `tv-extrapolate scan` CLI subcommands that use phenix.refine to independently verify the occupancy estimate produced by the it_tv pipeline.

**Architecture:** Two-step post-pipeline workflow. Step 1 (`refine-extrap`): refine the dark-state PDB against the pipeline's extrapolated MTZ (sites + ADP) to produce an excited-state model (`extrap.pdb`). Step 2 (`scan`): build mixed models `model(x) = x*extrap + (1-x)*ground` for an x-grid, run ADP-only phenix.refine on each against the raw triggered MTZ, extract Rwork/Rfree, and save results to CSV + plot. The PDB mixing logic lives in a new pure-Python module; the phenix wrapper and scan orchestration live in a second module; only `cli.py` is modified to wire subcommands.

**Tech Stack:** Python 3.12, pytest, matplotlib, subprocess (for phenix.refine). No new pip dependencies.

## Global Constraints

- Conda env: `tv-extrapolation` (Python 3.12). NOT `work`.
- `pytest` is the test runner. Run as: `conda run -n tv-extrapolation pytest tests/ -v`
- Phenix 2.0 binary lives at `/home/dmitrii/phenix-2.0-5867/phenix_bin/phenix.refine`. Tests must never call it — mock subprocess instead.
- All new modules go under `src/tv_extrapolation/`.
- Tests go in `tests/` (flat, no subdirs — matches existing layout).
- Commit after every task.

---

### Task 1: PDB mixing primitives

**Files:**
- Create: `src/tv_extrapolation/pdb_mix.py`
- Create: `tests/test_pdb_mix.py`

**Interfaces:**
- Produces:
  - `parse_atom_line(line: str) -> dict | None`
  - `write_pdb_atom_line(record, serial, name4, altloc, resname, chain, resseq, icode, x, y, z, occ, b, elem) -> str` (80 chars + `\n`)
  - `read_pdb_into_resmap(pdb_path: Path) -> tuple[list[str], dict]` — `(header_lines, ResMap)` where `ResMap = dict[tuple[str,int,str,str], dict[str, list[dict]]]` keyed by `(chain, resseq, icode, resname)` → `{altloc → [atom_dict]}`
  - `build_mixed_model(header, ground_map, extrap_map, x, *, mode="occupancy") -> list[str]` — atom lines only
  - `write_mixed_pdb(out_path: Path, header: list[str], atom_lines: list[str]) -> None`

- [ ] **Step 1: Write `tests/test_pdb_mix.py` (all tests, failing)**

```python
import textwrap
from pathlib import Path
import pytest
from tv_extrapolation.pdb_mix import (
    parse_atom_line,
    write_pdb_atom_line,
    read_pdb_into_resmap,
    build_mixed_model,
    write_mixed_pdb,
)

_ATOM_LINE = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C  \n"


def test_parse_atom_line_extracts_fields():
    a = parse_atom_line(_ATOM_LINE)
    assert a is not None
    assert a["chain"] == "A"
    assert a["resseq"] == 1
    assert a["resname"] == "ALA"
    assert a["name4"].strip() == "CA"
    assert a["x"] == pytest.approx(1.0)
    assert a["y"] == pytest.approx(2.0)
    assert a["z"] == pytest.approx(3.0)
    assert a["occ"] == pytest.approx(1.0)
    assert a["b"] == pytest.approx(10.0)


def test_parse_atom_line_returns_none_for_non_atom():
    assert parse_atom_line("REMARK  something\n") is None
    assert parse_atom_line("CRYST1  blah\n") is None


def test_write_pdb_atom_line_length():
    line = write_pdb_atom_line(
        "ATOM  ", 1, " CA ", " ", "ALA", "A", 1, " ",
        1.0, 2.0, 3.0, 1.0, 10.0, " C"
    )
    assert line.endswith("\n")
    assert len(line.rstrip("\n")) == 80


def test_read_pdb_into_resmap(tmp_path):
    pdb = tmp_path / "test.pdb"
    pdb.write_text(textwrap.dedent("""\
        CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1
        ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C  
        ATOM      2  CB  ALA A   1       2.000   3.000   4.000  1.00 12.00           C  
        ATOM      3  CA  GLY A   2       5.000   6.000   7.000  1.00  8.00           C  
        END
    """))
    header, resmap = read_pdb_into_resmap(pdb)
    assert any("CRYST1" in h for h in header)
    assert ("A", 1, " ", "ALA") in resmap
    assert ("A", 2, " ", "GLY") in resmap
    # ALA residue has two atoms in blank altloc
    assert len(resmap[("A", 1, " ", "ALA")][" "]) == 2


def _make_single_atom_pdb(tmp_path, filename, x, y, z, occ=1.0):
    pdb = tmp_path / filename
    line = write_pdb_atom_line(
        "ATOM  ", 1, " CA ", " ", "ALA", "A", 1, " ", x, y, z, occ, 10.0, " C"
    )
    pdb.write_text("CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1\n" + line + "END\n")
    return pdb


def test_build_mixed_model_x0_gives_ground_occupancy(tmp_path):
    ground_pdb = _make_single_atom_pdb(tmp_path, "ground.pdb", 0.0, 0.0, 0.0, occ=1.0)
    extrap_pdb = _make_single_atom_pdb(tmp_path, "extrap.pdb", 10.0, 10.0, 10.0, occ=1.0)
    _, gmap = read_pdb_into_resmap(ground_pdb)
    _, emap = read_pdb_into_resmap(extrap_pdb)
    lines = build_mixed_model([], gmap, emap, x=0.0, mode="occupancy")
    atoms = [parse_atom_line(l) for l in lines if l.startswith(("ATOM", "HETATM"))]
    atoms = [a for a in atoms if a is not None]
    # at x=0: ground contributes with occ=1*(1-0)=1.0, extrap with occ=1*0=0.0 (filtered)
    assert len(atoms) >= 1
    ground_atoms = [a for a in atoms if a["altloc"] in ("A", " ")]
    assert all(a["occ"] == pytest.approx(1.0) for a in ground_atoms)


def test_build_mixed_model_x1_gives_extrap_occupancy(tmp_path):
    ground_pdb = _make_single_atom_pdb(tmp_path, "ground.pdb", 0.0, 0.0, 0.0, occ=1.0)
    extrap_pdb = _make_single_atom_pdb(tmp_path, "extrap.pdb", 10.0, 10.0, 10.0, occ=1.0)
    _, gmap = read_pdb_into_resmap(ground_pdb)
    _, emap = read_pdb_into_resmap(extrap_pdb)
    lines = build_mixed_model([], gmap, emap, x=1.0, mode="occupancy")
    atoms = [parse_atom_line(l) for l in lines if l.startswith(("ATOM", "HETATM"))]
    atoms = [a for a in atoms if a is not None]
    extrap_atoms = [a for a in atoms if a["altloc"] == "Q"]
    assert len(extrap_atoms) >= 1
    assert all(a["occ"] == pytest.approx(1.0) for a in extrap_atoms)


def test_write_mixed_pdb_creates_file(tmp_path):
    out = tmp_path / "mixed.pdb"
    header = ["CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1"]
    atom_line = write_pdb_atom_line(
        "ATOM  ", 1, " CA ", " ", "ALA", "A", 1, " ", 0.0, 0.0, 0.0, 1.0, 10.0, " C"
    )
    write_mixed_pdb(out, header, [atom_line])
    text = out.read_text()
    assert "CRYST1" in text
    assert "ATOM" in text
    assert text.strip().endswith("END")
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
conda run -n tv-extrapolation pytest tests/test_pdb_mix.py -v 2>&1 | tail -20
```
Expected: `ImportError` or `ModuleNotFoundError` for `tv_extrapolation.pdb_mix`.

- [ ] **Step 3: Implement `src/tv_extrapolation/pdb_mix.py`**

```python
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

# (chain, resseq, icode, resname) → {altloc → [atom_dict]}
ResMap = dict[tuple[str, int, str, str], dict[str, list[dict]]]

_CALCULATED_PREFIXES = ("FC", "FWT", "DELFWT", "PHIC", "PHWT", "DELPHWT")


def parse_atom_line(line: str) -> dict | None:
    if not line.startswith(("ATOM  ", "HETATM")):
        return None
    try:
        name_raw = line[12:16]
        altloc = line[16:17].strip() or " "
        resname = line[17:20].strip()
        chain = line[21:22].strip() or " "
        resseq = int(line[22:26])
        icode = line[26:27].strip() or " "
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        occ_s = line[54:60].strip()
        b_s = line[60:66].strip()
        occ = float(occ_s) if occ_s else 1.0
        b = float(b_s) if b_s else 50.0
        elem_field = line[76:78].strip() if len(line) > 76 else ""
        elem = elem_field if elem_field else _guess_element(name_raw)
    except Exception:
        return None
    return {
        "record": line[0:6],
        "name4": _fix_name(name_raw),
        "altloc": altloc,
        "resname": resname,
        "chain": chain,
        "resseq": resseq,
        "icode": icode,
        "x": x, "y": y, "z": z,
        "occ": occ, "b": b,
        "element": elem,
    }


def _fix_name(name: str) -> str:
    n = (name or "").strip()
    return n[:4].ljust(4) if len(n) >= 4 else n.ljust(4)


def _guess_element(name4: str) -> str:
    s = (name4 or "").strip()
    if not s:
        return "  "
    if s[0].isdigit():
        e = s[1] if len(s) > 1 else " "
        if len(s) > 2 and s[2].isalpha() and s[2].islower():
            e += s[2]
        return e.upper().rjust(2)
    e = s[0]
    if len(s) > 1 and s[1].isalpha() and s[1].islower():
        e += s[1]
    return e.upper().rjust(2)


def write_pdb_atom_line(
    record: str, serial: int, name4: str, altloc: str,
    resname: str, chain: str, resseq: int, icode: str,
    x: float, y: float, z: float, occ: float, b: float, elem: str,
) -> str:
    record = record[:6].ljust(6)
    name4 = name4[:4].ljust(4)
    altloc = (altloc or " ")[:1]
    resname = resname[:3].rjust(3)
    chain = (chain or " ")[:1]
    resseq = int(resseq) % 10000
    icode = (icode or " ")[:1]
    elem = (elem.strip().upper() if elem else "").rjust(2)
    line = (
        f"{record}{serial:5d} {name4}{altloc}{resname} {chain}"
        f"{resseq:4d}{icode}   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occ:6.2f}{b:6.2f}"
    )
    line = line[:76].ljust(76)
    return line + elem + "  \n"


def read_pdb_into_resmap(pdb_path: Path) -> tuple[list[str], ResMap]:
    header: list[str] = []
    resmap: ResMap = defaultdict(lambda: defaultdict(list))
    with open(pdb_path) as fh:
        for line in fh:
            a = parse_atom_line(line)
            if a is None:
                header.append(line.rstrip("\n"))
            else:
                key = (a["chain"], a["resseq"], a["icode"], a["resname"])
                resmap[key][a["altloc"]].append(a)
    return header, dict(resmap)


def build_mixed_model(
    header: list[str],
    ground_map: ResMap,
    extrap_map: ResMap,
    x: float,
    *,
    mode: str = "occupancy",
) -> list[str]:
    out: list[str] = []
    serial = 1
    residue_keys = sorted(
        set(ground_map) | set(extrap_map),
        key=lambda k: (k[0], k[1], k[2], k[3]),
    )

    def emit(source_alts, src_labels, dest_alt, scale):
        nonlocal serial
        for label in src_labels:
            for a in source_alts.get(label, []):
                scaled_occ = scale * a["occ"]
                if scaled_occ < 0.001:
                    continue
                out.append(write_pdb_atom_line(
                    a["record"], serial, a["name4"], dest_alt,
                    a["resname"], a["chain"], a["resseq"], a["icode"],
                    a["x"], a["y"], a["z"], scaled_occ, a["b"], a["element"],
                ))
                serial += 1

    for key in residue_keys:
        g = ground_map.get(key, {})
        e = extrap_map.get(key, {})
        if mode == "coords":
            # single-conformer linear interpolation
            g_by_name = _best_by_name(g)
            e_by_name = _best_by_name(e)
            for name in sorted(set(g_by_name) | set(e_by_name)):
                ga, ea = g_by_name.get(name), e_by_name.get(name)
                if ga and ea:
                    cx = (1 - x) * ga["x"] + x * ea["x"]
                    cy = (1 - x) * ga["y"] + x * ea["y"]
                    cz = (1 - x) * ga["z"] + x * ea["z"]
                    b_out = (1 - x) * ga["b"] + x * ea["b"]
                    src = ga
                elif ga:
                    cx, cy, cz, b_out, src = ga["x"], ga["y"], ga["z"], ga["b"], ga
                else:
                    cx, cy, cz, b_out, src = ea["x"], ea["y"], ea["z"], ea["b"], ea  # type: ignore[assignment]
                chain, resseq, icode, resname = key
                out.append(write_pdb_atom_line(
                    src["record"], serial, src["name4"], " ",
                    resname, chain, resseq, icode,
                    cx, cy, cz, 1.0, b_out, src["element"],
                ))
                serial += 1
        else:
            emit(g, ["A", " "], "A", 1.0 - x)
            emit(g, ["B"],      "B", 1.0 - x)
            emit(e, ["A", " "], "Q", x)
            emit(e, ["B"],      "R", x)
    return out


def _best_by_name(alt_map: dict[str, list[dict]]) -> dict[str, dict]:
    by_name: dict[str, dict] = {}
    tmp: dict[str, dict[str, dict]] = {}
    for alt, atoms in alt_map.items():
        for a in atoms:
            tmp.setdefault(a["name4"].strip(), {})[alt] = a
    for name, adict in tmp.items():
        for pref in ("A", " ", "B"):
            if pref in adict:
                by_name[name] = adict[pref]
                break
        else:
            by_name[name] = next(iter(adict.values()))
    return by_name


def write_mixed_pdb(out_path: Path, header: list[str], atom_lines: list[str]) -> None:
    with open(out_path, "w") as fh:
        for h in header:
            if not h.startswith(("ATOM  ", "HETATM", "ANISOU", "TER", "END")):
                fh.write(h + "\n")
        for line in atom_lines:
            fh.write(line)
        fh.write("TER\nEND\n")
```

- [ ] **Step 4: Run tests — confirm all pass**

```bash
conda run -n tv-extrapolation pytest tests/test_pdb_mix.py -v 2>&1 | tail -20
```
Expected: all 8 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/tv_extrapolation/pdb_mix.py tests/test_pdb_mix.py
git commit -m "feat: add PDB mixing primitives for occupancy scan"
```

---

### Task 2: Phenix wrapper, scan orchestrator, and plot

**Files:**
- Create: `src/tv_extrapolation/occupancy_scan.py`
- Create: `tests/test_occupancy_scan.py`

**Interfaces:**
- Consumes: `read_pdb_into_resmap`, `build_mixed_model`, `write_mixed_pdb` from `pdb_mix`
- Produces:
  - `parse_refine_log_for_R(log_path: Path) -> tuple[float | None, float | None]`
  - `run_phenix_adp_refine(model_pdb, mtz, out_dir, *, cif_files=(), cpus=2, phenix_bin="phenix.refine", strategy="individual_adp") -> tuple[Path, bool]`
  - `ScanPoint(x, pdb_path, refine_dir, rwork, rfree, ok)`
  - `ScanResult(points, best, plot_path)`
  - `run_scan(ground_pdb, extrap_pdb, triggered_mtz, *, out_dir, x_grid, cif_files=(), cpus=2, phenix_bin="phenix.refine", mode="occupancy") -> ScanResult`
  - `plot_scan(result: ScanResult, out_path: Path) -> None`

- [ ] **Step 1: Write `tests/test_occupancy_scan.py` (all tests, failing)**

```python
from pathlib import Path
from unittest.mock import patch, MagicMock
import textwrap
import pytest
from tv_extrapolation.occupancy_scan import (
    parse_refine_log_for_R,
    run_phenix_adp_refine,
    run_scan,
    ScanResult,
    ScanPoint,
    plot_scan,
)
from tv_extrapolation.pdb_mix import write_pdb_atom_line, write_mixed_pdb


_PHENIX_LOG_OK = textwrap.dedent("""\
    ... lots of output ...
    Final R-work = 0.2134, R-free = 0.2561
    ... more output ...
""")

_PHENIX_LOG_FALLBACK = textwrap.dedent("""\
    R-work = 0.2200
    R-free = 0.2700
""")


def test_parse_refine_log_final_values(tmp_path):
    log = tmp_path / "refine.log"
    log.write_text(_PHENIX_LOG_OK)
    rwork, rfree = parse_refine_log_for_R(log)
    assert rwork == pytest.approx(0.2134)
    assert rfree == pytest.approx(0.2561)


def test_parse_refine_log_fallback(tmp_path):
    log = tmp_path / "refine.log"
    log.write_text(_PHENIX_LOG_FALLBACK)
    rwork, rfree = parse_refine_log_for_R(log)
    assert rwork == pytest.approx(0.2200)
    assert rfree == pytest.approx(0.2700)


def test_parse_refine_log_missing_file(tmp_path):
    rwork, rfree = parse_refine_log_for_R(tmp_path / "nonexistent.log")
    assert rwork is None
    assert rfree is None


def _write_minimal_pdb(path: Path) -> None:
    line = write_pdb_atom_line(
        "ATOM  ", 1, " CA ", " ", "ALA", "A", 1, " ", 0.0, 0.0, 0.0, 1.0, 10.0, " C"
    )
    write_mixed_pdb(path, ["CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1"], [line])


def test_run_phenix_adp_refine_calls_subprocess(tmp_path):
    model = tmp_path / "model.pdb"
    mtz = tmp_path / "data.mtz"
    model.touch(); mtz.touch()
    out_dir = tmp_path / "refine"
    log_content = _PHENIX_LOG_OK

    with patch("tv_extrapolation.occupancy_scan.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        # Write a fake log so the function can read it
        out_dir.mkdir()
        (out_dir / "phenix_refine.log").write_text(log_content)
        log_path, ok = run_phenix_adp_refine(model, mtz, out_dir, phenix_bin="phenix.refine")

    assert mock_run.called
    cmd = mock_run.call_args[0][0]
    assert "phenix.refine" in cmd[0]
    assert str(model) in cmd
    assert str(mtz) in cmd
    assert any("individual_adp" in c for c in cmd)


def test_run_scan_produces_scan_result(tmp_path):
    ground = tmp_path / "ground.pdb"
    extrap = tmp_path / "extrap.pdb"
    mtz = tmp_path / "triggered.mtz"
    _write_minimal_pdb(ground)
    _write_minimal_pdb(extrap)
    mtz.touch()
    out_dir = tmp_path / "scan"

    log_text = "Final R-work = 0.2000, R-free = 0.2500\n"

    def fake_refine(model_pdb, mtz_path, refine_dir, **kwargs):
        refine_dir.mkdir(parents=True, exist_ok=True)
        log = refine_dir / "phenix_refine.log"
        log.write_text(log_text)
        return log, True

    with patch("tv_extrapolation.occupancy_scan.run_phenix_adp_refine", side_effect=fake_refine):
        result = run_scan(
            ground, extrap, mtz,
            out_dir=out_dir,
            x_grid=[0.0, 0.1, 0.2],
        )

    assert isinstance(result, ScanResult)
    assert len(result.points) == 3
    assert all(p.rwork == pytest.approx(0.2000) for p in result.points)
    assert result.best is not None
    # all rfree equal, so best is the first
    assert result.best.x in [0.0, 0.1, 0.2]


def test_run_scan_best_is_lowest_rfree(tmp_path):
    ground = tmp_path / "ground.pdb"
    extrap = tmp_path / "extrap.pdb"
    mtz = tmp_path / "triggered.mtz"
    _write_minimal_pdb(ground)
    _write_minimal_pdb(extrap)
    mtz.touch()
    out_dir = tmp_path / "scan"

    rfree_by_x = {0.0: 0.30, 0.1: 0.22, 0.2: 0.28}

    def fake_refine(model_pdb, mtz_path, refine_dir, **kwargs):
        refine_dir.mkdir(parents=True, exist_ok=True)
        # Infer x from the directory name (e.g. scan_x010 → 0.10)
        stem = refine_dir.name  # e.g. "refine_x010"
        x_int = int(stem.split("x")[1]) if "x" in stem else 0
        x = x_int / 100.0
        rfree = rfree_by_x.get(round(x, 2), 0.35)
        log = refine_dir / "phenix_refine.log"
        log.write_text(f"Final R-work = 0.1800, R-free = {rfree:.4f}\n")
        return log, True

    with patch("tv_extrapolation.occupancy_scan.run_phenix_adp_refine", side_effect=fake_refine):
        result = run_scan(
            ground, extrap, mtz,
            out_dir=out_dir,
            x_grid=[0.0, 0.1, 0.2],
        )

    assert result.best is not None
    assert result.best.x == pytest.approx(0.1)
    assert result.best.rfree == pytest.approx(0.22)


def test_plot_scan_writes_file(tmp_path):
    ground = tmp_path / "ground.pdb"
    extrap = tmp_path / "extrap.pdb"
    mtz = tmp_path / "triggered.mtz"
    _write_minimal_pdb(ground)
    _write_minimal_pdb(extrap)
    mtz.touch()

    points = [
        ScanPoint(x=0.0, pdb_path=ground, refine_dir=tmp_path, rwork=0.25, rfree=0.30, ok=True),
        ScanPoint(x=0.1, pdb_path=extrap, refine_dir=tmp_path, rwork=0.22, rfree=0.27, ok=True),
    ]
    best = points[1]
    result = ScanResult(points=points, best=best, plot_path=None)

    out = tmp_path / "scan_plot.png"
    plot_scan(result, out)
    assert out.exists()
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
conda run -n tv-extrapolation pytest tests/test_occupancy_scan.py -v 2>&1 | tail -20
```
Expected: `ImportError` for `tv_extrapolation.occupancy_scan`.

- [ ] **Step 3: Implement `src/tv_extrapolation/occupancy_scan.py`**

```python
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pdb_mix import read_pdb_into_resmap, build_mixed_model, write_mixed_pdb


@dataclass
class ScanPoint:
    x: float
    pdb_path: Path
    refine_dir: Path
    rwork: float | None
    rfree: float | None
    ok: bool


@dataclass
class ScanResult:
    points: list[ScanPoint]
    best: ScanPoint | None
    plot_path: Path | None


def parse_refine_log_for_R(log_path: Path) -> tuple[float | None, float | None]:
    if not log_path.is_file():
        return None, None
    text = log_path.read_text()
    m = re.search(r"Final\s+R-work\s*=\s*([0-9.]+),?\s*R-free\s*=\s*([0-9.]+)", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"Start\s+R-work\s*=\s*([0-9.]+),?\s*R-free\s*=\s*([0-9.]+)", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m1 = re.search(r"R-work\s*=\s*([0-9.]+)", text)
    m2 = re.search(r"R-free\s*=\s*([0-9.]+)", text)
    return (float(m1.group(1)) if m1 else None), (float(m2.group(1)) if m2 else None)


def run_phenix_adp_refine(
    model_pdb: Path,
    mtz: Path,
    out_dir: Path,
    *,
    cif_files: list[Path] = (),
    cpus: int = 2,
    phenix_bin: str = "phenix.refine",
    strategy: str = "individual_adp",
) -> tuple[Path, bool]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "phenix_refine.log"
    cmd = [
        phenix_bin,
        str(model_pdb),
        str(mtz),
        *[str(c) for c in cif_files],
        f"output.prefix={out_dir / 'ref'}",
        f"strategy={strategy}",
        "main.number_of_macro_cycles=1",
        "hydrogens.refine=none",
    ]
    if cpus > 1:
        cmd.append(f"refinement.main.nproc={cpus}")
    with open(log_path, "wb") as fh:
        try:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
        except Exception as exc:
            fh.write(f"\nERROR: {exc}".encode())
            return log_path, False
    text = log_path.read_text(errors="replace")
    if "Unrecognized PHIL" in text or "Sorry:" in text:
        return log_path, False
    return log_path, True


def run_scan(
    ground_pdb: Path,
    extrap_pdb: Path,
    triggered_mtz: Path,
    *,
    out_dir: Path,
    x_grid: list[float],
    cif_files: list[Path] = (),
    cpus: int = 2,
    phenix_bin: str = "phenix.refine",
    mode: str = "occupancy",
) -> ScanResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, gmap = read_pdb_into_resmap(ground_pdb)
    _, emap = read_pdb_into_resmap(extrap_pdb)
    header, _ = read_pdb_into_resmap(ground_pdb)

    points: list[ScanPoint] = []
    for x in x_grid:
        x_tag = f"x{int(round(x * 100)):03d}"
        mixed_pdb = out_dir / f"mixed_{x_tag}.pdb"
        refine_dir = out_dir / f"refine_{x_tag}"

        atom_lines = build_mixed_model(header, gmap, emap, x, mode=mode)
        write_mixed_pdb(mixed_pdb, header, atom_lines)

        log_path, ok = run_phenix_adp_refine(
            mixed_pdb, triggered_mtz, refine_dir,
            cif_files=cif_files, cpus=cpus, phenix_bin=phenix_bin,
        )
        rwork, rfree = parse_refine_log_for_R(log_path)
        print(f"  x={x:.3f}  Rwork={rwork}  Rfree={rfree}  ok={ok}")
        points.append(ScanPoint(
            x=x, pdb_path=mixed_pdb, refine_dir=refine_dir,
            rwork=rwork, rfree=rfree, ok=ok,
        ))

    best = _best_point(points)
    result = ScanResult(points=points, best=best, plot_path=None)

    plot_path = out_dir / "occupancy_scan.png"
    plot_scan(result, plot_path)
    result.plot_path = plot_path

    _write_csv(result, out_dir / "scan_results.csv")
    return result


def _best_point(points: list[ScanPoint]) -> ScanPoint | None:
    scored = [(p.rfree if p.rfree is not None else p.rwork, p) for p in points]
    scored = [(s, p) for s, p in scored if s is not None]
    if not scored:
        return None
    return min(scored, key=lambda t: t[0])[1]


def _write_csv(result: ScanResult, path: Path) -> None:
    import csv
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["x", "rwork", "rfree", "ok", "pdb_path"])
        writer.writeheader()
        for p in result.points:
            writer.writerow({
                "x": p.x, "rwork": p.rwork, "rfree": p.rfree,
                "ok": p.ok, "pdb_path": str(p.pdb_path),
            })


def plot_scan(result: ScanResult, out_path: Path) -> None:
    xs = [p.x for p in result.points]
    rworks = [p.rwork for p in result.points]
    rfrees = [p.rfree for p in result.points]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.plot(xs, rworks, "o-", linewidth=2, markersize=8)
    ax1.set_ylabel("Rwork")
    ax1.set_title("Rwork and Rfree vs Occupancy x")
    ax1.grid(True)
    ax2.plot(xs, rfrees, "o-", linewidth=2, markersize=8, color="orange")
    ax2.set_xlabel("Occupancy x")
    ax2.set_ylabel("Rfree")
    ax2.grid(True)
    if result.best is not None:
        ax2.axvline(result.best.x, color="red", linestyle="--", label=f"best x={result.best.x:.3f}")
        ax2.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run tests — confirm all pass**

```bash
conda run -n tv-extrapolation pytest tests/test_occupancy_scan.py -v 2>&1 | tail -25
```
Expected: all 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/tv_extrapolation/occupancy_scan.py tests/test_occupancy_scan.py
git commit -m "feat: add occupancy scan orchestrator and phenix.refine wrapper"
```

---

### Task 3: CLI subcommands `refine-extrap` and `scan`

**Files:**
- Modify: `src/tv_extrapolation/cli.py`

**Interfaces:**
- Consumes: `run_phenix_adp_refine` from `occupancy_scan`; `run_scan` from `occupancy_scan`

CLI contract:

```
# Generate excited-state model by refining dark PDB against extrapolated MTZ:
tv-extrapolate refine-extrap dark.pdb extrap_ready.mtz \
    --out-dir results/ocp/ech_laser_30min/refine_extrap \
    [--cif ECH.cif] [--cpus 4] [--phenix-bin /path/to/phenix.refine] \
    [--strategy individual_sites+individual_adp]

# Scan occupancy x-grid using mixed models vs triggered MTZ:
tv-extrapolate scan ground.pdb extrap.pdb triggered.mtz \
    --out-dir results/ocp/ech_laser_30min/occupancy_scan \
    [--x-grid 0.0 0.05 0.1 0.15 0.2 0.25 0.3] \
    [--cif ECH.cif] [--cpus 4] [--phenix-bin /path/to/phenix.refine] \
    [--mode occupancy|coords]
```

- [ ] **Step 1: Write tests for both new subcommands (in `tests/test_cli.py`, append)**

Open `tests/test_cli.py` and append:

```python
from unittest.mock import patch, MagicMock
from pathlib import Path
from tv_extrapolation.occupancy_scan import ScanResult, ScanPoint


def _write_minimal_pdb_cli(path: Path) -> None:
    from tv_extrapolation.pdb_mix import write_pdb_atom_line, write_mixed_pdb
    line = write_pdb_atom_line(
        "ATOM  ", 1, " CA ", " ", "ALA", "A", 1, " ", 0.0, 0.0, 0.0, 1.0, 10.0, " C"
    )
    write_mixed_pdb(
        path,
        ["CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1"],
        [line],
    )


def test_refine_extrap_subcommand_calls_wrapper(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dark = tmp_path / "dark.pdb"
    mtz = tmp_path / "extrap.mtz"
    _write_minimal_pdb_cli(dark)
    mtz.touch()
    out_dir = tmp_path / "refine_out"

    fake_log = out_dir / "phenix_refine.log"

    def fake_refine(model_pdb, mtz_path, out_dir_arg, **kwargs):
        out_dir_arg.mkdir(parents=True, exist_ok=True)
        fake_log_path = out_dir_arg / "phenix_refine.log"
        fake_log_path.write_text("Final R-work = 0.2000, R-free = 0.2500\n")
        return fake_log_path, True

    with patch("tv_extrapolation.cli.run_phenix_adp_refine", side_effect=fake_refine) as mock_fn:
        exit_code = main([
            "refine-extrap", str(dark), str(mtz),
            "--out-dir", str(out_dir),
        ])

    assert exit_code == 0
    assert mock_fn.called


def test_scan_subcommand_calls_run_scan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ground = tmp_path / "ground.pdb"
    extrap = tmp_path / "extrap.pdb"
    mtz = tmp_path / "triggered.mtz"
    _write_minimal_pdb_cli(ground)
    _write_minimal_pdb_cli(extrap)
    mtz.touch()
    out_dir = tmp_path / "scan_out"

    fake_result = ScanResult(
        points=[ScanPoint(x=0.1, pdb_path=ground, refine_dir=out_dir, rwork=0.20, rfree=0.25, ok=True)],
        best=None,
        plot_path=None,
    )

    with patch("tv_extrapolation.cli.run_scan", return_value=fake_result) as mock_fn:
        exit_code = main([
            "scan", str(ground), str(extrap), str(mtz),
            "--out-dir", str(out_dir),
            "--x-grid", "0.0", "0.1", "0.2",
        ])

    assert exit_code == 0
    assert mock_fn.called
    kwargs = mock_fn.call_args
    assert kwargs[1]["x_grid"] == [0.0, 0.1, 0.2]
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
conda run -n tv-extrapolation pytest tests/test_cli.py::test_refine_extrap_subcommand_calls_wrapper tests/test_cli.py::test_scan_subcommand_calls_run_scan -v 2>&1 | tail -15
```
Expected: FAIL — `argument command: invalid choice: 'refine-extrap'`.

- [ ] **Step 3: Add the two subcommands to `src/tv_extrapolation/cli.py`**

Add imports at the top (after existing imports):

```python
from .occupancy_scan import run_phenix_adp_refine, run_scan
```

Add subparser definitions inside `main()`, before `args = parser.parse_args(argv)`:

```python
    # --- refine-extrap subcommand ---
    re_parser = subparsers.add_parser(
        "refine-extrap",
        help="Refine dark PDB against extrapolated MTZ to produce an excited-state model.",
    )
    re_parser.add_argument("dark_pdb", type=Path, help="Dark-state PDB (ground)")
    re_parser.add_argument("extrap_mtz", type=Path, help="Phenix-ready extrapolated MTZ")
    re_parser.add_argument("--out-dir", type=Path, required=True, dest="out_dir",
                           help="Output directory for refinement files")
    re_parser.add_argument("--cif", type=Path, action="append", default=[], dest="cif_files",
                           metavar="FILE", help="Ligand CIF restraint file (repeatable)")
    re_parser.add_argument("--cpus", type=int, default=2)
    re_parser.add_argument("--phenix-bin", default="phenix.refine", dest="phenix_bin",
                           help="Path to phenix.refine binary")
    re_parser.add_argument("--strategy", default="individual_sites+individual_adp",
                           help="Phenix refinement strategy (default: individual_sites+individual_adp)")

    # --- scan subcommand ---
    sc_parser = subparsers.add_parser(
        "scan",
        help="Scan occupancy x-grid: refine mixed models vs triggered MTZ, plot Rfree vs x.",
    )
    sc_parser.add_argument("ground_pdb", type=Path, help="Ground-state (dark) PDB")
    sc_parser.add_argument("extrap_pdb", type=Path, help="Excited-state PDB (from refine-extrap)")
    sc_parser.add_argument("triggered_mtz", type=Path, help="Raw triggered MTZ")
    sc_parser.add_argument("--out-dir", type=Path, required=True, dest="out_dir",
                           help="Output directory")
    sc_parser.add_argument("--x-grid", nargs="+", type=float, dest="x_grid",
                           default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
                           help="Occupancy x values to scan")
    sc_parser.add_argument("--cif", type=Path, action="append", default=[], dest="cif_files",
                           metavar="FILE")
    sc_parser.add_argument("--cpus", type=int, default=2)
    sc_parser.add_argument("--phenix-bin", default="phenix.refine", dest="phenix_bin")
    sc_parser.add_argument("--mode", choices=["occupancy", "coords"], default="occupancy")
```

Add handling inside the `if args.command == "run":` block (add as `elif` branches after the run block):

```python
    elif args.command == "refine-extrap":
        log_path, ok = run_phenix_adp_refine(
            args.dark_pdb, args.extrap_mtz, args.out_dir,
            cif_files=args.cif_files,
            cpus=args.cpus,
            phenix_bin=args.phenix_bin,
            strategy=args.strategy,
        )
        status = "ok" if ok else "FAILED"
        print(f"refine-extrap {status}: log at {log_path}")
        return 0 if ok else 1

    elif args.command == "scan":
        result = run_scan(
            args.ground_pdb, args.extrap_pdb, args.triggered_mtz,
            out_dir=args.out_dir,
            x_grid=args.x_grid,
            cif_files=args.cif_files,
            cpus=args.cpus,
            phenix_bin=args.phenix_bin,
            mode=args.mode,
        )
        if result.best is not None:
            print(f"Best x={result.best.x:.3f}  Rfree={result.best.rfree}  Rwork={result.best.rwork}")
        if result.plot_path:
            print(f"Plot: {result.plot_path}")
        print(f"CSV:  {args.out_dir / 'scan_results.csv'}")
        return 0
```

- [ ] **Step 4: Run all tests — confirm everything passes**

```bash
conda run -n tv-extrapolation pytest tests/ -v 2>&1 | tail -30
```
Expected: all tests PASSED (existing pipeline tests + all new tests).

- [ ] **Step 5: Smoke-test the CLI help**

```bash
conda run -n tv-extrapolation tv-extrapolate refine-extrap --help
conda run -n tv-extrapolation tv-extrapolate scan --help
```
Expected: both print usage without error.

- [ ] **Step 6: Commit**

```bash
git add src/tv_extrapolation/cli.py tests/test_cli.py
git commit -m "feat: add refine-extrap and scan CLI subcommands for occupancy verification"
```

---

## Self-Review

**Spec coverage:**
- ✅ Port `build_mixed_model` / PDB mixing from `Refinement_laser_implementation/scan_x_and_merge.py` → Task 1
- ✅ `run_phenix_adp_refine` with configurable binary, CIF, strategy → Task 2
- ✅ `parse_refine_log_for_R` (Phenix 2.0 format) → Task 2
- ✅ x-grid scan producing Rfree vs x → Task 2 `run_scan`
- ✅ Plot saved to file → Task 2 `plot_scan`
- ✅ CSV summary → Task 2 `_write_csv`
- ✅ `refine-extrap` CLI for generating `extrap.pdb` → Task 3
- ✅ `scan` CLI for the full x-scan → Task 3
- ✅ Phenix binary is configurable (`--phenix-bin`) → Task 3
- ✅ CIF restraint files configurable (`--cif`) → Task 3
- ✅ `--mode occupancy|coords` → Task 3

**Placeholder scan:** None found.

**Type consistency:**
- `run_phenix_adp_refine` signature used identically in Task 2 (implementation) and Task 3 (import + CLI call) ✅
- `ScanResult` / `ScanPoint` defined in Task 2, imported in Task 3 CLI test ✅
- `read_pdb_into_resmap` returns `(list[str], ResMap)` — used as `header, resmap = ...` everywhere ✅

**Workflow for OCP datasets (how to actually use this after implementation):**

```bash
cd /home/dmitrii/projects/tv_extrapolation
conda activate tv-extrapolation

# Step 1: generate excited-state model
tv-extrapolate refine-extrap \
  results/ocp/ech_laser_30min/OCP_ECH_dark_cell_corrected.pdb \
  results/ocp/ech_laser_30min/phenix_ready/ech_laser_30min_it_tv_extrapolated_phenix_ready.mtz \
  --out-dir results/ocp/ech_laser_30min/refine_extrap \
  --cif initial/ocp_ech/ECH.cif \
  --strategy "individual_sites+individual_adp" \
  --phenix-bin /home/dmitrii/phenix-2.0-5867/phenix_bin/phenix.refine

# Step 2: scan occupancy
tv-extrapolate scan \
  results/ocp/ech_laser_30min/OCP_ECH_dark_cell_corrected.pdb \
  results/ocp/ech_laser_30min/refine_extrap/ref_001.pdb \
  initial/ocp_ech/ech_laser_30min_triggered.mtz \
  --out-dir results/ocp/ech_laser_30min/occupancy_scan \
  --x-grid 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 \
  --cif initial/ocp_ech/ECH.cif \
  --phenix-bin /home/dmitrii/phenix-2.0-5867/phenix_bin/phenix.refine
```

Note: after `refine-extrap`, the excited-state PDB will be named `ref_XXX.pdb` in the out-dir (standard Phenix naming). Check the exact filename before running `scan`.
