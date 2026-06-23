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
