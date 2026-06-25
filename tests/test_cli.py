import csv
from pathlib import Path
from unittest.mock import patch, MagicMock

from tv_extrapolation.cli import main, _collect_configs
from tv_extrapolation.occupancy_scan import ScanResult, ScanPoint

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


def test_collect_configs_expands_directory(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "a.yaml").write_text("name: a\n")
    (sub / "b.yaml").write_text("name: b\n")
    (tmp_path / "not_yaml.txt").write_text("ignore\n")
    result = _collect_configs([tmp_path])
    assert sorted(p.name for p in result) == ["a.yaml", "b.yaml"]


def test_collect_configs_passes_files_through(tmp_path):
    f = tmp_path / "ds.yaml"
    f.write_text("name: x\n")
    result = _collect_configs([f])
    assert result == [f]
