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
