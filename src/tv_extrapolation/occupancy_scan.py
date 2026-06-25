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
    cycles: int = 1,
) -> tuple[Path, bool]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "phenix_refine.log"
    cmd = [
        phenix_bin,
        str(model_pdb),
        str(mtz),
        *[str(c) for c in cif_files],
        f"output.prefix={out_dir / 'ref'}",
        "output.overwrite=True",
        f"strategy={strategy}",
        f"main.number_of_macro_cycles={cycles}",
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
    strategy: str = "individual_adp",
    cycles: int = 1,
) -> ScanResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    header, gmap = read_pdb_into_resmap(ground_pdb)
    _, emap = read_pdb_into_resmap(extrap_pdb)

    points: list[ScanPoint] = []
    for x in x_grid:
        x_tag = f"x{int(round(x * 100)):03d}"
        mixed_pdb = out_dir / f"mixed_{x_tag}.pdb"
        refine_dir = out_dir / f"refine_{x_tag}"

        atom_lines = build_mixed_model(gmap, emap, x, mode=mode)
        write_mixed_pdb(mixed_pdb, header, atom_lines)

        log_path, ok = run_phenix_adp_refine(
            mixed_pdb, triggered_mtz, refine_dir,
            cif_files=cif_files, cpus=cpus, phenix_bin=phenix_bin,
            strategy=strategy, cycles=cycles,
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
