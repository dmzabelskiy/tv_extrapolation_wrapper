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
    lines = build_mixed_model(gmap, emap, x=0.0, mode="occupancy")
    atoms = [parse_atom_line(l) for l in lines if l.startswith(("ATOM", "HETATM"))]
    atoms = [a for a in atoms if a is not None]
    # at x=0: ground contributes with occ=1*(1-0)=1.0, extrap with occ=1*0=0.0 (filtered)
    assert len(atoms) >= 1
    ground_atoms = [a for a in atoms if a["altloc"] in ("A", " ")]
    assert all(a["occ"] == pytest.approx(1.0) for a in ground_atoms)
    extrap_atoms = [a for a in atoms if a["altloc"] == "Q"]
    assert len(extrap_atoms) == 0, "At x=0, no extrap (Q-altloc) atoms should appear"


def test_build_mixed_model_x1_gives_extrap_occupancy(tmp_path):
    ground_pdb = _make_single_atom_pdb(tmp_path, "ground.pdb", 0.0, 0.0, 0.0, occ=1.0)
    extrap_pdb = _make_single_atom_pdb(tmp_path, "extrap.pdb", 10.0, 10.0, 10.0, occ=1.0)
    _, gmap = read_pdb_into_resmap(ground_pdb)
    _, emap = read_pdb_into_resmap(extrap_pdb)
    lines = build_mixed_model(gmap, emap, x=1.0, mode="occupancy")
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
