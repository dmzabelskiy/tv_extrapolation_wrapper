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
