"""PDB parsing utilities — receptor-only vs label extraction."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Sequence


_SKIP_LIGANDS = frozenset(
    {
        "HOH",
        "WAT",
        "DOD",
        "NA",
        "CL",
        "MG",
        "ZN",
        "CA",
        "K",
        "SO4",
        "PO4",
        "GOL",
        "EDO",
        "PEG",
        "ACT",
        "ACE",
    }
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pdb_atoms(text: str) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        element = line[76:78].strip() if len(line) >= 78 else ""
        if not element:
            element = line[12:16].strip()[:1]
        atoms.append(
            {
                "record": line[:6].strip(),
                "serial": int(line[6:11].strip() or 0),
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21].strip(),
                "resseq": int(line[22:26].strip() or 0),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "element": element.upper() or "C",
                "raw_line": line.rstrip(),
            }
        )
    return atoms


def write_receptor_only_pdb(atoms: list[dict[str, Any]], dest: Path, *, chain: str | None = None) -> None:
    """Write ATOM records only (no HETATM) — Foliation/baseline input contract."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "REMARK pocket_bench receptor-only input",
        "REMARK ligand HETATM stripped; not used as Foliation features",
    ]
    n = 0
    for a in atoms:
        if a["record"] != "ATOM":
            continue
        if chain is not None and a["chain"] != chain:
            continue
        if a["element"] == "H":
            continue
        n += 1
        raw = a.get("raw_line")
        if raw and raw.startswith("ATOM"):
            lines.append(f"ATOM  {n:5d}{raw[11:]}")
        else:
            lines.append(
                f"ATOM  {n:5d} {a['name']:>4s} {a['resname']:>3s} {a['chain']}{a['resseq']:4d}    "
                f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}  1.00 20.00           {a['element']:>2s}"
            )
    lines.append("END")
    if n < 50:
        raise ValueError(f"receptor too small ({n} heavy atoms) for {dest}")
    dest.write_text("\n".join(lines) + "\n")


def detect_primary_ligand(atoms: list[dict[str, Any]], preferred: str | None = None) -> str:
    if preferred:
        return preferred
    counts: dict[str, int] = {}
    for a in atoms:
        if a["record"] != "HETATM":
            continue
        rn = a["resname"]
        if rn in _SKIP_LIGANDS or a["element"] == "H":
            continue
        counts[rn] = counts.get(rn, 0) + 1
    if not counts:
        raise ValueError("no ligand HETATM for labels")
    return max(counts, key=counts.get)


def ligand_heavy_coords(atoms: list[dict[str, Any]], resname: str) -> list[list[float]]:
    coords = []
    for a in atoms:
        if a["record"] != "HETATM" or a["resname"] != resname:
            continue
        if a["element"] == "H":
            continue
        coords.append([a["x"], a["y"], a["z"]])
    if len(coords) < 3:
        raise ValueError(f"ligand {resname} has <3 heavy atoms")
    return coords


def ligand_centroid(coords: list[list[float]]) -> list[float]:
    n = len(coords)
    return [sum(c[i] for c in coords) / n for i in range(3)]


def binding_residues(
    atoms: list[dict[str, Any]],
    ligand_coords: list[list[float]],
    *,
    cutoff_a: float = 4.5,
    chain: str | None = None,
) -> list[str]:
    """Residues with any heavy atom within cutoff of any ligand heavy atom (labels only)."""
    hits: set[str] = set()
    for a in atoms:
        if a["record"] != "ATOM" or a["element"] == "H":
            continue
        if chain is not None and a["chain"] != chain:
            continue
        ax, ay, az = a["x"], a["y"], a["z"]
        for lx, ly, lz in ligand_coords:
            d = math.sqrt((ax - lx) ** 2 + (ay - ly) ** 2 + (az - lz) ** 2)
            if d <= cutoff_a:
                hits.add(f"{a['chain']}:{a['resname']}{a['resseq']}")
                break
    return sorted(hits)


def residues_near_center(
    atoms: list[dict[str, Any]],
    center: Sequence[float],
    *,
    cutoff_a: float = 6.0,
    chain: str | None = None,
) -> list[str]:
    """Receptor residues near a predicted pocket center (for residue-F1)."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    hits: set[str] = set()
    for a in atoms:
        if a["record"] != "ATOM" or a["element"] == "H":
            continue
        if chain is not None and a["chain"] != chain:
            continue
        d = math.sqrt((a["x"] - cx) ** 2 + (a["y"] - cy) ** 2 + (a["z"] - cz) ** 2)
        if d <= cutoff_a:
            hits.add(f"{a['chain']}:{a['resname']}{a['resseq']}")
    return sorted(hits)


def assert_no_hetatm(pdb_path: Path) -> None:
    text = pdb_path.read_text(errors="ignore")
    for line in text.splitlines():
        if line.startswith("HETATM"):
            raise AssertionError(f"ligand leakage: HETATM present in {pdb_path}")
