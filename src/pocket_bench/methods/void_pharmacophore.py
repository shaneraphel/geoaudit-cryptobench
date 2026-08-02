"""Void-directed pharmacophore: H-bond vectors pointing into empty space.

Distinct from contact_wall empty-octant counts (measured null) and from
residue_chemistry alone (null). Members measure whether a side-chain donor or
acceptor points toward free volume rather than toward protein.

clinical_grade = false.
"""
from __future__ import annotations

import numpy as np

CONTACT = 8.0
VOID_PROBE = 3.5  # A along the H-bond axis with no heavy atom
SKIP = frozenset({"HOH", "WAT", "DOD"})

# Approximate side-chain donor/acceptor atom names by residue.
DONOR_ATOMS = {
    "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"], "TRP": ["NE1"],
    "ASN": ["ND2"], "GLN": ["NE2"], "LYS": ["NZ"], "ARG": ["NE", "NH1", "NH2"],
    "HIS": ["ND1", "NE2"], "CYS": ["SG"],
}
ACCEPTOR_ATOMS = {
    "ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"], "ASN": ["OD1"], "GLN": ["OE1"],
    "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"], "HIS": ["ND1", "NE2"],
}

COLUMNS = (
    "n_donors",
    "n_acceptors",
    "n_donors_into_void",
    "n_acceptors_into_void",
    "frac_donors_into_void_x100",
    "frac_acceptors_into_void_x100",
    "n_void_pharma_total",
    "donor_void_minus_acceptor_void",
    "n_heavy_along_donor_axis",
    "n_heavy_along_acceptor_axis",
    "rank_void_pharma",
    "rank_donor_void",
    "is_void_pharma_rich",
    "is_donor_void_gate",
)
N_COLUMNS = len(COLUMNS)
_COL = {n: j for j, n in enumerate(COLUMNS)}


def _atoms_named(atoms, names):
    out = []
    for a in atoms:
        if (a.get("name") or "").strip() in names and a.get("element") != "H":
            out.append(np.array([a["x"], a["y"], a["z"]], dtype=np.float64))
    return out


def _ca(atoms):
    for a in atoms:
        if (a.get("name") or "").strip() == "CA":
            return np.array([a["x"], a["y"], a["z"]], dtype=np.float64)
    return None


def _resname(atoms):
    return (atoms[0].get("resname") or "UNK").strip().upper() if atoms else "UNK"


def compute(atoms_by_res: list[list[dict]], resseqs: list[int]) -> np.ndarray:
    n = len(atoms_by_res)
    X = np.zeros((n, N_COLUMNS), dtype=np.float64)
    if n == 0:
        return X
    # all heavy coords for void tests
    heavies = []
    for atoms in atoms_by_res:
        for a in atoms:
            if a.get("element") != "H":
                heavies.append([a["x"], a["y"], a["z"]])
    H = np.asarray(heavies, dtype=np.float64) if heavies else np.zeros((0, 3))

    void_pharma = np.zeros(n)
    donor_void = np.zeros(n)
    for i, atoms in enumerate(atoms_by_res):
        aa = _resname(atoms)
        ca = _ca(atoms)
        if ca is None:
            continue
        donors = _atoms_named(atoms, DONOR_ATOMS.get(aa, []))
        accs = _atoms_named(atoms, ACCEPTOR_ATOMS.get(aa, []))
        n_d, n_a = len(donors), len(accs)
        d_void = a_void = 0
        d_heavy = a_heavy = 0
        for p in donors:
            # axis from CA through donor, probe beyond donor
            v = p - ca
            nv = np.linalg.norm(v)
            if nv < 1e-6:
                continue
            u = v / nv
            probe = p + u * VOID_PROBE
            if len(H):
                dist = np.linalg.norm(H - probe, axis=1)
                near = int((dist < 2.8).sum())
                d_heavy += near
                if near == 0:
                    d_void += 1
            else:
                d_void += 1
        for p in accs:
            v = p - ca
            nv = np.linalg.norm(v)
            if nv < 1e-6:
                continue
            u = v / nv
            probe = p + u * VOID_PROBE
            if len(H):
                dist = np.linalg.norm(H - probe, axis=1)
                near = int((dist < 2.8).sum())
                a_heavy += near
                if near == 0:
                    a_void += 1
            else:
                a_void += 1
        tot = d_void + a_void
        void_pharma[i] = tot
        donor_void[i] = d_void
        X[i, _COL["n_donors"]] = n_d
        X[i, _COL["n_acceptors"]] = n_a
        X[i, _COL["n_donors_into_void"]] = d_void
        X[i, _COL["n_acceptors_into_void"]] = a_void
        X[i, _COL["frac_donors_into_void_x100"]] = 100.0 * d_void / n_d if n_d else 0.0
        X[i, _COL["frac_acceptors_into_void_x100"]] = 100.0 * a_void / n_a if n_a else 0.0
        X[i, _COL["n_void_pharma_total"]] = tot
        X[i, _COL["donor_void_minus_acceptor_void"]] = d_void - a_void
        X[i, _COL["n_heavy_along_donor_axis"]] = d_heavy
        X[i, _COL["n_heavy_along_acceptor_axis"]] = a_heavy

    r_vp = np.argsort(np.argsort(void_pharma)) / max(n - 1, 1) * 100.0
    r_dv = np.argsort(np.argsort(donor_void)) / max(n - 1, 1) * 100.0
    for i in range(n):
        X[i, _COL["rank_void_pharma"]] = r_vp[i]
        X[i, _COL["rank_donor_void"]] = r_dv[i]
        X[i, _COL["is_void_pharma_rich"]] = int(void_pharma[i] >= 2)
        X[i, _COL["is_donor_void_gate"]] = int(donor_void[i] >= 1 and void_pharma[i] >= 1)
    return X


def consistency(X: np.ndarray) -> list[str]:
    bad = []
    if X.shape[1] != N_COLUMNS:
        bad.append(f"width {X.shape[1]} != {N_COLUMNS}")
    if np.isnan(X).any():
        bad.append("nan")
    return bad
