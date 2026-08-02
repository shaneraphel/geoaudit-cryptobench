"""Sequence-structure Toeplitz lag quantities (3D F/G discrete).

For each residue i and lag k in a fixed set, compare the geometric state at i
with the state at i±k when both exist: burial agreement, contact-graph
agreement, CA displacement, and side-chain volume contrast. Members are
different lag-statistics, not radii of one operator.

Prediction: union with seam_geometry should help short chains where local
sequence context carries cryptic signal that pLM exploits. Control: more_old;
permutation of lag columns within chain.

clinical_grade = false.
"""
from __future__ import annotations

import numpy as np

LAGS = (1, 2, 3, 4, 5, 8, 12, 16, 20)
CONTACT_CA = 8.0
SKIP = frozenset({"HOH", "WAT", "DOD"})
VOL = {
    "ALA": 67, "ARG": 148, "ASN": 96, "ASP": 91, "CYS": 86, "GLN": 114,
    "GLU": 109, "GLY": 48, "HIS": 118, "ILE": 124, "LEU": 124, "LYS": 135,
    "MET": 124, "PHE": 135, "PRO": 90, "SER": 73, "THR": 93, "TRP": 163,
    "TYR": 141, "VAL": 105,
}

COLUMNS = tuple(
    f"{stat}_lag{k}"
    for k in LAGS
    for stat in (
        "burial_agree",
        "ca_disp_x10",
        "vol_diff",
        "both_contact",
        "radial_diff_x100",
    )
)
N_COLUMNS = len(COLUMNS)
_COL = {n: j for j, n in enumerate(COLUMNS)}


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
    if n < 3:
        return X
    ca = np.full((n, 3), np.nan)
    vol = np.zeros(n)
    for i, atoms in enumerate(atoms_by_res):
        p = _ca(atoms)
        if p is not None:
            ca[i] = p
        vol[i] = VOL.get(_resname(atoms), 100)
    ok = np.isfinite(ca[:, 0])
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    burial = np.sum((d <= CONTACT_CA) & ok[None, :] & ok[:, None], axis=1)
    adj = (d <= CONTACT_CA) & ok[:, None] & ok[None, :]
    com = np.nanmean(ca[ok], axis=0)
    radial = np.linalg.norm(ca - com, axis=1)
    # map resseq -> index
    idx_of = {int(r): i for i, r in enumerate(resseqs)}

    for i in range(n):
        if not ok[i]:
            continue
        si = int(resseqs[i])
        for k in LAGS:
            for sign, tag in ((+1, ""),):  # both directions averaged below
                pass
            vals = {s: [] for s in (
                "burial_agree", "ca_disp_x10", "vol_diff",
                "both_contact", "radial_diff_x100")}
            for sign in (+1, -1):
                j = idx_of.get(si + sign * k)
                if j is None or not ok[j]:
                    continue
                vals["burial_agree"].append(
                    1.0 if abs(int(burial[i]) - int(burial[j])) <= 1 else 0.0)
                vals["ca_disp_x10"].append(
                    float(np.linalg.norm(ca[i] - ca[j]) * 10.0))
                vals["vol_diff"].append(abs(vol[i] - vol[j]))
                vals["both_contact"].append(1.0 if adj[i, j] else 0.0)
                vals["radial_diff_x100"].append(
                    abs(float(radial[i] - radial[j])) * 100.0)
            for stat, xs in vals.items():
                key = f"{stat}_lag{k}"
                X[i, _COL[key]] = float(np.mean(xs)) if xs else 0.0
    return X


def consistency(X: np.ndarray) -> list[str]:
    bad = []
    if X.shape[1] != N_COLUMNS:
        bad.append(f"width {X.shape[1]} != {N_COLUMNS}")
    if np.isnan(X).any():
        bad.append("nan")
    return bad
