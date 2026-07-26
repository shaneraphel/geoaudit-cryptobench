"""Chemistry and scale: the two axes the 35 algebraic invariants do not span.

Where the gap actually is
-------------------------
A closed-form linear functional of the 35 algebraic invariants reaches ROC-AUC
0.783 on the official fold while P2Rank reaches 0.793. The counting field cannot
be pushed past the linear functional of its own inputs, so no fusion rule closes
that gap; the inputs have to change. Two things the 35 invariants provably do
not contain:

**Chemistry.** Every one of the 35 is a function of atom POSITIONS alone. A
cryptic site is not merely a geometric depression: it is a hydrophobic,
aromatic-rich, conformationally flexible depression, and residue identity states
that in one integer. These wires are published physicochemical constants
(hydropathy, side-chain volume, aromatic ring count, formal charge at pH 7,
hydrogen-bond donor and acceptor counts, side-chain rotatable-bond count) plus
one frequency counted on the training fold. No constant here is fitted to
CryptoBench; the propensity is a bincount.

**Scale.** Each invariant is evaluated at one neighbourhood radius. P2Rank
aggregates its features over several neighbourhood sizes, and that is not a
modelling trick: a cryptic pocket has structure at the side-chain scale (6 A),
the loop scale (14 A) and the subdomain scale (20 A), and a single radius sees
one of them. The context transform contracts every wire over the geometric
adjacency at each radius,

    C_r[x]_i = ( sum_{j : |c_i - c_j| <= r} x_j ) / |N_r(i)| ,

which is the same unweighted counting gate used by the spatial gate, applied to
the inputs instead of the output. It is an average over a fixed adjacency: no
kernel, no bandwidth, no fitted coefficient.

Neither addition changes the character of the resolution path. Every wire is
still banded by its own chain's rank order into a quaternary digit, still
addresses a dense integer counter table, and is still fused by integer-weighted
addition.
"""
from __future__ import annotations

import numpy as np

from pocket_bench.methods.sequence_wires import (
    AA20,
    N_AA,
    apply_propensity,
    propensity_table,
)

# --- published per-residue constants ---------------------------------------
# Kyte-Doolittle hydropathy, J Mol Biol 157:105 (1982)
_KD = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5, "MET": 1.9,
    "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8, "TRP": -0.9, "TYR": -1.3,
    "PRO": -1.6, "HIS": -3.2, "GLU": -3.5, "GLN": -3.5, "ASP": -3.5, "ASN": -3.5,
    "LYS": -3.9, "ARG": -4.5,
}
# Side-chain volume in A^3, Zamyatnin, Prog Biophys Mol Biol 24:107 (1974)
_VOL = {
    "GLY": 60.1, "ALA": 88.6, "SER": 89.0, "CYS": 108.5, "ASP": 111.1,
    "PRO": 112.7, "ASN": 114.1, "THR": 116.1, "GLU": 138.4, "VAL": 140.0,
    "GLN": 143.8, "HIS": 153.2, "MET": 162.9, "ILE": 166.7, "LEU": 166.7,
    "LYS": 168.6, "ARG": 173.4, "PHE": 189.9, "TYR": 193.6, "TRP": 227.8,
}
# Aromatic rings in the side chain.
_AROM = {a: 0.0 for a in _KD}
_AROM.update({"PHE": 1.0, "TYR": 1.0, "HIS": 1.0, "TRP": 2.0})
# Formal charge at pH 7; histidine is partially protonated.
_CHARGE = {a: 0.0 for a in _KD}
_CHARGE.update({"ASP": -1.0, "GLU": -1.0, "LYS": 1.0, "ARG": 1.0, "HIS": 0.1})
# Side-chain hydrogen-bond donors and acceptors.
_HBD = {a: 0.0 for a in _KD}
_HBD.update({"ARG": 3.0, "LYS": 1.0, "TRP": 1.0, "ASN": 1.0, "GLN": 1.0,
             "HIS": 1.0, "SER": 1.0, "THR": 1.0, "TYR": 1.0, "CYS": 1.0})
_HBA = {a: 0.0 for a in _KD}
_HBA.update({"ASP": 2.0, "GLU": 2.0, "ASN": 1.0, "GLN": 1.0, "HIS": 1.0,
             "SER": 1.0, "THR": 1.0, "TYR": 1.0, "MET": 1.0, "CYS": 1.0})
# Rotatable side-chain bonds (chi angles): a proxy for the conformational
# freedom a cryptic site must have in order to open at all.
_CHI = {
    "GLY": 0.0, "ALA": 0.0, "PRO": 0.0, "SER": 1.0, "CYS": 1.0, "THR": 1.0,
    "VAL": 1.0, "ILE": 2.0, "LEU": 2.0, "ASP": 2.0, "ASN": 2.0, "HIS": 2.0,
    "PHE": 2.0, "TRP": 2.0, "TYR": 2.0, "MET": 3.0, "GLU": 3.0, "GLN": 3.0,
    "LYS": 4.0, "ARG": 4.0,
}

CHEM_NAMES = ("kd", "volume", "aromatic", "charge", "hbd", "hba", "chi")
_CHEM_TABLES = (_KD, _VOL, _AROM, _CHARGE, _HBD, _HBA, _CHI)

CONTEXT_RADII = (6.0, 14.0, 20.0)


def chemical_wires(codes: np.ndarray) -> np.ndarray:
    """``(R, 7)`` of published constants; an unknown residue takes the mean."""
    codes = np.asarray(codes, dtype=np.int64)
    cols = []
    for table in _CHEM_TABLES:
        vals = np.asarray([table[a] for a in AA20], dtype=np.float64)
        col = np.full(len(codes), float(vals.mean()), dtype=np.float64)
        ok = codes >= 0
        col[ok] = vals[np.clip(codes[ok], 0, N_AA - 1)]
        cols.append(col)
    return np.stack(cols, axis=1)


def context_transform(X: np.ndarray, ctr: np.ndarray, n_res_per: np.ndarray,
                      radii=CONTEXT_RADII) -> np.ndarray:
    """``(R, C*len(radii))``: every column averaged over each neighbourhood.

    Blocked per chain, because the adjacency is intra-chain by construction: a
    residue's neighbourhood cannot cross into a structure it is not part of.
    """
    X = np.asarray(X, dtype=np.float64)
    out = np.empty((X.shape[0], X.shape[1] * len(radii)), dtype=np.float64)
    off = 0
    for n in n_res_per:
        n = int(n)
        c = ctr[off:off + n]
        blk = X[off:off + n]
        for k, r in enumerate(radii):
            r2 = r * r
            acc = np.empty((n, X.shape[1]), dtype=np.float64)
            for i in range(0, n, 512):
                d2 = ((c[i:i + 512, None, :] - c[None, :, :]) ** 2).sum(-1)
                a = (d2 <= r2).astype(np.float64)
                acc[i:i + 512] = (a @ blk) / np.maximum(a.sum(1), 1.0)[:, None]
            out[off:off + n,
                k * X.shape[1]:(k + 1) * X.shape[1]] = acc
        off += n
    return out


def build_expanded(F: np.ndarray, codes: np.ndarray, ctr: np.ndarray,
                   n_res_per: np.ndarray, base_names: tuple[str, ...],
                   prop_table: np.ndarray | None = None,
                   y: np.ndarray | None = None,
                   ) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """``(X, names, propensity_table)``.

    ``prop_table`` is supplied when scoring (compiled on the training fold) and
    computed here only when ``y`` is given, which happens exactly once, on the
    training fold. A test residue never contributes a count to it.
    """
    chem = chemical_wires(codes)
    if prop_table is None:
        if y is None:
            raise ValueError("need either a compiled propensity table or labels")
        prop_table = propensity_table(codes, y)
    prop = apply_propensity(codes, prop_table)[:, None]

    local = np.concatenate([F, chem, prop], axis=1)
    local_names = tuple(base_names) + CHEM_NAMES + ("propensity",)

    ctx = context_transform(local, ctr, n_res_per)
    ctx_names = tuple(f"{nm}@{int(r)}" for r in CONTEXT_RADII
                      for nm in local_names)

    X = np.concatenate([local, ctx], axis=1)
    return X, local_names + ctx_names, prop_table
