"""Sequence input wires for the associative memory.

Four additional quaternary digits appended to the geometric address word. They
carry residue identity, which is information the geometric invariants provably do
not contain, and they are read directly off the receptor -- no alignment, no
profile database, no network.

What is NOT here, and why
-------------------------
True evolutionary conservation (a PSSM / HHblits profile) is the strongest known
sequence signal for ligand-binding sites, and it is deliberately absent: the
published CryptoBench artifacts carry no conservation column (the record schema is
apo/holo chain ids, pocket selections, ligand id, pRMSD, uniprot id and nothing
else), and no alignment tool is installed on this machine (hhblits, jackhmmer,
psiblast, hmmbuild all absent). ESM-2 embeddings are excluded by construction
because they are a neural network. So Track B measures what residue IDENTITY adds,
which is a strict lower bound on what full conservation would add.

S1 and S2 are published physicochemical constants, not fitted parameters. S4 is a
20-entry frequency counter compiled on the training fold only -- one integer
increment per training residue, no optimizer.
"""
from __future__ import annotations

import numpy as np

# Kyte-Doolittle hydropathy (J Mol Biol 157:105, 1982). Fixed literature constants.
_KD = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5, "MET": 1.9,
    "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8, "TRP": -0.9, "TYR": -1.3,
    "PRO": -1.6, "HIS": -3.2, "GLU": -3.5, "GLN": -3.5, "ASP": -3.5, "ASN": -3.5,
    "LYS": -3.9, "ARG": -4.5,
}
# Side-chain volumes, A^3 (Zamyatnin, Prog Biophys Mol Biol 24:107, 1974).
_VOL = {
    "GLY": 60.1, "ALA": 88.6, "SER": 89.0, "CYS": 108.5, "ASP": 111.1,
    "PRO": 112.7, "ASN": 114.1, "THR": 116.1, "GLU": 138.4, "VAL": 140.0,
    "GLN": 143.8, "HIS": 153.2, "MET": 162.9, "ILE": 166.7, "LEU": 166.7,
    "LYS": 168.6, "ARG": 173.4, "PHE": 189.9, "TYR": 193.6, "TRP": 227.8,
}
AA20 = tuple(sorted(_KD))
_INDEX = {a: i for i, a in enumerate(AA20)}
N_AA = len(AA20)
N_SEQ_WIRES = 4


def residue_codes(resnames: list[str]) -> np.ndarray:
    """Residue-type index in [0,19]; anything non-standard maps to -1."""
    return np.fromiter((_INDEX.get(str(r).strip().upper(), -1) for r in resnames),
                       dtype=np.int64, count=len(resnames))


def static_sequence_features(codes: np.ndarray,
                             centroids: np.ndarray,
                             radius: float = 10.0) -> np.ndarray:
    """S1..S3: hydropathy, volume, and neighbourhood mean hydropathy.

    S3 is the only one that mixes the two modalities: it contracts the sequence
    scalar over the geometric adjacency, so it states "this residue sits in a
    hydrophobic patch", which neither modality expresses alone.
    """
    codes = np.asarray(codes, dtype=np.int64)
    kd = np.asarray([_KD[a] for a in AA20], dtype=np.float64)
    vol = np.asarray([_VOL[a] for a in AA20], dtype=np.float64)
    ok = codes >= 0
    s1 = np.where(ok, kd[np.clip(codes, 0, N_AA - 1)], 0.0)
    s2 = np.where(ok, vol[np.clip(codes, 0, N_AA - 1)], 0.0)

    c = np.asarray(centroids, dtype=np.float64)
    n = len(codes)
    s3 = np.empty(n, dtype=np.float64)
    r2 = radius * radius
    block = 512
    for i in range(0, n, block):
        d2 = ((c[i:i + block, None, :] - c[None, :, :]) ** 2).sum(-1)
        a = (d2 <= r2).astype(np.float64)
        s3[i:i + block] = (a @ s1) / np.maximum(a.sum(1), 1.0)
    return np.stack([s1, s2, s3], axis=1)


def propensity_table(codes: np.ndarray, y: np.ndarray) -> np.ndarray:
    """S4 source: P(cryptic | residue type), counted on the training fold.

    Laplace-smoothed by one pseudo-count per class so a residue type unseen in
    training yields the neutral prior instead of an undefined ratio. Counting,
    not fitting.
    """
    codes = np.asarray(codes, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    ok = codes >= 0
    pos = np.bincount(codes[ok], weights=y[ok], minlength=N_AA)
    tot = np.bincount(codes[ok], minlength=N_AA).astype(np.float64)
    return (pos + 1.0) / (tot + 2.0)


def apply_propensity(codes: np.ndarray, table: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.int64)
    table = np.asarray(table, dtype=np.float64)
    out = np.full(len(codes), float(table.mean()), dtype=np.float64)
    ok = codes >= 0
    out[ok] = table[codes[ok]]
    return out
