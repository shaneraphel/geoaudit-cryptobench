"""Polar-gate family: charged / amphipathic cluster geometry.

Why this family
---------------
The worst seam-vs-pLM anti-ranks (``4j4e_F``, ``3ly8_A``, ``5i3t_E``) concentrate
charged and polar cryptic residues (Lys/Arg/Glu/Asp/Gln/Asn/His). Surface
exposure alone (cryptic_aperture) did not recover them — the dual is not
"more empty", it is "a polar gate that can rearrange". Residue-level AA
identity alone was already null; members here are *cluster* and *pair*
geometry of charges, not one-hot chemistry.

Members (different quantities, not a radius ladder):
salt-bridge frustration, sequential charged runs, opposite-charge proximity,
polar-patch cohesion, histidine gates, and ranks of those.

Prediction: union on seam_geometry should raise mean on the anti-rank subset
without collapsing the fold mean; more_old ≈ 0. Falsify if union ≤ more_old.

clinical_grade = false.
"""
from __future__ import annotations

import numpy as np

CONTACT_CA = 8.0
SALT = 6.0  # CA proxy for salt-bridge neighbourhood
SKIP = frozenset({"HOH", "WAT", "DOD"})

POS = frozenset({"LYS", "ARG"})
NEG = frozenset({"ASP", "GLU"})
POLAR = frozenset({"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS",
                   "ASP", "GLU", "LYS", "ARG"})
HIS = frozenset({"HIS"})
AROM = frozenset({"PHE", "TYR", "TRP", "HIS"})

COLUMNS = (
    "is_pos",
    "is_neg",
    "is_polar",
    "is_his",
    "charge",
    # Sequential charged / polar runs.
    "seq_charged_run_len",
    "seq_polar_run_len",
    "seq_opposite_charge_pm2",
    "seq_same_charge_pm2",
    # Spatial charge neighbourhood at CA.
    "n_pos_within_8A",
    "n_neg_within_8A",
    "n_polar_within_8A",
    "n_his_within_8A",
    "n_arom_within_8A",
    "charge_sum_8A",
    "charge_abs_sum_8A",
    "n_opposite_charge_6A",
    "n_same_charge_6A",
    "salt_bridge_proxy",
    "salt_frustration",  # same-charge contacts minus opposite
    # Polar patch cohesion.
    "polar_patch_size",
    "polar_patch_frac_x100",
    "charged_patch_size",
    # Gates: His near charged / aromatic near charged.
    "his_near_charge",
    "arom_near_charge",
    "polar_times_empty_shell",
    "charge_times_empty_shell",
    # Ranks.
    "rank_charge_abs_sum",
    "rank_salt_frustration",
    "rank_polar_patch",
    "rank_seq_charged_run",
    "polar_gate_flag",
)
N_COLUMNS = len(COLUMNS)
_COL = {n: j for j, n in enumerate(COLUMNS)}


def _ca(atoms):
    for a in atoms:
        if (a.get("name") or "").strip() == "CA":
            return np.array([a["x"], a["y"], a["z"]], dtype=np.float64)
    return None


def _rn(atoms):
    return (atoms[0].get("resname") or "UNK").strip().upper() if atoms else "UNK"


def _rank01(v):
    n = len(v)
    if n <= 1:
        return np.zeros(n)
    return v.argsort().argsort().astype(np.float64) / (n - 1)


def compute(atoms_by_res: list[list[dict]], resseqs: list[int]) -> np.ndarray:
    n = len(atoms_by_res)
    X = np.zeros((n, N_COLUMNS), dtype=np.float64)
    if n == 0:
        return X
    cas, names = [], []
    for atoms in atoms_by_res:
        cas.append(_ca(atoms))
        names.append(_rn(atoms))
    valid = [i for i, c in enumerate(cas) if c is not None]
    if len(valid) < 2:
        return X
    P = np.vstack([cas[i] for i in valid])
    dmat = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)

    charge = np.zeros(len(valid))
    is_pos = np.zeros(len(valid))
    is_neg = np.zeros(len(valid))
    is_pol = np.zeros(len(valid))
    is_his = np.zeros(len(valid))
    is_arom = np.zeros(len(valid))
    for k, i in enumerate(valid):
        rn = names[i]
        if rn in POS:
            is_pos[k] = 1
            charge[k] = 1
        elif rn in NEG:
            is_neg[k] = 1
            charge[k] = -1
        if rn in POLAR:
            is_pol[k] = 1
        if rn in HIS:
            is_his[k] = 1
        if rn in AROM:
            is_arom[k] = 1

    # Polar / charged connected components at 8A.
    def components(mask):
        nodes = [k for k in range(len(valid)) if mask[k] > 0.5]
        parent = {k: k for k in nodes}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a in nodes:
            for b in np.where(dmat[a] < CONTACT_CA)[0]:
                if mask[b] > 0.5:
                    ra, rb = find(a), find(int(b))
                    if ra != rb:
                        parent[rb] = ra
        size = {}
        for a in nodes:
            r = find(a)
            size[r] = size.get(r, 0) + 1
        out = np.zeros(len(valid))
        for a in nodes:
            out[a] = size[find(a)]
        return out

    polar_sz = components(is_pol)
    charged_sz = components((is_pos + is_neg) > 0.5)
    n_polar_nodes = max(1, int(is_pol.sum()))

    for k, i in enumerate(valid):
        X[i, _COL["is_pos"]] = is_pos[k]
        X[i, _COL["is_neg"]] = is_neg[k]
        X[i, _COL["is_polar"]] = is_pol[k]
        X[i, _COL["is_his"]] = is_his[k]
        X[i, _COL["charge"]] = charge[k]

        # Sequential runs in valid order.
        run_c = 1 if abs(charge[k]) > 0 else 0
        if run_c:
            t = k - 1
            while t >= 0 and abs(charge[t]) > 0:
                run_c += 1
                t -= 1
            t = k + 1
            while t < len(valid) and abs(charge[t]) > 0:
                run_c += 1
                t += 1
        run_p = 1 if is_pol[k] > 0 else 0
        if run_p:
            t = k - 1
            while t >= 0 and is_pol[t] > 0:
                run_p += 1
                t -= 1
            t = k + 1
            while t < len(valid) and is_pol[t] > 0:
                run_p += 1
                t += 1
        X[i, _COL["seq_charged_run_len"]] = run_c
        X[i, _COL["seq_polar_run_len"]] = run_p
        lo, hi = max(0, k - 2), min(len(valid), k + 3)
        window_c = charge[lo:hi]
        if abs(charge[k]) > 0:
            X[i, _COL["seq_opposite_charge_pm2"]] = int(
                np.sum(window_c * charge[k] < 0)
            )
            X[i, _COL["seq_same_charge_pm2"]] = int(
                np.sum(window_c * charge[k] > 0)
            )

        nbr8 = np.where(dmat[k] < CONTACT_CA)[0]
        nbr6 = np.where(dmat[k] < SALT)[0]
        X[i, _COL["n_pos_within_8A"]] = int(is_pos[nbr8].sum()) if nbr8.size else 0
        X[i, _COL["n_neg_within_8A"]] = int(is_neg[nbr8].sum()) if nbr8.size else 0
        X[i, _COL["n_polar_within_8A"]] = int(is_pol[nbr8].sum()) if nbr8.size else 0
        X[i, _COL["n_his_within_8A"]] = int(is_his[nbr8].sum()) if nbr8.size else 0
        X[i, _COL["n_arom_within_8A"]] = int(is_arom[nbr8].sum()) if nbr8.size else 0
        X[i, _COL["charge_sum_8A"]] = float(charge[nbr8].sum()) if nbr8.size else 0
        X[i, _COL["charge_abs_sum_8A"]] = (
            float(np.abs(charge[nbr8]).sum()) if nbr8.size else 0
        )
        if abs(charge[k]) > 0 and nbr6.size:
            opp = int(np.sum(charge[nbr6] * charge[k] < 0))
            same = int(np.sum(charge[nbr6] * charge[k] > 0))
            X[i, _COL["n_opposite_charge_6A"]] = opp
            X[i, _COL["n_same_charge_6A"]] = same
            X[i, _COL["salt_bridge_proxy"]] = opp
            X[i, _COL["salt_frustration"]] = same - opp
        X[i, _COL["polar_patch_size"]] = polar_sz[k]
        X[i, _COL["polar_patch_frac_x100"]] = 100.0 * polar_sz[k] / n_polar_nodes
        X[i, _COL["charged_patch_size"]] = charged_sz[k]
        X[i, _COL["his_near_charge"]] = (
            1.0
            if is_his[k] > 0 and nbr8.size and np.abs(charge[nbr8]).sum() > 0
            else 0.0
        )
        X[i, _COL["arom_near_charge"]] = (
            1.0
            if is_arom[k] > 0 and nbr8.size and np.abs(charge[nbr8]).sum() > 0
            else 0.0
        )
        empty = max(0, 12 - len(nbr8))
        X[i, _COL["polar_times_empty_shell"]] = is_pol[k] * empty
        X[i, _COL["charge_times_empty_shell"]] = abs(charge[k]) * empty

    # Ranks
    abs_sum = np.array([X[i, _COL["charge_abs_sum_8A"]] for i in range(n)])
    frust = np.array([X[i, _COL["salt_frustration"]] for i in range(n)])
    pp = np.array([X[i, _COL["polar_patch_size"]] for i in range(n)])
    run = np.array([X[i, _COL["seq_charged_run_len"]] for i in range(n)])
    r1, r2, r3, r4 = _rank01(abs_sum), _rank01(frust), _rank01(pp), _rank01(run)
    for i in range(n):
        X[i, _COL["rank_charge_abs_sum"]] = 100.0 * r1[i]
        X[i, _COL["rank_salt_frustration"]] = 100.0 * r2[i]
        X[i, _COL["rank_polar_patch"]] = 100.0 * r3[i]
        X[i, _COL["rank_seq_charged_run"]] = 100.0 * r4[i]
        X[i, _COL["polar_gate_flag"]] = (
            1.0
            if X[i, _COL["is_polar"]] > 0.5
            and (r3[i] > 0.6 or r4[i] > 0.6 or r2[i] > 0.7)
            else 0.0
        )
    return X


def consistency(X: np.ndarray) -> list[str]:
    bad = []
    if X.shape[1] != N_COLUMNS:
        bad.append(f"width {X.shape[1]} != {N_COLUMNS}")
    if np.isnan(X).any():
        bad.append("nan")
    if np.isinf(X).any():
        bad.append("inf")
    return bad
