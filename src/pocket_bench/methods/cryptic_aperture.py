"""Cryptic-aperture family: surface / dual-of-seam quantities.

Why this family
---------------
On several official-fold units (``4j4e_F``, ``2vqz_F``, ``3ly8_A``) the deployed
seam/geometry scores are *anti-correlated* with cryptic labels: inverting the
score would raise ROC-AUC from ~0.08–0.16 to ~0.84–0.94 on those chains. The
counting field learned "high packing-seam ⇒ cryptic"; those cryptic sites sit
on the dual geometry — exposed, low-packing, loop-like apertures that open
without looking like a buried peelable seam today.

Members are different measurements (not a radius ladder of one operator):
exposure proxies, backbone kink, terminus proximity, amphipathic contrast,
surface-patch cohesion, and within-chain ranks of those quantities.

Prediction (written before the lift)
------------------------------------
Union-attached to ``seam_geometry_field`` columns should raise official-fold
mean vs pLM-NN if the polarity deficit is collectible as dual columns;
``more_old`` near zero; row-permuted ≤ intact. Falsify if union ≤ more_old.

``clinical_grade`` is false. No affinity or therapeutic claim.
"""
from __future__ import annotations

import numpy as np

CONTACT_CA = 8.0
PACK_CA = 6.0
SKIP = frozenset({"HOH", "WAT", "DOD"})

HYDRO = frozenset({"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"})
POLAR = frozenset({"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS",
                   "ASP", "GLU", "LYS", "ARG"})
AROM = frozenset({"PHE", "TYR", "TRP", "HIS"})
CHARGE = {
    "ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1, "HIS": 0,
}
CHI = {
    "ALA": 0, "GLY": 0, "PRO": 0, "SER": 1, "CYS": 1, "THR": 1, "VAL": 1,
    "ASN": 2, "ASP": 2, "LEU": 2, "ILE": 2, "HIS": 2, "PHE": 2, "TYR": 2,
    "TRP": 2, "GLN": 3, "GLU": 3, "MET": 3, "LYS": 4, "ARG": 4,
}

COLUMNS = (
    # Exposure / inverse packing (dual of burial).
    "n_empty_ca_shell_8A",
    "frac_empty_ca_shell_x100",
    "inv_pack_6A_x100",
    "n_sc_heavy_exposed_proxy",
    "radial_surface_pct_x100",
    "hull_layer_is_outer",
    "dist_to_centroid_rank_x100",
    # Backbone aperture / kink (loop-like geometry).
    "ca_kink_angle_x100",
    "ca_kink_is_sharp",
    "ca_step_stretch_x100",
    "local_ca_rg_x100",
    "seq_terminus_proximity_x100",
    "is_n_terminal_20",
    "is_c_terminal_20",
    # Surface chemistry contrast (amphipathic aperture).
    "neigh8_n_hydro",
    "neigh8_n_polar",
    "neigh8_amphipathic_imbalance",
    "neigh8_charge_abs",
    "neigh8_n_aromatic",
    "neigh8_n_gly_pro",
    "self_chi",
    "exposure_times_chi",
    "exposure_times_amphipathic",
    "low_pack_times_kink",
    # Sequential hydrophobicity contrast along the chain.
    "seq_hydro_contrast_pm2",
    "seq_hydro_contrast_pm4",
    "seq_charge_contrast_pm3",
    # Surface-patch cohesion in the exposed shell.
    "n_exposed_neighbours",
    "exposed_patch_size",
    "exposed_patch_frac_x100",
    # Within-chain ranks of the load-bearing duals.
    "rank_inv_pack",
    "rank_kink",
    "rank_exposure_chi",
    "rank_amphipathic",
    "rank_terminus",
    "rank_exposed_patch",
    "aperture_flag",
)
N_COLUMNS = len(COLUMNS)
_COL = {n: j for j, n in enumerate(COLUMNS)}


def _atom_name(a: dict) -> str:
    return (a.get("name") or "").strip().upper()


def _resname(atoms: list[dict]) -> str:
    if not atoms:
        return "UNK"
    return (atoms[0].get("resname") or "UNK").strip().upper()


def _ca(atoms: list[dict]) -> np.ndarray | None:
    for a in atoms:
        if _atom_name(a) == "CA":
            return np.array([a["x"], a["y"], a["z"]], dtype=np.float64)
    return None


def _sc_heavy(atoms: list[dict]) -> np.ndarray:
    pts = []
    for a in atoms:
        if a.get("element") == "H":
            continue
        n = _atom_name(a)
        if n in {"N", "CA", "C", "O", "OXT"}:
            continue
        pts.append((a["x"], a["y"], a["z"]))
    return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 3))


def _hull_outer(pts: np.ndarray) -> np.ndarray:
    """Boolean: residue on the outer convex hull (iterative peel depth 0)."""
    n = len(pts)
    out = np.zeros(n, dtype=bool)
    if n < 4:
        out[:] = True
        return out
    remaining = np.arange(n)
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        # Centroid-radius fallback if scipy unavailable.
        c = pts.mean(axis=0)
        d = np.linalg.norm(pts - c, axis=1)
        thr = np.percentile(d, 85)
        return d >= thr
    try:
        hull = ConvexHull(pts[remaining])
        out[remaining[hull.vertices]] = True
    except Exception:
        c = pts.mean(axis=0)
        d = np.linalg.norm(pts - c, axis=1)
        thr = np.percentile(d, 85)
        return d >= thr
    return out


def _rank01(v: np.ndarray) -> np.ndarray:
    n = len(v)
    if n <= 1:
        return np.zeros(n)
    order = v.argsort().argsort().astype(np.float64)
    return order / (n - 1)


def compute(atoms_by_res: list[list[dict]], resseqs: list[int]) -> np.ndarray:
    n = len(atoms_by_res)
    X = np.zeros((n, N_COLUMNS), dtype=np.float64)
    if n == 0:
        return X

    cas = []
    names = []
    for atoms in atoms_by_res:
        cas.append(_ca(atoms))
        names.append(_resname(atoms))
    valid = [i for i, c in enumerate(cas) if c is not None]
    if len(valid) < 3:
        return X

    P = np.vstack([cas[i] for i in valid])
    idx = {i: k for k, i in enumerate(valid)}
    centroid = P.mean(axis=0)
    dist_c = np.linalg.norm(P - centroid, axis=1)
    outer = _hull_outer(P)

    # CA contact graph at 8 A among valid residues.
    dmat = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    neigh8 = [np.where(dmat[k] < CONTACT_CA)[0] for k in range(len(valid))]
    pack6 = [int(np.sum(dmat[k] < PACK_CA)) for k in range(len(valid))]

    # Max CA neighbours in a full shell ~ coordination number ~12; empty = gap.
    max_shell = 12
    inv_pack = np.zeros(n)
    kink = np.zeros(n)
    stretch = np.zeros(n)
    local_rg = np.zeros(n)
    amph = np.zeros(n)
    expo_chi = np.zeros(n)
    expo_amph = np.zeros(n)
    lowpk_kink = np.zeros(n)
    exposed_nbr = np.zeros(n)

    hydro_flag = np.array([1 if names[i] in HYDRO else 0 for i in valid])
    polar_flag = np.array([1 if names[i] in POLAR else 0 for i in valid])
    charge_v = np.array([CHARGE.get(names[i], 0) for i in valid])
    chi_v = np.array([CHI.get(names[i], 0) for i in valid])
    arom_flag = np.array([1 if names[i] in AROM else 0 for i in valid])
    gp_flag = np.array([1 if names[i] in {"GLY", "PRO"} else 0 for i in valid])

    # Exposure proxy: low pack6 → high inv_pack; empty shell count.
    for k, i in enumerate(valid):
        empty = max(0, max_shell - len(neigh8[k]))
        X[i, _COL["n_empty_ca_shell_8A"]] = empty
        X[i, _COL["frac_empty_ca_shell_x100"]] = 100.0 * empty / max_shell
        ip = 100.0 / (1.0 + pack6[k])
        inv_pack[i] = ip
        X[i, _COL["inv_pack_6A_x100"]] = ip
        sc = _sc_heavy(atoms_by_res[i])
        # Exposed proxy: fewer nearby CA → more "exposed" side-chain budget.
        X[i, _COL["n_sc_heavy_exposed_proxy"]] = len(sc) * empty
        X[i, _COL["radial_surface_pct_x100"]] = 100.0 * _rank01(dist_c)[k]
        X[i, _COL["hull_layer_is_outer"]] = 1.0 if outer[k] else 0.0
        X[i, _COL["dist_to_centroid_rank_x100"]] = 100.0 * _rank01(dist_c)[k]

        # Backbone kink from CA i-1, i, i+1 in sequence order among valid.
        # Use positional neighbours in the valid list (chain order preserved).
        if 0 < k < len(valid) - 1:
            a, b, c = P[k - 1], P[k], P[k + 1]
            u, v = a - b, c - b
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu > 1e-6 and nv > 1e-6:
                cos = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
                ang = float(np.degrees(np.arccos(cos)))
                # Sharp kink: angle far from ~110° (ideal tetrahedral CA).
                kink[i] = abs(ang - 110.0)
                X[i, _COL["ca_kink_angle_x100"]] = kink[i]
                X[i, _COL["ca_kink_is_sharp"]] = 1.0 if kink[i] > 25.0 else 0.0
                stretch[i] = 0.5 * (nu + nv)
                X[i, _COL["ca_step_stretch_x100"]] = 100.0 * stretch[i]

        if neigh8[k].size:
            pts = P[neigh8[k]]
            local_rg[i] = float(np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean()))
        X[i, _COL["local_ca_rg_x100"]] = 100.0 * local_rg[i]

        # Terminus proximity in residue-index space.
        term = min(k, len(valid) - 1 - k) / max(1, len(valid) - 1)
        # 0 at centre, 1 at terminus — invert so terminus is high.
        term_prox = 1.0 - term
        X[i, _COL["seq_terminus_proximity_x100"]] = 100.0 * term_prox
        X[i, _COL["is_n_terminal_20"]] = 1.0 if k < 20 else 0.0
        X[i, _COL["is_c_terminal_20"]] = 1.0 if k >= len(valid) - 20 else 0.0

        nh = int(hydro_flag[neigh8[k]].sum()) if neigh8[k].size else 0
        np_ = int(polar_flag[neigh8[k]].sum()) if neigh8[k].size else 0
        X[i, _COL["neigh8_n_hydro"]] = nh
        X[i, _COL["neigh8_n_polar"]] = np_
        amph[i] = abs(nh - np_)
        X[i, _COL["neigh8_amphipathic_imbalance"]] = amph[i]
        X[i, _COL["neigh8_charge_abs"]] = (
            int(np.abs(charge_v[neigh8[k]]).sum()) if neigh8[k].size else 0
        )
        X[i, _COL["neigh8_n_aromatic"]] = (
            int(arom_flag[neigh8[k]].sum()) if neigh8[k].size else 0
        )
        X[i, _COL["neigh8_n_gly_pro"]] = (
            int(gp_flag[neigh8[k]].sum()) if neigh8[k].size else 0
        )
        X[i, _COL["self_chi"]] = chi_v[k]
        expo_chi[i] = ip * chi_v[k]
        expo_amph[i] = ip * amph[i]
        lowpk_kink[i] = ip * kink[i]
        X[i, _COL["exposure_times_chi"]] = expo_chi[i]
        X[i, _COL["exposure_times_amphipathic"]] = expo_amph[i]
        X[i, _COL["low_pack_times_kink"]] = lowpk_kink[i]

        # Sequential hydrophobicity contrast.
        for w, col in ((2, "seq_hydro_contrast_pm2"), (4, "seq_hydro_contrast_pm4")):
            lo, hi = max(0, k - w), min(len(valid), k + w + 1)
            window = hydro_flag[lo:hi]
            X[i, _COL[col]] = abs(int(hydro_flag[k]) * len(window) - int(window.sum()))
        lo, hi = max(0, k - 3), min(len(valid), k + 4)
        X[i, _COL["seq_charge_contrast_pm3"]] = abs(
            int(charge_v[k]) * (hi - lo) - int(charge_v[lo:hi].sum())
        )

        # Exposed neighbours: neighbours that are also outer-hull.
        if neigh8[k].size:
            exposed_nbr[i] = int(outer[neigh8[k]].sum())
        X[i, _COL["n_exposed_neighbours"]] = exposed_nbr[i]

    # Exposed patch size: connected components on outer hull via 8A edges.
    outer_nodes = [k for k in range(len(valid)) if outer[k]]
    parent = {k: k for k in outer_nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for k in outer_nodes:
        for j in neigh8[k]:
            if outer[j]:
                union(k, j)
    size = {}
    for k in outer_nodes:
        r = find(k)
        size[r] = size.get(r, 0) + 1
    for k, i in enumerate(valid):
        if outer[k]:
            sz = size[find(k)]
            X[i, _COL["exposed_patch_size"]] = sz
            X[i, _COL["exposed_patch_frac_x100"]] = 100.0 * sz / max(1, len(outer_nodes))
        else:
            X[i, _COL["exposed_patch_size"]] = 0
            X[i, _COL["exposed_patch_frac_x100"]] = 0

    # Ranks (full chain, zeros for missing CA stay 0).
    full_inv = np.array([inv_pack[i] for i in range(n)])
    full_kink = np.array([kink[i] for i in range(n)])
    full_ec = np.array([expo_chi[i] for i in range(n)])
    full_am = np.array([amph[i] for i in range(n)])
    full_term = np.array([X[i, _COL["seq_terminus_proximity_x100"]] for i in range(n)])
    full_patch = np.array([X[i, _COL["exposed_patch_size"]] for i in range(n)])

    r_inv = _rank01(full_inv)
    r_kink = _rank01(full_kink)
    r_ec = _rank01(full_ec)
    r_am = _rank01(full_am)
    r_term = _rank01(full_term)
    r_patch = _rank01(full_patch)
    for i in range(n):
        X[i, _COL["rank_inv_pack"]] = 100.0 * r_inv[i]
        X[i, _COL["rank_kink"]] = 100.0 * r_kink[i]
        X[i, _COL["rank_exposure_chi"]] = 100.0 * r_ec[i]
        X[i, _COL["rank_amphipathic"]] = 100.0 * r_am[i]
        X[i, _COL["rank_terminus"]] = 100.0 * r_term[i]
        X[i, _COL["rank_exposed_patch"]] = 100.0 * r_patch[i]
        # Aperture flag: outer hull AND (high inv-pack OR sharp kink).
        X[i, _COL["aperture_flag"]] = (
            1.0
            if X[i, _COL["hull_layer_is_outer"]] > 0.5
            and (r_inv[i] > 0.7 or r_kink[i] > 0.7)
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
