"""Nonlocal-seam and packing-frustration quantities for compact domains.

Why this family
---------------
``geometry_field`` reaches fold-mean parity with pLM-NN but loses on short
chains, and catastrophically anti-ranks buried cryptic residues on a few units
(e.g. ``4j4e_F``: geometry 0.145 vs pLM 0.720). Contact-wall already counts
tertiary contacts and was null; residue chemistry alone was null. What those
families do not read is the *seam*: a residue that is packed in the core, has
long-range contact graph support, carries conformational budget in its
neighbourhood, and sits on a peelable depth layer — the geometry of a site that
can open without looking like a surface pocket today.

Members are different measurements (AGENT_MEMORY 2c), not radii of one operator.
Shell counts below use different atom classes at one cut, or one class at cuts
that enter different derived quantities, not a radius ladder as the family.

Prediction (written before the lift)
------------------------------------
Union attachment over geometry_field columns should move the official-fold
paired Δ vs pLM-NN from ≈0 toward positive if the short-chain buried deficit is
collectible; ``more_old`` near zero; row-permuted arm ≤ intact. Falsify if
union ≤ more_old.

``clinical_grade`` is false. No affinity or therapeutic claim.
"""
from __future__ import annotations

import numpy as np

CONTACT_CA = 8.0
PACK_CA = 6.0
HBOND_CUT = 3.5
SKIP = frozenset({"HOH", "WAT", "DOD"})

# Side-chain property tables — integers from structural formulae, not scales.
CHI = {
    "ALA": 0, "GLY": 0, "PRO": 0, "SER": 1, "CYS": 1, "THR": 1, "VAL": 1,
    "ASN": 2, "ASP": 2, "LEU": 2, "ILE": 2, "HIS": 2, "PHE": 2, "TYR": 2,
    "TRP": 2, "GLN": 3, "GLU": 3, "MET": 3, "LYS": 4, "ARG": 4,
}
CHARGE = {
    "ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1, "HIS": 0,
}
AROM = frozenset({"PHE", "TYR", "TRP", "HIS"})
POLAR = frozenset({"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS",
                   "ASP", "GLU", "LYS", "ARG"})
HYDRO = frozenset({"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"})

COLUMNS = (
    # Contact-graph sequence topology (finer than contact_wall's binary cut).
    "n_ca_contacts",
    "n_contacts_seq_lt_4",
    "n_contacts_seq_4_12",
    "n_contacts_seq_gt_12",
    "n_contacts_seq_gt_24",
    "max_seq_span_contact",
    "mean_seq_span_x10",
    "seq_span_entropy_x100",
    "frac_long_contacts_x100",
    # Graph shape of the 8 A CA neighbourhood.
    "degree",
    "n_triangles",
    "clustering_x100",
    "n_wedges",
    # Packing / burial without a radius ladder as the family.
    "n_ca_within_6A",
    "n_sc_heavy_within_5A",
    "n_bb_o_nonlocal_hbond",
    "n_bb_n_nonlocal_hbond",
    "hbond_frustration",
    # Local backbone geometry (discrete curvature / twist).
    "ca_angle_dev_x100",
    "ca_torsion_abs_x100",
    "ca_step_length_x100",
    # Neighbourhood inertia (one anisotropy number, not three eigenvalues).
    "neigh_rg_x100",
    "neigh_anisotropy_x100",
    "radial_pct_x100",
    "hull_depth",
    "on_convex_hull",
    # Conformational budget of the contact neighbourhood.
    "neigh_chi_sum",
    "neigh_n_gly",
    "neigh_n_pro",
    "neigh_n_aromatic",
    "neigh_n_charged",
    "neigh_charge_imbalance",
    "neigh_n_hydrophobic",
    "neigh_n_polar",
    "burial_times_chi",
    "long_contact_times_chi",
    "core_seam_flag",
    # Spectral / path geometry of the contact graph (distinct from degree/triangles).
    "fiedler_rank_x100",
    "laplacian_energy_local",
    "mean_path_len_x10",
    "n_paths3",
    "eccentricity",
    # Within-chain ranks of the load-bearing quantities.
    "rank_max_seq_span",
    "rank_hull_depth",
    "rank_hbond_frustration",
    "rank_burial_times_chi",
    "rank_long_contacts",
    "rank_clustering",
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


def _bb_atom(atoms: list[dict], name: str) -> np.ndarray | None:
    for a in atoms:
        if _atom_name(a) == name:
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


def _hull_depths(pts: np.ndarray) -> np.ndarray:
    """Iterative convex-hull peeling depth; 0 = on outer hull."""
    n = len(pts)
    depth = np.zeros(n, dtype=np.int64)
    if n < 4:
        return depth
    alive = np.ones(n, dtype=bool)
    layer = 0
    # SciPy optional — pure numpy fallback via extremal peeling on axes + diagonals.
    while alive.sum() >= 4 and layer < 40:
        idx = np.flatnonzero(alive)
        P = pts[idx]
        # Approximate hull: points that are extreme on many random directions.
        rng = np.random.default_rng(0)
        dirs = rng.normal(size=(32, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True).clip(1e-9)
        scores = P @ dirs.T
        extreme = np.zeros(len(idx), dtype=bool)
        extreme[np.argmax(scores, axis=0)] = True
        extreme[np.argmin(scores, axis=0)] = True
        if extreme.sum() < 3:
            break
        depth[idx[extreme]] = layer
        alive[idx[extreme]] = False
        layer += 1
    depth[alive] = layer
    return depth


def compute(atoms_by_res: list[list[dict]], resseqs: list[int]) -> np.ndarray:
    """Return (n_res, N_COLUMNS) float64 matrix for one polymer chain."""
    n = len(atoms_by_res)
    X = np.zeros((n, N_COLUMNS), dtype=np.float64)
    if n == 0:
        return X

    ca = np.full((n, 3), np.nan)
    names = []
    for i, atoms in enumerate(atoms_by_res):
        p = _ca(atoms)
        if p is not None:
            ca[i] = p
        names.append(_resname(atoms))
    ok = np.isfinite(ca[:, 0])
    if ok.sum() < 3:
        return X

    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = (d <= CONTACT_CA) & ok[:, None] & ok[None, :]
    seq = np.asarray(resseqs, dtype=np.int64)
    seqd = np.abs(seq[:, None] - seq[None, :])

    # Precompute backbone H-bond partners.
    bb_o = [ _bb_atom(atoms_by_res[i], "O") for i in range(n) ]
    bb_n = [ _bb_atom(atoms_by_res[i], "N") for i in range(n) ]

    depths = _hull_depths(np.nan_to_num(ca, nan=0.0))
    com = np.nanmean(ca[ok], axis=0)
    rad = np.linalg.norm(ca - com, axis=1)
    order_r = np.argsort(np.argsort(np.nan_to_num(rad, nan=0.0)))
    rad_rank = order_r / max(n - 1, 1)

    # Spectral / path features on the CA contact graph (once per chain).
    A = adj.astype(np.float64)
    deg_v = A.sum(axis=1)
    L = np.diag(deg_v) - A
    fiedler_rank = np.zeros(n)
    try:
        w, v = np.linalg.eigh(L)
        idx = np.argsort(w)
        fvec = np.abs(v[:, idx[1]]) if n >= 2 else np.zeros(n)
        fiedler_rank = np.argsort(np.argsort(fvec)) / max(n - 1, 1) * 100.0
    except np.linalg.LinAlgError:
        pass
    A3 = A @ A @ A
    n_paths3 = np.asarray(np.diag(A3), dtype=np.float64)  # closed walks length 3
    mean_path = np.zeros(n)
    ecc = np.zeros(n)
    lap_local = np.zeros(n)
    for i in range(n):
        # BFS distances up to 6
        dist = {i: 0}
        queue = [i]
        head = 0
        while head < len(queue):
            u = queue[head]; head += 1
            if dist[u] >= 6:
                continue
            for v in np.flatnonzero(A[u]):
                v = int(v)
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if len(dist) > 1:
            ds = [d for k, d in dist.items() if k != i]
            mean_path[i] = float(np.mean(ds))
            ecc[i] = float(max(ds))
        nbr = np.flatnonzero(A[i])
        nodes = np.concatenate([[i], nbr]) if len(nbr) else np.array([i])
        if len(nodes) >= 2:
            sub = A[np.ix_(nodes, nodes)]
            deg_s = sub.sum(axis=1)
            Ls = np.diag(deg_s) - sub
            try:
                lap_local[i] = float(np.sum(np.abs(np.linalg.eigvalsh(Ls))))
            except np.linalg.LinAlgError:
                pass

    for i in range(n):
        if not ok[i]:
            continue
        nbr = np.flatnonzero(adj[i])
        deg = int(len(nbr))
        spans = seqd[i, nbr] if deg else np.zeros(0, dtype=np.int64)
        n_lt4 = int((spans < 4).sum()) if deg else 0
        n_4_12 = int(((spans >= 4) & (spans <= 12)).sum()) if deg else 0
        n_gt12 = int((spans > 12).sum()) if deg else 0
        n_gt24 = int((spans > 24).sum()) if deg else 0
        max_span = int(spans.max()) if deg else 0
        mean_span = float(spans.mean()) if deg else 0.0
        if deg and max_span > 0:
            # coarse histogram entropy of span buckets
            buckets = np.array([
                n_lt4, n_4_12, n_gt12 - n_gt24, n_gt24
            ], dtype=np.float64)
            p = buckets / buckets.sum().clip(1)
            p = p[p > 0]
            ent = float(-(p * np.log(p)).sum())
        else:
            ent = 0.0
        frac_long = 100.0 * n_gt12 / deg if deg else 0.0

        # triangles / clustering among neighbours
        tri = 0
        wedges = 0
        if deg >= 2:
            sub = adj[np.ix_(nbr, nbr)]
            tri = int(sub.sum() // 2)
            wedges = deg * (deg - 1) // 2
        clust = 100.0 * tri / wedges if wedges else 0.0

        # packing
        n_ca6 = int(((d[i] <= PACK_CA) & ok).sum())
        sc_i = _sc_heavy(atoms_by_res[i])
        n_sc5 = 0
        if len(sc_i):
            for j in range(n):
                if i == j or not ok[j]:
                    continue
                sc_j = _sc_heavy(atoms_by_res[j])
                if not len(sc_j):
                    continue
                if np.linalg.norm(sc_i[:, None, :] - sc_j[None, :, :],
                                  axis=-1).min() <= 5.0:
                    n_sc5 += 1

        # nonlocal backbone H-bond geometry (not Seq±1)
        n_o = n_n = 0
        oi = bb_o[i]
        ni = bb_n[i]
        for j in nbr:
            if abs(int(seq[i]) - int(seq[j])) <= 1:
                continue
            oj, nj = bb_o[j], bb_n[j]
            if oi is not None and nj is not None:
                if np.linalg.norm(oi - nj) <= HBOND_CUT:
                    n_o += 1
            if ni is not None and oj is not None:
                if np.linalg.norm(ni - oj) <= HBOND_CUT:
                    n_n += 1
        # frustration: buried (many CA neighbours) but few H-bonds
        frustr = max(0, n_ca6 - 4 - (n_o + n_n))

        # backbone angle / torsion
        ang_dev = tors = step = 0.0
        if 0 < i < n - 1 and ok[i - 1] and ok[i + 1]:
            v1, v2 = ca[i] - ca[i - 1], ca[i + 1] - ca[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
                ang = float(np.degrees(np.arccos(c)))
                ang_dev = abs(ang - 110.0)  # peptide CA angle prior
                step = 0.5 * (n1 + n2)
        if 1 < i < n - 2 and ok[i - 2] and ok[i - 1] and ok[i + 1]:
            b1 = ca[i - 1] - ca[i - 2]
            b2 = ca[i] - ca[i - 1]
            b3 = ca[i + 1] - ca[i]
            n12 = np.cross(b1, b2)
            n23 = np.cross(b2, b3)
            nn12, nn23 = np.linalg.norm(n12), np.linalg.norm(n23)
            if nn12 > 1e-8 and nn23 > 1e-8:
                c = float(np.clip(np.dot(n12, n23) / (nn12 * nn23), -1, 1))
                tors = abs(float(np.degrees(np.arccos(c))))

        # neighbourhood inertia
        rg = aniso = 0.0
        if deg >= 3:
            P = ca[nbr] - ca[nbr].mean(axis=0)
            C = (P.T @ P) / deg
            w = np.sort(np.linalg.eigvalsh(C))
            rg = float(np.sqrt(max(w.sum(), 0.0)))
            aniso = 100.0 * (w[2] - w[0]) / max(w.sum(), 1e-9)

        # neighbour composition
        chi_sum = n_gly = n_pro = n_aro = n_chg = n_hyd = n_pol = 0
        qpos = qneg = 0
        for j in nbr:
            aa = names[j]
            chi_sum += CHI.get(aa, 0)
            n_gly += int(aa == "GLY")
            n_pro += int(aa == "PRO")
            n_aro += int(aa in AROM)
            q = CHARGE.get(aa, 0)
            if q > 0:
                qpos += 1
                n_chg += 1
            elif q < 0:
                qneg += 1
                n_chg += 1
            n_hyd += int(aa in HYDRO)
            n_pol += int(aa in POLAR)
        chi_i = CHI.get(names[i], 0)
        burial_chi = n_ca6 * chi_i
        long_chi = n_gt12 * chi_i
        core_seam = int(depths[i] >= 1 and n_gt12 >= 1 and chi_sum >= 4)

        row = {
            "n_ca_contacts": deg,
            "n_contacts_seq_lt_4": n_lt4,
            "n_contacts_seq_4_12": n_4_12,
            "n_contacts_seq_gt_12": n_gt12,
            "n_contacts_seq_gt_24": n_gt24,
            "max_seq_span_contact": max_span,
            "mean_seq_span_x10": mean_span * 10.0,
            "seq_span_entropy_x100": ent * 100.0,
            "frac_long_contacts_x100": frac_long,
            "degree": deg,
            "n_triangles": tri,
            "clustering_x100": clust,
            "n_wedges": wedges,
            "n_ca_within_6A": n_ca6,
            "n_sc_heavy_within_5A": n_sc5,
            "n_bb_o_nonlocal_hbond": n_o,
            "n_bb_n_nonlocal_hbond": n_n,
            "hbond_frustration": frustr,
            "ca_angle_dev_x100": ang_dev,
            "ca_torsion_abs_x100": tors,
            "ca_step_length_x100": step * 100.0,
            "neigh_rg_x100": rg * 100.0,
            "neigh_anisotropy_x100": aniso,
            "radial_pct_x100": rad_rank[i] * 100.0,
            "hull_depth": int(depths[i]),
            "on_convex_hull": int(depths[i] == 0),
            "neigh_chi_sum": chi_sum,
            "neigh_n_gly": n_gly,
            "neigh_n_pro": n_pro,
            "neigh_n_aromatic": n_aro,
            "neigh_n_charged": n_chg,
            "neigh_charge_imbalance": abs(qpos - qneg),
            "neigh_n_hydrophobic": n_hyd,
            "neigh_n_polar": n_pol,
            "burial_times_chi": burial_chi,
            "long_contact_times_chi": long_chi,
            "core_seam_flag": core_seam,
            "fiedler_rank_x100": float(fiedler_rank[i]),
            "laplacian_energy_local": float(lap_local[i]),
            "mean_path_len_x10": float(mean_path[i] * 10.0),
            "n_paths3": float(n_paths3[i]),
            "eccentricity": float(ecc[i]),
        }
        for k, v in row.items():
            X[i, _COL[k]] = v

    # within-chain ranks
    def _rank(col: str) -> np.ndarray:
        v = X[:, _COL[col]]
        return np.argsort(np.argsort(v)) / max(n - 1, 1) * 100.0

    X[:, _COL["rank_max_seq_span"]] = _rank("max_seq_span_contact")
    X[:, _COL["rank_hull_depth"]] = _rank("hull_depth")
    X[:, _COL["rank_hbond_frustration"]] = _rank("hbond_frustration")
    X[:, _COL["rank_burial_times_chi"]] = _rank("burial_times_chi")
    X[:, _COL["rank_long_contacts"]] = _rank("n_contacts_seq_gt_12")
    X[:, _COL["rank_clustering"]] = _rank("clustering_x100")
    return X


def consistency(X: np.ndarray) -> list[str]:
    bad = []
    if X.shape[1] != N_COLUMNS:
        bad.append(f"width {X.shape[1]} != {N_COLUMNS}")
    if np.isnan(X).any():
        bad.append("nan")
    return bad
