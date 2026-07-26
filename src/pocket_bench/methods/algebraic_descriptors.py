"""Exact algebraic and topological per-residue surface descriptors.

Thirty-five invariants, every one of them a closed-form algebraic or integer
topological quantity of the receptor alone. No fitted parameter, no gradient, no
RNG, no neural feature, no ligand.

Why thirty-five
---------------
The six-invariant word reaches ROC-AUC 0.7583 and P2Rank reaches 0.7930 with
roughly thirty-five descriptors. The gap is descriptive richness, not model
class. This module closes the richness gap without importing a model class: the
added quantities are eigenvalues of local Laplacians, Betti numbers of local
lining graphs, closed-form field gradients and inertia-tensor ratios.

The six groups
--------------
G1 surface exposure       (6)  occlusion and protrusion of the residue itself
G2 local spectral geometry(6)  Laplacian spectrum and Betti numbers of the
                               induced neighbourhood graph
G3 density field calculus (6)  closed-form gradient/Laplacian of the packing
                               density field rho
G4 ultrametric structure  (6)  non-Archimedean ball geometry
G5 curvature / anisotropy (6)  inertia-tensor ratios and normal divergence
G6 global position        (5)  depth, coordination, contact order

Grouping is not cosmetic. It is the partition the cascaded compiler consumes:
each group is compiled into its own dense quaternary table, so no group may
exceed six digits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pocket_bench.methods.density_topology import (
    MIN_BALL_ATOMS,
    branch_hinge,
    rotor_apply,
    ultrametric_balls,
)
from pocket_bench.methods.geometric_foundation import (
    _buriedness,
    _fibonacci_directions,
    _free_grid,
)
from pocket_bench.methods.sequence_wires import residue_codes
from pocket_bench.pdb_io import assert_no_hetatm, parse_pdb_atoms
from pocket_bench.spatial import cross_within, self_pairs_within

_ROTOR_THETA = 0.20
_MAX_HINGE_BALLS = 8
_NBR_R = 10.0             # residue-residue contact scale, same as everywhere
_MAX_LOCAL = 48           # cap on the induced subgraph order
_PINCH_R = 7.0            # lining radius at which a cleft actually disconnects

GROUPS: tuple[tuple[str, ...], ...] = (
    ("bur", "void", "probe_max", "protrusion", "exposure", "concavity"),
    ("fiedler", "lap_max", "spec_gap", "mean_degree", "betti0", "betti1"),
    ("rho", "grad_rho", "grad_radial", "lap_rho", "var_rho", "rho_rank_ball"),
    ("loose", "ball_size", "ball_radius", "ball_interface", "valuation",
     "rotor_gain"),
    ("anisotropy", "planarity", "sphericity", "normal_div", "angle_deficit",
     "hess_bur"),
    ("depth", "coordination", "contact_order", "ball_offset", "depth_rank"),
)
FEATURE_NAMES: tuple[str, ...] = tuple(n for g in GROUPS for n in g)
GROUP_SIZES: tuple[int, ...] = tuple(len(g) for g in GROUPS)
N_ALGEBRAIC = len(FEATURE_NAMES)          # 35


def _adjacency(ctr: np.ndarray, radius: float = _NBR_R):
    """Undirected residue contact graph as sorted CSR arrays."""
    n = len(ctr)
    pairs, _ = self_pairs_within(np.ascontiguousarray(ctr), radius)
    if len(pairs) == 0:
        return (np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int64))
    i = np.concatenate([pairs[:, 0], pairs[:, 1]])
    j = np.concatenate([pairs[:, 1], pairs[:, 0]])
    order = np.argsort(i, kind="stable")
    i, j = i[order], j[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, i + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, j


def _local_spectral(ctr, indptr, indices):
    """Laplacian spectrum and Betti numbers of every induced neighbourhood.

    For residue i let S_i be i together with its contacts. The induced subgraph
    G[S_i] is the local lining. Its combinatorial Laplacian L = D - A is real
    symmetric, so its spectrum is exact and ordered; lambda_2 is the algebraic
    connectivity of the lining and lambda_max its spectral radius. The Betti
    numbers are integer counts: b0 is the number of connected components of the
    lining with i deleted (how many disjoint walls face this residue) and
    b1 = E - V + C is the cycle rank (how many independent loops enclose it).
    A residue at the mouth of a cryptic cleft sees a lining that is well
    connected around the cleft but pinched across it, which is exactly a low
    lambda_2 at high b1.
    """
    n = len(ctr)
    fiedler = np.zeros(n)
    lap_max = np.zeros(n)
    mean_deg = np.zeros(n)
    betti0 = np.zeros(n)
    betti1 = np.zeros(n)
    for i in range(n):
        nb = indices[indptr[i]:indptr[i + 1]]
        if nb.size == 0:
            continue
        if nb.size > _MAX_LOCAL:                       # nearest ones, exactly
            d2 = ((ctr[nb] - ctr[i]) ** 2).sum(1)
            nb = nb[np.argsort(d2, kind="stable")[:_MAX_LOCAL]]
        S = np.concatenate([[i], nb])
        k = len(S)
        P = ctr[S]
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        A = (d2 <= _NBR_R * _NBR_R).astype(np.float64)
        np.fill_diagonal(A, 0.0)
        deg = A.sum(1)
        L = np.diag(deg) - A
        ev = np.linalg.eigvalsh(L)                     # exact, ascending
        fiedler[i] = float(ev[1]) if k > 1 else 0.0
        lap_max[i] = float(ev[-1])
        e_cnt = 0.5 * float(A.sum())
        mean_deg[i] = 2.0 * e_cnt / k
        # Components of the lining with the centre deleted. At the 10 A
        # contact radius the lining of a globular protein is always connected,
        # so b0 measured there is the constant 1 and carries no information;
        # the pinch that a cleft creates is only visible at the tighter radius
        # where the two walls stop touching each other.
        Ap = (d2 <= _PINCH_R * _PINCH_R).astype(np.float64)
        np.fill_diagonal(Ap, 0.0)
        B = Ap[1:, 1:]
        m = k - 1
        if m <= 0:
            betti0[i] = 0.0
            betti1[i] = 0.0
            continue
        seen = np.zeros(m, dtype=bool)
        comps = 0
        for s in range(m):
            if seen[s]:
                continue
            comps += 1
            stack = [s]
            seen[s] = True
            while stack:
                a = stack.pop()
                for b in np.nonzero(B[a])[0]:
                    if not seen[b]:
                        seen[b] = True
                        stack.append(int(b))
        betti0[i] = comps
        e_sub = 0.5 * float(B.sum())
        betti1[i] = e_sub - m + comps                  # cycle rank, an integer
    return fiedler, lap_max, mean_deg, betti0, betti1


def _neighbour_sums(indptr, indices, x):
    """Sum of ``x`` over each residue's contacts, and the contact count."""
    n = len(indptr) - 1
    cnt = np.diff(indptr).astype(np.float64)
    if len(indices) == 0:
        return np.zeros(n), cnt
    src = np.repeat(np.arange(n), np.diff(indptr))
    s = np.zeros(n)
    np.add.at(s, src, x[indices])
    return s, cnt


def _batched_gradient(ctr, rho_res, indptr, indices):
    """Closed-form least-squares gradient of the density field per residue.

    Over the neighbourhood the field is modelled by its first-order Taylor
    polynomial, and the normal equations of that fit are a single symmetric 3x3
    system per residue,

        A_i g_i = b_i,   A_i = sum_j (c_j - c_i)(c_j - c_i)^T,
                         b_i = sum_j (rho_j - rho_i)(c_j - c_i),

    solved in closed form. This is one tensor inversion, not an optimisation.
    """
    n = len(ctr)
    A = np.zeros((n, 3, 3))
    b = np.zeros((n, 3))
    if len(indices):
        src = np.repeat(np.arange(n), np.diff(indptr))
        d = ctr[indices] - ctr[src]
        dv = rho_res[indices] - rho_res[src]
        outer = d[:, :, None] * d[:, None, :]
        np.add.at(A, src, outer)
        np.add.at(b, src, dv[:, None] * d)
    tr = np.trace(A, axis1=1, axis2=2)
    reg = np.maximum(tr, 1e-12) * 1e-9
    A = A + np.eye(3)[None] * reg[:, None, None]
    try:
        g = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        g = np.zeros((n, 3))
    return g


def algebraic_residue_features(
    receptor_pdb: Path, *, chain: str | None = None,
    grid_step: float = 1.5, n_dirs: int = 30, cutoff: float = 11.0,
    perp: float = 1.8, atom_r: float = 2.6, max_pts: int = 6000,
):
    """``(resseq, F, codes, centroids)`` with F of shape (R, 35).

    Column order is exactly ``FEATURE_NAMES``, i.e. group-major, which is the
    order the cascaded compiler slices.
    """
    path = Path(receptor_pdb)
    assert_no_hetatm(path)
    atoms = parse_pdb_atoms(path.read_text(errors="ignore"))
    sel = [a for a in atoms
           if a["record"] == "ATOM" and a["element"] != "H"
           and (chain is None or a["chain"] == chain)]
    if len(sel) < 50:
        raise ValueError("too few receptor atoms")
    coords = np.asarray([[a["x"], a["y"], a["z"]] for a in sel], dtype=np.float64)
    res_of_atom = np.asarray([a["resseq"] for a in sel], dtype=np.int64)
    resseq, inv = np.unique(res_of_atom, return_inverse=True)
    n_res = len(resseq)
    _rn: dict[int, str] = {}
    for a, r in zip(sel, inv):
        _rn.setdefault(int(r), a["resname"])
    codes = residue_codes([_rn.get(i, "UNK") for i in range(n_res)])

    balls = ultrametric_balls(coords)
    labels = balls["labels"]
    rho = balls["rho"].astype(np.float64)
    tau = float(balls["tau"]) or 1.0
    tree_w = np.asarray(balls["tree_w"], dtype=np.float64)

    dirs = _fibonacci_directions(n_dirs)
    pts = _free_grid(coords, grid_step, atom_r, max_pts)
    if len(pts) == 0:
        raise ValueError("no_free_points")
    bur = _buriedness(pts, coords, dirs, cutoff, perp)

    cnt_at = np.zeros(n_res)
    np.add.at(cnt_at, inv, 1.0)
    cnt_at = np.maximum(cnt_at, 1.0)

    ctr = np.zeros((n_res, 3))
    np.add.at(ctr, inv, coords)
    ctr /= cnt_at[:, None]

    # ---- G1 surface exposure -------------------------------------------------
    near_r = 6.0
    bur_sum = np.zeros(n_res); bur_cnt = np.zeros(n_res)
    void_cnt = np.zeros(n_res); probe_max = np.zeros(n_res)
    pi, ai = cross_within(pts, coords, near_r)
    if pi.size:
        r_of = inv[ai]
        b = bur[pi]
        np.add.at(bur_sum, r_of, b)
        np.add.at(bur_cnt, r_of, 1.0)
        np.add.at(void_cnt, r_of, (b >= 0.55).astype(np.float64))
        np.maximum.at(probe_max, r_of, b)
    f_bur = np.where(bur_cnt > 0, bur_sum / np.maximum(bur_cnt, 1.0), 0.0)
    f_void = np.log1p(void_cnt)

    exp_cnt = np.zeros(n_res)
    pi5, ai5 = cross_within(pts, coords, 5.0)
    if pi5.size:
        np.add.at(exp_cnt, inv[ai5], 1.0)
    f_exposure = exp_cnt / cnt_at

    indptr, indices = _adjacency(ctr, _NBR_R)
    nb_cnt = np.diff(indptr).astype(np.float64)
    nb_sum_c = np.zeros((n_res, 3))
    if len(indices):
        src = np.repeat(np.arange(n_res), np.diff(indptr))
        np.add.at(nb_sum_c, src, ctr[indices])
    nb_mean_c = nb_sum_c / np.maximum(nb_cnt, 1.0)[:, None]
    f_protrusion = np.linalg.norm(nb_mean_c - ctr, axis=1) / _NBR_R

    protein_c = coords.mean(axis=0)
    rad = ctr - protein_c
    rad_n = rad / np.maximum(np.linalg.norm(rad, axis=1), 1e-9)[:, None]
    if len(indices):
        d_vec = ctr[indices] - ctr[src]
        d_len = np.maximum(np.linalg.norm(d_vec, axis=1), 1e-9)
        cos_out = (d_vec * rad_n[src]).sum(1) / d_len
        cs = np.zeros(n_res)
        np.add.at(cs, src, cos_out)
        f_concavity = cs / np.maximum(nb_cnt, 1.0)
    else:
        f_concavity = np.zeros(n_res)

    # ---- G2 local spectral geometry -----------------------------------------
    fiedler, lap_max, mean_deg, betti0, betti1 = _local_spectral(
        ctr, indptr, indices)
    spec_gap = fiedler / np.maximum(lap_max, 1e-9)

    # ---- G3 density field calculus -------------------------------------------
    f_rho = np.zeros(n_res)
    np.add.at(f_rho, inv, rho)
    f_rho /= cnt_at

    g = _batched_gradient(ctr, f_rho, indptr, indices)
    f_grad = np.linalg.norm(g, axis=1)
    f_grad_rad = (g * rad_n).sum(1)

    s_rho, _ = _neighbour_sums(indptr, indices, f_rho)
    mean_nb_rho = s_rho / np.maximum(nb_cnt, 1.0)
    f_lap_rho = mean_nb_rho - f_rho
    s_rho2, _ = _neighbour_sums(indptr, indices, f_rho ** 2)
    f_var_rho = np.maximum(s_rho2 / np.maximum(nb_cnt, 1.0) - mean_nb_rho ** 2,
                           0.0)

    ball_of_res = np.zeros(n_res, dtype=np.int64)
    for a_i, r_i in zip(labels, inv):
        ball_of_res[r_i] = a_i
    f_rho_rank_ball = np.zeros(n_res)
    for bl in np.unique(ball_of_res):
        m = np.nonzero(ball_of_res == bl)[0]
        if m.size <= 1:
            f_rho_rank_ball[m] = 0.5
            continue
        o = np.argsort(f_rho[m], kind="stable")
        rk = np.empty(m.size)
        rk[o] = np.arange(m.size)
        f_rho_rank_ball[m] = rk / (m.size - 1)

    # ---- G4 ultrametric structure --------------------------------------------
    n_balls = int(labels.max()) + 1
    ball_mean_rho = np.zeros(n_balls); ball_cnt = np.zeros(n_balls)
    np.add.at(ball_mean_rho, labels, rho)
    np.add.at(ball_cnt, labels, 1.0)
    ball_mean_rho /= np.maximum(ball_cnt, 1.0)
    global_rho = float(rho.mean()) or 1.0
    atom_loose = ball_mean_rho[labels] / global_rho
    f_loose = np.zeros(n_res)
    np.add.at(f_loose, inv, atom_loose)
    f_loose /= cnt_at

    f_ball_size = np.log1p(ball_cnt[ball_of_res])
    # ultrametric radius of a ball: the largest tree weight it can contain, which
    # is tau by construction, scaled by the ball's own density deflation
    f_ball_radius = tau * (global_rho / np.maximum(ball_mean_rho[ball_of_res],
                                                   1e-9))
    # The residue's own non-Archimedean valuation: the largest density-deflated
    # edge it carries in the contact graph, in units of the global ball radius
    # tau. A residue on a loose hinge carries a long deflated edge; a residue in
    # the packed core carries only short ones. This is the p-adic valuation of
    # the residue read off the same metric the ball partition was built from.
    f_valuation = np.zeros(n_res)
    if len(indices):
        rho_max = float(f_rho.max()) or 1.0
        dd = np.linalg.norm(ctr[indices] - ctr[src], axis=1)
        defl = (rho_max / np.maximum(np.minimum(f_rho[indices], f_rho[src]),
                                     1e-9)) ** 0.5
        np.maximum.at(f_valuation, src, dd * defl)
    f_valuation = f_valuation / tau

    f_ball_iface = np.zeros(n_res)
    if len(indices):
        same = (ball_of_res[indices] != ball_of_res[src]).astype(np.float64)
        np.add.at(f_ball_iface, src, same)
    f_ball_iface = f_ball_iface / np.maximum(nb_cnt, 1.0)

    ball_ctr = np.zeros((n_balls, 3))
    np.add.at(ball_ctr, labels, coords)
    ball_ctr /= np.maximum(ball_cnt, 1.0)[:, None]
    rg = float(np.sqrt(((coords - protein_c) ** 2).sum(1).mean())) or 1.0
    f_ball_offset = np.linalg.norm(ctr - ball_ctr[ball_of_res], axis=1) / rg

    f_gain = np.zeros(n_res)
    sizes = np.bincount(labels)
    hinges = [b for b in np.argsort(-sizes)[:_MAX_HINGE_BALLS]
              if sizes[b] >= MIN_BALL_ATOMS and sizes[b] < len(coords)]
    for bidx in hinges:
        mask = labels == bidx
        c_h, ax = branch_hinge(coords, mask)
        moved = rotor_apply(coords, ax, _ROTOR_THETA, c_h, mask)
        d2c = ((pts - c_h) ** 2).sum(1)
        loc = np.where(d2c <= (cutoff + 8.0) ** 2)[0]
        if loc.size == 0:
            continue
        b2 = _buriedness(pts[loc], moved, dirs, cutoff, perp)
        gain = b2 - bur[loc]
        pil, ail = cross_within(pts[loc], coords, near_r)
        if pil.size == 0:
            continue
        gg = np.zeros(n_res)
        np.maximum.at(gg, inv[ail], gain[pil])
        f_gain = np.maximum(f_gain, gg)

    # ---- G5 curvature and anisotropy -----------------------------------------
    M = np.zeros((n_res, 3, 3))
    if len(indices):
        d = ctr[indices] - ctr[src]
        np.add.at(M, src, d[:, :, None] * d[:, None, :])
    M = M / np.maximum(nb_cnt, 1.0)[:, None, None]
    ev = np.linalg.eigvalsh(M)                       # ascending, batched, exact
    l3, l2, l1 = ev[:, 0], ev[:, 1], ev[:, 2]
    tr_M = np.maximum(l1 + l2 + l3, 1e-12)
    f_aniso = (l1 - l3) / tr_M
    f_planar = (l2 - l3) / np.maximum(l1, 1e-12)
    f_spher = l3 / np.maximum(l1, 1e-12)

    if len(indices):
        div = ((rad_n[indices] - rad_n[src]) * (ctr[indices] - ctr[src])).sum(1)
        dv = np.zeros(n_res)
        np.add.at(dv, src, div)
        f_normal_div = dv / np.maximum(nb_cnt, 1.0)
        cosang = (rad_n[indices] * rad_n[src]).sum(1)
        ang = np.arccos(np.clip(cosang, -1.0, 1.0))
        asum = np.zeros(n_res)
        np.add.at(asum, src, ang)
        # Mean angular spread of the neighbourhood's outward normals, as a
        # deficit against the flat half-turn. Bounded in [0,1] by construction.
        f_angle_deficit = 1.0 - (asum / np.maximum(nb_cnt, 1.0)) / np.pi
    else:
        f_normal_div = np.zeros(n_res)
        f_angle_deficit = np.ones(n_res)

    s_bur, _ = _neighbour_sums(indptr, indices, f_bur)
    f_hess_bur = s_bur / np.maximum(nb_cnt, 1.0) - f_bur

    # ---- G6 global position ---------------------------------------------------
    f_depth = np.linalg.norm(ctr - protein_c, axis=1) / rg
    f_coord = nb_cnt.astype(np.float64)
    if len(indices):
        seq_sep = np.abs(resseq[indices] - resseq[src]).astype(np.float64)
        ss = np.zeros(n_res)
        np.add.at(ss, src, seq_sep)
        f_contact_order = ss / np.maximum(nb_cnt, 1.0) / max(n_res, 1)
    else:
        f_contact_order = np.zeros(n_res)
    o = np.argsort(f_depth, kind="stable")
    f_depth_rank = np.empty(n_res)
    f_depth_rank[o] = np.arange(n_res) / max(n_res - 1, 1)

    F = np.stack([
        f_bur, f_void, probe_max, f_protrusion, f_exposure, f_concavity,
        fiedler, lap_max, spec_gap, mean_deg, betti0, betti1,
        f_rho, f_grad, f_grad_rad, f_lap_rho, f_var_rho, f_rho_rank_ball,
        f_loose, f_ball_size, f_ball_radius, f_ball_iface, f_valuation, f_gain,
        f_aniso, f_planar, f_spher, f_normal_div, f_angle_deficit, f_hess_bur,
        f_depth, f_coord, f_contact_order, f_ball_offset, f_depth_rank,
    ], axis=1)
    assert F.shape[1] == N_ALGEBRAIC, (F.shape, N_ALGEBRAIC)
    return resseq, F, codes, ctr
