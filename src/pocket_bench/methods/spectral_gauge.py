"""Gauge-invariant spectral quantities on the residue contact graph.

The observation this family is built on
---------------------------------------
Let ``L = D - A`` be the combinatorial Laplacian of the CA contact graph, with
``L = sum_a lambda_a v_a v_a^T``. When ``lambda_a`` is simple, ``v_a`` is
determined only up to sign; on an eigenspace of multiplicity ``mu`` the basis is
determined only up to ``O(mu)``. Write

    G_L  =  prod_{simple} (Z/2)  x  prod_{degenerate} O(mu_j)

for that **gauge group**. A residue-level function built from the eigenvectors
is a *descriptor* only if it is ``G_L``-invariant. If it is not, its values
record the eigensolver's output convention, and a downstream detector inherits
an arbitrariness it cannot see.

``nonlocal_seam`` ranks ``|v_2(i)|``. That is invariant, but it is a gauge
*fixing* rather than the gauge-invariant content: taking the absolute value
throws away every relation between two residues, and the relations are where the
community structure of the contact graph lives. This family keeps the relations.

What is invariant, and why each one is
--------------------------------------
``P_k(i,i) = sum_{a<=k} v_a(i)^2``      the spectral projector diagonal; a sum of
                                        squares, so sign-blind, and basis-blind
                                        on a degenerate block because the
                                        projector onto an eigenspace does not
                                        depend on the basis chosen for it.

``P_k(i,j)``                            same argument off the diagonal. This is
                                        the quantity ``|v_2|`` discards.

``d_a(i) = #{ j ~ i : v_a(i) v_a(j) < 0 }``  the **discordance count**: how many
                                        of a residue's contacts sit on the other
                                        side of the a-th nodal partition. The
                                        product is fixed by ``v_a -> -v_a``, so
                                        the count is invariant, and it is an
                                        integer. For a degenerate block the
                                        individual signs are not defined and the
                                        invariant replacement is
                                        ``sign(P_block(i,j))``, which is what is
                                        computed there.

``R(i,j) = sum_{a>=2} (v_a(i) - v_a(j))^2 / lambda_a``   the resistance distance;
                                        a spectral function of the pair, and the
                                        standard nonlocal metric on a graph.

Not invariant, and therefore not used: ``v_a(i)``, ``sign(v_a(i))``, and the
within-chain rank of either.

Prediction, written before the run
----------------------------------
Two separate claims, with different expected answers, because conflating them is
how a family gets over-read.

*The field lift is expected to be null*, between -0.002 and +0.002.
``INDISTINGUISHABILITY_GROUP.json`` and ``BANK_TRUNCATION.json`` together say the
bank averages interchangeable correlated views rather than discriminating, and a
new view of the same contact graph is another correlated view. A lift above the
0.0026 reseed floor would falsify that reading, which is worth knowing either
way.

*The polarity diagnostic is the actual target.* Nineteen of the 192 official-fold
units anti-rank their cryptic residues; ``max(seam, -seam)`` chosen per unit
reaches 0.874 against pLM-NN's 0.8235, and no covariate screened so far
separates those nineteen from the rest by more than Cohen's ``d ~ 0.48``. If the
polarity defect is the shadow of an unfixed gauge, a ``G_L``-invariant column
should separate the nineteen where the gauge-fixed one does not. That is a
falsifiable statement about a specific, named set of units, and it is the reason
to build this family despite expecting the field lift to be null.

``clinical_grade`` is false.
"""
from __future__ import annotations

import numpy as np

CONTACT_CA = 8.0
N_EIG = 8               # nontrivial eigenvectors carried
DEGENERACY_TOL = 1e-8   # eigenvalues closer than this share a block

COLUMNS = (
    # -- projector diagonals: local spectral density at three depths ----------
    "proj_diag_k2_x1e4",
    "proj_diag_k4_x1e4",
    "proj_diag_k8_x1e4",
    # -- discordance across the nodal partitions, one integer each -----------
    "discord_v2",
    "discord_v3",
    "discord_v4",
    "discord_v5_v8_sum",
    "discord_total_k8",
    "discord_frac_k8_x100",
    # -- the same, weighted by how far into the spectrum the disagreement is --
    "discord_weighted_x100",
    "first_discordant_mode",
    "n_modes_fully_concordant",
    # -- projector off-diagonal aggregated over the contact neighbourhood -----
    "proj_offdiag_sum_k8_x1e4",
    "proj_offdiag_negative_count",
    "proj_offdiag_min_x1e4",
    # -- resistance distance: nonlocal, and its local statistics --------------
    "resist_mean_contacts_x1e3",
    "resist_max_contacts_x1e3",
    "resist_to_centroid_x1e3",
    "resist_rank_x100",
    # -- nodal-domain combinatorics -------------------------------------------
    "nodal_domain_size_v2",
    "nodal_domain_is_minority_v2",
    "on_nodal_boundary_v2",
    "n_boundary_contacts_k4",
    # -- within-chain ranks of the load-bearing members -----------------------
    "rank_discord_total",
    "rank_proj_diag_k8",
    "rank_resist_mean",
)
N_COLUMNS = len(COLUMNS)
_COL = {n: j for j, n in enumerate(COLUMNS)}


def _ca_coords(atoms_by_res: list[list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    n = len(atoms_by_res)
    ca = np.full((n, 3), np.nan)
    for i, atoms in enumerate(atoms_by_res):
        for a in atoms:
            if (a.get("name") or "").strip().upper() == "CA":
                ca[i] = (a["x"], a["y"], a["z"])
                break
    return ca, np.isfinite(ca[:, 0])


def compute(atoms_by_res: list[list[dict]], resseqs: list[int]) -> np.ndarray:
    """Return an ``(n_res, N_COLUMNS)`` matrix for one chain."""
    n = len(atoms_by_res)
    X = np.zeros((n, N_COLUMNS), dtype=np.float64)
    if n < 4:
        return X

    ca, ok = _ca_coords(atoms_by_res)
    if ok.sum() < 4:
        return X
    pts = np.nan_to_num(ca, nan=0.0)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    A = ((d <= CONTACT_CA) & ok[:, None] & ok[None, :]).astype(np.float64)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    try:
        w, V = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return X

    order = np.argsort(w)
    w, V = w[order], V[:, order]
    # Drop the constant mode(s): one per connected component, eigenvalue ~0.
    n_null = int((w < 1e-9).sum())
    k = min(N_EIG, max(0, len(w) - n_null))
    if k < 2:
        return X
    lam = w[n_null:n_null + k]
    U = V[:, n_null:n_null + k]

    # ---- gauge-invariant primitives ----------------------------------------
    # Projector onto the span of the first m carried modes. Basis-independent
    # by construction, so degeneracy needs no special handling here.
    def proj(m: int) -> np.ndarray:
        return U[:, :m] @ U[:, :m].T

    P2, P4, P8 = proj(min(2, k)), proj(min(4, k)), proj(k)

    # Discordance is sign-invariant per mode, but only well defined when the
    # eigenvalue is simple. Group near-equal eigenvalues and, inside a block,
    # use the sign of the block projector's off-diagonal, which is the
    # O(mu)-invariant replacement for a per-vector sign product.
    blocks: list[list[int]] = []
    for a in range(k):
        if blocks and abs(lam[a] - lam[blocks[-1][-1]]) <= DEGENERACY_TOL:
            blocks[-1].append(a)
        else:
            blocks.append([a])

    disc = np.zeros((n, k), dtype=np.int64)
    for blk in blocks:
        if len(blk) == 1:
            a = blk[0]
            prod = np.outer(U[:, a], U[:, a])
            neg = (prod < 0) & (A > 0)
            disc[:, a] = neg.sum(axis=1)
        else:
            Pb = U[:, blk] @ U[:, blk].T
            neg = (Pb < 0) & (A > 0)
            c = neg.sum(axis=1)
            for a in blk:
                disc[:, a] = c

    # Resistance distance restricted to contacts. Uses the pseudo-inverse of L
    # through the carried spectrum, which is the standard truncation.
    inv_lam = 1.0 / np.maximum(lam, 1e-12)
    Uw = U * inv_lam[None, :]
    Lp_diag = (U * Uw).sum(axis=1)
    Lp = U @ Uw.T
    R = Lp_diag[:, None] + Lp_diag[None, :] - 2.0 * Lp
    np.fill_diagonal(R, 0.0)

    com_idx = int(np.argmin(np.linalg.norm(pts - pts[ok].mean(axis=0), axis=1)))

    disc_tot = disc.sum(axis=1)
    weights = np.arange(1, k + 1, dtype=np.float64)
    disc_w = (disc * weights[None, :]).sum(axis=1)

    for i in range(n):
        if not ok[i]:
            continue
        nbr = np.flatnonzero(A[i])
        dgi = len(nbr)
        first_bad = 0
        for a in range(k):
            if disc[i, a] > 0:
                first_bad = a + 1
                break
        off = P8[i, nbr] if dgi else np.zeros(0)
        row = {
            "proj_diag_k2_x1e4": P2[i, i] * 1e4,
            "proj_diag_k4_x1e4": P4[i, i] * 1e4,
            "proj_diag_k8_x1e4": P8[i, i] * 1e4,
            "discord_v2": int(disc[i, 0]),
            "discord_v3": int(disc[i, 1]) if k > 1 else 0,
            "discord_v4": int(disc[i, 2]) if k > 2 else 0,
            "discord_v5_v8_sum": int(disc[i, 4:].sum()) if k > 4 else 0,
            "discord_total_k8": int(disc_tot[i]),
            "discord_frac_k8_x100": (100.0 * disc_tot[i] / (dgi * k)
                                     if dgi else 0.0),
            "discord_weighted_x100": disc_w[i] * 100.0 / max(dgi, 1),
            "first_discordant_mode": first_bad,
            "n_modes_fully_concordant": int((disc[i] == 0).sum()),
            "proj_offdiag_sum_k8_x1e4": float(off.sum()) * 1e4,
            "proj_offdiag_negative_count": int((off < 0).sum()),
            "proj_offdiag_min_x1e4": (float(off.min()) * 1e4 if dgi else 0.0),
            "resist_mean_contacts_x1e3": (float(R[i, nbr].mean()) * 1e3
                                          if dgi else 0.0),
            "resist_max_contacts_x1e3": (float(R[i, nbr].max()) * 1e3
                                         if dgi else 0.0),
            "resist_to_centroid_x1e3": float(R[i, com_idx]) * 1e3,
            "resist_rank_x100": 0.0,
            "nodal_domain_size_v2": 0,
            "nodal_domain_is_minority_v2": 0,
            "on_nodal_boundary_v2": int(disc[i, 0] > 0),
            "n_boundary_contacts_k4": int(disc[i, :min(4, k)].sum()),
            "rank_discord_total": 0.0,
            "rank_proj_diag_k8": 0.0,
            "rank_resist_mean": 0.0,
        }
        for key, v in row.items():
            X[i, _COL[key]] = v

    # Nodal domain of the Fiedler vector. The *sizes* of the two sides are
    # sign-invariant as an unordered pair, so "which side am I on" is not a
    # descriptor but "how big is my side" and "am I in the smaller one" are.
    s = np.sign(U[:, 0])
    pos, neg = int((s > 0).sum()), int((s < 0).sum())
    minority = 1 if pos <= neg else -1
    for i in range(n):
        if not ok[i]:
            continue
        X[i, _COL["nodal_domain_size_v2"]] = pos if s[i] > 0 else neg
        X[i, _COL["nodal_domain_is_minority_v2"]] = int(s[i] == minority)

    def _rank(col: str) -> np.ndarray:
        v = X[:, _COL[col]]
        return np.argsort(np.argsort(v)) / max(n - 1, 1) * 100.0

    X[:, _COL["rank_discord_total"]] = _rank("discord_total_k8")
    X[:, _COL["rank_proj_diag_k8"]] = _rank("proj_diag_k8_x1e4")
    X[:, _COL["rank_resist_mean"]] = _rank("resist_mean_contacts_x1e3")
    X[:, _COL["resist_rank_x100"]] = _rank("resist_mean_contacts_x1e3")
    return X


def consistency(X: np.ndarray) -> list[str]:
    bad = []
    if X.shape[1] != N_COLUMNS:
        bad.append(f"width {X.shape[1]} != {N_COLUMNS}")
    if not np.isfinite(X).all():
        bad.append("non-finite")
    return bad


def gauge_selftest(atoms_by_res: list[list[dict]], resseqs: list[int],
                   seed: int = 0) -> dict:
    """Every column must be unchanged when the eigenbasis signs are flipped.

    This is the test the family exists for, so it ships with the family rather
    than in a separate file that might not be run. It re-computes the matrix
    with a perturbed but mathematically equivalent eigenbasis — obtained by
    negating a random subset of eigenvectors — and requires bit-level agreement
    up to floating-point tolerance.

    A column that fails this is not a descriptor. It is a record of what the
    eigensolver happened to return.
    """
    X = compute(atoms_by_res, resseqs)
    rng = np.random.default_rng(seed)

    real_eigh = np.linalg.eigh

    def flipped(a):
        w, V = real_eigh(a)
        flip = rng.choice([-1.0, 1.0], size=V.shape[1])
        return w, V * flip[None, :]

    np.linalg.eigh = flipped
    try:
        Y = compute(atoms_by_res, resseqs)
    finally:
        np.linalg.eigh = real_eigh

    delta = np.abs(X - Y).max(axis=0)
    bad = [COLUMNS[j] for j in range(N_COLUMNS) if delta[j] > 1e-6]
    return {
        "gauge_invariant": not bad,
        "n_columns": N_COLUMNS,
        "n_columns_definition": "entries of spectral_gauge.COLUMNS, before "
                                "aggregation",
        "columns_failing_sign_flip": bad,
        "max_abs_deviation": float(delta.max()),
        "what_was_perturbed": ("a uniformly random sign was applied to each "
                               "Laplacian eigenvector, which produces a "
                               "mathematically equivalent eigenbasis"),
    }
