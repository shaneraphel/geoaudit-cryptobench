"""Four descriptor families that the contact-graph generator cannot produce.

Why a second module rather than more of the first
--------------------------------------------------
``operator_descriptors`` builds everything from two objects: the Laplacian of
the residue contact graph, and the second moment of the neighbourhood point
cloud. Both are invariant under permuting residues. That is a strong symmetry
and it is the wrong one here, because it throws away two things the receptor
actually has.

It throws away the chain. A protein is a curve in space, and the contact graph
is the curve with the curve deleted. No function of that graph's spectrum can
recover which neighbours are sequence-adjacent and which arrived from four
hundred residues away, and that distinction is close to the definition of the
structures this benchmark asks about: a lid, a loop clamped over a cleft, two
segments that touch only because the protein is folded.

It throws away the sign of curvature. The second moment of a point cloud is a
positive-definite tensor; it measures how spread the neighbourhood is along
three axes and cannot say whether the surface through those points curves
toward the residue or away from it. A flat patch and a saddle with the same
spread give the same tensor. The mouth of a cleft is a saddle.

So the four families here are, in order: the chain, the curvature, the
non-Archimedean ball profile, and the deformation. None of them is a variation
on the first module; each supplies information that is provably absent from it.

A. The local Toeplitz symbol
----------------------------
An operator whose entries depend only on ``i - j`` is Toeplitz. Let ``s(i)`` be
the chain position of residue ``i`` and, for a spatial neighbourhood ``N_r(i)``,
let

    g_i(k) = #{ j in N_r(i) : |s(i) - s(j)| = k }.

``g_i`` is the symbol of the Toeplitz part of the contact operator, localised at
``i``. Its low-order cosine coefficients

    ghat_i(m) = mean over j in N_r(i) of cos(pi m min(|s(i)-s(j)|, K0) / K0)

are taken with a cosine rather than a full transform because reversing the chain
must not change a geometric quantity, and with a fixed lag scale ``K0`` rather
than the chain length so that a 90-residue domain and a 600-residue one produce
comparable numbers. ``ghat(1)`` near one is a neighbourhood drawn from a single
run of sequence, a helix packed against itself; ``ghat(1)`` near zero is a
neighbourhood assembled from distant segments.

The integer companion is the segment count: the number of maximal runs of
consecutive chain positions among the neighbours. It is a small integer, it is
exactly computable, and it counts how many separate pieces of the chain have
been brought together at this residue.

B. The shape operator of the local surface
------------------------------------------
Take the neighbourhood cloud, put it in the frame of its own gyration tensor,
and call the smallest-eigenvalue direction the normal ``n``. Orient ``n``
outward, away from the chain's centroid, so that concave and convex are not
interchangeable. In coordinates ``(u, v, w)`` with ``w`` along ``n``, fit

    w = (1/2)(a u^2 + 2b uv + c v^2)

by least squares. The shape operator is ``S = [[a, b], [b, c]]``; its
eigenvalues are the principal curvatures ``k1 >= k2`` and from them

    Gaussian K = k1 k2,   mean H = (k1 + k2)/2,
    shape index = (2/pi) arctan((k1 + k2)/(k1 - k2)),
    curvedness  = sqrt((k1^2 + k2^2)/2).

The shape index is the scale-free coordinate on which cup, rut, saddle, ridge
and cap are separated intervals; a rut sits near -1/2 and a saddle at 0, and
those are the two values a cleft mouth takes. The residual of the fit is kept as
well, because it measures how badly the neighbourhood fails to be a surface at
all, which is what distinguishes a buried residue surrounded on all sides from
a surface residue on a flat patch.

C. The valuation profile
------------------------
Single linkage on the residue centroids induces an ultrametric: ``du(i, j)`` is
the largest edge on the path between them in the minimum spanning tree, and it
satisfies the strong triangle inequality ``du(i, k) <= max(du(i, j), du(j, k))``.
Under an ultrametric every point of a ball is its centre and balls are nested,
so the whole of the local structure is carried by the counting function

    B_i(t) = #{ j : du(i, j) <= t },

a step function whose jumps are the radii at which a new cluster is absorbed.
The existing bank samples this at one cut and keeps six numbers. Here the
profile is read at eight physical radii and the largest jump is recorded with
the radius at which it happens: a residue whose two walls join only at a large
``t`` is a residue between two walls.

D. Participation and shear of the soft modes
--------------------------------------------
For the anisotropic network model the Hessian is the block Laplacian, and its
low non-rigid modes are the deformations the fold is softest against. For mode
``m`` with displacement field ``u^m`` and eigenvalue ``lambda_m``, define at
residue ``i``

    participation  a_i^m = |u_i^m|^2
    local shear    sigma_i^m = mean over j in N(i) of |u_i^m - u_j^m|^2.

The shear is the quadratic form the Hessian itself penalises, localised. The
distinction matters and is the reason an earlier attempt with participation
alone bought almost nothing: a **lid** has high participation, a **hinge** has
low participation and high shear -- it barely moves while everything around it
moves relative to it -- and a rigid core interior has both low. Only the pair
separates the three, so the ratio ``sigma / a`` is carried explicitly as the
hinge indicator, together with the thermal sums ``sum_m a^m / lambda_m`` and
``sum_m sigma^m / lambda_m`` in which soft modes dominate as they should.

Every quantity in this module is a closed-form algebraic or integer function of
the receptor's residue centroids and chain order. No fitted parameter, no
gradient, no RNG, no learned component, no ligand.
"""
from __future__ import annotations

import numpy as np

from pocket_bench.methods.anisotropic_shear_oracle import low_shear_modes
from pocket_bench.spatial import self_pairs_within

CHAIN_SCALES: tuple[float, ...] = (8.0, 12.0, 16.0)
LAG_SCALE = 50            # the lag at which the Toeplitz symbol closes a period
N_SYMBOL = 4              # cosine coefficients kept
FAR_LAG = 20              # "brought together by the fold, not by the sequence"
NEAR_LAG = 4              # one turn of a helix
VALUATION_LEVELS: tuple[float, ...] = (4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0)
N_MODES = 5
MODE_REPORT = 3           # modes reported individually; the rest enter the sums
SHEAR_R = 10.0
_EPS = 1e-12


def _chain_names(r: float) -> list[str]:
    s = f"@{r:g}"
    return ([f"lag_mean{s}", f"lag_max{s}", f"lag_entropy{s}",
             f"frac_lag_far{s}", f"frac_lag_near{s}"]
            + [f"symbol{m}{s}" for m in range(1, N_SYMBOL + 1)]
            + [f"n_segments{s}", f"segment_len_mean{s}"])


def _shape_names(r: float) -> list[str]:
    s = f"@{r:g}"
    return [f"kappa1{s}", f"kappa2{s}", f"gauss_curv{s}", f"mean_curv{s}",
            f"shape_index{s}", f"curvedness{s}", f"quadric_residual{s}"]


def _valuation_names() -> list[str]:
    return ([f"valuation_ball@{t:g}" for t in VALUATION_LEVELS]
            + ["valuation_jump_max", "valuation_jump_level",
               "ultrametric_eccentricity"])


def _mode_names() -> list[str]:
    out = [f"mode_participation{m}" for m in range(1, MODE_REPORT + 1)]
    out += [f"mode_shear{m}" for m in range(1, MODE_REPORT + 1)]
    out += ["thermal_participation", "thermal_shear", "hinge_ratio",
            "hinge_rank", "mode1_radial_cos", "relative_motion"]
    return out


def feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for r in CHAIN_SCALES:
        names.extend(_chain_names(r))
    for r in CHAIN_SCALES:
        names.extend(_shape_names(r))
    names.extend(_valuation_names())
    names.extend(_mode_names())
    return tuple(names)


FEATURE_NAMES: tuple[str, ...] = feature_names()
N_CHAIN_OPERATOR = len(FEATURE_NAMES)


def _csr(ctr: np.ndarray, radius: float):
    n = len(ctr)
    pairs, _ = self_pairs_within(np.ascontiguousarray(ctr), radius)
    if len(pairs) == 0:
        return np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    i = np.concatenate([pairs[:, 0], pairs[:, 1]])
    j = np.concatenate([pairs[:, 1], pairs[:, 0]])
    order = np.argsort(i, kind="stable")
    i, j = i[order], j[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, i + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, j


# ------------------------------------------------------- A. Toeplitz symbol
def _chain_block(ctr: np.ndarray, radius: float) -> np.ndarray:
    n = len(ctr)
    out = np.zeros((n, len(_chain_names(radius))), dtype=np.float64)
    indptr, indices = _csr(ctr, radius)
    for i in range(n):
        nb = indices[indptr[i]:indptr[i + 1]]
        if nb.size == 0:
            continue
        lag = np.abs(nb.astype(np.int64) - i)
        col = 0
        out[i, col] = float(lag.mean()); col += 1
        out[i, col] = float(lag.max()); col += 1
        # Entropy of the lag distribution over octave bins: a neighbourhood
        # drawn from one run of sequence has low entropy whatever its size.
        bins = np.bincount(np.minimum(
            np.floor(np.log2(np.maximum(lag, 1))).astype(int), 9), minlength=10)
        p = bins / max(bins.sum(), 1)
        p = p[p > 0]
        out[i, col] = float(-(p * np.log(p)).sum()); col += 1
        out[i, col] = float((lag > FAR_LAG).mean()); col += 1
        out[i, col] = float((lag <= NEAR_LAG).mean()); col += 1
        clipped = np.minimum(lag, LAG_SCALE) / LAG_SCALE
        for m in range(1, N_SYMBOL + 1):
            out[i, col] = float(np.cos(np.pi * m * clipped).mean()); col += 1
        run = np.sort(nb)
        breaks = int((np.diff(run) > 1).sum()) + 1
        out[i, col] = float(breaks); col += 1
        out[i, col] = float(len(run) / breaks); col += 1
    return out


# ------------------------------------------------------- B. shape operator
def _shape_block(ctr: np.ndarray, radius: float) -> np.ndarray:
    n = len(ctr)
    out = np.zeros((n, len(_shape_names(radius))), dtype=np.float64)
    indptr, indices = _csr(ctr, radius)
    global_centre = ctr.mean(0)
    for i in range(n):
        nb = indices[indptr[i]:indptr[i + 1]]
        if nb.size < 6:
            continue
        P = ctr[nb] - ctr[i]
        G = (P.T @ P) / len(P)
        w, V = np.linalg.eigh(G)
        normal = V[:, 0]
        # The normal's sign is free; fixing it outward is what makes concave and
        # convex distinguishable rather than a labelling accident.
        if float(normal @ (ctr[i] - global_centre)) < 0:
            normal = -normal
        e_u, e_v = V[:, 2], V[:, 1]
        u, v = P @ e_u, P @ e_v
        w_ax = P @ normal
        M = np.stack([0.5 * u * u, u * v, 0.5 * v * v], axis=1)
        try:
            coef, *_ = np.linalg.lstsq(M, w_ax, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a, b, c = (float(x) for x in coef)
        resid = float(np.sqrt(np.mean((M @ coef - w_ax) ** 2)))
        S = np.array([[a, b], [b, c]])
        k = np.linalg.eigvalsh(S)
        k2, k1 = float(k[0]), float(k[1])       # k1 >= k2
        col = 0
        out[i, col] = k1; col += 1
        out[i, col] = k2; col += 1
        out[i, col] = k1 * k2; col += 1
        out[i, col] = 0.5 * (k1 + k2); col += 1
        denom = k1 - k2
        out[i, col] = (2.0 / np.pi) * np.arctan((k1 + k2) / denom) \
            if abs(denom) > _EPS else 0.0
        col += 1
        out[i, col] = float(np.sqrt(0.5 * (k1 * k1 + k2 * k2))); col += 1
        out[i, col] = resid
    return out


# --------------------------------------------------- C. valuation profile
def _valuation_block(ctr: np.ndarray) -> np.ndarray:
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import cophenet

    n = len(ctr)
    out = np.zeros((n, len(_valuation_names())), dtype=np.float64)
    if n < 3:
        return out
    d = pdist(ctr)
    Z = linkage(d, method="single")
    du = squareform(cophenet(Z))
    for i in range(n):
        row = du[i]
        counts = np.array([float((row <= t).sum()) for t in VALUATION_LEVELS])
        col = 0
        for cval in counts:
            out[i, col] = float(np.log1p(cval)); col += 1
        jumps = np.diff(counts)
        if len(jumps):
            k = int(np.argmax(jumps))
            out[i, col] = float(jumps[k]); col += 1
            out[i, col] = float(VALUATION_LEVELS[k + 1]); col += 1
        else:
            col += 2
        out[i, col] = float(row.max())
    return out


# ------------------------------------------- D. soft-mode hinge descriptors
def _mode_block(ctr: np.ndarray) -> np.ndarray:
    n = len(ctr)
    out = np.zeros((n, len(_mode_names())), dtype=np.float64)
    if n < 8:
        return out
    try:
        modes, vals = low_shear_modes(ctr, k=N_MODES)
    except Exception:  # noqa: BLE001 -- ARPACK can fail on odd geometries
        return out
    modes = np.asarray(modes, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    if modes.ndim != 3 or modes.shape[1] != n:
        return out
    K = modes.shape[0]

    indptr, indices = _csr(ctr, SHEAR_R)
    deg = np.maximum(np.diff(indptr), 1).astype(np.float64)
    part = (modes ** 2).sum(-1)                       # (K, n)
    shear = np.zeros((K, n), dtype=np.float64)
    nbr_mean = np.zeros((K, n, 3), dtype=np.float64)
    for i in range(n):
        nb = indices[indptr[i]:indptr[i + 1]]
        if nb.size == 0:
            continue
        diff = modes[:, nb, :] - modes[:, i, :][:, None, :]
        shear[:, i] = (diff ** 2).sum(-1).mean(1)
        nbr_mean[:, i, :] = modes[:, nb, :].mean(1)

    inv = 1.0 / np.maximum(vals[:K], _EPS)
    therm_a = (part * inv[:, None]).sum(0)
    therm_s = (shear * inv[:, None]).sum(0)
    hinge = therm_s / (therm_a + _EPS)

    col = 0
    for m in range(MODE_REPORT):
        out[:, col] = part[m] if m < K else 0.0; col += 1
    for m in range(MODE_REPORT):
        out[:, col] = shear[m] if m < K else 0.0; col += 1
    out[:, col] = therm_a; col += 1
    out[:, col] = therm_s; col += 1
    out[:, col] = hinge; col += 1
    # Rank within the chain, so the indicator means the same thing in a rigid
    # protein and a floppy one.
    order = np.argsort(np.argsort(hinge, kind="stable"), kind="stable")
    out[:, col] = order / max(n - 1, 1); col += 1
    radial = ctr - ctr.mean(0)
    norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial = radial / np.maximum(norm, _EPS)
    u1 = modes[0]
    n1 = np.linalg.norm(u1, axis=1, keepdims=True)
    out[:, col] = np.abs((u1 / np.maximum(n1, _EPS) * radial).sum(1)); col += 1
    out[:, col] = ((modes - nbr_mean) ** 2).sum(-1).mean(0)
    return out


def chain_operator_residue_features(ctr: np.ndarray) -> np.ndarray:
    """``(n_residues, N_CHAIN_OPERATOR)`` for one chain.

    Residue order is chain order: the caller must pass centroids in the order
    the chain runs, because family A reads the index as the sequence position.
    """
    ctr = np.ascontiguousarray(np.asarray(ctr, dtype=np.float64))
    blocks = [_chain_block(ctr, r) for r in CHAIN_SCALES]
    blocks += [_shape_block(ctr, r) for r in CHAIN_SCALES]
    blocks.append(_valuation_block(ctr))
    blocks.append(_mode_block(ctr))
    F = np.concatenate(blocks, axis=1)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
