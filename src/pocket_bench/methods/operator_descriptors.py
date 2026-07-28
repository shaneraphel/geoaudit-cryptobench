"""Operator-spectral residue descriptors, generated systematically rather than
one at a time.

Why this module exists
----------------------
``algebraic_descriptors`` computes, for every residue, the exact eigendecomposi-
tion of its local contact Laplacian, and then keeps three numbers from it:
lambda_2, lambda_max and their ratio. The eigenvectors are discarded entirely
and the rest of the spectrum with them. It does this at one radius, 10 A,
because that is the radius the contact graph was defined at. The Fisher
discriminant of the resulting thirty-five invariants tops out below P2Rank, and
adding invariants four at a time has not moved it.

The bottleneck is not that good invariants are hard to find. It is that the
existing bank samples a three-dimensional corner of a much larger space that the
same arithmetic already traverses. This module enumerates that space instead:

    descriptor  =  operator family  x  scale  x  functional

with the operators built from residue centroids alone, so nothing here needs
atoms, a fitted parameter, an RNG or a ligand, and every quantity remains a
closed-form algebraic function of the receptor.

The three axes
--------------
*Scales.* Five radii from 6 to 16 A. Six angstroms is under one residue-residue
contact and resolves the wall of a cleft; sixteen spans the cleft and both walls
at once. A cryptic site is precisely a place where those two views disagree, so
the disagreement is made a descriptor in its own right (see below).

*Operators.* At each scale, the induced neighbourhood carries a Gaussian-weighted
adjacency, its combinatorial Laplacian, and its normalised Laplacian whose
spectrum lies in [0, 2] and is therefore comparable across residues of very
different degree. One symmetric eigendecomposition per residue per scale serves
all three.

*Functionals.* Two kinds, and the distinction matters. Trace functionals
(moments, heat traces, spectral entropy) describe the neighbourhood as a whole.
Diagonal functionals -- the heat kernel and the resolvent evaluated at the centre
residue, and the mass the low eigenvectors place on it -- describe where the
centre sits inside its own neighbourhood. The second kind is what the old bank
threw away with the eigenvectors, and it is the kind that can distinguish a
residue at the mouth of a cleft from one on the wall beside it.

Deformation across scale
------------------------
The last family is not evaluated at a scale but between scales. For a selected
set of quantities the value at 6 A is compared with the value at 16 A, as a
difference and as a ratio. A buried residue in a rigid core looks the same at
both; a residue lining a cleft that only opens at the larger radius does not.
This is the deformation of the local operator under dilation of the scale
parameter, and it is where a cryptic site should be visible if it is visible
anywhere in the geometry of the apo structure.

What this module does not claim
-------------------------------
These are candidate descriptors. Whether any of them help is measured elsewhere,
on the training partition, by ``tools/expand_invariant_bank.py``. Nothing here
has been selected on anything.
"""
from __future__ import annotations

import numpy as np

from pocket_bench.spatial import self_pairs_within

SCALES: tuple[float, ...] = (6.0, 8.0, 10.0, 13.0, 16.0)
MAX_LOCAL = 48            # cap on the induced subgraph order, as in the old bank
HEAT_TIMES: tuple[float, ...] = (0.1, 0.5, 2.0)
RESOLVENT_SHIFT = 1.0
_EPS = 1e-12

# Quantities compared between the smallest and the largest scale. Chosen to span
# the three operator kinds rather than to be individually promising.
_DEFORM_KEYS: tuple[str, ...] = (
    "rho", "fiedler", "lap_max", "nspec_entropy", "heat_diag_t0.5",
    "resolvent_diag", "anisotropy", "planarity", "deg_mean", "gyration_trace",
)


def _per_scale_names(r: float) -> list[str]:
    s = f"@{r:g}"
    names = [
        # geometry of the neighbourhood
        f"rho{s}", f"deg_mean{s}", f"deg_var{s}", f"dist_mean{s}",
        f"dist_var{s}", f"dist_skew{s}",
        # gyration tensor of the neighbourhood cloud
        f"gyration_trace{s}", f"anisotropy{s}", f"planarity{s}",
        f"sphericity{s}", f"asphericity{s}", f"centre_offset{s}",
        # combinatorial Laplacian, trace functionals
        f"fiedler{s}", f"lap_max{s}", f"spec_gap{s}", f"lap_m1{s}",
        f"lap_m2{s}", f"lap_m3{s}", f"n_components{s}",
        # normalised Laplacian, trace functionals
        f"nspec_m1{s}", f"nspec_m2{s}", f"nspec_entropy{s}",
        f"nspec_max{s}", f"nspec_fiedler{s}",
    ]
    names += [f"heat_trace_t{t:g}{s}" for t in HEAT_TIMES]
    # diagonal functionals: where the centre sits in its own neighbourhood
    names += [f"heat_diag_t{t:g}{s}" for t in HEAT_TIMES]
    names += [f"resolvent_diag{s}", f"fiedler_mass{s}", f"top_mass{s}",
              f"participation{s}"]
    return names


def feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for r in SCALES:
        names.extend(_per_scale_names(r))
    lo, hi = f"{SCALES[0]:g}", f"{SCALES[-1]:g}"
    for k in _DEFORM_KEYS:
        names.append(f"{k}_dilation_diff@{lo}v{hi}")
        names.append(f"{k}_dilation_ratio@{lo}v{hi}")
    return tuple(names)


FEATURE_NAMES: tuple[str, ...] = feature_names()
N_OPERATOR = len(FEATURE_NAMES)


def _neighbourhood(ctr: np.ndarray, radius: float):
    """CSR contact lists at one radius."""
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


def _gyration(P: np.ndarray, centre: np.ndarray) -> tuple[float, ...]:
    """Invariants of the neighbourhood's gyration tensor about its own mean.

    Eigenvalues are sorted ascending, so the ratios are orientation-free: any
    rotation of the receptor leaves all of them fixed, which is the property
    that makes them admissible here.
    """
    Q = P - P.mean(0)
    G = (Q.T @ Q) / max(len(P), 1)
    w = np.linalg.eigvalsh(G)
    w = np.clip(w, 0.0, None)
    tot = float(w.sum())
    if tot <= _EPS:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    l1, l2, l3 = float(w[0]), float(w[1]), float(w[2])
    anis = (l3 - l1) / tot
    plan = 2.0 * (l2 - l1) / tot
    spher = 3.0 * l1 / tot
    asph = (l3 - 0.5 * (l1 + l2)) / tot
    offset = float(np.linalg.norm(centre - P.mean(0)))
    return tot, anis, plan, spher, asph, offset


def _spectra(P: np.ndarray, radius: float):
    """One eigendecomposition, reused by every functional at this scale.

    The adjacency is Gaussian-weighted rather than 0/1: a contact at 5.9 A and
    one at 6.1 A differ by a hair in the structure and by everything in a hard
    cutoff, and the weighting removes a discontinuity that has no physical
    counterpart. The normalised Laplacian is computed from the same weights so
    that its spectrum stays in [0, 2] and is comparable between a residue with
    eight neighbours and one with forty.
    """
    k = len(P)
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    W = np.exp(-d2 / (radius * radius))
    np.fill_diagonal(W, 0.0)
    deg = W.sum(1)
    L = np.diag(deg) - W
    ev, V = np.linalg.eigh(L)
    ev = np.clip(ev, 0.0, None)

    dsafe = np.where(deg > _EPS, deg, 1.0)
    Dm = 1.0 / np.sqrt(dsafe)
    Ln = np.eye(k) - (W * Dm[:, None]) * Dm[None, :]
    Ln = 0.5 * (Ln + Ln.T)
    nev = np.clip(np.linalg.eigvalsh(Ln), 0.0, 2.0)
    return d2, W, deg, ev, V, nev


def _scale_block(ctr: np.ndarray, radius: float) -> np.ndarray:
    """Every descriptor at one scale, for every residue of one chain."""
    n = len(ctr)
    names = _per_scale_names(radius)
    out = np.zeros((n, len(names)), dtype=np.float64)
    indptr, indices = _neighbourhood(ctr, radius)
    vol = (4.0 / 3.0) * np.pi * radius ** 3

    for i in range(n):
        nb = indices[indptr[i]:indptr[i + 1]]
        if nb.size > MAX_LOCAL:
            d2c = ((ctr[nb] - ctr[i]) ** 2).sum(1)
            nb = nb[np.argsort(d2c, kind="stable")[:MAX_LOCAL]]
        S = np.concatenate([[i], nb])
        P = ctr[S]
        k = len(S)
        col = 0

        dc = np.sqrt(((ctr[nb] - ctr[i]) ** 2).sum(1)) if nb.size else \
            np.zeros(0)
        out[i, col] = nb.size / vol; col += 1                      # rho
        if k < 3:
            # Too small for a spectrum or a tensor. Everything else stays zero,
            # which is the value that adds no information, and the residue count
            # is already recorded above.
            continue

        tot_g, anis, plan, spher, asph, offset = _gyration(P, ctr[i])
        d2, W, deg, ev, V, nev = _spectra(P, radius)

        out[i, col] = float(deg.mean()); col += 1                  # deg_mean
        out[i, col] = float(deg.var()); col += 1                   # deg_var
        out[i, col] = float(dc.mean()) if dc.size else 0.0; col += 1
        out[i, col] = float(dc.var()) if dc.size else 0.0; col += 1
        sd = float(dc.std())
        out[i, col] = (float(((dc - dc.mean()) ** 3).mean()) / (sd ** 3)
                       if dc.size and sd > _EPS else 0.0); col += 1

        out[i, col] = tot_g; col += 1
        out[i, col] = anis; col += 1
        out[i, col] = plan; col += 1
        out[i, col] = spher; col += 1
        out[i, col] = asph; col += 1
        out[i, col] = offset; col += 1

        out[i, col] = float(ev[1]); col += 1                       # fiedler
        lmax = float(ev[-1])
        out[i, col] = lmax; col += 1                               # lap_max
        out[i, col] = float(ev[1]) / lmax if lmax > _EPS else 0.0; col += 1
        out[i, col] = float(ev.mean()); col += 1                   # lap_m1
        out[i, col] = float((ev ** 2).mean()); col += 1
        out[i, col] = float((ev ** 3).mean()); col += 1
        out[i, col] = float((ev < 1e-8).sum()); col += 1           # components

        out[i, col] = float(nev.mean()); col += 1
        out[i, col] = float((nev ** 2).mean()); col += 1
        p = nev / max(float(nev.sum()), _EPS)
        p = p[p > _EPS]
        out[i, col] = float(-(p * np.log(p)).sum()); col += 1
        out[i, col] = float(nev[-1]); col += 1
        out[i, col] = float(nev[1]); col += 1

        for t in HEAT_TIMES:
            out[i, col] = float(np.exp(-t * ev).mean()); col += 1

        # Diagonal functionals at the centre, which is row 0 of S by
        # construction. These are the quantities the old bank discarded.
        v0 = V[0, :] ** 2
        for t in HEAT_TIMES:
            out[i, col] = float((v0 * np.exp(-t * ev)).sum()); col += 1
        out[i, col] = float((v0 / (ev + RESOLVENT_SHIFT)).sum()); col += 1
        out[i, col] = float(v0[1]); col += 1                       # fiedler_mass
        out[i, col] = float(v0[-1]); col += 1                      # top_mass
        out[i, col] = float(1.0 / max((v0 ** 2).sum(), _EPS)); col += 1
    return out


def operator_residue_features(ctr: np.ndarray) -> np.ndarray:
    """``(n_residues, N_OPERATOR)`` for one chain, from centroids alone."""
    ctr = np.ascontiguousarray(np.asarray(ctr, dtype=np.float64))
    n = len(ctr)
    blocks = {r: _scale_block(ctr, r) for r in SCALES}
    F = np.concatenate([blocks[r] for r in SCALES], axis=1)

    lo, hi = SCALES[0], SCALES[-1]
    lo_names = _per_scale_names(lo)
    hi_names = _per_scale_names(hi)
    extra = np.zeros((n, 2 * len(_DEFORM_KEYS)), dtype=np.float64)
    for t, key in enumerate(_DEFORM_KEYS):
        a = blocks[lo][:, lo_names.index(f"{key}@{lo:g}")]
        b = blocks[hi][:, hi_names.index(f"{key}@{hi:g}")]
        extra[:, 2 * t] = a - b
        extra[:, 2 * t + 1] = a / np.where(np.abs(b) > _EPS, b, np.nan)
    # A ratio against a vanishing denominator is undefined, not large; it takes
    # the neutral value rather than an arbitrary one.
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([F, extra], axis=1)
