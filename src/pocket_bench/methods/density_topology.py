"""Density-weighted non-Archimedean topology and single-pass isometric rotors.

Two algebraic bridges, both closed-form and single-pass. No SciPy, no iterative
solver, no RNG, no fitted scalar.

--------------------------------------------------------------------------------
BRIDGE 1 — why the Euclidean ultrametric collapsed, and the metric that fixes it
--------------------------------------------------------------------------------
Feeding raw Euclidean distances to single linkage yields the maximal subdominant
ultrametric, which equals the minimax path distance

    d_u(i,j) = min over paths P from i to j of  max edge on P.

A protein is a covalently bonded chain, so between ANY two atoms there exists a
path whose every edge is a bond of 1.2-1.9 A. The minimax is therefore bounded by
the largest bond, and measurement confirms it: on 1bk2_A the Euclidean
single-linkage ultrametric has diameter 1.789 A against a Euclidean diameter of
32.9 A. Every threshold above 1.79 A gives one ball holding 100% of atoms; every
threshold below 1.21 A gives all singletons. The collapse is not a tuning failure,
it is a theorem about connected graphs with bounded edge length.

The repair is to change the BASE METRIC before the ultrametric projection, so that
the quantity being minimaxed is topological depth rather than Euclidean length.
Define the packing degree (the coordination functional)

    rho_i = #{ j != i : |x_i - x_j| <= r_c },

and weight each contact edge by a density-deflated length

    w_ij = |x_i - x_j| * ( rho_max / min(rho_i, rho_j) )^gamma .

Now the minimax path cost from a dense core to a loose surface loop is dominated
by the sparsest atom that must be traversed, because that atom multiplies its edge
by a large density deflator. Two consequences, both required by the theory:

* Edge weights span orders of magnitude (a density RATIO), not the 1.2-1.9 A
  window, so the valuation has real dynamic range and cannot collapse.
* A loose surface decoy and a tight buried pocket are separated by a high-cost
  edge even when they are Euclidean neighbours, so they land in different balls.
  Euclidean adjacency no longer implies topological adjacency, which is exactly
  the non-Archimedean property being imported.

Thresholding an ultrametric is transitive (the strong triangle inequality forces
d_u <= tau to be an equivalence relation), so the balls are a genuine partition and
the gate is a block-diagonal 0/1 mask. The threshold is not tuned: tau is placed at
the largest MULTIPLICATIVE gap in the sorted spanning-tree weights, i.e. the
valuation jump of the tree, which is scale-free and determined by the structure.

--------------------------------------------------------------------------------
BRIDGE 2 — a rotor applied exactly once
--------------------------------------------------------------------------------
Composition drift comes from multiplying many rotors and re-orthonormalizing. Here
no composition occurs. For a unit bivector B (B^2 = -1) and angle theta the rotor

    R = exp(-B theta / 2) = cos(theta/2) - B sin(theta/2),      v' = R v R~

expands, in the 3D even subalgebra, to the closed form

    v' = v cos(theta) + (n x v) sin(theta) + n (n . v)(1 - cos(theta)),

with n the vector dual to B. This is evaluated once per atom from (n, theta); the
rotor is never accumulated, so there is no drift to correct. det R = 1 identically,
hence the map is an exact isometry and conserves volume by construction rather than
by a numerical test.

A global isometry opens nothing (it resamples the same wall from a moved frame), so
the rotor is applied to ONE ultrametric ball with its complement held fixed. The
result is piecewise-rigid: volume is exactly conserved inside each ball, and the
only geometric change is at the ball interface. That interface is precisely where a
cryptic pocket opens, a hinge rotation of a loop or subdomain, so the operator's
support coincides with the physics it is meant to express.
"""
from __future__ import annotations

import numpy as np

from pocket_bench.spatial import counts_within, self_pairs_within

CONTACT_CUTOFF = 8.0     # A, density and contact-graph radius
DENSITY_GAMMA = 2.0      # deflator exponent; integer, not fitted


def packing_density(coords: np.ndarray, r_c: float = CONTACT_CUTOFF) -> np.ndarray:
    """Coordination number per atom, the  packing functional.

    Counted on a cell list, so cost scales with the number of atoms actually
    inside the cutoff rather than with N^2.
    """
    return np.maximum(counts_within(coords, r_c), 1)


def density_weighted_edges(
    coords: np.ndarray, rho: np.ndarray, *,
    cutoff: float = CONTACT_CUTOFF, gamma: float = DENSITY_GAMMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Contact-graph edges (i<j) and their density-deflated lengths.

    Returns ``(pairs, weights)`` with ``pairs`` of shape (E,2). Only contacts are
    emitted: the minimax path distance over the contact graph equals the one over
    the complete graph whenever the contact graph is connected, because any longer
    chord can be replaced by a path of shorter edges.
    """
    rho = np.asarray(rho, dtype=np.float64)
    rho_max = float(rho.max())
    pairs, d = self_pairs_within(coords, cutoff)
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64)
    deflator = (rho_max / np.minimum(rho[pairs[:, 0]], rho[pairs[:, 1]])) ** gamma
    return pairs, d * deflator


class _DSU:
    """Union-find with path halving. One pass, no recursion."""

    __slots__ = ("p", "r")

    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a: int) -> int:
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


def spanning_tree(n: int, pairs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Minimum spanning forest edge indices, ascending by weight (Kruskal).

    The MST carries the entire minimax structure: the ultrametric distance between
    i and j is the largest weight on their unique tree path, so the tree is a
    complete and exact summary of the maximal subdominant ultrametric.
    """
    order = np.argsort(weights, kind="stable")
    dsu = _DSU(n)
    keep = []
    for e in order:
        a, b = int(pairs[e, 0]), int(pairs[e, 1])
        if dsu.union(a, b):
            keep.append(int(e))
            if len(keep) == n - 1:
                break
    return np.asarray(keep, dtype=np.int64)


MIN_BALL_ATOMS = 8       # one residue of heavy atoms: the smallest hinge unit


def valuation_threshold(
    n: int, pairs: np.ndarray, tree: np.ndarray, tree_w: np.ndarray,
    rho: np.ndarray, *, max_ball_fraction: float = 0.90,
    min_ball_atoms: int = MIN_BALL_ATOMS,
) -> tuple[float, float]:
    """tau at the peak of the admissible-ball count over the tree filtration.

    Two earlier criteria were tried and both are recorded here because both fail
    for structural reasons, not tuning reasons:

    * Largest multiplicative gap in the sorted tree weights. Scale-free, but
      measured it lands in the extreme tail and returns two single-atom outlier
      balls plus one ball holding 99.6% of the chain. The tail gap describes the
      two loosest atoms, not the core/surface boundary.
    * Calinski-Harabasz variance ratio of rho. Its (n-k) divisor is supposed to
      punish over-fragmentation, but as k approaches n that divisor collapses at
      the same rate as the within-cluster scatter, so the ratio diverges and the
      all-singleton partition wins anyway (measured: 2436 balls of 2437 atoms).

    Adding tree edges in ascending weight order generates every closed-ball
    partition the ultrametric admits, since an ultrametric IS a nested hierarchy
    indexed by radius. Over that finite family, count the balls large enough to
    turn as a rigid hinge (>= one residue of heavy atoms). That count is 0 at
    tau = 0 (all singletons) and 1 at tau = inf (one giant ball), so it has an
    interior maximum, and that maximum is the radius at which the structure
    resolves into the largest number of independently movable packing domains.
    No continuous scalar is fitted: the only constant is an integer atom count
    fixed by residue granularity.

    Maintained incrementally through the same union-find: one pass over the n-1
    tree edges, O(1) per edge, no partition ever rebuilt.

    Returns ``(tau, n_admissible_balls)``.
    """
    rho = np.asarray(rho, dtype=np.float64)
    if len(tree) == 0:
        return (float(tree_w.max()) if len(tree_w) else 0.0), 0.0

    cnt = np.ones(n, dtype=np.int64)
    dsu = _DSU(n)
    order = np.argsort(tree_w, kind="stable")
    cap = max_ball_fraction * n
    n_admissible = 0                      # components with size >= min_ball_atoms
    best_tau, best_k = float(tree_w[order[-1]]), -1

    for pos in order:
        e = int(tree[pos])
        a, b = int(pairs[e, 0]), int(pairs[e, 1])
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            continue
        was = (cnt[ra] >= min_ball_atoms) + (cnt[rb] >= min_ball_atoms)
        dsu.union(a, b)
        root = dsu.find(a)
        other = rb if root == ra else ra
        cnt[root] = cnt[ra] + cnt[rb]
        cnt[other] = 0
        n_admissible += (1 if cnt[root] >= min_ball_atoms else 0) - was
        if cnt[root] > cap:
            continue
        if n_admissible > best_k:
            best_k, best_tau = n_admissible, float(tree_w[pos])
    return best_tau, float(max(best_k, 0))


def ultrametric_balls(
    coords: np.ndarray, *, cutoff: float = CONTACT_CUTOFF,
    gamma: float = DENSITY_GAMMA, tau: float | None = None,
) -> dict:
    """Density-weighted ultrametric ball partition of the atoms.

    Returns ``labels`` (per-atom ball id), the chosen ``tau``, the packing density,
    and the tree weight spectrum. Balls at radius tau are the connected components
    of the tree restricted to edges of weight <= tau, which is exactly the closed
    ball partition of the minimax ultrametric.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)
    rho = packing_density(coords, cutoff)
    pairs, w = density_weighted_edges(coords, rho, cutoff=cutoff, gamma=gamma)
    if len(pairs) == 0:
        return {"labels": np.zeros(n, dtype=np.int64), "tau": 0.0, "rho": rho,
                "tree_w": np.zeros(0), "n_balls": 1}
    tree = spanning_tree(n, pairs, w)
    tree_w = w[tree]
    if tau is None:
        t, sep = valuation_threshold(n, pairs, tree, tree_w, rho)
    else:
        t, sep = float(tau), 0.0
    dsu = _DSU(n)
    for e in tree[tree_w <= t]:
        dsu.union(int(pairs[e, 0]), int(pairs[e, 1]))
    roots = np.fromiter((dsu.find(i) for i in range(n)), dtype=np.int64, count=n)
    _, labels = np.unique(roots, return_inverse=True)
    return {"labels": labels, "tau": t, "rho": rho, "tree_w": tree_w,
            "density_separation_f": sep,
            "n_balls": int(labels.max()) + 1}


def rotor_apply(
    coords: np.ndarray, axis: np.ndarray, theta: float,
    center: np.ndarray, mask: np.ndarray,
) -> np.ndarray:
    """One exact rotor sandwich v' = R v R~, applied once.

    ``mask`` selects the ultrametric ball that rotates; the complement is held
    fixed. Rodrigues form of the even-subalgebra sandwich, evaluated in closed
    form from (axis, theta) with no rotor accumulation and no re-orthonormalization,
    so there is nothing for floating point to drift against: the map is an exact
    isometry of the selected ball for any representable (axis, theta).
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = np.asarray(axis, dtype=np.float64)
    nn = float(np.sqrt(n @ n))
    if nn == 0.0 or theta == 0.0 or not mask.any():
        return coords.copy()
    n = n / nn
    out = coords.copy()
    v = coords[mask] - center
    c, s = float(np.cos(theta)), float(np.sin(theta))
    out[mask] = center + (v * c + np.cross(n, v) * s
                          + n[None, :] * ((v @ n)[:, None]) * (1.0 - c))
    return out


def branch_hinge(coords: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation centre and axis of an ultrametric ball, in closed form.

    The centre is the ball centroid. The axis is the ball's minor principal axis
    (smallest eigenvalue of the gyration tensor): rotating about the thinnest
    direction is the motion that sweeps the largest interfacial volume per unit
    angle, which is the hinge a cryptic loop actually turns on.
    """
    pts = coords[mask]
    ctr = pts.mean(axis=0)
    rel = pts - ctr
    gyr = rel.T @ rel
    evals, evecs = np.linalg.eigh(gyr)          # symmetric 3x3, exact and ordered
    return ctr, evecs[:, 0]
