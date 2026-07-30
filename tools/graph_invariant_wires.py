#!/usr/bin/env python3.12
"""Fifteen integer graph invariants of the contact lining, as a fourth column family.

Why this family and not another
-------------------------------
COLLECTABILITY_SCREEN.json measured where the deployed bank's pairwise interaction
comes from, and the answer is sharp. Over the 207,690 pairs of the 645-wire bus the
mean interaction is +1.06e-05, but pairs that read *one* local quantity under two
statistics average -6.29e-05, the only negative category and six times more negative
than the asymmetry family as a whole. Reading one operator at several radii produces
wires whose joint says less than their marginals added, which is the shape a linear
solve collects and a counting field cannot.

That yields a design rule with teeth: a new family is worth building only if its
members are *different quantities*. So this one is fifteen invariants of the same
graph rather than one invariant of fifteen graphs, and no two of them are a monotone
function of each other.

What the existing forty-three already cover, and what is left
------------------------------------------------------------
G2 reads the lining's Laplacian spectrum, its mean degree, the number of components
once the centre is deleted and the cycle rank E - V + C. G6 reads the degree at 10 A.
So spectra, connectivity and one cycle count are taken.

What is absent is the combinatorics of the cycles rather than their number: how short
they are, how they are packed into cliques, whether the wall is one path or a mesh,
where its cut points are. Those are integer counts, each a different invariant, and
none of them is recoverable from a Laplacian spectrum.

The graph
---------
The same lining every G2 descriptor uses: S_i is the residue and its contacts within
10 A, capped at the 48 nearest, exactly as _local_spectral caps it. Edges are drawn
at 7 A, the pinch radius, and for the same recorded reason -- at 10 A the lining of a
globular protein is a near-complete graph, so the girth is 3 and the diameter is 2 for
every residue and every count below is a constant. The wall is S_i with the centre
deleted, which is the graph G2's betti0 and betti1 are computed on.

Every invariant is computed at that one radius. Reading the same invariant at two
radii is the category the screen shows to be harmful, so the temptation to sweep is
declined here on measured grounds rather than aesthetic ones.

The fifteen
-----------
Through the centre:
  tri_centre     triangles containing i
  c4_centre      four-cycles containing i
  kcore          the largest k for which i survives iterated degree-k pruning of S_i
  ecc_centre     eccentricity of i, its distance to the furthest reachable vertex
  maxclique      the order of the largest clique containing i
  p4_centre      induced paths on four vertices with i as an endpoint
  expansion      how many vertices are two steps from i but not one
Of the wall alone, which is what the residue faces:
  tri_wall       triangles not involving i
  girth_wall     the length of its shortest cycle, 0 when it is a forest
  diam_wall      the diameter of its largest component
  wiener_wall    the sum of all pairwise distances inside that component
  bridges_wall   edges whose removal disconnects it
  artic_wall     vertices whose removal disconnects it
  leaves_wall    vertices of degree one
  bipartite_wall 1 when it has no odd cycle, 0 otherwise

Nothing here is a spectral quantity, a distance in angstroms, or a ratio. Every
value is a count, so the family is integer before the quaternary banding is applied
and the banding is the only lossy step.

What is predicted, before any field is compiled
-----------------------------------------------
The screen is the thing being tested, so its prediction is recorded first. If the
mean pairwise interaction of this family is positive and of the order of the deployed
bank's +1.06e-05, the counting field should collect it and the union attachment
should show a lift. If it is negative like the asymmetry family's -9.69e-06, it
should not, and the lift should go to a linear solve instead.

Either outcome is informative and the second is not a wasted run: the screen has so
far been fitted to three families after the fact, and a fourth family whose
prediction is written down before its lift is measured is the first real test of it.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from pocket_bench.paths import ROOT

CACHE = ROOT / "data/cryptobench_apo/_graphinv_cache_train.npz"
NAMES = (
    "tri_centre", "c4_centre", "kcore", "ecc_centre", "maxclique", "p4_centre",
    "expansion", "tri_wall", "girth_wall", "diam_wall", "wiener_wall",
    "bridges_wall", "artic_wall", "leaves_wall", "bipartite_wall",
)
# The lining, exactly as algebraic_descriptors._local_spectral builds it.
NBR_R = 10.0
MAX_LOCAL = 48
EDGE_R = 7.0
# A girth of 0 means acyclic. Cycles longer than this are reported as this value,
# which keeps the wire integer-valued and bounded; at 7 A on 48 vertices a longer
# shortest cycle is not observed in practice and the cap is recorded rather than
# assumed to be inactive.
GIRTH_CAP = 12


def _bfs_layers(adj: list[list[int]], s: int, n: int) -> np.ndarray:
    d = np.full(n, -1, dtype=np.int64)
    d[s] = 0
    q = deque([s])
    while q:
        a = q.popleft()
        for b in adj[a]:
            if d[b] < 0:
                d[b] = d[a] + 1
                q.append(b)
    return d


def _girth(adj: list[list[int]], n: int) -> int:
    """Shortest cycle length by a BFS from every vertex, 0 when acyclic.

    The standard bound: for each root, a non-tree edge closing at depths da, db
    gives a cycle of length da + db + 1. Taking the minimum over roots is exact
    for unweighted graphs up to the usual off-by-one on even cycles, and the
    off-by-one cannot occur here because the minimum over *all* roots is taken.
    """
    best = GIRTH_CAP + 1
    for s in range(n):
        d = np.full(n, -1, dtype=np.int64)
        par = np.full(n, -1, dtype=np.int64)
        d[s] = 0
        q = deque([s])
        while q:
            a = q.popleft()
            if 2 * int(d[a]) >= best:
                break
            for b in adj[a]:
                if d[b] < 0:
                    d[b] = d[a] + 1
                    par[b] = a
                    q.append(b)
                elif b != par[a]:
                    c = int(d[a]) + int(d[b]) + 1
                    if c < best:
                        best = c
    return 0 if best > GIRTH_CAP else int(best)


def _bridges_and_articulations(adj: list[list[int]], n: int) -> tuple[int, int]:
    """Counts from one iterative depth-first search, Tarjan's low-link rule."""
    disc = np.full(n, -1, dtype=np.int64)
    low = np.zeros(n, dtype=np.int64)
    parent = np.full(n, -1, dtype=np.int64)
    is_art = np.zeros(n, dtype=bool)
    bridges = 0
    timer = 0
    for root in range(n):
        if disc[root] >= 0:
            continue
        stack = [(root, iter(adj[root]))]
        disc[root] = low[root] = timer
        timer += 1
        root_children = 0
        while stack:
            a, it = stack[-1]
            advanced = False
            for b in it:
                if disc[b] < 0:
                    parent[b] = a
                    disc[b] = low[b] = timer
                    timer += 1
                    if a == root:
                        root_children += 1
                    stack.append((b, iter(adj[b])))
                    advanced = True
                    break
                if b != parent[a]:
                    low[a] = min(low[a], disc[b])
            if advanced:
                continue
            stack.pop()
            if stack:
                p = stack[-1][0]
                low[p] = min(low[p], low[a])
                if low[a] > disc[p]:
                    bridges += 1
                if p != root and low[a] >= disc[p]:
                    is_art[p] = True
        if root_children > 1:
            is_art[root] = True
    return bridges, int(is_art.sum())


def _max_clique_through(A: np.ndarray, v: int) -> int:
    """Largest clique containing ``v``, exactly, by branch and bound.

    Exact rather than greedy because a greedy bound is not an invariant: two
    graphs with the same clique number could take different values, and a wire
    that depends on a traversal order is not a function of the structure. The
    search is over ``v``'s neighbourhood only, which at 7 A is small.
    """
    nbrs = np.nonzero(A[v])[0]
    if nbrs.size == 0:
        return 1
    idx = {int(u): k for k, u in enumerate(nbrs)}
    m = len(nbrs)
    sub = A[np.ix_(nbrs, nbrs)].astype(bool)
    best = 0

    def expand(cand: list[int], size: int) -> None:
        nonlocal best
        if size + len(cand) <= best:
            return
        if not cand:
            best = max(best, size)
            return
        for k, u in enumerate(cand):
            if size + len(cand) - k <= best:
                return
            expand([w for w in cand[k + 1:] if sub[u, w]], size + 1)

    expand(list(range(m)), 0)
    return best + 1


def invariants_for_chain(ctr: np.ndarray) -> np.ndarray:
    """The fifteen counts for every residue of one chain."""
    n = len(ctr)
    out = np.zeros((n, len(NAMES)), dtype=np.float64)
    if n == 0:
        return out
    d2_all = ((ctr[:, None, :] - ctr[None, :, :]) ** 2).sum(-1)
    near = d2_all <= NBR_R * NBR_R
    np.fill_diagonal(near, False)

    for i in range(n):
        nb = np.nonzero(near[i])[0]
        if nb.size == 0:
            continue
        if nb.size > MAX_LOCAL:
            nb = nb[np.argsort(d2_all[i, nb], kind="stable")[:MAX_LOCAL]]
        S = np.concatenate([[i], nb])
        k = len(S)
        A = (d2_all[np.ix_(S, S)] <= EDGE_R * EDGE_R)
        np.fill_diagonal(A, False)
        Af = A.astype(np.int64)
        adj = [np.nonzero(A[a])[0].tolist() for a in range(k)]

        A2 = Af @ Af
        out[i, 0] = 0.5 * float((Af[0] * A2[0]).sum())          # tri_centre
        # Four-cycles through vertex 0: pairs of distinct neighbours-at-two-steps
        # counted through two distinct intermediates, minus the degenerate walks.
        c4 = 0
        deg0 = adj[0]
        for a_pos in range(k):
            if a_pos == 0 or A[0, a_pos]:
                continue
            common = int(A2[0, a_pos])
            if common >= 2:
                c4 += common * (common - 1) // 2
        out[i, 1] = float(c4)                                    # c4_centre

        # k-core of the centre: peel vertices of degree below k until the centre
        # would go, which is the standard degeneracy computation restricted to
        # the value at one vertex.
        deg = Af.sum(1).astype(np.int64)
        alive = np.ones(k, dtype=bool)
        core = 0
        kk = 1
        while alive[0]:
            changed = True
            while changed:
                changed = False
                for a in range(k):
                    if alive[a] and deg[a] < kk:
                        alive[a] = False
                        changed = True
                        for b in adj[a]:
                            if alive[b]:
                                deg[b] -= 1
            if alive[0]:
                core = kk
                kk += 1
        out[i, 2] = float(core)                                  # kcore

        dist0 = _bfs_layers(adj, 0, k)
        reach = dist0 >= 0
        out[i, 3] = float(dist0[reach].max())                    # ecc_centre
        out[i, 4] = float(_max_clique_through(Af, 0))            # maxclique

        # Induced P4 with the centre as an endpoint: 0-a-b-c with the three
        # non-consecutive pairs absent.
        p4 = 0
        for a in deg0:
            for b in adj[a]:
                if b == 0 or A[0, b]:
                    continue
                for c in adj[b]:
                    if c in (0, a) or A[0, c] or A[a, c]:
                        continue
                    p4 += 1
        out[i, 5] = float(p4)                                    # p4_centre
        two = int(((dist0 == 2)).sum())
        out[i, 6] = float(two)                                   # expansion

        # The wall: S with the centre deleted.
        B = A[1:, 1:]
        m = k - 1
        if m <= 0:
            continue
        Bf = B.astype(np.int64)
        wadj = [np.nonzero(B[a])[0].tolist() for a in range(m)]
        B2 = Bf @ Bf
        out[i, 7] = float(np.trace(Bf @ B2) / 6.0)               # tri_wall
        out[i, 8] = float(_girth(wadj, m))                       # girth_wall

        # Largest component of the wall, then its diameter and Wiener index.
        seen = np.zeros(m, dtype=bool)
        comps: list[list[int]] = []
        for s in range(m):
            if seen[s]:
                continue
            comp = []
            q = deque([s])
            seen[s] = True
            while q:
                a = q.popleft()
                comp.append(a)
                for b in wadj[a]:
                    if not seen[b]:
                        seen[b] = True
                        q.append(b)
            comps.append(comp)
        big = max(comps, key=len)
        diam = 0
        wiener = 0
        for s in big:
            d = _bfs_layers(wadj, s, m)
            dd = d[d >= 0]
            diam = max(diam, int(dd.max()))
            wiener += int(dd.sum())
        out[i, 9] = float(diam)                                  # diam_wall
        out[i, 10] = float(wiener // 2)                          # wiener_wall

        br, ar = _bridges_and_articulations(wadj, m)
        out[i, 11] = float(br)                                   # bridges_wall
        out[i, 12] = float(ar)                                   # artic_wall
        out[i, 13] = float((Bf.sum(1) == 1).sum())               # leaves_wall

        # Bipartite by two-colouring every component.
        colour = np.full(m, -1, dtype=np.int64)
        bip = 1
        for s in range(m):
            if colour[s] >= 0:
                continue
            colour[s] = 0
            q = deque([s])
            while q and bip:
                a = q.popleft()
                for b in wadj[a]:
                    if colour[b] < 0:
                        colour[b] = 1 - colour[a]
                        q.append(b)
                    elif colour[b] == colour[a]:
                        bip = 0
                        break
            if not bip:
                break
        out[i, 14] = float(bip)                                  # bipartite_wall
    return out


def build_or_load(force: bool = False, limit: int | None = None
                  ) -> tuple[np.ndarray, tuple[str, ...]]:
    if CACHE.is_file() and not force:
        z = np.load(CACHE, allow_pickle=False)
        print(f"reusing {CACHE.relative_to(ROOT)}  {z['X'].shape}")
        return z["X"], NAMES
    wide = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
    z = np.load(wide, allow_pickle=False)
    ctr, n_res = z["ctr"], z["n_res_per"]
    blocks = []
    off = 0
    t0 = time.perf_counter()
    todo = list(n_res) if limit is None else list(n_res)[:limit]
    for c, nn in enumerate(todo, start=1):
        nn = int(nn)
        blocks.append(invariants_for_chain(np.asarray(ctr[off:off + nn], float)))
        off += nn
        if c % 50 == 0 or c == len(todo):
            el = time.perf_counter() - t0
            print(f"  {c}/{len(todo)} chains, {el:.0f}s, "
                  f"{el / c * len(todo) / 60:.1f} min projected", flush=True)
    X = np.concatenate(blocks, axis=0)
    if limit is None:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, X=X)
        print(f"wrote {CACHE.relative_to(ROOT)}  {X.shape}")
    return X, NAMES


WIDE_CACHE = ROOT / "data/cryptobench_apo/_graphinv_wide_cache_train.npz"


def build_wide_or_load(force: bool = False) -> tuple[np.ndarray, tuple[str, ...]]:
    """The fifteen invariants under the fifteen deployed statistics: 225 wires.

    Why the expansion is applied at all, given that the screen shows same-quantity
    pairs to be the harmful category. Two reasons, and both are about size rather
    than taste. Fifteen columns make 105 pairs and about 112 tables under the union
    attachment, against 1,024 for the asymmetry family and 5,152 for the deployed
    bank -- too small a block to move a mean whatever it contains, so a null on it
    would say nothing about the family. And the expansion is exactly what the
    deployed construction does to get from 43 quantities to 645 wires, so applying
    it keeps the comparison against that bank honest.

    The cost is accepted with its eyes open: within a 15-by-15 block a random pair
    shares a quantity with probability about one in fifteen, against one in
    forty-five in the deployed bus, so this block carries proportionally more of
    the category the screen says is harmful. That is a reason to screen the
    expanded block rather than the raw one, which is what happens here.
    """
    if WIDE_CACHE.is_file() and not force:
        z = np.load(WIDE_CACHE, allow_pickle=False)
        print(f"reusing {WIDE_CACHE.relative_to(ROOT)}  {z['X'].shape}")
        return z["X"], tuple(str(s) for s in z["names"])
    from pocket_bench.methods.wide_descriptors import wide_transform

    X, names = build_or_load()
    z = np.load(ROOT / "data/cryptobench_apo/_wide_cache_train.npz",
                allow_pickle=False)
    t0 = time.perf_counter()
    W = wide_transform(X, np.asarray(z["ctr"], dtype=np.float64), z["n_res_per"])
    print(f"expanded {X.shape[1]} invariants to {W.shape[1]} wires in "
          f"{time.perf_counter() - t0:.0f}s")
    # Statistic-major, matching build_wide, so wire w is invariant w % 15 read
    # under statistic w // 15 and the screen's layout assumption holds.
    stats = ("identity", *[f"mean_r{r:g}" for r in (6, 10, 14, 20, 26)],
             *[f"sd_r{r:g}" for r in (6, 14, 20)],
             *[f"diff_r{r:g}" for r in (6, 14, 20)],
             *[f"rank_r{r:g}" for r in (6, 14, 20)])
    wnames = tuple(f"{s}:{nm}" for s in stats for nm in names)
    if len(wnames) != W.shape[1]:
        raise SystemExit(f"named {len(wnames)} wires for {W.shape[1]} columns; "
                         f"the statistic order does not match wide_transform")
    np.savez_compressed(WIDE_CACHE, X=W, names=np.array(wnames))
    print(f"wrote {WIDE_CACHE.relative_to(ROOT)}  {W.shape}")
    return W, wnames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="build only this many chains, for timing")
    ap.add_argument("--wide", action="store_true",
                    help="also expand through the fifteen deployed statistics")
    a = ap.parse_args(argv)
    X, names = build_or_load(a.force, a.limit)
    if a.wide:
        W, wnames = build_wide_or_load(a.force)
        print(f"\n  {W.shape[1]} wires over {W.shape[0]} residues")
        return 0
    print(f"\n  {X.shape[1]} invariants over {X.shape[0]} residues")
    print(f"  {'invariant':16s} {'min':>8s} {'median':>8s} {'max':>8s} "
          f"{'distinct':>9s}")
    for j, nm in enumerate(names):
        col = X[:, j]
        print(f"  {nm:16s} {col.min():8.0f} {np.median(col):8.0f} "
              f"{col.max():8.0f} {len(np.unique(col)):9d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
