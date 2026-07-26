"""Uniform cell-list neighbour search.

Every fixed-radius neighbour query in this repository was written as a blocked
brute-force scan: for a block of query points, materialize the full
``(block, n_atoms, 3)`` difference tensor and threshold it. That is exact but it
costs O(n_query * n_atoms) work and, more damagingly, O(block * n_atoms * 3)
peak memory. On a 104k-atom receptor a single 256-point block allocates 639 MB
and one structure takes minutes; the same query over a 8k-atom chain wastes
roughly two orders of magnitude of work on pairs that are nowhere near the
cutoff.

A cutoff query does not need the full matrix. Bin the atoms into cubes of side
``r``; any atom within ``r`` of a query point must lie in that point's cube or
one of its 26 neighbours. Work becomes O(n * k) with k the mean occupancy, and
peak memory is one cube-pair block.

Exactness
---------
The predicate is unchanged: a pair is reported iff ``d2 <= r*r`` evaluated on the
same float64 coordinates, so this returns exactly the set the brute-force scan
returned, never an approximation. Results are emitted in lexicographic
``(query, atom)`` order, which is the order ``np.nonzero`` produced on the dense
blocks, so downstream floating-point accumulations (``np.add.at``) and
tie-breaking sorts (Kruskal on equal weights) see the identical sequence and the
numbers are bit-identical. ``tests/test_spatial.py`` asserts that against the
reference implementation.
"""
from __future__ import annotations

import numpy as np

_OFFSETS = np.array(
    [(dx, dy, dz)
     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    dtype=np.int64,
)
# The 13 lexicographically-forward neighbours. Together with the home cube these
# visit every unordered cube pair exactly once, halving the self-pair work.
_FORWARD = _OFFSETS[[i for i, o in enumerate(_OFFSETS) if tuple(o) > (0, 0, 0)]]


class CellGrid:
    """Atoms binned into cubes of side ``cell``, with sorted bucket lookup."""

    __slots__ = ("coords", "cell", "origin", "dims", "uniq", "starts", "ends",
                 "order", "cube")

    def __init__(self, coords: np.ndarray, cell: float) -> None:
        self.coords = np.ascontiguousarray(coords, dtype=np.float64)
        self.cell = float(cell)
        self.origin = self.coords.min(axis=0)
        cube = np.floor((self.coords - self.origin) / self.cell).astype(np.int64)
        self.dims = cube.max(axis=0) + 1
        self.cube = cube
        flat = self._flatten(cube)
        self.order = np.argsort(flat, kind="stable")
        srt = flat[self.order]
        self.uniq, self.starts = np.unique(srt, return_index=True)
        self.ends = np.append(self.starts[1:], len(srt))

    def _flatten(self, cube: np.ndarray) -> np.ndarray:
        return ((cube[:, 0] * self.dims[1] + cube[:, 1]) * self.dims[2]
                + cube[:, 2])

    def _bucket_of(self, flat_ids: np.ndarray) -> list[np.ndarray]:
        """Atom indices for each requested flat cube id (empty array if absent)."""
        k = np.searchsorted(self.uniq, flat_ids)
        out = []
        for cid, ki in zip(flat_ids, k):
            if ki < len(self.uniq) and self.uniq[ki] == cid:
                out.append(self.order[self.starts[ki]:self.ends[ki]])
            else:
                out.append(_EMPTY)
        return out

    def _candidates(self, cube: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """Atom indices in the cubes ``cube + offsets`` that exist and are in range."""
        nb = cube[None, :] + offsets
        ok = np.all((nb >= 0) & (nb < self.dims[None, :]), axis=1)
        if not ok.any():
            return _EMPTY
        buckets = self._bucket_of(self._flatten(nb[ok]))
        buckets = [b for b in buckets if b.size]
        if not buckets:
            return _EMPTY
        return np.concatenate(buckets)


_EMPTY = np.zeros(0, dtype=np.int64)


def _query_cubes(q: np.ndarray, grid: CellGrid) -> tuple[np.ndarray, np.ndarray]:
    """Group query points by the atom-grid cube they fall in.

    Returns ``(cube_coords, groups)`` where ``groups[i]`` holds the query indices
    sharing ``cube_coords[i]``. Query points outside the atom bounding box get a
    cube index outside ``dims``; the neighbour scan simply finds no cubes there,
    which is correct because no atom can be within the cutoff.
    """
    cube = np.floor((q - grid.origin) / grid.cell).astype(np.int64)
    # Sorted on the integer triple itself rather than on a packed hash, so cubes
    # outside the atom box (negative indices) cannot collide with interior ones.
    order = np.lexsort((cube[:, 2], cube[:, 1], cube[:, 0]))
    srt = cube[order]
    brk = np.nonzero(np.any(srt[1:] != srt[:-1], axis=1))[0] + 1
    starts = np.concatenate(([0], brk))
    ends = np.concatenate((brk, [len(srt)]))
    groups = [order[s:e] for s, e in zip(starts, ends)]
    return cube, groups


def cross_within(q: np.ndarray, coords: np.ndarray,
                 r: float) -> tuple[np.ndarray, np.ndarray]:
    """All ``(qi, ai)`` with ``|q[qi] - coords[ai]|^2 <= r^2``.

    Sorted lexicographically by ``(qi, ai)`` -- the order the dense scan yielded.
    """
    q = np.ascontiguousarray(q, dtype=np.float64)
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    if len(q) == 0 or len(coords) == 0:
        return _EMPTY, _EMPTY
    grid = CellGrid(coords, r)
    cube, groups = _query_cubes(q, grid)
    r2 = r * r
    qi_all, ai_all = [], []
    for g in groups:
        cand = grid._candidates(cube[g[0]], _OFFSETS)
        if cand.size == 0:
            continue
        d2 = ((q[g][:, None, :] - coords[cand][None, :, :]) ** 2).sum(-1)
        li, lj = np.nonzero(d2 <= r2)
        if li.size:
            qi_all.append(g[li])
            ai_all.append(cand[lj])
    if not qi_all:
        return _EMPTY, _EMPTY
    qi = np.concatenate(qi_all)
    ai = np.concatenate(ai_all)
    k = np.lexsort((ai, qi))
    return qi[k], ai[k]


def counts_within(coords: np.ndarray, r: float,
                  *, exclude_self: bool = True) -> np.ndarray:
    """Number of atoms within ``r`` of each atom, without materializing pairs."""
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    grid = CellGrid(coords, r)
    out = np.zeros(n, dtype=np.int64)
    r2 = r * r
    for ki in range(len(grid.uniq)):
        home = grid.order[grid.starts[ki]:grid.ends[ki]]
        cand = grid._candidates(grid.cube[home[0]], _OFFSETS)
        if cand.size == 0:
            continue
        d2 = ((coords[home][:, None, :] - coords[cand][None, :, :]) ** 2).sum(-1)
        out[home] = (d2 <= r2).sum(axis=1)
    if exclude_self:
        out -= 1
    return out


def self_pairs_within(coords: np.ndarray,
                      r: float) -> tuple[np.ndarray, np.ndarray]:
    """Upper-triangular pairs ``(i, j)``, ``i < j``, within ``r``, plus distances.

    Sorted lexicographically by ``(i, j)``, matching the dense row-block scan.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64)
    grid = CellGrid(coords, r)
    r2 = r * r
    ii, jj, dd = [], [], []

    def _emit(a: np.ndarray, b: np.ndarray, d2: np.ndarray,
              li: np.ndarray, lj: np.ndarray) -> None:
        # Cube membership says nothing about atom index order, so the pair is
        # normalized rather than filtered; filtering on a < b would silently drop
        # every cross-cube pair whose home atom happens to have the larger index.
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        ii.append(lo); jj.append(hi); dd.append(np.sqrt(d2[li, lj]))

    for ki in range(len(grid.uniq)):
        home = grid.order[grid.starts[ki]:grid.ends[ki]]
        # Home cube: the block is symmetric, so the strict upper triangle of the
        # LOCAL indices yields each intra-cube pair exactly once.
        d2 = ((coords[home][:, None, :] - coords[home][None, :, :]) ** 2).sum(-1)
        li, lj = np.nonzero((d2 <= r2) & (np.arange(len(home))[:, None]
                                          < np.arange(len(home))[None, :]))
        if li.size:
            _emit(home[li], home[lj], d2, li, lj)
        # Forward cubes: disjoint from the home cube, so every hit is a new pair.
        cand = grid._candidates(grid.cube[home[0]], _FORWARD)
        if cand.size == 0:
            continue
        d2 = ((coords[home][:, None, :] - coords[cand][None, :, :]) ** 2).sum(-1)
        li, lj = np.nonzero(d2 <= r2)
        if li.size == 0:
            continue
        _emit(home[li], cand[lj], d2, li, lj)
    if not ii:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64)
    i = np.concatenate(ii); j = np.concatenate(jj); d = np.concatenate(dd)
    k = np.lexsort((j, i))
    return np.stack([i[k], j[k]], axis=1), d[k]
