"""Quotient tables: what a counting table can address once a symmetry is imposed.

The bound this lifts
--------------------
A dense quaternary table over ``d`` digits has ``L**d`` cells, and on this fold
it is admissible only while ``L**d <= rN`` -- ``N = 234838`` training residues at
base rate ``r = 0.0576``, so ``rN = 13524`` positives. Beyond that most cells can
never be driven by a single positive and the combinational layer degenerates.
For ``L = 4`` that is ``d <= log_4(13524) = 6.86``: seven digits, no more, which
is why thirty-five invariants had to be split across six tables and could never
interact across them.

The bound counts CELLS. A table required to be invariant under a group ``G``
acting on its digit positions does not have ``L**d`` free cells; it has as many
as ``G`` has orbits, and the admissibility condition applies to those. For the
full symmetric group ``S_d`` -- the table may depend on WHICH LEVELS occur and
how often, but not on which position carries them -- an orbit is a multiset of
``d`` digits from ``L`` levels, so

    orbits(d, L) = C(d + L - 1, d)

which is polynomial in ``d`` where ``L**d`` is exponential. At ``L = 4`` the
admissible width goes from ``6.86`` to ``41``: every invariant in the bank fits
inside one symmetric table. The exchange rate runs the other way too, and that
is the direction that turned out to matter here -- holding ``d = 6``, a dense
table is stuck at ``L = 4`` (4096 cells) while a symmetric one reaches ``L = 12``
(12376), so the digits can carry more resolution for the same budget.

What is bought and what is paid
-------------------------------
A symmetric table cannot say WHICH invariant is extreme, only how many are at
each level. That is a real loss and it is not always worth the width: a single
``S_35`` table over the whole bank scores 0.5997 on the training pick half,
against 0.7446 for the frozen dense bank. What works is the Young subgroup --
``S_6`` inside each thematic group, the identity across groups -- which keeps the
group identity that carries the signal and quotients only the ordering inside a
group of invariants that measure the same kind of thing.

Addressing, and why it is still counting
----------------------------------------
The canonical representative of an ``S_d`` orbit is the digit word in ascending
order, so the address is the base-``L`` value of the sorted word. Sorting six
integers is a comparator network and the encoding is integer multiply-add: the
query path acquires no float, no fitted parameter and no iteration, which is the
property the whole detector exists to demonstrate.
"""
from __future__ import annotations

from math import comb

import numpy as np


def n_orbits(d: int, levels: int) -> int:
    """Number of ``S_d`` orbits on words of length ``d`` over ``levels`` levels."""
    if d < 0 or levels < 1:
        raise ValueError(f"d={d}, levels={levels} is not a table")
    return comb(d + levels - 1, d)


def n_cells_dense(d: int, levels: int) -> int:
    return levels ** d


def widest_admissible(levels: int, budget: int, *, symmetric: bool) -> int:
    """Largest ``d`` whose table stays inside ``budget`` cells.

    ``budget`` is the number of positives available to drive the cells; a table
    with more cells than that cannot have them all populated by a positive even
    once, whatever the data looks like.
    """
    d = 0
    while True:
        nxt = d + 1
        size = n_orbits(nxt, levels) if symmetric else n_cells_dense(nxt, levels)
        if size > budget:
            return d
        d = nxt


def orbit_address(digits: np.ndarray, cols: list[int] | tuple[int, ...],
                  levels: int) -> np.ndarray:
    """Canonical ``S_d`` orbit address for each row: the sorted word, base ``L``.

    Sorting is what performs the quotient. Two residues whose digits agree as a
    multiset but differ in which invariant carries which digit land on the same
    address by construction, which is the whole content of the symmetry.
    """
    cols = list(cols)
    if not cols:
        raise ValueError("a table over no columns is not a table")
    if levels < 2:
        raise ValueError(f"levels={levels} leaves nothing to quantise")
    w = np.sort(np.asarray(digits)[:, cols], axis=1)
    if w.size and (w.min() < 0 or w.max() >= levels):
        raise ValueError(f"digits outside [0, {levels}) reached the address unit")
    code = np.zeros(w.shape[0], dtype=np.int64)
    for t in range(w.shape[1]):
        code = code * levels + w[:, t]
    return code


def dense_address(digits: np.ndarray, cols: list[int] | tuple[int, ...],
                  levels: int) -> np.ndarray:
    """Positional address, for the dense tables the quotient is compared against."""
    cols = list(cols)
    code = np.zeros(np.asarray(digits).shape[0], dtype=np.int64)
    for c in cols:
        code = code * levels + np.asarray(digits)[:, c]
    return code


def compile_cells(addr_fit: np.ndarray, y_fit: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count positives and totals per occupied address. Counting, nothing else.

    Returns the occupied addresses in ascending order beside their counts, which
    is the form the artifact stores and the form ``read_cells`` binary-searches.
    """
    uniq, inv = np.unique(np.asarray(addr_fit), return_inverse=True)
    tot = np.bincount(inv, minlength=len(uniq)).astype(np.int64)
    pos = np.bincount(inv, weights=np.asarray(y_fit, dtype=np.float64),
                      minlength=len(uniq)).astype(np.int64)
    return uniq, pos, tot


def read_cells(addrs: np.ndarray, pos: np.ndarray, tot: np.ndarray,
               query: np.ndarray, fallback: float) -> np.ndarray:
    """Cell fraction per query row.

    An address the training fold never occupied reads the base rate, which is
    the only value that asserts nothing about a cell nobody has counted.
    """
    addrs = np.asarray(addrs)
    q = np.asarray(query)
    out = np.full(len(q), float(fallback), dtype=np.float64)
    if len(addrs) == 0:
        return out
    i = np.searchsorted(addrs, q)
    i_clip = np.clip(i, 0, len(addrs) - 1)
    hit = (i < len(addrs)) & (addrs[i_clip] == q)
    if hit.any():
        j = i_clip[hit]
        t = np.asarray(tot, dtype=np.float64)[j]
        out[hit] = np.where(t > 0, np.asarray(pos, dtype=np.float64)[j] / np.maximum(t, 1.0),
                            fallback)
    return out
