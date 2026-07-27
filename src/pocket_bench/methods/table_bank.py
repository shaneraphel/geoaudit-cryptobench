"""A bank of small exact tables, held as counts rather than as a matrix.

What the bank is
----------------
Every wire is a quaternary digit, obtained by ranking the residue against the
other residues of its own chain and cutting at quartiles. A table is a short
tuple of wires; the digits of those wires concatenate into an address, and the
table stores two integers per address, counted on the training fold:

    tot_k[a]  residues of the training fold whose address in table k is a
    pos_k[a]  how many of them are cryptic

Scoring reads one cell per table and adds the cells with integer weights,

    S(i) = sum_k m_k * pos_k[a_k(i)] / tot_k[a_k(i)],    m_k integer.

Two-wire tables have sixteen cells and three-wire tables sixty-four, so with a
training fold of 235k residues a cell holds thousands of them and its ratio is
not a noisy quantity. This is the whole reason the bank is built from many
narrow tables rather than a few wide ones: a six-wire table has 4096 cells,
holds tens of residues per cell, and states the joint distribution exactly while
estimating it badly.

Why nothing is materialised
---------------------------
The obvious representation -- one float column per table -- costs rows x tables
x 4 bytes and reached several gigabytes at the pool sizes that work best, which
killed three runs outright and forced the earlier sweeps to stop at pool sizes
the method did not want to stop at. Addresses are cheap to recompute, so the
bank is never assembled: callers iterate over row blocks and each block's
values are formed on demand. Memory is then set by the block, not by the bank,
and the only object that grows with the number of tables is the K x K system
solved once at compile time.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

N_LEVELS = 4
BLOCK = 8192


def partition_tables(n_wires: int, width: int, rounds: int, seed: int
                     ) -> list[list[int]]:
    """``rounds`` random partitions of every wire into groups of ``width``.

    A partition rather than a random draw of tuples: within one round each wire
    appears exactly once, so no wire is over- or under-represented, and across
    rounds each wire meets a different set of partners. Coverage of the pair
    lattice comes from the number of rounds.
    """
    rng = np.random.default_rng(seed)
    tables: list[list[int]] = []
    for _ in range(rounds):
        perm = rng.permutation(n_wires)
        tables += [perm[i:i + width].tolist()
                   for i in range(0, n_wires, width)]
    return [t for t in tables if len(t) >= 2]


def cell_offsets(tables) -> np.ndarray:
    """Start of each table's cells inside one concatenated array."""
    sizes = [N_LEVELS ** len(t) for t in tables]
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


def addresses(D: np.ndarray, tables, offsets: np.ndarray,
              a: int, b: int) -> np.ndarray:
    """``(b - a, n_tables)`` addresses into the concatenated cell array."""
    out = np.empty((b - a, len(tables)), dtype=np.int64)
    for k, cols in enumerate(tables):
        acc = np.zeros(b - a, dtype=np.int64)
        for t, c in enumerate(cols):
            acc += D[a:b, c].astype(np.int64) * (N_LEVELS ** t)
        out[:, k] = acc + offsets[k]
    return out


def compile_cells(D: np.ndarray, y: np.ndarray, tables,
                  offsets: np.ndarray) -> np.ndarray:
    """Cell frequencies over the whole concatenated address space.

    An address never seen in training takes the fold's base rate, which is the
    only value that adds no information; it is a Z state in the sense the
    manuscript uses, and the count of them is worth reporting.
    """
    n = D.shape[0]
    total = int(offsets[-1])
    tot = np.zeros(total, dtype=np.int64)
    pos = np.zeros(total, dtype=np.float64)
    yf = y.astype(np.float64)
    for a in range(0, n, BLOCK):
        b = min(a + BLOCK, n)
        ad = addresses(D, tables, offsets, a, b)
        flat = ad.ravel()
        tot += np.bincount(flat, minlength=total)
        pos += np.bincount(flat, weights=np.repeat(yf[a:b], len(tables)),
                           minlength=total)
    rate = float(yf.mean())
    frac = np.where(tot > 0, pos / np.maximum(tot, 1), rate)
    return frac, tot


def blocks(D: np.ndarray, tables, offsets: np.ndarray, frac: np.ndarray,
           ) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(a, b, values)`` with ``values`` the table outputs of the rows."""
    n = D.shape[0]
    for a in range(0, n, BLOCK):
        b = min(a + BLOCK, n)
        yield a, b, frac[addresses(D, tables, offsets, a, b)]


def scatter_and_means(D, y, tables, offsets, frac):
    """Within-class scatter and class means of the table outputs."""
    K = len(tables)
    s1 = np.zeros(K); s0 = np.zeros(K)
    pos = y == 1
    n1 = int(pos.sum()); n0 = int(len(y) - n1)
    for a, b, v in blocks(D, tables, offsets, frac):
        p = pos[a:b]
        s1 += v[p].sum(0)
        s0 += v[~p].sum(0)
    mu1, mu0 = s1 / max(n1, 1), s0 / max(n0, 1)
    S = np.zeros((K, K))
    for a, b, v in blocks(D, tables, offsets, frac):
        p = pos[a:b]
        c = np.where(p[:, None], v - mu1, v - mu0)
        S += c.T @ c
    return S / max(len(y) - 2, 1), mu1, mu0


def integer_fanout(D, y, tables, offsets, frac, ridge: float, cap: int
                   ) -> np.ndarray:
    """Integer multiplicities from a regularised closed-form direction.

    One symmetric solve, rounded onto ``[-cap, cap]``. The ridge is not
    cosmetic: random pairs drawn from the pair lattice repeat as the pool grows,
    the scatter goes near-singular, and without it the direction chases the null
    space -- measured as a fall from 0.7844 to 0.6846 when the pool went from
    1032 to 1720 tables. Inference remains integer arithmetic over table cells.
    """
    S, mu1, mu0 = scatter_and_means(D, y, tables, offsets, frac)
    K = S.shape[0]
    S.flat[::K + 1] += ridge * float(np.trace(S)) / K + 1e-12
    w = np.linalg.solve(S, mu1 - mu0)
    peak = float(np.abs(w).max())
    if peak <= 0:
        return np.zeros(K, dtype=np.int64)
    return np.round(w / peak * cap).astype(np.int64)


def score(D, tables, offsets, frac, mult: np.ndarray) -> np.ndarray:
    """The integer-weighted sum of table cells, one value per residue."""
    out = np.empty(D.shape[0], dtype=np.float64)
    m = mult.astype(np.float64)
    for a, b, v in blocks(D, tables, offsets, frac):
        out[a:b] = v @ m
    return out


def chain_digits(F: np.ndarray, n_res_per) -> np.ndarray:
    """Quaternary digits by within-chain rank, ties sharing a mid-rank.

    Ranking inside the chain is what makes a wire mean the same thing in a
    57-residue chain and a 307-residue one, and it removes any dependence on
    absolute units, so no constant has to be carried from the training fold to
    score a new structure.
    """
    out = np.empty(F.shape, dtype=np.int8)
    off = 0
    for n in n_res_per:
        n = int(n)
        blk = F[off:off + n]
        r = np.empty(n)
        for j in range(F.shape[1]):
            x = blk[:, j]
            order = np.argsort(x, kind="stable")
            i = 0
            while i < n:
                k = i
                while k + 1 < n and x[order[k + 1]] == x[order[i]]:
                    k += 1
                r[order[i:k + 1]] = 0.5 * (i + k)
                i = k + 1
            out[off:off + n, j] = np.clip(
                np.floor(r / max(n - 1, 1) * N_LEVELS), 0, N_LEVELS - 1)
        off += n
    return out
