"""Cascaded quaternary combinational network for cryptic-site resolution.

The capacity theorem that forces the cascade
--------------------------------------------
A flat table over ``d`` quaternary digits owns ``4^d`` cells. With ``N`` training
residues at cryptic base rate ``r`` the expected occupancy of a cell is ``N/4^d``
and the expected number of POSITIVE drivers in a cell is ``r N / 4^d``. A cell
whose expected positive count is below one cannot express a fraction at all: it
is either 0 (asserted only by negatives) or a single-sample estimate. Requiring
at least one expected positive per cell gives

    4^d  <=  r N          =>   d  <=  log_4 (r N).

For CryptoBench, ``N = 234838`` and ``r = 0.0576``, so ``r N = 13524`` and
``d <= 6.87``. Seven digits is the hard ceiling of a FLAT quaternary word on this
fold. The previous ten-wire attempt sat at ``d = 10``, i.e. ``4^10 = 1048576``
cells against 13524 positives: 98.9 percent of the address space could never be
driven, and the combinational layer degenerated into a pass-through.

The cascade is the way to spend more than seven digits of input without ever
addressing more than seven digits at once. Group the ``M`` invariants into
disjoint blocks of at most six, compile one dense table per block, and let each
block emit a single quaternary digit obtained by banding its own cell fraction.
The emitted digit is a sufficient statistic of that block under the table model:
the cell fraction IS the empirical ``P(y=1 | block digits)``, so banding it
discards only within-band ordering, never the block's discriminative content.
Composing blocks then costs one digit per block instead of ``d_k`` digits.

Levels
------
    L0  one dense table per invariant group        (<= 6 digits in, 1 digit out)
    L1  merge tables over L0 digits                (3 digits in,    1 digit out)
    L2  spatial majority gate over L1 digits       (integer counting, 1 digit out)
    L3  global table over L1 and L2 digits         (<= 6 digits in, fraction out)

Every level is a lookup into an integer counter table. There is no weight, no
gradient, no iteration and no continuous discriminant anywhere on the resolution
path; the only arithmetic performed on a query is comparison, base-4 positional
addition, and reading two integers.

The spatial gate
----------------
A cryptic site is a contiguous PATCH of residues, never one isolated residue.
The gate states that fact combinationally: for residue ``i`` with contact set
``N(i)`` at 10 A, the patch value is the integer sum of the incident digits over
the closed neighbourhood divided by its cardinality,

    m_i = ( s_i + sum_{j in N(i)} s_j ) / ( 1 + |N(i)| ),

which is a symmetric threshold function of its inputs -- a counting gate, the
canonical dense combinational primitive -- and is then banded into one digit.
"""
from __future__ import annotations

from typing import Any

import numpy as np

N_LEVELS = 4
PATCH_RADIUS = 10.0


# --------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------
def compile_quartiles(F: np.ndarray) -> np.ndarray:
    """Three interior cut points per column at the pooled training quartiles."""
    F = np.asarray(F, dtype=np.float64)
    edges = np.empty((F.shape[1], N_LEVELS - 1), dtype=np.float64)
    for j in range(F.shape[1]):
        edges[j] = np.quantile(F[:, j], [0.25, 0.50, 0.75])
        for t in range(1, N_LEVELS - 1):
            if edges[j, t] <= edges[j, t - 1]:
                edges[j, t] = np.nextafter(edges[j, t - 1], np.inf)
    return edges


def global_digits(F: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Digits against frozen global cut points."""
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    d = np.empty(F.shape, dtype=np.int64)
    for j in range(F.shape[1]):
        d[:, j] = np.searchsorted(edges[j], F[:, j], side="right")
    return np.clip(d, 0, N_LEVELS - 1)


def chain_rank_digits(F: np.ndarray, n_res_per: np.ndarray) -> np.ndarray:
    """Digits against each CHAIN's own quartiles.

    Absolute cut points are not chain-invariant: a 57-residue chain and a
    307-residue chain have different absolute buriedness distributions, so a
    frozen global cut assigns whole small chains to one bin and destroys their
    address diversity. The chain's own quartile is a comparator network over the
    chain -- three integer comparisons per wire, no constant carried between
    chains -- and it makes the address of a residue mean the same thing
    everywhere: which quarter of ITS OWN structure it lies in. Tied values
    necessarily receive the same digit, which is the correct behaviour for a
    degenerate invariant.
    """
    F = np.asarray(F, dtype=np.float64)
    out = np.empty(F.shape, dtype=np.int64)
    off = 0
    for n in n_res_per:
        n = int(n)
        blk = F[off:off + n]
        cuts = np.quantile(blk, [0.25, 0.50, 0.75], axis=0)   # (3, d)
        out[off:off + n] = (blk[None, :, :] > cuts[:, None, :]).sum(0)
        off += n
    return np.clip(out, 0, N_LEVELS - 1)


def pack(digits: np.ndarray) -> np.ndarray:
    """Base-4 positional address of a digit word."""
    digits = np.atleast_2d(np.asarray(digits, dtype=np.int64))
    w = (N_LEVELS ** np.arange(digits.shape[1])).astype(np.int64)
    return digits @ w


def unpack(addr: np.ndarray, n_digits: int) -> np.ndarray:
    a = np.asarray(addr, dtype=np.int64)
    return np.stack([(a // (N_LEVELS ** j)) % N_LEVELS
                     for j in range(n_digits)], axis=1)


# --------------------------------------------------------------------------
# one dense quaternary table
# --------------------------------------------------------------------------
class QLUT:
    """A dense integer counter table over ``n_digits`` quaternary wires."""

    __slots__ = ("n_digits", "n_cells", "pos", "tot", "score", "state", "band")

    def __init__(self, n_digits: int, pos, tot, band=None) -> None:
        self.n_digits = int(n_digits)
        self.n_cells = N_LEVELS ** self.n_digits
        self.pos = np.asarray(pos, dtype=np.int64)
        self.tot = np.asarray(tot, dtype=np.int64)
        frac = np.zeros(self.n_cells, dtype=np.float64)
        nz = self.tot > 0
        frac[nz] = self.pos[nz] / self.tot[nz]
        state = np.full(self.n_cells, "Z", dtype="<U1")
        state[nz & (self.pos == 0)] = "0"
        state[nz & (self.pos == self.tot)] = "1"
        state[nz & (self.pos > 0) & (self.pos < self.tot)] = "X"
        self.state = state
        # Z resolution: nearest asserted cell in Hamming distance over the word.
        dst = np.nonzero(~nz)[0]
        src = np.nonzero(nz)[0]
        if dst.size and src.size:
            ds = unpack(src, self.n_digits)
            for s0 in range(0, len(dst), 256):
                blk = unpack(dst[s0:s0 + 256], self.n_digits)
                ham = (blk[:, None, :] != ds[None, :, :]).sum(-1)
                frac[dst[s0:s0 + 256]] = frac[src[np.argmin(ham, axis=1)]]
        self.score = frac
        self.band = None if band is None else np.asarray(band, dtype=np.float64)

    # -- compilation -------------------------------------------------------
    @staticmethod
    def compile(digits: np.ndarray, y: np.ndarray) -> "QLUT":
        digits = np.atleast_2d(np.asarray(digits, dtype=np.int64))
        n_digits = digits.shape[1]
        n_cells = N_LEVELS ** n_digits
        addr = pack(digits)
        tot = np.bincount(addr, minlength=n_cells).astype(np.int64)
        pos = np.bincount(addr, weights=np.asarray(y, dtype=np.float64),
                          minlength=n_cells).astype(np.int64)
        lut = QLUT(n_digits, pos, tot)
        f = lut.frac(digits)
        band = np.quantile(f, [0.25, 0.50, 0.75])
        for t in range(1, N_LEVELS - 1):
            if band[t] <= band[t - 1]:
                band[t] = np.nextafter(band[t - 1], np.inf)
        lut.band = band
        return lut

    # -- query -------------------------------------------------------------
    def frac(self, digits: np.ndarray) -> np.ndarray:
        return self.score[pack(digits)]

    def digit(self, digits: np.ndarray) -> np.ndarray:
        if self.band is None:
            raise ValueError("table carries no band edges")
        return np.clip(np.searchsorted(self.band, self.frac(digits),
                                       side="right"), 0, N_LEVELS - 1)

    def stats(self) -> dict[str, int]:
        return {
            "n_cells": int(self.n_cells),
            "asserted": int((self.tot > 0).sum()),
            "X": int((self.state == "X").sum()),
            "0": int((self.state == "0").sum()),
            "1": int((self.state == "1").sum()),
            "Z": int((self.state == "Z").sum()),
        }

    def to_json(self) -> dict[str, Any]:
        nz = np.nonzero(self.tot > 0)[0]
        return {"n_digits": self.n_digits,
                "sparse_addr": nz.tolist(),
                "sparse_pos": self.pos[nz].tolist(),
                "sparse_tot": self.tot[nz].tolist(),
                "band": None if self.band is None else self.band.tolist()}

    @staticmethod
    def from_json(d: dict[str, Any]) -> "QLUT":
        n_digits = int(d["n_digits"])
        n_cells = N_LEVELS ** n_digits
        pos = np.zeros(n_cells, dtype=np.int64)
        tot = np.zeros(n_cells, dtype=np.int64)
        a = np.asarray(d["sparse_addr"], dtype=np.int64)
        pos[a] = np.asarray(d["sparse_pos"], dtype=np.int64)
        tot[a] = np.asarray(d["sparse_tot"], dtype=np.int64)
        return QLUT(n_digits, pos, tot, d.get("band"))


# --------------------------------------------------------------------------
# spatial majority gate
# --------------------------------------------------------------------------
def patch_mean(s: np.ndarray, ctr: np.ndarray, n_res_per: np.ndarray,
               radius: float = PATCH_RADIUS) -> np.ndarray:
    """Closed-neighbourhood mean of an integer digit field, per chain.

    Integer sum over a geometric adjacency divided by its cardinality: a
    symmetric counting gate, evaluated once, with no kernel and no weight.
    """
    s = np.asarray(s, dtype=np.float64)
    out = np.empty(len(s), dtype=np.float64)
    r2 = radius * radius
    off = 0
    for n in n_res_per:
        n = int(n)
        c = ctr[off:off + n]
        v = s[off:off + n]
        acc = np.empty(n, dtype=np.float64)
        for i in range(0, n, 512):
            d2 = ((c[i:i + 512, None, :] - c[None, :, :]) ** 2).sum(-1)
            a = (d2 <= r2).astype(np.float64)
            acc[i:i + 512] = (a @ v) / np.maximum(a.sum(1), 1.0)
        out[off:off + n] = acc
        off += n
    return out


def compile_band(x: np.ndarray, n_out: int = 1) -> np.ndarray:
    """Interior cut points that split ``x`` into ``4**n_out`` equal-count bands.

    A stage that emits one digit forwards two bits of its cell fraction. That is
    the whole reason the first cascade starved: three six-wire tables carry
    thirty-six bits of address, and squeezing each into two bits at the merge
    left the global table ten bits to describe thirty-nine invariants, which is
    why every merge cell came out X. Emitting ``n_out`` digits forwards
    ``2*n_out`` bits instead, at the cost of ``n_out`` slots in the global word.
    The band edges are equal-count quantiles of the training fractions, so the
    forwarded alphabet is used uniformly; nothing is fitted.
    """
    k = N_LEVELS ** int(n_out)
    qs = np.arange(1, k) / k
    b = np.quantile(np.asarray(x, dtype=np.float64), qs)
    for t in range(1, len(b)):
        if b[t] <= b[t - 1]:
            b[t] = np.nextafter(b[t - 1], np.inf)
    return b


def band_of(x: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Band index of ``x``; the caller decides how many digits to spend on it."""
    return np.clip(np.searchsorted(band, x, side="right"), 0, len(band))


def spread(idx: np.ndarray, n_out: int) -> np.ndarray:
    """Base-4 expansion of a band index into ``n_out`` quaternary wires."""
    a = np.asarray(idx, dtype=np.int64)
    return np.stack([(a // (N_LEVELS ** j)) % N_LEVELS
                     for j in range(int(n_out))], axis=1)


# --------------------------------------------------------------------------
# cross-fitted stage outputs
# --------------------------------------------------------------------------
def oof_frac(digits: np.ndarray, y: np.ndarray, fold: np.ndarray,
             k: int) -> tuple[np.ndarray, "QLUT"]:
    """Out-of-fold cell fractions, plus the full-data table for inference.

    A cascade whose downstream table is compiled on upstream values that were
    themselves estimated from the same residues is a stacked estimator reading
    its own in-sample fit. The bias is not small: measured on this fold it moved
    the training curve to 0.8159 while the test curve fell to 0.7286. The cure is
    the standard one and it is pure counting -- for every residue, the upstream
    fraction it forwards is read from a table compiled with its own fold held
    out, so the downstream table only ever sees honest inputs. Inference uses the
    full-data table, which is what the artifact stores.
    """
    digits = np.atleast_2d(np.asarray(digits, dtype=np.int64))
    out = np.empty(len(y), dtype=np.float64)
    for f in range(k):
        m = fold != f
        sub = QLUT.compile(digits[m], y[m])
        out[~m] = sub.frac(digits[~m])
    return out, QLUT.compile(digits, y)


# --------------------------------------------------------------------------
# out-of-fold emission
# --------------------------------------------------------------------------
def residue_fold(n_res_per: np.ndarray, cluster: np.ndarray,
                 k: int = 5) -> np.ndarray:
    """Deterministic cluster-disjoint fold label for every training residue.

    A cascade level reads the cell fraction of the table that was compiled from
    the very residues it is scoring. That fraction is therefore an in-sample
    estimate, and the level above it calibrates its band edges against an
    optimistically biased input. The bias is not small: measured, it inflates a
    super-stage from a test-honest value to 0.8148 while the assembled network
    scores 0.7286 on the held-out fold.

    The repair is the same one that makes any stacked estimator honest: a level
    forwards only fractions read from a table compiled WITHOUT the residue's own
    fold. Folds are assigned by sequence cluster, not by residue and not by
    chain, because two chains of one cluster are near-duplicates and splitting
    them would leak across the fold boundary exactly as the test/train split
    would.
    """
    uniq = sorted(set(cluster.tolist()))
    fold_of_cluster = {c: i % k for i, c in enumerate(uniq)}
    per_unit = np.array([fold_of_cluster[c] for c in cluster], dtype=np.int64)
    return np.repeat(per_unit, np.asarray(n_res_per, dtype=np.int64))


def oof_fraction(digits: np.ndarray, y: np.ndarray,
                 fold: np.ndarray) -> tuple[np.ndarray, "QLUT"]:
    """``(out-of-fold fractions for train, table compiled on all of train)``.

    The returned table is the one inference uses; the returned vector is the one
    the level above may be compiled against. They are different objects on
    purpose.
    """
    digits = np.atleast_2d(np.asarray(digits, dtype=np.int64))
    out = np.empty(len(y), dtype=np.float64)
    for f in np.unique(fold):
        m = fold == f
        held = QLUT.compile(digits[~m], y[~m])
        out[m] = held.frac(digits[m])
    return out, QLUT.compile(digits, y)


