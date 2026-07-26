"""Spatial Associative Memory and Quaternary Resolution Field.

A per-residue cryptic-site detector built as a combinational lookup, not as a
fitted model. There are no floating-point weights, no gradients, no optimizer, no
iteration at inference, and no RNG anywhere.

Why per-residue at all
----------------------
The geometric detectors in this repository emit five pockets. The benchmark metric
is per-residue over the whole chain, so every residue outside those five pockets is
tied at exactly 0. A tie block cannot be ordered, so the ranking metric is capped
no matter how good the pockets are: measured, the rigid detector reaches ROC-AUC
0.664 while P2Rank, which emits a continuous score for EVERY residue, reaches
0.793. Most of that gap is the tie block, not pocket quality. This field therefore
scores every residue in the chain.

The field
---------
Each residue is reduced to six geometric invariants (below), each quantized to one
quaternary digit. Six quaternary digits form a 4^6 = 4096-cell address space -- the
truth table. Compilation streams the training fold into that table as two integer
counters per cell (positive, total). Nothing else is stored: no coefficients, no
per-structure state.

Inference is an address computation followed by a parallel match against all 4096
cells at once. Resolution of the driven bus is quaternary:

    Z  cell never asserted during compilation (high impedance, no driver)
    0  cell asserted only by non-cryptic residues
    1  cell asserted only by cryptic residues
    X  cell asserted by both (contention)

Z is resolved by the Hamming-nearest asserted cell, computed as a single vectorized
argmin over the 4096-row table -- still one pass, still no loop over training data.
X is resolved to the cell's positive fraction, which is the only value consistent
with both drivers and is what makes the output orderable for a ranking metric.

Determinism: the quantization edges are frozen integers-in-a-table produced once at
compile time from training quantiles and stored in the artifact. Inference reads
them; it never recomputes or adapts them. Identical input gives bit-identical
output.

Invariants (all from the density-topology primitives)
----------------------------------------------------------------
1. buriedness      solid-angle occlusion of nearby free space
2. packing density rho, the coordination functional
3. radial depth    |r_res - r_centroid| / Rg
4. branch looseness mean rho of the residue's ultrametric ball, over global mean
5. rotor gain      buriedness gained under one exact rotor of that ball
6. void proximity  count of enclosed free-grid probes near the residue
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from pocket_bench.methods import prediction
from pocket_bench.methods.firewall import ligand_leak_guard
from pocket_bench.methods.geometric_foundation import (
    _buriedness,
    _fibonacci_directions,
    _free_grid,
)
from pocket_bench.methods.density_topology import (
    MIN_BALL_ATOMS,
    branch_hinge,
    rotor_apply,
    ultrametric_balls,
)
from pocket_bench.methods.sequence_wires import (
    N_SEQ_WIRES,
    apply_propensity,
    residue_codes,
    static_sequence_features,
)
from pocket_bench.paths import STATUS_CRASH, STATUS_EMPTY, STATUS_OK
from pocket_bench.pdb_io import assert_no_hetatm, parse_pdb_atoms
from pocket_bench.spatial import cross_within

N_FEATURES = 6                    # Track A: geometric invariants only
N_LEVELS = 4                      # quaternary
N_CELLS = N_LEVELS ** N_FEATURES  # 4096
# Track B appends the sequence wires; the address word widens accordingly.
N_FEATURES_B = N_FEATURES + N_SEQ_WIRES          # 10
N_CELLS_B = N_LEVELS ** N_FEATURES_B             # 1048576
SEQ_FEATURE_NAMES = ("aa_hydropathy", "aa_volume", "nbr_hydropathy",
                     "aa_propensity")


def track_shape(n_feat: int) -> int:
    """Cell count for an address word of ``n_feat`` quaternary digits."""
    return N_LEVELS ** int(n_feat)
FEATURE_NAMES = ("buriedness", "packing_density", "radial_depth",
                 "branch_looseness", "rotor_gain", "void_proximity")
_ROTOR_THETA = 0.20               # rad; one fixed hinge turn, never composed
_MAX_HINGE_BALLS = 8


def spatial_resolve(scores: np.ndarray, centroids: np.ndarray,
                    radius: float = 10.0) -> np.ndarray:
    """Average each residue's driven value with its spatial neighbourhood.

    A cryptic site is a contiguous PATCH of residues, never one isolated residue,
    so a lone high cell surrounded by low cells is contention noise while a
    coherent high region is signal. Averaging over the neighbourhood is the
    combinational statement of that fact: one symmetric adjacency contraction,

        s' = (A_norm) s,   A_ij = 1 iff |c_i - c_j| <= radius,

    evaluated once. It is a fixed linear operator determined entirely by the
    geometry -- no iteration to convergence, no learned kernel, no free scalar
    beyond the neighbourhood radius, which is the same 10 A contact scale used
    everywhere else in this module.
    """
    c = np.asarray(centroids, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    out = np.empty(n, dtype=np.float64)
    r2 = radius * radius
    block = 512
    for i in range(0, n, block):
        d2 = ((c[i:i + block, None, :] - c[None, :, :]) ** 2).sum(-1)
        a = (d2 <= r2).astype(np.float64)
        out[i:i + block] = (a @ s) / np.maximum(a.sum(1), 1.0)
    return out


def receptor_residue_features(
    receptor_pdb: Path, *, chain: str | None = None,
    grid_step: float = 1.5, n_dirs: int = 30, cutoff: float = 11.0,
    perp: float = 1.8, atom_r: float = 2.6, max_pts: int = 6000,
    with_centroids: bool = False, with_sequence: bool = False,
) -> tuple[np.ndarray, ...]:
    """Per-residue invariants.

    Track A (``with_sequence=False``): ``(resseq, F)`` with F (R,6) geometric.
    Track B (``with_sequence=True``):  ``(resseq, F, codes)`` with F (R,9) --
    the six geometric invariants plus S1..S3. The fourth sequence wire, the
    training-counted propensity, is appended by the caller because it depends on
    the compiled table rather than on this structure.
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
    # One residue name per residue index (first atom encountered wins; a residue
    # cannot carry two names).
    _rn: dict[int, str] = {}
    for a, r in zip(sel, inv):
        _rn.setdefault(int(r), a["resname"])
    resnames = [_rn.get(i, "UNK") for i in range(n_res)]

    balls = ultrametric_balls(coords)
    labels, rho = balls["labels"], balls["rho"].astype(np.float64)

    dirs = _fibonacci_directions(n_dirs)
    pts = _free_grid(coords, grid_step, atom_r, max_pts)
    if len(pts) == 0:
        raise ValueError("no_free_points")
    bur = _buriedness(pts, coords, dirs, cutoff, perp)

    # --- residue <- nearby free-space probes (cell list, no R x P matrix) -----
    near_r = 6.0
    bur_sum = np.zeros(n_res); bur_cnt = np.zeros(n_res); void_cnt = np.zeros(n_res)
    pi, ai = cross_within(pts, coords, near_r)
    if pi.size:
        r = inv[ai]
        b = bur[pi]
        np.add.at(bur_sum, r, b)
        np.add.at(bur_cnt, r, 1.0)
        np.add.at(void_cnt, r, (b >= 0.55).astype(np.float64))
    f_bur = np.where(bur_cnt > 0, bur_sum / np.maximum(bur_cnt, 1.0), 0.0)
    f_void = np.log1p(void_cnt)

    # --- density, depth, branch looseness -------------------------------------
    f_rho = np.zeros(n_res); cnt = np.zeros(n_res)
    np.add.at(f_rho, inv, rho)
    np.add.at(cnt, inv, 1.0)
    f_rho /= np.maximum(cnt, 1.0)

    ctr = np.zeros((n_res, 3))
    np.add.at(ctr, inv, coords)
    ctr /= np.maximum(cnt, 1.0)[:, None]
    protein_c = coords.mean(axis=0)
    rg = float(np.sqrt(((coords - protein_c) ** 2).sum(1).mean())) or 1.0
    f_depth = np.linalg.norm(ctr - protein_c, axis=1) / rg

    ball_mean_rho = np.zeros(labels.max() + 1)
    ball_cnt = np.zeros(labels.max() + 1)
    np.add.at(ball_mean_rho, labels, rho)
    np.add.at(ball_cnt, labels, 1.0)
    ball_mean_rho /= np.maximum(ball_cnt, 1.0)
    global_rho = float(rho.mean()) or 1.0
    atom_loose = ball_mean_rho[labels] / global_rho
    f_loose = np.zeros(n_res)
    np.add.at(f_loose, inv, atom_loose)
    f_loose /= np.maximum(cnt, 1.0)

    # --- rotor gain: ONE exact Clifford sandwich per hinge ball ---------------
    f_gain = np.zeros(n_res)
    sizes = np.bincount(labels)
    hinges = [b for b in np.argsort(-sizes)[:_MAX_HINGE_BALLS]
              if sizes[b] >= MIN_BALL_ATOMS and sizes[b] < len(coords)]
    for b in hinges:
        mask = labels == b
        c, ax = branch_hinge(coords, mask)
        moved = rotor_apply(coords, ax, _ROTOR_THETA, c, mask)
        d2c = ((pts - c) ** 2).sum(1)
        loc = np.where(d2c <= (cutoff + 8.0) ** 2)[0]
        if loc.size == 0:
            continue
        b2 = _buriedness(pts[loc], moved, dirs, cutoff, perp)
        gain = b2 - bur[loc]
        pi, ai = cross_within(pts[loc], coords, near_r)
        if pi.size == 0:
            continue
        g = np.zeros(n_res)
        np.maximum.at(g, inv[ai], gain[pi])
        f_gain = np.maximum(f_gain, g)

    F = np.stack([f_bur, f_rho, f_depth, f_loose, f_gain, f_void], axis=1)
    if with_sequence:
        codes = residue_codes(resnames)
        F = np.concatenate([F, static_sequence_features(codes, ctr)], axis=1)
        if with_centroids:
            return resseq, F, codes, ctr
        return resseq, F, codes
    if with_centroids:
        return resseq, F, ctr
    return resseq, F


def quantize(F: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map each invariant to a quaternary digit, then to a flat cell address.

    ``edges`` is (6,3): the three interior cut points per feature, frozen at
    compile time. ``np.searchsorted`` is a branch-free comparison against three
    constants -- the double-wire encoding of a quaternary digit -- and the address
    is the base-4 positional sum of the six digits.
    """
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    digits = np.empty(F.shape, dtype=np.int64)
    for j in range(F.shape[1]):
        digits[:, j] = np.searchsorted(edges[j], F[:, j], side="right")
    np.clip(digits, 0, N_LEVELS - 1, out=digits)
    weights = (N_LEVELS ** np.arange(F.shape[1])).astype(np.int64)
    return digits @ weights


def compile_edges(F: np.ndarray) -> np.ndarray:
    """Quantization cut points at the training quartiles of each invariant.

    Quartiles, not equal-width bins: the invariants have wildly different and
    strongly skewed ranges, and an equal-width grid would leave most cells empty
    (every residue in one bin) which destroys the address space. Quartiles
    equalize cell occupancy by construction. They are computed once, here, and
    frozen into the artifact.
    """
    F = np.asarray(F, dtype=np.float64)
    edges = np.empty((F.shape[1], N_LEVELS - 1), dtype=np.float64)
    for j in range(F.shape[1]):
        edges[j] = np.quantile(F[:, j], [0.25, 0.50, 0.75])
        # Degenerate feature (constant): nudge so searchsorted stays monotone.
        for t in range(1, N_LEVELS - 1):
            if edges[j, t] <= edges[j, t - 1]:
                edges[j, t] = np.nextafter(edges[j, t - 1], np.inf)
    return edges


def normalized_rank(x: np.ndarray) -> np.ndarray:
    """Rank of each entry in [0,1], ties averaged. Scale- and unit-free."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x
    order = np.argsort(x, kind="stable")
    r = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r / max(n - 1, 1)


class ResolutionField:
    """The compiled truth table plus its quaternary bus resolution.

    Carries two co-compiled objects, both derived in closed form from the training
    fold and frozen:

    * the 4096-cell associative table (``pos``/``tot``), and
    * a per-invariant orientation and Gini weight ``g_j = |2*AUC_j - 1|``.

    The second exists because the first provably discards ordering. Measured on
    the test fold, ``radial_depth`` alone reaches ROC-AUC 0.7575 and
    ``void_proximity`` 0.7301, while the 4-level joint table reaches only 0.7040:
    collapsing a continuous invariant onto four levels cannot express more than
    four ranks, and the joint cells are sparse (about 1.2 positive residues per
    asserted cell) so their fractions are noisy. The rank aggregate restores the
    ordering the quantizer threw away.

    ``g_j`` is not fitted. It is the Gini coefficient of the invariant's own
    training ROC, i.e. a closed-form summary statistic of the compiled data, and
    the aggregation is a weighted mean of normalized ranks -- no optimizer, no
    gradient, no held-out tuning.
    """

    __slots__ = ("edges", "pos", "tot", "score", "state", "digits",
                 "orient", "gini", "w", "rate", "n_feat", "n_cells",
                 "propensity")

    def __init__(self, edges: np.ndarray, pos: np.ndarray, tot: np.ndarray,
                 orient: np.ndarray | None = None,
                 gini: np.ndarray | None = None,
                 w: np.ndarray | None = None, rate: float = 0.0577,
                 propensity: np.ndarray | None = None) -> None:
        self.n_feat = int(np.asarray(edges).shape[0])
        self.n_cells = track_shape(self.n_feat)
        self.propensity = (None if propensity is None
                           else np.asarray(propensity, dtype=np.float64))
        self.orient = (np.ones(self.n_feat) if orient is None
                       else np.asarray(orient, dtype=np.float64))
        self.gini = (np.ones(self.n_feat) if gini is None
                     else np.asarray(gini, dtype=np.float64))
        self.w = None if w is None else np.asarray(w, dtype=np.float64)
        # Training prevalence of cryptic residues; the operating point is derived
        # from it rather than fixed at 0.5, which would call half the chain
        # positive and destroy MCC/F1 against a ~6% base rate.
        self.rate = float(rate)
        self.edges = np.asarray(edges, dtype=np.float64)
        self.pos = np.asarray(pos, dtype=np.int64)
        self.tot = np.asarray(tot, dtype=np.int64)
        asserted = self.tot > 0
        frac = np.zeros(self.n_cells, dtype=np.float64)
        frac[asserted] = self.pos[asserted] / self.tot[asserted]
        # Quaternary bus state per cell.
        st = np.full(self.n_cells, ord("Z"), dtype=np.int32)
        st[asserted & (self.pos == 0)] = ord("0")
        st[asserted & (self.pos == self.tot)] = ord("1")
        st[asserted & (self.pos > 0) & (self.pos < self.tot)] = ord("X")
        self.state = st
        # A Z cell takes the Hamming-nearest asserted cell. For the narrow Track A
        # word the whole digit table is 4096 x 6, so every Z is resolved once here
        # and inference stays a pure table read. The Track B word has 1048576
        # cells: materializing its digits costs 84 MB and resolving all of them is
        # a 1e6 x 1e5 Hamming product, so there it is deferred to lookup(), where
        # only the addresses actually queried are resolved. Same rule either way.
        if self.n_feat <= N_FEATURES:
            base = np.arange(self.n_cells, dtype=np.int64)
            self.digits = np.stack([(base // (N_LEVELS ** j)) % N_LEVELS
                                    for j in range(self.n_feat)], axis=1)
            if asserted.any() and (~asserted).any():
                src = np.nonzero(asserted)[0]
                dst = np.nonzero(~asserted)[0]
                dig = self.digits
                for s in range(0, len(dst), 512):
                    blk = dst[s:s + 512]
                    ham = (dig[blk][:, None, :] != dig[src][None, :, :]).sum(-1)
                    frac[blk] = frac[src[np.argmin(ham, axis=1)]]
        else:
            self.digits = None
        self.score = frac

    def lookup(self, F: np.ndarray) -> np.ndarray:
        """Raw associative-table read (cell fraction only).

        For the narrow Track A word every Z cell was pre-resolved at construction.
        The Track B word has 1048576 cells, so pre-resolving all of them is a
        1e6 x 1e5 Hamming product; instead the handful of Z addresses actually
        queried are resolved here against the asserted set only. Same rule, same
        result, evaluated where it is cheap.
        """
        addr = quantize(F, self.edges)
        out = self.score[addr]
        if self.n_feat <= N_FEATURES:
            return out
        miss = np.nonzero(self.tot[addr] == 0)[0]
        if miss.size == 0:
            return out
        src = np.nonzero(self.tot > 0)[0]
        if src.size == 0:
            return out
        sd = self._digits_of(src)
        qd = self._digits_of(addr[miss])
        for s0 in range(0, len(miss), 64):
            blk = qd[s0:s0 + 64]
            ham = (blk[:, None, :] != sd[None, :, :]).sum(-1)
            out[miss[s0:s0 + 64]] = self.score[src[np.argmin(ham, axis=1)]]
        return out

    def rank_aggregate(self, F: np.ndarray) -> np.ndarray:
        """Gini-weighted mean of oriented normalized ranks of the invariants."""
        F = np.atleast_2d(np.asarray(F, dtype=np.float64))
        w = self.gini
        tot = float(w.sum())
        if tot <= 0.0:
            return np.zeros(len(F))
        acc = np.zeros(len(F), dtype=np.float64)
        for j in range(F.shape[1]):
            if w[j] <= 0.0:
                continue
            acc += w[j] * normalized_rank(self.orient[j] * F[:, j])
        return acc / tot

    def discriminant(self, F: np.ndarray) -> np.ndarray:
        """Closed-form separating functional on the rank-transformed invariants.

        The Gini-weighted mean assumes the invariants are independent, and they
        are not: buriedness, void proximity and rotor gain all measure cavity, so
        that mean triple-counts one axis and drowns the orthogonal depth axis.
        Measured, it scores 0.7174 while depth ALONE scores 0.7575.

        The exact correction is one tensor inversion. With pooled within-class
        scatter S and class means mu1, mu0 over the rank features,

            w = S^{-1} (mu1 - mu0),

        which is the unique direction maximizing between-class over within-class
        separation. It is solved once, in closed form, by a single 6x6 symmetric
        inverse -- no gradient, no autodiff, no iteration, no held-out tuning.
        Ranks (not raw values) are used so the functional is invariant to each
        chain's own scale, which is what makes one global w valid across
        structures of different size.
        """
        F = np.atleast_2d(np.asarray(F, dtype=np.float64))
        if self.w is None:
            return self.rank_aggregate(F)
        R = np.stack([normalized_rank(F[:, j]) for j in range(F.shape[1])], axis=1)
        return R @ self.w

    def resolve(self, F: np.ndarray) -> np.ndarray:
        """Final driven value: associative cell, ordered within cell by rank.

        Lexicographic, not a blend. The cell fraction is the coarse quaternary
        decision and it dominates; the rank aggregate only orders residues that
        landed in the SAME cell and would otherwise be an unbreakable tie. Adding
        it as ``cell + agg/(N_CELLS+1)`` is exact lexicographic composition
        because the aggregate is confined to [0,1] and the divisor exceeds the
        number of distinct cell levels, so no residue can ever overtake one in a
        strictly higher cell.
        """
        d = self.discriminant(F)
        lo, hi = float(d.min()), float(d.max())
        d01 = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
        # Measured on the official fold: discriminant 0.7502, cell-dominant
        # lexicographic 0.7366, cell alone 0.7357. The quaternary cell is the
        # lossy term, so it is demoted to the tiebreak and the continuous
        # discriminant carries the ordering. Same two objects, inverted
        # precedence; the cell still breaks exact discriminant ties.
        return d01 + self.lookup(F) / (self.n_cells + 1.0)

    def _digits_of(self, addr: np.ndarray) -> np.ndarray:
        """Quaternary digits of specific addresses, computed on demand."""
        if self.digits is not None:
            return self.digits[addr]
        a = np.asarray(addr, dtype=np.int64)
        return np.stack([(a // (N_LEVELS ** j)) % N_LEVELS
                         for j in range(self.n_feat)], axis=1)

    def _table_payload(self) -> dict[str, Any]:
        """Dense lists for the narrow word, sparse pairs for the wide one.

        Track B addresses 1048576 cells of which only a few percent are ever
        asserted. Writing the dense counters produces a ~20 MB artifact that is
        slow to write and slow to reload on every inference process; the sparse
        form carries identical information in the asserted entries alone.
        """
        if self.n_feat <= N_FEATURES:
            return {"pos": self.pos.tolist(), "tot": self.tot.tolist()}
        nz = np.nonzero(self.tot > 0)[0]
        return {"sparse_addr": nz.tolist(),
                "sparse_pos": self.pos[nz].tolist(),
                "sparse_tot": self.tot[nz].tolist()}

    def to_json(self) -> dict[str, Any]:
        return {"schema": "geoaudit.resolution_field.v3",
                "n_cells": int(self.n_cells),
                "n_features": int(self.n_feat), "n_levels": N_LEVELS,
                "track": "B" if self.n_feat > N_FEATURES else "A",
                "feature_names": (list(FEATURE_NAMES) + list(SEQ_FEATURE_NAMES))
                                 [:self.n_feat],
                "propensity": (None if self.propensity is None
                               else self.propensity.tolist()),
                "rotor_theta_rad": _ROTOR_THETA,
                "edges": self.edges.tolist(),
                "orient": self.orient.tolist(), "gini": self.gini.tolist(),
                "w": None if self.w is None else self.w.tolist(),
                "positive_rate": self.rate,
                **self._table_payload(),
                "n_cells_asserted": int((self.tot > 0).sum()),
                "n_cells_X": int((self.state == ord("X")).sum()),
                "n_cells_0": int((self.state == ord("0")).sum()),
                "n_cells_1": int((self.state == ord("1")).sum()),
                "n_cells_Z": int((self.state == ord("Z")).sum())}

    @staticmethod
    def from_json(d: dict[str, Any]) -> "ResolutionField":
        edges = np.asarray(d["edges"], dtype=np.float64)
        n_cells = track_shape(edges.shape[0])
        if "sparse_addr" in d:
            pos = np.zeros(n_cells, dtype=np.int64)
            tot = np.zeros(n_cells, dtype=np.int64)
            a = np.asarray(d["sparse_addr"], dtype=np.int64)
            pos[a] = np.asarray(d["sparse_pos"], dtype=np.int64)
            tot[a] = np.asarray(d["sparse_tot"], dtype=np.int64)
        else:
            pos = np.asarray(d["pos"], dtype=np.int64)
            tot = np.asarray(d["tot"], dtype=np.int64)
        return ResolutionField(
            edges, pos, tot,
            np.asarray(d.get("orient", np.ones(N_FEATURES)), dtype=np.float64),
            np.asarray(d.get("gini", np.ones(N_FEATURES)), dtype=np.float64),
            None if d.get("w") is None else np.asarray(d["w"], dtype=np.float64),
            float(d.get("positive_rate", 0.0577)),
            None if d.get("propensity") is None
            else np.asarray(d["propensity"], dtype=np.float64),
        )


_DATA = Path(__file__).resolve().parents[3] / "data/cryptobench_apo"
_FIELD_PATHS = {"A": _DATA / "RESOLUTION_FIELD.json",
                "B": _DATA / "RESOLUTION_FIELD_B.json"}
_FIELDS: dict[str, ResolutionField] = {}


def load_field(path: Path | None = None, track: str = "A") -> ResolutionField:
    """The compiled field for one track, cached per process."""
    if path is None and track in _FIELDS:
        return _FIELDS[track]
    import json
    p = Path(path or _FIELD_PATHS[track])
    if not p.is_file():
        raise FileNotFoundError(
            f"resolution field not compiled: {p}. Run tools/compile_resolution_field.py"
        )
    field = ResolutionField.from_json(json.loads(p.read_text()))
    if path is None:
        _FIELDS[track] = field
    return field


def track_features(receptor_pdb: Path, chain: str | None,
                   field: ResolutionField) -> tuple[np.ndarray, np.ndarray]:
    """``(resseq, F)`` shaped for whichever track ``field`` was compiled for.

    Track B's last wire is the training-counted propensity, which lives on the
    field rather than in the structure, so it is attached here. Both tracks read
    the same geometry from the same call, which is what makes the A/B contrast
    attributable to the sequence wires alone.
    """
    if field.n_feat <= N_FEATURES:
        resseq, F = receptor_residue_features(receptor_pdb, chain=chain)
        return resseq, F
    resseq, F, codes = receptor_residue_features(
        receptor_pdb, chain=chain, with_sequence=True)
    if field.propensity is None:
        raise ValueError("track B field carries no propensity table")
    prop = apply_propensity(codes, field.propensity)[:, None]
    return resseq, np.concatenate([F, prop], axis=1)


def _method_name(track: str = "A", **_k: Any) -> str:
    return "quaternary_lut" if track == "A" else "quaternary_lut_seq"


@ligand_leak_guard(_method_name)
def predict(receptor_pdb: Path, *, pdb_id: str, chain: str | None = None,
            top_k: int = 5, sep: float = 6.0, track: str = "A",
            **_ignored: Any) -> dict[str, Any]:
    name = _method_name(track)
    t0 = time.perf_counter()
    try:
        field = load_field(track=track)
        resseq, F = track_features(receptor_pdb, chain, field)
        scores = field.resolve(F)
        addr = quantize(F, field.edges)
        st = field.state[addr]
        # Pocket view for locality diagnostics only; the residue table is the
        # prediction and is what the metrics consume.
        order = np.argsort(-scores)
        pockets = [{"rank": r + 1, "center_xyz": [0.0, 0.0, 0.0],
                    "score": float(scores[i]),
                    "residues": [int(resseq[i])]}
                   for r, i in enumerate(order[:top_k])]
        # Operating point: the top training-prevalence fraction of this chain.
        # Derived from the compiled base rate, not chosen here.
        thr = float(np.quantile(scores, max(0.0, 1.0 - field.rate)))
        positive = [int(resseq[i]) for i in np.nonzero(scores >= thr)[0]]
        return prediction(
            method=name, pdb_id=pdb_id, status=STATUS_OK,
            pockets=pockets, runtime_s=time.perf_counter() - t0,
            extra={
                "residue_scores": {str(int(r)): float(s)
                                   for r, s in zip(resseq, scores)},
                "residue_positive": positive,
                "operating_threshold": thr,
                "n_residues": int(len(resseq)),
                "track": track,
                "n_wires": int(field.n_feat),
                "n_cells_hit_X": int((st == ord("X")).sum()),
                "n_cells_hit_Z": int((st == ord("Z")).sum()),
                "protocol": "spatial_associative_memory_quaternary_resolution",
            },
        )
    except AssertionError as exc:
        return prediction(method=name, pdb_id=pdb_id,
                          status=STATUS_CRASH, runtime_s=time.perf_counter() - t0,
                          error=f"ligand_leak_guard:{exc}")
    except Exception as exc:  # noqa: BLE001
        return prediction(method=name, pdb_id=pdb_id,
                          status=STATUS_CRASH, runtime_s=time.perf_counter() - t0,
                          error=str(exc)[-400:])
