"""Forty-eight quantities of what the crystallographer could not pin down.

Why this family, under the screen that named the other three
------------------------------------------------------------
``AGENT_MEMORY`` 2i's rule is one sentence: a family is worth building only if it
reads bytes the deployed pipeline throws away. Three families have passed it --
backbone atom positions, side-chain atom positions, and the connectivity of the
empty space -- and all three read *coordinates*. This one reads none.

Every deposited ATOM record carries a temperature factor in columns 61-66, an
occupancy in 55-60, and an alternate-location indicator in column 17. The
pipeline reads the coordinates and the element. It has never read the B-factor
at all: ``pdb_io.parse_pdb_atoms`` does not extract that field. It reads
occupancy and altLoc only to *discard* alternates, keeping the highest-occupancy
copy and dropping the rest.

Measured over 200 training receptors: **every one** has a varying B-factor
column, and 24 % carry at least one alternate conformer, with a median of 1.6 %
of residues bearing one and a maximum of 38 %. So the primary input here is
universally available and universally unused, and the secondary one is sparse
but real.

Why it should matter physically, stated so it can be wrong
-----------------------------------------------------------
A cryptic site is defined by motion: it is a pocket that is absent in the apo
structure and present in the holo one, so the residues that form it are residues
whose apo positions are not the whole story. A temperature factor is the
refinement's own estimate of how badly one atom's position is determined, and an
alternate conformer is the crystallographer stating outright that one position
was not enough. Both are per-atom statements about local disorder, made by the
experiment, and this pipeline discards both.

That is an argument, not a result. It is written here before the measurement so
that it can be checked against the outcome rather than adjusted to it.

What is predicted, in advance
------------------------------
``AGENT_MEMORY`` 2m closed by committing that the next family would have its
expected overlap written down before its stack was run, because the void family
turned a sign-prediction into a magnitude-prediction and one instance is not a
law. So:

* **Expected raw lift: small, +0.001 to +0.003.** B-factors are strongly
  correlated with solvent exposure, and the deployed bank already reads burial
  through many wires. Much of what a B-factor says is therefore already present
  by another route, which is precisely the re-encoding failure mode 2i names.
* **Expected overlap with the ``geometry 528`` stack: high, 30-50 %,** against
  void's 12 % and the two conformation families' 21 % with each other. Void reads
  the space between atoms and keeps 88 % of its value when stacked; this family
  reads a scalar per atom that is largely a function of how exposed that atom is,
  and exposure is what the existing wires measure best.
* **The permutation arm should be negative** if the family is live at all, by the
  same argument that made it negative for the other three.
* **If the raw lift exceeds +0.004, the prediction was wrong in an interesting
  direction** and the reason to look at is group E: alternate conformers are not
  a function of exposure, and they are the one input here that is a direct
  statement about multiple occupancy rather than about uncertainty.

Design constraints that shaped the quantity list
-------------------------------------------------
**A B-factor is not comparable across structures.** It absorbs resolution, the
refinement protocol, the scaling model and whether TLS was used; a 30 Å² in a
1.2 Å structure and a 30 Å² in a 3.0 Å structure are different statements. So no
quantity here is a raw B in Å². Every one is a within-chain rank, a ratio between
two parts of the same residue, a difference against the same chain's median, or
an integer count. That also makes the three aggregations meaningful: summing a
rank over a contact shell is summing comparable things.

**Alternates are kept, not collapsed.** This module parses the receptor itself
rather than going through ``pdb_io``, for two reasons. That parser discards the
alternates this family is about, and ``AGENTS.md`` says not to modify a shared
tool for an unrelated reason -- the same judgement that put a retry-count fix in
a probe rather than in ``external_inventory.py``. Twenty lines of parsing here
cost less than a change with repository-wide blast radius.

**Nothing here is a function of residue type.** That is the test 2i imposes and
the test the chemistry family failed. A B-factor rank is a property of the
refinement of one residue, not of which residue it is.

The forty-eight quantities
---------------------------
A. within the residue (10) -- how mobility is distributed over one residue's own
   atoms, and whether its side chain moves more than its anchor.
B. against the chain (6) -- where the residue sits in its own structure's
   distribution, in ranks and robust scores.
C. along the sequence (7) -- runs, curvature and the length of the mobile stretch
   a residue belongs to. Integer where possible.
D. against the contact shell (9) -- rank among neighbours, local maxima, and the
   contrast between a residue and the residues touching it. These are here rather
   than left to the wire builder's shell aggregation because a rank within a
   shell is not a linear function of the shell's values.
E. alternate conformers (10) -- counts, occupancy spread, and how far apart the
   alternates actually are.
F. occupancy (6) -- partial occupancy without an alternate is a different
   statement from an alternate, and both are discarded today.

Nothing here reads a label, a ligand, or a holo structure.
"""
from __future__ import annotations

import numpy as np

# Column 61-66 is the temperature factor, 55-60 the occupancy, 17 the altLoc
# indicator, in the fixed-column PDB format. Slices are zero-based half-open.
B_COLS = slice(60, 66)
OCC_COLS = slice(54, 60)
ALTLOC_COL = 16

SKIP = frozenset({"HOH", "DOD", "WAT"})

# Integer scale for quantities reported as a fraction. permille rather than
# percent because a contact shell of thirty residues summing percents loses the
# resolution the quantiser then bands on.
PERMILLE = 1000
CENTI = 100

# Backbone atom names, for the side-chain/backbone split.
BACKBONE = frozenset({"N", "CA", "C", "O", "OXT"})

# Two residues are in contact when their heavy-atom centroids are within this,
# the same radius the other three families' wire builders use. Held identical so
# that a difference between families is not a difference in neighbourhood.
CONTACT_RADIUS = 8.0

# A run of "mobile" residues is a run above the chain's own median rank. The
# median rather than a fixed B because the whole point is that B does not
# transfer between structures.
MOBILE_RANK = 500  # permille

COLUMNS = (
    # A. within the residue
    "b_ca_permille", "b_backbone_permille", "b_sidechain_permille",
    "b_sc_over_bb_centi", "b_range_within_centi", "b_spread_within_centi",
    "b_terminal_permille", "b_gradient_centi", "b_max_permille",
    "b_min_permille",
    # B. against the chain
    "b_rank_permille", "b_robust_z_centi", "b_quartile", "b_above_median",
    "b_decile", "b_top_tenth",
    # C. along the sequence
    "b_prev_permille", "b_next_permille", "b_curvature_centi",
    "b_run_length", "b_run_position_permille", "b_seq_gradient_centi",
    "b_mobile_segments_near",
    # D. against the contact shell
    "b_shell_rank_permille", "b_minus_shell_centi", "b_shell_spread_centi",
    "b_is_shell_max", "b_shell_max_margin_centi", "b_shell_size",
    "b_shell_above_median", "b_shell_mobile_fraction_permille",
    "b_contrast_sign",
    # E. alternate conformers
    "alt_atoms", "alt_labels", "alt_backbone", "alt_sidechain",
    "alt_occupancy_spread_centi", "alt_max_displacement_centi",
    "alt_mean_displacement_centi", "alt_in_shell", "alt_shell_permille",
    "alt_cluster_size",
    # F. occupancy
    "occ_min_centi", "occ_mean_centi", "occ_partial_atoms",
    "occ_deficit_centi", "occ_partial_in_shell", "occ_is_full",
)

N_COLUMNS = len(COLUMNS)
IDX = {c: i for i, c in enumerate(COLUMNS)}

# Quantities that are properties of a residue's neighbourhood rather than of the
# residue, so two residues in one shell legitimately share them. Named because a
# test asserting every column varies residue by residue would be wrong about
# these, and weakening the test instead of naming them is how a real constant
# gets through.
SHELL_LEVEL = frozenset({
    "b_shell_spread_centi", "b_shell_size", "b_shell_above_median",
    "b_shell_mobile_fraction_permille", "alt_in_shell", "alt_shell_permille",
    "occ_partial_in_shell",
})


def parse_displacement(text: str, chain: str
                       ) -> dict[tuple[int, str], list[dict]]:
    """Every atom of one chain with its B, occupancy and altLoc, alternates kept.

    ``pdb_io.parse_pdb_atoms`` is not used, and not extended. It drops the
    B-factor entirely and collapses alternates to the highest-occupancy copy,
    which are the two inputs this family exists to read; extending it would put a
    change with repository-wide blast radius in the path of one new family, and
    ``AGENTS.md`` records why that trade is refused.

    Only the first model is read, which is the same convention ``pdb_io`` uses
    and matters for the NMR and ensemble deposits in this corpus.
    """
    out: dict[tuple[int, str], list[dict]] = {}
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) < 66 or line[21] != chain:
            continue
        element = line[76:78].strip().upper() or line[12:16].strip()[:1]
        if element == "H" or line[17:20].strip() in SKIP:
            continue
        try:
            b = float(line[B_COLS])
            occ = float(line[OCC_COLS])
            x, y, z = (float(line[30:38]), float(line[38:46]),
                       float(line[46:54]))
        except ValueError:
            continue
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        key = (resseq, line[26].strip())
        out.setdefault(key, []).append({
            "name": line[12:16].strip(),
            "altloc": line[ALTLOC_COL],
            "b": b,
            "occ": occ,
            "xyz": (x, y, z),
        })
    return out


def _ranks_permille(v: np.ndarray) -> np.ndarray:
    """Within-chain rank of each value, in permille, ties averaged.

    Rank rather than value because a B-factor does not transfer between
    structures: it absorbs resolution, the refinement protocol and the scaling
    model. A rank says "this residue is the mobile one in *this* structure",
    which is the only comparison the corpus supports.
    """
    n = len(v)
    if n == 0:
        return v
    if n == 1:
        return np.array([PERMILLE // 2], dtype=np.float64)
    order = np.argsort(v, kind="stable")
    r = np.empty(n, dtype=np.float64)
    r[order] = np.arange(n, dtype=np.float64)
    # Average the ranks of tied values, so a chain with a flat B column gives
    # every residue the same middle rank instead of an ordering by file order.
    uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, r)
    r = (sums / counts)[inv]
    return r * PERMILLE / max(n - 1, 1)


def _robust_z_centi(v: np.ndarray) -> np.ndarray:
    """(value - median) / MAD, in hundredths, clipped.

    The median and the median absolute deviation rather than mean and standard
    deviation, because a single very mobile loop moves a mean and a standard
    deviation far enough to flatten everything else. Clipped because the
    quantiser bands on within-chain quartiles anyway and an unbounded tail buys
    nothing but a wider float.
    """
    if len(v) == 0:
        return v
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad <= 0.0:
        return np.zeros(len(v))
    return np.clip((v - med) / mad, -8.0, 8.0) * CENTI


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Length of the True-run each position belongs to, and its offset in it."""
    n = len(mask)
    length = np.zeros(n, dtype=np.float64)
    offset = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        for k in range(i, j):
            length[k] = j - i
            offset[k] = (k - i) * PERMILLE / max(j - i - 1, 1)
        i = j
    return length, offset


def _centroids(residues: dict[tuple[int, str], list[dict]],
               order: list[tuple[int, str]]) -> np.ndarray:
    """Heavy-atom centroid per entry of ``order``, alternates weighted by occupancy.

    Derived here rather than taken as an argument. The obvious alternative was to
    accept the ``ctr`` the wire builders already compute, so that group D's
    contact graph would be identical to the graph the shell aggregation uses --
    but those arrays are keyed differently. ``ctr`` is indexed by *unique residue
    sequence number*, merging insertion-coded residues into one row, while
    ``order`` keys on ``(resseq, icode)``. On a chain carrying insertion codes the
    two lengths differ, which is a silent misalignment on every row after the
    first insertion rather than an error -- it raised here only because the
    shapes happened to be compared.

    So the radius and the centroid rule are held identical to the wire builders'
    and the indexing is made local. The graphs differ only in that an
    insertion-coded residue is its own node here.
    """
    out = np.zeros((len(order), 3), dtype=np.float64)
    for i, key in enumerate(order):
        atoms = residues.get(key, [])
        if not atoms:
            continue
        pts = np.array([a["xyz"] for a in atoms], dtype=np.float64)
        w = np.array([max(a["occ"], 1e-6) for a in atoms], dtype=np.float64)
        out[i] = np.average(pts, axis=0, weights=w)
    return out


def compute(residues: dict[tuple[int, str], list[dict]],
            order: list[tuple[int, str]]) -> np.ndarray:
    """The forty-eight quantities for one chain, as ``(n_residues, 48)``.

    One row per entry of ``order``, which keys on ``(resseq, icode)``. The caller
    subsets to the evaluation universe afterwards, exactly as the other three
    families' builders do.
    """
    n = len(order)
    out = np.zeros((n, N_COLUMNS), dtype=np.float64)
    if n == 0:
        return out
    j = IDX
    ctr = _centroids(residues, order)

    # --- per-residue aggregates over the atoms actually deposited -------------
    b_mean = np.zeros(n)
    b_ca = np.zeros(n)
    b_bb = np.zeros(n)
    b_sc = np.zeros(n)
    has_sc = np.zeros(n, dtype=bool)
    for i, key in enumerate(order):
        atoms = residues.get(key, [])
        if not atoms:
            continue
        # One value per atom name: alternates of the same atom are averaged by
        # occupancy, so a residue modelled in two conformations is not counted
        # as twice as many atoms.
        by_name: dict[str, list[dict]] = {}
        for a in atoms:
            by_name.setdefault(a["name"], []).append(a)
        bs, names = [], []
        for name, group in by_name.items():
            w = np.array([max(g["occ"], 1e-6) for g in group])
            bs.append(float(np.average([g["b"] for g in group], weights=w)))
            names.append(name)
        bs = np.array(bs)
        b_mean[i] = bs.mean()
        bb_mask = np.array([nm in BACKBONE for nm in names])
        if bb_mask.any():
            b_bb[i] = bs[bb_mask].mean()
        if (~bb_mask).any():
            b_sc[i] = bs[~bb_mask].mean()
            has_sc[i] = True
        for nm, bv in zip(names, bs):
            if nm == "CA":
                b_ca[i] = bv

        out[i, j["b_range_within_centi"]] = (bs.max() - bs.min()) * CENTI
        out[i, j["b_spread_within_centi"]] = float(bs.std()) * CENTI

        # Terminal atom: the deposited heavy atom furthest from CA along the
        # side chain. Its mobility relative to CA is the residue's own gradient,
        # which is a statement about the side chain and not about the fold.
        ca_xyz = next((a["xyz"] for a in atoms if a["name"] == "CA"), None)
        if ca_xyz is not None:
            sc = [a for a in atoms if a["name"] not in BACKBONE]
            if sc:
                d = [float(np.linalg.norm(np.array(a["xyz"])
                                          - np.array(ca_xyz))) for a in sc]
                term = sc[int(np.argmax(d))]
                out[i, j["b_gradient_centi"]] = (term["b"] - b_ca[i]) * CENTI

    b_max_atom = np.array([max((a["b"] for a in residues.get(k, [])), default=0.0)
                           for k in order])
    b_min_atom = np.array([min((a["b"] for a in residues.get(k, [])), default=0.0)
                           for k in order])
    term_b = np.array([
        max((a["b"] for a in residues.get(k, []) if a["name"] not in BACKBONE),
            default=0.0) for k in order])

    # --- A / B. ranks against this chain -------------------------------------
    out[:, j["b_ca_permille"]] = _ranks_permille(b_ca)
    out[:, j["b_backbone_permille"]] = _ranks_permille(b_bb)
    out[:, j["b_sidechain_permille"]] = _ranks_permille(b_sc)
    out[:, j["b_terminal_permille"]] = _ranks_permille(term_b)
    out[:, j["b_max_permille"]] = _ranks_permille(b_max_atom)
    out[:, j["b_min_permille"]] = _ranks_permille(b_min_atom)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((b_bb > 0) & has_sc, b_sc / np.maximum(b_bb, 1e-9), 1.0)
    out[:, j["b_sc_over_bb_centi"]] = np.clip(ratio, 0.0, 8.0) * CENTI

    rank = _ranks_permille(b_mean)
    out[:, j["b_rank_permille"]] = rank
    out[:, j["b_robust_z_centi"]] = _robust_z_centi(b_mean)
    out[:, j["b_quartile"]] = np.minimum((rank * 4) // (PERMILLE + 1), 3)
    out[:, j["b_decile"]] = np.minimum((rank * 10) // (PERMILLE + 1), 9)
    mobile = rank >= MOBILE_RANK
    out[:, j["b_above_median"]] = mobile.astype(np.float64)
    out[:, j["b_top_tenth"]] = (rank >= 900).astype(np.float64)

    # --- C. along the sequence ------------------------------------------------
    resseq = np.array([k[0] for k in order], dtype=np.int64)
    adjacent = np.zeros(max(n - 1, 0), dtype=bool)
    if n > 1:
        adjacent = np.diff(resseq) == 1
    prev_r = np.full(n, PERMILLE / 2.0)
    next_r = np.full(n, PERMILLE / 2.0)
    if n > 1:
        prev_r[1:] = np.where(adjacent, rank[:-1], PERMILLE / 2.0)
        next_r[:-1] = np.where(adjacent, rank[1:], PERMILLE / 2.0)
    out[:, j["b_prev_permille"]] = prev_r
    out[:, j["b_next_permille"]] = next_r
    # Second difference along the chain: positive where a residue is a local
    # minimum of mobility between two more mobile neighbours, negative at a
    # local peak. A hinge and a rigid anchor differ in this sign.
    out[:, j["b_curvature_centi"]] = (prev_r - 2.0 * rank + next_r) * CENTI \
        / PERMILLE
    out[:, j["b_seq_gradient_centi"]] = (next_r - prev_r) * CENTI / PERMILLE
    run_len, run_pos = _runs(mobile)
    out[:, j["b_run_length"]] = run_len
    out[:, j["b_run_position_permille"]] = run_pos
    # How many separate mobile stretches lie within five residues in sequence.
    # One long loop and three short ones are different local situations that a
    # run length alone reports identically.
    near = np.zeros(n)
    for i in range(n):
        lo, hi = max(i - 5, 0), min(i + 6, n)
        seg = mobile[lo:hi]
        near[i] = float(np.sum(seg[1:] & ~seg[:-1]) + (1 if seg[0] else 0))
    out[:, j["b_mobile_segments_near"]] = near

    # --- D. against the contact shell ----------------------------------------
    d = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = d <= CONTACT_RADIUS
    shell_n = adj.sum(axis=1)
    out[:, j["b_shell_size"]] = shell_n.astype(np.float64)
    for i in range(n):
        nb = np.flatnonzero(adj[i])
        if len(nb) == 0:
            out[i, j["b_shell_rank_permille"]] = PERMILLE / 2.0
            out[i, j["b_contrast_sign"]] = 0.0
            continue
        vals = rank[nb]
        out[i, j["b_shell_rank_permille"]] = (
            float(np.sum(vals < rank[i]) + 0.5 * np.sum(vals == rank[i]))
            * PERMILLE / len(vals))
        out[i, j["b_minus_shell_centi"]] = (rank[i] - vals.mean()) * CENTI \
            / PERMILLE
        out[i, j["b_shell_spread_centi"]] = float(vals.std()) * CENTI / PERMILLE
        out[i, j["b_is_shell_max"]] = float(rank[i] >= vals.max())
        out[i, j["b_shell_max_margin_centi"]] = (rank[i] - vals.max()) * CENTI \
            / PERMILLE
        out[i, j["b_shell_above_median"]] = float(np.sum(mobile[nb]))
        out[i, j["b_shell_mobile_fraction_permille"]] = (
            float(np.mean(mobile[nb])) * PERMILLE)
        out[i, j["b_contrast_sign"]] = float(np.sign(rank[i] - vals.mean()))

    # --- E. alternate conformers ---------------------------------------------
    has_alt = np.zeros(n, dtype=bool)
    for i, key in enumerate(order):
        atoms = residues.get(key, [])
        alts = [a for a in atoms if a["altloc"] != " "]
        if not alts:
            out[i, j["occ_is_full"]] = 1.0 if atoms else 0.0
            continue
        has_alt[i] = True
        out[i, j["alt_atoms"]] = float(len(alts))
        labels = {a["altloc"] for a in alts}
        out[i, j["alt_labels"]] = float(len(labels))
        out[i, j["alt_backbone"]] = float(
            sum(1 for a in alts if a["name"] in BACKBONE))
        out[i, j["alt_sidechain"]] = float(
            sum(1 for a in alts if a["name"] not in BACKBONE))
        occs = np.array([a["occ"] for a in alts])
        out[i, j["alt_occupancy_spread_centi"]] = (
            float(occs.max() - occs.min()) * CENTI)
        # How far apart the alternates actually sit. A residue modelled in two
        # conformations 0.2 A apart is a refinement detail; one modelled in two
        # conformations 3 A apart is a statement that the side chain occupies
        # two distinct positions, which is what a cryptic site is made of.
        by_name: dict[str, list[dict]] = {}
        for a in alts:
            by_name.setdefault(a["name"], []).append(a)
        gaps = []
        for group in by_name.values():
            if len(group) < 2:
                continue
            pts = np.array([g["xyz"] for g in group])
            dd = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
            gaps.append(float(dd.max()))
        if gaps:
            out[i, j["alt_max_displacement_centi"]] = max(gaps) * CENTI
            out[i, j["alt_mean_displacement_centi"]] = float(
                np.mean(gaps)) * CENTI

    for i in range(n):
        nb = np.flatnonzero(adj[i])
        out[i, j["alt_in_shell"]] = float(np.sum(has_alt[nb])) if len(nb) else 0.0
        out[i, j["alt_shell_permille"]] = (
            float(np.mean(has_alt[nb])) * PERMILLE if len(nb) else 0.0)

    # Connected components of alternate-bearing residues under the contact
    # graph. One disordered side chain and a patch of six touching disordered
    # side chains are different objects, and a per-residue count cannot tell
    # them apart.
    comp = np.zeros(n, dtype=np.int64) - 1
    size: dict[int, int] = {}
    c = 0
    for i in range(n):
        if not has_alt[i] or comp[i] >= 0:
            continue
        stack, members = [i], []
        comp[i] = c
        while stack:
            k = stack.pop()
            members.append(k)
            for m in np.flatnonzero(adj[k]):
                if has_alt[m] and comp[m] < 0:
                    comp[m] = c
                    stack.append(m)
        size[c] = len(members)
        c += 1
    for i in range(n):
        if comp[i] >= 0:
            out[i, j["alt_cluster_size"]] = float(size[comp[i]])

    # --- F. occupancy ---------------------------------------------------------
    partial = np.zeros(n, dtype=bool)
    for i, key in enumerate(order):
        atoms = residues.get(key, [])
        if not atoms:
            out[i, j["occ_min_centi"]] = CENTI
            out[i, j["occ_mean_centi"]] = CENTI
            continue
        # Alternates of one atom share an occupancy budget, so their occupancies
        # are summed per atom name before being read as "is this atom fully
        # occupied". Without that every alternate looks like partial occupancy
        # and group F becomes a copy of group E.
        by_name: dict[str, float] = {}
        for a in atoms:
            by_name[a["name"]] = by_name.get(a["name"], 0.0) + a["occ"]
        occ = np.array(list(by_name.values()))
        out[i, j["occ_min_centi"]] = float(occ.min()) * CENTI
        out[i, j["occ_mean_centi"]] = float(occ.mean()) * CENTI
        np_partial = int(np.sum(occ < 0.999))
        out[i, j["occ_partial_atoms"]] = float(np_partial)
        out[i, j["occ_deficit_centi"]] = float(np.sum(
            np.clip(1.0 - occ, 0.0, None))) * CENTI
        out[i, j["occ_is_full"]] = float(np_partial == 0)
        partial[i] = np_partial > 0
    for i in range(n):
        nb = np.flatnonzero(adj[i])
        out[i, j["occ_partial_in_shell"]] = (
            float(np.sum(partial[nb])) if len(nb) else 0.0)

    return out


def consistency(x: np.ndarray, order: list[tuple[int, str]]) -> list[str]:
    """Invariants that must hold on any chain, returned as complaints.

    Written as a list of statements rather than assertions so a caller can report
    all of them at once. Each one is a property that would have to break for a
    number in this family to mean something other than what its name says.
    """
    bad: list[str] = []
    if x.shape[1] != N_COLUMNS:
        bad.append(f"{x.shape[1]} columns, expected {N_COLUMNS}")
        return bad
    if not np.isfinite(x).all():
        bad.append("non-finite value")
    j = IDX

    for name in ("b_ca_permille", "b_backbone_permille", "b_sidechain_permille",
                 "b_rank_permille", "b_terminal_permille", "b_max_permille",
                 "b_min_permille", "b_shell_rank_permille",
                 "b_run_position_permille", "b_alt_shell" if False else
                 "alt_shell_permille", "b_shell_mobile_fraction_permille"):
        v = x[:, j[name]]
        if v.size and (v.min() < -1e-9 or v.max() > PERMILLE + 1e-6):
            bad.append(f"{name} outside [0, {PERMILLE}]: "
                       f"{v.min():.3f}..{v.max():.3f}")

    for name in ("b_above_median", "b_top_tenth", "b_is_shell_max",
                 "occ_is_full"):
        v = x[:, j[name]]
        if v.size and not np.isin(v, (0.0, 1.0)).all():
            bad.append(f"{name} is not an indicator")

    q = x[:, j["b_quartile"]]
    if q.size and (q.min() < 0 or q.max() > 3):
        bad.append("b_quartile outside 0..3")
    dec = x[:, j["b_decile"]]
    if dec.size and (dec.min() < 0 or dec.max() > 9):
        bad.append("b_decile outside 0..9")

    for name in ("alt_atoms", "alt_labels", "alt_backbone", "alt_sidechain",
                 "alt_in_shell", "alt_cluster_size", "occ_partial_atoms",
                 "occ_partial_in_shell", "b_run_length", "b_shell_size",
                 "b_shell_above_median", "b_mobile_segments_near"):
        v = x[:, j[name]]
        if v.size and (v.min() < -1e-9 or np.abs(v - np.round(v)).max() > 1e-9):
            bad.append(f"{name} is not a non-negative integer count")

    # A residue with no alternate cannot carry an alternate displacement, and a
    # residue with one must have at least two atoms sharing a name for the
    # displacement to be defined -- so the implication runs one way only.
    no_alt = x[:, j["alt_atoms"]] == 0
    if no_alt.any():
        for name in ("alt_labels", "alt_backbone", "alt_sidechain",
                     "alt_max_displacement_centi", "alt_occupancy_spread_centi"):
            if np.abs(x[no_alt, j[name]]).max(initial=0.0) > 1e-9:
                bad.append(f"{name} is non-zero on a residue with no alternate")

    if (x[:, j["alt_backbone"]] + x[:, j["alt_sidechain"]]
            != x[:, j["alt_atoms"]]).any():
        bad.append("alt_backbone + alt_sidechain != alt_atoms")

    if (x[:, j["alt_cluster_size"]] > 0).any():
        if x[(x[:, j["alt_cluster_size"]] > 0), j["alt_atoms"]].min() <= 0:
            bad.append("a residue is in an alternate cluster without an "
                       "alternate of its own")

    full = x[:, j["occ_is_full"]] == 1.0
    if full.any() and x[full, j["occ_partial_atoms"]].max(initial=0.0) > 0:
        bad.append("occ_is_full set on a residue with a partial-occupancy atom")

    if len(order) != x.shape[0]:
        bad.append(f"{x.shape[0]} rows for {len(order)} residues")
    return bad
