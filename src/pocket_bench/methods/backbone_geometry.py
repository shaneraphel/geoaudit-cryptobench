"""Backbone geometry: thirteen quantities the centroid representation destroys.

Why this module exists
----------------------
Every local quantity the deployed detector reads is a function of residue
centroids and the contact graph they induce. That is stated in the wire
appendix's own conventions: ``c_i`` is the centroid of the heavy atoms of residue
``i``, and neighbourhoods are the residues whose centroid lies within 10 A. Six
wire families have been added on top of that and every one measured null, and
they have one thing in common -- each is a function of data the pipeline already
reads, so each is a re-encoding.

Reducing a residue to one point destroys the backbone. Two different backbone
conformations can present the same set of centroids, because recovering a torsion
needs the positions of N and C, and those are discarded at parse time. So nothing
here is recoverable from the deployed bank, and that is the whole point: this is
the first family that is not a re-encoding.

It also has a mechanism rather than a hope. Cryptic pockets open by backbone
motion; the residues that line them sit disproportionately in loops and at the
ends of helices; and the most reproducible probing result about protein language
models is that their representations encode secondary structure strongly. If the
deficit against pLM-NN has a structural explanation, backbone conformation is
where it should live, and if it does not, this family should be null like the
other six and the deficit is evolutionary rather than geometric.

The quantities, and why each is its own thing
---------------------------------------------
A family whose members are near-duplicates cannot be told apart from a family
that is simply larger, so these are chosen to be different kinds of measurement
rather than forty-four versions of one. Forty-four is the count after the
expansion; the first thirteen are listed here and are the block measured in
``BACKBONE_WIRES_LIFT.json``, and the six groups after them are described where
they are computed, in ``_expand``. The expansion is strictly additive: the first
thirteen columns are bit-identical before and after it, so the earlier
measurement remains a statement about a subset of this family and not about a
different one.

``rama_region``       a combinatorial label. The Ramachandran torus is partitioned
                      into four named cells -- right-handed helical, extended,
                      left-handed helical, and everything else -- and the residue
                      is assigned the cell it lands in. No fitting: the cell
                      boundaries are the conventional ones and are constants here.
``cos_phi, sin_phi``  the C(i-1)-N-CA-C torsion, as a point on the circle.
``cos_psi, sin_psi``  the N-CA-C-N(i+1) torsion, likewise.
``ca_turn``           the angle at CA(i) subtended by CA(i-1) and CA(i+1). About
                      89 degrees through a helix and 120-130 through a strand,
                      and it needs no torsion, so it survives a missing N or C.
``cos_ca_tor``        the CA(i-1..i+2) dihedral, as a point on the circle. About
``sin_ca_tor``        +50 degrees in a helix and near 180 in a strand, and it is
                      the one quantity here that distinguishes a right-handed
                      helix from its mirror image.
``hb_donated``        whether this residue's amide nitrogen donates a backbone
                      hydrogen bond.
``hb_accepted``       how many backbone donors this residue's carbonyl accepts.
                      Not the same quantity as the one above and not derivable
                      from it: a carbonyl can accept twice and a nitrogen donates
                      at most once.
``hb_lag``            the sequence separation to the donated bond's acceptor, 0
                      when none. About 4 through a helix, large through a sheet,
                      and this single integer is what separates the two forms of
                      regular structure that the turn angle alone confuses.
``cb_radial``         the cosine between CA->CB and the outward radial direction.
                      Whether the side chain points into the protein or out of
                      it, which is a property of the backbone's orientation and
                      not of the side chain's identity.
``ca_density``        CA atoms within 8 A. Backbone packing with the side chains
                      removed, which is a different quantity from the centroid
                      coordination the bank already has: a chain of glycines and
                      a chain of tryptophans with the same fold agree here and
                      disagree there.

Why angles are emitted as cosine and sine rather than as degrees
----------------------------------------------------------------
The detector quantises every wire by its rank within the chain. A rank order is
meaningless on a circle: -179 and +179 degrees are a two-degree apart
conformation and sit at opposite ends of any ranking, so a banded torsion angle
would place the two halves of the extended region in different cells and call
them maximally different. Cosine and sine are genuine functions on the circle,
they are continuous across the wrap, and together they determine the angle. The
cost is two wires per torsion instead of one, which is the honest price of not
lying about the topology.

Boundaries and breaks
---------------------
A torsion needs residues on both sides, and a crystallographic chain is not
always continuous. Consecutive residues whose C(i)-N(i+1) distance exceeds
``PEPTIDE_BOND_MAX`` are not bonded, and every quantity spanning that gap is
undefined. Undefined takes the neutral value -- zero for a cosine, a sine, a
count or a lag, and the ``other`` cell for the region label -- following the
appendix's convention that a boundary is neutral in the rank order rather than
extreme in it.

Hydrogens are absent from these files, so the hydrogen bond is decided
geometrically without them: the nitrogen and the acceptor oxygen within
``HBOND_MAX_NO``, the carbonyl pointing at the nitrogen rather than away, and a
sequence separation of at least two so that a residue's own and its neighbour's
carbonyl cannot count. This is a coarser criterion than an energy with an
inferred hydrogen position, and it is deliberately coarse: the quantity that
matters downstream is a count and a lag, both of which survive a criterion that
is a little loose, while an inferred hydrogen would put a modelling decision
inside a wire.
"""
from __future__ import annotations

import numpy as np

# The conventional partition of the Ramachandran torus. Boundaries are constants
# of the field, not thresholds fitted here; they are wide because the purpose is
# to name a basin, not to score a structure.
RAMA_OTHER, RAMA_ALPHA_R, RAMA_BETA, RAMA_ALPHA_L = 0, 1, 2, 3
RAMA_CELLS = {
    RAMA_OTHER: "other, or undefined at a chain end or break",
    RAMA_ALPHA_R: "right-handed helical",
    RAMA_BETA: "extended",
    RAMA_ALPHA_L: "left-handed helical",
}

PEPTIDE_BOND_MAX = 2.0      # angstrom, C(i) to N(i+1); a real bond is near 1.33
HBOND_MAX_NO = 3.5          # angstrom, N to O
HBOND_MIN_COS = 0.34        # about 70 degrees off the C=O axis, the widest cone
                            # that still calls the nitrogen "beyond the oxygen"
HBOND_MIN_LAG = 2           # residues; excludes a residue's own and adjacent C=O
HBOND_LAG_CAP = 30          # beyond this the lag is "far" and the value saturates
CA_DENSITY_RADIUS = 8.0     # angstrom
CB_BOND = 1.53              # angstrom, CA to CB

RUN_CAP = 12                # residues; a helix longer than this is "long"
WINDOW = 4                  # residues each side, one alpha turn
SEQ_SPAN_CAP = 60           # residues; beyond this a contact is simply "distant"
DIST_CAP = 8.0              # angstrom; the ceiling for a nearest-partner distance
CA_RADII = (6.0, 8.0, 12.0)
LONG_RANGE_LAG = 8          # |i-j| above which a contact is tertiary, not local

# The first thirteen are the family measured in BACKBONE_WIRES_LIFT.json and
# their order is fixed so that a rebuild reproduces that column block exactly.
# Everything after them is the expansion, and it is grouped by *kind* of
# measurement rather than by convenience: a family whose members are near
# duplicates cannot be told apart from a family that is merely larger, which is
# the rule five null families produced.
COLUMNS = (
    # torsion and conformation
    "rama_region",
    "cos_phi", "sin_phi",
    "cos_psi", "sin_psi",
    "ca_turn",
    "cos_ca_tor", "sin_ca_tor",
    "hb_donated", "hb_accepted", "hb_lag",
    "cb_radial",
    "ca_density",
    # A. the conformation of the neighbourhood in sequence, not in space
    "cos_omega", "sin_omega",
    "cos_phi_plus_psi", "sin_phi_plus_psi",
    "cos_phi_minus_psi", "sin_phi_minus_psi",
    "rama_prev", "rama_next",
    "rama_run", "rama_variety", "dist_to_cell_change",
    # B. discrete differential geometry of the CA trace
    "ca_span5", "ca_span7", "ca_tetra_volume", "ca_edge",
    # C. hydrogen bonding: saturation, direction, and what is unsatisfied
    "hb_lag_signed", "hb_accept_lag", "hb_window_count",
    "n_nearest_acceptor", "o_nearest_donor", "hb_carbonyls_near_n",
    # D. where the side chain points, without asking which side chain it is
    "cb_tangent", "cb_binormal", "cb_density", "cb_hemisphere",
    # E. packing split into the local and the tertiary
    "ca_density_6", "ca_density_12", "ca_longrange", "ca_seq_span",
    # F. the peptide plane
    "plane_radial", "plane_twist",
)
N_COLUMNS = len(COLUMNS)
N_ORIGINAL_COLUMNS = 13

BACKBONE_ATOMS = ("N", "CA", "C", "O")


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
              p3: np.ndarray) -> np.ndarray:
    """Signed dihedral of four point sequences, returned as (cos, sin).

    The angle is never formed, because forming it and taking its cosine again
    would round-trip through a discontinuity for nothing. b1 x b2 and b2 x b3 are
    the normals of the two planes; their dot gives the cosine and the triple
    product with the normalised central bond gives the sine, which is what
    carries the sign and therefore the handedness.
    """
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    b2n = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-9)
    m1 = np.cross(n1, b2n)
    x = (n1 * n2).sum(-1)
    # The sine is negated because this construction measures the rotation of the
    # second plane onto the first, and IUPAC fixes the sign the other way round.
    # Getting this backwards is not a cosmetic error: it exchanges the
    # right-handed and left-handed helical cells of the Ramachandran partition,
    # and the first run of this file put 127 of one chain's 254 residues in the
    # left-handed cell, which is the sign that catches it.
    y = -(m1 * n2).sum(-1)
    r = np.maximum(np.hypot(x, y), 1e-12)
    return x / r, y / r


def _virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Where CB would sit, from N, CA and C alone.

    Glycine has no CB and a sentinel would make every glycine extreme in the rank
    order of ``cb_radial`` rather than neutral, which is the opposite of what the
    appendix's boundary convention asks for. The tetrahedral construction places
    it exactly: the bisector of the N and C directions gives the plane, the
    normal gives the out-of-plane component, and the standard L-amino-acid
    geometry fixes the mixture. This is also applied to residues whose CB is
    simply missing from the deposit, so one rule covers both cases.
    """
    b = ca - n
    cc = c - ca
    a = np.cross(b, cc)
    return (-0.58273431 * a + 0.56802827 * b - 0.54067466 * cc) + ca


def chain_backbone(atoms: list[dict], resseq_order: list[tuple[int, str]]
                   ) -> dict[str, np.ndarray]:
    """Backbone atom positions per residue, in the order the caller scores them.

    Missing atoms are returned as NaN so that every downstream quantity can say
    "undefined here" rather than silently using a wrong coordinate. The residue
    order is supplied rather than inferred, because the evaluation universe is
    the sorted set of integer resseq and file order is not always that.
    """
    want = {k: i for i, k in enumerate(resseq_order)}
    n = len(resseq_order)
    out = {a: np.full((n, 3), np.nan) for a in BACKBONE_ATOMS + ("CB",)}
    for at in atoms:
        if at["record"] != "ATOM":
            continue
        name = at["name"]
        if name not in out:
            continue
        i = want.get((at["resseq"], at["icode"].strip()))
        if i is None:
            continue
        out[name][i] = (at["x"], at["y"], at["z"])
    return out


def _bonded(c: np.ndarray, nxt_n: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(nxt_n - c, axis=-1)
    return np.isfinite(d) & (d <= PEPTIDE_BOND_MAX)


def _segments(link: np.ndarray, n: int) -> np.ndarray:
    """A segment label per residue; a chain break starts a new one.

    Every sequence-local quantity below is computed inside a segment. Reading a
    previous residue's conformation across a break would describe two pieces of
    protein that are not joined, and on a chain with three gaps that is three
    fabricated neighbours rather than a rounding error.
    """
    seg = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        seg[i] = seg[i - 1] + (0 if (i - 1 < len(link) and link[i - 1]) else 1)
    return seg


def _shift(v: np.ndarray, k: int, seg: np.ndarray, fill: float) -> np.ndarray:
    """``v`` moved ``k`` places along the chain, neutral across a break."""
    n = len(v)
    out = np.full(n, fill, dtype=np.float64)
    idx = np.arange(n)
    src = idx - k
    ok = (src >= 0) & (src < n)
    ok[ok] &= seg[src[ok]] == seg[idx[ok]]
    out[ok] = v[src[ok]]
    return out


def _expand(out: np.ndarray, col: dict, N, CA, C, O, cb, link) -> None:
    """The thirty-one quantities beyond the first thirteen.

    Grouped by kind rather than by convenience, because a family of near
    duplicates is indistinguishable from a family that is merely bigger. Each
    group answers a different question about the same backbone:

    A. what conformation do the residues *next to this one in sequence* have,
       which is what turns a residue in helical conformation into a helix;
    B. what does the CA trace do over four and six steps, which is the classic
       discriminator between a helix (span5 near 6.2 A) and a strand (near 13);
    C. which backbone polar groups are satisfied and which are not, since an
       unsatisfied buried donor or carbonyl is a position that has nothing to
       hydrogen bond to and is a recognised signature of a site that opens;
    D. where the side chain points, asked of the backbone rather than of the
       residue's identity, which AGENT_MEMORY 2i closed as a source of anything
       new;
    E. packing split into the local and the tertiary, because a residue packed
       by its own helix and one packed by a distant strand have the same
       neighbour count and are not in the same situation;
    F. the peptide plane, whose orientation is the one backbone degree of
       freedom that neither torsion captures.
    """
    n = len(CA)
    seg = _segments(link, n)
    fin = lambda a: np.isfinite(a).all(1)  # noqa: E731

    # --- A. conformation in the sequence neighbourhood -----------------------
    # omega is the torsion of the peptide bond *preceding* this residue, the
    # same convention phi follows, so a cis-proline shows on the residue whose
    # bond is cis rather than on its predecessor.
    if n > 1:
        ok = link & fin(CA[:-1]) & fin(C[:-1]) & fin(N[1:]) & fin(CA[1:])
        i = np.flatnonzero(ok)
        if len(i):
            cs, sn = _dihedral(CA[i], C[i], N[i + 1], CA[i + 1])
            out[i + 1, col["cos_omega"]] = cs
            out[i + 1, col["sin_omega"]] = sn

    # The torus has two natural diagonals. phi+psi separates the extended region
    # from the helical one along a different axis from either angle alone, and
    # under a rank quantiser that is a different partition of the same points,
    # not a relabelling of it.
    cph, sph = out[:, col["cos_phi"]], out[:, col["sin_phi"]]
    cps, sps = out[:, col["cos_psi"]], out[:, col["sin_psi"]]
    live = (np.hypot(cph, sph) > 0.5) & (np.hypot(cps, sps) > 0.5)
    out[live, col["cos_phi_plus_psi"]] = (cph * cps - sph * sps)[live]
    out[live, col["sin_phi_plus_psi"]] = (sph * cps + cph * sps)[live]
    out[live, col["cos_phi_minus_psi"]] = (cph * cps + sph * sps)[live]
    out[live, col["sin_phi_minus_psi"]] = (sph * cps - cph * sps)[live]

    cell = out[:, col["rama_region"]]
    out[:, col["rama_prev"]] = _shift(cell, 1, seg, RAMA_OTHER)
    out[:, col["rama_next"]] = _shift(cell, -1, seg, RAMA_OTHER)

    # How long the run of this conformation is, and how far the nearest change
    # sits. A helix is a run; a single residue in helical conformation inside a
    # loop is not, and no per-residue torsion can tell them apart.
    run = np.ones(n, dtype=np.float64)
    dist = np.full(n, float(RUN_CAP), dtype=np.float64)
    for i in range(n):
        same = 1
        for d in (1, -1):
            j = i + d
            while (0 <= j < n and seg[j] == seg[i] and cell[j] == cell[i]
                   and same < RUN_CAP):
                same += 1
                j += d
            k = i + d
            step = 1
            while 0 <= k < n and seg[k] == seg[i] and step <= RUN_CAP:
                if cell[k] != cell[i]:
                    dist[i] = min(dist[i], step)
                    break
                k += d
                step += 1
        run[i] = min(same, RUN_CAP)
    out[:, col["rama_run"]] = run
    out[:, col["dist_to_cell_change"]] = dist

    lo = np.maximum(np.arange(n) - 2, 0)
    hi = np.minimum(np.arange(n) + 3, n)
    out[:, col["rama_variety"]] = [
        len({cell[k] for k in range(lo[i], hi[i]) if seg[k] == seg[i]})
        for i in range(n)]

    # --- B. the CA trace over four and six steps ----------------------------
    okca = fin(CA)
    for k, name in ((2, "ca_span5"), (3, "ca_span7")):
        if n > 2 * k:
            i = np.arange(k, n - k)
            good = okca[i - k] & okca[i + k] & (seg[i - k] == seg[i + k])
            j = i[good]
            if len(j):
                out[j, col[name]] = np.linalg.norm(CA[j + k] - CA[j - k],
                                                   axis=1)
    if n > 4:
        i = np.arange(2, n - 2)
        good = (okca[i - 2] & okca[i - 1] & okca[i + 1] & okca[i + 2]
                & (seg[i - 2] == seg[i + 2]))
        j = i[good]
        if len(j):
            a, b, c = CA[j - 1] - CA[j - 2], CA[j + 1] - CA[j - 2], \
                CA[j + 2] - CA[j - 2]
            out[j, col["ca_tetra_volume"]] = (
                np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0)
    if n > 1:
        i = np.arange(n - 1)
        good = okca[i] & okca[i + 1] & (seg[i] == seg[i + 1])
        j = i[good]
        if len(j):
            out[j, col["ca_edge"]] = np.linalg.norm(CA[j + 1] - CA[j], axis=1)

    # --- C. hydrogen bonding, satisfaction and its absence ------------------
    okN, okO = fin(N), fin(O) & fin(C)
    idxN, idxO = np.flatnonzero(okN), np.flatnonzero(okO)
    if len(idxN) and len(idxO):
        d = np.linalg.norm(N[idxN][:, None, :] - O[idxO][None, :, :], axis=-1)
        lag = np.abs(idxN[:, None] - idxO[None, :])
        far = lag >= HBOND_MIN_LAG
        # The nearest possible partner regardless of geometry. This is not one
        # minus "did it bond": a nitrogen with an acceptor at 3.6 A and one with
        # nothing inside 8 A both fail the bond test and are not in the same
        # situation, and the second is the buried unsatisfied donor that marks a
        # position with nothing to hydrogen bond to.
        dn = np.where(far, d, np.inf)
        out[idxN, col["n_nearest_acceptor"]] = np.minimum(
            dn.min(axis=1), DIST_CAP)
        out[idxO, col["o_nearest_donor"]] = np.minimum(
            dn.min(axis=0), DIST_CAP)
        out[idxN, col["hb_carbonyls_near_n"]] = (
            (dn <= HBOND_MAX_NO).sum(axis=1).astype(np.float64))

        co = O[idxO] - C[idxO]
        con = co / np.maximum(np.linalg.norm(co, axis=1, keepdims=True), 1e-9)
        to_n = N[idxN][:, None, :] - O[idxO][None, :, :]
        to_n = to_n / np.maximum(np.linalg.norm(to_n, axis=-1, keepdims=True),
                                 1e-9)
        ok = ((d <= HBOND_MAX_NO) & far
              & ((to_n * con[None, :, :]).sum(-1) > HBOND_MIN_COS))
        if ok.any():
            dd = np.where(ok, d, np.inf)
            best = dd.argmin(axis=1)
            has = np.isfinite(dd[np.arange(len(idxN)), best])
            rows = idxN[has]
            signed = (idxO[best] - idxN)[has]
            out[rows, col["hb_lag_signed"]] = np.clip(
                signed, -HBOND_LAG_CAP, HBOND_LAG_CAP)
            # The acceptor's own view: how far away, in sequence, is the donor
            # it accepts from. A carbonyl accepting at lag 4 is inside a helix;
            # one accepting at lag 30 is holding a sheet together.
            da = np.where(ok, d, np.inf)
            bestd = da.argmin(axis=0)
            hasd = np.isfinite(da[bestd, np.arange(len(idxO))])
            out[idxO[hasd], col["hb_accept_lag"]] = np.minimum(
                np.abs(idxO - idxN[bestd])[hasd], HBOND_LAG_CAP)

    donated = out[:, col["hb_donated"]]
    accepted = np.minimum(out[:, col["hb_accepted"]], 4.0)
    tot = donated + accepted
    win = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a = max(0, i - WINDOW)
        b = min(n, i + WINDOW + 1)
        m = seg[a:b] == seg[i]
        win[i] = tot[a:b][m].sum()
    out[:, col["hb_window_count"]] = win

    # --- D. where the side chain points -------------------------------------
    okcb = fin(cb) & okca
    if okcb.any():
        v = np.zeros((n, 3))
        v[okcb] = cb[okcb] - CA[okcb]
        vn = np.linalg.norm(v, axis=1)
        vhat = np.zeros_like(v)
        nz = vn > 1e-9
        vhat[nz] = v[nz] / vn[nz, None]
        tang = np.zeros((n, 3))
        if n > 2:
            i = np.arange(1, n - 1)
            good = okca[i - 1] & okca[i + 1] & (seg[i - 1] == seg[i + 1])
            j = i[good]
            tang[j] = CA[j + 1] - CA[j - 1]
        tn = np.linalg.norm(tang, axis=1)
        that = np.zeros_like(tang)
        nz2 = tn > 1e-9
        that[nz2] = tang[nz2] / tn[nz2, None]
        out[:, col["cb_tangent"]] = np.clip((vhat * that).sum(1), -1.0, 1.0)
        binorm = np.zeros((n, 3))
        if n > 2:
            i = np.arange(1, n - 1)
            good = okca[i - 1] & okca[i + 1] & (seg[i - 1] == seg[i + 1])
            j = i[good]
            binorm[j] = np.cross(CA[j] - CA[j - 1], CA[j + 1] - CA[j])
        bn = np.linalg.norm(binorm, axis=1)
        bhat = np.zeros_like(binorm)
        nz3 = bn > 1e-9
        bhat[nz3] = binorm[nz3] / bn[nz3, None]
        out[:, col["cb_binormal"]] = np.clip((vhat * bhat).sum(1), -1.0, 1.0)

        idx = np.flatnonzero(okcb)
        P = cb[idx]
        for s in range(0, len(idx), 512):
            e = min(s + 512, len(idx))
            dv = P[None, :, :] - P[s:e, None, :]
            d2 = (dv ** 2).sum(-1)
            out[idx[s:e], col["cb_density"]] = (d2 <= 64.0).sum(1) - 1
            # Of the side chains within 10 A, how many lie in the half space
            # this one points into. A residue whose side chain points into a
            # crowd is in a different situation from one pointing out of it,
            # and the two have the same neighbour count.
            near = d2 <= 100.0
            proj = np.einsum("ijk,ik->ij", dv, vhat[idx[s:e]])
            out[idx[s:e], col["cb_hemisphere"]] = (near & (proj > 0)).sum(1)

    # --- E. packing, local against tertiary ---------------------------------
    idx = np.flatnonzero(okca)
    if len(idx) > 1:
        P = CA[idx]
        lagm = np.abs(idx[:, None] - idx[None, :])
        for s in range(0, len(idx), 512):
            e = min(s + 512, len(idx))
            d2 = ((P[s:e, None, :] - P[None, :, :]) ** 2).sum(-1)
            for r, name in zip(CA_RADII, ("ca_density_6", "ca_density",
                                          "ca_density_12")):
                out[idx[s:e], col[name]] = (d2 <= r * r).sum(1) - 1
            close = d2 <= 64.0
            out[idx[s:e], col["ca_longrange"]] = (
                close & (lagm[s:e] > LONG_RANGE_LAG)).sum(1)
            spans = np.where(close, lagm[s:e], 0).max(axis=1)
            out[idx[s:e], col["ca_seq_span"]] = np.minimum(spans, SEQ_SPAN_CAP)

    # --- F. the peptide plane ------------------------------------------------
    if n > 1:
        ok = link & fin(CA[:-1]) & fin(C[:-1]) & fin(N[1:])
        i = np.flatnonzero(ok)
        if len(i):
            nrm = np.cross(C[i] - CA[i], N[i + 1] - C[i])
            ln = np.linalg.norm(nrm, axis=1)
            good = ln > 1e-9
            j, nh = i[good], nrm[good] / ln[good, None]
            centre = np.nanmean(CA[okca], axis=0) if okca.any() else np.zeros(3)
            rad = CA[j] - centre
            rn = np.maximum(np.linalg.norm(rad, axis=1), 1e-9)
            out[j, col["plane_radial"]] = np.clip(
                np.abs((nh * (rad / rn[:, None])).sum(1)), -1.0, 1.0)
            keep = np.zeros((n, 3))
            keep[j] = nh
            nxt = _shift_rows(keep, -1, seg)
            both = (np.linalg.norm(keep, axis=1) > 0) & (
                np.linalg.norm(nxt, axis=1) > 0)
            out[both, col["plane_twist"]] = np.clip(
                (keep[both] * nxt[both]).sum(1), -1.0, 1.0)


def _shift_rows(v: np.ndarray, k: int, seg: np.ndarray) -> np.ndarray:
    n = len(v)
    out = np.zeros_like(v)
    idx = np.arange(n)
    src = idx - k
    ok = (src >= 0) & (src < n)
    ok[ok] &= seg[src[ok]] == seg[idx[ok]]
    out[ok] = v[src[ok]]
    return out


def compute(bb: dict[str, np.ndarray]) -> np.ndarray:
    """The thirteen quantities for one chain, as an (n_residues, 13) array.

    Every quantity is a function of backbone atom positions only. None of them is
    a function of residue type, which is what makes this family different in kind
    from the chemistry family that measured null.
    """
    N, CA, C, O = (bb["N"], bb["CA"], bb["C"], bb["O"])
    n = len(CA)
    out = np.zeros((n, N_COLUMNS), dtype=np.float64)
    if n == 0:
        return out
    col = {name: j for j, name in enumerate(COLUMNS)}

    # Which consecutive pairs are actually joined by a peptide bond. Everything
    # spanning a break stays at its neutral value.
    link = np.zeros(n - 1, dtype=bool) if n > 1 else np.zeros(0, dtype=bool)
    if n > 1:
        link = _bonded(C[:-1], N[1:])

    # phi needs C(i-1); psi needs N(i+1).
    if n > 1:
        ok = link & np.isfinite(C[:-1]).all(1) & np.isfinite(N[1:]).all(1)
        i = np.flatnonzero(ok)
        if len(i):
            good = (np.isfinite(N[i + 1]).all(1) & np.isfinite(CA[i + 1]).all(1)
                    & np.isfinite(C[i + 1]).all(1))
            j = i[good]
            if len(j):
                cs, sn = _dihedral(C[j], N[j + 1], CA[j + 1], C[j + 1])
                out[j + 1, col["cos_phi"]] = cs
                out[j + 1, col["sin_phi"]] = sn
            good = (np.isfinite(N[i]).all(1) & np.isfinite(CA[i]).all(1)
                    & np.isfinite(C[i]).all(1))
            j = i[good]
            if len(j):
                cs, sn = _dihedral(N[j], CA[j], C[j], N[j + 1])
                out[j, col["cos_psi"]] = cs
                out[j, col["sin_psi"]] = sn

    # The Ramachandran cell, from the angles the two circles determine. Only
    # residues with both torsions defined get a cell; the rest stay at "other".
    have = ((out[:, col["cos_phi"]] != 0) | (out[:, col["sin_phi"]] != 0)) & \
           ((out[:, col["cos_psi"]] != 0) | (out[:, col["sin_psi"]] != 0))
    if have.any():
        phi = np.degrees(np.arctan2(out[have, col["sin_phi"]],
                                    out[have, col["cos_phi"]]))
        psi = np.degrees(np.arctan2(out[have, col["sin_psi"]],
                                    out[have, col["cos_psi"]]))
        cell = np.full(len(phi), RAMA_OTHER, dtype=np.float64)
        alpha_r = (phi >= -160) & (phi <= -20) & (psi >= -120) & (psi <= 50)
        beta = ((phi >= -180) & (phi <= -20)
                & ((psi >= 90) | (psi <= -150)))
        alpha_l = (phi >= 20) & (phi <= 100) & (psi >= -20) & (psi <= 90)
        cell[beta] = RAMA_BETA
        cell[alpha_r] = RAMA_ALPHA_R
        cell[alpha_l] = RAMA_ALPHA_L
        out[have, col["rama_region"]] = cell

    # The CA trace: a turn angle at every interior residue, and a torsion over
    # four consecutive CA atoms. Both need only CA, so they survive a deposit
    # that is missing an N or a C.
    if n > 2:
        a, b, c = CA[:-2], CA[1:-1], CA[2:]
        u, v = a - b, c - b
        okc = np.isfinite(u).all(1) & np.isfinite(v).all(1)
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        okc &= (nu > 1e-6) & (nv > 1e-6)
        cosang = np.zeros(n - 2)
        cosang[okc] = np.clip((u[okc] * v[okc]).sum(1) / (nu[okc] * nv[okc]),
                              -1.0, 1.0)
        turn = np.zeros(n - 2)
        turn[okc] = np.degrees(np.arccos(cosang[okc]))
        out[1:-1, col["ca_turn"]] = turn
    if n > 3:
        ok4 = np.ones(n - 3, dtype=bool)
        for k in range(4):
            ok4 &= np.isfinite(CA[k:n - 3 + k]).all(1)
        i = np.flatnonzero(ok4)
        if len(i):
            cs, sn = _dihedral(CA[i], CA[i + 1], CA[i + 2], CA[i + 3])
            out[i + 1, col["cos_ca_tor"]] = cs
            out[i + 1, col["sin_ca_tor"]] = sn

    # Backbone hydrogen bonds, without hydrogens. A nitrogen donates at most one,
    # so the donated column is an indicator and the lag is the separation to the
    # partner it chose; a carbonyl can accept more than one, so that column is a
    # count. The two are different quantities and neither determines the other.
    okN = np.isfinite(N).all(1)
    okO = np.isfinite(O).all(1) & np.isfinite(C).all(1)
    if okN.any() and okO.any():
        idxN = np.flatnonzero(okN)
        idxO = np.flatnonzero(okO)
        d = np.linalg.norm(N[idxN][:, None, :] - O[idxO][None, :, :], axis=-1)
        close = d <= HBOND_MAX_NO
        # The carbonyl must point at the nitrogen rather than away from it.
        co = O[idxO] - C[idxO]
        con = co / np.maximum(np.linalg.norm(co, axis=1, keepdims=True), 1e-9)
        to_n = N[idxN][:, None, :] - O[idxO][None, :, :]
        to_n = to_n / np.maximum(np.linalg.norm(to_n, axis=-1, keepdims=True),
                                 1e-9)
        # In C=O...H-N the nitrogen lies beyond the oxygen, away from the
        # carbonyl carbon, so O->N and C->O point the same way. Requiring the
        # opposite admits every carbonyl that merely passes close, and the
        # symptom is unmistakable: the modal donated lag through a helix comes
        # out at 2 instead of the 4 an alpha turn fixes.
        pointing = (to_n * con[None, :, :]).sum(-1) > HBOND_MIN_COS
        lag = np.abs(idxN[:, None] - idxO[None, :])
        ok = close & pointing & (lag >= HBOND_MIN_LAG)
        if ok.any():
            # The donor keeps its closest acceptable acceptor.
            dd = np.where(ok, d, np.inf)
            best = dd.argmin(axis=1)
            has = np.isfinite(dd[np.arange(len(idxN)), best])
            rows = idxN[has]
            out[rows, col["hb_donated"]] = 1.0
            out[rows, col["hb_lag"]] = np.minimum(
                lag[np.arange(len(idxN)), best][has], HBOND_LAG_CAP)
            out[idxO, col["hb_accepted"]] = ok.sum(axis=0).astype(np.float64)

    # Where the side chain points, relative to the outward radial direction of
    # the chain. Glycine and any residue with a missing CB get the constructed
    # position, so the column is defined everywhere the backbone is.
    cb = bb["CB"].copy()
    need = ~np.isfinite(cb).all(1)
    full = np.isfinite(N).all(1) & np.isfinite(CA).all(1) & np.isfinite(C).all(1)
    fix = need & full
    if fix.any():
        cb[fix] = _virtual_cb(N[fix], CA[fix], C[fix])
    okcb = np.isfinite(cb).all(1) & np.isfinite(CA).all(1)
    if okcb.any():
        centre = np.nanmean(CA[np.isfinite(CA).all(1)], axis=0)
        v = cb[okcb] - CA[okcb]
        r = CA[okcb] - centre
        nv = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        nr = np.maximum(np.linalg.norm(r, axis=1), 1e-9)
        out[okcb, col["cb_radial"]] = np.clip((v * r).sum(1) / (nv * nr),
                                              -1.0, 1.0)

    # Backbone packing with the side chains removed.
    okca = np.isfinite(CA).all(1)
    if okca.sum() > 1:
        idx = np.flatnonzero(okca)
        P = CA[idx]
        r2 = CA_DENSITY_RADIUS ** 2
        dens = np.zeros(len(idx), dtype=np.float64)
        for s in range(0, len(idx), 512):
            e = min(s + 512, len(idx))
            d2 = ((P[s:e, None, :] - P[None, :, :]) ** 2).sum(-1)
            dens[s:e] = (d2 <= r2).sum(1) - 1
        out[idx, col["ca_density"]] = dens

    _expand(out, col, N, CA, C, O, cb, link)
    return out


def consistency(x: np.ndarray) -> list[str]:
    """Chemical and geometric facts the array must satisfy, whatever the input.

    These are properties of the definitions rather than of any structure, so a
    violation is a bug in this file and not an unusual protein.
    """
    bad = []
    col = {name: j for j, name in enumerate(COLUMNS)}
    for name in ("cos_phi", "sin_phi", "cos_psi", "sin_psi", "cos_ca_tor",
                 "sin_ca_tor", "cb_radial"):
        v = x[:, col[name]]
        if np.any(np.abs(v) > 1 + 1e-9):
            bad.append(f"{name} leaves [-1, 1]")
    for name, lo, hi in (("ca_turn", 0.0, 180.0),
                         ("hb_donated", 0.0, 1.0),
                         ("hb_lag", 0.0, float(HBOND_LAG_CAP)),
                         ("rama_region", 0.0, 3.0)):
        v = x[:, col[name]]
        if np.any(v < lo - 1e-9) or np.any(v > hi + 1e-9):
            bad.append(f"{name} leaves [{lo}, {hi}]")
    if np.any(x[:, col["hb_accepted"]] < 0):
        bad.append("hb_accepted is negative")
    if np.any(x[:, col["ca_density"]] < 0):
        bad.append("ca_density is negative")
    # A residue that donates nothing cannot have a lag, and one that donates must.
    don = x[:, col["hb_donated"]] > 0
    if np.any(x[~don, col["hb_lag"]] != 0):
        bad.append("a residue with no donated bond carries a lag")
    if np.any(x[don, col["hb_lag"]] < HBOND_MIN_LAG):
        bad.append("a donated bond is closer in sequence than the rule allows")
    # A cosine and a sine that are both zero mean undefined, which is the only
    # way the pair may leave the unit circle.
    for name in ("cos_omega", "sin_omega", "cos_phi_plus_psi",
                 "sin_phi_plus_psi", "cos_phi_minus_psi", "sin_phi_minus_psi",
                 "cb_tangent", "cb_binormal", "plane_radial", "plane_twist"):
        v = x[:, col[name]]
        if np.any(np.abs(v) > 1 + 1e-9):
            bad.append(f"{name} leaves [-1, 1]")
    for name, lo, hi in (("rama_prev", 0.0, 3.0), ("rama_next", 0.0, 3.0),
                         ("rama_run", 0.0, float(RUN_CAP)),
                         ("rama_variety", 0.0, 5.0),
                         ("dist_to_cell_change", 0.0, float(RUN_CAP)),
                         ("hb_lag_signed", -float(HBOND_LAG_CAP),
                          float(HBOND_LAG_CAP)),
                         ("hb_accept_lag", 0.0, float(HBOND_LAG_CAP)),
                         ("n_nearest_acceptor", 0.0, DIST_CAP),
                         ("o_nearest_donor", 0.0, DIST_CAP),
                         ("ca_seq_span", 0.0, float(SEQ_SPAN_CAP))):
        v = x[:, col[name]]
        if np.any(v < lo - 1e-9) or np.any(v > hi + 1e-9):
            bad.append(f"{name} leaves [{lo}, {hi}]")
    for name in ("hb_window_count", "hb_carbonyls_near_n", "cb_density",
                 "cb_hemisphere", "ca_density_6", "ca_density_12",
                 "ca_longrange", "ca_edge", "ca_span5", "ca_span7"):
        if np.any(x[:, col[name]] < 0):
            bad.append(f"{name} is negative")
    # Packing is monotone in the radius by construction, and a violation means
    # the three counts were not computed over the same point set.
    if np.any(x[:, col["ca_density_6"]] > x[:, col["ca_density"]] + 1e-9):
        bad.append("ca_density_6 exceeds ca_density at the larger radius")
    if np.any(x[:, col["ca_density"]] > x[:, col["ca_density_12"]] + 1e-9):
        bad.append("ca_density exceeds ca_density_12")
    # A signed lag and the unsigned one describe the same bond.
    signed = x[:, col["hb_lag_signed"]]
    if np.any(np.abs(signed) - x[:, col["hb_lag"]] > 1e-9):
        bad.append("hb_lag_signed and hb_lag disagree about the same bond")
    # A donor with a partner inside the bonding distance cannot be recorded as
    # having no carbonyl nearby.
    don = x[:, col["hb_donated"]] > 0
    if np.any(x[don, col["hb_carbonyls_near_n"]] < 1):
        bad.append("a residue donates a bond with no carbonyl near its nitrogen")
    for a, b in (("cos_phi", "sin_phi"), ("cos_psi", "sin_psi"),
                 ("cos_ca_tor", "sin_ca_tor"), ("cos_omega", "sin_omega"),
                 ("cos_phi_plus_psi", "sin_phi_plus_psi"),
                 ("cos_phi_minus_psi", "sin_phi_minus_psi")):
        r = np.hypot(x[:, col[a]], x[:, col[b]])
        live = r > 1e-9
        if np.any(np.abs(r[live] - 1.0) > 1e-6):
            bad.append(f"({a}, {b}) is neither on the unit circle nor zero")
    return bad
