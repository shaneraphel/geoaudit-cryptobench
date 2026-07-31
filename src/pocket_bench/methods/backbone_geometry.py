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
rather than thirteen versions of one:

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

COLUMNS = (
    "rama_region",
    "cos_phi", "sin_phi",
    "cos_psi", "sin_psi",
    "ca_turn",
    "cos_ca_tor", "sin_ca_tor",
    "hb_donated", "hb_accepted", "hb_lag",
    "cb_radial",
    "ca_density",
)
N_COLUMNS = len(COLUMNS)

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
    for a, b in (("cos_phi", "sin_phi"), ("cos_psi", "sin_psi"),
                 ("cos_ca_tor", "sin_ca_tor")):
        r = np.hypot(x[:, col[a]], x[:, col[b]])
        live = r > 1e-9
        if np.any(np.abs(r[live] - 1.0) > 1e-6):
            bad.append(f"({a}, {b}) is neither on the unit circle nor zero")
    return bad
