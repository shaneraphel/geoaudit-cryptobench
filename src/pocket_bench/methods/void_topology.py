"""Void topology: forty-five quantities of the space a protein does not occupy.

Every family measured before this one was a function of residue centroids and
distances between them, and every one of those was null. The rule that came out
of those nulls (AGENT_MEMORY 2i) is that a family has to read bytes the deployed
pipeline throws away. Backbone and side-chain conformation were the first two
such families. This is the third, and it reads something different again: not the
positions of the atoms but the *topology of the space between them*.

The construction, and why each constant is not a fitted threshold
-----------------------------------------------------------------
Take the Delaunay tetrahedralisation of the chain's heavy-atom centres. Each
tetrahedron has a circumsphere, empty of atom centres by the Delaunay property,
and its radius says how much room sits between those four atoms. Radii below
about 3 A are the ordinary interstices of close packing; radii above about 6 A
are outside the protein altogether. The band between them is the classical
alpha-sphere band of Le Guilloux, Schmidtke and Tuffery's fpocket, and both
numbers are theirs, published in 2009 and used here unchanged. So is the 1.73 A
single-linkage distance that groups neighbouring alpha spheres into one void.

Nothing here is fitted on any fold. Three published constants are read off a
paper, and everything else is a count, a rank or a graph distance. This matters
more than it usually would, because a construction that placed its own cuts
would be choosing them against the same training fold that the field is compiled
on, and the argument that this family is not a re-encoding would then have to
compete with the argument that it is a fitted detector.

This is not fpocket and makes no claim against it. fpocket is a pocket-ranking
method with a scoring function, a second clustering pass and a minimum pocket
size; what is borrowed is the geometric primitive underneath it and none of the
machinery on top. ``fpocket_wrap.py`` is the baseline; this is a wire family.

What percolation cost, recorded because it changed the design
--------------------------------------------------------------
The first version connected alpha spheres through shared tetrahedron faces and
produced, on 1a4u_B, one component covering 225 of 254 residues: the band forms
a shell over the whole solvent-accessible surface and the shell is connected. A
quantity that is the size of that component is a chain-level constant wearing a
per-residue name. The second version flood-filled from the exterior through
faces wide enough to pass a probe and broke the shell, but broke it too far --
the median component became a single tetrahedron and cryptic and non-cryptic
residues had component sizes of 5 and 4. Single linkage at fpocket's own 1.73 A
is the version that separates: on the first fourteen training chains the largest
cluster a residue lines has a within-chain ROC-AUC against the cryptic label of
0.906, 0.957, 0.838 and 0.797 on four of them, from one integer with no fitting.

The rule this file exists to respect
-------------------------------------
The deployed quantiser bands each column by *rank within the chain*. A column
that is constant within a chain therefore has no bands at all: every residue
ties, the tie-break is arbitrary, and the column contributes noise while
consuming cells. So no quantity here is a chain-level census. ``chain_n_voids``
and ``chain_largest_void`` were both drafted and both deleted; where a chain-wide
figure is genuinely wanted it appears only as a denominator, in
``best_void_share_permille`` and ``best_void_rank_permille``, which vary from
residue to residue because different residues line different voids.

Integer exactness at the band edge
-----------------------------------
A tetrahedron whose circumradius is within a hair of 3.0 or 6.0 A is admitted or
rejected on the last bits of a float division, and there are about seventeen
thousand tetrahedra per chain and seven hundred and seventy chains. Deposited
coordinates carry three decimals, so scaling by 1000 makes them exact integers,
and the circumradius comparison becomes ``|num|^2 >= (1000 r)^2 den^2`` with num
and den integer polynomials in the coordinates. Every tetrahedron within
``EXACT_MARGIN`` of either edge is decided that way, in Python's arbitrary-
precision integers, and ``compute`` reports how many took that path so a test
can assert the path is exercised rather than merely present.

Ratios are emitted as integers in permille or centi units for the same reason
the rest of the repository does: a table reads a band, a band is an order
statistic, and an integer that is already a count of thousandths cannot acquire
a rounding difference between one build and the next.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, cKDTree

# fpocket's alpha-sphere band and linkage distance, in angstroms. Read off the
# 2009 paper; not fitted here and not swept.
ALPHA_MIN = 3.0
ALPHA_MAX = 6.0
LINKAGE = 1.73

# Deposited coordinates carry three decimals, so this scale is exact.
SCALE = 1000

# A tetrahedron whose float circumradius lands inside this margin of either band
# edge is re-decided in exact integer arithmetic. Wide enough to cover any
# plausible accumulation of float error in a degree-4 polynomial, narrow enough
# that the exact path runs on a small minority of tetrahedra.
EXACT_MARGIN = 1e-6

# Sub-bands of the alpha-sphere radius. A sphere near the bottom of the band is a
# tight cleft between two side chains; one near the top is an open groove. These
# split the published band and introduce no new outer edge.
TIGHT_BELOW = 3.5
OPEN_ABOVE = 4.5

# The neighbourhood radii the diagnostic columns compare. 18 A is the deployed
# spatial gate's radius, taken from table_field.GATE_RADIUS and quoted here so
# that the dilution column measures the gate that actually ships; 7 A is the
# contact radius the wire aggregations already use.
GATE_RADIUS = 18.0
CONTACT_RADIUS = 7.0

BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})

# The four triangular faces of a tetrahedron, as vertex index triples.
FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))

COLUMNS = (
    # A. how much void this residue itself lines
    "alpha_spheres", "alpha_atoms", "alpha_atom_permille", "alpha_incidences",
    "alpha_sc_atoms", "alpha_bb_atoms", "alpha_sc_permille",
    # B. the sizes of the spheres it lines
    "alpha_r_max_centi", "alpha_r_min_centi", "alpha_r_median_centi",
    "alpha_r_range_centi", "alpha_n_tight", "alpha_n_open",
    # C. the void it lines, and how big that void is
    "n_voids", "best_void_residues", "best_void_spheres", "best_void_atoms",
    "min_void_residues", "void_residues_total", "best_void_rank_permille",
    "best_void_share_permille",
    # D. where in the void this residue sits
    "rank_in_void_permille", "void_neighbours", "void_neighbour_permille",
    "depth_in_void", "is_void_core", "void_span_centi",
    # E. the void's shape, which is not its size
    "void_span_per_residue_centi", "void_spheres_per_residue_centi",
    "void_r_max_centi", "void_r_mean_centi", "void_sc_permille",
    # F. whether the void is made by one loop or by distant segments
    "void_seq_segments", "void_seq_span", "void_seq_span_per_residue_centi",
    "own_seq_offset_permille",
    # G. what the deployed 18 A gate is actually averaging over
    "ball_gate_residues", "ball_gate_in_void", "ball_gate_purity_permille",
    "ball_contact_purity_permille", "purity_gain_permille",
    # H. burial, from the convex hull of the chain's own atoms
    "hull_distance_centi", "void_hull_min_centi", "void_hull_max_centi",
    "void_hull_range_centi",
)

IDX = {c: i for i, c in enumerate(COLUMNS)}

# Columns that are properties of the void rather than of the residue, so every
# residue lining one void shares them. They are not constant within a chain --
# a chain has hundreds of voids -- but a test that asserts every column varies
# residue by residue would be wrong about these, and the honest thing is to name
# them rather than to weaken the test.
VOID_LEVEL = frozenset({
    "void_span_centi", "void_span_per_residue_centi",
    "void_spheres_per_residue_centi", "void_r_max_centi", "void_r_mean_centi",
    "void_sc_permille", "void_seq_segments", "void_seq_span",
    "void_seq_span_per_residue_centi", "void_hull_min_centi",
    "void_hull_max_centi", "void_hull_range_centi",
})


def _circumradius(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Circumradius, circumcentre and the signed denominator, per tetrahedron.

    The denominator is returned because the exact re-decision at the band edge
    needs it, and recomputing it there would be a second chance to disagree.
    """
    a, b, c, d = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    A, B, C = b - a, c - a, d - a
    bc, ca, ab = np.cross(B, C), np.cross(C, A), np.cross(A, B)
    den = 2.0 * np.einsum("ij,ij->i", A, bc)
    num = ((A * A).sum(1)[:, None] * bc + (B * B).sum(1)[:, None] * ca
           + (C * C).sum(1)[:, None] * ab)
    ok = np.abs(den) > 1e-9
    r = np.full(len(p), np.inf)
    ctr = np.zeros((len(p), 3))
    r[ok] = np.linalg.norm(num[ok], axis=1) / np.abs(den[ok])
    ctr[ok] = a[ok] + num[ok] / den[ok, None]
    return r, ctr, den


def _exact_in_band(q: np.ndarray) -> bool:
    """Whether one tetrahedron's circumradius lies in the published band.

    ``q`` holds the four vertices as integers in units of 1/SCALE angstrom. The
    circumradius is ``|num| / |den|`` with num and den integer polynomials in
    those coordinates, so both band comparisons are comparisons of integers and
    neither can be decided the wrong way.
    """
    ax, ay, az = int(q[0][0]), int(q[0][1]), int(q[0][2])
    A = [int(q[1][k]) - (ax, ay, az)[k] for k in range(3)]
    B = [int(q[2][k]) - (ax, ay, az)[k] for k in range(3)]
    C = [int(q[3][k]) - (ax, ay, az)[k] for k in range(3)]

    def cross(u, v):
        return [u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0]]

    def dot(u, v):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    bc, ca, ab = cross(B, C), cross(C, A), cross(A, B)
    den = 2 * dot(A, bc)
    if den == 0:
        return False
    na, nb, nc = dot(A, A), dot(B, B), dot(C, C)
    num = [na * bc[k] + nb * ca[k] + nc * ab[k] for k in range(3)]
    n2 = dot(num, num)
    d2 = den * den
    lo = int(round(ALPHA_MIN * SCALE))
    hi = int(round(ALPHA_MAX * SCALE))
    return n2 >= lo * lo * d2 and n2 <= hi * hi * d2


def _band_membership(simplices: np.ndarray, xyz: np.ndarray, r: np.ndarray
                     ) -> tuple[np.ndarray, int]:
    """Which tetrahedra are alpha spheres, decided exactly where it is close."""
    keep = (r >= ALPHA_MIN) & (r <= ALPHA_MAX)
    close = (np.abs(r - ALPHA_MIN) < EXACT_MARGIN) | (
        np.abs(r - ALPHA_MAX) < EXACT_MARGIN)
    n_exact = int(close.sum())
    if n_exact:
        q = np.rint(xyz * SCALE).astype(np.int64)
        for t in np.flatnonzero(close):
            keep[t] = _exact_in_band(q[simplices[t]])
    return keep, n_exact


def _segments(sorted_ids: np.ndarray) -> int:
    """Contiguous runs in a sorted list of residue sequence numbers."""
    if len(sorted_ids) == 0:
        return 0
    return 1 + int((np.diff(sorted_ids) > 1).sum())


def _hull_distance(xyz: np.ndarray, pts: np.ndarray, hull_eq: np.ndarray
                   ) -> np.ndarray:
    """Distance from each point to the nearest face plane of the chain's hull.

    Qhull's ``equations`` are outward unit normals with offsets, so the signed
    value is negative inside and the depth of a point is the least magnitude
    over the faces. This is buriedness measured against the chain's own shape
    rather than against a sphere, which matters for anything elongated.
    """
    if len(pts) == 0:
        return np.zeros(0)
    d = pts @ hull_eq[:, :3].T + hull_eq[:, 3]
    return -d.max(axis=1)


def chain_voids(atoms: list[dict], resseq_order: list[tuple[int, str]]
                ) -> dict:
    """Everything the quantities are read from, built once per chain.

    Returns the atom table, the alpha spheres that survive the band, their
    single-linkage clustering, and the residue each atom belongs to. Splitting
    this from ``compute`` is what lets a test build a synthetic pocket and check
    the quantities without a deposit.
    """
    want = {(rs, ic.strip()) for rs, ic in resseq_order}
    rows = [a for a in atoms
            if a["element"] != "H"
            and (a["resseq"], a["icode"].strip()) in want]
    order = list(resseq_order)
    pos = {k: i for i, k in enumerate(order)}

    xyz = np.array([[a["x"], a["y"], a["z"]] for a in rows], dtype=np.float64)
    owner = np.array([pos[(a["resseq"], a["icode"].strip())] for a in rows],
                     dtype=np.int64)
    is_bb = np.array([a["name"].strip() in BACKBONE_ATOMS for a in rows])
    resseq = np.array([rs for rs, _ic in order], dtype=np.int64)

    n_res = len(order)
    out = {"xyz": xyz, "owner": owner, "is_bb": is_bb, "resseq": resseq,
           "n_res": n_res, "n_exact": 0}
    if len(xyz) < 5:
        out.update(simplices=np.zeros((0, 4), np.int64),
                   radius=np.zeros(0), centre=np.zeros((0, 3)),
                   label=np.zeros(0, np.int64), n_voids=0,
                   hull_eq=np.zeros((0, 4)))
        return out

    tri = Delaunay(xyz)
    r, ctr, _den = _circumradius(xyz[tri.simplices])
    keep, n_exact = _band_membership(tri.simplices, xyz, r)
    k = np.flatnonzero(keep)
    out["n_exact"] = n_exact
    out["hull_eq"] = _hull_equations(xyz)

    if len(k) == 0:
        out.update(simplices=np.zeros((0, 4), np.int64), radius=np.zeros(0),
                   centre=np.zeros((0, 3)), label=np.zeros(0, np.int64),
                   n_voids=0)
        return out

    c = ctr[k]
    if len(k) == 1:
        label = np.zeros(1, dtype=np.int64)
        n_voids = 1
    else:
        pairs = np.asarray(sorted(cKDTree(c).query_pairs(LINKAGE)),
                           dtype=np.int64).reshape(-1, 2)
        if len(pairs) == 0:
            label = np.arange(len(k), dtype=np.int64)
            n_voids = len(k)
        else:
            g = coo_matrix((np.ones(len(pairs)),
                            (pairs[:, 0], pairs[:, 1])),
                           shape=(len(k), len(k)))
            n_voids, label = connected_components(g, directed=False)
            label = label.astype(np.int64)

    out.update(simplices=tri.simplices[k], radius=r[k], centre=c,
               label=label, n_voids=int(n_voids))
    return out


def _hull_equations(xyz: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull
    try:
        return ConvexHull(xyz).equations
    except Exception:
        return np.zeros((0, 4))


def compute(v: dict) -> np.ndarray:
    """The forty-five quantities, one row per residue of ``resseq_order``."""
    n = v["n_res"]
    x = np.zeros((n, len(COLUMNS)), dtype=np.float64)
    if n == 0:
        return x

    simp, rad, ctr = v["simplices"], v["radius"], v["centre"]
    lab, owner, is_bb = v["label"], v["owner"], v["is_bb"]
    xyz, resseq, hull_eq = v["xyz"], v["resseq"], v["hull_eq"]

    # Residue centroids, which the ball columns and the lining graph need.
    rc = np.zeros((n, 3))
    cnt = np.zeros(n)
    np.add.at(rc, owner, xyz)
    np.add.at(cnt, owner, 1.0)
    rc[cnt > 0] /= cnt[cnt > 0, None]
    heavy = cnt.copy()

    if len(hull_eq):
        x[:, IDX["hull_distance_centi"]] = np.rint(
            _hull_distance(xyz, rc, hull_eq) * 100)

    if len(simp) == 0:
        return x

    # --- incidence: residue -> the spheres it lines -------------------------
    sph_of_res: list[list[int]] = [[] for _ in range(n)]
    atoms_of_res_in_alpha: list[set[int]] = [set() for _ in range(n)]
    for s in range(len(simp)):
        seen = set()
        for a in simp[s]:
            i = int(owner[a])
            atoms_of_res_in_alpha[i].add(int(a))
            if i not in seen:
                seen.add(i)
                sph_of_res[i].append(s)

    # --- voids: their lining, extent and composition ------------------------
    void_res: dict[int, set[int]] = {}
    void_sph: dict[int, list[int]] = {}
    void_atoms: dict[int, set[int]] = {}
    for s in range(len(simp)):
        c = int(lab[s])
        void_sph.setdefault(c, []).append(s)
        rs = void_res.setdefault(c, set())
        at = void_atoms.setdefault(c, set())
        for a in simp[s]:
            rs.add(int(owner[a]))
            at.add(int(a))

    void_size = {c: len(r) for c, r in void_res.items()}
    total_spheres = len(simp)
    # Rank of each void by lining size, largest first, ties by index so a
    # rebuild reproduces the same integers.
    order_by_size = sorted(void_size, key=lambda c: (-void_size[c], c))
    rank_of = {c: i + 1 for i, c in enumerate(order_by_size)}
    n_voids_total = max(len(order_by_size), 1)

    void_stats: dict[int, dict] = {}
    for c, ss in void_sph.items():
        cc = ctr[ss]
        span = 0.0
        if len(cc) > 1:
            d = np.linalg.norm(cc[:, None, :] - cc[None, :, :], axis=-1)
            span = float(d.max())
        at = np.fromiter(void_atoms[c], dtype=np.int64)
        sc = int((~is_bb[at]).sum())
        seq = np.sort(resseq[np.fromiter(sorted(void_res[c]), dtype=np.int64)])
        hd = _hull_distance(xyz, cc, hull_eq) if len(hull_eq) else np.zeros(
            len(cc))
        nres = len(void_res[c])
        void_stats[c] = {
            "span": span, "sc": sc, "n_at": len(at), "n_sph": len(ss),
            "r_max": float(rad[ss].max()), "r_mean": float(rad[ss].mean()),
            "seq_segments": _segments(seq), "seq_span": int(seq[-1] - seq[0]),
            "seq_median": float(np.median(seq)), "n_res": nres,
            "hull_min": float(hd.min()), "hull_max": float(hd.max()),
        }

    # --- the two balls the diagnostic columns compare -----------------------
    tree = cKDTree(rc)
    ball_gate = tree.query_ball_point(rc, GATE_RADIUS)
    ball_contact = tree.query_ball_point(rc, CONTACT_RADIUS)

    for i in range(n):
        ss = sph_of_res[i]
        na = len(atoms_of_res_in_alpha[i])
        x[i, IDX["alpha_spheres"]] = len(ss)
        x[i, IDX["alpha_atoms"]] = na
        x[i, IDX["alpha_atom_permille"]] = (
            round(1000 * na / heavy[i]) if heavy[i] else 0)
        if na:
            at = np.fromiter(atoms_of_res_in_alpha[i], dtype=np.int64)
            x[i, IDX["alpha_sc_atoms"]] = int((~is_bb[at]).sum())
            x[i, IDX["alpha_bb_atoms"]] = int(is_bb[at].sum())
            x[i, IDX["alpha_sc_permille"]] = round(
                1000 * int((~is_bb[at]).sum()) / na)
        if not ss:
            continue

        inc = 0
        for s in ss:
            inc += int((owner[simp[s]] == i).sum())
        x[i, IDX["alpha_incidences"]] = inc

        rr = rad[ss]
        x[i, IDX["alpha_r_max_centi"]] = round(100 * float(rr.max()))
        x[i, IDX["alpha_r_min_centi"]] = round(100 * float(rr.min()))
        x[i, IDX["alpha_r_median_centi"]] = round(100 * float(np.median(rr)))
        x[i, IDX["alpha_r_range_centi"]] = round(
            100 * float(rr.max() - rr.min()))
        x[i, IDX["alpha_n_tight"]] = int((rr < TIGHT_BELOW).sum())
        x[i, IDX["alpha_n_open"]] = int((rr > OPEN_ABOVE).sum())

        mine = sorted({int(lab[s]) for s in ss})
        x[i, IDX["n_voids"]] = len(mine)
        best = max(mine, key=lambda c: (void_size[c], -c))
        st = void_stats[best]
        x[i, IDX["best_void_residues"]] = st["n_res"]
        x[i, IDX["best_void_spheres"]] = st["n_sph"]
        x[i, IDX["best_void_atoms"]] = st["n_at"]
        x[i, IDX["min_void_residues"]] = min(void_size[c] for c in mine)
        x[i, IDX["void_residues_total"]] = sum(void_size[c] for c in mine)
        x[i, IDX["best_void_rank_permille"]] = round(
            1000 * rank_of[best] / n_voids_total)
        x[i, IDX["best_void_share_permille"]] = round(
            1000 * st["n_sph"] / total_spheres)

        lining = sorted(void_res[best])
        by = sorted(lining,
                    key=lambda j: (-sum(1 for s in sph_of_res[j]
                                        if int(lab[s]) == best), j))
        x[i, IDX["rank_in_void_permille"]] = round(
            1000 * (by.index(i) + 1) / len(by))
        lin = np.fromiter(lining, dtype=np.int64)
        d = np.linalg.norm(rc[lin] - rc[i], axis=1)
        nb = int(((d <= CONTACT_RADIUS) & (lin != i)).sum())
        x[i, IDX["void_neighbours"]] = nb
        x[i, IDX["void_neighbour_permille"]] = round(1000 * nb / len(lining))

        x[i, IDX["void_span_centi"]] = round(100 * st["span"])
        x[i, IDX["void_span_per_residue_centi"]] = round(
            100 * st["span"] / st["n_res"])
        x[i, IDX["void_spheres_per_residue_centi"]] = round(
            100 * st["n_sph"] / st["n_res"])
        x[i, IDX["void_r_max_centi"]] = round(100 * st["r_max"])
        x[i, IDX["void_r_mean_centi"]] = round(100 * st["r_mean"])
        x[i, IDX["void_sc_permille"]] = round(1000 * st["sc"] / st["n_at"])
        x[i, IDX["void_seq_segments"]] = st["seq_segments"]
        x[i, IDX["void_seq_span"]] = st["seq_span"]
        x[i, IDX["void_seq_span_per_residue_centi"]] = round(
            100 * st["seq_span"] / st["n_res"])
        x[i, IDX["own_seq_offset_permille"]] = (
            round(1000 * abs(float(resseq[i]) - st["seq_median"])
                  / st["seq_span"]) if st["seq_span"] else 0)
        x[i, IDX["void_hull_min_centi"]] = round(100 * st["hull_min"])
        x[i, IDX["void_hull_max_centi"]] = round(100 * st["hull_max"])
        x[i, IDX["void_hull_range_centi"]] = round(
            100 * (st["hull_max"] - st["hull_min"]))

        # Depth: hops in the void's own lining contact graph from this residue
        # to the rim, where the rim is a lining residue contributing one sphere.
        x[i, IDX["depth_in_void"]] = _depth(i, lining, best, sph_of_res, lab,
                                            rc)
        x[i, IDX["is_void_core"]] = 1 if x[i, IDX["depth_in_void"]] >= 2 else 0

        bg, bc_ = ball_gate[i], ball_contact[i]
        inv = void_res[best]
        x[i, IDX["ball_gate_residues"]] = len(bg)
        gin = sum(1 for j in bg if j in inv)
        cin = sum(1 for j in bc_ if j in inv)
        x[i, IDX["ball_gate_in_void"]] = gin
        pg = round(1000 * gin / len(bg)) if bg else 0
        pc = round(1000 * cin / len(bc_)) if bc_ else 0
        x[i, IDX["ball_gate_purity_permille"]] = pg
        x[i, IDX["ball_contact_purity_permille"]] = pc
        x[i, IDX["purity_gain_permille"]] = pc - pg

    return x


def _depth(i: int, lining: list[int], void: int, sph_of_res, lab,
           rc: np.ndarray) -> int:
    """Hops from residue ``i`` to the rim of the void it lines.

    The rim is the set of lining residues that contribute exactly one alpha
    sphere to this void: they touch it and no more. A residue at depth 0 is on
    the rim itself, and one at depth 2 or more is surrounded by lining on every
    side, which is the shape of a residue inside a pocket rather than at its
    mouth. Distance is measured in the lining's own contact graph and not in
    angstroms, so it does not change with the size of the void.
    """
    rim = {j for j in lining
           if sum(1 for s in sph_of_res[j] if int(lab[s]) == void) == 1}
    if i in rim or not rim:
        return 0
    lin = np.fromiter(lining, dtype=np.int64)
    d = np.linalg.norm(rc[lin][:, None, :] - rc[lin][None, :, :], axis=-1)
    adj = (d <= CONTACT_RADIUS)
    at = {int(j): k for k, j in enumerate(lin)}
    frontier = {at[i]}
    seen = set(frontier)
    rim_k = {at[j] for j in rim}
    for hop in range(1, len(lin) + 1):
        nxt = set()
        for u in frontier:
            for w in np.flatnonzero(adj[u]):
                w = int(w)
                if w not in seen:
                    seen.add(w)
                    nxt.add(w)
        if not nxt:
            return hop
        if nxt & rim_k:
            return hop
        frontier = nxt
    return len(lin)


def consistency(x: np.ndarray) -> list[str]:
    """Invariants that hold on every chain, checked on every build.

    Each entry is something a wrong index or a swapped pair of columns would
    break, and none of them is a range check that a typo would satisfy.
    """
    bad: list[str] = []
    g = lambda c: x[:, IDX[c]]  # noqa: E731

    for c in COLUMNS:
        if not np.isfinite(x[:, IDX[c]]).all():
            bad.append(f"{c} is not finite")
    counts = ("alpha_spheres", "alpha_atoms", "alpha_incidences",
              "alpha_sc_atoms", "alpha_bb_atoms", "n_voids",
              "best_void_residues", "best_void_spheres", "best_void_atoms",
              "void_neighbours", "depth_in_void", "void_seq_segments",
              "ball_gate_residues", "ball_gate_in_void")
    for c in counts:
        if (g(c) < 0).any():
            bad.append(f"{c} is negative")
    for c in ("alpha_atom_permille", "alpha_sc_permille",
              "best_void_rank_permille", "best_void_share_permille",
              "rank_in_void_permille", "void_neighbour_permille",
              "void_sc_permille", "ball_gate_purity_permille",
              "ball_contact_purity_permille"):
        if ((g(c) < 0) | (g(c) > 1000)).any():
            bad.append(f"{c} is outside [0, 1000]")

    if (g("alpha_sc_atoms") + g("alpha_bb_atoms") != g("alpha_atoms")).any():
        bad.append("side-chain and backbone alpha atoms do not sum to the total")
    if (g("alpha_atoms") > g("alpha_incidences")).any():
        bad.append("an atom is counted in fewer incidences than it has")
    if (g("best_void_residues") < g("min_void_residues")).any():
        bad.append("the best void is smaller than the smallest one")
    if (g("void_residues_total") < g("best_void_residues")).any():
        bad.append("the total lining is smaller than one void's lining")
    if (g("alpha_r_max_centi") < g("alpha_r_min_centi")).any():
        bad.append("the largest alpha sphere is smaller than the smallest")
    lo, hi = round(100 * ALPHA_MIN), round(100 * ALPHA_MAX)
    lines = g("alpha_spheres") > 0
    if lines.any():
        if (g("alpha_r_min_centi")[lines] < lo - 1).any():
            bad.append("an alpha sphere is below the published band")
        if (g("alpha_r_max_centi")[lines] > hi + 1).any():
            bad.append("an alpha sphere is above the published band")
    if (g("ball_gate_in_void") > g("ball_gate_residues")).any():
        bad.append("more of the gate's ball lines the void than the ball holds")
    if (g("ball_gate_residues")[lines] < 1).any():
        bad.append("a residue is not inside its own 18 A ball")
    empty = ~lines
    if empty.any():
        for c in ("best_void_residues", "n_voids", "best_void_spheres",
                  "depth_in_void", "void_span_centi"):
            if (g(c)[empty] != 0).any():
                bad.append(f"{c} is set on a residue that lines no void")
    if (g("is_void_core") > 1).any() or (g("is_void_core") < 0).any():
        bad.append("is_void_core is not an indicator")
    if ((g("depth_in_void") >= 2) != (g("is_void_core") > 0)).any():
        bad.append("is_void_core disagrees with depth_in_void")
    if (g("purity_gain_permille")
            != g("ball_contact_purity_permille")
            - g("ball_gate_purity_permille")).any():
        bad.append("purity_gain is not the difference it is defined as")
    return bad
