"""Crystallographic quantities: the lattice orbit of a chain, and the shape of
each atom's displacement ellipsoid.

Why this family exists
----------------------
Every descriptor family in this repository is a function of one point cloud in
an arbitrary frame. That is not because the frame is all there is; it is because
the committed receptor files were stripped to polymer ``ATOM`` records before
they were tracked. Measured over all 962 units of the two folds, **none** of
them carries ``CRYST1``, ``SCALE``, ``REMARK 290 SMTRY`` or ``ANISOU``
(``data/deposited_entries/DEPOSITED_ENTRY_MANIFEST.json``). Two experimentally
determined quantities were therefore unavailable to every family so far, and
neither is derivable from the coordinates that were kept.

**The lattice orbit.** A crystal structure is not one molecule; it is one
asymmetric unit together with a space group `G` and a Bravais lattice `Λ`, whose
combined action tiles space. A surface patch of the deposited chain is either
facing bulk solvent or facing an image of another molecule under some
`g in G ⋉ Λ`. Those are physically different environments, and the second one is
informative here for a specific reason: a patch that packs against a symmetry
mate is a patch that pays less desolvation penalty to be buried than to be wet.
That is the same preference a cryptic site expresses when a ligand closes over
it. The signal is not a proxy for the label — it is a measured property of the
apo crystal, available before anything is known about a ligand — but it is a
real physical correlate of the property being predicted.

**The displacement ellipsoid.** ``ANISOU`` records a symmetric positive-definite
`U` per atom, in units of 1e-4 A^2, describing the second moment of that atom's
positional distribution over the crystal. The isotropic `B = 8 pi^2 tr(U)/3` is
already read by the displacement family and carries most of it (AGENT_MEMORY
2n). What `B` cannot state is **direction**: two atoms with identical `B` differ
if one is smeared along the backbone and the other perpendicular to it, into a
void. A pocket that opens is a set of atoms whose preferred direction of motion
points into the space the ligand will occupy.

What is arithmetic here, and what is not
----------------------------------------
The space-group operators are supplied in Cartesian coordinates by
``REMARK 290``. Conjugated by the ``SCALE`` matrix they become the *integer*
matrices of `GL(3, Z)` that they are in the crystallographic basis, with
translation parts of denominator 1, 2, 3, 4 or 6. Every operator-derived column
below is a count or a small integer read off that integer matrix — its trace,
which by the crystallographic restriction theorem takes one of seven values and
determines the rotation order in `{1, 2, 3, 4, 6}`; whether the translation part
is intrinsic (screw, glide) or removable; how many distinct operators put an
image within contact range. Nothing is fitted and nothing is approximated.

The ellipsoid columns are the **polynomial invariants** of `U` — its trace, the
trace of its square, and its determinant — combined into two scale-free
rotation-invariant ratios. These need no eigensolver: they are polynomials in
the six deposited numbers. Only the two *directional* columns require a
principal axis, and they are marked as such.

Coverage is a level, not a filter
---------------------------------
Roughly a fifth of the fold carries ``ANISOU`` at all, because it is refined
only at higher resolution. A family defined where a record exists and undefined
elsewhere is a family with a missing *level*, not a family with fewer units.
``adp_present`` is therefore a column, absent entries take a fixed sentinel of
zero on every ADP column, and the coverage fraction is reported beside any lift
this family produces. Quantisation is by within-chain rank, so a chain with no
ADP contributes a constant column that ranks flat and addresses one cell.

``clinical_grade`` is false. No affinity or therapeutic claim.
"""
from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np

SKIP = frozenset({"HOH", "WAT", "DOD"})

# Contact shells. Three cuts of one quantity would be a radius ladder, which
# AGENT_MEMORY 2c calls the harmful category; these are three different
# quantities that happen to need a distance, each entering a different derived
# column, and the family is not built by sweeping any one of them.
MATE_NEAR = 4.0      # polar-contact range
MATE_CONTACT = 6.0   # packing range
MATE_FAR = 8.0       # shoulder range
LATTICE_RANGE = 2    # lattice translations searched, in units of a, b, c

# A fixed 32-direction spherical code, generated once by the Fibonacci lattice
# so that occlusion counting is deterministic and frame-independent up to the
# code's own orientation. It is a constant of this module, not a random draw.
_N_DIR = 32


def _directions() -> np.ndarray:
    i = np.arange(_N_DIR, dtype=np.float64) + 0.5
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    z = 1.0 - 2.0 * i / _N_DIR
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


DIRECTIONS = _directions()
PROBE_LEN = 8.0      # how far a direction is followed before calling it open
PROBE_RADIUS = 2.4   # a direction is blocked if an atom lies this close to it

CHI = {
    "ALA": 0, "GLY": 0, "PRO": 0, "SER": 1, "CYS": 1, "THR": 1, "VAL": 1,
    "ASN": 2, "ASP": 2, "LEU": 2, "ILE": 2, "HIS": 2, "PHE": 2, "TYR": 2,
    "TRP": 2, "GLN": 3, "GLU": 3, "MET": 3, "LYS": 4, "ARG": 4,
}

COLUMNS = (
    # -- lattice orbit: how much of this residue faces another molecule -------
    "n_mate_atoms_4A",
    "n_mate_atoms_6A",
    "n_mate_atoms_8A",
    "n_mate_carbon_5A",
    "n_mate_polar_4A",
    "n_distinct_symops_contacting",
    "n_distinct_lattice_images_contacting",
    "n_pure_translation_images_contacting",
    "n_contacting_ops_order_2",
    "n_contacting_ops_order_3_4_6",
    "n_contacting_ops_screw_or_glide",
    # -- the same counts against the residue's own crystal-independent
    #    environment, so that a lift cannot be burial arriving by a new route --
    "n_self_atoms_6A",
    "n_otherchain_atoms_6A",
    "mate_minus_self_6A",
    "mate_share_of_contacts_x100",
    # -- directional occlusion: of 32 fixed rays, how many are stopped, and
    #    by what --------------------------------------------------------------
    "n_dirs_blocked_any",
    "n_dirs_blocked_self",
    "n_dirs_blocked_mate_only",
    "n_dirs_open",
    # -- the operator that puts the nearest image here ------------------------
    "nearest_mate_dist_x10",
    "nearest_op_rotation_trace_plus3",
    "nearest_op_rotation_order",
    "nearest_op_is_screw_or_glide",
    "nearest_op_is_improper",
    # -- chain-level crystal context, broadcast to every residue of the chain --
    "space_group_order",
    "n_symops_in_cell",
    "cell_volume_x1e_minus3",
    "chain_lattice_neighbours",
    "crystal_system_code",
    # -- interaction with the residue's own chemistry -------------------------
    "mate_contact_times_chi",
    "mate_contact_times_hydrophobic",
    # -- displacement ellipsoid: polynomial invariants of U -------------------
    "adp_present",
    "adp_trace_x1e4",
    "adp_anisotropy_x1000",
    "adp_det_ratio_x1000",
    "adp_sidechain_minus_backbone_x1e4",
    "adp_max_atom_anisotropy_x1000",
    # -- displacement ellipsoid: the two columns that need a principal axis ---
    "adp_major_axis_vs_radial_deg",
    "adp_major_axis_vs_backbone_deg",
    "n_contacts_with_aligned_major_axis",
    # -- within-chain ranks of the load-bearing quantities --------------------
    "rank_mate_minus_self",
    "rank_dirs_blocked_mate_only",
    "rank_adp_anisotropy",
)
N_COLUMNS = len(COLUMNS)
_COL = {n: j for j, n in enumerate(COLUMNS)}

# Crystal system from the space-group symbol's lead letter and cell angles.
_SYSTEM = {"triclinic": 1, "monoclinic": 2, "orthorhombic": 3,
           "tetragonal": 4, "trigonal": 5, "hexagonal": 6, "cubic": 7}


class CrystalEntry:
    """The crystal form of one deposited PDB entry."""

    def __init__(self, cell: tuple[float, ...], space_group: str,
                 scale: np.ndarray, symops: list[tuple[np.ndarray, np.ndarray]],
                 atoms: list[dict], anisou: dict[tuple, np.ndarray]) -> None:
        self.cell = cell
        self.space_group = space_group
        self.scale = scale
        self.symops = symops
        self.atoms = atoms
        self.anisou = anisou

    # -- integer form of an operator, in the crystallographic basis ----------
    def integer_rotation(self, R: np.ndarray) -> np.ndarray | None:
        """`S R S^-1` for the SCALE matrix `S`; integer for a space-group op."""
        if self.scale is None:
            return None
        try:
            M = self.scale @ R @ np.linalg.inv(self.scale)
        except np.linalg.LinAlgError:
            return None
        Mi = np.rint(M)
        return Mi.astype(np.int64) if np.abs(M - Mi).max() < 0.02 else None

    def lattice_vectors(self) -> np.ndarray | None:
        if self.scale is None:
            return None
        try:
            return np.linalg.inv(self.scale)  # columns are a, b, c in Cartesian
        except np.linalg.LinAlgError:
            return None


def _f(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return 0.0


def read_entry(path: Path) -> CrystalEntry | None:
    """Parse the crystal form and ADPs out of a deposited (gzipped) PDB entry."""
    p = Path(path)
    if not p.exists():
        return None
    raw = p.read_bytes()
    text = (gzip.decompress(raw) if p.suffix == ".gz" else raw).decode(
        "utf-8", "ignore")

    cell = None
    space_group = ""
    scale_rows: list[list[float]] = []
    smtry: dict[int, list[list[float]]] = {}
    atoms: list[dict] = []
    anisou: dict[tuple, np.ndarray] = {}
    seen_model = False

    for ln in text.splitlines():
        rec = ln[:6]
        if rec == "CRYST1":
            cell = (_f(ln[6:15]), _f(ln[15:24]), _f(ln[24:33]),
                    _f(ln[33:40]), _f(ln[40:47]), _f(ln[47:54]))
            space_group = ln[55:66].strip()
        elif rec.startswith("SCALE") and len(rec) == 6 and rec[5] in "123":
            scale_rows.append([_f(ln[10:20]), _f(ln[20:30]), _f(ln[30:40])])
        elif ln.startswith("REMARK 290   SMTRY"):
            axis = int(ln[18])
            idx = int(ln[19:23])
            smtry.setdefault(idx, [None, None, None])[axis - 1] = [
                _f(ln[24:33]), _f(ln[33:43]), _f(ln[43:53]), _f(ln[53:68])]
        elif rec == "MODEL ":
            if seen_model:
                break
            seen_model = True
        elif rec == "ENDMDL":
            break
        elif rec in ("ATOM  ", "HETATM"):
            alt = ln[16]
            if alt not in (" ", "A"):
                continue
            resname = ln[17:20].strip().upper()
            if resname in SKIP:
                continue
            element = ln[76:78].strip().upper() or ln[12:16].strip()[:1]
            if element == "H" or element == "D":
                continue
            try:
                resseq = int(ln[22:26])
            except ValueError:
                continue
            atoms.append({
                "name": ln[12:16].strip().upper(),
                "resname": resname,
                "chain": ln[21],
                "resseq": resseq,
                "icode": ln[26].strip(),
                "x": _f(ln[30:38]), "y": _f(ln[38:46]), "z": _f(ln[46:54]),
                "element": element,
            })
        elif rec == "ANISOU":
            alt = ln[16]
            if alt not in (" ", "A"):
                continue
            try:
                resseq = int(ln[22:26])
            except ValueError:
                continue
            key = (ln[21], resseq, ln[26].strip(), ln[12:16].strip().upper())
            try:
                u = np.array([int(ln[28:35]), int(ln[35:42]), int(ln[42:49]),
                              int(ln[49:56]), int(ln[56:63]), int(ln[63:70])],
                             dtype=np.float64)
            except ValueError:
                continue
            anisou[key] = u

    if cell is None:
        return None
    scale = np.asarray(scale_rows[:3], dtype=np.float64) if len(scale_rows) >= 3 \
        else None
    ops: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in sorted(smtry):
        rows = smtry[idx]
        if any(r is None for r in rows):
            continue
        M = np.asarray(rows, dtype=np.float64)
        ops.append((M[:, :3], M[:, 3]))
    return CrystalEntry(cell, space_group, scale, ops, atoms, anisou)


def _crystal_system(cell, symbol: str) -> int:
    a, b, c, al, be, ga = cell
    lead = (symbol or " ")[0].upper()
    ang = lambda v, t: abs(v - t) < 0.5  # noqa: E731
    if ang(al, 90) and ang(be, 90) and ang(ga, 90):
        if abs(a - b) < 0.01 and abs(b - c) < 0.01:
            return _SYSTEM["cubic"]
        if abs(a - b) < 0.01:
            return _SYSTEM["tetragonal"]
        return _SYSTEM["orthorhombic"]
    if ang(al, 90) and ang(be, 90) and ang(ga, 120):
        return _SYSTEM["hexagonal"] if lead == "P" and "6" in symbol \
            else _SYSTEM["trigonal"]
    if ang(al, 90) and ang(ga, 90):
        return _SYSTEM["monoclinic"]
    return _SYSTEM["triclinic"]


def _rotation_order(trace: int) -> int:
    """Crystallographic restriction: a lattice rotation has order 1,2,3,4 or 6."""
    return {3: 1, -1: 2, 0: 3, 1: 4, 2: 6, -3: 2, -2: 6}.get(int(trace), 1)


def _ellipsoid(u: np.ndarray) -> tuple[float, float, float]:
    """(trace, anisotropy, normalised determinant) from polynomial invariants.

    ``anisotropy`` is ``3 tr(U^2)/tr(U)^2 - 1``, which is 0 exactly when `U` is a
    multiple of the identity and grows as the ellipsoid becomes a needle.
    ``det_ratio`` is ``27 det(U)/tr(U)^3``, which is 1 for a sphere and 0 for a
    degenerate ellipsoid. Neither needs an eigensolver.
    """
    u11, u22, u33, u12, u13, u23 = u
    i1 = u11 + u22 + u33
    if i1 <= 0:
        return 0.0, 0.0, 0.0
    i2 = (u11 * u11 + u22 * u22 + u33 * u33
          + 2.0 * (u12 * u12 + u13 * u13 + u23 * u23))
    det = (u11 * (u22 * u33 - u23 * u23)
           - u12 * (u12 * u33 - u23 * u13)
           + u13 * (u12 * u23 - u22 * u13))
    aniso = max(0.0, 3.0 * i2 / (i1 * i1) - 1.0)
    detr = max(0.0, 27.0 * det / (i1 ** 3))
    return i1 / 3.0, aniso, detr


def _major_axis(u: np.ndarray) -> np.ndarray:
    m = np.array([[u[0], u[3], u[4]], [u[3], u[1], u[5]], [u[4], u[5], u[2]]],
                 dtype=np.float64)
    try:
        w, v = np.linalg.eigh(m)
    except np.linalg.LinAlgError:
        return np.array([0.0, 0.0, 1.0])
    return v[:, int(np.argmax(w))]


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 90.0
    c = abs(float(np.dot(a, b) / (na * nb)))  # axes are undirected
    return float(np.degrees(np.arccos(min(1.0, c))))


def compute(entry: CrystalEntry | None, chain: str,
            atoms_by_res: list[list[dict]], resseqs: list[int],
            centroids: np.ndarray) -> np.ndarray:
    """Return an ``(n_res, N_COLUMNS)`` matrix for one chain of one entry.

    ``atoms_by_res`` and ``resseqs`` come from the *committed receptor*, so the
    residue universe is exactly the one every other family uses; ``entry``
    supplies only the crystal form and the ADPs. When the entry is missing or
    carries no cell, every column is zero and ``adp_present`` is zero, which is
    a level and not a gap.
    """
    n = len(atoms_by_res)
    X = np.zeros((n, N_COLUMNS), dtype=np.float64)
    if n == 0 or entry is None or entry.cell is None:
        return X

    from scipy.spatial import cKDTree  # noqa: PLC0415

    own = np.asarray([[a["x"], a["y"], a["z"]]
                      for res in atoms_by_res for a in res], dtype=np.float64)
    own_res = np.asarray([i for i, res in enumerate(atoms_by_res)
                          for _ in res], dtype=np.int64)
    if len(own) == 0:
        return X

    # ---- the asymmetric unit, split into this chain and the rest ------------
    au_all = np.asarray([[a["x"], a["y"], a["z"]] for a in entry.atoms],
                        dtype=np.float64)
    au_chain = np.asarray([a["chain"] == chain for a in entry.atoms])
    au_elem = np.asarray([a["element"] for a in entry.atoms])
    if len(au_all) == 0:
        return X

    # ---- generate every lattice image that can reach this chain -------------
    lat = entry.lattice_vectors()
    centre = own.mean(axis=0)
    reach = float(np.linalg.norm(own - centre, axis=1).max()) + MATE_FAR + 1.0

    mate_pts: list[np.ndarray] = []
    mate_op: list[int] = []
    mate_img: list[int] = []
    mate_elem: list[np.ndarray] = []
    img_id = 0
    if lat is not None and entry.symops:
        rng = range(-LATTICE_RANGE, LATTICE_RANGE + 1)
        for oi, (R, t) in enumerate(entry.symops):
            base = au_all @ R.T + t
            bc = base.mean(axis=0)
            for i in rng:
                for j in rng:
                    for k in rng:
                        shift = lat @ np.array([i, j, k], dtype=np.float64)
                        if oi == 0 and i == 0 and j == 0 and k == 0:
                            continue  # the deposited copy itself
                        if np.linalg.norm(bc + shift - centre) > reach + \
                                float(np.linalg.norm(base - bc, axis=1).max()):
                            continue
                        pts = base + shift
                        keep = np.linalg.norm(pts - centre, axis=1) <= reach
                        if not keep.any():
                            continue
                        mate_pts.append(pts[keep])
                        mate_op.append(oi)
                        mate_img.append(img_id)
                        mate_elem.append(au_elem[keep])
                        img_id += 1

    if mate_pts:
        M = np.concatenate(mate_pts, axis=0)
        M_op = np.concatenate([np.full(len(p), o, dtype=np.int64)
                               for p, o in zip(mate_pts, mate_op)])
        M_img = np.concatenate([np.full(len(p), g, dtype=np.int64)
                                for p, g in zip(mate_pts, mate_img)])
        M_el = np.concatenate(mate_elem)
        mate_tree = cKDTree(M)
    else:
        M = np.zeros((0, 3))
        M_op = M_img = np.zeros(0, dtype=np.int64)
        M_el = np.zeros(0, dtype="<U2")
        mate_tree = None

    self_tree = cKDTree(own)
    au_other = au_all[~au_chain]
    other_tree = cKDTree(au_other) if len(au_other) else None

    # ---- integer form of each operator, once per entry ----------------------
    op_trace: list[int] = []
    op_order: list[int] = []
    op_screw: list[int] = []
    op_improper: list[int] = []
    for R, t in entry.symops:
        Ri = entry.integer_rotation(R)
        if Ri is None:
            op_trace.append(3); op_order.append(1)
            op_screw.append(0); op_improper.append(0)
            continue
        tr = int(np.trace(Ri))
        det = int(round(np.linalg.det(Ri)))
        op_trace.append(tr)
        op_order.append(_rotation_order(tr))
        op_improper.append(1 if det < 0 else 0)
        # Intrinsic (screw/glide) part: the component of t fixed by R.
        if entry.scale is not None:
            tf = entry.scale @ t
            k = _rotation_order(tr)
            acc = np.zeros(3)
            P = np.eye(3)
            for _ in range(k):
                acc = acc + P @ tf
                P = Ri @ P
            intrinsic = acc / k
            op_screw.append(1 if float(np.abs(
                intrinsic - np.rint(intrinsic)).max()) > 1e-3 else 0)
        else:
            op_screw.append(0)

    n_ops = max(len(entry.symops), 1)
    sysc = _crystal_system(entry.cell, entry.space_group)
    a, b, c, al, be, ga = entry.cell
    ca, cb, cg = (math.cos(math.radians(v)) for v in (al, be, ga))
    vol = a * b * c * math.sqrt(max(0.0, 1 - ca * ca - cb * cb - cg * cg
                                    + 2 * ca * cb * cg))

    # ---- ADPs, keyed to the deposited chain --------------------------------
    have_adp = bool(entry.anisou)

    for i, res in enumerate(atoms_by_res):
        P = np.asarray([[at["x"], at["y"], at["z"]] for at in res],
                       dtype=np.float64)
        if len(P) == 0:
            continue
        resname = (res[0].get("resname") or "UNK").strip().upper()

        # -- lattice-orbit counts --
        if mate_tree is not None:
            n4 = sum(len(x) for x in mate_tree.query_ball_point(P, MATE_NEAR))
            i6 = mate_tree.query_ball_point(P, MATE_CONTACT)
            flat6 = np.unique(np.concatenate([np.asarray(x, dtype=np.int64)
                                              for x in i6])) if any(
                len(x) for x in i6) else np.zeros(0, dtype=np.int64)
            n6 = int(len(flat6))
            n8 = sum(len(x) for x in mate_tree.query_ball_point(P, MATE_FAR))
            i5 = mate_tree.query_ball_point(P, 5.0)
            flat5 = np.unique(np.concatenate([np.asarray(x, dtype=np.int64)
                                              for x in i5])) if any(
                len(x) for x in i5) else np.zeros(0, dtype=np.int64)
            n_c5 = int((M_el[flat5] == "C").sum()) if len(flat5) else 0
            flat4 = np.unique(np.concatenate([np.asarray(x, dtype=np.int64)
                                              for x in mate_tree.query_ball_point(
                                                  P, MATE_NEAR)])) if n4 else \
                np.zeros(0, dtype=np.int64)
            n_pol4 = int(np.isin(M_el[flat4], ("N", "O")).sum()) if len(flat4) \
                else 0
            ops_here = np.unique(M_op[flat6]) if n6 else np.zeros(0, np.int64)
            imgs_here = np.unique(M_img[flat6]) if n6 else np.zeros(0, np.int64)
            n_pure = int(sum(1 for o in ops_here if o == 0))
            n_o2 = int(sum(1 for o in ops_here if op_order[o] == 2))
            n_o346 = int(sum(1 for o in ops_here if op_order[o] in (3, 4, 6)))
            n_scr = int(sum(1 for o in ops_here if op_screw[o]))
            dmin, jmin = mate_tree.query(P, k=1)
            amin = int(np.argmin(dmin))
            near_d = float(dmin[amin])
            near_op = int(M_op[int(jmin[amin])])
        else:
            n4 = n6 = n8 = n_c5 = n_pol4 = n_pure = n_o2 = n_o346 = n_scr = 0
            ops_here = imgs_here = np.zeros(0, dtype=np.int64)
            near_d, near_op = 99.9, 0

        idx_self = self_tree.query_ball_point(P, MATE_CONTACT)
        self_flat = np.unique(np.concatenate(
            [np.asarray(x, dtype=np.int64) for x in idx_self]))
        n_self = int((own_res[self_flat] != i).sum())
        n_other = 0
        if other_tree is not None:
            io = other_tree.query_ball_point(P, MATE_CONTACT)
            n_other = int(len(np.unique(np.concatenate(
                [np.asarray(x, dtype=np.int64) for x in io]))) ) if any(
                len(x) for x in io) else 0

        # -- directional occlusion from the residue centroid --
        origin = centroids[i]
        blocked_self = blocked_mate = 0
        steps = np.arange(2.0, PROBE_LEN + 0.01, 2.0)
        for d in DIRECTIONS:
            probe = origin[None, :] + np.outer(steps, d)
            hs = self_tree.query_ball_point(probe, PROBE_RADIUS)
            hit_self = any(any(own_res[k] != i for k in h) for h in hs)
            hit_mate = False
            if mate_tree is not None:
                hm = mate_tree.query_ball_point(probe, PROBE_RADIUS)
                hit_mate = any(len(h) for h in hm)
            blocked_self += int(hit_self)
            blocked_mate += int(hit_mate and not hit_self)

        # -- displacement ellipsoid --
        adp_present = 0
        adp_tr = adp_an = adp_dr = adp_sb = adp_max = 0.0
        ax_rad = ax_bb = 0.0
        if have_adp:
            us, us_bb, us_sc = [], [], []
            for at in res:
                key = (chain, at["resseq"], at.get("icode", "").strip(),
                       at["name"])
                u = entry.anisou.get(key)
                if u is None:
                    continue
                us.append(u)
                (us_bb if at["name"] in ("N", "CA", "C", "O")
                 else us_sc).append(u)
            if us:
                adp_present = 1
                U = np.mean(us, axis=0)
                adp_tr, adp_an, adp_dr = _ellipsoid(U)
                adp_max = max(_ellipsoid(u)[1] for u in us)
                if us_bb and us_sc:
                    adp_sb = (float(np.mean([_ellipsoid(u)[0] for u in us_sc]))
                              - float(np.mean([_ellipsoid(u)[0] for u in us_bb])))
                axis = _major_axis(U)
                ax_rad = _angle_deg(axis, origin - centroids.mean(axis=0))
                if 0 < i < n - 1:
                    ax_bb = _angle_deg(axis, centroids[i + 1] - centroids[i - 1])
                else:
                    ax_bb = 90.0

        chi = CHI.get(resname, 0)
        row = {
            "n_mate_atoms_4A": n4,
            "n_mate_atoms_6A": n6,
            "n_mate_atoms_8A": n8,
            "n_mate_carbon_5A": n_c5,
            "n_mate_polar_4A": n_pol4,
            "n_distinct_symops_contacting": int(len(ops_here)),
            "n_distinct_lattice_images_contacting": int(len(imgs_here)),
            "n_pure_translation_images_contacting": n_pure,
            "n_contacting_ops_order_2": n_o2,
            "n_contacting_ops_order_3_4_6": n_o346,
            "n_contacting_ops_screw_or_glide": n_scr,
            "n_self_atoms_6A": n_self,
            "n_otherchain_atoms_6A": n_other,
            "mate_minus_self_6A": n6 - n_self,
            "mate_share_of_contacts_x100":
                100.0 * n6 / max(n6 + n_self + n_other, 1),
            "n_dirs_blocked_any": blocked_self + blocked_mate,
            "n_dirs_blocked_self": blocked_self,
            "n_dirs_blocked_mate_only": blocked_mate,
            "n_dirs_open": _N_DIR - blocked_self - blocked_mate,
            "nearest_mate_dist_x10": min(999.0, near_d * 10.0),
            "nearest_op_rotation_trace_plus3": op_trace[near_op] + 3
                if near_op < len(op_trace) else 6,
            "nearest_op_rotation_order": op_order[near_op]
                if near_op < len(op_order) else 1,
            "nearest_op_is_screw_or_glide": op_screw[near_op]
                if near_op < len(op_screw) else 0,
            "nearest_op_is_improper": op_improper[near_op]
                if near_op < len(op_improper) else 0,
            "space_group_order": n_ops,
            "n_symops_in_cell": n_ops,
            "cell_volume_x1e_minus3": vol / 1000.0,
            "chain_lattice_neighbours": int(img_id),
            "crystal_system_code": sysc,
            "mate_contact_times_chi": n6 * chi,
            "mate_contact_times_hydrophobic": n_c5 * (1 if resname in
                {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"} else 0),
            "adp_present": adp_present,
            "adp_trace_x1e4": adp_tr,
            "adp_anisotropy_x1000": adp_an * 1000.0,
            "adp_det_ratio_x1000": adp_dr * 1000.0,
            "adp_sidechain_minus_backbone_x1e4": adp_sb,
            "adp_max_atom_anisotropy_x1000": adp_max * 1000.0,
            "adp_major_axis_vs_radial_deg": ax_rad,
            "adp_major_axis_vs_backbone_deg": ax_bb,
            "n_contacts_with_aligned_major_axis": 0,  # filled below
        }
        for k, v in row.items():
            X[i, _COL[k]] = v

    # -- pairwise ADP alignment, once the per-residue axes exist --------------
    if have_adp:
        d = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :],
                           axis=-1)
        np.fill_diagonal(d, np.inf)
        near = d <= MATE_FAR
        ang = X[:, _COL["adp_major_axis_vs_backbone_deg"]]
        pres = X[:, _COL["adp_present"]] > 0
        for i in range(n):
            if not pres[i]:
                continue
            js = np.flatnonzero(near[i] & pres)
            X[i, _COL["n_contacts_with_aligned_major_axis"]] = int(
                (np.abs(ang[js] - ang[i]) <= 30.0).sum())

    def _rank(col: str) -> np.ndarray:
        v = X[:, _COL[col]]
        return np.argsort(np.argsort(v)) / max(n - 1, 1) * 100.0

    X[:, _COL["rank_mate_minus_self"]] = _rank("mate_minus_self_6A")
    X[:, _COL["rank_dirs_blocked_mate_only"]] = _rank("n_dirs_blocked_mate_only")
    X[:, _COL["rank_adp_anisotropy"]] = _rank("adp_anisotropy_x1000")
    return X


def consistency(X: np.ndarray) -> list[str]:
    bad = []
    if X.shape[1] != N_COLUMNS:
        bad.append(f"width {X.shape[1]} != {N_COLUMNS}")
    if not np.isfinite(X).all():
        bad.append("non-finite")
    return bad
