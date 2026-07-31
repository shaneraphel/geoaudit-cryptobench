"""Side-chain conformation: the second family the centroid representation destroys.

Why this module exists
----------------------
``AGENT_MEMORY`` 2i states the screen that six null families produced: a wire
family is worth measuring only if it reads bytes the deployed pipeline throws
away. The pipeline reduces a residue to ``c_i``, the centroid of its heavy
atoms, and builds every neighbourhood from centroid distances. Backbone geometry
was the first family that survived that screen and it is not null: 44 quantities,
+0.00441 against the deployed detector on 12 of 12 splits, and its row-permuted
control at −0.00213.

Side-chain *conformation* survives the same screen for a reason that is stated
rather than hoped. A centroid does not determine a torsion. A leucine whose CG
sits gauche⁺ and one whose CG sits trans have different χ₁, different terminal
atom positions, different packing at atom resolution and different exposed
surface, and they can present the same centroid to within the width of the
quantiser. Two independent facts make that concrete here:

* ``pdb_io.parse_pdb_atoms`` keeps ``name`` for every atom, so CG, CD, CE, CZ,
  OG, ND1, NE2 and the rest are on disk and are read by nothing;
* the deployed bank's only side-chain content is seven per-type constants
  (``kd``, ``volume``, ``aromatic``, ``charge``, ``hbd``, ``hba``, ``chi``),
  which ``AGENT_MEMORY`` 2i proved injective on the twenty types unquantised.
  Those constants are functions of *identity*. Nothing in them varies when the
  same leucine changes rotamer, which is exactly what this family measures.

Why identity is the thing to avoid, and how each quantity avoids it
-------------------------------------------------------------------
Chemistry 42 measured +0.000165 with its own control ahead of it, and the
post-mortem in 2i is that all fourteen of its columns were functions of residue
type, so the bank already determined them. This family is built under the
opposite constraint: **a quantity that a residue's three-letter code determines
is not admitted**, and where a raw measurement would be type-determined it is
either normalised by an identity-derived denominator computed from the same
residue, or paired with the identity-free version of itself.

Two examples of the discipline, both of which changed a definition here:

* ``sc_extension``, the distance from CA to the furthest side-chain atom, is
  mostly a statement that tryptophan is bigger than serine. Its companion
  ``sc_extension_ratio`` divides by the contour length of the same side chain —
  the sum of the bonded distances along its own atom path — so a fully extended
  arginine and a fully extended lysine both read near 1 and a curled one reads
  low, whatever the identity. The ratio is the quantity with a mechanism; the
  raw length is kept beside it only so that the pair can be compared.
* ``sc_polar_atoms`` is purely type-determined and would be inadmissible alone.
  It is present as the denominator of ``sc_satisfaction_ratio`` and beside
  ``sc_polar_unsatisfied``, which counts buried polar atoms with no hydrogen-bond
  partner and is a conformational quantity: the same asparagine is satisfied in
  one rotamer and frustrated in another.

The groups, and why each is a different kind of measurement
-----------------------------------------------------------
A family whose members are near-duplicates cannot be distinguished from a family
that is merely larger. That was the rule five null families produced and it is
the rule here. Each group answers a different question:

``A`` **torsions and rotamer combinatorics.** χ₁ to χ₄ as points on the circle,
      the rotamer well each lands in as a combinatorial label, the joint cell,
      and the angular distance to the nearest canonical well. A non-rotameric
      side chain is strained, and strain is a recognised signature of a site that
      opens.
``B`` **completeness and disorder.** Which side-chain atoms the deposit does not
      contain. A crystallographer omits an atom when its density is not there,
      and absent density is mobility. This reads the *absence* of bytes, which no
      re-encoding of present bytes can express.
``C`` **shape.** Extension against contour length, span, curl, the offset between
      the side-chain centroid and the residue centroid the pipeline uses. The
      last of these is identically zero for the deployed representation, because
      the deployed representation is the residue centroid.
``D`` **direction.** Where the side chain points relative to the outward radial,
      the backbone tangent and its own stub. A buried side chain pointing outward
      is a different situation from a buried one pointing inward and the centroid
      cannot tell them apart.
``E`` **packing at atom resolution.** The bank counts residues within a radius.
      This counts atoms, distinct contributing residues, and the ratio of the
      two, which separates one neighbour presenting five atoms from five
      neighbours presenting one each.
``F`` **open directions.** From the terminal atom and from CB, how many of twelve
      fixed directions have no heavy atom within a cutoff, how far the freest one
      runs, and whether an open direction has an open antipode. This is the van
      der Waals wall of the site as seen from the residue, and a cryptic pocket
      is a direction that is closed in the apo structure and open in the holo.
``G`` **polar satisfaction.** Side-chain hydrogen bonds donated and accepted,
      split by whether the partner is backbone or side chain, and the count of
      buried polar atoms with no partner at all.
``H`` **chirality.** The signed tetrahedral volume at CA, its magnitude, and the
      second chiral centre that isoleucine and threonine carry at CB. These are
      exact integers of sign and are the one quantity here that distinguishes a
      structure from its mirror image.
``I`` **rotamer agreement with the neighbourhood.** How many contacting residues
      sit in the same χ₁ well, how many wells are represented among them, and how
      many contacting side chains are non-rotameric. A cluster of strained side
      chains is a different object from one strained side chain.
``J`` **side-chain to backbone coupling.** χ₁ against φ and against ψ as points
      on the circle, and the joint Ramachandran × rotamer cell as a label. The
      coupling is neither a backbone quantity nor a side-chain quantity and is in
      neither family alone.
``K`` **the chemistry of the neighbourhood at atom resolution.** Carbon, oxygen,
      nitrogen and sulfur counts near the side chain. The bank's composition
      wires count residues by class; these count atoms, and a buried carbonyl
      oxygen belonging to a distant lysine is invisible to the former.

Angles are emitted as cosine and sine, never as degrees
--------------------------------------------------------
The detector quantises every wire by its rank within the chain, and a rank order
on a circle is a lie: −179° and +179° are two degrees apart and sit at opposite
ends of any ranking. Cosine and sine are genuine functions on the circle,
continuous across the wrap, and together they determine the angle. The cost is
two wires per torsion, which is the price of not misdescribing the topology.
This is the same convention ``backbone_geometry`` adopted and for the same
reason.

Undefined is neutral, never extreme
------------------------------------
Glycine has no χ₁, alanine has no χ₂, a disordered lysine may have no CE, and a
chain break removes φ. Every such case takes the neutral value — zero for a
cosine, a sine, a count, a ratio or a strain, and the ``undefined`` cell for a
label — following the appendix convention that a boundary sits in the middle of
the rank order rather than at its end. A sentinel of −999 would make every
glycine the extreme residue of its chain on eight wires at once.

Nothing here reads a label, the test fold, or any external unit.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Chemical topology. These tables are the IUPAC atom names of the twenty
# standard residues and the conventional chi-angle definitions. They are
# constants of chemistry, not thresholds fitted on anything here.
# ---------------------------------------------------------------------------

# The heavy atoms of each side chain, in the order they are bonded outward from
# CB. The order matters: the contour length in group C walks this list, and the
# terminal atom is the last entry that the deposit actually contains.
SIDECHAIN_ATOMS: dict[str, tuple[str, ...]] = {
    "ALA": ("CB",),
    "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "ASN": ("CB", "CG", "OD1", "ND2"),
    "ASP": ("CB", "CG", "OD1", "OD2"),
    "CYS": ("CB", "SG"),
    "GLN": ("CB", "CG", "CD", "OE1", "NE2"),
    "GLU": ("CB", "CG", "CD", "OE1", "OE2"),
    "GLY": (),
    "HIS": ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("CB", "CG1", "CG2", "CD1"),
    "LEU": ("CB", "CG", "CD1", "CD2"),
    "LYS": ("CB", "CG", "CD", "CE", "NZ"),
    "MET": ("CB", "CG", "SD", "CE"),
    "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("CB", "CG", "CD"),
    "SER": ("CB", "OG"),
    "THR": ("CB", "OG1", "CG2"),
    "TRP": ("CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("CB", "CG1", "CG2"),
}

# The bonded path from CB outward, used for the contour length. A ring is walked
# once around its longer arm rather than both ways, because a contour length is a
# path and a ring has two; taking the longer arm makes the ratio in group C a
# statement about how far the tip reaches along the chain it is attached by.
SIDECHAIN_PATH: dict[str, tuple[str, ...]] = {
    "ALA": ("CB",),
    "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1"),
    "ASN": ("CB", "CG", "OD1"),
    "ASP": ("CB", "CG", "OD1"),
    "CYS": ("CB", "SG"),
    "GLN": ("CB", "CG", "CD", "OE1"),
    "GLU": ("CB", "CG", "CD", "OE1"),
    "GLY": (),
    "HIS": ("CB", "CG", "CD2", "NE2"),
    "ILE": ("CB", "CG1", "CD1"),
    "LEU": ("CB", "CG", "CD1"),
    "LYS": ("CB", "CG", "CD", "CE", "NZ"),
    "MET": ("CB", "CG", "SD", "CE"),
    "PHE": ("CB", "CG", "CD1", "CE1", "CZ"),
    "PRO": ("CB", "CG", "CD"),
    "SER": ("CB", "OG"),
    "THR": ("CB", "OG1"),
    "TRP": ("CB", "CG", "CD2", "CE2", "CZ2", "CH2"),
    "TYR": ("CB", "CG", "CD1", "CE1", "CZ", "OH"),
    "VAL": ("CB", "CG1"),
}

# Conventional chi definitions. Each entry is the four atom names whose dihedral
# is that chi. Absent entries mean the residue has no such chi at all, which is a
# property of its chemistry and takes the neutral value rather than a sentinel.
CHI_ATOMS: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "ARG": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")),
    "ASN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "ASP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "CYS": (("N", "CA", "CB", "SG"),),
    "GLN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")),
    "GLU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")),
    "HIS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
    "ILE": (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")),
    "LEU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "LYS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")),
    "MET": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"),
            ("CB", "CG", "SD", "CE")),
    "PHE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "PRO": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "SER": (("N", "CA", "CB", "OG"),),
    "THR": (("N", "CA", "CB", "OG1"),),
    "TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "TYR": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "VAL": (("N", "CA", "CB", "CG1"),),
}

# Side-chain nitrogen and oxygen that can donate, and that can accept. Sulfur is
# excluded from both: its hydrogen bonds are weak, its geometry is looser, and
# admitting it would put a modelling judgement inside a wire.
SC_DONORS: dict[str, tuple[str, ...]] = {
    "ARG": ("NE", "NH1", "NH2"), "ASN": ("ND2",), "GLN": ("NE2",),
    "HIS": ("ND1", "NE2"), "LYS": ("NZ",), "SER": ("OG",),
    "THR": ("OG1",), "TRP": ("NE1",), "TYR": ("OH",),
}
SC_ACCEPTORS: dict[str, tuple[str, ...]] = {
    "ASN": ("OD1",), "ASP": ("OD1", "OD2"), "GLN": ("OE1",),
    "GLU": ("OE1", "OE2"), "HIS": ("ND1", "NE2"), "SER": ("OG",),
    "THR": ("OG1",), "TYR": ("OH",),
}

# The second chiral centre. Isoleucine and threonine are the two standard
# residues whose CB carries four distinct substituents, so their configuration is
# a fact about the deposit that a centroid cannot carry.
SECOND_CENTRE: dict[str, tuple[str, str]] = {
    "ILE": ("CG1", "CG2"),
    "THR": ("OG1", "CG2"),
}

BACKBONE_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})

# ---------------------------------------------------------------------------
# Constants of the geometry. None is fitted; each is either a textbook value or
# a cutoff whose role is to make a count finite.
# ---------------------------------------------------------------------------
ROTAMER_WELLS = (60.0, 180.0, 300.0)   # gauche+, trans, gauche-, in degrees
NON_ROTAMERIC_DEG = 30.0               # beyond this from every well is strained
CLOSE_CONTACT = 4.5                    # angstrom, an atom-resolution contact
WIDE_CONTACT = 6.0                     # angstrom, the second packing shell
CLASH_DISTANCE = 3.2                   # angstrom, closer than any comfortable pair
HBOND_MAX = 3.5                        # angstrom, donor heavy atom to acceptor
BURIAL_ATOMS = 12                      # neighbours above which a polar atom is
                                       # called buried, so that an unsatisfied
                                       # surface hydroxyl is not called frustrated
FREE_CAP = 12.0                        # angstrom, the ceiling on a free run
CONTACT_RADIUS_RESIDUE = 8.0           # angstrom, the residue-level graph reused
                                       # from the chemistry and backbone families
VIRTUAL_CA_BOND = 4.2                  # angstrom, CA(i) to CA(i+1); a real one is
                                       # 3.8, and beyond this the two residues are
                                       # not consecutive in the polymer

# Twelve directions, the vertices of a regular icosahedron, normalised. They come
# in six antipodal pairs, which is what makes the through-hole count in group F
# expressible. A fixed set is used rather than a random one so that the family is
# a deterministic function of the coordinates.
#
# An atom is assigned to the *nearest* of the twelve rather than to every
# direction within a cone about it. The first version used a 30-degree cone and
# it left holes: the covering radius of these twelve points is 37.4 degrees, so
# the +x axis sits outside every cone at once and a wall placed along it closed
# nothing. Nearest-direction assignment is the Voronoi partition of the sphere by
# these twelve points, it covers by construction, and it removes a constant that
# had no principled value.
_PHI = (1.0 + 5.0 ** 0.5) / 2.0
_ICOSA = np.array([
    [0.0, 1.0, _PHI], [0.0, -1.0, _PHI], [0.0, 1.0, -_PHI], [0.0, -1.0, -_PHI],
    [1.0, _PHI, 0.0], [-1.0, _PHI, 0.0], [1.0, -_PHI, 0.0], [-1.0, -_PHI, 0.0],
    [_PHI, 0.0, 1.0], [_PHI, 0.0, -1.0], [-_PHI, 0.0, 1.0], [-_PHI, 0.0, -1.0],
], dtype=np.float64)
DIRECTIONS = _ICOSA / np.linalg.norm(_ICOSA, axis=1, keepdims=True)
# Index of the antipode of each direction, used by the through-hole count.
ANTIPODE = np.array([np.argmin(DIRECTIONS @ d) for d in DIRECTIONS])

ROT_UNDEFINED, ROT_GAUCHE_PLUS, ROT_TRANS, ROT_GAUCHE_MINUS = 0, 1, 2, 3

COLUMNS = (
    # A. torsions and rotamer combinatorics
    "cos_chi1", "sin_chi1", "cos_chi2", "sin_chi2",
    "cos_chi3", "sin_chi3", "cos_chi4", "sin_chi4",
    "rot1", "rot2", "rot_joint", "n_chi_defined",
    "chi1_strain", "chi2_strain", "n_non_rotameric", "chi1_non_rotameric",
    # B. completeness and disorder
    "sc_atoms_observed", "sc_atoms_expected", "sc_atoms_missing",
    "sc_complete", "sc_missing_fraction", "has_real_cb", "terminal_missing",
    # C. shape
    "sc_extension", "sc_contour", "sc_extension_ratio", "sc_span",
    "sc_spread", "sc_curl", "ca_to_sc_centroid", "sc_centroid_offset",
    "sc_out_of_plane",
    # D. direction
    "sc_radial", "term_radial", "sc_minus_cb_radial", "sc_tangent",
    "sc_binormal", "sc_hemisphere",
    # E. packing at atom resolution
    "atoms_near_sc_close", "atoms_near_sc_wide", "atoms_near_term_close",
    "atoms_near_term_wide", "residues_near_sc_close", "atoms_per_residue",
    "sc_sc_contacts", "sc_bb_contacts", "sc_sc_fraction", "atoms_near_cb_wide",
    "burial_gradient",
    # F. open directions
    "open_cones_term", "open_cones_cb", "open_cone_ratio", "through_holes",
    "deepest_free", "free_anisotropy", "tightest_free",
    "shell_empty_close", "shell_empty_wide",
    # G. polar satisfaction
    "sc_hb_donated", "sc_hb_accepted", "sc_hb_to_backbone",
    "sc_hb_to_sidechain", "sc_polar_atoms", "sc_polar_buried",
    "sc_polar_unsatisfied", "sc_satisfaction", "sc_hb_min_lag",
    # H. chirality
    "ca_chirality", "ca_tetra_volume", "second_centre_chirality",
    "chi_handedness", "local_frame_chirality",
    # I. rotamer agreement with the neighbourhood
    "rot1_agree", "rot1_disagree", "rot1_variety", "non_rotameric_near",
    "sc_clashes",
    # J. side-chain to backbone coupling
    "cos_chi1_minus_phi", "sin_chi1_minus_phi",
    "cos_chi1_minus_psi", "sin_chi1_minus_psi", "rama_rot_cell",
    # K. neighbourhood chemistry at atom resolution
    "carbons_near_sc", "oxygens_near_sc", "nitrogens_near_sc",
    "sulfurs_near_sc", "polar_fraction_near_sc",
)
N_COLUMNS = len(COLUMNS)

_COL = {name: j for j, name in enumerate(COLUMNS)}


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
              p3: np.ndarray) -> tuple[float, float]:
    """Signed dihedral of four points, returned as (cos, sin).

    The angle is never formed. ``b1 x b2`` and ``b2 x b3`` are the normals of the
    two planes; their dot gives the cosine and the triple product with the
    normalised central bond gives the sine, which carries the sign and therefore
    the handedness. The sine is negated for the same reason it is negated in
    ``backbone_geometry``: this construction measures the rotation of the second
    plane onto the first and IUPAC fixes the sign the other way. Getting it
    backwards exchanges gauche⁺ and gauche⁻, which is not cosmetic — it would
    make every rotamer label in group A the mirror of the truth.
    """
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    nb2 = np.linalg.norm(b2)
    if nb2 < 1e-9:
        return 0.0, 0.0
    m1 = np.cross(n1, b2 / nb2)
    x = float(n1 @ n2)
    y = -float(m1 @ n2)
    r = float(np.hypot(x, y))
    if r < 1e-12:
        return 0.0, 0.0
    return x / r, y / r


def _rotamer_well(cos_a: float, sin_a: float) -> tuple[int, float]:
    """Which of the three staggered wells, and how far from it in degrees.

    The wells are the classical gauche⁺, trans and gauche⁻ at 60, 180 and 300
    degrees. They are constants of tetrahedral carbon, not values fitted on this
    dataset, and no library is consulted. The returned distance is the angular
    separation to the nearest well, which is the strain measure group A uses.
    """
    if cos_a == 0.0 and sin_a == 0.0:
        return ROT_UNDEFINED, 0.0
    ang = np.degrees(np.arctan2(sin_a, cos_a)) % 360.0
    best, best_d = 0, 360.0
    for k, w in enumerate(ROTAMER_WELLS):
        d = abs(ang - w)
        d = min(d, 360.0 - d)
        if d < best_d:
            best, best_d = k, d
    return best + 1, best_d


def chain_sidechain(atoms: list[dict],
                    resseq_order: list[tuple[int, str]]
                    ) -> tuple[list[dict[str, np.ndarray]], list[str]]:
    """Per-residue atom dictionaries and residue names, in scoring order.

    The residue order is supplied rather than inferred, for the reason
    ``backbone_wires`` records: the evaluation universe is the sorted set of
    integer resseq, the polymer is keyed by resseq *and* insertion code, and on a
    chain carrying insertion codes those two lists have different lengths. Every
    atom this module reads is looked up through the supplied order, so an
    off-by-one cannot enter here without the caller's map being wrong.
    """
    want = {k: i for i, k in enumerate(resseq_order)}
    n = len(resseq_order)
    per: list[dict[str, np.ndarray]] = [dict() for _ in range(n)]
    names = [""] * n
    for at in atoms:
        if at["record"] != "ATOM":
            continue
        i = want.get((at["resseq"], at["icode"].strip()))
        if i is None:
            continue
        names[i] = at["resname"]
        per[i][at["name"]] = np.array((at["x"], at["y"], at["z"]),
                                      dtype=np.float64)
    return per, names


def _atom_table(per: list[dict[str, np.ndarray]], names: list[str]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every heavy atom of the chain as one array, with owner and role.

    Returns coordinates, the residue index owning each atom, whether the atom is
    a side-chain atom, and its element as an integer code. Group E counts atoms
    rather than residues, and doing that from per-residue dictionaries in Python
    would be the slow part of this file; one flat table lets a single tree answer
    every neighbour question.
    """
    xs, owner, is_sc, elem = [], [], [], []
    code = {"C": 0, "O": 1, "N": 2, "S": 3}
    for i, d in enumerate(per):
        for name, p in d.items():
            if name.startswith("H") or name in ("D", "OXT"):
                continue
            e = name[0]
            if e not in code:
                continue
            xs.append(p)
            owner.append(i)
            is_sc.append(name not in BACKBONE_NAMES)
            elem.append(code[e])
    if not xs:
        return (np.zeros((0, 3)), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64))
    return (np.asarray(xs), np.asarray(owner, dtype=np.int64),
            np.asarray(is_sc, dtype=bool), np.asarray(elem, dtype=np.int64))


def compute(per: list[dict[str, np.ndarray]], names: list[str],
            phi: np.ndarray | None = None, psi: np.ndarray | None = None,
            rama: np.ndarray | None = None) -> np.ndarray:
    """The side-chain quantities for one chain, as an ``(n_residues, N)`` array.

    ``phi``, ``psi`` and ``rama`` are the backbone quantities group J couples to.
    They are passed in rather than recomputed so that the coupling is between the
    two families as they are actually deployed, and they are optional so that
    this module can be tested on a synthetic side chain with no backbone context.
    Absent, group J takes its neutral value and nothing else changes.
    """
    n = len(per)
    out = np.zeros((n, N_COLUMNS), dtype=np.float64)
    if n == 0:
        return out

    from scipy.spatial import cKDTree

    xyz, owner, is_sc, elem = _atom_table(per, names)
    tree = cKDTree(xyz) if len(xyz) else None

    ca = np.full((n, 3), np.nan)
    for i, d in enumerate(per):
        if "CA" in d:
            ca[i] = d["CA"]
    have_ca = np.isfinite(ca).all(1)
    centre = ca[have_ca].mean(axis=0) if have_ca.any() else np.zeros(3)

    # The residue-level contact graph, the same 8 A one the chemistry and
    # backbone families aggregate over, used here only by group I, which asks a
    # question about the neighbours' rotamers rather than about their positions.
    ctr = np.full((n, 3), np.nan)
    for i, d in enumerate(per):
        heavy = [p for name, p in d.items()
                 if not name.startswith("H") and name != "OXT"]
        if heavy:
            ctr[i] = np.mean(heavy, axis=0)

    rot1 = np.zeros(n, dtype=np.int64)
    non_rot = np.zeros(n, dtype=bool)

    # ---------------- pass one: everything local to the residue -------------
    for i, d in enumerate(per):
        rn = names[i]
        sc_names = SIDECHAIN_ATOMS.get(rn, ())
        present = [a for a in sc_names if a in d]
        out[i, _COL["sc_atoms_expected"]] = len(sc_names)
        out[i, _COL["sc_atoms_observed"]] = len(present)
        out[i, _COL["sc_atoms_missing"]] = len(sc_names) - len(present)
        out[i, _COL["sc_complete"]] = float(
            len(present) == len(sc_names) and len(sc_names) > 0)
        out[i, _COL["sc_missing_fraction"]] = (
            (len(sc_names) - len(present)) / len(sc_names) if sc_names else 0.0)
        out[i, _COL["has_real_cb"]] = float("CB" in d)
        if sc_names:
            out[i, _COL["terminal_missing"]] = float(sc_names[-1] not in d)

        # --- A. torsions and rotamer combinatorics --------------------------
        chis = CHI_ATOMS.get(rn, ())
        n_def = 0
        strains: list[float] = []
        wells: list[int] = []
        for k, quad in enumerate(chis[:4]):
            if not all(a in d for a in quad):
                break
            c, s = _dihedral(d[quad[0]], d[quad[1]], d[quad[2]], d[quad[3]])
            out[i, _COL[f"cos_chi{k + 1}"]] = c
            out[i, _COL[f"sin_chi{k + 1}"]] = s
            n_def += 1
            w, dist = _rotamer_well(c, s)
            wells.append(w)
            strains.append(dist)
        out[i, _COL["n_chi_defined"]] = n_def
        if wells:
            out[i, _COL["rot1"]] = wells[0]
            rot1[i] = wells[0]
            out[i, _COL["chi1_strain"]] = strains[0] / 60.0
            out[i, _COL["chi1_non_rotameric"]] = float(
                strains[0] > NON_ROTAMERIC_DEG)
            non_rot[i] = strains[0] > NON_ROTAMERIC_DEG
        if len(wells) > 1:
            out[i, _COL["rot2"]] = wells[1]
            out[i, _COL["chi2_strain"]] = strains[1] / 60.0
            out[i, _COL["rot_joint"]] = 1 + 3 * (wells[0] - 1) + (wells[1] - 1)
        out[i, _COL["n_non_rotameric"]] = sum(
            1 for s in strains if s > NON_ROTAMERIC_DEG)
        # The handedness of the first two torsions as one label. sign(sin) is +1
        # for a rotation one way and -1 for the other, so the pair takes nine
        # values and is a combinatorial statement about the side chain's twist
        # that neither cosine carries.
        if n_def >= 1:
            s1 = int(np.sign(out[i, _COL["sin_chi1"]]))
            s2 = int(np.sign(out[i, _COL["sin_chi2"]])) if n_def >= 2 else 0
            out[i, _COL["chi_handedness"]] = 1 + 3 * (s1 + 1) + (s2 + 1)

        # --- C. shape --------------------------------------------------------
        pts = np.asarray([d[a] for a in present]) if present else None
        if pts is not None and np.isfinite(ca[i]).all():
            dists = np.linalg.norm(pts - ca[i], axis=1)
            out[i, _COL["sc_extension"]] = float(dists.max())
            sc_ctr = pts.mean(axis=0)
            out[i, _COL["ca_to_sc_centroid"]] = float(
                np.linalg.norm(sc_ctr - ca[i]))
            out[i, _COL["sc_spread"]] = float(
                np.linalg.norm(pts - sc_ctr, axis=1).mean())
            if np.isfinite(ctr[i]).all():
                out[i, _COL["sc_centroid_offset"]] = float(
                    np.linalg.norm(sc_ctr - ctr[i]))
            if len(pts) > 1:
                pair = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
                out[i, _COL["sc_span"]] = float(pair.max())
            # Non-planarity: the largest tetrahedron the first four side-chain
            # atoms span. A phenyl ring is flat and reads near zero; a strained
            # or misbuilt one does not. This is a signed volume taken in absolute
            # value, so it is a shape statement and not a chirality statement.
            if len(pts) >= 4:
                v = pts[1:4] - pts[0]
                out[i, _COL["sc_out_of_plane"]] = abs(
                    float(np.linalg.det(v))) / 6.0

        path = [a for a in SIDECHAIN_PATH.get(rn, ()) if a in d]
        if len(path) >= 2:
            p = np.asarray([d[a] for a in path])
            contour = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
            out[i, _COL["sc_contour"]] = contour
            if contour > 1e-9:
                out[i, _COL["sc_extension_ratio"]] = float(
                    np.linalg.norm(p[-1] - p[0]) / contour)
        if "CB" in d and present and np.isfinite(ca[i]).all():
            term = d[present[-1]]
            u = d["CB"] - ca[i]
            v = term - d["CB"]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu > 1e-9 and nv > 1e-9:
                out[i, _COL["sc_curl"]] = float(np.clip(u @ v / (nu * nv),
                                                        -1.0, 1.0))

        # --- D. direction -----------------------------------------------------
        if np.isfinite(ca[i]).all() and present:
            rad = ca[i] - centre
            nr = np.linalg.norm(rad)
            term = d[present[-1]]
            sc_ctr = np.asarray([d[a] for a in present]).mean(axis=0)
            if nr > 1e-9:
                for key, vec in (("sc_radial", sc_ctr - ca[i]),
                                 ("term_radial", term - ca[i])):
                    nv = np.linalg.norm(vec)
                    if nv > 1e-9:
                        out[i, _COL[key]] = float(
                            np.clip(vec @ rad / (nv * nr), -1.0, 1.0))
                if "CB" in d:
                    cb = d["CB"] - ca[i]
                    ncb = np.linalg.norm(cb)
                    if ncb > 1e-9:
                        cbr = float(np.clip(cb @ rad / (ncb * nr), -1.0, 1.0))
                        out[i, _COL["sc_minus_cb_radial"]] = (
                            out[i, _COL["sc_radial"]] - cbr)
            out[i, _COL["sc_hemisphere"]] = float(
                np.sign(out[i, _COL["sc_radial"]]))

        # Where the side chain points relative to the *backbone's own frame*
        # rather than to the protein's outward radial. The two are different
        # questions: a residue on the inside of a curved sheet has one answer
        # from the radial and another from its own tangent, and the tangent
        # version survives on a chain whose overall shape is not globular.
        # Consecutive CA positions are used only when the virtual bond is short
        # enough to be a real one, so a chain break contributes nothing.
        prev_ok = (i > 0 and np.isfinite(ca[i - 1]).all()
                   and np.isfinite(ca[i]).all()
                   and np.linalg.norm(ca[i] - ca[i - 1]) <= VIRTUAL_CA_BOND)
        next_ok = (i + 1 < n and np.isfinite(ca[i + 1]).all()
                   and np.isfinite(ca[i]).all()
                   and np.linalg.norm(ca[i + 1] - ca[i]) <= VIRTUAL_CA_BOND)
        if prev_ok and next_ok and present:
            tangent = ca[i + 1] - ca[i - 1]
            curvature = ca[i + 1] + ca[i - 1] - 2.0 * ca[i]
            binormal = np.cross(tangent, curvature)
            sc_ctr = np.asarray([d[a] for a in present]).mean(axis=0)
            v = sc_ctr - ca[i]
            nv = np.linalg.norm(v)
            for key, axis in (("sc_tangent", tangent),
                              ("sc_binormal", binormal)):
                na = np.linalg.norm(axis)
                if nv > 1e-9 and na > 1e-9:
                    out[i, _COL[key]] = float(
                        np.clip(v @ axis / (nv * na), -1.0, 1.0))
            if "CB" in d:
                # The handedness of the CB stub against the direction the chain
                # is travelling. Unlike ca_chirality this is not fixed by the
                # L configuration: it reverses with the local backbone
                # conformation, so it is a statement about the fold and not
                # about the amino acid.
                m = np.stack([d["CB"] - ca[i], ca[i + 1] - ca[i],
                              ca[i - 1] - ca[i]])
                out[i, _COL["local_frame_chirality"]] = float(
                    np.sign(np.linalg.det(m)))

        # --- H. chirality ------------------------------------------------------
        if all(a in d for a in ("N", "CA", "C", "CB")):
            v = np.stack([d["CB"] - d["CA"], d["N"] - d["CA"],
                          d["C"] - d["CA"]])
            det = float(np.linalg.det(v)) / 6.0
            out[i, _COL["ca_chirality"]] = float(np.sign(det))
            out[i, _COL["ca_tetra_volume"]] = abs(det)
        sec = SECOND_CENTRE.get(rn)
        if sec and all(a in d for a in sec + ("CB", "CA")):
            v = np.stack([d[sec[0]] - d["CB"], d[sec[1]] - d["CB"],
                          d["CA"] - d["CB"]])
            out[i, _COL["second_centre_chirality"]] = float(
                np.sign(np.linalg.det(v)))

        # --- G. polar atoms present (the denominator; satisfaction is pass two)
        polar = [a for a in SC_DONORS.get(rn, ()) + SC_ACCEPTORS.get(rn, ())
                 if a in d]
        out[i, _COL["sc_polar_atoms"]] = len(set(polar))

    # ---------------- pass two: everything that needs the neighbours ---------
    if tree is not None and len(xyz):
        for i, d in enumerate(per):
            rn = names[i]
            present = [a for a in SIDECHAIN_ATOMS.get(rn, ()) if a in d]
            if not present:
                continue
            pts = np.asarray([d[a] for a in present])

            close = tree.query_ball_point(pts, CLOSE_CONTACT)
            wide = tree.query_ball_point(pts, WIDE_CONTACT)
            close_idx = {j for lst in close for j in lst if owner[j] != i}
            wide_idx = {j for lst in wide for j in lst if owner[j] != i}
            ci = np.fromiter(close_idx, dtype=np.int64, count=len(close_idx))
            wi = np.fromiter(wide_idx, dtype=np.int64, count=len(wide_idx))
            out[i, _COL["atoms_near_sc_close"]] = len(ci)
            out[i, _COL["atoms_near_sc_wide"]] = len(wi)
            if len(ci):
                res = np.unique(owner[ci])
                out[i, _COL["residues_near_sc_close"]] = len(res)
                out[i, _COL["atoms_per_residue"]] = len(ci) / len(res)
                out[i, _COL["sc_sc_contacts"]] = int(is_sc[ci].sum())
                out[i, _COL["sc_bb_contacts"]] = int((~is_sc[ci]).sum())
                out[i, _COL["sc_sc_fraction"]] = float(is_sc[ci].mean())
                # A clash is a pair closer than any comfortable contact. It is
                # counted rather than flagged because one is a modelling
                # imperfection and several is a strained region.
                dd = np.linalg.norm(pts[:, None, :] - xyz[ci][None, :, :],
                                    axis=-1)
                out[i, _COL["sc_clashes"]] = int((dd < CLASH_DISTANCE).sum())
            if len(wi):
                e = elem[wi]
                out[i, _COL["carbons_near_sc"]] = int((e == 0).sum())
                out[i, _COL["oxygens_near_sc"]] = int((e == 1).sum())
                out[i, _COL["nitrogens_near_sc"]] = int((e == 2).sum())
                out[i, _COL["sulfurs_near_sc"]] = int((e == 3).sum())
                out[i, _COL["polar_fraction_near_sc"]] = float(
                    ((e == 1) | (e == 2)).mean())

            term = d[present[-1]]
            for key, pt, rad in (("atoms_near_term_close", term, CLOSE_CONTACT),
                                 ("atoms_near_term_wide", term, WIDE_CONTACT)):
                idx = [j for j in tree.query_ball_point(pt, rad)
                       if owner[j] != i]
                out[i, _COL[key]] = len(idx)
            if "CB" in d:
                idx = [j for j in tree.query_ball_point(d["CB"], WIDE_CONTACT)
                       if owner[j] != i]
                out[i, _COL["atoms_near_cb_wide"]] = len(idx)
                out[i, _COL["burial_gradient"]] = (
                    out[i, _COL["atoms_near_term_wide"]] - len(idx))

            # --- F. open directions -------------------------------------------
            free_term = _free_runs(tree, xyz, owner, i, term)
            out[i, _COL["open_cones_term"]] = int(
                (free_term >= FREE_CAP).sum())
            out[i, _COL["deepest_free"]] = float(free_term.max())
            out[i, _COL["tightest_free"]] = float(free_term.min())
            out[i, _COL["free_anisotropy"]] = float(
                free_term.max() - free_term.min())
            open_mask = free_term >= FREE_CAP
            # A through-hole is a *pair* of opposite open directions, so each is
            # counted once by keeping only the half of the pairs whose index is
            # the lower of the two. Counting the mask against its own antipode
            # map without that restriction counts every hole twice, which is
            # what the isolated-residue test caught.
            lower = np.arange(len(ANTIPODE)) < ANTIPODE
            out[i, _COL["through_holes"]] = int(
                (open_mask & open_mask[ANTIPODE] & lower).sum())
            out[i, _COL["shell_empty_close"]] = int(
                (free_term > CLOSE_CONTACT).sum())
            out[i, _COL["shell_empty_wide"]] = int(
                (free_term > WIDE_CONTACT).sum())
            if "CB" in d:
                free_cb = _free_runs(tree, xyz, owner, i, d["CB"])
                out[i, _COL["open_cones_cb"]] = int((free_cb >= FREE_CAP).sum())
                if out[i, _COL["open_cones_cb"]] > 0:
                    out[i, _COL["open_cone_ratio"]] = (
                        out[i, _COL["open_cones_term"]]
                        / out[i, _COL["open_cones_cb"]])

            # --- G. hydrogen bonds of the side chain ---------------------------
            donors = [a for a in SC_DONORS.get(rn, ()) if a in d]
            acceptors = [a for a in SC_ACCEPTORS.get(rn, ()) if a in d]
            n_don = n_acc = n_bb = n_sc = 0
            lags: list[int] = []
            buried = unsat = 0
            for a in set(donors + acceptors):
                p = d[a]
                partners = [j for j in tree.query_ball_point(p, HBOND_MAX)
                            if owner[j] != i and elem[j] in (1, 2)]
                near = [j for j in tree.query_ball_point(p, WIDE_CONTACT)
                        if owner[j] != i]
                is_buried = len(near) >= BURIAL_ATOMS
                buried += int(is_buried)
                if not partners:
                    unsat += int(is_buried)
                for j in partners:
                    if is_sc[j]:
                        n_sc += 1
                    else:
                        n_bb += 1
                    lags.append(abs(int(owner[j]) - i))
                if a in donors:
                    n_don += len(partners)
                if a in acceptors:
                    n_acc += len(partners)
            out[i, _COL["sc_hb_donated"]] = n_don
            out[i, _COL["sc_hb_accepted"]] = n_acc
            out[i, _COL["sc_hb_to_backbone"]] = n_bb
            out[i, _COL["sc_hb_to_sidechain"]] = n_sc
            out[i, _COL["sc_polar_buried"]] = buried
            out[i, _COL["sc_polar_unsatisfied"]] = unsat
            npolar = out[i, _COL["sc_polar_atoms"]]
            if npolar > 0:
                out[i, _COL["sc_satisfaction"]] = (n_bb + n_sc) / npolar
            out[i, _COL["sc_hb_min_lag"]] = min(lags) if lags else 0

    # --- I. rotamer agreement with the neighbourhood -------------------------
    ok = np.isfinite(ctr).all(1)
    if ok.sum() > 1:
        idx = np.flatnonzero(ok)
        dd = np.linalg.norm(ctr[idx][:, None, :] - ctr[idx][None, :, :],
                            axis=-1)
        np.fill_diagonal(dd, np.inf)
        adj = dd <= CONTACT_RADIUS_RESIDUE
        for a, i in enumerate(idx):
            nb = idx[adj[a]]
            if not len(nb):
                continue
            theirs = rot1[nb]
            defined = theirs[theirs > 0]
            out[i, _COL["rot1_agree"]] = int((defined == rot1[i]).sum()
                                             if rot1[i] > 0 else 0)
            out[i, _COL["rot1_disagree"]] = int((defined != rot1[i]).sum()
                                                if rot1[i] > 0 else 0)
            out[i, _COL["rot1_variety"]] = int(len(np.unique(defined)))
            out[i, _COL["non_rotameric_near"]] = int(non_rot[nb].sum())

    # --- J. side-chain to backbone coupling ----------------------------------
    if phi is not None and psi is not None:
        c1, s1 = out[:, _COL["cos_chi1"]], out[:, _COL["sin_chi1"]]
        have = (c1 != 0) | (s1 != 0)
        for key, (cb_, sb_) in (("phi", (phi[:, 0], phi[:, 1])),
                                ("psi", (psi[:, 0], psi[:, 1]))):
            # cos(chi - theta) and sin(chi - theta) from the two circle points,
            # without ever forming either angle. The difference of two points on
            # the circle is a point on the circle, and it is the coupling rather
            # than either coordinate that group J is about.
            cc = c1 * cb_ + s1 * sb_
            ss = s1 * cb_ - c1 * sb_
            out[have, _COL[f"cos_chi1_minus_{key}"]] = cc[have]
            out[have, _COL[f"sin_chi1_minus_{key}"]] = ss[have]
    if rama is not None:
        have = out[:, _COL["rot1"]] > 0
        out[have, _COL["rama_rot_cell"]] = (
            1 + 4 * (out[have, _COL["rot1"]] - 1) + rama[have])

    return out


def _free_runs(tree, xyz: np.ndarray, owner: np.ndarray, i: int,
               origin: np.ndarray) -> np.ndarray:
    """How far each of the twelve directions runs before it meets an atom.

    Every atom belonging to another residue within ``FREE_CAP`` is assigned to
    the nearest of the twelve directions, and each direction keeps the closest
    atom assigned to it. The answer saturates at ``FREE_CAP``, so an unobstructed
    direction and a very unobstructed one read the same and the wire stays
    bounded. This is the van der Waals wall as seen from one atom, and it is a
    quantity the centroid representation cannot state at all: a centroid has no
    directions.
    """
    idx = [j for j in tree.query_ball_point(origin, FREE_CAP) if owner[j] != i]
    free = np.full(len(DIRECTIONS), FREE_CAP)
    if not idx:
        return free
    v = xyz[idx] - origin
    dist = np.linalg.norm(v, axis=1)
    good = dist > 1e-9
    if not good.any():
        return free
    v, dist = v[good], dist[good]
    nearest = np.argmax((v / dist[:, None]) @ DIRECTIONS.T, axis=1)
    np.minimum.at(free, nearest, dist)
    return np.minimum(free, FREE_CAP)


def consistency(x: np.ndarray) -> list[str]:
    """Facts the array must satisfy whatever the input, so a breach is a bug.

    These are properties of the definitions rather than of any protein, which is
    the standard ``backbone_geometry`` set after two of its three first
    quantities were wrong and neither raised. A number whose correct value is
    known before the module runs is the only check that catches a sign error.
    """
    bad: list[str] = []
    for name in ("cos_chi1", "sin_chi1", "cos_chi2", "sin_chi2",
                 "cos_chi3", "sin_chi3", "cos_chi4", "sin_chi4",
                 "sc_radial", "term_radial", "sc_curl",
                 "cos_chi1_minus_phi", "sin_chi1_minus_phi",
                 "cos_chi1_minus_psi", "sin_chi1_minus_psi"):
        v = x[:, _COL[name]]
        if np.any(np.abs(v) > 1 + 1e-9):
            bad.append(f"{name} leaves [-1, 1]")
    for name in ("rot1", "rot2"):
        v = x[:, _COL[name]]
        if np.any((v < 0) | (v > 3)):
            bad.append(f"{name} is not one of the four rotamer labels")
    for name in ("sc_atoms_observed", "sc_atoms_expected", "sc_atoms_missing",
                 "n_chi_defined", "n_non_rotameric", "sc_polar_atoms",
                 "atoms_near_sc_close", "atoms_near_sc_wide",
                 "open_cones_term", "open_cones_cb", "through_holes",
                 "sc_clashes", "sc_polar_buried", "sc_polar_unsatisfied"):
        v = x[:, _COL[name]]
        if np.any(v < 0):
            bad.append(f"{name} is negative")
    if np.any(x[:, _COL["sc_atoms_observed"]] > x[:, _COL["sc_atoms_expected"]]):
        bad.append("more side-chain atoms observed than the residue has")
    if np.any(x[:, _COL["n_chi_defined"]] > 4):
        bad.append("more than four chi angles")
    if np.any(x[:, _COL["sc_extension_ratio"]] > 1.0 + 1e-6):
        bad.append("an end-to-end distance longer than its own contour")
    if np.any(x[:, _COL["sc_polar_unsatisfied"]]
              > x[:, _COL["sc_polar_atoms"]]):
        bad.append("more unsatisfied polar atoms than the residue has")
    if np.any(np.abs(x[:, _COL["ca_chirality"]]) > 1):
        bad.append("ca_chirality is not a sign")
    if not np.isfinite(x).all():
        bad.append("non-finite value")
    return bad
