"""Per-residue chemical principles, as integers, and why each one is here.

What this module is for
-----------------------
The 645 deployed wires are geometric invariants of atom positions. They are
very good at describing the shape a structure currently has, and they contain
nothing at all about what it could become. A *cryptic* pocket is defined by the
second thing: the site is closed in the apo structure and the label says it
opens. Every quantity below is chosen because it bears on whether a
neighbourhood can rearrange, or on what it would present if it did, and none of
it is derivable from coordinates alone -- it comes from knowing which residue
sits at each position.

``composition_wires.py`` already reads residue identity, and reads it as eight
crude classes: aliphatic, aromatic, polar, positive, negative, and then
glycine, proline and cysteine on their own. That partition throws away most of
what a chemist knows about a side chain. Lysine and arginine are both
"positive", but arginine's guanidinium donates five hydrogen bonds in a planar
bidentate arrangement and lysine's ammonium donates three from a point;
tryptophan and phenylalanine are both "aromatic", but one carries an indole NH
and nine ring atoms and the other carries neither. The eight classes are labels.
What follows are quantities.

Why quantities and not more classes
-----------------------------------
``docs/AGENT_MEMORY.md`` 2c records the one design rule this repository has that
was measured rather than argued: a new column family is worth building only if
its members are *different quantities*. Reading one operator at several radii
produces wires whose joint says less than their marginals added -- the
same-quantity pairs of the deployed bus carry -6.29e-05 mean interaction, the
only negative category there is -- so a parameter sweep of one good operator is
not collectible however good the operator is. Ten different properties of a side
chain are ten quantities. Ten radii of one property are one.

The quantities, and the reasoning behind each
---------------------------------------------
``chi_rotatable``  The number of side-chain dihedrals with rotameric freedom, by
    the standard rotamer-library convention. This is the module's central
    quantity and the one nothing in the deployed bus can see. A pocket that is
    cryptic must open, and side chains open it: a neighbourhood of glycine,
    alanine and proline has no conformational budget to open with, and one of
    lysine, arginine, methionine and glutamate has a great deal. Proline is
    zero because the pyrrolidine ring locks chi1 and chi2, and alanine is zero
    because a methyl's rotation produces no distinct rotamer.

``sc_hbd`` and ``sc_hba``  Donatable hydrogens and acceptor atoms on the side
    chain only, backbone excluded, because the backbone contributes the same
    amide donor and carbonyl acceptor at every position and adds nothing that
    distinguishes one site from another. Counted separately rather than summed
    into "polar": a surface that can donate and a surface that can accept
    complement different ligands, and the difference is exactly what a
    pharmacophore model is about.

``formal_charge``  At pH 7.4, so aspartate and glutamate are -1, lysine and
    arginine +1, and histidine is 0. Histidine's side-chain pKa near 6.0 means
    it is mostly neutral at physiological pH; calling it "positive", which the
    eight-class partition does, is wrong about the majority species. Its
    switchability is recorded separately.

``ph_switchable``  Histidine alone. It is the one residue whose charge state
    changes across the physiological range, which makes it the usual hinge of
    pH-dependent conformational change and a frequent metal ligand.

``aromatic_ring_atoms``  Ring atoms of the side chain: 6 for phenylalanine and
    tyrosine, 9 for tryptophan's indole, 5 for histidine's imidazole. Aromatic
    side chains are the classic gate of a cryptic site -- a phenylalanine
    rotating out of the way is one of the commonest opening mechanisms -- and
    the count distinguishes the size of the gate, which the class label does
    not.

``sc_carbon`` and ``sc_polar_atoms``  Heavy-atom composition of the side chain,
    carbon against N, O and S. Their ratio is a hydrophobicity that needs no
    fitted scale, and both are exact integers read off a structural formula
    rather than a published scale with a citation and a choice of which
    publication.

``beta_branched``  Valine, isoleucine and threonine, which branch at C-beta.
    They restrict the backbone conformations available to their own residue and
    their neighbours, so they act against the flexibility that ``chi_rotatable``
    measures, and a neighbourhood can be high in both.

``backbone_flexible`` and ``backbone_constrained``  Glycine and proline. Glycine
    has no side chain and the widest Ramachandran freedom of any residue;
    proline has no amide NH and its ring fixes phi. They are the two residues
    whose effect is on the *backbone* rather than the side chain, which is why
    the eight-class partition already gives each its own class, and here they
    stay separate for the same reason but as counts of a property rather than
    as an identity.

``metal_ligating``  Histidine, cysteine, aspartate, glutamate and methionine:
    the side chains that coordinate a divalent metal. Not a claim that a metal
    is present; a count of the capability, which is a property of the
    neighbourhood a ligand or an ion would encounter.

``nucleophilic``  Cysteine's thiolate above all, then serine and threonine
    hydroxyls and lysine's amine. This is where a covalent ligand attaches, and
    a cryptic pocket lined with one is a different proposition from one that is
    not.

``sc_volume``  Zamyatnin side-chain volume in cubic angstroms, rounded to an
    integer. Included because the space a side chain vacates when it rotates is
    the space a ligand occupies, and because it is the one quantity here that
    is a measured constant rather than a count. The literature value is used
    unmodified and the citation travels with it.

What is deliberately absent
---------------------------
No hydropathy scale beyond the two composition counts, because there are a
dozen published scales that disagree and choosing one is choosing a fitted
quantity with a citation attached. No secondary-structure assignment, because
that is an algorithm's output rather than a property of the residue. No
conservation, because this repository computes no alignment and says so.

Nothing here reads a label, a fold assignment or a structure's coordinates;
these are constants of the twenty amino acids.
"""
from __future__ import annotations

import numpy as np

# The canonical order used by the expanded cache's ``codes`` column. Any change
# here silently re-indexes every table below, so it is asserted against the
# cache's own ordering by the builder that consumes this module.
AA20 = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)

# Side-chain dihedrals with rotameric freedom, standard rotamer-library
# convention. ALA is 0 because a methyl rotation gives no distinct rotamer and
# PRO is 0 because the pyrrolidine ring locks chi1 and chi2.
CHI_ROTATABLE = {
    "GLY": 0, "ALA": 0, "PRO": 0,
    "SER": 1, "CYS": 1, "THR": 1, "VAL": 1,
    "ILE": 2, "LEU": 2, "ASP": 2, "ASN": 2, "HIS": 2, "PHE": 2, "TYR": 2,
    "TRP": 2,
    "MET": 3, "GLU": 3, "GLN": 3,
    "LYS": 4, "ARG": 4,
}

# Donatable hydrogens on the side chain at pH 7.4. ARG is 5: one on NE and two
# on each of NH1 and NH2. HIS is 1 because the neutral imidazole carries its
# proton on one ring nitrogen at a time. ASP and GLU are 0 because they are
# deprotonated.
SC_HBD = {
    "GLY": 0, "ALA": 0, "VAL": 0, "LEU": 0, "ILE": 0, "PRO": 0, "PHE": 0,
    "MET": 0, "ASP": 0, "GLU": 0,
    "SER": 1, "THR": 1, "TYR": 1, "CYS": 1, "HIS": 1, "TRP": 1,
    "ASN": 2, "GLN": 2,
    "LYS": 3,
    "ARG": 5,
}

# Side-chain atoms with a lone pair available to accept. TRP is 0: the indole
# nitrogen is a donor and its lone pair is in the aromatic system. CYS and MET
# sulfur are counted, weakly but really.
SC_HBA = {
    "GLY": 0, "ALA": 0, "VAL": 0, "LEU": 0, "ILE": 0, "PRO": 0, "PHE": 0,
    "TRP": 0, "LYS": 0, "ARG": 0,
    "SER": 1, "THR": 1, "TYR": 1, "CYS": 1, "MET": 1, "ASN": 1, "GLN": 1,
    "HIS": 1,
    "ASP": 2, "GLU": 2,
}

# Formal charge of the side chain at pH 7.4. HIS is 0: its pKa near 6.0 leaves
# it mostly neutral, and the eight-class partition calling it positive is wrong
# about the majority species.
FORMAL_CHARGE = {a: 0 for a in AA20}
FORMAL_CHARGE.update({"ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1})

# The one residue whose charge state moves across the physiological range.
PH_SWITCHABLE = {a: 0 for a in AA20}
PH_SWITCHABLE["HIS"] = 1

# Ring atoms of the side chain: benzene 6, phenol 6, indole 9, imidazole 5.
AROMATIC_RING_ATOMS = {a: 0 for a in AA20}
AROMATIC_RING_ATOMS.update({"PHE": 6, "TYR": 6, "TRP": 9, "HIS": 5})

# Heavy atoms of the side chain, counted off the structural formula.
SC_CARBON = {
    "GLY": 0, "ALA": 1, "SER": 1, "CYS": 1,
    "THR": 2, "ASN": 2, "ASP": 2,
    "VAL": 3, "PRO": 3, "MET": 3, "GLN": 3, "GLU": 3,
    "LEU": 4, "ILE": 4, "LYS": 4, "HIS": 4, "ARG": 4,
    "PHE": 7, "TYR": 7,
    "TRP": 9,
}
SC_POLAR_ATOMS = {
    "GLY": 0, "ALA": 0, "VAL": 0, "LEU": 0, "ILE": 0, "PRO": 0, "PHE": 0,
    "SER": 1, "CYS": 1, "THR": 1, "TYR": 1, "MET": 1, "TRP": 1, "LYS": 1,
    "ASN": 2, "ASP": 2, "GLN": 2, "GLU": 2, "HIS": 2,
    "ARG": 3,
}

# Branching at C-beta, which restricts backbone conformation.
BETA_BRANCHED = {a: 0 for a in AA20}
BETA_BRANCHED.update({"VAL": 1, "ILE": 1, "THR": 1})

BACKBONE_FLEXIBLE = {a: 0 for a in AA20}
BACKBONE_FLEXIBLE["GLY"] = 1
BACKBONE_CONSTRAINED = {a: 0 for a in AA20}
BACKBONE_CONSTRAINED["PRO"] = 1

# Side chains that coordinate a divalent metal. A capability, not a claim that
# a metal is present.
METAL_LIGATING = {a: 0 for a in AA20}
METAL_LIGATING.update({"HIS": 1, "CYS": 1, "ASP": 1, "GLU": 1, "MET": 1})

# Where a covalent ligand attaches. Cysteine is 2 rather than 1 because the
# thiolate is the one nucleophile in a protein that is routinely targeted, and
# the ordering between it and a serine hydroxyl is the whole point of the
# quantity.
NUCLEOPHILIC = {a: 0 for a in AA20}
NUCLEOPHILIC.update({"CYS": 2, "SER": 1, "THR": 1, "LYS": 1, "TYR": 1})

# Zamyatnin side-chain volume, cubic angstroms, Prog Biophys Mol Biol 24:107
# (1974), rounded to an integer. The one measured constant here.
SC_VOLUME = {
    "GLY": 60, "ALA": 89, "SER": 89, "CYS": 109, "ASP": 111, "PRO": 113,
    "ASN": 114, "THR": 116, "GLU": 138, "VAL": 140, "GLN": 144, "HIS": 153,
    "MET": 163, "ILE": 167, "LEU": 167, "LYS": 169, "ARG": 173, "PHE": 190,
    "TYR": 194, "TRP": 228,
}

# Order fixed here so that a column's meaning does not depend on dict ordering
# elsewhere. Adding a property appends; it must never insert.
PROPERTIES: tuple[tuple[str, dict], ...] = (
    ("chi_rotatable", CHI_ROTATABLE),
    ("sc_hbd", SC_HBD),
    ("sc_hba", SC_HBA),
    ("formal_charge", FORMAL_CHARGE),
    ("ph_switchable", PH_SWITCHABLE),
    ("aromatic_ring_atoms", AROMATIC_RING_ATOMS),
    ("sc_carbon", SC_CARBON),
    ("sc_polar_atoms", SC_POLAR_ATOMS),
    ("beta_branched", BETA_BRANCHED),
    ("backbone_flexible", BACKBONE_FLEXIBLE),
    ("backbone_constrained", BACKBONE_CONSTRAINED),
    ("metal_ligating", METAL_LIGATING),
    ("nucleophilic", NUCLEOPHILIC),
    ("sc_volume", SC_VOLUME),
)


def property_names() -> tuple[str, ...]:
    return tuple(name for name, _ in PROPERTIES)


def table() -> np.ndarray:
    """``(20, n_properties)`` integer table, rows in ``AA20`` order.

    Every table is required to name all twenty residues. A missing key would
    otherwise become a silent zero, which for ``chi_rotatable`` would read as
    "this side chain cannot move" -- the strongest statement the module makes,
    arrived at by omission.
    """
    out = np.zeros((len(AA20), len(PROPERTIES)), dtype=np.int64)
    for j, (name, d) in enumerate(PROPERTIES):
        missing = [a for a in AA20 if a not in d]
        if missing:
            raise SystemExit(f"{name} does not name {missing}; a missing "
                             f"residue becomes a silent zero")
        extra = [k for k in d if k not in AA20]
        if extra:
            raise SystemExit(f"{name} names {extra}, which are not among the "
                             f"twenty")
        for i, a in enumerate(AA20):
            out[i, j] = int(d[a])
    return out


def consistency() -> dict:
    """Relations between the tables that must hold, checked rather than assumed.

    These are not tests of the constants against the literature -- that is what
    reading them is for -- but of the tables against each other, which catches
    the failure mode where one table is edited and its neighbour is not.
    """
    t = table()
    idx = {n: i for i, n in enumerate(property_names())}
    aa = {a: i for i, a in enumerate(AA20)}
    checks: dict[str, bool] = {}

    # A residue with aromatic ring atoms has at least one side-chain carbon.
    arom = t[:, idx["aromatic_ring_atoms"]]
    checks["aromatic_residues_have_carbon"] = bool(
        np.all(t[arom > 0, idx["sc_carbon"]] > 0))

    # Charge and pH-switchability are exclusive: histidine is the switchable
    # one and it is neutral, and the charged four do not switch.
    ch = t[:, idx["formal_charge"]]
    sw = t[:, idx["ph_switchable"]]
    checks["charged_residues_do_not_switch"] = bool(
        np.all(sw[ch != 0] == 0))
    checks["only_histidine_switches"] = bool(
        sw.sum() == 1 and sw[aa["HIS"]] == 1)

    # Glycine has no side chain, so every side-chain quantity is zero.
    g = t[aa["GLY"]]
    sc_cols = [idx[n] for n in ("chi_rotatable", "sc_hbd", "sc_hba",
                                "aromatic_ring_atoms", "sc_carbon",
                                "sc_polar_atoms")]
    checks["glycine_has_no_side_chain"] = bool(np.all(g[sc_cols] == 0))

    # Proline is the constrained one and has no rotameric freedom.
    checks["proline_is_locked"] = bool(
        t[aa["PRO"], idx["chi_rotatable"]] == 0
        and t[aa["PRO"], idx["backbone_constrained"]] == 1)

    # Volume orders with heavy-atom count over the residues where both are
    # meaningful: tryptophan is the largest by both and glycine the smallest.
    vol = t[:, idx["sc_volume"]]
    heavy = t[:, idx["sc_carbon"]] + t[:, idx["sc_polar_atoms"]]
    checks["tryptophan_is_largest_by_both"] = bool(
        int(np.argmax(vol)) == aa["TRP"] and int(np.argmax(heavy)) == aa["TRP"])
    checks["glycine_is_smallest_by_both"] = bool(
        int(np.argmin(vol)) == aa["GLY"] and int(np.argmin(heavy)) == aa["GLY"])

    # The quantity the module exists for must actually vary, and vary widely.
    chi = t[:, idx["chi_rotatable"]]
    checks["chi_spans_zero_to_four"] = bool(chi.min() == 0 and chi.max() == 4)

    # Every property must distinguish something. A column of one value is a
    # column that cannot enter a table.
    checks["every_property_varies"] = bool(
        all(len(np.unique(t[:, j])) > 1 for j in range(t.shape[1])))

    return {
        "n_residues": len(AA20),
        "n_properties": len(PROPERTIES),
        "properties": list(property_names()),
        "checks": checks,
        "ok": all(checks.values()),
        "failed": sorted(k for k, v in checks.items() if not v),
    }
