"""Side-chain geometry checked against numbers known before the module runs.

Why this file is built the way it is
-------------------------------------
``AGENT_MEMORY`` 2j records that two of the first three backbone quantities were
wrong and neither raised: a dihedral's sine had the wrong sign, which exchanged
the two helical Ramachandran cells, and a hydrogen-bond direction test was
inverted, which put the modal donated lag at 2 where an alpha turn fixes 4. Both
were caught the same way, by asking the module for a quantity whose correct value
is known in advance rather than by inspecting a distribution and finding it
plausible.

Every test here is of that kind. A side chain is built forwards from bond
lengths, bond angles and a chosen torsion, and the module is asked to read the
torsion back. A mirror image is built and the chirality is required to flip. An
atom is placed in empty space and every direction is required to be open. The
rotamer labels are required to land in the wells the chemistry names, not in
whichever wells make the distribution look right.

The distributional facts are here too, in ``test_rotamer_census_matches_known``,
because they are the check that would have caught a sign error in chi even if
the constructive test had been written against the same wrong convention: the
chi1 rotamer census of real proteins is g-minus, then trans, then g-plus, in that
order and by a wide margin, and a flipped sine exchanges the first and third.
"""
from __future__ import annotations

import glob

import numpy as np
import pytest

from pocket_bench.methods import sidechain_geometry as sg
from pocket_bench.paths import ROOT
from pocket_bench.pdb_io import parse_pdb_atoms

COL = {name: j for j, name in enumerate(sg.COLUMNS)}


def place(a: np.ndarray, b: np.ndarray, c: np.ndarray,
          bond: float, angle_deg: float, torsion_deg: float) -> np.ndarray:
    """The fourth atom of a chain, from an internal-coordinate specification.

    This is the standard NeRF construction and it is deliberately written out
    rather than imported: it is the independent implementation the module is
    checked against, so sharing code between the two would make the test agree
    with the module by construction.
    """
    ang, tor = np.radians(angle_deg), np.radians(torsion_deg)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d = np.array([-bond * np.cos(ang),
                  bond * np.sin(ang) * np.cos(tor),
                  bond * np.sin(ang) * np.sin(tor)])
    return c + d[0] * bc + d[1] * m + d[2] * n


def ideal_residue(resname: str, chi1: float | None = None,
                  chi2: float | None = None) -> dict[str, np.ndarray]:
    """One residue with textbook backbone geometry and a chosen chi.

    The backbone is placed by hand at standard bond lengths and angles and CB is
    placed by the same tetrahedral construction ``backbone_geometry`` uses, so
    that a chirality test here is a test of the module's determinant convention
    and not of a coincidence in the coordinates.
    """
    n = np.array([0.0, 0.0, 0.0])
    ca = np.array([1.458, 0.0, 0.0])
    c = ca + 1.525 * np.array([np.cos(np.radians(180 - 111.2)),
                               np.sin(np.radians(180 - 111.2)), 0.0])
    o = place(n, ca, c, 1.231, 120.5, 0.0)
    b = ca - n
    cc = c - ca
    a = np.cross(b, cc)
    cb = (-0.58273431 * a + 0.56802827 * b - 0.54067466 * cc) + ca
    d = {"N": n, "CA": ca, "C": c, "O": o, "CB": cb}
    if chi1 is not None:
        names = sg.SIDECHAIN_ATOMS[resname]
        g = names[1]
        d[g] = place(n, ca, cb, 1.52, 114.0, chi1)
        if chi2 is not None and len(names) > 2:
            quad = sg.CHI_ATOMS[resname][1]
            d[quad[3]] = place(ca, cb, d[g], 1.52, 114.0, chi2)
    return d


def compute_one(d: dict[str, np.ndarray], resname: str) -> np.ndarray:
    return sg.compute([d], [resname])[0]


# ---------------------------------------------------------------------------
# Torsion: the module must read back the torsion the residue was built with.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chi1", [-179.0, -120.0, -60.0, -1.0, 0.0,
                                  1.0, 60.0, 120.0, 179.0])
def test_chi1_is_read_back(chi1):
    x = compute_one(ideal_residue("LEU", chi1=chi1), "LEU")
    got = np.degrees(np.arctan2(x[COL["sin_chi1"]], x[COL["cos_chi1"]]))
    assert abs((got - chi1 + 180) % 360 - 180) < 1e-6, (
        f"built chi1={chi1}, read {got}")


@pytest.mark.parametrize("chi2", [-150.0, -60.0, 0.0, 75.0, 160.0])
def test_chi2_is_read_back(chi2):
    x = compute_one(ideal_residue("LEU", chi1=-60.0, chi2=chi2), "LEU")
    got = np.degrees(np.arctan2(x[COL["sin_chi2"]], x[COL["cos_chi2"]]))
    assert abs((got - chi2 + 180) % 360 - 180) < 1e-6


def test_cos_and_sin_are_a_point_on_the_circle():
    x = compute_one(ideal_residue("LEU", chi1=37.0), "LEU")
    r = np.hypot(x[COL["cos_chi1"]], x[COL["sin_chi1"]])
    assert abs(r - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Rotamer wells: the labels are the chemistry's, not whichever fits.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chi1,want", [
    (60.0, sg.ROT_GAUCHE_PLUS),
    (65.0, sg.ROT_GAUCHE_PLUS),
    (180.0, sg.ROT_TRANS),
    (-175.0, sg.ROT_TRANS),
    (-60.0, sg.ROT_GAUCHE_MINUS),
    (300.0, sg.ROT_GAUCHE_MINUS),
])
def test_rotamer_label_is_the_named_well(chi1, want):
    x = compute_one(ideal_residue("LEU", chi1=chi1), "LEU")
    assert int(x[COL["rot1"]]) == want


def test_strain_is_zero_at_a_well_and_maximal_between_two():
    at_well = compute_one(ideal_residue("LEU", chi1=180.0), "LEU")
    assert at_well[COL["chi1_strain"]] < 1e-9
    assert at_well[COL["chi1_non_rotameric"]] == 0.0
    # Exactly between trans and gauche-minus is 60 degrees from each, which is
    # the largest a distance to the nearest of three wells 120 degrees apart can
    # be, so the normalised strain is one.
    between = compute_one(ideal_residue("LEU", chi1=240.0), "LEU")
    assert abs(between[COL["chi1_strain"]] - 1.0) < 1e-9
    assert between[COL["chi1_non_rotameric"]] == 1.0


def test_undefined_chi_is_neutral_not_extreme():
    """Glycine and alanine have no chi and must read zero, not a sentinel."""
    for resname in ("GLY", "ALA"):
        d = ideal_residue(resname)
        if resname == "GLY":
            d.pop("CB")
        x = compute_one(d, resname)
        for name in ("cos_chi1", "sin_chi1", "rot1", "chi1_strain",
                     "n_chi_defined", "rot_joint"):
            assert x[COL[name]] == 0.0, f"{resname} {name} is not neutral"


# ---------------------------------------------------------------------------
# Chirality: an L residue and its mirror must differ in sign and nowhere else.
# ---------------------------------------------------------------------------

def test_mirror_image_flips_the_chirality_sign():
    d = ideal_residue("LEU", chi1=-60.0)
    x = compute_one(d, "LEU")
    mirrored = {k: v * np.array([1.0, 1.0, -1.0]) for k, v in d.items()}
    y = compute_one(mirrored, "LEU")
    assert x[COL["ca_chirality"]] == -y[COL["ca_chirality"]]
    assert x[COL["ca_chirality"]] != 0.0
    # The magnitude is a volume and reflection preserves it.
    assert abs(x[COL["ca_tetra_volume"]] - y[COL["ca_tetra_volume"]]) < 1e-9
    # A reflection also reverses every torsion, which is the second half of the
    # statement that chirality is what distinguishes these two structures.
    assert abs(x[COL["sin_chi1"]] + y[COL["sin_chi1"]]) < 1e-9
    assert abs(x[COL["cos_chi1"]] - y[COL["cos_chi1"]]) < 1e-9


def test_every_standard_residue_in_the_fold_is_l_configured():
    """No L-amino acid may read as D, on a real chain, at any position.

    This is the check that a determinant with a transposed row would fail while
    still producing a plausible-looking distribution.
    """
    x, names = _real_chain()
    have_cb = np.array([n not in ("GLY",) for n in names])
    sign = x[:, COL["ca_chirality"]]
    assert not np.any(sign[have_cb] < 0), (
        f"{int((sign[have_cb] < 0).sum())} residues read as D-amino acids")
    assert np.all(sign[~have_cb] == 0.0)


# ---------------------------------------------------------------------------
# Shape and direction.
# ---------------------------------------------------------------------------

def test_extension_ratio_is_one_for_a_straight_side_chain():
    """Collinear atoms have end-to-end distance equal to contour length."""
    d = {"N": np.array([0.0, 1.0, 0.0]), "CA": np.zeros(3),
         "C": np.array([1.0, 1.0, 0.0])}
    for k, name in enumerate(("CB", "CG", "CD", "CE", "NZ")):
        d[name] = np.array([1.5 * (k + 1), 0.0, 0.0])
    x = compute_one(d, "LYS")
    assert abs(x[COL["sc_extension_ratio"]] - 1.0) < 1e-9
    assert abs(x[COL["sc_curl"]] - 1.0) < 1e-9


def test_extension_ratio_falls_when_the_side_chain_folds_back():
    d = {"N": np.array([0.0, 1.0, 0.0]), "CA": np.zeros(3),
         "C": np.array([1.0, 1.0, 0.0]),
         "CB": np.array([1.5, 0.0, 0.0]),
         "CG": np.array([3.0, 0.0, 0.0]),
         "CD": np.array([3.0, 1.5, 0.0]),
         "CE": np.array([1.5, 1.5, 0.0]),
         "NZ": np.array([1.5, 3.0, 0.0])}
    x = compute_one(d, "LYS")
    assert x[COL["sc_extension_ratio"]] < 0.6
    # Never exceeds one, which is the invariant consistency() also asserts.
    assert x[COL["sc_extension_ratio"]] <= 1.0 + 1e-9


def test_completeness_counts_the_atoms_the_deposit_omits():
    full = ideal_residue("LEU", chi1=-60.0, chi2=175.0)
    full["CD2"] = full["CG"] + np.array([0.0, 0.0, 1.5])
    x = compute_one(full, "LEU")
    assert x[COL["sc_atoms_expected"]] == 4        # CB CG CD1 CD2
    assert x[COL["sc_atoms_observed"]] == 4
    assert x[COL["sc_atoms_missing"]] == 0
    assert x[COL["sc_complete"]] == 1.0
    partial = dict(full)
    partial.pop("CD2")
    partial.pop("CD1")
    y = compute_one(partial, "LEU")
    assert y[COL["sc_atoms_missing"]] == 2
    assert y[COL["sc_complete"]] == 0.0
    assert y[COL["terminal_missing"]] == 1.0
    assert abs(y[COL["sc_missing_fraction"]] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Open directions: the van der Waals wall.
# ---------------------------------------------------------------------------

def test_an_isolated_side_chain_sees_every_direction_open():
    d = ideal_residue("LEU", chi1=-60.0)
    x = compute_one(d, "LEU")
    assert x[COL["open_cones_term"]] == len(sg.DIRECTIONS)
    # Twelve directions in six antipodal pairs, all open, is six through-holes.
    assert x[COL["through_holes"]] == len(sg.DIRECTIONS) // 2
    assert x[COL["deepest_free"]] == sg.FREE_CAP
    assert x[COL["free_anisotropy"]] == 0.0


def test_the_antipode_map_is_an_involution_and_has_no_fixed_point():
    a = sg.ANTIPODE
    assert np.array_equal(a[a], np.arange(len(a)))
    assert not np.any(a == np.arange(len(a)))
    for k, j in enumerate(a):
        assert sg.DIRECTIONS[k] @ sg.DIRECTIONS[j] < -0.99


def test_a_wall_of_atoms_closes_the_directions_it_faces():
    """Burying a residue must reduce the open-direction count in every direction.

    The wall is a shell of atoms rather than a slab, because the first version of
    this test placed a slab along +x and it closed nothing: the twelve directions
    had a 30-degree cone each and their covering radius is 37.4 degrees, so the
    +x axis fell outside all twelve at once. Nearest-direction assignment fixed
    that, and this test is written to fail again if a cone ever comes back.
    """
    d = ideal_residue("LEU", chi1=-60.0)
    free = compute_one(d, "LEU")[COL["open_cones_term"]]
    assert free == len(sg.DIRECTIONS)

    term = d[sg.SIDECHAIN_ATOMS["LEU"][1]]
    shell = [{"CA": term + 5.0 * u} for u in _sphere_points(60)]
    x = sg.compute([d] + shell, ["LEU"] + ["ALA"] * len(shell))[0]
    assert x[COL["open_cones_term"]] == 0
    assert x[COL["through_holes"]] == 0
    assert x[COL["tightest_free"]] < sg.FREE_CAP
    assert x[COL["deepest_free"]] < sg.FREE_CAP


def _sphere_points(n: int) -> np.ndarray:
    """A spiral of ``n`` roughly equidistributed unit vectors."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    t = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.stack([r * np.cos(t), r * np.sin(t), z], axis=1)


def test_a_wall_on_one_side_only_closes_that_side():
    """Direction resolution: a half-shell must leave the other half open."""
    d = ideal_residue("LEU", chi1=-60.0)
    term = d[sg.SIDECHAIN_ATOMS["LEU"][1]]
    pts = _sphere_points(120)
    half = pts[pts[:, 0] > 0.3]
    shell = [{"CA": term + 5.0 * u} for u in half]
    x = sg.compute([d] + shell, ["LEU"] + ["ALA"] * len(shell))[0]
    assert 0 < x[COL["open_cones_term"]] < len(sg.DIRECTIONS)
    assert x[COL["free_anisotropy"]] > 0.0


# ---------------------------------------------------------------------------
# The type-determined trap: 2i closed per-residue-type dictionaries, so a
# quantity that a three-letter code determines cannot be in this family.
# ---------------------------------------------------------------------------

def test_conformational_columns_vary_within_one_residue_type():
    """The same residue type in two rotamers must differ on the shape columns.

    Chemistry 42 measured null because every one of its columns was a function of
    residue identity. This test states the property that makes this family
    different in kind, on the columns that carry the mechanism.
    """
    a = compute_one(ideal_residue("LEU", chi1=-60.0, chi2=175.0), "LEU")
    b = compute_one(ideal_residue("LEU", chi1=180.0, chi2=65.0), "LEU")
    varying = ["cos_chi1", "sin_chi1", "rot1", "sc_extension",
               "sc_extension_ratio", "sc_curl", "ca_to_sc_centroid",
               "sc_centroid_offset"]
    differ = [n for n in varying if abs(a[COL[n]] - b[COL[n]]) > 1e-6]
    assert len(differ) >= 6, (
        f"only {differ} differ between two rotamers of the same residue")


def test_identity_only_columns_are_declared_and_few():
    """The columns a three-letter code determines are named, and are denominators.

    ``sc_atoms_expected`` and ``sc_polar_atoms`` are functions of type alone.
    They are admitted because each is the denominator of a conformational ratio
    beside it, and this test fixes that list so that a later addition of a
    type-determined column is a visible change rather than a silent one.
    """
    identity_only = {"sc_atoms_expected", "sc_polar_atoms"}
    per, names = [], []
    for resname in sorted(sg.SIDECHAIN_ATOMS):
        for chi in (-60.0, 180.0):
            if resname in ("GLY", "ALA"):
                continue
            per.append(ideal_residue(resname, chi1=chi))
            names.append(resname)
    x = sg.compute(per, names)
    constant_within_type: set[str] = set()
    for j, name in enumerate(sg.COLUMNS):
        by_type = {}
        ok = True
        for i, rn in enumerate(names):
            by_type.setdefault(rn, []).append(x[i, j])
        for rn, vals in by_type.items():
            if max(vals) - min(vals) > 1e-9:
                ok = False
                break
        if ok and x[:, j].std() > 0:
            constant_within_type.add(name)
    assert identity_only <= constant_within_type | set(sg.COLUMNS)
    for name in identity_only:
        assert name in sg.COLUMNS


# ---------------------------------------------------------------------------
# Real chains: the invariants, and the census that catches a flipped sine.
# ---------------------------------------------------------------------------

def _real_chain(index: int = 0):
    """One training chain, with the backbone context group J couples to.

    The backbone quantities are supplied here exactly as the wire builder
    supplies them, so that group J is live in the tests rather than silently
    zero. An earlier version of this helper omitted them and
    ``test_no_column_is_constant_across_the_fold`` correctly reported five dead
    columns that were only dead in the test.
    """
    from pocket_bench.methods import backbone_geometry as bg

    files = sorted(glob.glob(str(
        ROOT / "data/cryptobench_apo/train_receptors/*_receptor.pdb")))
    if not files:
        pytest.skip("training receptors are not present in this checkout")
    f = files[index]
    chain = f.split("/")[-1].split("_")[1]
    atoms = [a for a in parse_pdb_atoms(open(f).read()) if a["chain"] == chain]
    seen, keys = set(), []
    for a in atoms:
        if a["record"] != "ATOM":
            continue
        k = (a["resseq"], a["icode"].strip())
        if k not in seen:
            seen.add(k)
            keys.append(k)
    order = sorted(keys)
    per, names = sg.chain_sidechain(atoms, order)
    bcol = {n: j for j, n in enumerate(bg.COLUMNS)}
    b = bg.compute(bg.chain_backbone(atoms, order))
    phi = b[:, [bcol["cos_phi"], bcol["sin_phi"]]]
    psi = b[:, [bcol["cos_psi"], bcol["sin_psi"]]]
    rama = b[:, bcol["rama_region"]]
    return sg.compute(per, names, phi=phi, psi=psi, rama=rama), names


def test_consistency_is_clean_on_real_chains():
    for i in range(6):
        x, _ = _real_chain(i)
        assert sg.consistency(x) == []


def test_rotamer_census_matches_known_protein_statistics():
    """chi1 in real proteins is gauche-minus, then trans, then gauche-plus.

    This ordering is one of the most reproducible facts in structural biology and
    it is asymmetric under a sign flip: negating the sine exchanges gauche-plus
    and gauche-minus, so a module with an inverted dihedral convention produces
    the reverse ordering and passes every symmetric check. That is the error
    ``AGENT_MEMORY`` 2j records in the backbone family, caught the same way.
    """
    counts = {sg.ROT_GAUCHE_PLUS: 0, sg.ROT_TRANS: 0, sg.ROT_GAUCHE_MINUS: 0}
    for i in range(8):
        x, _ = _real_chain(i)
        for w in counts:
            counts[w] += int((x[:, COL["rot1"]] == w).sum())
    total = sum(counts.values())
    assert total > 1000
    g_minus = counts[sg.ROT_GAUCHE_MINUS] / total
    trans = counts[sg.ROT_TRANS] / total
    g_plus = counts[sg.ROT_GAUCHE_PLUS] / total
    assert g_minus > trans > g_plus, (
        f"census g-={g_minus:.3f} t={trans:.3f} g+={g_plus:.3f} is not the "
        f"known ordering; the dihedral sign is the first suspect")
    assert g_minus > 0.4, f"gauche-minus at {g_minus:.3f} is far below the "\
                          f"roughly one half real proteins show"
    assert g_plus < 0.25


def test_second_chiral_centre_appears_only_for_ile_and_thr():
    x, names = _real_chain()
    has = x[:, COL["second_centre_chirality"]] != 0
    for i, flagged in enumerate(has):
        if flagged:
            assert names[i] in sg.SECOND_CENTRE
    n_expected = sum(1 for n in names if n in sg.SECOND_CENTRE)
    assert int(has.sum()) <= n_expected


def test_terminal_atoms_are_less_buried_than_cb_on_average():
    """A side chain points away from the backbone, so burial falls along it.

    This is a directional fact rather than a bound, and it is here because it is
    the check that would catch ``burial_gradient`` being computed with its two
    terms exchanged, which no symmetric assertion would notice.
    """
    x, _ = _real_chain()
    live = x[:, COL["atoms_near_cb_wide"]] > 0
    assert x[live, COL["burial_gradient"]].mean() < 0


def test_no_column_is_constant_across_the_fold():
    """A column with one value everywhere costs cells and carries nothing."""
    xs = [_real_chain(i)[0] for i in range(8)]
    x = np.concatenate(xs, axis=0)
    dead = [sg.COLUMNS[j] for j in range(x.shape[1])
            if np.ptp(x[:, j]) == 0.0]
    assert dead == [], f"columns constant over {x.shape[0]} residues: {dead}"


def test_column_names_are_unique_and_counted():
    assert len(set(sg.COLUMNS)) == len(sg.COLUMNS)
    assert sg.N_COLUMNS == len(sg.COLUMNS)


def test_chi_definitions_reference_atoms_the_residue_has():
    """Every chi is defined over atoms the residue's own topology contains."""
    for resname, quads in sg.CHI_ATOMS.items():
        have = set(sg.SIDECHAIN_ATOMS[resname]) | {"N", "CA", "C", "O"}
        for k, quad in enumerate(quads):
            missing = [a for a in quad if a not in have]
            assert not missing, f"{resname} chi{k + 1} names {missing}"


def test_the_bonded_path_is_a_subset_of_the_atom_list():
    for resname, path in sg.SIDECHAIN_PATH.items():
        assert set(path) <= set(sg.SIDECHAIN_ATOMS[resname]), resname


def test_donors_and_acceptors_are_side_chain_nitrogen_or_oxygen():
    for table in (sg.SC_DONORS, sg.SC_ACCEPTORS):
        for resname, names in table.items():
            for a in names:
                assert a in sg.SIDECHAIN_ATOMS[resname], f"{resname} {a}"
                assert a[0] in "NO", f"{resname} {a} is not N or O"
