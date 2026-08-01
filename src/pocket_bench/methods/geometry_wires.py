"""The 624 geometry columns for one receptor, from the file rather than a cache.

Why this module has to exist
----------------------------
The four wire families were measured by ``tools/*_wires.py``, and every one of
those builds its matrix by walking the training manifest and indexing the
training wide cache. That is the right shape for measuring a lift on twelve
halvings of the training fold and the wrong shape for scoring anything else: an
external unit is a receptor file with no row in any cache.

So the same quantities are assembled here from a receptor path, in the same order
and under the same three aggregations, so that a field compiled on the training
fold can be applied to a structure the fold has never seen.

The one thing that makes this trustworthy
------------------------------------------
Nothing in the assembly is novel --- every quantity comes from the same
``compute`` in the same module the builders call --- so the only way this can be
wrong is by attaching a correct quantity to the wrong residue. That failure is
silent: every lookup succeeds and every score comes out plausible.

``tests/test_geometry_wires.py`` therefore requires this module to reproduce the
four cached training matrices **exactly**, column for column and row for row, on
real chains. Not to a tolerance --- bit-identical, because both paths run the
same float operations in the same order on the same coordinates, so anything
other than equality means the two are not computing the same thing. That test is
the whole warrant for reading an external set with a field compiled over these
columns.

Layout
------
``COLUMNS`` is the concatenation the measurement used, in that order:

* backbone, 44 quantities, aggregation-major -> 132
* side chain, 87 -> 261
* void topology, 45 -> 135
* displacement, the 32 temperature-factor quantities only -> 96

The displacement family contributes 32 of its 48 quantities because the other 16
were measured and are not there: ``displacement alt 48`` scored +0.00052 against
its own control at +0.00052, the same number to five decimals. Carrying them
would spend 384 tables on a measured zero, and it would also push the block past
the 645 deployed wires, which the attachment harness refuses because a round
pairs each new column with a distinct wire.

The three aggregations are the residue's own value, the sum over residues whose
centroids lie within 8 A, and the sum two steps out on that contact graph. The
radius is the builders' and is held identical so that a comparison against any
family is a comparison of quantities and not of neighbourhoods.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pocket_bench.methods.backbone_geometry import (
    COLUMNS as BB_COLUMNS,
    chain_backbone,
    compute as backbone_compute,
)
from pocket_bench.methods.displacement import (
    COLUMNS as DISP_COLUMNS,
    compute as displacement_compute,
    parse_displacement,
)
from pocket_bench.methods.sidechain_geometry import (
    COLUMNS as SC_COLUMNS,
    chain_sidechain,
    compute as sidechain_compute,
)
from pocket_bench.methods.void_topology import (
    COLUMNS as VOID_COLUMNS,
    chain_voids,
    compute as void_compute,
)
from pocket_bench.pdb_io import parse_pdb_atoms

CONTACT_RADIUS = 8.0
AGGREGATIONS = ("own", "contact", "walk2")
SKIP = frozenset({"HOH", "WAT", "DOD"})

# backbone_geometry exports no index map, so it is derived here the same
# way sidechain_wires derives it, from the column tuple.
BB_IDX = {n: j for j, n in enumerate(BB_COLUMNS)}

# How many of the displacement family's quantities are carried, and which.
# Groups A-D are the temperature-factor quantities and occupy 0..31; E and F are
# the alternate-conformer and occupancy groups and measured null.
N_DISP_CARRIED = 32

FAMILIES = (
    ("backbone", BB_COLUMNS),
    ("sidechain", SC_COLUMNS),
    ("void", VOID_COLUMNS),
    ("displacement", DISP_COLUMNS[:N_DISP_CARRIED]),
)

COLUMNS = tuple(
    f"{fam}~{agg}~{q}"
    for fam, quantities in FAMILIES
    for agg in AGGREGATIONS
    for q in quantities
)
N_COLUMNS = len(COLUMNS)


def residue_rows(atoms: list[dict], chain: str
                 ) -> tuple[list[tuple[int, str]], np.ndarray, np.ndarray]:
    """The polymer's residues, the evaluation universe, and the map between them.

    Identical to the four builders' ``_residue_rows``, and duplicated from them
    for the reason each of those records: they feed pinned artifacts and
    importing from one would put this file inside that artifact's blast radius.
    The duplication is held honest by the equality test rather than by care.
    """
    keep = [a for a in atoms
            if a["chain"] == chain and a["element"] != "H"
            and a["resname"] not in SKIP]
    poly: dict[tuple[int, str], list[tuple[float, float, float]]] = {}
    for a in keep:
        poly.setdefault((a["resseq"], a["icode"].strip()), []).append(
            (a["x"], a["y"], a["z"]))
    order = sorted(poly)

    universe = sorted({r for r, _ic in order})
    first: dict[int, int] = {}
    for j, (r, _ic) in enumerate(order):
        first.setdefault(r, j)
    take = np.array([first[r] for r in universe], dtype=np.int64)

    merged: dict[int, list] = {}
    for (r, _ic), pts in poly.items():
        merged.setdefault(r, []).extend(pts)
    ctr = np.array([np.mean(merged[r], axis=0) for r in universe],
                   dtype=np.float64)
    return order, ctr, take


def aggregate(ctr: np.ndarray, prop: np.ndarray) -> np.ndarray:
    """The three aggregations for one chain, identical to the builders'."""
    d = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = (d <= CONTACT_RADIUS).astype(np.float64)
    two = adj @ adj
    np.fill_diagonal(two, 0.0)
    return np.concatenate([prop, adj @ prop, two @ prop], axis=1)


def raw_quantities(receptor_pdb: str | Path, chain: str
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-residue quantities for one chain, before aggregation.

    Returns ``(resseq, ctr, blocks)`` where ``blocks`` is the four families'
    unaggregated quantities side by side on the evaluation universe, in the order
    ``FAMILIES`` declares.
    """
    path = Path(receptor_pdb)
    text = path.read_text(errors="ignore")
    atoms = parse_pdb_atoms(text)
    order, ctr, take = residue_rows(atoms, chain)
    if len(order) == 0:
        raise ValueError(f"{path.name} chain {chain} has no polymer residues")
    mine = [a for a in atoms if a["chain"] == chain]

    bb = backbone_compute(chain_backbone(mine, order))
    per, names = chain_sidechain(mine, order)
    sc = sidechain_compute(
        per, names,
        phi=bb[:, [BB_IDX["cos_phi"], BB_IDX["sin_phi"]]],
        psi=bb[:, [BB_IDX["cos_psi"], BB_IDX["sin_psi"]]],
        rama=bb[:, BB_IDX["rama_region"]])
    void = void_compute(chain_voids(mine, order))
    disp = displacement_compute(parse_displacement(text, chain), order)

    blocks = np.concatenate(
        [bb[take], sc[take], void[take], disp[take, :N_DISP_CARRIED]], axis=1)
    resseq = np.array([r for r, _ic in
                       sorted({(r, "") for r, _ic in order})], dtype=np.int64)
    return resseq, ctr, blocks


def geometry_columns(receptor_pdb: str | Path, chain: str
                     ) -> tuple[np.ndarray, np.ndarray]:
    """``(resseq, X)`` with ``X`` of shape ``(n_residues, 624)``.

    Each family is aggregated separately and the blocks are then concatenated,
    which is what the measurement did: the four builders each produced their own
    three-aggregation matrix and ``straddling_attachment`` joined them. Doing it
    in the other order --- concatenating first and aggregating once --- gives the
    same numbers for these particular aggregations, and would not if an
    aggregation were ever added that is not linear in the property, so the
    measurement's order is kept rather than the convenient one.
    """
    resseq, ctr, blocks = raw_quantities(receptor_pdb, chain)
    out, off = [], 0
    for _fam, quantities in FAMILIES:
        k = len(quantities)
        out.append(aggregate(ctr, blocks[:, off:off + k]))
        off += k
    assert off == blocks.shape[1], (off, blocks.shape)
    X = np.concatenate(out, axis=1)
    if X.shape[1] != N_COLUMNS:
        raise AssertionError(f"{X.shape[1]} columns, expected {N_COLUMNS}")
    return resseq, X
