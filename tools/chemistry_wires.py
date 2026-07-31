#!/usr/bin/env python3.12
"""Wires from the fourteen residue chemical quantities, and what they cost.

The family
----------
``residue_chemistry.py`` gives fourteen integer properties of a side chain.
This builds three aggregations of each over a chain's contact geometry:

    own      the residue's own value
    contact  the sum over residues within CONTACT_RADIUS
    walk2    the sum over residues two steps away on the contact graph

Forty-two columns. The variety is in the fourteen quantities, which are
different from each other in the sense AGENT_MEMORY 2c measured; the three
aggregations of one property are the same-quantity category, which that section
records as the only negative one in the deployed bus, and they are here anyway
because fourteen columns make too small a table block for a null to mean
anything. That trade is stated rather than hidden, and if the family fails the
first thing to check is whether the aggregations diluted it.

Why this family and not a fifth re-reading of the contact graph
---------------------------------------------------------------
Four families have been measured and none moved the detector: composition,
asymmetry, graph invariants, and then all three attachments including the
straddling one, which recovered the union attachment's own loss and gained
nothing. The honest prior for a fifth is poor and is stated in PREDICTION
below.

What is different here is not the aggregation, it is the content.
``chi_rotatable`` counts rotameric dihedrals, and a cryptic pocket is by
definition a site that is closed and opens. The 645 deployed wires are
geometric invariants of atom positions and cannot see conformational freedom;
``composition_wires.py`` reads residue identity but as eight classes, and its
aliphatic class holds ALA, VAL, LEU, ILE and MET whose chi runs 0, 1, 2, 2, 3.
So this is the first family carrying a quantity about what the structure could
become rather than about the shape it currently has.

The prediction, recorded before the run
---------------------------------------
Given four failures, a lift above the 0.0026 reseed floor would be the first
family to move this detector and should be treated as surprising rather than
expected. The range committed to is -0.002 to +0.003 for the union attachment.
Anything above the floor falsifies the reading that the wire axis is closed;
anything inside it confirms it and the axis stays closed with a fifth data
point instead of four.

Nothing here reads the test fold or any external unit.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pocket_bench.methods.residue_chemistry import (
    AA20,
    property_names,
    table as chem_table,
)
from pocket_bench.methods.sequence_wires import AA20 as SEQ_AA20
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.chemistry_wires.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
CODES = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
CACHE = ROOT / "data/cryptobench_apo/_chemwire_cache_train.npz"

CONTACT_RADIUS = 8.0
AGGREGATIONS = ("own", "contact", "walk2")

PREDICTION = {
    "committed_before_the_run": True,
    "field_lift_low": -0.002,
    "field_lift_high": +0.003,
    "reseed_floor": 0.0026,
    "why_the_prior_is_poor": "four families and three attachments measured, "
                             "none moved the detector; the largest effect "
                             "anywhere on this axis is +0.0010",
    "what_would_be_surprising": "a union-attachment lift above the reseed "
                                "floor, which would be the first family to "
                                "move this detector and would falsify the "
                                "reading that the wire axis is closed",
}


def column_names() -> tuple[str, ...]:
    return tuple(f"{agg}~{p}" for agg in AGGREGATIONS for p in property_names())


def build_chain(ctr: np.ndarray, prop: np.ndarray) -> np.ndarray:
    """The three aggregations for one chain.

    ``prop`` is ``(n_residues, n_properties)`` already mapped through the
    residue table, so this function performs no chemistry -- it is contact
    geometry applied to whatever quantities it is handed, which keeps the
    chemistry in one file and the geometry in another.
    """
    n = int(ctr.shape[0])
    d = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = (d <= CONTACT_RADIUS).astype(np.float64)

    two = adj @ adj
    np.fill_diagonal(two, 0.0)

    return np.concatenate([prop, adj @ prop, two @ prop], axis=1)


def build_or_load(force: bool = False) -> tuple[np.ndarray, tuple[str, ...]]:
    names = column_names()
    if CACHE.is_file() and not force:
        z = np.load(CACHE, allow_pickle=False)
        if tuple(str(s) for s in z["names"]) == names:
            print(f"reusing {CACHE.relative_to(ROOT)}  {z['C'].shape}",
                  flush=True)
            return z["C"], names
        print("cached column names differ; rebuilding", flush=True)

    # The residue table is indexed by the expanded cache's ``codes``, and that
    # index is only meaningful if both files order the twenty the same way. A
    # mismatch would not raise anywhere: every lookup would succeed and every
    # residue would be assigned another residue's chemistry.
    if tuple(AA20) != tuple(SEQ_AA20):
        raise SystemExit(
            f"residue_chemistry.AA20 and sequence_wires.AA20 disagree; the "
            f"codes in the expanded cache index the latter, so every residue "
            f"would silently receive another residue's properties.\n"
            f"  chemistry: {AA20}\n  sequence:  {tuple(SEQ_AA20)}")

    w = np.load(WIDE, allow_pickle=False)
    e = np.load(CODES, allow_pickle=False)
    for key in ("units", "n_res_per", "y"):
        if not np.array_equal(w[key], e[key]):
            raise SystemExit(f"the wide and expanded caches disagree about "
                             f"{key}; their rows are not the same residues")
    if not np.array_equal(w["ctr"], e["ctr"]):
        raise SystemExit("the two caches disagree about residue centroids")

    codes = e["codes"]
    if int(codes.min()) < 0:
        raise SystemExit(
            f"{int((codes < 0).sum())} residues carry no standard residue "
            f"code; a chemistry lookup over them would silently be row zero, "
            f"which is alanine")
    if int(codes.max()) >= len(AA20):
        raise SystemExit(f"codes reach {int(codes.max())} against "
                         f"{len(AA20)} residues")

    prop_all = chem_table()[codes].astype(np.float64)
    ctr, n_res = w["ctr"], w["n_res_per"]

    t0 = time.perf_counter()
    blocks = []
    off = 0
    for i, n in enumerate(n_res):
        n = int(n)
        blocks.append(build_chain(ctr[off:off + n], prop_all[off:off + n]))
        off += n
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(n_res)} chains  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    C = np.concatenate(blocks, axis=0)
    if C.shape != (len(codes), len(names)):
        raise SystemExit(f"built {C.shape}, expected "
                         f"{(len(codes), len(names))}")
    C = C.astype(np.float32)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, C=C, names=np.asarray(names))
    print(f"wrote {CACHE.relative_to(ROOT)}  {C.shape} in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    return C, names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    C, names = build_or_load(a.rebuild)
    print(f"{C.shape[1]} columns over {C.shape[0]} residues")
    print(f"  {len(property_names())} quantities x "
          f"{len(AGGREGATIONS)} aggregations")
    nz = [(n, float(C[:, j].std())) for j, n in enumerate(names)]
    flat = [n for n, s in nz if s == 0.0]
    if flat:
        raise SystemExit(f"columns with no variation: {flat}")
    print("  every column varies")
    print(json.dumps(PREDICTION, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
