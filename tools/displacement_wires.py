#!/usr/bin/env python3
"""Wires from the deposited displacement fields, the fourth discarded input.

What this is
------------
``src/pocket_bench/methods/displacement.py`` computes 48 quantities from three
fields of every ATOM record that this pipeline has never read: the temperature
factor in columns 61-66, the occupancy in 55-60, and the alternate-location
indicator in column 17. This turns them into columns the counting field can read
under the same three aggregations ``chemistry_wires.py``, ``backbone_wires.py``,
``sidechain_wires.py`` and ``void_wires.py`` use -- the residue's own value, the
sum over residues within 8 A, and the sum two steps out on the contact graph --
so a comparison against any of those families is a comparison of the quantities
and not of the attachment.

Why this family
---------------
``AGENT_MEMORY`` 2i: a family is worth measuring only if it reads bytes the
pipeline throws away. Six families that failed that screen measured null. Three
that passed it are worth +0.0044, +0.0048 and +0.0026, and all three read
coordinates. This one reads none of them.

``pdb_io.parse_pdb_atoms`` never extracts the B-factor, and it reads occupancy
and altLoc only in order to *discard* alternates, keeping the highest-occupancy
copy. Over 200 training receptors every single one has a varying B-factor column
and 24 % carry at least one alternate conformer. The primary input is
universally available and entirely unused.

The physical argument, which is checkable rather than rhetorical: a cryptic site
is a site whose apo coordinates are not the whole story, a temperature factor is
the refinement's own estimate of how badly one atom's position is determined,
and an alternate conformer is the crystallographer saying one position was not
enough. Both are per-atom statements about local disorder made by the
experiment.

What is predicted, before the run
----------------------------------
``AGENT_MEMORY`` 2m committed that the next family would carry a written
prediction of its overlap, because the void family turned a sign-prediction into
a magnitude-prediction and one instance is not a law. The prediction lives in the
module docstring and is repeated here in one line so that it cannot be quietly
dropped: **raw lift +0.001 to +0.003, and 30-50 % overlap with the geometry 528
stack against void's 12 %**, because a B-factor is largely a function of solvent
exposure and exposure is what the deployed wires already measure best. A raw lift
above +0.004 falsifies the reasoning in an interesting direction and the place to
look is group E, alternate conformers, which is not a function of exposure.

The controls, and there are two
-------------------------------
``more_old`` adds the same number of extra tables over the deployed wires with
the family absent, which catches a family that is only buying cells. It does not
catch a family whose columns carry ordinary per-residue information that happens
not to be about displacement, so ``--permuted`` is built here too: the same 48
quantities, aggregated identically, rows shuffled within each chain under a
fixed seed. Every column's multiset over the chain is preserved exactly and only
the correspondence between a row and its residue is destroyed. That arm ran
-0.0021, -0.0033 and -0.0014 on the three live families.

Alignment is checked, not assumed
---------------------------------
The rows of the wide cache are residues in an order this file must reproduce
exactly, and nothing would raise if it did not: every lookup would succeed and
every residue would be handed another residue's B-factor. So each residue's
heavy-atom centroid is recomputed and compared against the cache's own ``ctr``
row by row.

One extra check this family needs and the others do not: a receptor whose
B-factor column is constant carries no information, and a *corpus* whose
B-factor columns were mostly constant would make this family null for a reason
that has nothing to do with the screen. The build counts those chains and prints
the count, so a null result can be read against how much signal was present.

Nothing here reads a label, the test fold, or any external unit.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from pocket_bench.methods.displacement import (
    COLUMNS as DISP_COLUMNS,
    compute,
    consistency,
    parse_displacement,
)
from pocket_bench.pdb_io import parse_pdb_atoms
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.displacement_wires.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
CACHE = ROOT / "data/cryptobench_apo/_displacement_cache_train.npz"
CACHE_PERM = ROOT / "data/cryptobench_apo/_displacement_perm_cache_train.npz"

CONTACT_RADIUS = 8.0
AGGREGATIONS = ("own", "contact", "walk2")
PERMUTATION_SEED = 20260731
CENTROID_TOLERANCE = 5e-4

SKIP = frozenset({"HOH", "WAT", "DOD"})


def column_names() -> tuple[str, ...]:
    return tuple(f"{agg}~{q}" for agg in AGGREGATIONS for q in DISP_COLUMNS)


def build_chain(ctr: np.ndarray, prop: np.ndarray) -> np.ndarray:
    """The three aggregations for one chain, identical to the other families'."""
    d = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = (d <= CONTACT_RADIUS).astype(np.float64)
    two = adj @ adj
    np.fill_diagonal(two, 0.0)
    return np.concatenate([prop, adj @ prop, two @ prop], axis=1)


def _residue_rows(atoms: list[dict], chain: str
                  ) -> tuple[list[tuple[int, str]], np.ndarray, np.ndarray]:
    """The polymer's residues, the evaluation universe, and the map between them.

    Duplicated from ``void_wires`` deliberately rather than imported, for the
    reason ``sidechain_wires`` records: that file feeds a pinned artifact and
    importing from it would put this file inside that artifact's blast radius
    for no benefit.
    """
    keep = [a for a in atoms
            if a["chain"] == chain and a["element"] != "H"
            and a["resname"] not in SKIP]
    poly: dict[tuple[int, str], list[tuple[float, float, float]]] = {}
    for a in keep:
        poly.setdefault((a["resseq"], a["icode"].strip()), []).append(
            (a["x"], a["y"], a["z"]))
    order = sorted(poly)

    universe = sorted({r for r, _ in order})
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


def raw_quantities() -> tuple[np.ndarray, np.ndarray, dict]:
    """The 48 per-residue quantities for every training residue."""
    w = np.load(WIDE, allow_pickle=False)
    units = [str(u) for u in w["units"]]
    n_res, ctr_cached = w["n_res_per"], w["ctr"]
    w.close()

    paths = {f"{e['pdb']}_{e['chain']}": (ROOT / e["receptor_path"], e["chain"])
             for e in json.loads(MANIFEST.read_text())["entries"]}

    out = np.zeros((int(n_res.sum()), len(DISP_COLUMNS)), dtype=np.float64)
    worst, worst_unit, off, t0 = 0.0, "", 0, time.perf_counter()
    flat_b = 0
    with_alt = 0
    n_alt_residues = 0
    for i, (u, n) in enumerate(zip(units, n_res)):
        n = int(n)
        if u not in paths:
            raise SystemExit(f"{u} is in the wide cache and not in the manifest")
        path, chain = paths[u]
        text = path.read_text()
        atoms = parse_pdb_atoms(text)
        order, ctr, take = _residue_rows(atoms, chain)
        if len(take) != n:
            raise SystemExit(
                f"{u}: the receptor gives {len(take)} universe rows and the "
                f"wide cache holds {n}; the rows are not the same residues")
        d = float(np.abs(ctr - ctr_cached[off:off + n]).max())
        if d > worst:
            worst, worst_unit = d, u
        if d > CENTROID_TOLERANCE:
            raise SystemExit(
                f"{u}: recomputed centroids differ from the cache by {d:.2e}, "
                f"above {CENTROID_TOLERANCE:.0e}. The residue order this file "
                f"derives is not the cache's order, and every displacement "
                f"quantity would be attached to the wrong residue without "
                f"raising")

        residues = parse_displacement(text, chain)
        bs = [a["b"] for group in residues.values() for a in group]
        if not bs or max(bs) - min(bs) < 1e-9:
            flat_b += 1
        alts = sum(1 for group in residues.values()
                   if any(a["altloc"] != " " for a in group))
        if alts:
            with_alt += 1
            n_alt_residues += alts

        x = compute(residues, order)[take]
        bad = consistency(x, [order[k] for k in take])
        if bad:
            raise SystemExit(f"{u}: displacement quantities violate {bad}")
        out[off:off + n] = x
        off += n
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(units)} chains  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)

    coverage = {
        "n_chains": len(units),
        "n_with_a_flat_b_factor_column": flat_b,
        "n_with_at_least_one_alternate": with_alt,
        "n_residues_carrying_an_alternate": n_alt_residues,
    }
    print(f"  centroids agree with the cache to {worst:.2e} (worst on "
          f"{worst_unit})", flush=True)
    print(f"  {flat_b} of {len(units)} chains have a flat B-factor column; "
          f"{with_alt} carry an alternate, over {n_alt_residues} residues",
          flush=True)
    return out, n_res, coverage


def _aggregate(prop: np.ndarray, n_res: np.ndarray) -> np.ndarray:
    w = np.load(WIDE, allow_pickle=False)
    ctr = w["ctr"]
    w.close()
    blocks, off = [], 0
    for n in n_res:
        n = int(n)
        blocks.append(build_chain(ctr[off:off + n], prop[off:off + n]))
        off += n
    return np.concatenate(blocks, axis=0).astype(np.float32)


def build_or_load(force: bool = False, permuted: bool = False
                  ) -> tuple[np.ndarray, tuple[str, ...]]:
    names = column_names()
    cache = CACHE_PERM if permuted else CACHE
    if cache.is_file() and not force:
        z = np.load(cache, allow_pickle=False)
        if tuple(str(s) for s in z["names"]) == names:
            print(f"reusing {cache.relative_to(ROOT)}  {z['C'].shape}",
                  flush=True)
            return z["C"], names
        print("cached column names differ; rebuilding", flush=True)

    prop, n_res, _coverage = raw_quantities()
    if permuted:
        rng = np.random.default_rng(PERMUTATION_SEED)
        off = 0
        for n in n_res:
            n = int(n)
            prop[off:off + n] = prop[off:off + n][rng.permutation(n)]
            off += n
    C = _aggregate(prop, n_res)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, C=C, names=np.asarray(names))
    print(f"wrote {cache.relative_to(ROOT)}  {C.shape}", flush=True)
    return C, names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--permuted", action="store_true",
                    help="build the row-permuted control instead")
    a = ap.parse_args(argv)
    C, names = build_or_load(a.rebuild, a.permuted)
    print(f"\n{C.shape[1]} columns over {C.shape[0]} residues")
    nz = (C != 0).mean(axis=0)
    print(f"  non-zero share: min {nz.min():.3f}  median "
          f"{float(np.median(nz)):.3f}  max {nz.max():.3f}")
    dead = [names[j] for j in np.flatnonzero(np.ptp(C, axis=0) == 0)]
    if dead:
        print(f"  {len(dead)} constant columns: {dead[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
