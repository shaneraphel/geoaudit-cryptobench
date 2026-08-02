#!/usr/bin/env python3.12
"""Wires from the van der Waals contact wall — the fifth geometry family.

What this is
------------
``src/pocket_bench/methods/contact_wall.py`` measures the contact sheet a ligand
would have to displace: Bondi-tight and soft contacts, open octants, plane
asperity, carbon/polar imbalance. This file turns those quantities into columns
the counting field can read under the same three aggregations the other families
use (own / contact-8A / walk-2), so a comparison is about the quantities and not
the attachment.

Why this family
---------------
Side-chain geometry reads rotamers; void topology reads empty space; neither
reads the packed face those rotamers press against. A cryptic opening is often
that face moving. AGENT_MEMORY 2c: members must be different measurements, not
radii of one operator — the column list in ``contact_wall.COLUMNS`` is that list.

Prediction (written before any lift number existed)
---------------------------------------------------
* Raw lift **+0.001 to +0.004** on 12 cluster-disjoint halvings.
* Overlap with side-chain / void: **30–50%** of its additive sum.
* ``more_old`` control near zero or negative; permuted arm negative if live.
* Falsification route: lift ≤ 0 with permutation matching intact → null family
  (cell budget). Lift > +0.006 → look at open-octant / asperity subgroup, which
  is the part least redundant with burial wires.

Nothing here reads a label, the test fold, or any external unit.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from pocket_bench.methods.contact_wall import (
    COLUMNS as WALL_COLUMNS,
    compute,
    consistency,
)
from pocket_bench.pdb_io import parse_pdb_atoms
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.contact_wall_wires.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
CACHE = ROOT / "data/cryptobench_apo/_contact_wall_cache_train.npz"
CACHE_PERM = ROOT / "data/cryptobench_apo/_contact_wall_perm_cache_train.npz"

CONTACT_RADIUS = 8.0
AGGREGATIONS = ("own", "contact", "walk2")
PERMUTATION_SEED = 20260802
CENTROID_TOLERANCE = 5e-4
SKIP = frozenset({"HOH", "WAT", "DOD"})
N_JOBS = min(9, os.cpu_count() or 4)


def column_names() -> tuple[str, ...]:
    return tuple(f"{agg}~{q}" for agg in AGGREGATIONS for q in WALL_COLUMNS)


def build_chain(ctr: np.ndarray, prop: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = (d <= CONTACT_RADIUS).astype(np.float64)
    two = adj @ adj
    np.fill_diagonal(two, 0.0)
    return np.concatenate([prop, adj @ prop, two @ prop], axis=1)


def _residue_rows(atoms: list[dict], chain: str):
    keep = [a for a in atoms
            if a["chain"] == chain and a["element"] != "H"
            and a["resname"] not in SKIP]
    poly: dict[tuple[int, str], list[tuple[float, float, float]]] = {}
    atoms_by: dict[tuple[int, str], list[dict]] = {}
    for a in keep:
        key = (a["resseq"], a["icode"].strip())
        poly.setdefault(key, []).append((a["x"], a["y"], a["z"]))
        atoms_by.setdefault(key, []).append(a)
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
    return order, ctr, take, atoms_by


def _one_chain(args):
    u, n, path, chain, ctr_slice = args
    text = Path_read(path)
    atoms = parse_pdb_atoms(text)
    order, ctr, take, atoms_by = _residue_rows(atoms, chain)
    if len(take) != n:
        return ("err", u, f"universe {len(take)} != cache {n}")
    d = float(np.abs(ctr - ctr_slice).max())
    if d > CENTROID_TOLERANCE:
        return ("err", u, f"centroid drift {d:.2e}")
    atoms_by_res = [atoms_by[order[k]] for k in range(len(order))]
    resseqs = [order[k][0] for k in range(len(order))]
    x_full = compute(atoms_by_res, resseqs)
    x = x_full[take]
    bad = consistency(x)
    if bad:
        return ("err", u, f"consistency {bad}")
    return ("ok", u, d, x)


from pathlib import Path as _Path

def Path_read(path):
    return _Path(path).read_text()


def raw_quantities() -> tuple[np.ndarray, np.ndarray]:
    w = np.load(WIDE, allow_pickle=False)
    units = [str(u) for u in w["units"]]
    n_res, ctr_cached = w["n_res_per"], w["ctr"]
    w.close()
    paths = {f"{e['pdb']}_{e['chain']}": (str(ROOT / e["receptor_path"]), e["chain"])
             for e in json.loads(MANIFEST.read_text())["entries"]}

    jobs = []
    off = 0
    for u, n in zip(units, n_res):
        n = int(n)
        if u not in paths:
            raise SystemExit(f"{u} missing from manifest")
        path, chain = paths[u]
        jobs.append((u, n, path, chain, ctr_cached[off:off + n].copy()))
        off += n

    out = np.zeros((int(n_res.sum()), len(WALL_COLUMNS)), dtype=np.float64)
    worst, worst_unit = 0.0, ""
    t0 = time.perf_counter()
    done = 0
    # map unit -> offset
    offsets = {}
    off = 0
    for u, n in zip(units, n_res):
        offsets[u] = off
        off += int(n)

    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futs = [ex.submit(_one_chain, j) for j in jobs]
        for fut in as_completed(futs):
            rec = fut.result()
            if rec[0] == "err":
                raise SystemExit(f"{rec[1]}: {rec[2]}")
            _, u, d, x = rec
            if d > worst:
                worst, worst_unit = d, u
            out[offsets[u]:offsets[u] + len(x)] = x
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(units)} chains  "
                      f"{time.perf_counter() - t0:.0f}s  jobs={N_JOBS}",
                      flush=True)
    print(f"  centroids agree to {worst:.2e} (worst {worst_unit})", flush=True)
    return out, n_res


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

    prop, n_res = raw_quantities()
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
    ap.add_argument("--permuted", action="store_true")
    a = ap.parse_args(argv)
    C, names = build_or_load(a.rebuild, a.permuted)
    print(f"\n{C.shape[1]} columns over {C.shape[0]} residues")
    nz = (C != 0).mean(axis=0)
    print(f"  non-zero share: min {nz.min():.3f}  median "
          f"{float(np.median(nz)):.3f}  max {nz.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
