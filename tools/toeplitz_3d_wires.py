#!/usr/bin/env python3.12
"""Train wires for toeplitz_3d family. clinical_grade=false."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pocket_bench.methods.toeplitz_3d import COLUMNS as TQ, compute, consistency
from pocket_bench.pdb_io import parse_pdb_atoms
from pocket_bench.paths import ROOT

WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
CACHE = ROOT / "data/cryptobench_apo/_toeplitz_3d_cache_train.npz"
CONTACT_RADIUS = 8.0
AGGREGATIONS = ("own", "contact", "walk2")
CENTROID_TOLERANCE = 5e-4
SKIP = frozenset({"HOH", "WAT", "DOD"})
N_JOBS = min(9, os.cpu_count() or 4)


def column_names():
    return tuple(f"{a}~{q}" for a in AGGREGATIONS for q in TQ)


def build_chain(ctr, prop):
    d = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = (d <= CONTACT_RADIUS).astype(np.float64)
    two = adj @ adj
    np.fill_diagonal(two, 0.0)
    return np.concatenate([prop, adj @ prop, two @ prop], axis=1)


def _residue_rows(atoms, chain):
    keep = [a for a in atoms if a["chain"] == chain and a["element"] != "H"
            and a["resname"] not in SKIP]
    poly, atoms_by = {}, {}
    for a in keep:
        key = (a["resseq"], a["icode"].strip())
        poly.setdefault(key, []).append((a["x"], a["y"], a["z"]))
        atoms_by.setdefault(key, []).append(a)
    order = sorted(poly)
    universe = sorted({r for r, _ in order})
    first = {}
    for j, (r, _) in enumerate(order):
        first.setdefault(r, j)
    take = np.array([first[r] for r in universe], dtype=np.int64)
    merged = {}
    for (r, _), pts in poly.items():
        merged.setdefault(r, []).extend(pts)
    ctr = np.array([np.mean(merged[r], axis=0) for r in universe], dtype=np.float64)
    return order, ctr, take, atoms_by


def _one(args):
    u, n, path, chain, ctr_slice = args
    atoms = parse_pdb_atoms(Path(path).read_text())
    order, ctr, take, atoms_by = _residue_rows(atoms, chain)
    if len(take) != n:
        return ("err", u, f"universe {len(take)} != {n}")
    drift = float(np.abs(ctr - ctr_slice).max())
    if drift > CENTROID_TOLERANCE:
        return ("err", u, f"drift {drift:.2e}")
    x = compute([atoms_by[k] for k in order], [k[0] for k in order])[take]
    bad = consistency(x)
    if bad:
        return ("err", u, str(bad))
    return ("ok", u, drift, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    names = column_names()
    if CACHE.is_file() and not a.rebuild:
        z = np.load(CACHE)
        if tuple(str(s) for s in z["names"]) == names:
            print("reusing", CACHE, z["C"].shape); return 0
    w = np.load(WIDE)
    units = [str(u) for u in w["units"]]
    n_res, ctr_cached = w["n_res_per"], w["ctr"]
    w.close()
    paths = {f"{e['pdb']}_{e['chain']}": (str(ROOT / e["receptor_path"]), e["chain"])
             for e in json.loads(MANIFEST.read_text())["entries"]}
    jobs, offsets, off = [], {}, 0
    for u, n in zip(units, n_res):
        n = int(n)
        path, chain = paths[u]
        jobs.append((u, n, path, chain, ctr_cached[off:off + n].copy()))
        offsets[u] = off; off += n
    out = np.zeros((int(n_res.sum()), len(TQ)), dtype=np.float64)
    t0 = time.perf_counter(); done = 0
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        for fut in as_completed([ex.submit(_one, j) for j in jobs]):
            rec = fut.result()
            if rec[0] == "err":
                raise SystemExit(f"{rec[1]}: {rec[2]}")
            _, u, _, x = rec
            out[offsets[u]:offsets[u] + len(x)] = x
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(units)} {time.perf_counter()-t0:.0f}s", flush=True)
    # aggregate
    w = np.load(WIDE); ctr = w["ctr"]; w.close()
    blocks, off = [], 0
    for n in n_res:
        n = int(n)
        blocks.append(build_chain(ctr[off:off + n], out[off:off + n])); off += n
    C = np.concatenate(blocks, axis=0).astype(np.float32)
    np.savez_compressed(CACHE, C=C, names=np.asarray(names))
    print(f"wrote {CACHE} {C.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
