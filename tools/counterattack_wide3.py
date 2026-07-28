#!/usr/bin/env python3
"""Do the generated invariants help the counting field, or only its linear shadow?

The Fisher ceiling measured in ``expand_invariant_bank`` is a ceiling on *linear*
readouts of a bank. It is not a ceiling on the counting field, and the evidence
for that is already in this repository: the 645-wire table field scores 0.7992
on the official fold, above the 0.783 Fisher ceiling of the 35 invariants it is
built from. A table is an arbitrary function of the cell its digits address, so
it is not bounded by any linear form over those digits.

The +0.0081 that the 267 generated descriptors buy on the Fisher ceiling is
therefore the wrong number to decide on. The right one is what they buy the
counting field, and that is what this measures: the identical harness of
``counterattack_wide2`` -- same seeded cluster-disjoint halving, same random
partition bases, same integer fan-out, same three gates -- with one thing
changed, the wire set.

The wire set
------------
The 645 existing wires are kept exactly as they are. The 267 generated
descriptors are added under four statistics rather than the fifteen the
original wires get, because they are already multi-scale objects: each was
computed on a neighbourhood of 6 to 16 A, so averaging them again over 6 A
would mostly restate them. What they lack is standing at pocket scale, so each
is carried raw, and as its mean, centred difference and local rank over a 20 A
neighbourhood -- well outside any radius used to build it. 267 x 4 = 1068, for
1713 wires in total.

Memory. ``chain_digits`` emits int8, and the within-chain rank that defines a
digit depends on nothing outside the chain, so the digits of a wire block can be
computed and the floating-point block thrown away before the next one is built.
1713 wires of digits is 402 MB; the float matrix that was never held would have
been 3.2 GB.

The official test fold is not read here.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_wide3.py
"""
from __future__ import annotations

import gc
import json
import time

import numpy as np

from pocket_bench.methods.table_bank import (
    cell_offsets,
    chain_digits,
    compile_cells,
    integer_fanout,
    partition_tables,
    score,
)
from pocket_bench.paths import ROOT

from counterattack_ridge import spread_matched_gate
from counterattack_select import SEED, per_unit_auc
from expand_invariant_bank import BANKS, build_cache

WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT_BY_ARM = {
    "both": ROOT / "results/architecture_sweep/COUNTERATTACK_WIDE3.json",
    "existing": ROOT / "results/architecture_sweep/COUNTERATTACK_WIDE3_CONTROL.json",
}
GATES = ((14.0, 1.0), (18.0, 0.5), (18.0, 1.0))
CONTEXT_RADIUS = 20.0
BLOCK = 512
REFERENCE = {"645_wire_table_field_same_split": 0.8045,
             "516_wire_table_field": 0.8010,
             "172_wire_table_field": 0.7932}


def context_wires(F: np.ndarray, ctr: np.ndarray, n_res_per,
                  radius: float = CONTEXT_RADIUS) -> np.ndarray:
    """``(R, 4C)``: each descriptor raw, and its mean, centred difference and
    local rank over a ``radius`` neighbourhood inside its own chain."""
    F = np.asarray(F, dtype=np.float64)
    C = F.shape[1]
    out = np.empty((F.shape[0], 4 * C), dtype=np.float64)
    r2 = radius * radius
    off = 0
    for n in n_res_per:
        n = int(n)
        c = ctr[off:off + n]
        blk = F[off:off + n]
        mean = np.empty((n, C))
        rank = np.empty((n, C))
        for i in range(0, n, BLOCK):
            j = min(i + BLOCK, n)
            a = (((c[i:j, None, :] - c[None, :, :]) ** 2).sum(-1) <= r2
                 ).astype(np.float64)
            cnt = np.maximum(a.sum(1), 1.0)[:, None]
            mean[i:j] = (a @ blk) / cnt
            less = blk[i:j, None, :] > blk[None, :, :]
            rank[i:j] = (less * a[:, :, None]).sum(1) / cnt
        out[off:off + n] = np.concatenate(
            [blk, mean, blk - mean, rank], axis=1)
        off += n
    return out


def _digits_of_generated(n_res, ctr, workers: int) -> tuple[np.ndarray, int]:
    """Digits for the generated wires, one bank at a time so the float blocks
    never coexist."""
    blocks, n_desc = [], 0
    for kind in BANKS:
        F = build_cache(kind, workers)
        n_desc += F.shape[1]
        t0 = time.perf_counter()
        W = context_wires(F, ctr, n_res)
        del F
        gc.collect()
        blocks.append(chain_digits(W, n_res))
        print(f"  {kind}: {W.shape[1]} wires digitised in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        del W
        gc.collect()
    return np.concatenate(blocks, axis=1), n_desc


def main(workers: int = 8, wires: str = "both",
         ridges: tuple[float, ...] = (0.03, 0.1, 0.3),
         bases: tuple[str, ...] = ("pairs8", "pairs16", "triples")) -> int:
    z = np.load(WIDE, allow_pickle=False)
    y, n_res, ctr = z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    print(f"digitising the {z['X'].shape[1]} existing wires", flush=True)
    D_old = chain_digits(z["X"], n_res)
    del z
    gc.collect()

    n_old = D_old.shape[1]
    n_desc = 0
    if wires == "both":
        print("digitising the generated wires", flush=True)
        D_new, n_desc = _digits_of_generated(n_res, ctr, workers)
        D = np.concatenate([D_old, D_new], axis=1)
        del D_new
    else:
        # The control arm. Same harness, same ridge grid, same seed, only the
        # generated wires withheld -- so a difference between the arms is the
        # wires and not the grid they were searched over.
        D = D_old
    del D_old
    gc.collect()

    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"]
                  for e in json.loads(MANIFEST.read_text())["entries"]}
    clusters = sorted({cluster_of[u] for u in units})
    rng = np.random.default_rng(SEED)
    rng.shuffle(clusters)
    fit_clusters = set(clusters[:len(clusters) // 2])
    is_fit = np.array([cluster_of[u] in fit_clusters for u in units])
    row_unit = np.repeat(np.arange(len(units)), n_res)
    fm, pm = is_fit[row_unit], ~is_fit[row_unit]
    n_pick_per = np.array([n for n, f in zip(n_res, is_fit) if not f])
    yfit, ypick, ctr_pick = y[fm], y[pm], ctr[pm]

    Dfit = np.ascontiguousarray(D[fm])
    Dpick = np.ascontiguousarray(D[pm])
    n_wires = D.shape[1]
    del D
    gc.collect()
    print(f"\n{n_wires} wires ({n_old} existing + {n_wires - n_old} generated "
          f"from {n_desc} descriptors); fit {len(yfit)} rows, pick "
          f"{len(ypick)} rows; ridges "
          f"{', '.join(f'{r:g}' for r in ridges)}", flush=True)

    all_bases = {"pairs8": ("pairs x8", [(2, 8)]),
                 "pairs16": ("pairs x16", [(2, 16)]),
                 "triples": ("pairs x16 + triples x8", [(2, 16), (3, 8)])}
    results = []
    for basis_name, spec in (all_bases[b] for b in bases):
        tables = []
        for width, rounds in spec:
            tables += partition_tables(n_wires, width, rounds, SEED)
        offsets = cell_offsets(tables)
        frac, tot = compile_cells(Dfit, yfit, tables, offsets)
        empty = int((tot == 0).sum())
        print(f"\n{basis_name}: {len(tables)} tables, {len(frac)} cells, "
              f"{empty} never addressed ({100.0 * empty / len(frac):.2f}%), "
              f"{tot[tot > 0].mean():.0f} residues per occupied cell",
              flush=True)
        for ridge in ridges:
            m = integer_fanout(Dfit, yfit, tables, offsets, frac, ridge, 32)
            S = score(Dpick, tables, offsets, frac, m)
            raw = per_unit_auc(S, ypick, n_pick_per)
            best, gname = -1.0, ""
            for r, wt in GATES:
                a = per_unit_auc(
                    spread_matched_gate(S, ctr_pick, n_pick_per, r, wt),
                    ypick, n_pick_per)
                if a > best:
                    best, gname = a, f"r{int(r)} w{wt}"
            print(f"  ridge {ridge:<5g} cap 32  raw {raw:.4f}  "
                  f"gated {best:.4f} ({gname})  fan-out "
                  f"{int(np.abs(m).sum())}", flush=True)
            results.append({
                "basis": basis_name, "spec": spec, "n_tables": len(tables),
                "n_cells": int(len(frac)), "n_cells_never_addressed": empty,
                "mean_residues_per_occupied_cell": float(tot[tot > 0].mean()),
                "ridge": ridge, "cap": 32, "gate": gname,
                "total_fan_out": int(np.abs(m).sum()),
                "n_tables_used": int((m != 0).sum()),
                "pick_half_roc_auc_raw": raw,
                "pick_half_roc_auc": best,
            })
        del frac, tot
        gc.collect()

    winner = max(results, key=lambda r: r["pick_half_roc_auc"])
    OUT = OUT_BY_ARM[wires]
    delta = winner["pick_half_roc_auc"] - REFERENCE[
        "645_wire_table_field_same_split"]
    print(f"\nbest: {winner['basis']}, ridge {winner['ridge']:g}, gate "
          f"{winner['gate']} -> pick-half {winner['pick_half_roc_auc']:.4f}")
    print(f"the same harness on 645 wires    {REFERENCE['645_wire_table_field_same_split']:.4f}")
    print(f"generated wires are worth        {delta:+.4f} to the counting field")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_wide3.v1",
        "arm": wires,
        "ridge_grid": list(ridges),
        "bases": list(bases),
        "clinical_grade": False,
        "question": "the generated banks lift the Fisher ceiling by +0.0081, "
                    "but a table is not a linear form over its digits. This "
                    "asks what they are worth to the counting field itself, "
                    "under the harness that produced the 645-wire number.",
        "n_wires": n_wires,
        "n_wires_existing": n_old,
        "n_wires_generated": n_wires - n_old,
        "n_generated_descriptors": n_desc,
        "generated_wire_statistics": ["raw", f"mean@{CONTEXT_RADIUS:g}",
                                      f"centred@{CONTEXT_RADIUS:g}",
                                      f"localrank@{CONTEXT_RADIUS:g}"],
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED},
        "reference_same_split": REFERENCE,
        "candidates": results,
        "selected": winner,
        "delta_vs_645_wires": round(float(delta), 6),
        "test_fold_read": False,
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--wires", choices=("both", "existing"), default="both",
                   help="'existing' is the control arm: the 645 wires alone, "
                        "through this same harness and ridge grid")
    p.add_argument("--ridges", type=float, nargs="+",
                   default=[0.03, 0.1, 0.3])
    p.add_argument("--bases", nargs="+",
                   choices=("pairs8", "pairs16", "triples"),
                   default=["pairs8", "pairs16", "triples"])
    a = p.parse_args()
    raise SystemExit(main(a.workers, a.wires, tuple(a.ridges),
                          tuple(a.bases)))
