#!/usr/bin/env python3.12
"""The subgroup of residue permutations the wire bank cannot see, and what it costs.

The object
----------
Fix a chain with residue set ``V``, ``|V| = n``. Every wire is quantised by
within-chain rank into quartiles, giving a digit matrix ``D in {0,1,2,3}^(n x W)``.
Every table reads two columns, so the score of residue ``i`` before the spatial
gate is a function of the row ``D[i,:]`` alone:

    S_raw(i) = f(D[i,:])          for some f fixed by the compiled field.

**Definition (indistinguishability).** Residues ``i`` and ``j`` are
*field-indistinguishable* when ``D[i,:] = D[j,:]``. Then ``S_raw(i) = S_raw(j)``
identically, for **every** choice of table pairing, every integer fan-out, and
every cell content — because all of those are functions of the digit row and
nothing else.

**Definition (deformation subgroup).** Let the orbits of the indistinguishability
relation partition ``V`` into blocks ``B_1, ..., B_m``. The *deformation
subgroup* of the chain is the Young subgroup

    D(V) = S_{B_1} x ... x S_{B_m}  <=  S_n,        |D(V)| = prod_r |B_r|!

the permutations of residues that leave the digit matrix invariant up to
relabelling. It is the exact group of rearrangements the wire bank is blind to.

Why this is worth measuring rather than asserting
-------------------------------------------------
It is an **architecture-level ceiling that no amount of table work can move**.
Every open item on the wire axis — pairings, fan-out, quantisation ladder, bank
size — reorders or reweights functions of ``D``. None of them can separate two
rows that are equal. So the size of ``D(V)`` bounds what the pre-gate ordering
can achieve, and the interesting number is not the group order but the part of
it that straddles the label: a block containing both a cryptic and a non-cryptic
residue is an error the architecture cannot fix.

The prediction, written before the run
--------------------------------------
Per-unit PR-AUC is where this field trails pLM-NN (−0.0100) while leading on the
pooled residue read. PR-AUC at low recall is decided by the first few residues,
and ties there are resolved arbitrarily. If the blocks are large near the top of
the ranking, that is a mechanism for the deficit; if almost every residue is a
singleton, the deficit is about ranking quality and this object is a curiosity.

`ORDER_COMPOSITION_SCREEN.json` is the other half of this: it measured that the
raw ordering is worse than the gated one everywhere including the top. If the
blocks are large, ties are part of why.

``clinical_grade`` is false.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from pocket_bench.methods.table_bank import chain_digits
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.indistinguishability_group.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
OUT = ROOT / "results/architecture_sweep/INDISTINGUISHABILITY_GROUP.json"


def _blocks(Dc: np.ndarray) -> np.ndarray:
    """Block index per residue, from exact equality of digit rows."""
    view = np.ascontiguousarray(Dc).view(
        np.dtype((np.void, Dc.dtype.itemsize * Dc.shape[1])))
    _uniq, inv = np.unique(view.ravel(), return_inverse=True)
    return inv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--top-q", type=float, default=0.10,
                    help="operating point already compiled on the training fold")
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res = z["X"], z["y"], z["n_res_per"]
    units = [str(u) for u in z["units"]]
    z.close()
    D = chain_digits(np.asarray(W, dtype=np.float64), n_res).astype(np.uint8)
    print(f"digits {D.shape} in {time.perf_counter() - t0:.0f}s", flush=True)

    n_res_tot = int(D.shape[0])
    in_block = 0            # residues in a block of size >= 2
    straddling = 0          # residues in a block holding both classes
    log_order = 0.0         # log10 |D(V)| summed over chains
    max_block = 0
    max_block_unit = ""
    block_hist: Counter = Counter()
    n_cryptic = 0
    cryptic_in_block = 0
    cryptic_straddling = 0
    per_unit = []

    from math import lgamma, log10
    off = 0
    for u, n in zip(units, n_res):
        n = int(n)
        Dc, yc = D[off:off + n], y[off:off + n]
        off += n
        blk = _blocks(Dc)
        counts = np.bincount(blk)
        sizes = counts[blk]
        big = sizes >= 2
        in_block += int(big.sum())
        n_cryptic += int(yc.sum())
        cryptic_in_block += int((big & (yc > 0)).sum())

        # A block is straddling when it holds at least one of each class.
        pos_per = np.bincount(blk, weights=(yc > 0).astype(float))
        strad = (pos_per[blk] > 0) & (pos_per[blk] < counts[blk])
        straddling += int(strad.sum())
        cryptic_straddling += int((strad & (yc > 0)).sum())

        for c in counts:
            if c >= 2:
                block_hist[int(c)] += 1
        log_order += sum(lgamma(int(c) + 1) for c in counts) / np.log(10)
        m = int(counts.max())
        if m > max_block:
            max_block, max_block_unit = m, u
        per_unit.append({
            "unit": u, "n": n, "n_blocks": int(counts.size),
            "n_in_nontrivial_block": int(big.sum()),
            "max_block": m,
            "log10_group_order": round(
                sum(lgamma(int(c) + 1) for c in counts) / float(np.log(10)), 3),
        })

    # ---- the filtration -----------------------------------------------------
    # At the full width the subgroup is trivial and that is nearly forced: 645
    # wires at two bits each address 4^645 states for a few hundred residues.
    # The object with content is the *filtration* D_1 >= D_2 >= ... >= D_W = 1
    # obtained by reading only the first k wires. Two curves are recorded: how
    # fast the blocks break up, and how fast the label stops straddling them.
    # The second is the one that matters, because separating two residues the
    # label agrees on is free and separating two it disagrees on is the task.
    ks = [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128,
                      192, 256, 384, 512, 645) if k <= D.shape[1]]
    filt = []
    for k in ks:
        blk_res = 0
        strad_res = 0
        strad_cryptic = 0
        off2 = 0
        for n in n_res:
            n = int(n)
            Dc, yc = D[off2:off2 + n, :k], y[off2:off2 + n]
            off2 += n
            blk = _blocks(Dc)
            counts = np.bincount(blk)
            sizes = counts[blk]
            blk_res += int((sizes >= 2).sum())
            pos_per = np.bincount(blk, weights=(yc > 0).astype(float))
            strad = (pos_per[blk] > 0) & (pos_per[blk] < counts[blk])
            strad_res += int(strad.sum())
            strad_cryptic += int((strad & (yc > 0)).sum())
        filt.append({
            "n_wires": k,
            "frac_residues_in_nontrivial_block": round(
                blk_res / max(n_res_tot, 1), 5),
            "frac_residues_label_straddling": round(
                strad_res / max(n_res_tot, 1), 5),
            "frac_cryptic_label_straddling": round(
                strad_cryptic / max(n_cryptic, 1), 5),
        })
        print(f"  k={k:4d}  in-block {filt[-1]['frac_residues_in_nontrivial_block']:.4f}"
              f"  straddling {filt[-1]['frac_residues_label_straddling']:.4f}"
              f"  cryptic-straddling {filt[-1]['frac_cryptic_label_straddling']:.4f}",
              flush=True)
    trivial_at = next((f["n_wires"] for f in filt
                       if f["frac_residues_in_nontrivial_block"] == 0.0), None)
    label_free_at = next((f["n_wires"] for f in filt
                          if f["frac_cryptic_label_straddling"] == 0.0), None)

    out = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "definition_indistinguishable": (
            "two residues of one chain whose 645-wire quartile digit rows are "
            "equal; their pre-gate scores are then identical for every table "
            "pairing, every integer fan-out and every cell content, because "
            "all of those are functions of the digit row alone"),
        "definition_deformation_subgroup": (
            "the Young subgroup S_{B_1} x ... x S_{B_m} of the symmetric group "
            "on the chain's residues, where the B_r are the indistinguishability "
            "blocks; it is the exact group of residue rearrangements the wire "
            "bank cannot detect"),
        "why_it_is_a_ceiling": (
            "every open item on the wire axis reorders or reweights functions "
            "of the digit matrix; none can separate two equal rows, so this "
            "bounds the pre-gate ordering independently of the table topology"),
        "what_it_does_not_bound": (
            "the gated score, which adds each residue's neighbourhood mean and "
            "therefore separates indistinguishable residues whenever their "
            "neighbourhoods differ. The bound is on the wires, not on the "
            "detector, and the gap between the two is what the gate is worth"),
        "n_wires": int(D.shape[1]),
        "n_residues": n_res_tot,
        "n_residues_definition": "training-fold residues in the wide cache",
        "n_units": len(units),
        "n_cryptic_residues": n_cryptic,
        "n_cryptic_residues_definition": "residues labelled cryptic-binding by "
                                         "CryptoBench in the training fold",
        "residues_in_a_nontrivial_block": in_block,
        "residues_in_a_nontrivial_block_fraction": round(
            in_block / max(n_res_tot, 1), 5),
        "cryptic_residues_in_a_nontrivial_block": cryptic_in_block,
        "cryptic_residues_in_a_nontrivial_block_fraction": round(
            cryptic_in_block / max(n_cryptic, 1), 5),
        "residues_in_a_label_straddling_block": straddling,
        "residues_in_a_label_straddling_block_definition": (
            "residues whose block holds at least one cryptic and at least one "
            "non-cryptic residue; the wires cannot order these two apart and "
            "no table work can"),
        "residues_in_a_label_straddling_block_fraction": round(
            straddling / max(n_res_tot, 1), 5),
        "cryptic_residues_in_a_straddling_block": cryptic_straddling,
        "cryptic_residues_in_a_straddling_block_fraction": round(
            cryptic_straddling / max(n_cryptic, 1), 5),
        "largest_block": max_block,
        "largest_block_unit": max_block_unit,
        "block_size_histogram": dict(sorted(block_hist.items())[:25]),
        "log10_total_group_order_over_the_fold": round(log_order, 2),
        "log10_total_group_order_definition": (
            "sum over chains of log10 of the Young subgroup order; a single "
            "number for the whole fold, quoted because it is the natural size "
            "of the object and not because it is actionable"),
        "filtration": filt,
        "filtration_definition": (
            "reading only the first k wires gives a coarser digit row and a "
            "larger subgroup D_k; the sequence D_1 >= D_2 >= ... >= D_W is a "
            "filtration of the symmetric group by wire count, and these are "
            "its two summary curves"),
        "wires_at_which_the_subgroup_becomes_trivial": trivial_at,
        "wires_at_which_no_cryptic_residue_straddles": label_free_at,
        "why_the_second_number_is_the_interesting_one": (
            "separating two residues the label agrees on costs address space "
            "and buys nothing; separating two it disagrees on is the task. The "
            "gap between these two counts is the address space the bank spends "
            "on distinctions the label does not care about"),
        "seconds": round(time.perf_counter() - t0, 1),
        "per_unit": sorted(per_unit, key=lambda r: -r["max_block"])[:40],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", a.out)
    for k in ("n_residues", "n_cryptic_residues",
              "residues_in_a_nontrivial_block_fraction",
              "cryptic_residues_in_a_nontrivial_block_fraction",
              "residues_in_a_label_straddling_block_fraction",
              "cryptic_residues_in_a_straddling_block_fraction",
              "largest_block", "log10_total_group_order_over_the_fold"):
        print(f"  {k:52s} {out[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
