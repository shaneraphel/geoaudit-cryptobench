#!/usr/bin/env python3.12
"""How many of the 5,152 tables does the readout actually use? Training folds only.

Where this comes from
---------------------
GRAM_CONDITIONING.json measured three things about the deployed configuration and
the third reframes the architecture. 76.5 per cent of the trace of the K x K
within-class scatter sits in the top one per cent of its directions -- roughly 52
of 5,152 -- and the cosine between the solved direction and the class mean
difference is 0.025 on every bank tried, including the one that ships. The bank is
massively over-complete and the readout has never used more than a small effective
subspace of it.

Two column families follow from that rather than from the columns. The
wire-asymmetry columns and the composition columns are of completely different
mathematical character, a linear solve values both, and the counting field
collects neither. If the solve is already confined to about fifty effective
directions then several hundred additional tables cannot enter it whatever they
carry, and the invisibility of both families is a property of the readout.

The prediction that makes, and what this tool measures
-----------------------------------------------------
A bank of order 52 to 200 tables, chosen on the fit half, should match the
5,152-table bank. If it does, the deployed detector carries two orders of
magnitude more table than it uses, and a detector that is 1.79 MB of integers
becomes one that is tens of kilobytes at the same accuracy. That is a stronger
claim than any of the lifts being chased, and it is falsifiable in the obvious
direction: if accuracy falls off smoothly with K' rather than plateauing early,
the trace concentration was misleading and the tables are doing distributed work
the spectrum does not see.

Two selection rules, because one of them uses the full solve and one does not.

``by multiplicity``  Rank tables by the absolute value of the integer
                     multiplicity the full 5,152-table solve assigns them, keep
                     the top K', re-solve on that subset alone. This is the rule
                     the spectrum suggests, and it is fit-half only, but it needs
                     the full solve to exist first.

``by gini``          Rank tables by their own between-cell label variance
                     V = sum_c (k_c - p n_c)^2 / (n_c N), the same integer-count
                     functional the fan-out ranking already uses. This needs no
                     full solve, so a bank chosen this way could be compiled
                     directly at any target size. It is the rule that would
                     actually ship.

``random``           A subset of the same size drawn at a fixed seed. Without it,
                     a plateau proves nothing: if random subsets of 200 tables
                     also match the full bank then the tables are interchangeable
                     and neither ranking is doing work.

What is held fixed
------------------
Cell frequencies are compiled once per split over the whole 5,152-table bank and
the subsets read the same numbers, so nothing about a cell changes when the bank
is truncated -- only which cells the score sums over, and the integer
multiplicities, which are re-solved on each subset because the whole question is
what the solve does with fewer tables. Quantisation, width, rounds, seed, ridge,
cap and gate are the deployed values throughout.

Reproduction gate: K' = 5152 by multiplicity is the full bank and must return the
frozen deployed numbers. It is run as an arm rather than assumed.

Training folds only. No test residue and no external unit is read.

Usage: PYTHONPATH=src:tools python3.12 tools/bank_truncation.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from expand_invariant_bank import SEED  # noqa: E402
from quantisation_ladder import compile_at, offsets_at  # noqa: E402
from select_architecture_on_train import cluster_half_split, per_unit_auc  # noqa: E402

from pocket_bench.methods.table_bank import (
    BLOCK,
    N_LEVELS,
    chain_digits,
    partition_tables,
)
from pocket_bench.methods.table_field import (
    FAN_OUT_CAP,
    GATE_RADIUS,
    GATE_WEIGHT,
    PARTITION_ROUNDS,
    PARTITION_SEED,
    RIDGE,
    TABLE_WIDTH,
    apply_gate,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.bank_truncation.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
OUT = ROOT / "results/architecture_sweep/BANK_TRUNCATION.json"

SIZES = (13, 26, 52, 104, 208, 416, 832, 1664, 3328, 5152)
RULES = ("by multiplicity", "by gini", "random")
SUBSET_SEED = 20260730


def table_outputs(D, tables, offsets, frac, n_levels, cols=None):
    """Yield ``(a, b, values)`` for a chosen column subset of the bank.

    ``cols`` indexes into ``tables``. Addresses are recomputed per block, as in
    the deployed bank, so nothing of size rows-by-tables is ever materialised.
    """
    idx = range(len(tables)) if cols is None else cols
    sel = [(k, tables[k]) for k in idx]
    n = D.shape[0]
    for a in range(0, n, BLOCK):
        b = min(a + BLOCK, n)
        out = np.empty((b - a, len(sel)), dtype=np.float64)
        for j, (k, cs) in enumerate(sel):
            acc = np.zeros(b - a, dtype=np.int64)
            for t, c in enumerate(cs):
                acc += D[a:b, c].astype(np.int64) * (n_levels ** t)
            out[:, j] = frac[acc + offsets[k]]
        yield a, b, out


def solve_multiplicities(D, y, tables, offsets, frac, n_levels, cols):
    """The deployed fan-out, restricted to a subset of the bank."""
    K = len(cols)
    s1 = np.zeros(K)
    s0 = np.zeros(K)
    pos = y == 1
    n1 = int(pos.sum())
    n0 = int(len(y) - n1)
    for a, b, v in table_outputs(D, tables, offsets, frac, n_levels, cols):
        p = pos[a:b]
        s1 += v[p].sum(0)
        s0 += v[~p].sum(0)
    mu1, mu0 = s1 / max(n1, 1), s0 / max(n0, 1)
    S = np.zeros((K, K))
    for a, b, v in table_outputs(D, tables, offsets, frac, n_levels, cols):
        p = pos[a:b]
        c = np.where(p[:, None], v - mu1, v - mu0)
        S += c.T @ c
    S /= max(len(y) - 2, 1)
    S.flat[::K + 1] += RIDGE * float(np.trace(S)) / K + 1e-12
    w = np.linalg.solve(S, mu1 - mu0)
    peak = float(np.abs(w).max())
    if peak <= 0:
        return np.zeros(K, dtype=np.int64)
    return np.round(w / peak * FAN_OUT_CAP).astype(np.int64)


def score_subset(D, tables, offsets, frac, mult, n_levels, cols):
    out = np.empty(D.shape[0], dtype=np.float64)
    m = mult.astype(np.float64)
    for a, b, v in table_outputs(D, tables, offsets, frac, n_levels, cols):
        out[a:b] = v @ m
    return out


def per_table_gini(D, y, tables, offsets, frac, n_levels):
    """``V`` for each table on its own, from integer counts.

    The same functional the fan-out ranking uses. Computed from the compiled cell
    counts rather than from the outputs, so it costs one pass.
    """
    n = D.shape[0]
    p = float(y.mean())
    K = len(tables)
    v = np.zeros(K)
    yf = y.astype(np.float64)
    total = int(offsets[-1])
    tot = np.zeros(total, dtype=np.float64)
    posc = np.zeros(total, dtype=np.float64)
    for a in range(0, n, BLOCK):
        b = min(a + BLOCK, n)
        for k, cs in enumerate(tables):
            acc = np.zeros(b - a, dtype=np.int64)
            for t, c in enumerate(cs):
                acc += D[a:b, c].astype(np.int64) * (n_levels ** t)
            ad = acc + offsets[k]
            tot += np.bincount(ad, minlength=total)
            posc += np.bincount(ad, weights=yf[a:b], minlength=total)
    for k in range(K):
        lo, hi = int(offsets[k]), int(offsets[k + 1])
        nc, kc = tot[lo:hi], posc[lo:hi]
        num = (kc - p * nc) ** 2
        v[k] = float(np.where(nc > 0, num / np.maximum(nc, 1.0), 0.0).sum() / n)
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=0)
    ap.add_argument("--rules", type=str, default="")
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args(argv)

    cdoc = json.loads(COUNTING.read_text())
    frozen = {int(k.split()[-2]): np.asarray(v, dtype=float)
              for k, v in cdoc["per_split"].items()}
    n_splits = a.splits or cdoc["protocol"]["n_splits"]
    rules = ([s.strip() for s in a.rules.split(",") if s.strip()]
             if a.rules else list(RULES))
    unknown = [r for r in rules if r not in RULES]
    if unknown:
        raise SystemExit(f"unknown rules {unknown}; known: {list(RULES)}")

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    n_wires = int(W.shape[1])
    if n_wires not in frozen:
        raise SystemExit(f"frozen artifact has widths {sorted(frozen)}")

    tables = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                              PARTITION_SEED)
    offsets = offsets_at(tables, N_LEVELS)
    K_full = len(tables)
    sizes = [s for s in SIZES if s <= K_full]
    if K_full not in sizes:
        sizes.append(K_full)

    t0 = time.perf_counter()
    D = chain_digits(np.asarray(W, dtype=np.float64), n_res)
    print(f"banded {n_wires} wires in {time.perf_counter() - t0:.0f}s; "
          f"{K_full} tables; sizes {sizes}; rules {rules}", flush=True)
    row = np.repeat(np.arange(len(n_res)), n_res)
    rng = np.random.default_rng(SUBSET_SEED)
    random_order = rng.permutation(K_full)

    got: dict[str, dict[int, list[float]]] = {
        r: {s: [] for s in sizes} for r in rules}

    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        Dfit, yfit = D[fit], y[fit]

        t1 = time.perf_counter()
        frac, _tc = compile_at(Dfit, yfit, tables, offsets, N_LEVELS)
        full_mult = solve_multiplicities(Dfit, yfit, tables, offsets, frac,
                                        N_LEVELS, list(range(K_full)))
        by_mult = np.argsort(-np.abs(full_mult), kind="stable")
        gini = (per_table_gini(Dfit, yfit, tables, offsets, frac, N_LEVELS)
                if "by gini" in rules else None)
        by_gini = np.argsort(-gini, kind="stable") if gini is not None else None
        print(f"  split {s + 1}/{n_splits}  compile+full solve "
              f"{time.perf_counter() - t1:.0f}s  deployed(frozen) "
              f"{frozen[n_wires][s]:.4f}", flush=True)

        orders = {"by multiplicity": by_mult, "by gini": by_gini,
                  "random": random_order}
        for rule in rules:
            order = orders[rule]
            for k in sizes:
                cols = sorted(int(c) for c in order[:k])
                t2 = time.perf_counter()
                mult = solve_multiplicities(Dfit, yfit, tables, offsets, frac,
                                            N_LEVELS, cols)
                sc = apply_gate(score_subset(D[pick], tables, offsets, frac,
                                            mult, N_LEVELS, cols),
                                ctr[pick], n_pick)
                auc = float(per_unit_auc(sc, y[pick], n_pick))
                got[rule][k].append(auc)
                print(f"      {rule:16s} K'={k:5d}  {auc:.4f}  "
                      f"{auc - frozen[n_wires][s]:+.4f}  "
                      f"{time.perf_counter() - t2:.0f}s", flush=True)

    base = frozen[n_wires][:n_splits]

    def stat(v):
        v = np.asarray(v)
        d = v - base
        return {"mean": round(float(v.mean()), 6),
                "delta_mean": round(float(d.mean()), 6),
                "delta_min": round(float(d.min()), 6),
                "delta_max": round(float(d.max()), 6),
                "n_splits_positive": int((d > 0).sum()),
                "n_splits": int(len(d))}

    curves = {r: {str(k): stat(v) for k, v in got[r].items()} for r in rules}

    # The full bank by multiplicity is the deployed bank and must return it.
    repro = None
    if "by multiplicity" in rules and K_full in got["by multiplicity"]:
        d = np.abs(np.asarray(got["by multiplicity"][K_full]) - base)
        repro = {
            "arm": f"by multiplicity, K'={K_full}",
            "max_absolute_difference": round(float(d.max()), 9),
            "reproduces_the_deployed_arm": bool(d.max() < 1e-6),
            "why_this_is_here": "the largest subset is the whole bank, so this "
                                "arm must return the frozen deployed numbers. "
                                "Running it rather than assuming it is what "
                                "makes the truncated arms comparable to them",
        }

    # The smallest K' whose interval of per-split deltas does not fall below a
    # stated tolerance. Reported per rule so a plateau is a fact and not a
    # reading of a graph.
    def plateau(rule, tol):
        for k in sorted(got[rule]):
            if float(np.mean(np.asarray(got[rule][k]) - base)) >= -tol:
                return k
        return None

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "GRAM_CONDITIONING.json found 76.5 per cent of the scatter "
                    "trace in the top one per cent of its directions and a "
                    "solved-direction cosine of 0.025 on every bank including "
                    "the deployed one. If the readout uses only a small "
                    "effective subspace, how many of the 5,152 tables does it "
                    "need",
        "why_it_matters": "if a bank of order 52 to 200 matches 5,152, the "
                          "deployed detector carries two orders of magnitude "
                          "more table than it uses, and 1.79 MB of integers "
                          "becomes tens of kilobytes at the same accuracy. That "
                          "is a stronger claim than the lifts being chased",
        "what_would_falsify_it": "accuracy falling off smoothly with K' instead "
                                 "of plateauing early, which would mean the "
                                 "trace concentration was misleading and the "
                                 "tables do distributed work the spectrum does "
                                 "not see",
        "selection_rules": {
            "by multiplicity": "absolute integer multiplicity from the full "
                               "5,152-table solve; the rule the spectrum "
                               "suggests, fit-half only, but it needs the full "
                               "solve to exist",
            "by gini": "each table's own V = sum_c (k_c - p n_c)^2 / (n_c N), "
                       "the integer-count functional the fan-out ranking "
                       "already uses. Needs no full solve, so a bank chosen "
                       "this way could be compiled directly at any target size. "
                       "This is the rule that would ship",
            "random": f"a subset of the same size at seed {SUBSET_SEED}. "
                      f"Without it a plateau proves nothing: if random subsets "
                      f"also match the full bank then the tables are "
                      f"interchangeable and neither ranking is doing work",
        },
        "held_fixed": {
            "n_wires": n_wires,
            "n_levels": N_LEVELS,
            "table_width": TABLE_WIDTH,
            "partition_rounds": PARTITION_ROUNDS,
            "partition_seed": PARTITION_SEED,
            "n_tables_full": K_full,
            "ridge": RIDGE,
            "fan_out_cap": FAN_OUT_CAP,
            "gate_radius": GATE_RADIUS,
            "gate_weight": GATE_WEIGHT,
            "cells_compiled_once": "cell frequencies are compiled over the whole "
                                   "bank per split and the subsets read the same "
                                   "numbers, so truncation changes which cells "
                                   "the score sums over and the multiplicities, "
                                   "and nothing about a cell",
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "select_and_solve_on": "the fit half only",
            "evaluate_on": "the pick half",
            "metric": "mean per-unit ROC-AUC",
            "baseline_was_not_recomputed": str(COUNTING.relative_to(ROOT)),
        },
        "sizes": sizes,
        "deployed_arm_frozen": {"mean": round(float(base.mean()), 6)},
        "curves": curves,
        "smallest_bank_within_tolerance": {
            f"{tol}": {r: plateau(r, tol) for r in rules}
            for tol in (0.001, 0.002, 0.005)
        },
        "reproduction_check": repro,
        "per_split": {r: {str(k): [round(float(x), 6) for x in v]
                          for k, v in got[r].items()} for r in rules},
        "per_split_deployed_frozen": [round(float(x), 6) for x in base],
        "n_units": int(len(n_res)),
        "n_residues": int(len(y)),
        "n_positive_residues": int(y.sum()),
    }

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    print(f"\n  deployed (frozen): {base.mean():.6f}")
    for r in rules:
        print(f"  {r}:")
        for k in sizes:
            c = curves[r][str(k)]
            print(f"    K'={k:5d}  {c['mean']:.6f}  {c['delta_mean']:+.6f}  "
                  f"on {c['n_splits_positive']}/{c['n_splits']}")
    print(f"\n  smallest bank within tolerance: "
          f"{json.dumps(doc['smallest_bank_within_tolerance'])}")
    if repro:
        print(f"  full bank reproduces deployed: "
              f"{repro['reproduces_the_deployed_arm']} "
              f"(max |diff| {repro['max_absolute_difference']:.2e})")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
