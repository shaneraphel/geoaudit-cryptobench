#!/usr/bin/env python3.12
"""Move the quantisation cut points. Training folds only.

The hypothesis this tests, and where it came from
------------------------------------------------
Every wire becomes a quaternary digit by within-chain rank cut at quartiles, so
the extreme level of a wire is the extreme 25 % of its chain. Cryptic residues
are 5.76 % of the fold. Measured on the training cache over the 24 wires with
the largest decile enrichment, the extreme 2 % of a wire is far richer than the
extreme quartile that the deployed cut lumps it into:

    void          top 2 %  0.2934    top 25 %  0.1291    2.27x
    void~d20      top 2 %  0.2579    top 25 %  0.1201    2.15x
    depth~d20     bot 2 %  0.2187    bot 25 %  0.1235    1.77x
    angle_deficit~d6  bot 2 %  0.2147  bot 25 %  0.1006  2.13x

Against a base rate of 0.0576 the extreme 2 % of ``void`` is 5.1x enriched and
the deployed top level reaches 2.2x. A level that contains both a 29 % band and
a 9 % band states one frequency for both, and no integer weight over that cell
recovers the difference. That part is a hard information loss rather than a
weighting problem, which is why it is worth an arm.

What this does NOT assume
-------------------------
Enrichment inside a band is not ROC-AUC, and the field reads all four levels
with a signed weight, so a coarse ladder is not obviously worse at ranking. The
measurement above bounds what is lost inside a cell; it does not predict the
lift. That is what this tool measures.

Why the cheapest arms cost nothing
----------------------------------
Arms B, C and D keep four levels and width-2 tables, so the bank has exactly the
same 5,152 tables and 16 cells each. Only the rank fractions at which the digit
changes move. Nothing about the detector's size, its integer arithmetic or its
fold discipline changes, and no constant crosses the fold boundary: the cuts are
rank fractions of a chain against itself, as before.

Arm E adds levels, so its cells grow to 6**2 = 36 and the bank grows with them.
It is reported separately for that reason.

Comparison is against the frozen 645-wire arm rather than a recomputation of it,
on the same seeds, as IS_FISHER_A_CEILING.json and UNION_BANK_COUNTING_FIELD.json
already do. The tool checks the split count and the wire width before scoring.

Training folds only. No test residue and no external unit is read.

Usage: PYTHONPATH=src:tools python3.12 tools/quantisation_ladder.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from expand_invariant_bank import SEED  # noqa: E402
from select_architecture_on_train import cluster_half_split, per_unit_auc  # noqa: E402

from pocket_bench.methods.table_bank import BLOCK, partition_tables
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

SCHEMA = "geoaudit.quantisation_ladder.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
OUT = ROOT / "results/architecture_sweep/QUANTISATION_LADDER.json"

# Rank fractions at which the digit increments. The deployed ladder is the
# uniform quartile one and is named so the artifact can state that the arms are
# measured against the same construction, not merely a similar one.
LADDERS: dict[str, tuple[float, ...]] = {
    "uniform quartiles (deployed)": (0.25, 0.50, 0.75),
    "tails at 10 %": (0.10, 0.50, 0.90),
    "tails at 5 %": (0.05, 0.50, 0.95),
    "tails at 2 %": (0.02, 0.50, 0.98),
    "six levels, tails at 2 and 10 %": (0.02, 0.10, 0.50, 0.90, 0.98),
}
DEPLOYED = "uniform quartiles (deployed)"


def chain_digits_at(F: np.ndarray, n_res_per, cuts: tuple[float, ...]
                    ) -> np.ndarray:
    """Digits by within-chain rank, incrementing at each fraction in ``cuts``.

    A local copy of ``table_bank.chain_digits`` rather than a change to it.
    ``TABLE_FIELD.json`` carries a ``code_sha256`` over ``table_bank.py`` among
    seven other files, so editing that module to add a parameter would
    invalidate the compiled field for a reason unrelated to the field. Tie
    handling is identical: tied values share a mid-rank, so they cannot be split
    across two levels.
    """
    cuts = tuple(float(c) for c in cuts)
    if not all(0.0 < c < 1.0 for c in cuts):
        raise SystemExit(f"cuts must lie strictly inside (0, 1): {cuts}")
    if list(cuts) != sorted(cuts) or len(set(cuts)) != len(cuts):
        raise SystemExit(f"cuts must be strictly increasing: {cuts}")
    edges = np.asarray(cuts, dtype=np.float64)
    out = np.empty(F.shape, dtype=np.int8)
    off = 0
    for n in n_res_per:
        n = int(n)
        blk = F[off:off + n]
        r = np.empty(n)
        for j in range(F.shape[1]):
            x = blk[:, j]
            order = np.argsort(x, kind="stable")
            i = 0
            while i < n:
                k = i
                while k + 1 < n and x[order[k + 1]] == x[order[i]]:
                    k += 1
                r[order[i:k + 1]] = 0.5 * (i + k)
                i = k + 1
            out[off:off + n, j] = np.searchsorted(
                edges, r / max(n - 1, 1), side="right")
        off += n
    return out


def offsets_at(tables, n_levels: int) -> np.ndarray:
    """``cell_offsets`` for a ladder that is not necessarily quaternary."""
    sizes = [n_levels ** len(t) for t in tables]
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


def addresses_at(D, tables, offsets, n_levels: int, a: int, b: int
                 ) -> np.ndarray:
    out = np.empty((b - a, len(tables)), dtype=np.int64)
    for k, cols in enumerate(tables):
        acc = np.zeros(b - a, dtype=np.int64)
        for t, c in enumerate(cols):
            acc += D[a:b, c].astype(np.int64) * (n_levels ** t)
        out[:, k] = acc + offsets[k]
    return out


def compile_at(D, y, tables, offsets, n_levels: int):
    n = D.shape[0]
    total = int(offsets[-1])
    tot = np.zeros(total, dtype=np.int64)
    pos = np.zeros(total, dtype=np.float64)
    yf = y.astype(np.float64)
    for a in range(0, n, BLOCK):
        b = min(a + BLOCK, n)
        flat = addresses_at(D, tables, offsets, n_levels, a, b).ravel()
        tot += np.bincount(flat, minlength=total)
        pos += np.bincount(flat, weights=np.repeat(yf[a:b], len(tables)),
                           minlength=total)
    rate = float(yf.mean())
    frac = np.where(tot > 0, pos / np.maximum(tot, 1), rate)
    return frac, tot


def blocks_at(D, tables, offsets, frac, n_levels: int):
    n = D.shape[0]
    for a in range(0, n, BLOCK):
        b = min(a + BLOCK, n)
        yield a, b, frac[addresses_at(D, tables, offsets, n_levels, a, b)]


def fanout_at(D, y, tables, offsets, frac, n_levels: int) -> np.ndarray:
    K = len(tables)
    s1 = np.zeros(K)
    s0 = np.zeros(K)
    pos = y == 1
    n1 = int(pos.sum())
    n0 = int(len(y) - n1)
    for a, b, v in blocks_at(D, tables, offsets, frac, n_levels):
        p = pos[a:b]
        s1 += v[p].sum(0)
        s0 += v[~p].sum(0)
    mu1, mu0 = s1 / max(n1, 1), s0 / max(n0, 1)
    S = np.zeros((K, K))
    for a, b, v in blocks_at(D, tables, offsets, frac, n_levels):
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


def score_at(D, tables, offsets, frac, mult, n_levels: int) -> np.ndarray:
    out = np.empty(D.shape[0], dtype=np.float64)
    m = mult.astype(np.float64)
    for a, b, v in blocks_at(D, tables, offsets, frac, n_levels):
        out[a:b] = v @ m
    return out


def cell_occupancy(tot: np.ndarray) -> dict:
    """How many cells were never addressed, and the median count of those that
    were. A ladder that resolves the tail also thins the cells, and a cell that
    holds ten residues states a noisy frequency; this is the number that says
    whether an arm's ladder has gone too far."""
    seen = tot[tot > 0]
    return {
        "n_cells": int(tot.size),
        "n_cells_never_addressed": int((tot == 0).sum()),
        "fraction_never_addressed": round(float((tot == 0).mean()), 6),
        "median_count_of_addressed_cells": int(np.median(seen)) if seen.size
        else 0,
        "min_count_of_addressed_cells": int(seen.min()) if seen.size else 0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=0,
                    help="0 means as many as the frozen artifact used")
    ap.add_argument("--arms", type=str, default="",
                    help="comma-separated ladder names; default is all")
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args(argv)

    cdoc = json.loads(COUNTING.read_text())
    frozen = {int(k.split()[-2]): np.asarray(v, dtype=float)
              for k, v in cdoc["per_split"].items()}
    n_splits = a.splits or cdoc["protocol"]["n_splits"]

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    n_wires = int(W.shape[1])
    if n_wires not in frozen:
        raise SystemExit(
            f"the frozen artifact reports widths {sorted(frozen)}; the wide "
            f"cache carries {n_wires} wires and cannot be compared against it")

    wanted = ([s.strip() for s in a.arms.split(",") if s.strip()]
              if a.arms else list(LADDERS))
    unknown = [w for w in wanted if w not in LADDERS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {list(LADDERS)}")

    tables = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                              PARTITION_SEED)
    print(f"{len(tables)} tables of width {TABLE_WIDTH} over {n_wires} wires; "
          f"{n_splits} splits; arms: {len(wanted)}", flush=True)

    Wf = np.asarray(W, dtype=np.float64)
    got: dict[str, list[float]] = {}
    occupancy: dict[str, dict] = {}
    for name in wanted:
        cuts = LADDERS[name]
        n_levels = len(cuts) + 1
        offsets = offsets_at(tables, n_levels)
        t0 = time.perf_counter()
        D = chain_digits_at(Wf, n_res, cuts)
        print(f"\n{name}: {n_levels} levels, {n_levels ** TABLE_WIDTH} cells "
              f"per table, {int(offsets[-1])} cells total; banded in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        rows: list[float] = []
        for s in range(n_splits):
            is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
            row = np.repeat(np.arange(len(n_res)), n_res)
            fit, pick = is_fit[row], ~is_fit[row]
            n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
            t1 = time.perf_counter()
            frac, tot = compile_at(D[fit], y[fit], tables, offsets, n_levels)
            mult = fanout_at(D[fit], y[fit], tables, offsets, frac, n_levels)
            sc = apply_gate(score_at(D[pick], tables, offsets, frac, mult,
                                     n_levels), ctr[pick], n_pick)
            rows.append(float(per_unit_auc(sc, y[pick], n_pick)))
            if s == 0:
                occupancy[name] = cell_occupancy(tot)
            print(f"  split {s + 1}/{n_splits}  {rows[-1]:.4f}  "
                  f"deployed(frozen) {frozen[n_wires][s]:.4f}  "
                  f"{time.perf_counter() - t1:.0f}s", flush=True)
        got[name] = rows
        del D

    base = frozen[n_wires][:n_splits]

    def summarise(v):
        v = np.asarray(v)
        return {"mean": round(float(v.mean()), 6),
                "min": round(float(v.min()), 6),
                "max": round(float(v.max()), 6)}

    def compare(v):
        d = np.asarray(v) - base
        return {"mean": round(float(d.mean()), 6),
                "min": round(float(d.min()), 6),
                "max": round(float(d.max()), 6),
                "n_splits_positive": int((d > 0).sum()),
                "n_splits": int(len(d)),
                "positive_on_every_split": bool((d > 0).all())}

    # The deployed ladder is recomputed as one of the arms whenever it is asked
    # for, and it must reproduce the frozen numbers. If it does not, the two
    # tools disagree about the same quantity and no arm here can be trusted.
    reproduction = None
    if DEPLOYED in got:
        d = np.abs(np.asarray(got[DEPLOYED]) - base)
        reproduction = {
            "recomputed_the_deployed_ladder": True,
            "max_absolute_difference_from_frozen": round(float(d.max()), 9),
            "reproduces_frozen_arm": bool(d.max() < 1e-6),
            "why_this_is_here": (
                "this tool reimplements banding, addressing, compilation and "
                "fan-out locally rather than editing table_bank.py, which a "
                "code digest pins. Running the deployed ladder through the "
                "local copy and requiring the frozen numbers back is what "
                "makes the other arms comparable to them"),
        }

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "the deployed quantisation cuts every wire at within-chain "
                    "quartiles, and measured on the training cache the extreme "
                    "2 % of the strongest wires is about twice as rich in "
                    "cryptic residues as the extreme quartile containing it. "
                    "Does moving the cut points to resolve the tail lift the "
                    "counting field, at the same bank and the same cell count",
        "what_the_screen_measured": {
            "base_positive_rate": 0.0576,
            "examples": {
                "void": {"top_2_percent": 0.2934, "top_25_percent": 0.1291},
                "void~d20": {"top_2_percent": 0.2579,
                             "top_25_percent": 0.1201},
                "depth~d20": {"bottom_2_percent": 0.2187,
                              "bottom_25_percent": 0.1235},
                "angle_deficit~d6": {"bottom_2_percent": 0.2147,
                                     "bottom_25_percent": 0.1006},
            },
            "what_it_bounds": "the frequency spread inside one cell, which no "
                              "integer weight over that cell can express",
            "what_it_does_not_predict": "the ROC-AUC lift, which is what the "
                                        "arms below measure",
        },
        "held_fixed_across_arms": {
            "n_wires": n_wires,
            "table_width": TABLE_WIDTH,
            "partition_rounds": PARTITION_ROUNDS,
            "partition_seed": PARTITION_SEED,
            "n_tables": len(tables),
            "ridge": RIDGE,
            "fan_out_cap": FAN_OUT_CAP,
            "gate_radius": GATE_RADIUS,
            "gate_weight": GATE_WEIGHT,
            "what_changes": "only the within-chain rank fractions at which the "
                            "digit increments, and for the six-level arm the "
                            "number of levels and therefore the cell count",
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "compile_on": "the fit half only",
            "evaluate_on": "the pick half",
            "metric": "mean per-unit ROC-AUC",
            "baseline_was_not_recomputed": str(COUNTING.relative_to(ROOT)),
            "banding": "per chain against that chain's own order statistics, "
                       "computed over the whole fold before splitting because "
                       "it cannot see the split",
        },
        "ladders": {k: list(LADDERS[k]) for k in wanted},
        "arms": {k: summarise(v) for k, v in got.items()},
        "deployed_arm_frozen": summarise(base),
        "minus_deployed": {k: compare(v) for k, v in got.items()},
        "cell_occupancy_on_split_1": occupancy,
        "reproduction_check": reproduction,
        "per_split": {k: [round(float(x), 6) for x in v]
                      for k, v in got.items()},
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
    for k in got:
        c = doc["minus_deployed"][k]
        print(f"  {k:34s} {doc['arms'][k]['mean']:.6f}  "
              f"{c['mean']:+.6f}  on {c['n_splits_positive']}/"
              f"{c['n_splits']} splits")
    if reproduction is not None:
        print(f"\n  reproduction of the deployed ladder: "
              f"{'OK' if reproduction['reproduces_frozen_arm'] else 'MISMATCH'}"
              f" (max |diff| "
              f"{reproduction['max_absolute_difference_from_frozen']:.2e})")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
