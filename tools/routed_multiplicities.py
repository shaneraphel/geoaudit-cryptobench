#!/usr/bin/env python3.12
"""One set of integer multiplicities per region instead of one globally.

Why this is the only structural change the measurements have not excluded
-----------------------------------------------------------------------
Five results now stand, all on training folds, all with everything else held.
The quantisation ladder is at its optimum for the count budget
(QUANTISATION_LADDER.json). Pairing choice is seed noise at the magnitude
available (SELECTED_PAIRINGS.json). Appended column families are visible to a
linear solve and invisible to the field, on two families of different
mathematical character (COMPOSITION_WIRES.json). Integer rounding of the
multiplicities is free at cos 0.9992 and the ridge contributes only 4.1 per cent
of the diagonal (GRAM_CONDITIONING.json). And the bank is not compressible:
accuracy is smooth and roughly logarithmic in the table count with no knee, and
at 52 tables the loss is 0.0324 (BANK_TRUNCATION.json).

Every parameter varied in those five is a *global* choice applied identically to
all 5,152 tables: one ladder, one pairing draw, one attachment, one solve, one
table count. What has not been varied is that the assignment is global at all.

The reason to suspect it is a number from GRAM_CONDITIONING.json:
cos(m, mu_1 - mu_0) = 0.025 on every bank including the deployed one. The solved
direction is nearly orthogonal to the class mean difference. One global direction
nearly orthogonal to the class mean is what a single direction fitted to
heterogeneous regions looks like -- a table that separates cryptic from
non-cryptic residues in a buried core and anti-separates them on an exposed loop
receives one weight, and the solve is left reconciling the two.

The construction
----------------
A region is itself a quantised cell, which keeps the change inside the
architecture's own language. Choose R router wires by a stated rule on the fit
half; a residue's region is the tuple of its digits on those wires, so the region
index is read off the residue's own address and needs no information the detector
does not already have. With one router wire and four levels there are four
regions.

Cell frequencies are compiled ONCE on the whole fit half and shared by every
region. Only the multiplicities are per-region:

    S(i) = sum_k m_k^{(g(i))} f_k[a_k(i)],   g(i) the region of residue i.

This matters for what is being tested. If the frequencies were also per-region
the cells would thin by a factor of R and the experiment would confound routing
with count starvation, which QUANTISATION_LADDER.json already showed is the
dominant effect when cells thin. Holding the frequencies global isolates the
assignment.

At inference the detector is still exact integer arithmetic over table cells. It
carries R integer vectors instead of one, which is R x 5,152 integers -- tens of
kilobytes -- against the 1.79 MB the cell counts already occupy.

Controls, because R regions is also R times the parameters
---------------------------------------------------------
``global`` is R = 1 and must reproduce the frozen deployed numbers. Run as an arm
rather than assumed.

``random router`` partitions residues into the same number of regions of the same
sizes at a fixed seed, ignoring geometry. If random routing gains as much as
informed routing, the gain is capacity and not structure, and the experiment has
found nothing. This is the arm that decides whether the result means anything.

``chain-size router`` partitions by chain length quartile, a routing that is real
but has nothing to do with the local environment. It separates "any partition
helps" from "this partition helps".

What would falsify the hypothesis
---------------------------------
Informed routing failing to beat both the global arm and the random router. In
that case the global assignment is not the constraint either, every parameter of
the construction has been measured at its optimum, and the deficit is not
reachable from this architecture without a change of a kind not yet imagined.
That is a useful outcome and it is why the controls are in the same run.

Training folds only. No test residue and no external unit is read.

Usage: PYTHONPATH=src:tools python3.12 tools/routed_multiplicities.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bank_truncation import table_outputs  # noqa: E402
from expand_invariant_bank import SEED  # noqa: E402
from quantisation_ladder import compile_at, offsets_at  # noqa: E402
from select_architecture_on_train import cluster_half_split, per_unit_auc  # noqa: E402

from pocket_bench.methods.table_bank import (
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

SCHEMA = "geoaudit.routed_multiplicities.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
OUT = ROOT / "results/architecture_sweep/ROUTED_MULTIPLICITIES.json"
ROUTER_SEED = 20260731
ARMS = ("global", "gini router", "random router", "chain-size router")


def solve_on_rows(D, y, tables, offsets, frac, n_levels, rows) -> np.ndarray:
    """The deployed fan-out restricted to a subset of rows.

    Identical arithmetic to ``integer_fanout``; only which rows enter the class
    means and the scatter changes. Returns integers on ``[-cap, cap]``.
    """
    K = len(tables)
    Dr, yr = D[rows], y[rows]
    if len(yr) < 2 or yr.sum() == 0 or yr.sum() == len(yr):
        return np.zeros(K, dtype=np.int64)
    s1 = np.zeros(K)
    s0 = np.zeros(K)
    pos = yr == 1
    n1, n0 = int(pos.sum()), int(len(yr) - int(pos.sum()))
    for a, b, v in table_outputs(Dr, tables, offsets, frac, n_levels):
        p = pos[a:b]
        s1 += v[p].sum(0)
        s0 += v[~p].sum(0)
    mu1, mu0 = s1 / max(n1, 1), s0 / max(n0, 1)
    S = np.zeros((K, K))
    for a, b, v in table_outputs(Dr, tables, offsets, frac, n_levels):
        p = pos[a:b]
        c = np.where(p[:, None], v - mu1, v - mu0)
        S += c.T @ c
    S /= max(len(yr) - 2, 1)
    S.flat[::K + 1] += RIDGE * float(np.trace(S)) / K + 1e-12
    w = np.linalg.solve(S, mu1 - mu0)
    peak = float(np.abs(w).max())
    if peak <= 0:
        return np.zeros(K, dtype=np.int64)
    return np.round(w / peak * FAN_OUT_CAP).astype(np.int64)


def score_routed(D, tables, offsets, frac, mults, region, n_levels):
    """Score with each residue reading its own region's multiplicities."""
    out = np.empty(D.shape[0], dtype=np.float64)
    M = np.stack([m.astype(np.float64) for m in mults])
    for a, b, v in table_outputs(D, tables, offsets, frac, n_levels):
        out[a:b] = np.einsum("rk,rk->r", v, M[region[a:b]])
    return out


def chain_size_regions(n_res_per, n_regions: int) -> np.ndarray:
    """Region per residue by the quartile of its chain's length."""
    lens = np.asarray([int(n) for n in n_res_per], dtype=np.int64)
    order = np.argsort(lens, kind="stable")
    band = np.empty(len(lens), dtype=np.int64)
    edges = np.linspace(0, len(lens), n_regions + 1).astype(int)
    for r in range(n_regions):
        band[order[edges[r]:edges[r + 1]]] = r
    return np.repeat(band, lens)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=0)
    ap.add_argument("--arms", type=str, default="")
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args(argv)

    cdoc = json.loads(COUNTING.read_text())
    frozen = {int(k.split()[-2]): np.asarray(v, dtype=float)
              for k, v in cdoc["per_split"].items()}
    n_splits = a.splits or cdoc["protocol"]["n_splits"]
    wanted = ([s.strip() for s in a.arms.split(",") if s.strip()]
              if a.arms else list(ARMS))
    unknown = [w for w in wanted if w not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {list(ARMS)}")

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    names = [str(s) for s in z["names"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    n_wires = int(W.shape[1])
    if n_wires not in frozen:
        raise SystemExit(f"frozen artifact has widths {sorted(frozen)}")

    tables = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                              PARTITION_SEED)
    offsets = offsets_at(tables, N_LEVELS)
    t0 = time.perf_counter()
    D = chain_digits(np.asarray(W, dtype=np.float64), n_res)
    print(f"banded {n_wires} wires in {time.perf_counter() - t0:.0f}s; "
          f"{len(tables)} tables; {N_LEVELS} regions when routed; "
          f"arms {wanted}", flush=True)

    row = np.repeat(np.arange(len(n_res)), n_res)
    size_region_all = chain_size_regions(n_res, N_LEVELS)
    rng = np.random.default_rng(ROUTER_SEED)
    random_region_all = rng.integers(0, N_LEVELS, size=len(y))

    got: dict[str, list[float]] = {k: [] for k in wanted}
    routers: list[dict] = []

    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        Dfit, yfit = D[fit], y[fit]

        t1 = time.perf_counter()
        frac, _tc = compile_at(Dfit, yfit, tables, offsets, N_LEVELS)
        print(f"  split {s + 1}/{n_splits}  compile "
              f"{time.perf_counter() - t1:.0f}s  deployed(frozen) "
              f"{frozen[n_wires][s]:.4f}", flush=True)

        # Router chosen by a stated rule on the fit half: the single wire whose
        # own between-cell label variance is largest. Not tuned, and recorded per
        # split so the choice can be checked. Computed by direct counting rather
        # than through per_table_gini, which wants a compiled cell array for a
        # whole bank; for one wire at a time the four counts are the cheap route.
        p = float(yfit.mean())
        v_wire = np.empty(n_wires)
        for j in range(n_wires):
            dj = Dfit[:, j]
            nc = np.bincount(dj, minlength=N_LEVELS).astype(float)
            kc = np.bincount(dj, weights=yfit.astype(float),
                             minlength=N_LEVELS)
            num = (kc - p * nc) ** 2
            v_wire[j] = float(np.where(nc > 0, num / np.maximum(nc, 1.0),
                                       0.0).sum() / len(yfit))
        router = int(np.argmax(v_wire))
        routers.append({"split": s + 1, "wire_index": router,
                        "wire_name": names[router],
                        "own_variance": round(float(v_wire[router]), 9)})
        gini_region_all = D[:, router].astype(np.int64)
        print(f"      router wire: {names[router]} (index {router})",
              flush=True)

        region_all = {
            "gini router": gini_region_all,
            "random router": random_region_all,
            "chain-size router": size_region_all,
        }

        for arm in wanted:
            t2 = time.perf_counter()
            if arm == "global":
                m = solve_on_rows(D, y, tables, offsets, frac, N_LEVELS,
                                  np.flatnonzero(fit))
                sc = score_routed(D[pick], tables, offsets, frac, [m],
                                  np.zeros(int(pick.sum()), dtype=np.int64),
                                  N_LEVELS)
            else:
                reg = region_all[arm]
                fit_idx = np.flatnonzero(fit)
                mults = []
                for r in range(N_LEVELS):
                    rows = fit_idx[reg[fit_idx] == r]
                    mults.append(solve_on_rows(D, y, tables, offsets, frac,
                                               N_LEVELS, rows))
                sc = score_routed(D[pick], tables, offsets, frac, mults,
                                  reg[pick], N_LEVELS)
            sc = apply_gate(sc, ctr[pick], n_pick)
            got[arm].append(float(per_unit_auc(sc, y[pick], n_pick)))
            print(f"      {arm:20s} {got[arm][-1]:.4f}  "
                  f"{got[arm][-1] - frozen[n_wires][s]:+.4f}  "
                  f"{time.perf_counter() - t2:.0f}s", flush=True)

    base = frozen[n_wires][:n_splits]

    def summarise(v):
        v = np.asarray(v)
        return {"mean": round(float(v.mean()), 6),
                "min": round(float(v.min()), 6),
                "max": round(float(v.max()), 6)}

    def compare(v, ref):
        d = np.asarray(v) - np.asarray(ref)
        return {"mean": round(float(d.mean()), 6),
                "min": round(float(d.min()), 6),
                "max": round(float(d.max()), 6),
                "n_splits_positive": int((d > 0).sum()),
                "n_splits": int(len(d)),
                "positive_on_every_split": bool((d > 0).all())}

    repro = None
    if "global" in got:
        d = np.abs(np.asarray(got["global"]) - base)
        repro = {
            "arm": "global",
            "max_absolute_difference": round(float(d.max()), 9),
            "reproduces_the_deployed_arm": bool(d.max() < 1e-6),
            "why_this_is_here": "the global arm is one region and is therefore "
                                "the deployed detector. It must return the "
                                "frozen numbers, and running it rather than "
                                "assuming it is what makes the routed arms "
                                "comparable",
        }

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "five measurements have found every global parameter of this "
                    "construction at or near its optimum, and the solved "
                    "direction is nearly orthogonal to the class mean difference "
                    "on every bank including the deployed one. Does giving each "
                    "region its own integer multiplicities recover anything",
        "why_this_is_the_remaining_change": {
            "quantisation": "results/architecture_sweep/QUANTISATION_LADDER.json",
            "pairings": "results/architecture_sweep/SELECTED_PAIRINGS.json",
            "appended_columns": "results/architecture_sweep/COMPOSITION_WIRES.json",
            "rounding_and_ridge": "results/architecture_sweep/GRAM_CONDITIONING.json",
            "bank_size": "results/architecture_sweep/BANK_TRUNCATION.json",
            "what_they_share": "each varies a choice applied identically to all "
                               "5,152 tables. None varies the fact that the "
                               "assignment is global",
            "the_number_that_motivates_it": "cos(m, mu_1 - mu_0) = 0.025, which "
                                            "is what one direction fitted to "
                                            "heterogeneous regions looks like",
        },
        "construction": {
            "n_regions": N_LEVELS,
            "region_of_a_residue": "the digit of one router wire, so the region "
                                   "index is read off the residue's own address "
                                   "and needs no information the detector does "
                                   "not already have",
            "router_rule": "the single wire whose own between-cell label "
                           "variance on the fit half is largest; stated, not "
                           "tuned, and recorded per split",
            "cell_frequencies": "compiled once on the whole fit half and shared "
                                "by every region",
            "why_frequencies_stay_global": "per-region frequencies would thin "
                                           "every cell by a factor of R, and "
                                           "QUANTISATION_LADDER.json showed cell "
                                           "thinning is the dominant effect when "
                                           "it happens. Holding them global "
                                           "isolates the assignment from count "
                                           "starvation",
            "inference_cost": f"{N_LEVELS} integer vectors of {len(tables)} "
                              f"entries instead of one; tens of kilobytes "
                              f"against the 1.79 MB the cell counts occupy",
            "still_integer_at_inference": True,
        },
        "controls": {
            "random router": f"residues partitioned into {N_LEVELS} regions at "
                             f"seed {ROUTER_SEED}, ignoring geometry. R regions "
                             f"is also R times the parameters, so if random "
                             f"routing gains as much the gain is capacity and "
                             f"not structure. This arm decides whether the "
                             f"result means anything",
            "chain-size router": "partition by chain-length quartile: a real "
                                 "partition with nothing to do with the local "
                                 "environment, which separates 'any partition "
                                 "helps' from 'this partition helps'",
        },
        "what_would_falsify_the_hypothesis": "informed routing failing to beat "
                                             "both the global arm and the random "
                                             "router. Then the global assignment "
                                             "is not the constraint either, every "
                                             "parameter of the construction has "
                                             "been measured at its optimum, and "
                                             "the deficit is not reachable from "
                                             "this architecture",
        "held_fixed": {
            "n_wires": n_wires,
            "n_levels": N_LEVELS,
            "table_width": TABLE_WIDTH,
            "partition_rounds": PARTITION_ROUNDS,
            "partition_seed": PARTITION_SEED,
            "n_tables": len(tables),
            "ridge": RIDGE,
            "fan_out_cap": FAN_OUT_CAP,
            "gate_radius": GATE_RADIUS,
            "gate_weight": GATE_WEIGHT,
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "route_and_solve_on": "the fit half only",
            "evaluate_on": "the pick half",
            "metric": "mean per-unit ROC-AUC",
            "baseline_was_not_recomputed": str(COUNTING.relative_to(ROOT)),
        },
        "routers_chosen": routers,
        "arms": {k: summarise(v) for k, v in got.items()},
        "deployed_arm_frozen": summarise(base),
        "minus_deployed": {k: compare(v, base) for k, v in got.items()},
        "minus_random_router": ({k: compare(got[k], got["random router"])
                                 for k in got if k != "random router"}
                                if "random router" in got else None),
        "reproduction_check": repro,
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
        print(f"  {k:20s} {doc['arms'][k]['mean']:.6f}  {c['mean']:+.6f}  on "
              f"{c['n_splits_positive']}/{c['n_splits']}")
    if doc["minus_random_router"]:
        print("\n  against the random router:")
        for k, c in doc["minus_random_router"].items():
            print(f"    {k:20s} {c['mean']:+.6f}  on "
                  f"{c['n_splits_positive']}/{c['n_splits']}")
    if repro:
        print(f"\n  global arm reproduces deployed: "
              f"{repro['reproduces_the_deployed_arm']} "
              f"(max |diff| {repro['max_absolute_difference']:.2e})")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
