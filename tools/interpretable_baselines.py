#!/usr/bin/env python3
"""Is the accuracy in the architecture, or in the wires it reads?

The counting field is a pile of small lookup tables over quantised local
quantities. A reader who has seen a generalised additive model will ask the
obvious question: given the same 645 wires, would a logistic regression, or an
additive model over the same bins, do just as well? If so the interesting object
is the feature set and the table machinery is decoration.

This answers that on the training partition, with the published configuration
frozen and the held-out fold unread. Every arm below sees the same wires, the
same seeded cluster-disjoint halving, the same quantisation where it quantises
at all, and the same gate search. The only thing that varies is the readout.

The ladder
----------
**Ridge direction on raw wires.** A linear readout on the continuous wires, no
bins anywhere. This is the control for "the features are doing the work".

**Logistic regression on raw wires.** The same linear readout fitted to the
likelihood a reader would expect, by Newton steps on the standardised wires.
Reported because a Fisher direction and a logistic fit are not the same
estimator and it would be convenient, but not honest, to report only whichever
of the two came out lower.

**Additive over bins (width 1).** One table per wire, four cells each, fused by
the same integer solve. This is a generalised additive model over exactly the
bins the field uses: it keeps the quantisation and throws away the interactions.

**One round of pairs (width 2, one partition).** Each wire meets exactly one
partner, once. This is a GA-squared-M with a single random pairing.

**Sixteen rounds of pairs.** The published field. The difference from the row
above is the whole of what repeated pair coverage buys.

**Sixteen rounds, continuous weights.** The published field with the integer
rounding removed from the fusion, which prices the fan-out cap.

**Spatial smoothing** is not an arm but a column: every arm is reported raw and
gated, so the smoothing is priced once per readout rather than once overall.

Nothing here is a selection. The published configuration was frozen long before
this file existed and is not revised by it, whichever row comes out highest.

Usage: PYTHONPATH=src:tools python3.12 tools/interpretable_baselines.py [--check]
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np

from pocket_bench.methods import table_bank
from pocket_bench.methods.table_bank import partition_tables
from pocket_bench.paths import ROOT

from counterattack_ridge import ridge_direction
from counterattack_select import SEED, per_unit_auc, roc_auc
from counterattack_ridge import spread_matched_gate
from sensitivity_sweep import (
    FROZEN,
    GATES,
    _gated_auc,
    _split,
    cell_offsets,
    compile_cells,
    digits,
    integer_fanout,
    score,
)

CACHE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
OUT = ROOT / "results/architecture_sweep/INTERPRETABLE_BASELINES.json"
SENS = ROOT / "results/architecture_sweep/SENSITIVITY_SWEEP.json"
SCHEMA = "geoaudit.interpretable_baselines.v1"

CHUNK = 8192
# Enough draws that the interval's own third decimal is stable, and a seed that
# is not the split's, so a lucky halving cannot be reused as a lucky resample.
N_BOOT = 4000
BOOT_SEED = SEED + 1


def _per_unit_aucs(score, y, n_res_per) -> list[float]:
    """The vector ``per_unit_auc`` averages over.

    Kept separately because a difference of two means over the same 96 chains
    is a paired quantity, and reporting it without the pairing throws away most
    of what makes it estimable.
    """
    out, off = [], 0
    for n in n_res_per:
        n = int(n)
        s, t = score[off:off + n], y[off:off + n]
        off += n
        if t.sum() == 0 or t.sum() == n:
            out.append(float("nan"))
            continue
        a = roc_auc(list(s), list(t))
        out.append(float("nan") if a is None else float(a))
    return out


def _paired_ci(a: list[float], b: list[float]) -> dict:
    """Bootstrap the paired mean difference over the chains both arms scored."""
    x = np.array(a, dtype=float)
    z = np.array(b, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(z))
    d = x[ok] - z[ok]
    if len(d) == 0:
        return {"n_paired": 0}
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    draws = d[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"n_paired": int(len(d)),
            "delta": round(float(d.mean()), 6),
            "ci95": [round(float(lo), 6), round(float(hi), 6)],
            "excludes_zero": bool(lo > 0 or hi < 0)}
# Newton on a 645-parameter logistic converges in a handful of steps; the cap is
# a guard against a pathological fit, not a schedule. The tolerance is on the
# largest coordinate move, so it does not loosen as the parameter count grows.
NEWTON_STEPS = 25
NEWTON_TOL = 1e-6
# The same relative ridge the field's own solve uses, so the linear arms are not
# handicapped by a regularisation choice made for a different estimator.
LOGIT_RIDGE = FROZEN["ridge"]


def _standardise(X, rows_fit):
    """Centre and scale by the fit half only.

    A statistic taken over both halves would carry information about the half
    the arm is scored on. It would barely move these numbers, and it would make
    the comparison unquotable.
    """
    n_fit = int(rows_fit.sum())
    mean = np.zeros(X.shape[1])
    for a in range(0, X.shape[0], CHUNK):
        b = min(a + CHUNK, X.shape[0])
        m = rows_fit[a:b]
        if m.any():
            mean += X[a:b][m].astype(np.float64).sum(0)
    mean /= max(n_fit, 1)
    var = np.zeros(X.shape[1])
    for a in range(0, X.shape[0], CHUNK):
        b = min(a + CHUNK, X.shape[0])
        m = rows_fit[a:b]
        if m.any():
            var += ((X[a:b][m].astype(np.float64) - mean) ** 2).sum(0)
    var /= max(n_fit - 1, 1)
    sd = np.sqrt(var)
    # A wire that never varies on the fit half carries no information and would
    # divide by zero; it is passed through as a column of zeros instead of being
    # dropped, so the coefficient vector stays aligned with the wire names.
    sd[sd <= 0] = 1.0
    return mean, sd


def _logistic(X, y, rows, mean, sd, ridge):
    """L2-penalised logistic regression by Newton steps, in float64 chunks.

    Written out rather than imported. The alternative was a new dependency for
    one number in one table, and every library the frozen numbers depend on is
    recorded in the environment artifact; adding one for a baseline would put a
    reported figure behind a pin nobody wrote down.
    """
    K = X.shape[1]
    w = np.zeros(K + 1)
    idx = np.flatnonzero(rows)
    yv = y[idx].astype(np.float64)
    n = len(idx)
    moves = []
    for _ in range(NEWTON_STEPS):
        g = np.zeros(K + 1)
        H = np.zeros((K + 1, K + 1))
        for a in range(0, n, CHUNK):
            sl = idx[a:a + CHUNK]
            Z = (X[sl].astype(np.float64) - mean) / sd
            Z = np.hstack([Z, np.ones((len(sl), 1))])
            eta = np.clip(Z @ w, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-eta))
            g += Z.T @ (yv[a:a + CHUNK] - p)
            H += (Z * (p * (1.0 - p))[:, None]).T @ Z
        # The penalty is on the wires, never on the intercept: shrinking the
        # intercept would pull the fitted base rate away from the observed one.
        pen = ridge * float(np.trace(H[:K, :K])) / max(K, 1) + 1e-9
        g[:K] -= pen * w[:K]
        H.flat[::K + 2] += pen
        H[K, K] -= pen
        step = np.linalg.solve(H, g)
        w += step
        moves.append(float(np.abs(step).max()))
        if moves[-1] < NEWTON_TOL:
            break
    return w, moves


def _apply_logistic(X, rows, mean, sd, w):
    out = np.empty(int(rows.sum()))
    idx = np.flatnonzero(rows)
    for a in range(0, len(idx), CHUNK):
        sl = idx[a:a + CHUNK]
        Z = (X[sl].astype(np.float64) - mean) / sd
        out[a:a + len(sl)] = Z @ w[:-1] + w[-1]
    return out


def _apply_linear(X, rows, w):
    out = np.empty(int(rows.sum()))
    idx = np.flatnonzero(rows)
    for a in range(0, len(idx), CHUNK):
        sl = idx[a:a + CHUNK]
        out[a:a + len(sl)] = X[sl].astype(np.float64) @ w
    return out


def _row(name, what, S, ypick, n_pick, ctr_pick, extra=None):
    raw = per_unit_auc(S, ypick, n_pick)
    gated, gname = _gated_auc(S, ctr_pick, n_pick, ypick)
    # The gate that won, applied again, so the per-chain vector behind the
    # reported mean is the vector of the number actually reported.
    r, wt = next((r, w) for r, w in GATES if gname == f"r{int(r)} w{w}")
    G = spread_matched_gate(S, ctr_pick, n_pick, r, wt)
    per_unit = _per_unit_aucs(G, ypick, n_pick)
    if abs(float(np.nanmean(per_unit)) - gated) > 1e-9:
        raise SystemExit(f"{name}: the per-chain vector does not average to "
                         f"the reported gated AUC; the gate was reapplied "
                         f"differently from the way it was searched")
    d = {"arm": name, "what_it_is": what,
         "pick_half_roc_auc_raw": raw,
         "pick_half_roc_auc": gated,
         "spatial_smoothing_gain": round(gated - raw, 6),
         "gate": gname,
         "per_unit_gated_auc": [None if np.isnan(v) else round(v, 6)
                                for v in per_unit]}
    d.update(extra or {})
    print(f"  {name:<28s} raw {raw:.4f}  gated {gated:.4f}  "
          f"(+{gated - raw:.4f})", flush=True)
    return d


def build() -> dict:
    t_start = time.perf_counter()
    with np.load(CACHE, allow_pickle=False) as z:
        y, n_res, ctr = z["y"], z["n_res_per"], z["ctr"]
        units = [str(u) for u in z["units"]]
        X = z["X"]
        n_wires = int(X.shape[1])

        is_fit, fm, pm = _split(units, n_res)
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        yfit, ypick, ctr_pick = y[fm], y[pm], ctr[pm]
        print(f"{n_wires} wires, fit {int(fm.sum())} rows, "
              f"pick {int(pm.sum())} rows", flush=True)

        rows = []

        # ---- the two linear arms, on the continuous wires ------------------
        print("linear readouts on the raw wires", flush=True)
        Xfit = np.ascontiguousarray(X[fm])
        w_lin = ridge_direction(Xfit, yfit, FROZEN["ridge"])
        del Xfit
        gc.collect()
        rows.append(_row(
            "ridge direction", "a linear readout on the 645 continuous wires, "
            "the same regularised class-separating direction the field's fusion "
            "solves for, but on the wires themselves rather than on cell rates",
            _apply_linear(X, pm, w_lin), ypick, n_pick, ctr_pick,
            {"n_parameters": n_wires, "quantised": False}))

        mean, sd = _standardise(X, fm)
        w_log, moves = _logistic(X, y, fm, mean, sd, LOGIT_RIDGE)
        rows.append(_row(
            "logistic regression", "L2-penalised logistic regression on the "
            "same 645 wires, standardised on the fit half, fitted by Newton "
            "steps to the tolerance recorded here",
            _apply_logistic(X, pm, mean, sd, w_log), ypick, n_pick, ctr_pick,
            {"n_parameters": n_wires + 1, "quantised": False,
             "newton_steps": len(moves),
             "largest_final_coefficient_move": round(moves[-1], 10),
             "converged": bool(moves[-1] < NEWTON_TOL)}))

        # ---- everything below shares one quantisation ----------------------
        D = digits(X, n_res, FROZEN["levels"], FROZEN["ranking"])
        del X
    gc.collect()
    Dfit = np.ascontiguousarray(D[fm])
    Dpick = np.ascontiguousarray(D[pm])
    del D
    gc.collect()

    banks = [
        ("additive over bins",
         [[j] for j in range(n_wires)],
         "one table per wire, four cells each: a generalised additive model "
         "over exactly the bins the field uses, with every interaction removed",
         1, 1),
        ("pairs, one round",
         partition_tables(n_wires, 2, 1, SEED),
         "one random partition of the wires into pairs, so each wire meets "
         "exactly one partner: pairwise interactions, covered once",
         2, 1),
        ("pairs, sixteen rounds",
         partition_tables(n_wires, FROZEN["width"], FROZEN["rounds"], SEED),
         "the published field: sixteen independent partitions into pairs, "
         "fused by the integer solve",
         FROZEN["width"], FROZEN["rounds"]),
    ]

    published_gated = None
    for name, tables, what, width, rounds in banks:
        print(f"{name}: {len(tables)} tables", flush=True)
        offsets = cell_offsets(tables, FROZEN["levels"])
        frac, tot = compile_cells(Dfit, yfit, tables, offsets, FROZEN["levels"])
        m = integer_fanout(Dfit, yfit, tables, offsets, frac,
                           FROZEN["ridge"], FROZEN["cap"], FROZEN["levels"])
        S = score(Dpick, tables, offsets, frac, m, FROZEN["levels"])
        r = _row(name, what, S, ypick, n_pick, ctr_pick,
                 {"width": width, "rounds": rounds, "n_tables": len(tables),
                  "n_cells": int(len(frac)),
                  "n_cells_never_addressed": int((tot == 0).sum()),
                  "n_tables_used": int((m != 0).sum()),
                  "quantised": True, "integer_fan_out": True})
        rows.append(r)

        if rounds == FROZEN["rounds"] and width == FROZEN["width"]:
            published_gated = r["pick_half_roc_auc"]
            # The same bank and the same solve, with the rounding removed. The
            # continuous direction is what the integer multiplicities are a
            # rounding of, so this prices the rounding and nothing else.
            wc = _continuous_fanout(Dfit, yfit, tables, offsets, frac,
                                    FROZEN["ridge"], FROZEN["levels"])
            Sc = score(Dpick, tables, offsets, frac, wc, FROZEN["levels"])
            rows.append(_row(
                "pairs, sixteen rounds, unrounded",
                "the published field with the integer rounding removed from "
                "the fusion: the continuous solve that the multiplicities are "
                "a rounding of, scored the same way",
                Sc, ypick, n_pick, ctr_pick,
                {"width": width, "rounds": rounds, "n_tables": len(tables),
                 "quantised": True, "integer_fan_out": False}))
        del frac, tot
        gc.collect()

    return _summarise(rows, published_gated, time.perf_counter() - t_start)


def _continuous_fanout(D, y, tables, offsets, frac, ridge, levels):
    """``integer_fanout`` up to, but not including, the rounding.

    Deliberately a copy of the shipped solve rather than a parameter on it: the
    shipped one returns integers by contract and several gates check that it
    does.
    """
    K = len(tables)
    s1 = np.zeros(K)
    s0 = np.zeros(K)
    pos = y == 1
    n1 = int(pos.sum())
    n0 = int(len(y) - n1)
    from sensitivity_sweep import blocks
    for a, b, v in blocks(D, tables, offsets, frac, levels):
        p = pos[a:b]
        s1 += v[p].sum(0)
        s0 += v[~p].sum(0)
    mu1, mu0 = s1 / max(n1, 1), s0 / max(n0, 1)
    S = np.zeros((K, K))
    for a, b, v in blocks(D, tables, offsets, frac, levels):
        p = pos[a:b]
        c = np.where(p[:, None], v - mu1, v - mu0)
        S += c.T @ c
    S /= max(len(y) - 2, 1)
    S.flat[::K + 1] += ridge * float(np.trace(S)) / K + 1e-12
    w = np.linalg.solve(S, mu1 - mu0)
    peak = float(np.abs(w).max())
    return w / peak * FROZEN["cap"] if peak > 0 else w


def _by(rows, name):
    return next(r for r in rows if r["arm"] == name)


def _summarise(rows, published_gated, seconds) -> dict:
    best = max(rows, key=lambda r: r["pick_half_roc_auc"])
    pub = _by(rows, "pairs, sixteen rounds")
    # A point difference between two arms on one half of one fold is not an
    # ordering. Every arm therefore carries its paired interval against the
    # published readout, over the same chains, so a row that cannot be
    # separated from it says so on its own line.
    against = {r["arm"]: _paired_ci(pub["per_unit_gated_auc"],
                                    r["per_unit_gated_auc"])
               for r in rows if r["arm"] != pub["arm"]}
    lin = max(_by(rows, "ridge direction")["pick_half_roc_auc"],
              _by(rows, "logistic regression")["pick_half_roc_auc"])
    add = _by(rows, "additive over bins")["pick_half_roc_auc"]
    one = _by(rows, "pairs, one round")["pick_half_roc_auc"]
    unr = _by(rows, "pairs, sixteen rounds, unrounded")["pick_half_roc_auc"]
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "question": "whether the counting field's accuracy is in its readout "
                    "or in the wires it reads, measured on the training "
                    "partition with the published configuration frozen",
        "frozen_configuration": FROZEN,
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED},
        "metric": "per-unit ROC-AUC on the pick half, reported before and "
                  "after the same gate search",
        "gates_searched": [{"radius": r, "weight": w} for r, w in GATES],
        "rows": rows,
        "published_readout_against_each_other_arm": against,
        "arms_it_cannot_be_separated_from": sorted(
            k for k, v in against.items() if not v["excludes_zero"]),
        "resampling": {"draws": N_BOOT, "seed": BOOT_SEED,
                       "unit": "one pick-half chain, paired across arms"},
        "what_each_step_is_worth": {
            "the_bins_and_the_tables_over_a_linear_readout": round(add - lin, 6),
            "pairing_the_wires_over_an_additive_model": round(one - add, 6),
            "repeating_the_pairing_sixteen_times": round(pub[
                "pick_half_roc_auc"] - one, 6),
            "rounding_the_fusion_to_integers": round(
                pub["pick_half_roc_auc"] - unr, 6),
            "the_spatial_mean_on_the_published_field": pub[
                "spatial_smoothing_gain"],
        },
        "best_arm": best["arm"],
        "best_pick_half_roc_auc": best["pick_half_roc_auc"],
        "published_pick_half_roc_auc": published_gated,
        "published_is_best": bool(best["arm"] == "pairs, sixteen rounds"),
        "selection_note": "the published configuration was frozen before this "
                          "comparison and is not revised by it, whichever arm "
                          "comes out highest",
        "not_an_ebm": "InterpretML's EBM is not among the arms. The additive "
                      "row is a generalised additive model over the same bins "
                      "and the pairwise rows are pairwise-interaction models "
                      "over the same bins, fitted by this repository's own "
                      "solve rather than by cyclic gradient boosting, so the "
                      "comparison isolates the readout structure and not the "
                      "fitting procedure. A boosted fit is a different "
                      "experiment and is not claimed here",
        "seconds": round(seconds, 1),
        "reads_test_fold": False,
    }


def _write(doc) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")


def check() -> int:
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    bad = []
    if d.get("schema") != SCHEMA:
        bad.append("unexpected schema")
    if d.get("reads_test_fold"):
        bad.append("this comparison must not read the held-out fold")
    rows = d.get("rows") or []
    names = {r["arm"] for r in rows}
    want = {"ridge direction", "logistic regression", "additive over bins",
            "pairs, one round", "pairs, sixteen rounds",
            "pairs, sixteen rounds, unrounded"}
    if not want <= names:
        bad.append(f"missing arms: {sorted(want - names)}")
    # Every arm must have been scored on the same fold halves by the same
    # metric, so a difference between two of them is a difference of readout.
    if d.get("frozen_configuration") != FROZEN:
        bad.append("the frozen configuration has moved since this was written")
    if d.get("split", {}).get("seed") != SEED:
        bad.append("the split seed does not match the one the field was "
                   "selected under")
    # The published arm has to be the published method. If the sweep and this
    # file disagree about what the frozen configuration scores on the same half,
    # one of them is not running the shipped code.
    if SENS.exists() and rows:
        s = json.loads(SENS.read_text()).get("published_pick_half_roc_auc")
        p = d.get("published_pick_half_roc_auc")
        if s is not None and p is not None and abs(s - p) > 1e-9:
            bad.append(f"the published arm scores {p} here and {s} in the "
                       f"sensitivity sweep; the two are meant to be the same "
                       f"configuration on the same half")
    # The summary differences have to be the rows subtracted, not a retelling.
    if want <= names:
        w = d["what_each_step_is_worth"]
        pub = _by(rows, "pairs, sixteen rounds")["pick_half_roc_auc"]
        one = _by(rows, "pairs, one round")["pick_half_roc_auc"]
        unr = _by(rows, "pairs, sixteen rounds, unrounded")["pick_half_roc_auc"]
        add = _by(rows, "additive over bins")["pick_half_roc_auc"]
        for key, got in (
                ("repeating_the_pairing_sixteen_times", pub - one),
                ("rounding_the_fusion_to_integers", pub - unr),
                ("pairing_the_wires_over_an_additive_model", one - add)):
            if abs(w[key] - got) > 5e-6:
                bad.append(f"{key} is recorded as {w[key]} and the rows give "
                           f"{got:.6f}")
        if abs(w["the_spatial_mean_on_the_published_field"]
               - (pub - _by(rows, "pairs, sixteen rounds")[
                   "pick_half_roc_auc_raw"])) > 5e-6:
            bad.append("the spatial-mean gain is not the published arm's own "
                       "gated minus raw")
        best = max(rows, key=lambda r: r["pick_half_roc_auc"])
        if (best["arm"] == "pairs, sixteen rounds") != d.get("published_is_best"):
            bad.append("the artifact disagrees with its own rows about whether "
                       "the published readout is the best one")
    # Every reported mean has to be the mean of the vector stored beside it,
    # and every interval has to be about the arm it names.
    for r in rows:
        v = [x for x in (r.get("per_unit_gated_auc") or []) if x is not None]
        if not v:
            bad.append(f"{r['arm']} stores no per-chain vector")
        elif abs(sum(v) / len(v) - r["pick_half_roc_auc"]) > 1e-5:
            bad.append(f"{r['arm']}: the stored per-chain vector averages to "
                       f"{sum(v) / len(v):.6f}, not the reported "
                       f"{r['pick_half_roc_auc']}")
    against = d.get("published_readout_against_each_other_arm") or {}
    if rows and set(against) != {r["arm"] for r in rows} - {
            "pairs, sixteen rounds"}:
        bad.append("the paired intervals do not cover every other arm")
    unresolved = sorted(k for k, v in against.items()
                        if not v.get("excludes_zero"))
    if unresolved != (d.get("arms_it_cannot_be_separated_from") or []):
        bad.append("the list of arms the published readout cannot be "
                   "separated from disagrees with the intervals above it")
    for b in bad:
        print(f"FAIL {OUT.relative_to(ROOT)}: {b}")
    if bad:
        return 1
    print(f"OK {OUT.relative_to(ROOT)}: {len(rows)} readouts on one split, "
          f"published {d['published_pick_half_roc_auc']:.4f}, best "
          f"{d['best_arm']} {d['best_pick_half_roc_auc']:.4f}, test fold unread")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        return check()
    doc = build()
    _write(doc)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    w = doc["what_each_step_is_worth"]
    for k, v in w.items():
        print(f"  {k:<50s} {v:+.4f}")
    print("\nthe published readout against each other arm, paired over the "
          "same chains")
    for arm, ci in doc["published_readout_against_each_other_arm"].items():
        mark = "" if ci["excludes_zero"] else "   <- contains zero"
        print(f"  {arm:<34s} {ci['delta']:+.4f} "
              f"[{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]{mark}")
    if not doc["published_is_best"]:
        print(f"\nthe published readout is not the best arm here: "
              f"{doc['best_arm']} scores {doc['best_pick_half_roc_auc']:.4f} "
              f"against {doc['published_pick_half_roc_auc']:.4f}. That is "
              f"reported, not adopted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
