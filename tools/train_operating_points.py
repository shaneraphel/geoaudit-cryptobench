#!/usr/bin/env python3
"""Every threshold either method will be held to, chosen on the training fold.

The matched-operating-point read needed one threshold per method and got it from
a single pooled-F1 grid search. A reviewer asked for two things that search does
not supply. The first is MCC: F1 and MCC do not peak at the same calling
fraction in general, and a comparison that only ever optimises F1 cannot say
whether the ranking survives a different summary of the same confusion matrix.
The second is the asymmetry in how the two thresholds were obtained. Our q was
chosen on the fold whose residues had just been counted into the cells, so it is
in-sample for us; P2Rank never saw this fold at all, so its q is out-of-sample
for it. Selecting our q on data our cells were fitted on and P2Rank's on data it
was not is an advantage to us of unknown size, and "of unknown size" is not
something a paper gets to leave in a comparison it is relying on.

So this measures four thresholds per method:

  pooled F1  chosen on the whole training fold
  pooled MCC chosen on the whole training fold
  pooled F1  chosen on the pick half, with our cells counted on the fit half
  pooled MCC chosen the same way

The last two are the honest ones for our method, because the field they threshold
never saw the residues that chose the threshold. They cost a second compile.
P2Rank's numbers are computed on the same halves so the two arms are cut the same
way, though for P2Rank the distinction is vacuous by construction: no part of
this fold trained it.

Pooled, not per-unit-averaged: the grid search that produced the shipped q
pooled the confusion counts over all training units, and a threshold selected
under one objective cannot be compared against a threshold selected under
another. Nothing here reads the held-out fold, and ``--check`` asserts as much.

Usage:
  PYTHONPATH=src python3.12 tools/train_operating_points.py
  PYTHONPATH=src python3.12 tools/train_operating_points.py --check
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
FIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
P2CACHE = ROOT / "results/architecture_sweep/_p2rank_train_residues.npz"
P2OP = ROOT / "results/architecture_sweep/P2RANK_TRAIN_OPERATING_POINT.json"
OUT = ROOT / "results/architecture_sweep/TRAIN_OPERATING_POINTS.json"

SCHEMA = "geoaudit.train_operating_points.v1"
Q_GRID = [round(float(q), 2) for q in np.arange(0.02, 0.41, 0.01)]
SPLIT_SEED = 20260725
# Scoring the whole fold at once needs the 645-wire matrix as float64, which is
# 1.2 GB and put this machine into swap the last time a tool did it. Units are
# scored independently apart from the gate, which is within-chain, so a block of
# units is a safe unit of work.
UNIT_BLOCK = 64


def top_q_call(s: np.ndarray, q: float) -> np.ndarray:
    """The shipped rule: per-chain top-q by score, ties by residue order."""
    n = len(s)
    k = max(1, int(round(q * n)))
    out = np.zeros(n, dtype=bool)
    out[np.argsort(-s, kind="stable")[:k]] = True
    return out


def pooled_counts(scores: list[np.ndarray], truths: list[np.ndarray],
                  q: float) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for s, t in zip(scores, truths):
        call = top_q_call(s, q)
        tp += int((call & t).sum())
        fp += int((call & ~t).sum())
        fn += int((~call & t).sum())
        tn += int((~call & ~t).sum())
    return tp, fp, tn, fn


def f1_of(tp: int, fp: int, tn: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else 0.0


def mcc_of(tp: int, fp: int, tn: int, fn: int) -> float:
    d = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn - fp * fn) / d) if d > 0 else 0.0


OBJECTIVES = {"pooled_f1": f1_of, "pooled_mcc": mcc_of}


def curve(scores: list[np.ndarray], truths: list[np.ndarray]) -> list[dict]:
    rows = []
    for q in Q_GRID:
        tp, fp, tn, fn = pooled_counts(scores, truths, q)
        rows.append({"q": q, "n_called": tp + fp,
                     **{name: round(fn_(tp, fp, tn, fn), 6)
                        for name, fn_ in OBJECTIVES.items()}})
    return rows


def _argmax(rows: list[dict], objective: str) -> dict:
    """The first q attaining the maximum, so ties resolve to the smaller call.

    A tie broken towards a larger calling fraction would quietly buy recall at
    the same objective value, which is the kind of free choice a preregistered
    threshold exists to remove.
    """
    best = max(r[objective] for r in rows)
    row = next(r for r in rows if r[objective] == best)
    return {"q": row["q"], "value": row[objective], "n_called": row["n_called"],
            "ties_at": [r["q"] for r in rows if r[objective] == best]}


def _cluster_of() -> dict[str, str]:
    return {f"{e['pdb']}_{e['chain']}": e["cluster_id"]
            for e in json.loads(MANIFEST.read_text())["entries"]}


def _fit_half(units: list[str]) -> np.ndarray:
    """The same cluster-disjoint halving the architecture selection used.

    Reproduced from the seed rather than read from an artifact, and asserted
    against the sweep's row count below, because a threshold chosen on a
    different half than the field was fitted on would be in-sample again
    without anything saying so.
    """
    cluster_of = _cluster_of()
    clusters = sorted({cluster_of[u] for u in units})
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(clusters)
    fit = set(clusters[:len(clusters) // 2])
    return np.array([cluster_of[u] in fit for u in units])


def _score_in_blocks(field, X: np.ndarray, ctr: np.ndarray,
                     n_res: np.ndarray) -> list[np.ndarray]:
    """Per-unit scores from a compiled field, a block of units at a time."""
    offs = np.concatenate([[0], np.cumsum(n_res)])
    out: list[np.ndarray] = []
    for a in range(0, len(n_res), UNIT_BLOCK):
        b = min(a + UNIT_BLOCK, len(n_res))
        lo, hi = int(offs[a]), int(offs[b])
        s = field.score_matrix(np.asarray(X[lo:hi], dtype=np.float64),
                               ctr[lo:hi], list(n_res[a:b]))
        for j in range(a, b):
            out.append(s[int(offs[j]) - lo:int(offs[j + 1]) - lo].copy())
        del s
    return out


def _our_scores_shipped() -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    from pocket_bench.methods.table_field import TableField
    field = TableField.load(FIELD)
    with np.load(CACHE, allow_pickle=False) as z:
        units = [str(u) for u in z["units"]]
        n_res = z["n_res_per"].astype(np.int64)
        y, ctr = z["y"].astype(bool), z["ctr"]
        scores = _score_in_blocks(field, z["X"], ctr, n_res)
    offs = np.concatenate([[0], np.cumsum(n_res)])
    truths = [y[int(offs[i]):int(offs[i + 1])] for i in range(len(n_res))]
    return units, scores, truths


def _our_scores_fit_half(units: list[str]):
    """Compile on the fit half, score the pick half with what it produced."""
    from pocket_bench.methods.table_field import TableField, compile_field
    is_fit = _fit_half(units)
    with np.load(CACHE, allow_pickle=False) as z:
        n_res = z["n_res_per"].astype(np.int64)
        names, prop = z["names"], z["propensity_table"]
        row_unit = np.repeat(np.arange(len(units)), n_res)
        fm, pm = is_fit[row_unit], ~is_fit[row_unit]
        X = np.asarray(z["X"], dtype=np.float64)
        y, ctr = z["y"].astype(np.int64), z["ctr"]
        doc = compile_field(X[fm], y[fm], ctr[fm], list(n_res[is_fit]),
                            names, prop)
        Xp, ctrp = X[pm].copy(), ctr[pm]
        del X
        gc.collect()
        field = TableField(doc)
        scores = _score_in_blocks(field, Xp, ctrp, n_res[~is_fit])
        del Xp
        gc.collect()
    yp = y[pm].astype(bool)
    offs = np.concatenate([[0], np.cumsum(n_res[~is_fit])])
    truths = [yp[int(offs[i]):int(offs[i + 1])] for i in range(int((~is_fit).sum()))]
    return ([u for u, f in zip(units, is_fit) if not f], scores, truths,
            {"fit_units": int(is_fit.sum()), "pick_units": int((~is_fit).sum()),
             "fit_residues": int(fm.sum()), "pick_residues": int(pm.sum()),
             "q_the_fit_half_field_would_ship": doc["operating_point"]["q"]})


def _p2rank_scores():
    if not P2CACHE.is_file():
        raise SystemExit(
            f"missing {P2CACHE.relative_to(ROOT)}. It is deliberately "
            f"untracked; regenerate it with `make p2op`, which re-runs P2Rank "
            f"over the training receptors")
    with np.load(P2CACHE, allow_pickle=False) as z:
        units = [str(u) for u in z["unit_id"]]
        n_res = z["n_res"].astype(np.int64)
        score, truth = z["score"], z["truth"].astype(bool)
        native = z["native"].astype(bool)
    offs = np.concatenate([[0], np.cumsum(n_res)])
    cut = lambda a: [a[int(offs[i]):int(offs[i + 1])] for i in range(len(n_res))]
    return units, cut(score), cut(truth), cut(native)


def _native_pooled(calls: list[np.ndarray], truths: list[np.ndarray]) -> dict:
    """What P2Rank's own pocket assignment scores, pooled, for the forecast.

    The comparison the paper published gave P2Rank this call and gave us a
    tuned one. How much the tuning is worth on the training fold is the only
    honest basis for predicting how much of the held-out margin it accounts
    for, and that requires the untuned number under both objectives.
    """
    tp = fp = tn = fn = 0
    for c, t in zip(calls, truths):
        tp += int((c & t).sum())
        fp += int((c & ~t).sum())
        fn += int((~c & t).sum())
        tn += int((~c & ~t).sum())
    return {"n_called": tp + fp,
            **{name: round(fn_(tp, fp, tn, fn), 6)
               for name, fn_ in OBJECTIVES.items()}}


def _subset(units: list[str], scores, truths, keep: list[str]):
    idx = {u: i for i, u in enumerate(units)}
    return ([scores[idx[u]] for u in keep], [truths[idx[u]] for u in keep])


def build() -> dict:
    t0 = time.time()
    our_units, our_s, our_t = _our_scores_shipped()
    p2_units, p2_s, p2_t, p2_native = _p2rank_scores()
    if our_units != p2_units:
        raise SystemExit(
            f"the two training caches disagree about the unit set "
            f"({len(our_units)} vs {len(p2_units)}); a threshold chosen on one "
            f"could not be compared against a threshold chosen on the other")

    pick_units, our_pick_s, our_pick_t, split = _our_scores_fit_half(our_units)
    p2_pick_s, p2_pick_t = _subset(p2_units, p2_s, p2_t, pick_units)

    full = {"table_field": curve(our_s, our_t), "p2rank": curve(p2_s, p2_t)}
    half = {"table_field": curve(our_pick_s, our_pick_t),
            "p2rank": curve(p2_pick_s, p2_pick_t)}

    selected = {}
    for method in ("table_field", "p2rank"):
        for objective in OBJECTIVES:
            selected[f"{method}/{objective}/full_fold"] = _argmax(
                full[method], objective)
            selected[f"{method}/{objective}/pick_half"] = _argmax(
                half[method], objective)

    shipped_q = json.loads(FIELD.read_text())["operating_point"]["q"]
    p2_prior = json.loads(P2OP.read_text()) if P2OP.is_file() else {}

    p2_untuned = _native_pooled(p2_native, p2_t)
    tuning_gain = {}
    for objective in OBJECTIVES:
        at_q = selected[f"p2rank/{objective}/full_fold"]["value"]
        tuning_gain[objective] = {
            "p2rank_at_its_native_pocket_assignment": p2_untuned[objective],
            "p2rank_at_its_tuned_q": at_q,
            "the_tuning_is_worth": round(at_q - p2_untuned[objective], 6),
            "our_value_at_our_tuned_q": selected[
                f"table_field/{objective}/full_fold"]["value"],
            "our_margin_over_untuned_p2rank": round(
                selected[f"table_field/{objective}/full_fold"]["value"]
                - p2_untuned[objective], 6),
            "our_margin_over_tuned_p2rank": round(
                selected[f"table_field/{objective}/full_fold"]["value"] - at_q,
                6),
        }

    # The whole reason the pick-half thresholds are here: if they differ from
    # the in-sample ones, the shipped q was partly a fit to the fold that
    # counted the cells, and the matched comparison inherits that.
    optimism = {}
    for method in ("table_field", "p2rank"):
        for objective in OBJECTIVES:
            a = selected[f"{method}/{objective}/full_fold"]["q"]
            b = selected[f"{method}/{objective}/pick_half"]["q"]
            optimism[f"{method}/{objective}"] = {
                "q_in_sample": a, "q_out_of_sample": b,
                "same": abs(a - b) < 1e-9, "shift": round(b - a, 4)}

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "question": (
            "at what per-chain calling fraction is each method's confusion "
            "matrix best, under each of two objectives, chosen on the "
            "training fold alone and on a half of it the scored field never "
            "saw"),
        "test_fold_touched": False,
        "reads_test_fold": False,
        "rule": {
            "binarisation": "per-chain top-q by score",
            "tie_break": "stable sort of the negated score, so equal scores "
                         "are called in ascending residue number",
            "objective": "pooled over all units in the selection set: the "
                         "TP/FP/TN/FN are summed and the summary computed "
                         "once, not averaged per unit",
            "grid": {"low": Q_GRID[0], "high": Q_GRID[-1], "step": 0.01,
                     "n": len(Q_GRID)},
            "argmax_tie_break": "the smallest q attaining the maximum",
        },
        "n_units": len(our_units),
        "n_residues": int(sum(len(s) for s in our_s)),
        "n_positives": int(sum(int(t.sum()) for t in our_t)),
        "split": {
            "criterion": "cluster_id, seeded shuffle, disjoint halves",
            "seed": SPLIT_SEED,
            "identical_to": "the split tools/sensitivity_sweep.py uses",
            **split,
        },
        "curves": {"full_fold": full, "pick_half": half},
        "selected": selected,
        "p2rank_native_call_on_the_training_fold": p2_untuned,
        "what_tuning_p2rank_is_worth_on_the_training_fold": tuning_gain,
        "in_sample_optimism": optimism,
        "agreement_with_prior_artifacts": {
            "shipped_q": shipped_q,
            "our_f1_q_on_full_fold": selected["table_field/pooled_f1/full_fold"]["q"],
            "reproduces_shipped_q": abs(
                selected["table_field/pooled_f1/full_fold"]["q"] - shipped_q) < 1e-9,
            "p2rank_f1_q_previously_reported": p2_prior.get("p2rank_selected_q"),
            "our_p2rank_f1_q": selected["p2rank/pooled_f1/full_fold"]["q"],
            "reproduces_p2rank_q": (
                p2_prior.get("p2rank_selected_q") is None
                or abs(selected["p2rank/pooled_f1/full_fold"]["q"]
                       - p2_prior["p2rank_selected_q"]) < 1e-9),
            "p2rank_f1_previously_reported": p2_prior.get(
                "p2rank_pooled_train_f1_at_selected_q"),
            "our_p2rank_f1": selected["p2rank/pooled_f1/full_fold"]["value"],
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }


def _report(d: dict) -> None:
    print(f"\ntraining-fold operating points, {d['n_units']} units, "
          f"{d['n_positives']} positives")
    for k in sorted(d["selected"]):
        s = d["selected"][k]
        tie = "" if len(s["ties_at"]) == 1 else f"  (ties {s['ties_at']})"
        print(f"  {k:<40} q={s['q']:.2f}  value={s['value']:.4f}{tie}")
    for obj, g in d["what_tuning_p2rank_is_worth_on_the_training_fold"].items():
        print(f"  {obj}: P2Rank native {g['p2rank_at_its_native_pocket_assignment']:.4f} "
              f"-> tuned {g['p2rank_at_its_tuned_q']:.4f} "
              f"(+{g['the_tuning_is_worth']:.4f});  our margin "
              f"{g['our_margin_over_untuned_p2rank']:+.4f} untuned -> "
              f"{g['our_margin_over_tuned_p2rank']:+.4f} tuned")
    print("  in-sample vs out-of-sample q:")
    for k, v in d["in_sample_optimism"].items():
        flag = "same" if v["same"] else f"SHIFTS {v['shift']:+.2f}"
        print(f"    {k:<28} {v['q_in_sample']:.2f} -> "
              f"{v['q_out_of_sample']:.2f}   {flag}")
    a = d["agreement_with_prior_artifacts"]
    print(f"  reproduces shipped q: {a['reproduces_shipped_q']}   "
          f"reproduces P2Rank's q: {a['reproduces_p2rank_q']}")


def _check() -> int:
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    if d.get("schema") != SCHEMA:
        print(f"FAILED: schema {d.get('schema')}")
        return 1
    if d.get("reads_test_fold") or d.get("test_fold_touched"):
        print("FAILED: the artifact claims to have touched the held-out fold")
        return 1
    a = d["agreement_with_prior_artifacts"]
    if not a["reproduces_shipped_q"]:
        print(f"FAILED: this search puts our q at {a['our_f1_q_on_full_fold']} "
              f"but the shipped field carries {a['shipped_q']}")
        return 1
    if not a["reproduces_p2rank_q"]:
        print(f"FAILED: this search puts P2Rank's q at {a['our_p2rank_f1_q']} "
              f"but the earlier artifact reported "
              f"{a['p2rank_f1_previously_reported']}")
        return 1
    # Every selected q has to be the argmax of the curve it claims to come from.
    for key, sel in d["selected"].items():
        method, objective, where = key.split("/")
        rows = d["curves"][where][method]
        best = max(r[objective] for r in rows)
        if abs(sel["value"] - best) > 1e-9:
            print(f"FAILED: {key} records {sel['value']} but its curve peaks "
                  f"at {best}")
            return 1
    _report(d)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        return _check()
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
