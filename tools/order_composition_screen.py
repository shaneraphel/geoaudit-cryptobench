#!/usr/bin/env python3.12
"""Does composing the gated and ungated rankings recover precision at the top?

The deficit this targets
------------------------
On the official fold the counting field is at parity with pLM-NN on per-unit
ROC-AUC (+0.0106) and *ahead* on the pooled residue read (0.8535 vs 0.8466),
while sitting *behind* on per-unit PR-AUC (−0.0100). Those three facts together
say something specific: the field orders the whole chain better, and orders the
first few residues worse. PR-AUC is dominated by the top of the list, which is
exactly what a chemist reads.

The mechanism this tests
------------------------
The spatial gate adds back the neighbourhood mean of the score. It exists
because a cryptic site is a contiguous patch and a per-residue function cannot
say so, and it is worth a great deal: it is the difference between ranking
patches and ranking residues. But averaging over a 14 A ball necessarily makes
the residues *inside* one patch resemble each other, and the ordering within the
winning patch is the thing PR-AUC at low recall is made of.

So the hypothesis is that the two rankings are good at different things — the
gated one at finding the patch, the raw one at ordering inside it — and that
composing them recovers the top without giving up the whole-chain ordering.

The arms, all parameter-free order compositions
-----------------------------------------------
``gated``   the deployed score. Baseline.
``raw``     the same score before the gate is added. Not expected to win
            overall; included because the hypothesis is false if the raw
            ordering is simply worse everywhere, including at the top.
``borda``   ``rank(gated) + rank(raw)``, integer ranks summed. A Borda count
            over two orderings. No coefficient, no threshold, no radius: the
            only input is the two orders, and the sum of two ranks is the
            oldest parameter-free way to combine them.
``lex``     lexicographic. Take the top ``q`` by the gated score, where ``q`` is
            the operating point already compiled on the training fold, and
            reorder only those by the raw score; everything below keeps its
            gated order. This says "trust the gate to choose the patch, trust
            the raw score to order inside it" as literally as it can be said.

Nothing here is fitted. ``borda`` introduces no constant at all; ``lex`` reuses
a constant already selected on the training fold for a different purpose.

Why this runs on the training fold first
----------------------------------------
Four arms scored on the test fold and the best one reported is test-fold
selection, whatever the arms are made of. This screens on twelve
cluster-disjoint halves of the *training* fold and reports every arm. Only the
arm that wins here is read on the official fold, once, by
``grand_baseline_read.py``.

The metric here is per-unit **PR-AUC**, because that is the deficit. Per-unit
ROC-AUC is reported beside it, because an arm that buys the top by giving up the
whole ranking has not helped.

``clinical_grade`` is false.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from expand_invariant_bank import SEED  # noqa: E402
from select_architecture_on_train import cluster_half_split  # noqa: E402

from pocket_bench.methods.table_bank import (
    cell_offsets,
    chain_digits,
    compile_cells,
    integer_fanout,
    partition_tables,
    score,
)
from pocket_bench.methods.table_field import (
    FAN_OUT_CAP,
    PARTITION_ROUNDS,
    PARTITION_SEED,
    RIDGE,
    TABLE_WIDTH,
    apply_gate,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.order_composition_screen.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT = ROOT / "results/architecture_sweep/ORDER_COMPOSITION_SCREEN.json"

# The operating point compiled on the training fold for the deployed field.
# Reused, not re-selected. See TABLE_FIELD.json -> operating_point.q
DEFAULT_Q = 0.10


def _ranks(v: np.ndarray) -> np.ndarray:
    """Integer ranks, 0 = smallest, ties broken stably by position."""
    return np.argsort(np.argsort(v, kind="stable"), kind="stable")


def _compose(raw: np.ndarray, gated: np.ndarray, n_res, q: float
             ) -> dict[str, np.ndarray]:
    out = {"gated": gated.copy(), "raw": raw.copy(),
           "borda": np.empty_like(gated), "lex": np.empty_like(gated)}
    off = 0
    for n in n_res:
        n = int(n)
        g, r = gated[off:off + n], raw[off:off + n]
        rg, rr = _ranks(g), _ranks(r)
        out["borda"][off:off + n] = rg + rr

        k = max(1, int(round(q * n)))
        top = np.argsort(-g, kind="stable")[:k]
        # Below the operating point: keep the gated order, compressed under the
        # top block. Inside it: order by the raw score. Ranks are integers and
        # the two blocks never interleave, so this is one permutation.
        lex = rg.astype(np.float64).copy()
        lex[top] = float(n) + _ranks(r[top]).astype(np.float64)
        out["lex"][off:off + n] = lex
        off += n
    return out


def _per_unit(sc: np.ndarray, y: np.ndarray, n_res) -> tuple[float, float, int]:
    roc, prc, off = [], [], 0
    for n in n_res:
        n = int(n)
        yy, ss = y[off:off + n], sc[off:off + n]
        off += n
        if yy.sum() == 0 or yy.sum() == n:
            continue
        roc.append(roc_auc_score(yy, ss))
        prc.append(average_precision_score(yy, ss))
    return float(np.mean(roc)), float(np.mean(prc)), len(roc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=12)
    ap.add_argument("--q", type=float, default=DEFAULT_Q)
    ap.add_argument("--gate-radius", type=float, default=14.0)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)

    import pocket_bench.methods.table_field as tf
    tf.GATE_RADIUS = a.gate_radius

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    z.close()
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"]
                  for e in json.loads(MANIFEST.read_text())["entries"]}

    t0 = time.perf_counter()
    D = chain_digits(np.asarray(W, dtype=np.float64), n_res)
    tabs = partition_tables(D.shape[1], TABLE_WIDTH, PARTITION_ROUNDS,
                            PARTITION_SEED)
    offs = cell_offsets(tabs)
    print(f"banded {D.shape[1]} columns, {len(tabs)} tables, "
          f"{time.perf_counter() - t0:.0f}s", flush=True)

    row = np.repeat(np.arange(len(n_res)), n_res)
    arms = ("gated", "raw", "borda", "lex")
    got: dict[str, dict[str, list[float]]] = {
        k: {"roc": [], "pr": []} for k in arms}

    for s in range(a.splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        t1 = time.perf_counter()

        frac, _t = compile_cells(D[fit], y[fit], tabs, offs)
        mult = integer_fanout(D[fit], y[fit], tabs, offs, frac, RIDGE,
                              FAN_OUT_CAP)
        raw = score(D[pick], tabs, offs, frac, mult)
        gated = apply_gate(raw, ctr[pick], n_pick)

        comp = _compose(raw, gated, n_pick, a.q)
        line = []
        for k in arms:
            roc, pr, _n = _per_unit(comp[k], y[pick], n_pick)
            got[k]["roc"].append(roc)
            got[k]["pr"].append(pr)
            line.append(f"{k} {pr:.4f}/{roc:.4f}")
        print(f"  split {s + 1}/{a.splits}  " + "  ".join(line)
              + f"  {time.perf_counter() - t1:.0f}s", flush=True)

    base_pr = np.asarray(got["gated"]["pr"])
    base_roc = np.asarray(got["gated"]["roc"])
    summary = {}
    for k in arms:
        pr = np.asarray(got[k]["pr"])
        roc = np.asarray(got[k]["roc"])
        summary[k] = {
            "mean_pr_auc": float(pr.mean()),
            "mean_roc_auc": float(roc.mean()),
            "delta_pr_vs_gated": float((pr - base_pr).mean()),
            "delta_roc_vs_gated": float((roc - base_roc).mean()),
            "splits_pr_better": int((pr > base_pr).sum()),
            "splits_roc_better": int((roc > base_roc).sum()),
            "n_splits": a.splits,
            "per_split_pr": [float(v) for v in pr],
            "per_split_roc": [float(v) for v in roc],
        }

    best = max((k for k in arms if k != "gated"),
               key=lambda k: summary[k]["delta_pr_vs_gated"])
    out = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "question": ("whether composing the gated and ungated orderings "
                     "recovers per-unit PR-AUC, which is where pLM-NN still "
                     "leads, without giving up per-unit ROC-AUC"),
        "arms_are_parameter_free": (
            "borda introduces no constant; lex reuses the operating point q "
            "already compiled on the training fold; neither fits anything"),
        "protocol": {
            "n_splits": a.splits,
            "split": "cluster-disjoint halves, seeds 20260725 onward",
            "compile_on": "the fit half only",
            "evaluate_on": "the pick half",
            "gate_radius": a.gate_radius,
            "q": a.q,
            "primary_metric": "mean per-unit PR-AUC",
            "secondary_metric": "mean per-unit ROC-AUC, so that an arm buying "
                                "the top by losing the whole ranking is visible",
        },
        "reseed_floor": 0.0026,
        "reseed_floor_definition": (
            "split-to-split standard deviation of the deployed field's mean "
            "per-unit ROC-AUC under reseeding alone; quoted for scale, and it "
            "is a ROC-AUC floor, so a PR-AUC effect should not be compared to "
            "it directly"),
        "n_train_units": len(units),
        "n_train_units_definition": "chains in the training wide cache",
        "arms": summary,
        "best_non_baseline_arm_by_pr": best,
        "what_happens_next": (
            f"only '{best}' is read on the official fold, once, and only if "
            f"its PR-AUC gain here is positive on a clear majority of splits "
            f"and its ROC-AUC does not regress"),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print("\nWROTE", a.out)
    print(f"\n{'arm':8s} {'PR-AUC':>9s} {'dPR':>9s} {'splits':>7s} "
          f"{'ROC-AUC':>9s} {'dROC':>9s} {'splits':>7s}")
    for k in arms:
        s = summary[k]
        print(f"{k:8s} {s['mean_pr_auc']:9.4f} {s['delta_pr_vs_gated']:+9.4f} "
              f"{s['splits_pr_better']:4d}/{a.splits:<2d} "
              f"{s['mean_roc_auc']:9.4f} {s['delta_roc_vs_gated']:+9.4f} "
              f"{s['splits_roc_better']:4d}/{a.splits:<2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
