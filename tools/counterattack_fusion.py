#!/usr/bin/env python3
"""Close the gap between the counting fusion and the ceiling of its own inputs.

On the training-fold split the expanded wires support a continuous functional at
ROC-AUC 0.781 while the table bank reaches 0.755. The tables are not the
bottleneck -- inside a six-wire table the joint distribution is modelled exactly
-- so the loss is in how the tables are combined: fifteen tables whose
informativeness varies twofold are summed with multiplicities 1..15, which is a
very coarse way to say which of them to believe.

Four sharper combination rules, all of them still integer fan-out of a table
output and none of them a real-valued coefficient:

  rank        m_k = rank of the table's compiled Gini            (the incumbent)
  ratio       m_k = round(g_k / min g), capped                   (finer, still integer)
  square      m_k = (rank)^2                                     (steeper)
  greedy      m_k = how many times a stagewise search picked table k

The greedy rule needs to be stated plainly because it is the one a reviewer will
question: it repeatedly appends whichever table most increases the pooled AUC of
the running integer sum, ON THE FIT HALF, for a fixed number of rounds. That is
stagewise additive selection. It performs no gradient step and produces no real
coefficient -- the output is a multiset of tables, i.e. an integer multiplicity
vector -- but it is a search over the fit half, and the paper says so.

Everything is ranked on the pick half, whose clusters the fit half never saw.
The official test fold is not read here.

Usage: PYTHONPATH=src python3.12 tools/counterattack_fusion.py
"""
from __future__ import annotations

import json

import numpy as np

from pocket_bench.paths import ROOT

from counterattack_select import (  # noqa: E402  (sibling tool, same directory)
    GATE_RADII,
    L,
    SEED,
    _unit,
    chain_digits,
    gate,
    per_unit_auc,
    pooled_auc,
)

CACHE = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
DIGITS = ROOT / "data/cryptobench_apo/_expanded_digits_train.npy"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_FUSION.json"
GREEDY_ROUNDS = 40


def build_tables(order, k, width=6, stride=None):
    """Windows over the top-k wires, taken in WIRE-INDEX order.

    The ordering inside the window matters and is easy to get wrong. Grouping by
    strength puts the k strongest wires in one table, and the strongest wires are
    the ones most correlated with each other -- ``depth`` and ``depth_rank`` are
    the same quantity twice, ``concavity@14`` and ``concavity@20`` nearly so --
    so that table spends six digits re-stating one fact while the last table
    holds six near-blind wires. Index order interleaves the groups (base
    invariant, chemistry, each context radius), which is why it measures better:
    each table then sees several different kinds of evidence at once.
    """
    cols = sorted(order[:k].tolist())
    stride = width if stride is None else stride
    tabs = [cols[i:i + width] for i in range(0, len(cols) - 1, stride)]
    return [t for t in tabs if len(t) >= 2]


def compile_bank(Dfit, yfit, Dpick, tables, rate):
    yf = yfit.astype(np.float64)
    fr_fit, fr_pick, gini = [], [], []
    for cols in tables:
        n_cells = L ** len(cols)
        a_fit = np.zeros(len(yfit), dtype=np.int64)
        a_pick = np.zeros(Dpick.shape[0], dtype=np.int64)
        for t, c in enumerate(cols):
            a_fit += Dfit[:, c] * (L ** t)
            a_pick += Dpick[:, c] * (L ** t)
        tot = np.bincount(a_fit, minlength=n_cells).astype(np.float64)
        pos = np.bincount(a_fit, weights=yf, minlength=n_cells)
        frac = np.where(tot > 0, pos / np.maximum(tot, 1.0), rate)
        fr_fit.append(frac[a_fit])
        fr_pick.append(frac[a_pick])
        gini.append(abs(2.0 * pooled_auc(frac[a_fit], yfit) - 1.0))
    return fr_fit, fr_pick, np.asarray(gini)


def greedy_multiplicity(fr_fit, yfit, rounds=GREEDY_ROUNDS):
    """Stagewise integer selection on the fit half. Returns counts per table."""
    k = len(fr_fit)
    mult = np.zeros(k, dtype=np.int64)
    running = np.zeros_like(fr_fit[0])
    best_auc = 0.5
    for _ in range(rounds):
        cand_auc = [pooled_auc(running + fr_fit[j], yfit) for j in range(k)]
        j = int(np.argmax(cand_auc))
        if cand_auc[j] <= best_auc + 1e-6:
            break
        best_auc = cand_auc[j]
        mult[j] += 1
        running = running + fr_fit[j]
    return mult, best_auc


def main() -> int:
    z = np.load(CACHE, allow_pickle=False)
    X, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    names = [str(s) for s in z["names"]]
    units = [str(u) for u in z["units"]]
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
    rate = float(yfit.mean())

    if DIGITS.exists():
        D = np.load(DIGITS)
        print(f"digits from cache {D.shape}", flush=True)
    else:
        D = chain_digits(X, n_res)
        np.save(DIGITS, D)
        print(f"digits computed and cached {D.shape}", flush=True)
    Dfit, Dpick = D[fm], D[pm]

    g = np.array([abs(2.0 * pooled_auc(Dfit[:, j].astype(float), yfit) - 1.0)
                  for j in range(D.shape[1])])
    order = np.argsort(-g)

    results = []

    def record(name, score, extra=None):
        a = per_unit_auc(score, ypick, n_pick_per)
        results.append({"architecture": name, "pick_half_roc_auc": a,
                        **(extra or {})})
        print(f"  {name:56s} {a:.4f}", flush=True)
        return a

    for k, width, stride in ((90, 6, 6), (90, 6, 3), (120, 6, 3), (172, 6, 4)):
        tables = build_tables(order, k, width, stride)
        fr_fit, fr_pick, gi = compile_bank(Dfit, yfit, Dpick, tables, rate)
        tag = f"top-{k} w{width} s{stride} ({len(tables)} tables)"

        rank_m = np.empty(len(tables), dtype=np.int64)
        rank_m[np.argsort(gi)] = np.arange(1, len(tables) + 1)
        ratio_m = np.clip(np.round(gi / max(gi.min(), 1e-9)), 1, 12).astype(int)
        square_m = rank_m ** 2
        greedy_m, fit_auc = greedy_multiplicity(fr_fit, yfit)

        for label, m in (("rank", rank_m), ("ratio", ratio_m),
                         ("square", square_m), ("greedy", greedy_m)):
            if m.sum() == 0:
                continue
            S = np.sum([int(w) * f for w, f in zip(m, fr_pick)], axis=0)
            G = np.sum([_unit(gate(S, ctr_pick, n_pick_per, r))
                        for r in GATE_RADII], axis=0)
            record(f"{tag}, {label} multiplicity + gate",
                   _unit(S) + _unit(G),
                   {"n_tables": len(tables), "n_wires": k,
                    "width": width, "stride": stride, "rule": label,
                    "multiplicity": m.tolist(),
                    "fit_half_pooled_auc": (fit_auc if label == "greedy"
                                            else None)})

    winner = max(results, key=lambda r: r["pick_half_roc_auc"])
    print(f"\nselected on the training fold alone: {winner['architecture']}")
    print(f"  pick-half ROC-AUC {winner['pick_half_roc_auc']:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_fusion.v1",
        "clinical_grade": False,
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED,
                  "n_fit_units": int(is_fit.sum()),
                  "n_pick_units": len(units) - int(is_fit.sum())},
        "wire_order_top20": [names[j] for j in order[:20]],
        "candidates": results,
        "selected": winner,
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
