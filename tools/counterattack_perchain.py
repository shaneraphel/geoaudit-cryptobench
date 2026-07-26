#!/usr/bin/env python3
"""Make the threshold gate self-contained per chain, without losing the score.

The selection sweeps normalised the accumulator and the spatial gate by their
maximum over the WHOLE fold before adding them. That is fine for ranking inside
a sweep, but it is not a deployable detector: the relative weight of the two
terms then depends on which other structures happen to be in the batch, and the
repository's stated contract is that scoring one receptor reads nothing but that
receptor. So the normalisation has to move inside the chain, and the question is
what that costs.

Three self-contained fusions are measured here on the same pick half used for
every other selection decision:

    per-chain unit      each term divided by its own chain's maximum magnitude
    per-chain rank      each term replaced by its rank within the chain, summed
                        as integers -- no division anywhere
    accumulator only    the integer gate output with no spatial term at all

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_perchain.py
"""
from __future__ import annotations

import json

import numpy as np

from pocket_bench.paths import ROOT

from counterattack_select import GATE_RADII, SEED, _unit, gate, per_unit_auc
from counterattack_threshold import closed_form_direction, quantize

CACHE = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
DIGITS = ROOT / "data/cryptobench_apo/_expanded_digits_train.npy"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_PERCHAIN.json"


def per_chain_unit(x, n_res_per):
    out = np.empty_like(x, dtype=np.float64)
    off = 0
    for n in n_res_per:
        n = int(n)
        out[off:off + n] = _unit(x[off:off + n])
        off += n
    return out


def per_chain_rank(x, n_res_per):
    """Rank within the chain, scaled to [0, 1] by the chain's own length."""
    out = np.empty(len(x), dtype=np.float64)
    off = 0
    for n in n_res_per:
        n = int(n)
        blk = x[off:off + n]
        r = np.empty(n, dtype=np.float64)
        r[np.argsort(blk, kind="stable")] = np.arange(n, dtype=np.float64)
        out[off:off + n] = r / max(n - 1, 1)
        off += n
    return out


def main() -> int:
    z = np.load(CACHE, allow_pickle=False)
    y, n_res, ctr = z["y"], z["n_res_per"], z["ctr"]
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

    D = np.load(DIGITS)
    Dfit, Dpick = D[fm], D[pm]

    def pooled(x, yv):
        n_pos, n_neg = int(yv.sum()), int(len(yv) - yv.sum())
        r = np.empty(len(x))
        r[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
        return (r[yv == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    g = np.array([abs(2.0 * pooled(Dfit[:, j].astype(float), yfit) - 1.0)
                  for j in range(D.shape[1])])
    cols = sorted(np.argsort(-g)[:172].tolist())
    wi = quantize(closed_form_direction(Dfit[:, cols], yfit), 128)
    S = (Dpick[:, cols].astype(np.int64) @ wi).astype(np.float64)
    gates = [gate(S, ctr_pick, n_pick_per, r) for r in GATE_RADII]

    results = {}

    def record(name, score):
        a = per_unit_auc(score, ypick, n_pick_per)
        results[name] = a
        print(f"  {name:42s} {a:.4f}", flush=True)

    record("accumulator only (no spatial term)", S)
    record("global unit (sweep convention, not deployable)",
           _unit(S) + np.sum([_unit(x) for x in gates], axis=0))
    record("per-chain unit",
           per_chain_unit(S, n_pick_per)
           + np.sum([per_chain_unit(x, n_pick_per) for x in gates], axis=0))
    record("per-chain rank",
           per_chain_rank(S, n_pick_per)
           + np.sum([per_chain_rank(x, n_pick_per) for x in gates], axis=0))

    best = max(
        (k for k in results if "not deployable" not in k),
        key=lambda k: results[k])
    print(f"\nbest self-contained fusion: {best} {results[best]:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_perchain.v1",
        "clinical_grade": False,
        "question": "what does making the fusion self-contained per chain cost",
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED},
        "pick_half_roc_auc": results,
        "selected": best,
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
