"""Find the table width at which interactions are affordable, and use it.

The two readouts tried so far fail in opposite directions. The width-six bank
models interactions exactly inside each window, but a 4096-cell table on 234838
residues at a 5.8 percent base rate leaves about fifty residues and three
positives per cell, so each fraction carries a relative standard error near 60
percent. The linear functional estimates its coefficients on the whole fold, but
represents no interaction at all. On the training split they score 0.763 and
0.787.

The width is the free variable and nobody had varied it. A width-two quaternary
table has sixteen cells, so a cell holds about 14700 residues and its fraction
has a relative standard error near 3 percent: at this width the joint
distribution is not a noisy quantity, it is a measured one. There are only
C(K,2) such tables over the top K wires, and their fractions can be combined by
the same single closed-form solve used for the linear readout.

This is the classical bias-variance statement applied to table width rather than
to a penalty. Interactions are not unaffordable here; interactions of order six
are.

  pair_K       C(K,2) width-two fractions over the top K wires
  pair_K+lin   the same, with the 172 first-order digits appended
  triple_K     width-three tables over a screened subset, 64 cells each

Both halves come from the training fold and their clusters are disjoint. The
official test fold is not read here.

Usage: PYTHONPATH=src:tools python3.12 tools/pairwise_readout_select.py
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np

from pocket_bench.paths import ROOT

from counterattack_select import (  # noqa: E402  (sibling tool)
    GATE_RADII,
    L,
    SEED,
    chain_digits,
    per_unit_auc,
    pooled_auc,
)
from final_readout_select import gate_matrix  # noqa: E402

CACHE = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT = ROOT / "results/architecture_sweep/PAIRWISE_READOUT_SELECTION.json"
RIDGE = 1e-4


def pair_fractions(Dfit, yfit, Dpick, cols, rate):
    """Width-two cell fractions for every pair, fit half compiled, both mapped."""
    pairs = list(combinations(cols, 2))
    Pf = np.empty((Dfit.shape[0], len(pairs)), dtype=np.float32)
    Pp = np.empty((Dpick.shape[0], len(pairs)), dtype=np.float32)
    yf = yfit.astype(np.float64)
    for t, (i, j) in enumerate(pairs):
        af = Dfit[:, i] * L + Dfit[:, j]
        ap = Dpick[:, i] * L + Dpick[:, j]
        tot = np.bincount(af, minlength=L * L).astype(np.float64)
        pos = np.bincount(af, weights=yf, minlength=L * L)
        frac = np.where(tot > 0, pos / np.maximum(tot, 1.0), rate)
        Pf[:, t] = frac[af]
        Pp[:, t] = frac[ap]
    return Pf, Pp, pairs


RIDGES = (1e-6, 1e-4, 1e-2, 1e-1, 1.0)


def gram(A, t, chunk=8192):
    """``(A'A, A't)`` accumulated in float64 without materialising A in float64.

    Pair fractions are strongly collinear -- two tables sharing a wire agree on a
    quarter of their address -- so the Gram matrix is near-singular and a float32
    accumulation loses the small eigenvalues that carry the interaction signal.
    That is not a modelling result, it is round-off, and it is what made a first
    pass report that adding tables destroys the fit.
    """
    n, m = A.shape
    G = np.zeros((m, m), dtype=np.float64)
    b = np.zeros(m, dtype=np.float64)
    for i in range(0, n, chunk):
        blk = A[i:i + chunk].astype(np.float64)
        G += blk.T @ blk
        b += blk.T @ t[i:i + chunk]
    return G, b


def ridge_path(Afit, yfit, Apick, ridges=RIDGES):
    """Scores for a path of ridge values from one eigendecomposition."""
    mu = Afit.mean(0, dtype=np.float64)
    sd = Afit.std(0, dtype=np.float64)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Af = (Afit - mu.astype(np.float32)) / sd.astype(np.float32)
    Ap = (Apick - mu.astype(np.float32)) / sd.astype(np.float32)
    t = (yfit - yfit.mean()).astype(np.float64)
    G, b = gram(Af, t)
    scale = float(np.trace(G)) / G.shape[0]
    ev, U = np.linalg.eigh(G)
    Ub = U.T @ b
    for lam in ridges:
        w = U @ (Ub / (ev + lam * scale + 1e-12))
        yield lam, Ap @ w.astype(np.float32)


def best_gated(s, ypick, n_pick_per, ctr_pick):
    s = np.asarray(s, dtype=np.float64)
    raw = per_unit_auc(s, ypick, n_pick_per)
    best, how = raw, "none"
    for r in (14.0, 18.0):
        g = gate_matrix(s, ctr_pick, n_pick_per, r)[:, 0]
        sc = np.std(s) / max(np.std(g), 1e-12)
        for w in (0.25, 0.5, 1.0):
            a = per_unit_auc(s + w * g * sc, ypick, n_pick_per)
            if a > best:
                best, how = a, f"r={r:g},w={w:g}"
    return raw, best, how


def main() -> int:
    z = np.load(CACHE, allow_pickle=False)
    X, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    names = [str(s) for s in z["names"]]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}

    clusters = sorted({cluster_of[u] for u in units})
    rng = np.random.default_rng(SEED)
    rng.shuffle(clusters)
    fit_clusters = set(clusters[:len(clusters) // 2])
    is_fit = np.array([cluster_of[u] in fit_clusters for u in units])
    row_unit = np.repeat(np.arange(len(units)), n_res)
    fm, pm = is_fit[row_unit], ~is_fit[row_unit]
    n_pick_per = np.array([n for n, f in zip(n_res, is_fit) if not f])
    yfit, ypick = y[fm].astype(np.float64), y[pm]
    ctr_pick = ctr[pm]
    rate = float(yfit.mean())
    print(f"fit {int(fm.sum())} residues / pick {int(pm.sum())}; "
          f"{X.shape[1]} wires, base rate {rate:.4f}", flush=True)

    D = chain_digits(X, n_res)
    Dfit, Dpick = D[fm], D[pm]
    g = np.array([abs(2.0 * pooled_auc(Dfit[:, j].astype(float), yfit) - 1.0)
                  for j in range(D.shape[1])])
    order = np.argsort(-g)

    results = []
    for K in (30, 45, 60):
        cols = sorted(int(j) for j in order[:K])
        n_pairs = K * (K - 1) // 2
        cells = n_pairs * L * L
        print(f"\ntop-{K}: {n_pairs} width-two tables, {cells} cells, "
              f"~{int(fm.sum()) / (L * L)} residues per cell", flush=True)
        Pf, Pp, _pairs = pair_fractions(Dfit, yfit, Dpick, cols, rate)

        for tag, Af, Ap in (
            (f"pair_{K}", Pf, Pp),
            (f"pair_{K}+lin",
             np.hstack([Pf, Dfit.astype(np.float32)]),
             np.hstack([Pp, Dpick.astype(np.float32)])),
        ):
            for lam, s in ridge_path(Af, yfit, Ap):
                raw, best, how = best_gated(s, ypick, n_pick_per, ctr_pick)
                results.append({"readout": tag, "ridge": lam,
                                "n_parameters": int(Af.shape[1]),
                                "pick_half_roc_auc_raw": round(float(raw), 4),
                                "pick_half_roc_auc": round(float(best), 4),
                                "gate": how})
                print(f"  {tag:16s} dim={Af.shape[1]:5d} ridge={lam:<7g} "
                      f"raw={raw:.4f} gated={best:.4f} ({how})", flush=True)
            del Af, Ap
        del Pf, Pp

    results.sort(key=lambda r: -r["pick_half_roc_auc"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "pairwise_readout_selection/v1",
        "split": "training fold, cluster-disjoint fit/pick halves",
        "seed": SEED, "ridge": RIDGE, "table_width": 2, "n_levels": L,
        "reference": {"width_six_bank": 0.7631, "first_order_linear": 0.7873},
        "candidates": results,
        "selected": results[0],
    }, indent=2) + "\n")
    print(f"\nselected: {results[0]['readout']} "
          f"{results[0]['pick_half_roc_auc']:.4f}\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
