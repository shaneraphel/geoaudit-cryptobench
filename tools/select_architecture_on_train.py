#!/usr/bin/env python3
"""Choose the combinational architecture without ever consulting the test fold.

Seven fusion architectures were compared while developing the algebraic field.
Comparing them on the official test fold and then reporting the winner's test
score is selection on the evaluation set: the reported number then carries the
maximum of seven noisy estimates, and it is optimistic by an amount nobody has
measured. The defect is invisible in the final artifact, which is why it has to
be excluded by construction rather than argued away.

So the choice is made here, on the TRAINING fold only, split into two
cluster-disjoint halves:

    fit half   compile every candidate's tables
    pick half  score them; the winner is the architecture with the highest
               per-unit mean ROC-AUC on units whose clusters the fit half
               never saw

The split is on ``cluster_id`` from ``train_manifest.json``. Be precise about
what that field is, because an earlier version of this docstring was not: it is
the UniProt accession, written by ``tools/build_training_fold.py``, and it is
NOT the MMseqs2 10 % cluster. The 10 % clustering is real, but it lives upstream
in CryptoBench's own fold construction, and the cluster assignment is not
distributed with the data -- ``folds.json`` ships fold membership, not cluster
labels. Splitting on the accession is therefore finer than splitting on a 10 %
cluster: two accessions that MMseqs2 would have grouped can land on opposite
sides here. What this split does guarantee is that no single protein is both fit
on and picked on.

The coarser guarantee is available, and ``tools/crossvalidate_architecture.py``
uses it: CryptoBench ships four training folds, train-0 to train-3, built by its
authors under that 10 % clustering and pairwise disjoint in both PDB id and
accession. Holding each out in turn is a cluster-level cross-validation of this
selection under the benchmark's own clustering rather than a proxy for it. This
file stays as it is, on the accession halves, because it is what the frozen
result was selected with; the cross-validation reports whether that choice
survives the coarser and the repeated splits.

The test fold is then read exactly once, for the single architecture named here.

Usage: PYTHONPATH=src python3.12 tools/select_architecture_on_train.py
"""
from __future__ import annotations

import json

import numpy as np

from pocket_bench.metrics import roc_auc
from pocket_bench.paths import ROOT

CACHE = ROOT / "data/cryptobench_apo/_cascade_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT = ROOT / "results/architecture_sweep/TRAIN_ONLY_SELECTION.json"
L = 4
RADII = (6.0, 8.0, 10.0, 14.0, 18.0)
SEED = 20260725


# ---------------------------------------------------------------- quantization
def chain_levels(F, n_res_per, levels):
    """Each column banded by its own chain's rank order: a comparator network."""
    out = np.empty(F.shape, dtype=np.int64)
    off = 0
    for n in n_res_per:
        n = int(n)
        blk = F[off:off + n]
        for j in range(F.shape[1]):
            x = blk[:, j]
            order = np.argsort(x, kind="stable")
            r = np.empty(n)
            i = 0
            while i < n:
                k = i
                while k + 1 < n and x[order[k + 1]] == x[order[i]]:
                    k += 1
                r[order[i:k + 1]] = 0.5 * (i + k)
                i = k + 1
            out[off:off + n, j] = np.clip(
                np.floor(r / max(n - 1, 1) * levels).astype(np.int64),
                0, levels - 1)
        off += n
    return out


def per_unit_auc(score, y, n_res_per):
    aucs = []
    off = 0
    for n in n_res_per:
        n = int(n)
        s, t = score[off:off + n], y[off:off + n]
        off += n
        if t.sum() == 0 or t.sum() == n:
            continue
        a = roc_auc(list(s), list(t))
        if a is not None:
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


def patch_mean(s, ctr, n_res_per, radius):
    out = np.empty(len(s))
    r2 = radius * radius
    off = 0
    for n in n_res_per:
        n = int(n)
        c, v = ctr[off:off + n], s[off:off + n]
        acc = np.empty(n)
        for i in range(0, n, 512):
            d2 = ((c[i:i + 512, None, :] - c[None, :, :]) ** 2).sum(-1)
            a = (d2 <= r2).astype(np.float64)
            acc[i:i + 512] = (a @ v) / np.maximum(a.sum(1), 1.0)
        out[off:off + n] = acc
        off += n
    return out


def _unit(x):
    m = float(np.abs(x).max())
    return x / m if m > 0 else x


def pooled_auc(x, y):
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ------------------------------------------------------------------ candidates
def bank_fracs(Dfit, yfit, Dpick, tables, levels, rate):
    yf = yfit.astype(np.float64)
    fr_pick, gini = [], []
    for cols in tables:
        n_cells = levels ** len(cols)
        a_fit = np.zeros(len(yfit), dtype=np.int64)
        a_pick = np.zeros(Dpick.shape[0], dtype=np.int64)
        for t, c in enumerate(cols):
            a_fit += Dfit[:, c] * (levels ** t)
            a_pick += Dpick[:, c] * (levels ** t)
        tot = np.bincount(a_fit, minlength=n_cells).astype(np.float64)
        pos = np.bincount(a_fit, weights=yf, minlength=n_cells)
        frac = np.where(tot > 0, pos / np.maximum(tot, 1.0), rate)
        fr_pick.append(frac[a_pick])
        gini.append(abs(2.0 * pooled_auc(frac[a_fit], yfit) - 1.0))
    return fr_pick, np.asarray(gini)


def load_train_fold():
    """The cached training-fold descriptors, and each unit's cluster label."""
    z = np.load(CACHE, allow_pickle=False)
    F, y, n_res, ctr = z["F"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    missing = [u for u in units if u not in cluster_of]
    if missing:
        raise SystemExit(f"{len(missing)} cached units absent from the manifest, "
                         f"e.g. {missing[:3]}")
    return F, y, n_res, ctr, units, cluster_of


def build_level_cache(F, n_res, levels=(4, 8, 16, 64)) -> dict[int, np.ndarray]:
    """Banding depends only on the descriptors, never on the split.

    Hoisted out of the candidate loop because it is the expensive part and it is
    identical for every split, which is what makes repeating the selection over
    many splits affordable.
    """
    return {n: chain_levels(F, n_res, n) for n in levels}


def cluster_half_split(units, cluster_of, seed):
    """Halve the units so that no cluster label straddles the two halves.

    Reproducible from the seed alone. Note what the guarantee is worth: see the
    note on ``cluster_id`` in this module's docstring. Whole clusters move
    together, which is all this function promises.
    """
    clusters = sorted({cluster_of[u] for u in units})
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    fit_clusters = set(clusters[:len(clusters) // 2])
    is_fit_unit = np.array([cluster_of[u] in fit_clusters for u in units])
    straddling = {c for c in clusters
                  if len({bool(f) for u, f in zip(units, is_fit_unit)
                          if cluster_of[u] == c}) > 1}
    if straddling:
        raise SystemExit(f"{len(straddling)} clusters span both halves: "
                         f"{sorted(straddling)[:5]}")
    return is_fit_unit, clusters


def evaluate_candidates(F, y, n_res, ctr, level_cache, is_fit_unit,
                        verbose=True) -> list[dict]:
    """Score every candidate architecture on one fit/pick split.

    Split in, ranking out, and nothing else: the architectures are defined once,
    here, so that repeating the selection over other splits cannot silently
    compare a different set of candidates than the frozen run did.
    """
    row_unit = np.repeat(np.arange(len(n_res)), n_res)
    fit_mask = is_fit_unit[row_unit]
    pick_mask = ~fit_mask

    n_pick_per = np.array([n for n, f in zip(n_res, is_fit_unit) if not f])
    rate = float(y[fit_mask].mean())
    M = F.shape[1]

    results = []
    D4 = level_cache[4]
    Dfit4, Dpick4 = D4[fit_mask], D4[pick_mask]
    yfit, ypick = y[fit_mask], y[pick_mask]
    ctr_pick = ctr[pick_mask]

    thematic = [list(range(i, min(i + 6, M))) for i in range(0, M, 6)]

    def record(name, score, extra=None):
        a = per_unit_auc(score, ypick, n_pick_per)
        results.append({"architecture": name, "pick_half_roc_auc": a,
                        **(extra or {})})
        if verbose:
            print(f"  {name:44s} pick-half ROC-AUC = {a:.4f}", flush=True)
        return a

    # 1-2. flat single table, the control
    for w in (6, 7):
        cols = list(range(w))
        fr, _ = bank_fracs(Dfit4, yfit, Dpick4, [cols], 4, rate)
        record(f"flat single table, {w} wires", fr[0])

    # 3. parallel bank, unweighted counting fusion
    fr_th, gini_th = bank_fracs(Dfit4, yfit, Dpick4, thematic, 4, rate)
    S_uni = np.sum(fr_th, axis=0)
    record("parallel bank (thematic 6x6), uniform fusion", S_uni)

    # 4. same bank, integer multiplicity threshold fusion
    order = np.argsort(gini_th)
    mult = np.empty(len(thematic), dtype=np.int64)
    mult[order] = np.arange(1, len(thematic) + 1)
    S_thr = np.sum([m * f for m, f in zip(mult, fr_th)], axis=0)
    record("parallel bank, integer-multiplicity fusion", S_thr,
           {"multiplicity": mult.tolist()})

    # 5-6. either fusion, plus the multi-scale spatial counting gate
    best_named, best_auc = None, -1.0
    for label, S in (("uniform", S_uni), ("integer-multiplicity", S_thr)):
        G = np.sum([_unit(patch_mean(S, ctr_pick, n_pick_per, r))
                    for r in RADII], axis=0)
        a = record(f"parallel bank, {label} fusion + multi-scale gate",
                   _unit(S) + _unit(G))
        if a > best_auc:
            best_auc, best_named = a, f"{label} + gate"

    # 7. resolution trade: fewer wires per table, more levels per wire
    for levels, w in ((8, 4), (16, 3), (64, 2)):
        D = level_cache[levels]
        tabs = [list(range(i, min(i + w, M))) for i in range(0, M, w)]
        fr, _ = bank_fracs(D[fit_mask], yfit, D[pick_mask], tabs, levels, rate)
        record(f"wide-bus bank, {levels} levels x {w} wires/table",
               np.sum(fr, axis=0))

    return results


def main() -> int:
    F, y, n_res, ctr, units, cluster_of = load_train_fold()
    is_fit_unit, clusters = cluster_half_split(units, cluster_of, SEED)

    row_unit = np.repeat(np.arange(len(units)), n_res)
    fit_mask = is_fit_unit[row_unit]
    pick_mask = ~fit_mask
    n_fit_units = int(is_fit_unit.sum())
    n_pick_units = len(units) - n_fit_units
    n_fit_res = int(fit_mask.sum())
    rate = float(y[fit_mask].mean())
    print(f"train fold {len(units)} units / {len(clusters)} clusters -> "
          f"fit {n_fit_units} units ({n_fit_res} residues), "
          f"pick {n_pick_units} units", flush=True)

    level_cache = build_level_cache(F, n_res)
    results = evaluate_candidates(F, y, n_res, ctr, level_cache, is_fit_unit)

    winner = max(results, key=lambda r: r["pick_half_roc_auc"])
    print(f"\nselected on the training fold alone: {winner['architecture']} "
          f"({winner['pick_half_roc_auc']:.4f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.train_only_architecture_selection.v1",
        "clinical_grade": False,
        "why": "the architecture is chosen without reading the official test "
               "fold, so the reported test score is not the maximum of several "
               "test-fold estimates",
        "split": {
            "source": "data/cryptobench_apo/train_manifest.json",
            "criterion": "cluster_id (the UniProt accession, not the MMseqs2 "
                         "10% cluster), seeded shuffle, disjoint halves",
            "seed": SEED,
            "n_clusters": len(clusters),
            "n_fit_units": n_fit_units,
            "n_pick_units": n_pick_units,
            "n_fit_residues": n_fit_res,
            "n_pick_residues": int(pick_mask.sum()),
            "fit_base_rate": rate,
        },
        "candidates": results,
        "selected": winner,
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
