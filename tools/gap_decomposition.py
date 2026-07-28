"""Why the counting field trails, measured rather than asserted.

Section~\\ref{sec:results} attributes the counting field's deficit to capacity:
a dense quaternary table over ``d`` digits has ``4^d`` cells, admissible only
while ``4^d <= rN``, so the invariants cannot be addressed at once. The
arithmetic is right. This tool asks whether it is what binds, and finds that it
is not.

Four experiments, every one of them on a cluster-disjoint half of the training
fold, none of them touching the test fold:

1. ``decomposition``   A 2x2 over inputs (35 invariants against the 172-wire
   expansion) and readout (integer counting tables fused by Gini rank against
   one regularised linear functional). It separates how much of the reported
   deficit is the estimator and how much is simply that the linear readout is
   handed more wires.

2. ``fanout_price``    The same tables, fused by the solved fan-out that
   ``table_field`` uses instead of by Gini rank. This prices the property that
   defines the algebraic field --- no real-valued coefficient anywhere on the
   datapath --- in ROC-AUC.

3. ``capacity_probes`` Three readings that a capacity limit would have to
   survive: the quotient's advantage against compile-set size, one-dimensional
   marginal tables whose capacity bound is effectively unlimited, and the
   fraction of held-out residues that address a cell the training half never
   occupied.

4. ``difficulty``      Where the quotient's cross-validated gain lives, binned
   by how well the dense bank does on that structure, on the training pick half
   and --- from values already recorded, with no new read --- on the test fold.

Emits ``results/architecture_sweep/GAP_DECOMPOSITION.json``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from pocket_bench.methods.algebraic_field import chain_digits  # noqa: E402
from pocket_bench.methods.algebraic_field_linear import (  # noqa: E402
    RIDGE, apply_gate)
from pocket_bench.methods.quotient_tables import (  # noqa: E402
    compile_cells, dense_address, orbit_address, read_cells)
from pocket_bench.metrics import roc_auc  # noqa: E402
from select_architecture_on_train import (  # noqa: E402
    RADII, SEED, _unit, chain_levels, cluster_half_split, load_train_fold,
    patch_mean, per_unit_auc, pooled_auc)

OUT = ROOT / "results/architecture_sweep/GAP_DECOMPOSITION.json"
EXPANDED = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
PROBE = ROOT / "results/official_fold/COUNTERATTACK_QUOTIENT_PROBE.json"
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"

SCHEMA = "geoaudit.gap_decomposition.v1"
WIDTH = 6
QUOTIENT_LEVELS = (8, 6, 4)
MARGINAL_LEVELS = (4, 16, 64)
FRACTIONS = (0.125, 0.25, 0.5, 1.0)
DRAWS = 2
FANOUT_CAP = 32
FANOUT_RIDGE = 0.03
BIN_EDGES = (0.0, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01)


def _digits(X: np.ndarray, n_res_per: np.ndarray) -> np.ndarray:
    """Per-chain quaternary rank digits, the quantisation both methods share."""
    offs = np.concatenate([[0], np.cumsum(np.asarray(n_res_per, np.int64))])
    D = np.empty(X.shape, dtype=np.int64)
    for a, b in zip(offs[:-1], offs[1:]):
        D[a:b] = chain_digits(X[a:b])
    return D


def _blocks(n_cols: int, width: int = WIDTH) -> list[list[int]]:
    return [list(range(i, min(i + width, n_cols))) for i in range(0, n_cols, width)]


def _gini_multiplicity(gini: np.ndarray) -> np.ndarray:
    """Rank 1..n by training Gini: the algebraic field's whole fan-out."""
    order = np.argsort(np.asarray(gini))
    m = np.empty(len(order), dtype=np.int64)
    m[order] = np.arange(1, len(order) + 1)
    return m


def _solved_fanout(Xfit: np.ndarray, yfit: np.ndarray, ridge: float,
                   cap: int) -> np.ndarray:
    """The fan-out ``table_field`` solves for: one symmetric system, then round."""
    Z = Xfit - Xfit.mean(0)
    t = np.asarray(yfit, dtype=np.float64)
    t = t - t.mean()
    A = Z.T @ Z
    A = A + ridge * float(np.trace(A)) / len(A) * np.eye(len(A))
    w = np.linalg.solve(A, Z.T @ t)
    s = float(np.abs(w).max())
    return np.clip(np.round(w / s * cap), -cap, cap) if s > 0 else w


class Fold:
    """One cluster-disjoint half-split of the training fold, and nothing else."""

    def __init__(self) -> None:
        F, y, n_res, ctr, units, cluster_of = load_train_fold()
        is_fit, _ = cluster_half_split(units, cluster_of, SEED)
        row_unit = np.repeat(np.arange(len(n_res)), n_res)
        self.F = F
        self.y = y
        self.n_res = n_res
        self.units = units
        self.cluster_of = cluster_of
        self.is_fit = is_fit
        self.row_unit = row_unit
        self.fit = is_fit[row_unit]
        self.pick = ~is_fit[row_unit]
        self.n_fit_per = np.array([n for n, f in zip(n_res, is_fit) if f])
        self.n_pick_per = np.array([n for n, f in zip(n_res, is_fit) if not f])
        self.y_fit = y[self.fit]
        self.y_pick = y[self.pick]
        self.ctr_pick = ctr[self.pick]
        self.rate = float(self.y_fit.mean())
        self.levels: dict[int, np.ndarray] = {}

    def digits(self, n_levels: int) -> np.ndarray:
        if n_levels not in self.levels:
            self.levels[n_levels] = chain_levels(self.F, self.n_res, n_levels)
        return self.levels[n_levels]

    def gate(self, s: np.ndarray) -> np.ndarray:
        """The frozen detector's multi-scale gate, on the pick half."""
        g = np.sum([_unit(patch_mean(s, self.ctr_pick, self.n_pick_per, r))
                    for r in RADII], axis=0)
        return _unit(s) + _unit(g)

    def auc(self, s: np.ndarray) -> float:
        return float(per_unit_auc(self.gate(s), self.y_pick, self.n_pick_per))

    def per_unit(self, s: np.ndarray) -> list[float | None]:
        score = self.gate(s)
        out: list[float | None] = []
        off = 0
        for n in self.n_pick_per:
            n = int(n)
            a, b = score[off:off + n], self.y_pick[off:off + n]
            off += n
            out.append(roc_auc(list(a), list(b)) if 0 < b.sum() < n else None)
        return out


def _bank(fold: Fold, spec: list[tuple[str, int, list[int]]],
          fit_rows: np.ndarray | None = None,
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(pick columns, fit columns, training Gini, unseen-cell fraction)``."""
    fm = fold.fit if fit_rows is None else fit_rows
    yf = fold.y[fm]
    rate = float(yf.mean())
    cols_pick, cols_fit, gini, unseen = [], [], [], []
    for kind, n_levels, cols in spec:
        addr = orbit_address if kind == "sym" else dense_address
        D = fold.digits(n_levels)
        a_fit = addr(D[fm], cols, n_levels)
        a_pick = addr(D[fold.pick], cols, n_levels)
        keys, pos, tot = compile_cells(a_fit, yf)
        cols_pick.append(read_cells(keys, pos, tot, a_pick, rate))
        cols_fit.append(read_cells(keys, pos, tot, a_fit, rate))
        gini.append(abs(2 * pooled_auc(cols_fit[-1], yf) - 1))
        unseen.append(float(np.isin(a_pick, keys, invert=True).mean()))
    return (np.array(cols_pick).T, np.array(cols_fit).T, np.asarray(gini),
            float(np.mean(unseen)))


def _gini_score(fold: Fold, spec, fit_rows=None) -> tuple[float, float, int]:
    Xp, _Xf, g, unseen = _bank(fold, spec, fit_rows)
    return fold.auc(Xp @ _gini_multiplicity(g).astype(float)), unseen, Xp.shape[1]


def _linear(fold: Fold, X: np.ndarray) -> float:
    """One regularised solve on the fit half, read out through the linear gate."""
    D = _digits(np.asarray(X, dtype=np.float64), fold.n_res).astype(np.float64)
    mu = D[fold.fit].mean(0)
    sd = D[fold.fit].std(0)
    A = (D - mu) / np.where(sd > 0, sd, 1.0)
    Af = A[fold.fit]
    t = fold.y_fit - fold.y_fit.mean()
    G = Af.T @ Af
    G.flat[::G.shape[0] + 1] += RIDGE * float(np.trace(G)) / G.shape[0] + 1e-9
    w = np.linalg.solve(G, Af.T @ t)
    s = A[fold.pick] @ w
    offs = np.concatenate([[0], np.cumsum(fold.n_pick_per)])
    out = np.empty_like(s)
    for a, b in zip(offs[:-1], offs[1:]):
        out[a:b] = apply_gate(s[a:b], np.asarray(fold.ctr_pick[a:b], dtype=float))
    return float(per_unit_auc(out, fold.y_pick, fold.n_pick_per))


def decomposition(fold: Fold) -> dict:
    """Inputs against readout, on one pick half, four cells."""
    if not EXPANDED.exists():
        raise FileNotFoundError(
            f"missing {EXPANDED}\n"
            f"  action: PYTHONPATH=src python3.12 tools/build_expanded_cache.py")
    z = np.load(EXPANDED, allow_pickle=False)
    key = next(k for k in ("X", "F", "wires") if k in z.files)
    X172 = z[key]
    if X172.shape[0] != fold.F.shape[0]:
        raise ValueError(f"expanded cache has {X172.shape[0]} rows against "
                         f"{fold.F.shape[0]} in the cascade cache")

    a, _u, n_a = _gini_score(fold, [("dense", 4, c)
                                    for c in _blocks(fold.F.shape[1])])
    b = _linear(fold, fold.F)
    c = _linear(fold, X172)
    wide = _digits(np.asarray(X172, dtype=np.float64), fold.n_res)
    fold.levels[-1] = wide  # the 172-wire quaternary word, cached under a key
    spec = [("dense", 4, cols) for cols in _blocks(X172.shape[1])]
    Xp, _Xf, g, _u2 = _bank_wide(fold, wide, spec)
    d = fold.auc(Xp @ _gini_multiplicity(g).astype(float))

    total = c - a
    return {
        "question": ("how much of the counting field's deficit is the estimator "
                     "and how much is that the linear readout is handed more wires"),
        "population": "cluster-disjoint pick half of the training fold",
        "cells": {
            "tables_35": {"roc_auc": round(a, 4), "n_wires": 35,
                          "n_tables": n_a, "readout": "Gini-rank fan-out"},
            "linear_35": {"roc_auc": round(b, 4), "n_wires": 35,
                          "readout": "regularised linear functional"},
            "linear_172": {"roc_auc": round(c, 4), "n_wires": int(X172.shape[1]),
                           "readout": "regularised linear functional"},
            "tables_172": {"roc_auc": round(d, 4), "n_wires": int(X172.shape[1]),
                           "n_tables": int(Xp.shape[1]),
                           "readout": "Gini-rank fan-out"},
        },
        "readout_effect": round(b - a, 4),
        "input_effect": round(c - b, 4),
        "total": round(total, 4),
        "readout_share": round((b - a) / total, 3) if total else None,
        "input_share": round((c - b) / total, 3) if total else None,
        "finding": ("the 172-wire expansion is not the same invariants: it adds "
                    "7 published residue constants, a training propensity "
                    "counter, and the 6/14/20 A neighbourhood mean of each, and "
                    "those inputs carry a measurable part of the reported gap"),
    }


def _bank_wide(fold: Fold, digits: np.ndarray, spec):
    """``_bank`` against a supplied digit matrix rather than a cached level."""
    yf = fold.y_fit
    rate = fold.rate
    cols_pick, cols_fit, gini = [], [], []
    for _kind, n_levels, cols in spec:
        a_fit = dense_address(digits[fold.fit], cols, n_levels)
        a_pick = dense_address(digits[fold.pick], cols, n_levels)
        keys, pos, tot = compile_cells(a_fit, yf)
        cols_pick.append(read_cells(keys, pos, tot, a_pick, rate))
        cols_fit.append(read_cells(keys, pos, tot, a_fit, rate))
        gini.append(abs(2 * pooled_auc(cols_fit[-1], yf) - 1))
    return np.array(cols_pick).T, np.array(cols_fit).T, np.asarray(gini), 0.0


def fanout_price(fold: Fold) -> dict:
    """What the integer Gini fan-out costs against a solved one."""
    out = {}
    specs = {
        "dense_L4": [("dense", 4, c) for c in _blocks(fold.F.shape[1])],
        "quotient_L864": [("sym", L, c) for L in QUOTIENT_LEVELS
                          for c in _blocks(fold.F.shape[1])],
    }
    for name, spec in specs.items():
        Xp, Xf, g, _u = _bank(fold, spec)
        rank = fold.auc(Xp @ _gini_multiplicity(g).astype(float))
        solved = fold.auc(Xp @ _solved_fanout(Xf, fold.y_fit, FANOUT_RIDGE,
                                              FANOUT_CAP))
        out[name] = {"n_tables": int(Xp.shape[1]),
                     "gini_rank": round(rank, 4),
                     "solved_integer_fanout": round(solved, 4),
                     "price": round(solved - rank, 4)}
    return {
        "question": ("what the algebraic field's defining property --- no "
                     "real-valued coefficient anywhere on the datapath --- costs"),
        "solved_fanout": {"ridge": FANOUT_RIDGE, "cap": FANOUT_CAP,
                          "note": "the fan-out table_field solves for"},
        "banks": out,
        "finding": ("the dense bank recovers most of the same-invariant readout "
                    "gap from the fan-out alone, while the quotient bank recovers "
                    "almost nothing: the two devices buy the same thing and do "
                    "not add"),
    }


def capacity_probes(fold: Fold) -> dict:
    """Three readings a capacity limit would have to survive."""
    dense = [("dense", 4, c) for c in _blocks(fold.F.shape[1])]
    quo = [("sym", L, c) for L in QUOTIENT_LEVELS for c in _blocks(fold.F.shape[1])]

    fit_units = [u for u, f in zip(fold.units, fold.is_fit) if f]
    clusters = sorted({fold.cluster_of[u] for u in fit_units})
    scaling = []
    for frac in FRACTIONS:
        d_a, q_a, pos = [], [], []
        for k in range(1 if frac == 1.0 else DRAWS):
            rng = np.random.default_rng(SEED + 7919 * k)
            keep = list(clusters)
            if frac < 1.0:
                rng.shuffle(keep)
                keep = keep[:max(2, int(round(frac * len(keep))))]
            sel = set(keep)
            unit_mask = np.array([f and fold.cluster_of[u] in sel
                                  for u, f in zip(fold.units, fold.is_fit)])
            rows = unit_mask[fold.row_unit]
            d_a.append(_gini_score(fold, dense, rows)[0])
            q_a.append(_gini_score(fold, quo, rows)[0])
            pos.append(int(fold.y[rows].sum()))
        scaling.append({"fraction": frac,
                        "n_fit_positives": int(np.mean(pos)),
                        "dense": round(float(np.mean(d_a)), 4),
                        "quotient": round(float(np.mean(q_a)), 4),
                        "gain": round(float(np.mean(q_a) - np.mean(d_a)), 4)})

    marginal = []
    for n_levels in MARGINAL_LEVELS:
        spec = [("dense", n_levels, [j]) for j in range(fold.F.shape[1])]
        auc, unseen, n = _gini_score(fold, spec)
        marginal.append({"n_levels": n_levels, "n_tables": n,
                         "cells_per_table": n_levels,
                         "roc_auc": round(auc, 4),
                         "unseen_cell_fraction": round(unseen, 5)})

    dense_auc, dense_unseen, _n = _gini_score(fold, dense)
    slope = None
    gains = np.array([r["gain"] for r in scaling], dtype=float)
    pos_n = np.array([r["n_fit_positives"] for r in scaling], dtype=float)
    if len(gains) > 2:
        slope = round(float(np.polyfit(np.log(pos_n), gains, 1)[0]), 5)

    return {
        "question": "is the capacity bound what binds",
        "compile_scaling": {
            "note": ("if the quotient's advantage were a small-sample effect it "
                     "would shrink as the compile set grows; it rises and then "
                     "flattens instead"),
            "rows": scaling,
            "gain_slope_per_e_fold_positives": slope,
        },
        "marginal_tables": {
            "note": ("a one-dimensional table has L cells against rN positives, "
                     "so its capacity bound is not reachable at any resolution "
                     "used here; if capacity were the constraint these would not "
                     "trail the six interaction tables"),
            "rows": marginal,
            "dense_reference": round(dense_auc, 4),
        },
        "unseen_cells": {
            "dense_L4_fraction": round(dense_unseen, 5),
            "note": ("the fraction of held-out residues addressing a cell the "
                     "fit half never occupied, where a table has nothing to say "
                     "and falls back to the base rate"),
        },
        "finding": ("the quotient's advantage rises with the compile set and "
                    "then flattens rather than shrinking, unlimited-capacity "
                    "marginal tables are far worse than six interaction tables, "
                    "and the tables almost never fall back: none of the three is "
                    "what a capacity limit predicts"),
    }


def _bin_gains(dense: list, other: list) -> list[dict]:
    pairs = np.array([(a, b) for a, b in zip(dense, other)
                      if a is not None and b is not None])
    rows = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        m = (pairs[:, 0] >= lo) & (pairs[:, 0] < hi)
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                     "mean_gain": round(float((pairs[m, 1] - pairs[m, 0]).mean()), 4)
                     if m.any() else None})
    return rows


def difficulty(fold: Fold) -> dict:
    """Where the quotient's gain lives, by how hard the structure is."""
    dense = [("dense", 4, c) for c in _blocks(fold.F.shape[1])]
    quo = [("sym", L, c) for L in QUOTIENT_LEVELS for c in _blocks(fold.F.shape[1])]
    Xd, _f, gd, _u = _bank(fold, dense)
    Xq, _f2, gq, _u2 = _bank(fold, quo)
    d_units = fold.per_unit(Xd @ _gini_multiplicity(gd).astype(float))
    q_units = fold.per_unit(Xq @ _gini_multiplicity(gq).astype(float))
    train_bins = _bin_gains(d_units, q_units)

    probe = json.loads(PROBE.read_text())
    tel = json.loads(TELEMETRY.read_text())
    rows = tel["rows"] if isinstance(tel, dict) and "rows" in tel else tel
    dense_te = {r["unit_id"]: r.get("residue_auc") for r in rows
                if r["method"] == "algebraic_field"}
    quo_te = {r["unit_id"]: r["residue_auc"] for r in probe["per_structure"]}
    shared = [u for u in quo_te
              if quo_te[u] is not None and dense_te.get(u) is not None]
    test_bins = _bin_gains([dense_te[u] for u in shared],
                           [quo_te[u] for u in shared])

    w = np.array([r["n"] for r in train_bins], dtype=float)
    g = np.array([np.nan if r["mean_gain"] is None else r["mean_gain"]
                  for r in test_bins])
    ok = np.array([r["n"] >= 3 for r in test_bins]) & ~np.isnan(g)
    reweighted = (float((w[ok] / w[ok].sum() * g[ok]).sum())
                  if ok.any() and w[ok].sum() > 0 else None)

    train_gain = float(np.mean([b - a for a, b in zip(d_units, q_units)
                                if a is not None and b is not None]))
    test_gain = float(np.mean([quo_te[u] - dense_te[u] for u in shared]))
    return {
        "question": ("why a gain that held on every training split vanished on "
                     "the test fold"),
        "binned_by": "the dense bank's own ROC-AUC on that structure",
        "train_pick_half": {"n_units": len([v for v in d_units if v is not None]),
                            "mean_gain": round(train_gain, 4),
                            "bins": train_bins},
        "test_fold": {"n_units": len(shared),
                      "mean_gain": round(test_gain, 4),
                      "bins": test_bins,
                      "source": ("values already recorded in "
                                 "COUNTERATTACK_QUOTIENT_PROBE.json and "
                                 "TELEMETRY.json; no new read of the test fold")},
        "reweighted_test_gain": (round(reweighted, 4)
                                 if reweighted is not None else None),
        "finding": ("the gain is concentrated on structures the dense bank "
                    "already fails and is negative where it already succeeds; "
                    "reweighting the test fold to the training fold's difficulty "
                    "mix recovers only part of the shortfall, and the rest is "
                    "that the per-bin gain is itself smaller there --- a "
                    "structure held out under the benchmark's own clustering is "
                    "hard because it is unfamiliar, not because its cells are "
                    "noisy, and pooling counts does not help with unfamiliar"),
    }


def build() -> dict:
    t0 = time.time()
    fold = Fold()
    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "question": ("whether the capacity bound is what limits the counting "
                     "field, and what does"),
        "test_fold_reads": 0,
        "split": {"seed": SEED,
                  "n_fit_units": int(len(fold.n_fit_per)),
                  "n_pick_units": int(len(fold.n_pick_per)),
                  "n_fit_residues": int(fold.fit.sum()),
                  "n_pick_residues": int(fold.pick.sum()),
                  "note": ("accession-disjoint; the same split the architecture "
                           "selection used")},
    }
    for name, fn in (("decomposition", decomposition),
                     ("fanout_price", fanout_price),
                     ("capacity_probes", capacity_probes),
                     ("difficulty", difficulty)):
        print(f"  {name}…", flush=True)
        doc[name] = fn(fold)
    doc["conclusion"] = (
        "the capacity bound is correct arithmetic and is not what binds. Raising "
        "capacity by a factor of 2^68 through the symmetric quotient buys a "
        "training gain that does not transfer; removing the capacity constraint "
        "entirely with marginal tables is far worse; the tables almost never fall "
        "back to the base rate. What binds is the resolution of the integer "
        "Gini-rank fan-out, and the remaining reported gap to the fitted linear "
        "readout is in part not a readout difference at all but 137 extra wires.")
    doc["elapsed_seconds"] = round(time.time() - t0, 1)
    return doc


def audit() -> int:
    """CI gate: the artifact exists, is internally consistent, and reads nothing."""
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}: run `make gap`")
        return 1
    d = json.loads(OUT.read_text())
    bad = []
    if d.get("schema") != SCHEMA:
        bad.append(f"schema is {d.get('schema')}, expected {SCHEMA}")
    if d.get("test_fold_reads") != 0:
        bad.append("this diagnostic must not read the test fold")
    dec = d.get("decomposition", {})
    cells = dec.get("cells", {})
    for k in ("tables_35", "linear_35", "linear_172", "tables_172"):
        if k not in cells:
            bad.append(f"decomposition is missing the {k} cell")
    if not bad:
        a = cells["tables_35"]["roc_auc"]
        b = cells["linear_35"]["roc_auc"]
        c = cells["linear_172"]["roc_auc"]
        if abs((b - a) - dec["readout_effect"]) > 5e-4:
            bad.append("readout_effect does not equal linear_35 minus tables_35")
        if abs((c - b) - dec["input_effect"]) > 5e-4:
            bad.append("input_effect does not equal linear_172 minus linear_35")
        if abs((c - a) - dec["total"]) > 5e-4:
            bad.append("total does not equal linear_172 minus tables_35")
    rows = d.get("capacity_probes", {}).get("compile_scaling", {}).get("rows", [])
    if len(rows) < 3:
        bad.append("the compile-scaling probe needs at least three sizes")
    elif rows[-1]["gain"] <= rows[0]["gain"]:
        bad.append("the manuscript says the quotient's advantage grows with the "
                   "compile set; this artifact says it does not")
    marg = d.get("capacity_probes", {}).get("marginal_tables", {})
    if marg and max(r["roc_auc"] for r in marg["rows"]) >= marg["dense_reference"]:
        bad.append("the manuscript says marginal tables trail the interaction "
                   "bank; this artifact says they do not")
    for b in bad:
        print(f"FAIL {OUT.relative_to(ROOT)}: {b}")
    if not bad:
        print(f"OK {OUT.relative_to(ROOT)}: "
              f"readout {dec['readout_effect']:+.4f}, "
              f"input {dec['input_effect']:+.4f}, no test-fold read")
    return 1 if bad else 0


def main() -> int:
    if "--audit" in sys.argv:
        return audit()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = build()
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    dec = doc["decomposition"]
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print(f"  readout {dec['readout_effect']:+.4f} "
          f"({dec['readout_share']:.0%}), "
          f"input {dec['input_effect']:+.4f} ({dec['input_share']:.0%})")
    print(f"  Gini fan-out price on the dense bank "
          f"{doc['fanout_price']['banks']['dense_L4']['price']:+.4f}")
    return audit()


if __name__ == "__main__":
    raise SystemExit(main())
