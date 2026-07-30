#!/usr/bin/env python3.12
"""Choose the table pairings instead of drawing them. Training folds only.

Where this comes from
---------------------
``partition_tables`` draws 16 random permutations of the 645 wires and cuts each
into pairs, so every pairing in the deployed bank is arbitrary. Two frozen
artifacts say that is the wrong thing to leave to a seed.

IS_FISHER_A_CEILING.json: the counting field beats a ridge Fisher solve over the
same wires by +0.0053 on 11 of 12 splits. A counting field is linear in the
indicator functions of quantised cells, so the whole of that margin lives in what
a *pair* of wires says jointly and neither says alone.

UNION_BANK_COUNTING_FIELD.json: widening the bus from 645 to 774 wires keeps only
296 of the 5,152 existing pairings, and that redraw alone turned a +0.0010 result
into -0.0007. Pairings looked like they carried the margin.

If they do, pairing wires that actually interact should beat pairing them at
random. That is measurable on the fit half without touching the pick half.

The criterion, and why it is a variance and not an information
--------------------------------------------------------------
Everything is a rational function of integer counts on the fit half. With N rows,
K positive and p = K/N, a partition of the rows into cells c with counts n_c and
positive counts k_c has between-cell label variance

    V = sum_c (k_c - p * n_c)^2 / (n_c * N)

which is the Gini reduction the bank's fan-out ranking already uses, written so
that no division by an empty cell occurs. For a single wire the cells are its
four levels; for a pair, its sixteen. Refining a partition cannot decrease V, so
V_pair >= max(V_u, V_v) always.

Two criteria run as separate arms because they ask different questions.

``interaction``  V_pair - V_u - V_v, the second-order term of a variance ANOVA:
what the joint says beyond both marginals added. It can be negative when two
wires are redundant. This is the criterion the argument above predicts, because
each wire already appears in 16 tables so its marginal is not scarce -- what is
scarce in the bank is interaction.

``pair variance``  V_pair alone, marginals included. Run because the prediction
is an argument, and an argument measured against no alternative is a correlate.

Controls, so that a positive result could be disbelieved
-------------------------------------------------------
``another seed`` redraws random pairings at a different seed. If selection beats
the deployed bank by about what a fresh draw does, the effect is seed sensitivity
and not selection. This arm is the one that decided the experiment.

``anti-selected`` minimises the interaction criterion. If the criterion means
anything, deliberately pairing the least interacting wires must lose. A criterion
that has never failed is indistinguishable from one that cannot.

Protocol
--------
Pairings are chosen per split from that split's fit half only, then compiled on
the fit half and scored on the pick half, exactly as the cell counts and the
integer fan-out already are. No pick-half row and no test residue enters the
selection. The selected pairings are integers and would ship with the detector as
a list of index pairs.

Rounds are edge-disjoint: round 1 takes a greedy maximum-weight matching over all
pairs, round 2 the same over the pairs no earlier round used, and so on for 16
rounds, so the bank holds the same 5,152 tables as the deployed one and every
wire appears once per round. Greedy rather than blossom because ``networkx`` is
not a declared dependency here; the artifact records what fraction of the
unconstrained ideal weight each round captured, so the approximation is visible
rather than assumed.

A note on why this file was written twice. The first copy was authored, run to
completion, and then deleted along with every other untracked file by a failed
workspace switch, while its artifact survived. An artifact whose tool is missing
is not reproducible, which is the one property this repository claims, so it was
re-authored against the artifact's own recorded specification and required to
reproduce it before being committed.

Training folds only. No test residue and no external unit is read.

Usage: PYTHONPATH=src:tools python3.12 tools/selected_pairings.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from expand_invariant_bank import SEED  # noqa: E402
from quantisation_ladder import (  # noqa: E402
    cell_occupancy,
    compile_at,
    fanout_at,
    offsets_at,
    score_at,
)
from select_architecture_on_train import cluster_half_split, per_unit_auc  # noqa: E402

from pocket_bench.methods.table_bank import (
    N_LEVELS,
    chain_digits,
    partition_tables,
)
from pocket_bench.methods.table_field import (
    GATE_RADIUS,
    GATE_WEIGHT,
    PARTITION_ROUNDS,
    PARTITION_SEED,
    TABLE_WIDTH,
    apply_gate,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.selected_pairings.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
OUT = ROOT / "results/architecture_sweep/SELECTED_PAIRINGS.json"

CONTROL_SEED = PARTITION_SEED + 1
ROW_BLOCK = 16384
ARMS = ("interaction", "pair variance", "another seed", "anti-selected")


def joint_counts(D: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All pairwise 4x4 contingency tables of (total, positive), at once.

    One-hot the digits into an ``N x (n_wires * N_LEVELS)`` indicator and form
    its Gram matrix: entry ``(u*4+a, v*4+b)`` counts the rows at level ``a`` of
    wire ``u`` and level ``b`` of wire ``v``, which is exactly that pair's joint
    table. Every one of the 207,690 pairs therefore comes out of one matrix
    product rather than a separate pass, and on this machine it takes a second.
    Accumulated over row blocks so the indicator is never materialised whole.
    """
    n, w = D.shape
    m = w * N_LEVELS
    tot = np.zeros((m, m), dtype=np.float64)
    pos = np.zeros((m, m), dtype=np.float64)
    cols = (np.arange(w) * N_LEVELS)[None, :]
    for a in range(0, n, ROW_BLOCK):
        b = min(a + ROW_BLOCK, n)
        oh = np.zeros((b - a, m), dtype=np.float32)
        np.put_along_axis(oh, D[a:b].astype(np.int64) + cols, 1.0, axis=1)
        tot += oh.T @ oh
        p = y[a:b] == 1
        if p.any():
            ohp = oh[p]
            pos += ohp.T @ ohp
    return tot, pos


def variance_terms(tot: np.ndarray, pos: np.ndarray, n_wires: int, p: float,
                   n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-wire ``V_u`` and the full pairwise ``V_pair`` matrix.

    Empty cells contribute nothing and are masked rather than divided by.
    """
    def v_of(n_c, k_c):
        num = (k_c - p * n_c) ** 2
        return np.where(n_c > 0, num / np.maximum(n_c, 1.0), 0.0).sum(axis=-1)

    m = n_wires * N_LEVELS
    d = np.arange(m)
    n_lvl = tot[d, d].reshape(n_wires, N_LEVELS)
    k_lvl = pos[d, d].reshape(n_wires, N_LEVELS)
    v_wire = v_of(n_lvl, k_lvl) / n_rows

    nb = tot.reshape(n_wires, N_LEVELS, n_wires, N_LEVELS).transpose(0, 2, 1, 3)
    kb = pos.reshape(n_wires, N_LEVELS, n_wires, N_LEVELS).transpose(0, 2, 1, 3)
    v_pair = v_of(nb.reshape(n_wires, n_wires, -1),
                  kb.reshape(n_wires, n_wires, -1)) / n_rows
    return v_wire, v_pair


def bank_variance(tables, v_pair: np.ndarray, v_wire: np.ndarray) -> dict:
    """Mean per-table interaction and pair variance of a bank, on one half.

    Evaluated on whichever half's counts were passed in. Running it on the fit
    half and again on the pick half asks whether a pairing chosen for its
    apparent interaction still has that interaction on rows it was not chosen
    from. Selection optimism -- picking the pairs whose fit-half interaction is
    largest partly because their noise is largest -- would show as a high fit
    value collapsing on the pick side, and it is the first thing to check when a
    bank with more measured interaction scores worse.
    """
    u = np.fromiter((t[0] for t in tables), dtype=np.int64, count=len(tables))
    v = np.fromiter((t[1] for t in tables), dtype=np.int64, count=len(tables))
    inter = v_pair[u, v] - v_wire[u] - v_wire[v]
    return {"mean_interaction": round(float(inter.mean()), 9),
            "mean_pair_variance": round(float(v_pair[u, v].mean()), 9)}


def greedy_rounds(weight: np.ndarray, rounds: int, maximise: bool
                  ) -> tuple[list[list[int]], list[dict]]:
    """``rounds`` edge-disjoint greedy matchings, best-weight edge first.

    Within a round each wire is used at most once, so a round is a matching and
    the bank keeps the property that no wire is over-represented. An odd wire
    count leaves one wire unpaired per round and it is dropped, which is what
    ``partition_tables`` already does.
    """
    w = np.asarray(weight, dtype=np.float64)
    n = w.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    flat = w[iu, ju]
    used = np.zeros(len(flat), dtype=bool)
    tables: list[list[int]] = []
    report: list[dict] = []
    for r in range(rounds):
        avail = np.flatnonzero(~used)
        vals = flat[avail]
        order = avail[np.argsort(-vals if maximise else vals, kind="stable")]
        free = np.ones(n, dtype=bool)
        picked: list[int] = []
        for e in order:
            a, b = int(iu[e]), int(ju[e])
            if free[a] and free[b]:
                free[a] = free[b] = False
                picked.append(int(e))
                if len(picked) == n // 2:
                    break
        used[picked] = True
        got = flat[picked]
        # The ideal for one round is the n//2 best available edges ignoring the
        # matching constraint, which no matching can beat. The ratio says how
        # much the greedy rule gave up, without needing blossom to find out.
        srt = np.sort(vals)
        ideal = srt[::-1][:n // 2] if maximise else srt[:n // 2]
        tables += [[int(iu[e]), int(ju[e])] for e in picked]
        report.append({
            "round": r + 1,
            "n_tables": len(picked),
            "weight_captured": round(float(got.sum()), 9),
            "unconstrained_ideal": round(float(ideal.sum()), 9),
            "fraction_of_ideal": round(float(got.sum() / ideal.sum()), 6)
            if ideal.sum() != 0 else None,
        })
    return tables, report


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

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    n_wires = int(W.shape[1])
    if n_wires not in frozen:
        raise SystemExit(f"the frozen artifact reports widths "
                         f"{sorted(frozen)}; the cache carries {n_wires} wires")
    if TABLE_WIDTH != 2:
        raise SystemExit(f"this tool selects pairs and the deployed width is "
                         f"{TABLE_WIDTH}; a matching needs width 2")

    wanted = ([s.strip() for s in a.arms.split(",") if s.strip()]
              if a.arms else list(ARMS))
    unknown = [w for w in wanted if w not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {list(ARMS)}")

    t0 = time.perf_counter()
    D = chain_digits(np.asarray(W, dtype=np.float64), n_res)
    print(f"banded {n_wires} wires in {time.perf_counter() - t0:.0f}s; "
          f"{n_splits} splits; arms: {wanted}", flush=True)

    control = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                              CONTROL_SEED)
    deployed = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                               PARTITION_SEED)
    row = np.repeat(np.arange(len(n_res)), n_res)
    deployed_set = {tuple(sorted(t)) for t in deployed}

    got: dict[str, list[float]] = {k: [] for k in wanted}
    occupancy: dict[str, dict] = {}
    matching: dict[str, list[dict]] = {}
    overlap: dict[str, list[int]] = {k: [] for k in wanted}
    survival: dict[str, dict] = {}

    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        Dfit, yfit = D[fit], y[fit]
        n_rows = int(Dfit.shape[0])
        p = float(yfit.mean())

        t1 = time.perf_counter()
        tot, pos = joint_counts(Dfit, yfit)
        v_wire, v_pair = variance_terms(tot, pos, n_wires, p, n_rows)
        inter = v_pair - v_wire[:, None] - v_wire[None, :]
        gram = time.perf_counter() - t1

        banks: dict[str, list[list[int]]] = {}
        for name in wanted:
            if name == "interaction":
                tb, rep = greedy_rounds(inter, PARTITION_ROUNDS, True)
            elif name == "pair variance":
                tb, rep = greedy_rounds(v_pair, PARTITION_ROUNDS, True)
            elif name == "anti-selected":
                tb, rep = greedy_rounds(inter, PARTITION_ROUNDS, False)
            else:
                tb, rep = control, []
            banks[name] = tb
            if s == 0:
                matching[name] = rep
            overlap[name].append(
                sum(1 for t in tb if tuple(sorted(t)) in deployed_set))

        if s == 0:
            # The same tables scored on the half they were not chosen from. One
            # split is enough for a diagnostic that asks about a mechanism
            # rather than about an ordering.
            tp, pp = joint_counts(D[pick], y[pick])
            vw_p, vp_p = variance_terms(tp, pp, n_wires,
                                        float(y[pick].mean()),
                                        int(pick.sum()))
            for name in list(wanted) + ["deployed"]:
                tb = deployed if name == "deployed" else banks[name]
                f = bank_variance(tb, v_pair, v_wire)
                q = bank_variance(tb, vp_p, vw_p)
                survival[name] = {
                    "on_the_fit_half_it_was_chosen_from": f,
                    "on_the_pick_half_it_was_not": q,
                    "interaction_surviving": round(
                        q["mean_interaction"] / f["mean_interaction"], 6)
                    if f["mean_interaction"] else None,
                }
            del tp, pp, vp_p

        print(f"  split {s + 1}/{n_splits}  gram {gram:.0f}s  "
              f"deployed(frozen) {frozen[n_wires][s]:.4f}", flush=True)
        for name in wanted:
            tb = banks[name]
            off = offsets_at(tb, N_LEVELS)
            t2 = time.perf_counter()
            frac, tcount = compile_at(Dfit, yfit, tb, off, N_LEVELS)
            mult = fanout_at(Dfit, yfit, tb, off, frac, N_LEVELS)
            sc = apply_gate(score_at(D[pick], tb, off, frac, mult, N_LEVELS),
                            ctr[pick], n_pick)
            got[name].append(float(per_unit_auc(sc, y[pick], n_pick)))
            if s == 0:
                occupancy[name] = cell_occupancy(tcount)
            print(f"      {name:16s} {got[name][-1]:.4f}  "
                  f"{got[name][-1] - frozen[n_wires][s]:+.4f}  "
                  f"{time.perf_counter() - t2:.0f}s", flush=True)
        del tot, pos, v_pair, inter

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

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "the deployed bank draws all 5,152 pairings from one seed, "
                    "and two frozen artifacts say the counting field's margin "
                    "lives in what paired wires say jointly. Does choosing the "
                    "pairings on the fit half beat drawing them",
        "criterion": {
            "definition": "V = sum_c (k_c - p n_c)^2 / (n_c N) over the cells "
                          "of a partition of the fit half, a rational function "
                          "of integer counts and the Gini reduction the "
                          "fan-out ranking already uses",
            "interaction": "V_pair - V_u - V_v, the second-order variance ANOVA "
                           "term; predicted by the argument because each wire "
                           "already appears in 16 tables so its marginal is not "
                           "scarce, interaction is",
            "pair variance": "V_pair alone; run because the prediction is an "
                             "argument and an argument measured against no "
                             "alternative is a correlate",
            "computed_by": "one Gram matrix of the level indicators per split, "
                           "which yields every pair's 4x4 contingency table at "
                           "once instead of 207,690 separate passes",
        },
        "controls": {
            "another seed": f"random pairings at seed {CONTROL_SEED}; if "
                            f"selection wins by about what a redraw wins, the "
                            f"effect is seed sensitivity",
            "anti-selected": "minimises the interaction criterion; if the "
                             "criterion means anything this must lose",
        },
        "held_fixed_across_arms": {
            "n_wires": n_wires,
            "n_levels": N_LEVELS,
            "quantisation": "within-chain quartiles, the deployed ladder",
            "table_width": TABLE_WIDTH,
            "partition_rounds": PARTITION_ROUNDS,
            "n_tables": len(deployed),
            "gate_radius": GATE_RADIUS,
            "gate_weight": GATE_WEIGHT,
            "what_changes": "only which wires are paired",
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "select_pairings_on": "the fit half only, recomputed per split",
            "compile_on": "the fit half only",
            "evaluate_on": "the pick half",
            "metric": "mean per-unit ROC-AUC",
            "baseline_was_not_recomputed": str(COUNTING.relative_to(ROOT)),
            "why_selection_is_not_a_leak": "the pairings are a compiled "
                                           "quantity of the fit half, like the "
                                           "cell counts, the integer fan-out "
                                           "and the residue propensity table "
                                           "already are. No pick-half row is "
                                           "read while choosing them",
        },
        "arms": {k: summarise(v) for k, v in got.items()},
        "deployed_arm_frozen": summarise(base),
        "minus_deployed": {k: compare(v) for k, v in got.items()},
        "pairings_shared_with_the_deployed_bank": {
            k: {"per_split": v, "of_n_tables": len(deployed)}
            for k, v in overlap.items()},
        "greedy_matching_on_split_1": matching,
        "cell_occupancy_on_split_1": occupancy,
        "does_the_chosen_interaction_survive_on_split_1": {
            "what_this_asks": "a pairing chosen because its fit-half "
                              "interaction is large may have been chosen partly "
                              "because its noise is large. The same tables are "
                              "re-measured on the pick half, which had no part "
                              "in choosing them. A selected bank whose "
                              "interaction collapses there, next to a random "
                              "bank whose does not, is selection optimism and "
                              "not a failure of the criterion to mean anything",
            "banks": survival,
        },
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
        print(f"  {k:16s} {doc['arms'][k]['mean']:.6f}  {c['mean']:+.6f}  on "
              f"{c['n_splits_positive']}/{c['n_splits']} splits")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
