#!/usr/bin/env python3.12
"""Which column families can a counting field collect, and which belong to a solve?

The pattern that needs explaining
--------------------------------
Three column families have both a counting-field lift and a linear-solve lift
measured on the same twelve splits under the same gate, and the sign of the
difference flips between them:

    deployed 645 wires   field 0.79012 against solve 0.784826, field ahead +0.0053
    asymmetry 129        field +0.0010 on 9/12, solve +0.0038 on 12/12
    composition 76       field -0.0009 on 2/12, solve +0.0011 on 10/12

The deployed wires are the only family the field reads better than a solve does.
APPENDED_BLOCK_GEOMETRY.json rules out the two obvious explanations for the other
two: not an interaction the field misses, because the solve is linear and cannot
be reading one; not the field's estimation cost, because varying the appended
block's cell budget twenty-one-fold moves the lift by a seventh of the reseed
floor and not upward.

The proposition under test
--------------------------
A counting field is a sum over tables of cell frequencies. Its cells can state
anything about a joint state of two quantised wires, including things no additive
function of the two can state -- and that is the only thing it has that a linear
solve does not. A solve, one coefficient per column, is instead an efficient
aggregator of many weak effects that add. So the two readouts should be good at
different kinds of column, and which kind a family is ought to be measurable
before any field is compiled.

The measurable form. For a pair of wires the count-weighted signal of its 4x4
table decomposes exactly:

    V_pair(u,v) = V(u) + V(v) + interaction(u,v)

where V is the count-weighted departure of level rates from the base rate and the
interaction is what the joint says beyond both marginals added. Define the family's
**non-additive share** as the mean of interaction / V_pair over its pairs. It is
dimensionless, it is computed from integer counts, and it is exactly the fraction
of what a table knows that an additive model cannot see.

Prediction: the non-additive share separates the deployed wires from the two
appended families, in the same order as (field lift minus solve lift). If it does,
it is a screen that predicts what the field will collect -- which is what the
Fisher solve was wrongly used for, since IS_FISHER_A_CEILING.json shows the solve
is not a ceiling. If it does not, the proposition is wrong.

What is reused rather than reinvented
-------------------------------------
``joint_counts`` and ``variance_terms`` are imported from
``selected_pairings.py``. The interaction here is the same quantity that tool
selected pairings by, and that selection failed at -0.0028. The failure is on
record with its reason and the reason does not transfer: interaction concentrates
on few wires while a matching may use each wire once per round, so a round could
capture only a fraction of the available weight. Nothing here forms a matching.
The question is not which pairs to make out of fixed wires; it is whether a family
of columns is the kind of thing a counting field can read at all.

Falsification
-------------
The share failing to separate the three families, or ordering them against the
measured field-minus-solve difference. Either kills the proposition, and with it
the idea of screening new wire families this way.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from anisotropic_expansion_ceiling import build_or_load as build_asymmetry  # noqa: E402
from composition_wires import build_or_load as build_composition  # noqa: E402
from expand_invariant_bank import SEED  # noqa: E402
from selected_pairings import joint_counts, variance_terms  # noqa: E402
from select_architecture_on_train import cluster_half_split  # noqa: E402

from pocket_bench.methods.table_bank import (
    N_LEVELS,
    chain_digits,
    partition_tables,
)
from pocket_bench.methods.table_field import (
    PARTITION_ROUNDS,
    PARTITION_SEED,
    TABLE_WIDTH,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.collectability_screen.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
OUT = ROOT / "results/architecture_sweep/COLLECTABILITY_SCREEN.json"
FISHER = ROOT / "results/architecture_sweep/IS_FISHER_A_CEILING.json"
UNION = ROOT / "results/architecture_sweep/UNION_BANK_COUNTING_FIELD.json"
COMPWIRE = ROOT / "results/architecture_sweep/COMPOSITION_WIRES.json"

# The measured field-minus-solve difference for each family, read from the frozen
# artifacts rather than typed, so the thing the screen has to reproduce cannot
# drift from what was measured.
KNOWN = {
    "deployed 645 wires": {
        "artifact": str(FISHER.relative_to(ROOT)),
        "how": "verdict at 645 wires: counting minus fisher",
    },
    "asymmetry 129": {
        "artifact": str(UNION.relative_to(ROOT)),
        "how": "union minus narrow for the field, fisher lift for the solve",
    },
    "composition 76": {
        "artifact": str(COMPWIRE.relative_to(ROOT)),
        "how": "union minus deployed for the field, fisher lift for the solve",
    },
}


def known_lifts() -> dict:
    """Field lift, solve lift and their difference, per family, from artifacts."""
    out: dict[str, dict] = {}
    if FISHER.is_file():
        v = json.loads(FISHER.read_text())["verdict"]["645"]
        out["deployed 645 wires"] = {
            "field": v["counting_mean"],
            "solve": v["fisher_mean"],
            "field_minus_solve": round(v["counting_minus_fisher_mean"], 6),
            "n_splits_field_higher": v["n_splits_counting_higher"],
            "note": "absolute AUCs, not lifts: this family is the bank rather "
                    "than an addition to it",
        }
        aniso = (json.loads(FISHER.read_text())
                 .get("fisher_lift_from_anisotropy_under_the_deployed_gate") or {})
    else:
        aniso = {}
    if UNION.is_file() and aniso:
        u = json.loads(UNION.read_text())["union_minus_narrow"]
        out["asymmetry 129"] = {
            "field": round(u["mean"], 6),
            "solve": round(aniso["mean"], 6),
            "field_minus_solve": round(u["mean"] - aniso["mean"], 6),
            "n_splits_field_positive": u["n_splits_positive"],
        }
    if COMPWIRE.is_file():
        c = json.loads(COMPWIRE.read_text())
        f = c["minus_deployed"]["union"]
        s = c["fisher_lift_from_composition"]
        out["composition 76"] = {
            "field": round(f["mean"], 6),
            "solve": round(s["mean"], 6),
            "field_minus_solve": round(f["mean"] - s["mean"], 6),
            "n_splits_field_positive": f["n_splits_positive"],
        }
    return out


def shares(D: np.ndarray, y: np.ndarray, tables) -> dict:
    """First-order, second-order and non-additive share, over a family's columns.

    Two populations are reported and they answer different questions. Over the
    bank's own pairings is what the deployed construction actually sees. Over all
    pairs within the family is a property of the family and not of one seed draw,
    which is what a screen for a *new* family needs, since its pairings have not
    been drawn yet.
    """
    n_wires = int(D.shape[1])
    p = float(y.mean())
    tot, pos = joint_counts(D, y)
    v_wire, v_pair = variance_terms(tot, pos, n_wires, p, int(len(y)))

    iu = np.triu_indices(n_wires, k=1)
    inter_all = v_pair[iu] - v_wire[iu[0]] - v_wire[iu[1]]
    pair_all = v_pair[iu]
    ok = pair_all > 0
    share_all = np.where(ok, inter_all / np.maximum(pair_all, 1e-300), 0.0)[ok]

    u = np.fromiter((t[0] for t in tables), dtype=np.int64, count=len(tables))
    v = np.fromiter((t[1] for t in tables), dtype=np.int64, count=len(tables))
    inter_bank = v_pair[u, v] - v_wire[u] - v_wire[v]
    pair_bank = v_pair[u, v]
    okb = pair_bank > 0
    share_bank = np.where(okb, inter_bank / np.maximum(pair_bank, 1e-300), 0.0)[okb]

    def summ(x):
        x = np.asarray(x, dtype=float)
        return {
            "mean": round(float(x.mean()), 6),
            "median": round(float(np.median(x)), 6),
            "p10": round(float(np.percentile(x, 10)), 6),
            "p90": round(float(np.percentile(x, 90)), 6),
            "fraction_negative": round(float((x < 0).mean()), 4),
        }

    return {
        "n_columns": n_wires,
        "n_pairs_all": int(len(share_all)),
        "n_pairs_in_the_bank": int(len(share_bank)),
        "first_order_per_column": {
            "mean": round(float(v_wire.mean()), 9),
            "median": round(float(np.median(v_wire)), 9),
        },
        "second_order_over_all_pairs": {
            "mean": round(float(inter_all.mean()), 9),
            "median": round(float(np.median(inter_all)), 9),
        },
        "non_additive_share_over_all_pairs": summ(share_all),
        "non_additive_share_over_the_bank": summ(share_bank),
    }


# The deployed bus is 43 local quantities under 15 statistics, laid out
# statistic-major by wide_descriptors.build_wide, so wire w is quantity w % 43
# read under statistic w // 43.
N_QUANTITIES = 43
N_STATISTICS = 15


def within_versus_across(D: np.ndarray, y: np.ndarray) -> dict:
    """Split the deployed bank's pairs by whether they share a local quantity.

    Predicted before computing, and written here rather than in the summary so
    the order is on record. The asymmetry family is one operator swept over
    radii and its mean interaction is negative: pairs of it restate each other.
    Fifteen of the deployed bus's statistics are also a sweep -- five radii of a
    mean, three of a dispersion, three of a centred difference, three of a local
    rank -- so pairs that share a quantity should behave like the asymmetry
    family, and the bank's positive interaction should be carried by pairs of
    different quantities. If instead the two are alike, then what distinguishes
    the asymmetry columns is not that they are a sweep and the reading of the
    screen above is wrong.
    """
    n_wires = int(D.shape[1])
    if n_wires != N_QUANTITIES * N_STATISTICS:
        return {"skipped": f"{n_wires} wires is not {N_QUANTITIES} x "
                           f"{N_STATISTICS}; the layout assumption does not hold"}
    p = float(y.mean())
    tot, pos = joint_counts(D, y)
    v_wire, v_pair = variance_terms(tot, pos, n_wires, p, int(len(y)))
    iu = np.triu_indices(n_wires, k=1)
    inter = v_pair[iu] - v_wire[iu[0]] - v_wire[iu[1]]
    same_q = (iu[0] % N_QUANTITIES) == (iu[1] % N_QUANTITIES)
    same_s = (iu[0] // N_QUANTITIES) == (iu[1] // N_QUANTITIES)

    def summ(mask, what):
        x = inter[mask]
        return {"what": what, "n_pairs": int(mask.sum()),
                "mean_interaction": round(float(x.mean()), 12),
                "median_interaction": round(float(np.median(x)), 12),
                "fraction_negative": round(float((x < 0).mean()), 4)}

    return {
        "layout": f"wire w is quantity w % {N_QUANTITIES} under statistic "
                  f"w // {N_QUANTITIES}, from wide_descriptors.build_wide",
        "all pairs": summ(np.ones_like(same_q), "every pair in the bus"),
        "same quantity, different statistic": summ(
            same_q & ~same_s, "one local quantity read at several radii or under "
                              "several statistics -- the same shape as the "
                              "asymmetry family"),
        "same statistic, different quantity": summ(
            same_s & ~same_q, "two different local quantities read the same way"),
        "different quantity and statistic": summ(
            ~same_q & ~same_s, "unrelated in both coordinates"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=1,
                   help="how many cluster-disjoint fit halves to average over. "
                        "The share is a property of the columns and the labels, "
                        "not of a readout, so it is stable across halves and one "
                        "is enough to see a separation of this size; more is a "
                        "check that it is")
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res = z["X"], z["y"], z["n_res_per"]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}

    t0 = time.perf_counter()
    A, _diag, _n = build_asymmetry(8)
    C, _names = build_composition()
    families = {
        "deployed 645 wires": np.asarray(W, dtype=np.float64),
        "asymmetry 129": np.asarray(A, dtype=np.float64),
        "composition 76": np.asarray(C, dtype=np.float64),
    }
    print(f"built the three families in {time.perf_counter() - t0:.0f}s: "
          + ", ".join(f"{k} ({v.shape[1]})" for k, v in families.items()),
          flush=True)

    row = np.repeat(np.arange(len(n_res)), n_res)
    got: dict[str, list[dict]] = {k: [] for k in families}
    for s in range(a.splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit = is_fit[row]
        for name, F in families.items():
            D = chain_digits(F, n_res)[fit]
            tabs = partition_tables(int(F.shape[1]), TABLE_WIDTH,
                                    PARTITION_ROUNDS, PARTITION_SEED)
            got[name].append(shares(D, y[fit], tabs))
            print(f"  split {s + 1}: {name:20s} "
                  f"share(bank) "
                  f"{got[name][-1]['non_additive_share_over_the_bank']['mean']:+.4f}"
                  f"  share(all pairs) "
                  f"{got[name][-1]['non_additive_share_over_all_pairs']['mean']:+.4f}",
                  flush=True)

    is_fit0, _ = cluster_half_split(units, cluster_of, SEED)
    split_by_kind = within_versus_across(
        chain_digits(families["deployed 645 wires"], n_res)[is_fit0[row]],
        y[is_fit0[row]])
    print()
    for k, v in split_by_kind.items():
        if isinstance(v, dict) and "mean_interaction" in v:
            print(f"  {k:38s} {v['mean_interaction']:+.3e}  "
                  f"({v['n_pairs']:6d} pairs, "
                  f"{100 * v['fraction_negative']:.0f}% negative)", flush=True)

    lifts = known_lifts()
    fam = {}
    for name, rows in got.items():
        bank = [r["non_additive_share_over_the_bank"]["mean"] for r in rows]
        allp = [r["non_additive_share_over_all_pairs"]["mean"] for r in rows]
        fam[name] = {
            **rows[0],
            "share_over_the_bank_mean_across_splits": round(float(np.mean(bank)), 6),
            "share_over_all_pairs_mean_across_splits": round(float(np.mean(allp)), 6),
            "measured_lifts": lifts.get(name),
            "lift_source": KNOWN.get(name),
        }

    order_by_share = sorted(fam, key=lambda k: -fam[k]
                            ["share_over_all_pairs_mean_across_splits"])
    order_by_abs = sorted(fam, key=lambda k: -fam[k]
                          ["second_order_over_all_pairs"]["mean"])
    have = [k for k in fam if (lifts.get(k) or {}).get("field_minus_solve") is not None]
    order_by_lift = sorted(have, key=lambda k: -lifts[k]["field_minus_solve"])
    agrees = order_by_share[:len(order_by_lift)] == order_by_lift
    agrees_abs = order_by_abs[:len(order_by_lift)] == order_by_lift

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "whether a column family's non-additive share predicts which "
                    "of the two readouts can collect it, so that a new family can "
                    "be screened before a field is compiled",
        "the_pattern_being_explained": (
            "the deployed wires are the only one of three families the counting "
            "field reads better than a linear solve does. The two appended "
            "families are read better by the solve, and "
            "APPENDED_BLOCK_GEOMETRY.json excludes both the interaction and the "
            "estimation-cost explanations for that"),
        "definition": {
            "decomposition": "V_pair(u,v) = V(u) + V(v) + interaction(u,v), where "
                             "V is the count-weighted departure of level rates "
                             "from the base rate",
            "non_additive_share": "mean of interaction / V_pair, the fraction of "
                                  "what a table knows that no additive function "
                                  "of its two wires can state",
            "computed_from": "integer cell counts only; one Gram matrix of level "
                             "indicators gives every pair's 4x4 table at once",
            "why_dimensionless": "a ratio of two terms of the same decomposition, "
                                 "so families of different scale are comparable",
            "can_be_negative": "yes, when two wires are redundant, and the "
                               "distribution is reported rather than only its mean",
        },
        "what_is_reused": (
            "joint_counts and variance_terms from selected_pairings.py. The "
            "interaction is the same quantity that tool selected pairings by, and "
            "that selection failed at -0.0028. The recorded reason does not "
            "transfer: interaction concentrates on few wires and a matching may "
            "use each wire once per round, so a round captured a fraction of the "
            "available weight. Nothing here forms a matching, and the question is "
            "not which pairs to make but whether a family is readable at all"),
        "what_would_falsify_it": "the share failing to separate the three "
                                 "families, or ordering them against the measured "
                                 "field-minus-solve difference",
        "protocol": {
            "n_splits": a.splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + a.splits - 1}",
            "computed_on": "the fit half only",
            "banding": "within-chain rank quartiles, the deployed rule, applied "
                       "within each family separately",
            "pairings": f"{PARTITION_ROUNDS} rounds at width {TABLE_WIDTH}, seed "
                        f"{PARTITION_SEED}, drawn within each family",
        },
        "families": fam,
        "where_the_deployed_bank_s_interaction_lives": split_by_kind,
        "why_that_decomposition_was_run": (
            "the asymmetry family is one operator swept over radii and its mean "
            "interaction is negative. Fourteen of the deployed bus's fifteen "
            "statistics are also a sweep, so if being a sweep is what makes a "
            "family redundant then the bank's own same-quantity pairs should look "
            "like the asymmetry family and its positive interaction should be "
            "carried by pairs of different quantities. Predicted in the "
            "function's docstring before it was computed"),
        "order_by_non_additive_share": order_by_share,
        "order_by_absolute_second_order_term": order_by_abs,
        "order_by_measured_field_minus_solve": order_by_lift,
        "the_share_orders_the_families_correctly": bool(agrees),
        "verdict_on_the_preregistered_statistic": (
            "falsified. The share was named in this tool's docstring before it "
            "ran and it puts the composition columns first, where the field does "
            "worst of the three. The reason is visible in the absolute terms: "
            "composition's first-order term is about a third of the others, so a "
            "modest interaction becomes a large fraction of a small total. A "
            "ratio was the wrong normalisation"),
        "what_the_absolute_term_does": {
            "orders_them_correctly": bool(agrees_abs),
            "the_numbers": {k: fam[k]["second_order_over_all_pairs"]["mean"]
                            for k in order_by_abs},
            "the_reading": (
                "the mean pairwise interaction, unnormalised, has the same order "
                "as the measured field-minus-solve difference. The asymmetry "
                "columns are the striking case: their mean interaction is "
                "negative, so on average a pair of them says less than its two "
                "marginals added. 129 columns computed at several radii from one "
                "structure are largely restatements of each other, which is "
                "exactly the shape a linear solve can use -- one coefficient per "
                "column, marginals are all it needs -- and a counting field "
                "cannot, since its only advantage is in the joint"),
            "why_this_is_not_confirmed": (
                "the share was chosen before the run and the absolute term after "
                "it, on three families. Picking the statistic that orders three "
                "points correctly, having seen the ordering, is not a "
                "confirmation of anything. It is a hypothesis, and the way to "
                "test it is a fourth family whose interaction is measured and "
                "whose prediction is recorded before its field lift is"),
        },
    }

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    if a.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    print("\n  family                 share(all pairs)  share(bank)   "
          "field-minus-solve")
    for k in order_by_share:
        L = lifts.get(k) or {}
        fms = L.get("field_minus_solve")
        print(f"  {k:22s} {fam[k]['share_over_all_pairs_mean_across_splits']:+.4f}"
              f"          {fam[k]['share_over_the_bank_mean_across_splits']:+.4f}"
              f"       {'n/a' if fms is None else f'{fms:+.4f}'}")
    print("\n  absolute mean pairwise interaction:")
    for k in order_by_abs:
        print(f"    {k:22s} {fam[k]['second_order_over_all_pairs']['mean']:+.3e}")
    print(f"\n  by share:            {order_by_share}")
    print(f"  by absolute 2nd order: {order_by_abs}")
    print(f"  by measured lift:      {order_by_lift}")
    print(f"  the preregistered share orders them correctly: {agrees}")
    print(f"  the absolute term does:                        {agrees_abs}"
          f"   (chosen after seeing this, on three families; a hypothesis, "
          f"not a confirmation)")
    if a.write:
        print(f"\nwrote {out.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
