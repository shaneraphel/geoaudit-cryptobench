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
from selected_pairings import ROW_BLOCK, joint_counts, variance_terms  # noqa: E402
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
APPENDED = ROOT / "results/architecture_sweep/APPENDED_FAMILY_LIFT.json"
GRAPHINV = ROOT / "data/cryptobench_apo/_graphinv_cache_train.npz"

# The range this screen committed to for the prospective family, before that
# family's field lift was measured. Kept as a constant so the verdict below is a
# comparison against a recorded number and not a sentence written afterwards.
PREDICTED = {
    "family": "graph invariants 225",
    "field_lift_low": -0.002,
    "field_lift_high": +0.001,
    "recorded_in": "docs/AGENT_MEMORY.md and this tool, before "
                   "tools/appended_family_lift.py was run",
    "falsified_if": "a field lift outside that range, in either direction",
}

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
    # The prospective family. Absent until its lift is measured, which is what
    # keeps it out of the agreement check while it is still a prediction; present
    # afterwards, because a prospective point that is measured and then withheld
    # from the check is the check being chosen after the answer.
    if APPENDED.is_file():
        g = json.loads(APPENDED.read_text())
        f = g["minus_narrow"]["union"]
        s = g["solve_lift_from_the_family"]
        out[g["family"]] = {
            "field": round(f["mean"], 6),
            "solve": round(s["mean"], 6),
            "field_minus_solve": round(f["mean"] - s["mean"], 6),
            "n_splits_field_positive": f["n_splits_positive"],
            "measured_after_the_screen": True,
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


def _level_counts(D: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-wire (total, positive) counts at each level, without a Gram matrix."""
    w = int(D.shape[1])
    p = y == 1
    tot = np.empty((w, N_LEVELS), dtype=np.float64)
    pos = np.empty((w, N_LEVELS), dtype=np.float64)
    for j in range(w):
        col = D[:, j].astype(np.int64)
        tot[j] = np.bincount(col, minlength=N_LEVELS)[:N_LEVELS]
        pos[j] = np.bincount(col[p], minlength=N_LEVELS)[:N_LEVELS]
    return tot, pos


def _straddling_gram(Dold: np.ndarray, Dnew: np.ndarray, y: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Only the off-diagonal block of the joint Gram, never the whole thing.

    ``joint_counts`` on the concatenated columns would answer this, and the first
    version of this function did exactly that. It allocates a Gram over
    ``(645 + n) * 4`` indicators and then two dense ``(645 + n)^2 x 16`` tensors
    inside ``variance_terms``, of which the straddling block is under a tenth --
    about 1.1 GB of transient for 18 MB of answer, which is what killed the run
    on a machine with the swap already full. The cross block is a single
    rectangular product and is formed directly.
    """
    n = int(Dold.shape[0])
    wo, wn = int(Dold.shape[1]), int(Dnew.shape[1])
    mo, mn = wo * N_LEVELS, wn * N_LEVELS
    tot = np.zeros((mo, mn), dtype=np.float64)
    pos = np.zeros((mo, mn), dtype=np.float64)
    colo = (np.arange(wo) * N_LEVELS)[None, :]
    coln = (np.arange(wn) * N_LEVELS)[None, :]
    for a in range(0, n, ROW_BLOCK):
        b = min(a + ROW_BLOCK, n)
        oh_o = np.zeros((b - a, mo), dtype=np.float32)
        np.put_along_axis(oh_o, Dold[a:b].astype(np.int64) + colo, 1.0, axis=1)
        oh_n = np.zeros((b - a, mn), dtype=np.float32)
        np.put_along_axis(oh_n, Dnew[a:b].astype(np.int64) + coln, 1.0, axis=1)
        tot += oh_o.T @ oh_n
        m = y[a:b] == 1
        if m.any():
            pos += oh_o[m].T @ oh_n[m]
    return tot, pos


def _check_native_dtype_digits(F: np.ndarray, n_res_per, n_chains: int = 40
                               ) -> dict:
    """Require the native columns to digitise exactly as their float64 upcast.

    The wide cache is float32 and upcasting it costs 1,156 MB against the 578 MB
    the columns already occupy, which is the single largest allocation this tool
    makes and the one that put it over the edge on a machine with the swap full.
    Skipping the upcast is safe by argument -- float32 to float64 is exact, so
    both the stable argsort and the equality test that groups ties see the same
    order -- and an argument is not a measurement. Digitise a few chains both
    ways and require the int8 arrays to be equal, on the family that is actually
    being read rather than on a synthetic one.
    """
    k = min(int(n_chains), len(n_res_per))
    n = int(np.sum(n_res_per[:k]))
    sl, heads = F[:n], n_res_per[:k]
    same = np.array_equal(chain_digits(sl, heads),
                          chain_digits(np.asarray(sl, dtype=np.float64), heads))
    if not same:
        raise SystemExit(
            f"digitising {F.dtype} columns does not reproduce the float64 "
            f"upcast on the first {k} chains; refusing to trade a number for "
            f"memory")
    return {
        "checked_on": f"the first {k} chains, {n} residues, "
                      f"{int(F.shape[1])} columns",
        "native_dtype": str(F.dtype),
        "identical": bool(same),
        "why": "the upcast is the largest allocation the tool makes; skipping "
               "it must not move a digit",
    }


def _check_straddling_against_the_canonical_path(Dold: np.ndarray,
                                                 Dnew: np.ndarray,
                                                 y: np.ndarray) -> dict:
    """Require the lean cross path to return what ``joint_counts`` returns.

    The habit this repository has paid for twice: a tool that reimplements a
    piece of the pipeline reported a live axis at -0.0096 on 0 of 12 purely
    because its local copy of the solve did not standardise. ``_straddling_gram``
    and ``_level_counts`` are a reimplementation of the same decomposition, so
    the concatenated path is run once on a small column subset and the two are
    required to agree to floating-point noise. Small because the concatenated
    path is the one that cannot be afforded at full width -- which is the whole
    reason the lean one exists.
    """
    o = Dold[:, :24]
    nw = Dnew[:, :12]
    n_o, n_n = int(o.shape[1]), int(nw.shape[1])
    p = float(y.mean())
    tot, pos = joint_counts(np.concatenate([o, nw], axis=1), y)
    v_wire, v_pair = variance_terms(tot, pos, n_o + n_n, p, int(len(y)))
    want = v_pair[:n_o, n_o:] - v_wire[:n_o, None] - v_wire[None, n_o:]

    def v_of(n_c, k_c):
        num = (k_c - p * n_c) ** 2
        return np.where(n_c > 0, num / np.maximum(n_c, 1.0), 0.0).sum(axis=-1)

    to, po = _level_counts(o, y)
    tn, pn = _level_counts(nw, y)
    ct, cp = _straddling_gram(o, nw, y)
    nb = ct.reshape(n_o, N_LEVELS, n_n, N_LEVELS).transpose(0, 2, 1, 3)
    kb = cp.reshape(n_o, N_LEVELS, n_n, N_LEVELS).transpose(0, 2, 1, 3)
    got = (v_of(nb.reshape(n_o, n_n, -1), kb.reshape(n_o, n_n, -1)) / len(y)
           - v_of(to, po)[:, None] / len(y) - v_of(tn, pn)[None, :] / len(y))

    err = float(np.abs(got - want).max())
    scale = float(np.abs(want).max())
    ok = err <= 1e-12 + 1e-9 * scale
    if not ok:
        raise SystemExit(
            f"the lean cross path disagrees with joint_counts by {err:.3e} "
            f"against a scale of {scale:.3e}; refusing to report a screen from "
            f"a reimplementation that does not reproduce the canonical one")
    return {
        "checked_on": f"{n_o} deployed wires against {n_n} new columns",
        "max_absolute_disagreement": float(err),
        "scale_of_the_quantity": scale,
        "agrees": bool(ok),
        "why": "the lean path forms only the straddling Gram block; the "
               "canonical path forms the whole thing and is unaffordable at "
               "full width. They must agree where both can be run",
    }


def cross_interaction(Dold: np.ndarray, Dnew: np.ndarray, y: np.ndarray) -> dict:
    """Mean interaction of pairs that straddle the deployed bus and a new family.

    The term the first version of this screen left out, and the one a family is
    actually added under. A family's internal interaction says what its own tables
    could know; it says nothing about whether the deployed bank already knows it.
    Two families with the same internal interaction can differ entirely in how much
    of it is new, and the union attachment does not even form straddling tables --
    it puts new tables over new columns alone -- so a family redundant with the bus
    contributes tables whose content is already present, which the fan-out then has
    to decorrelate away.

    The decomposition is the same one the rest of this tool uses, read on the
    straddling pairs only: ``interaction(u,v) = V_pair(u,v) - V(u) - V(v)`` for
    ``u`` a deployed wire and ``v`` a column of the new family.
    """
    n_old, n_new = int(Dold.shape[1]), int(Dnew.shape[1])
    n_rows = int(len(y))
    p = float(y.mean())

    def v_of(n_c, k_c):
        num = (k_c - p * n_c) ** 2
        return np.where(n_c > 0, num / np.maximum(n_c, 1.0), 0.0).sum(axis=-1)

    to, po = _level_counts(Dold, y)
    tn, pn = _level_counts(Dnew, y)
    v_old = v_of(to, po) / n_rows
    v_new = v_of(tn, pn) / n_rows

    tot, pos = _straddling_gram(Dold, Dnew, y)
    nb = tot.reshape(n_old, N_LEVELS, n_new, N_LEVELS).transpose(0, 2, 1, 3)
    kb = pos.reshape(n_old, N_LEVELS, n_new, N_LEVELS).transpose(0, 2, 1, 3)
    v_pair = v_of(nb.reshape(n_old, n_new, -1),
                  kb.reshape(n_old, n_new, -1)) / n_rows

    inter = v_pair - v_old[:, None] - v_new[None, :]
    return {
        "n_straddling_pairs": int(inter.size),
        "mean_interaction": round(float(inter.mean()), 12),
        "median_interaction": round(float(np.median(inter)), 12),
        "fraction_negative": round(float((inter < 0).mean()), 4),
        "mean_first_order_of_the_new_columns": round(float(v_new.mean()), 12),
        "what_it_asks": "whether what the family knows jointly is already known by "
                        "the deployed bus, which its own internal interaction "
                        "cannot say",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=1,
                   help="how many cluster-disjoint fit halves to average over. "
                        "The share is a property of the columns and the labels, "
                        "not of a readout, so it is stable across halves and one "
                        "is enough to see a separation of this size; more is a "
                        "check that it is")
    ap.add_argument("--cross", action="store_true",
                    help="also measure each family's interaction with the deployed "
                         "bus, which its internal interaction cannot say anything "
                         "about. Costs one Gram matrix over 645 + n columns")
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    z = np.load(WIDE, allow_pickle=False)
    y, n_res = z["y"], z["n_res_per"]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}

    # One family is built, digitised, and released before the next is built.
    #
    # The obvious form of this held all five as float64 at once -- about 2 GB for
    # 234,838 rows -- and was killed twice by the kernel, silently and with no
    # traceback, on a machine whose swap was already full. Nothing downstream
    # reads the float columns: ``chain_digits`` ranks within each chain over all
    # of that chain's residues, which does not depend on the split, so the int8
    # digits are the only thing any later stage needs and they are a tenth of the
    # size. Building lazily bounds the peak at one family rather than five, and
    # changes no number, because the digits of a family do not depend on which
    # other families exist.
    builders: dict[str, object] = {
        "deployed 645 wires": lambda: z["X"],
        "asymmetry 129": lambda: build_asymmetry(8)[0],
        "composition 76": lambda: build_composition()[0],
    }
    # The prospective test. When this tool was first written the family had no
    # measured field lift, so it appeared with the screen's value and no lift
    # beside it. APPENDED_FAMILY_LIFT.json has since measured it, and
    # ``known_lifts`` now picks that up, which is what lets it enter the
    # agreement check it was built to test.
    if GRAPHINV.is_file():
        from graph_invariant_wires import (
            build_or_load as build_graphinv,
            build_wide_or_load as build_graphinv_wide,
        )
        builders["graph invariants 15"] = lambda: build_graphinv()[0]
        # The 225-wire expansion is what a union attachment would actually add;
        # fifteen columns make about 112 tables against the asymmetry family's
        # 1,024, too small a block for a null on it to mean anything. Both are
        # screened so the expansion's own cost is visible.
        builders["graph invariants 225"] = lambda: build_graphinv_wide()[0]

    t0 = time.perf_counter()
    row = np.repeat(np.arange(len(n_res)), n_res)
    full: dict[str, np.ndarray] = {}
    widths: dict[str, int] = {}
    dtype_check = None
    for name, build in builders.items():
        F = np.asarray(build())
        widths[name] = int(F.shape[1])
        if dtype_check is None and F.dtype != np.float64:
            dtype_check = _check_native_dtype_digits(F, n_res)
            print(f"  digits from {F.dtype} match the float64 upcast on "
                  f"{dtype_check['checked_on']}: {dtype_check['identical']}",
                  flush=True)
        full[name] = chain_digits(F, n_res)
        del F
    z.close()
    print(f"built and digitised {len(full)} families in "
          f"{time.perf_counter() - t0:.0f}s: "
          + ", ".join(f"{k} ({w})" for k, w in widths.items())
          + f"; {sum(d.nbytes for d in full.values()) / 2**20:.0f} MB of int8 "
            f"held, the float columns released as each was digitised",
          flush=True)

    got: dict[str, list[dict]] = {k: [] for k in full}
    for s in range(a.splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit = is_fit[row]
        for name, D_all in full.items():
            tabs = partition_tables(widths[name], TABLE_WIDTH,
                                    PARTITION_ROUNDS, PARTITION_SEED)
            got[name].append(shares(D_all[fit], y[fit], tabs))
            print(f"  split {s + 1}: {name:20s} "
                  f"share(bank) "
                  f"{got[name][-1]['non_additive_share_over_the_bank']['mean']:+.4f}"
                  f"  share(all pairs) "
                  f"{got[name][-1]['non_additive_share_over_all_pairs']['mean']:+.4f}",
                  flush=True)

    is_fit0, _ = cluster_half_split(units, cluster_of, SEED)
    fit0 = is_fit0[row]
    split_by_kind = within_versus_across(full["deployed 645 wires"][fit0],
                                         y[fit0])
    print()
    for k, v in split_by_kind.items():
        if isinstance(v, dict) and "mean_interaction" in v:
            print(f"  {k:38s} {v['mean_interaction']:+.3e}  "
                  f"({v['n_pairs']:6d} pairs, "
                  f"{100 * v['fraction_negative']:.0f}% negative)", flush=True)

    cross = {}
    if a.cross:
        yf = y[fit0]
        Dold = full["deployed 645 wires"][fit0]
        checked = None
        for name in list(full):
            if name == "deployed 645 wires":
                continue
            if checked is None:
                checked = _check_straddling_against_the_canonical_path(
                    Dold, Dnew, yf)
                print(f"  lean cross path reproduces joint_counts to "
                      f"{checked['max_absolute_disagreement']:.2e}", flush=True)
            cross[name] = cross_interaction(Dold, Dnew, yf)
            print(f"  cross with the deployed bus: {name:22s} "
                  f"{cross[name]['mean_interaction']:+.3e}  "
                  f"({100 * cross[name]['fraction_negative']:.0f}% negative)",
                  flush=True)
        if checked is not None:
            cross["_reimplementation_check"] = checked
        if dtype_check is not None:
            cross["_native_dtype_check"] = dtype_check

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
    # Only families with a measured lift can order anything. Two orderings are
    # reported and they are not interchangeable. The three the statistic was
    # chosen on can only ever agree with it, since it was picked for that. The
    # one that tests anything is the ordering with the prospective family in it,
    # whose lift was measured after the statistic was fixed.
    have = [k for k in fam if (lifts.get(k) or {}).get("field_minus_solve") is not None]
    fitted_on = [k for k in have
                 if not (lifts[k] or {}).get("measured_after_the_screen")]
    order_by_share = [k for k in order_by_share if k in have]
    order_by_abs = [k for k in order_by_abs if k in have]
    order_by_lift = sorted(have, key=lambda k: -lifts[k]["field_minus_solve"])

    def _agree(order: list[str], among: list[str]) -> bool:
        a_ = [k for k in order if k in among]
        b_ = [k for k in order_by_lift if k in among]
        return a_ == b_

    agrees = _agree(order_by_share, fitted_on)
    agrees_abs = _agree(order_by_abs, fitted_on)
    agrees_prospective = _agree(order_by_abs, have)
    agrees_share_prospective = _agree(order_by_share, have)

    pname = PREDICTED["family"]
    verdict = None
    if pname in fam and pname in lifts:
        measured = lifts[pname]["field"]
        inside = PREDICTED["field_lift_low"] <= measured <= PREDICTED["field_lift_high"]
        # The field lift alone, over the three families that are additions to the
        # bank rather than the bank itself. The screen claims to order this too,
        # and on three points a reversal is not conclusive -- one ordering in six
        # -- so the count is reported rather than a word.
        added = [k for k in have if k != "deployed 645 wires"]
        by_screen = [k for k in order_by_abs if k in added]
        by_field = sorted(added, key=lambda k: -lifts[k]["field"])
        verdict = {
            "predicted": PREDICTED,
            "measured_field_lift": measured,
            "measured_solve_lift": lifts[pname]["solve"],
            "measured_field_minus_solve": lifts[pname]["field_minus_solve"],
            "n_splits_field_positive": lifts[pname]["n_splits_field_positive"],
            "prediction_held": bool(inside),
            "how_far_outside": None if inside else round(
                float(measured - PREDICTED["field_lift_low"]), 6),
            "reseed_floor_for_scale": 0.0026,
            "the_reading": (
                "the screen said no material field lift and was right that there "
                "is none, and wrong about what happens instead. It predicted a "
                "null between -0.002 and +0.001; the family costs "
                f"{measured:+.4f} on {lifts[pname]['n_splits_field_positive']} of "
                "12 splits, more than twice the reseed floor below the bottom of "
                "the predicted range. A family can be actively harmful and no "
                "internal statistic of it said so"),
            "and_the_solve_loses_too": (
                "the new part. On the two earlier families the solve gained what "
                "the field did not, +0.0011 and +0.0038, which is what made "
                "'belongs to a solve' a meaningful category. Here the solve loses "
                f"{lifts[pname]['solve']:+.4f} as well, so the family is not "
                "collectible by either readout and the screen has no category for "
                "that"),
            "order_by_field_lift_alone": {
                "by_screen": by_screen,
                "by_measured_field_lift": by_field,
                "agree": by_screen == by_field,
                "reversed": by_screen == by_field[::-1],
                "n_families": len(added),
                "why_the_count_matters": (
                    "three points. A reversal is one ordering in six by chance, so "
                    "this is suggestive of an inverted screen and does not "
                    "establish one"),
            },
        }

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
        "interaction_with_the_deployed_bus": cross or None,
        "why_the_cross_term_was_added": (
            "the first version screened only a family's internal interaction, which "
            "says what its own tables could know and nothing about whether the "
            "deployed bus already knows it. Two families with equal internal "
            "interaction can differ entirely in how much of it is new, and the union "
            "attachment forms no straddling tables at all, so a redundant family "
            "contributes tables whose content is already present for the fan-out to "
            "decorrelate away") if cross else None,
        "prospective_test": {
            "family": "graph invariants 225",
            "also_screened_unexpanded": "graph invariants 15",
            "why_it_exists": (
                "the screen was fitted to three families after the fact, so it is "
                "a hypothesis. This family is built to the rule the screen "
                "implies -- fifteen different invariants of one graph rather than "
                "one invariant of fifteen graphs -- and its interaction is "
                "measured here before its field lift is measured anywhere"),
            "screen_value": (fam.get("graph invariants 225") or {}).get(
                "second_order_over_all_pairs"),
            "screen_value_unexpanded": (fam.get("graph invariants 15") or {}).get(
                "second_order_over_all_pairs"),
            "what_the_screen_predicts": (
                "a mean pairwise interaction of the order of the deployed bank's "
                "+1.06e-05 means the counting field should collect this family and "
                "the union attachment should lift. A negative one like the "
                "asymmetry family's -9.69e-06 means it should not, and the lift "
                "should appear in a linear solve instead"),
            "field_lift_measured": verdict is not None,
            "outcome": verdict,
        } if "graph invariants 225" in fam else None,
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
            "and_that_fourth_family_has_now_been_measured": {
                "orders_them_correctly_with_it_included": bool(agrees_prospective),
                "the_share_does_with_it_included": bool(agrees_share_prospective),
                "families_the_statistic_was_chosen_on": fitted_on,
                "families_with_a_measured_lift_now": have,
                "see": "prospective_test.outcome",
            } if verdict is not None else None,
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
    print("\n  absolute mean pairwise interaction (families with a measured lift "
          "first, then the ones awaiting one):")
    for k in order_by_abs + [x for x in fam if x not in order_by_abs]:
        flag = "" if k in order_by_abs else "   <- no field lift measured yet"
        print(f"    {k:22s} "
              f"{fam[k]['second_order_over_all_pairs']['mean']:+.3e}{flag}")
    print(f"\n  by share:            {order_by_share}")
    print(f"  by absolute 2nd order: {order_by_abs}")
    print(f"  by measured lift:      {order_by_lift}")
    print(f"  on the {len(fitted_on)} families the statistic was chosen on:")
    print(f"    the preregistered share orders them correctly: {agrees}")
    print(f"    the absolute term does:                        {agrees_abs}")
    if verdict is not None:
        o = verdict["order_by_field_lift_alone"]
        print(f"\n  the prospective family is now measured, so it enters the check:")
        print(f"    predicted field lift  "
              f"[{PREDICTED['field_lift_low']:+.4f}, "
              f"{PREDICTED['field_lift_high']:+.4f}]")
        print(f"    measured field lift   "
              f"{verdict['measured_field_lift']:+.4f} on "
              f"{verdict['n_splits_field_positive']}/12"
              f"   -> prediction held: {verdict['prediction_held']}")
        print(f"    measured solve lift   "
              f"{verdict['measured_solve_lift']:+.4f}"
              f"   (the earlier two families gained here; this one does not)")
        print(f"    absolute term orders all {len(have)} correctly: "
              f"{agrees_prospective}")
        print(f"    on field lift alone over the {o['n_families']} appended "
              f"families: agree={o['agree']} reversed={o['reversed']}")
    if a.write:
        print(f"\nwrote {out.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
