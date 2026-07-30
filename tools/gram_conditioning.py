#!/usr/bin/env python3.12
"""Is the combination step the binding constraint? Training folds only.

The conjecture this settles
--------------------------
Four measurements now share one shape and no single explanation has been checked.

On the 645 deployed wires the counting field beats a ridge Fisher solve by
+0.0053 on 11 of 12 splits (IS_FISHER_A_CEILING.json). But on every column family
*appended* to those wires the ordering reverses: the wire-asymmetry columns give
the solve +0.0038 on 12/12 and the field -0.0007
(ANISOTROPIC_COUNTING_FIELD.json), and the composition columns give the solve
+0.0011 on 10/12 and the field -0.0009 (COMPOSITION_WIRES.json). Meanwhile
choosing pairings to maximise measured interaction raises the bank's mean
per-table interaction about twentyfold, that interaction survives to held-out
rows at a ratio of 1.04, and the field is still worse for it
(SELECTED_PAIRINGS.json).

So the field's advantage is not a property of the readout in general; it is a
property of the readout at the bank size and bank composition that ships. The one
mechanism consistent with all four is the combination step. The multiplicities
are the rounded solution of a single ridge system

    (S + lambda I) m = mu_1 - mu_0,   S the K x K within-class scatter of the
                                      table outputs, K = 5152 or more,

and ``integer_fanout``'s own docstring records the failure mode: "random pairs
drawn from the pair lattice repeat as the pool grows, the scatter goes
near-singular, and without it the direction chases the null space -- measured as
a fall from 0.7844 to 0.6846 when the pool went from 1032 to 1720 tables."

If that is what is happening then adding tables, or concentrating them on fewer
wires, must show up as a worse-conditioned S. This tool measures that directly.

What is measured
----------------
For each bank, on the fit half of each split, the scatter S is formed exactly as
``integer_fanout`` forms it and then characterised without being inverted:

``eigenvalue spectrum``   the symmetric eigenvalues of S, from which the
                          condition number, the effective rank at several
                          thresholds, and the fraction of trace in the leading
                          subspace are read. All are properties of S and none
                          requires the solve.
``ridge dominance``       what fraction of the diagonal the ridge contributes,
                          ``lambda / (lambda + median(diag S))``. If the ridge is
                          most of the diagonal then the solve is barely reading
                          the data.
``direction alignment``   the cosine between the solved direction and the class
                          mean difference. At perfect conditioning these
                          coincide up to scale; a small cosine means the inverse
                          has rotated the direction into the small-eigenvalue
                          subspace, which is precisely "chasing the null space".
``rounding loss``         the cosine between the real solution and its integer
                          rounding onto [-cap, cap]. This is the only part of the
                          combination step that is intrinsic to the integer
                          architecture rather than to the conditioning, and it
                          must be separated from it before conditioning is blamed.

Banks compared, all at the deployed quantisation, width and rounds:
the deployed draw, a redraw at another seed, the interaction-selected bank, and
the union bank over 645 + 76 composition columns. The prediction is an ordering:
interaction-selected worst conditioned, union next, the two random draws best and
indistinguishable from each other.

What a null here would mean
---------------------------
If conditioning does not order the banks the way their AUC does, then the
combination step is not the constraint and the four measurements above need a
different explanation. That is a useful outcome and the tool is written to make it
visible rather than to confirm the conjecture: the ordering is reported next to
the AUC ordering, and the artifact states whether they agree.

No claim is made that improving conditioning would improve AUC. That is a further
experiment and it is named in the artifact.

Training folds only. No test residue and no external unit is read.

Usage: PYTHONPATH=src:tools python3.12 tools/gram_conditioning.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from composition_wires import build_or_load, union_tables  # noqa: E402
from expand_invariant_bank import SEED  # noqa: E402
from quantisation_ladder import blocks_at, compile_at, offsets_at  # noqa: E402
from selected_pairings import (  # noqa: E402
    greedy_rounds,
    joint_counts,
    variance_terms,
)
from select_architecture_on_train import cluster_half_split  # noqa: E402

from pocket_bench.methods.table_bank import (
    N_LEVELS,
    chain_digits,
    partition_tables,
)
from pocket_bench.methods.table_field import (
    FAN_OUT_CAP,
    PARTITION_ROUNDS,
    PARTITION_SEED,
    RIDGE,
    TABLE_WIDTH,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.gram_conditioning.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
OUT = ROOT / "results/architecture_sweep/GRAM_CONDITIONING.json"
CONTROL_SEED = PARTITION_SEED + 1

# The AUC each bank scored, from the artifacts, so the conditioning ordering can
# be set beside the ordering it is meant to explain rather than beside a memory.
AUC_FROM_ARTIFACTS = {
    "deployed": ("results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json",
                 0.790120),
    "another seed": ("results/architecture_sweep/SELECTED_PAIRINGS.json",
                     0.787470),
    "interaction-selected": (
        "results/architecture_sweep/SELECTED_PAIRINGS.json", 0.787327),
    "union 645+76": ("results/architecture_sweep/COMPOSITION_WIRES.json",
                     0.789260),
}


def scatter_and_means(D, y, tables, offsets, frac, n_levels):
    """The K x K within-class scatter and the class means, as the deployed
    fan-out forms them. Copied rather than imported because ``table_bank`` is
    pinned by a code digest over eight files."""
    K = len(tables)
    s1 = np.zeros(K)
    s0 = np.zeros(K)
    pos = y == 1
    n1 = int(pos.sum())
    n0 = int(len(y) - n1)
    for a, b, v in blocks_at(D, tables, offsets, frac, n_levels):
        p = pos[a:b]
        s1 += v[p].sum(0)
        s0 += v[~p].sum(0)
    mu1, mu0 = s1 / max(n1, 1), s0 / max(n0, 1)
    S = np.zeros((K, K))
    for a, b, v in blocks_at(D, tables, offsets, frac, n_levels):
        p = pos[a:b]
        c = np.where(p[:, None], v - mu1, v - mu0)
        S += c.T @ c
    return S / max(len(y) - 2, 1), mu1, mu0


def characterise(S: np.ndarray, mu1: np.ndarray, mu0: np.ndarray) -> dict:
    K = S.shape[0]
    diag_median = float(np.median(np.diag(S)))
    lam = RIDGE * float(np.trace(S)) / K + 1e-12
    w = np.linalg.eigvalsh(S)
    w = np.clip(w, 0.0, None)
    tot = float(w.sum())
    top = np.sort(w)[::-1]
    csum = np.cumsum(top)

    Sr = S.copy()
    Sr.flat[::K + 1] += lam
    delta = mu1 - mu0
    m_real = np.linalg.solve(Sr, delta)
    peak = float(np.abs(m_real).max())
    m_int = (np.round(m_real / peak * FAN_OUT_CAP) if peak > 0
             else np.zeros(K))

    def cos(a, b):
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

    return {
        "n_tables": int(K),
        "eigenvalues": {
            "max": round(float(w.max()), 12),
            "min": round(float(w.min()), 12),
            "condition_number_unridged": (
                round(float(w.max() / w.min()), 3) if w.min() > 0
                else "singular"),
            "condition_number_with_ridge": round(
                float((w.max() + lam) / (w.min() + lam)), 3),
            "n_below_1e-12_times_max": int((w < 1e-12 * w.max()).sum()),
            "effective_rank_at_1e-6": int((w > 1e-6 * w.max()).sum()),
            "effective_rank_at_1e-3": int((w > 1e-3 * w.max()).sum()),
            "trace_fraction_in_top_1_percent_of_directions": round(
                float(csum[max(K // 100 - 1, 0)] / tot), 6) if tot > 0 else None,
        },
        "ridge_dominance": {
            "lambda": round(lam, 12),
            "median_diagonal": round(diag_median, 12),
            "lambda_share_of_diagonal": round(
                float(lam / (lam + diag_median)), 6) if diag_median > 0
            else None,
            "why_it_matters": "if the ridge is most of the diagonal the solve "
                              "is barely reading the scatter, and the "
                              "multiplicities approach a scaled copy of the "
                              "class mean difference",
        },
        "direction": {
            "cosine_solution_to_mean_difference": round(cos(m_real, delta), 6),
            "why_it_matters": "at perfect conditioning the solved direction and "
                              "the mean difference coincide up to scale. A "
                              "small cosine means the inverse has rotated the "
                              "direction into the small-eigenvalue subspace, "
                              "which is what 'chasing the null space' means",
        },
        "integer_rounding": {
            "cosine_rounded_to_real": round(cos(m_int, m_real), 6),
            "fan_out_cap": FAN_OUT_CAP,
            "n_multiplicities_saturated_at_cap": int(
                (np.abs(m_int) >= FAN_OUT_CAP).sum()),
            "n_multiplicities_rounded_to_zero": int((m_int == 0).sum()),
            "why_it_is_reported_separately": "rounding loss is intrinsic to the "
                                             "integer architecture and "
                                             "conditioning loss is not; "
                                             "blaming conditioning requires "
                                             "showing the rounding is not the "
                                             "cause",
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=3,
                    help="conditioning is a property of the bank and varies "
                         "little across splits; three is the default and the "
                         "spread across them is reported")
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args(argv)

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res = z["X"], z["y"], z["n_res_per"]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    n_wires = int(W.shape[1])

    C, _names = build_or_load()
    n_new = int(C.shape[1])
    tables_union, _tw, bank = union_tables(n_wires, n_new)

    t0 = time.perf_counter()
    D_narrow = chain_digits(np.asarray(W, dtype=np.float64), n_res)
    D_full = chain_digits(np.asarray(np.concatenate([W, C], axis=1),
                                     dtype=np.float64), n_res)
    print(f"banded both column sets in {time.perf_counter() - t0:.0f}s",
          flush=True)

    deployed = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                               PARTITION_SEED)
    control = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                              CONTROL_SEED)
    row = np.repeat(np.arange(len(n_res)), n_res)

    per_split: dict[str, list[dict]] = {}
    for s in range(a.splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit = is_fit[row]
        yf = y[fit]

        t1 = time.perf_counter()
        tot, pos = joint_counts(D_narrow[fit], yf)
        v_wire, v_pair = variance_terms(tot, pos, n_wires, float(yf.mean()),
                                        int(fit.sum()))
        selected, _rep = greedy_rounds(
            v_pair - v_wire[:, None] - v_wire[None, :], PARTITION_ROUNDS, True)
        del tot, pos, v_pair
        print(f"  split {s + 1}/{a.splits}  selection in "
              f"{time.perf_counter() - t1:.0f}s", flush=True)

        banks = {
            "deployed": (D_narrow, deployed),
            "another seed": (D_narrow, control),
            "interaction-selected": (D_narrow, selected),
            "union 645+76": (D_full, tables_union),
        }
        for name, (D, tb) in banks.items():
            off = offsets_at(tb, N_LEVELS)
            t2 = time.perf_counter()
            frac, _tc = compile_at(D[fit], yf, tb, off, N_LEVELS)
            S, mu1, mu0 = scatter_and_means(D[fit], yf, tb, off, frac,
                                            N_LEVELS)
            rec = characterise(S, mu1, mu0)
            per_split.setdefault(name, []).append(rec)
            e = rec["eigenvalues"]
            print(f"      {name:22s} K={rec['n_tables']:5d}  "
                  f"cond(ridged)={e['condition_number_with_ridge']:.3g}  "
                  f"effrank1e-6={e['effective_rank_at_1e-6']:5d}  "
                  f"cos(sol,mu)={rec['direction']['cosine_solution_to_mean_difference']:.4f}"
                  f"  {time.perf_counter() - t2:.0f}s", flush=True)
            del S

    def spread(name, path):
        vals = [_dig(r, path) for r in per_split[name]]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            return None
        return {"mean": round(float(np.mean(vals)), 6),
                "min": round(float(np.min(vals)), 6),
                "max": round(float(np.max(vals)), 6)}

    def _dig(rec, path):
        node = rec
        for k in path:
            node = node[k]
        return node

    summary = {}
    for name in per_split:
        summary[name] = {
            "n_tables": per_split[name][0]["n_tables"],
            "auc_from_artifact": AUC_FROM_ARTIFACTS.get(name, (None, None))[1],
            "auc_source": AUC_FROM_ARTIFACTS.get(name, (None, None))[0],
            "condition_number_with_ridge": spread(
                name, ("eigenvalues", "condition_number_with_ridge")),
            "effective_rank_at_1e-6": spread(
                name, ("eigenvalues", "effective_rank_at_1e-6")),
            "trace_fraction_in_top_1_percent": spread(
                name, ("eigenvalues",
                       "trace_fraction_in_top_1_percent_of_directions")),
            "lambda_share_of_diagonal": spread(
                name, ("ridge_dominance", "lambda_share_of_diagonal")),
            "cosine_solution_to_mean_difference": spread(
                name, ("direction", "cosine_solution_to_mean_difference")),
            "cosine_rounded_to_real": spread(
                name, ("integer_rounding", "cosine_rounded_to_real")),
        }

    # Does the conditioning ordering agree with the AUC ordering? Stated as a
    # fact about two orderings and not as a causal claim.
    named = [n for n in summary if summary[n]["auc_from_artifact"] is not None]
    by_auc = sorted(named, key=lambda n: -summary[n]["auc_from_artifact"])
    by_cond = sorted(
        named, key=lambda n: summary[n]["condition_number_with_ridge"]["mean"])
    by_cos = sorted(
        named,
        key=lambda n: -summary[n]["cosine_solution_to_mean_difference"]["mean"])

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "on the deployed wires the counting field beats a ridge "
                    "solve by +0.0053, and on every column family appended to "
                    "them the ordering reverses while a bank with twenty times "
                    "the surviving interaction scores worse. Is the single "
                    "ridge solve that assigns the integer multiplicities the "
                    "binding constraint",
        "what_would_falsify_the_conjecture": "the conditioning ordering not "
                                             "matching the AUC ordering. Both "
                                             "orderings are printed below so "
                                             "that is visible rather than "
                                             "argued",
        "what_is_not_claimed": "that improving the conditioning would improve "
                               "AUC. Nothing here changes the combination rule; "
                               "this measures the rule that is deployed",
        "protocol": {
            "n_splits": a.splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + a.splits - 1}",
            "scatter_formed_on": "the fit half only, exactly as integer_fanout "
                                 "forms it",
            "ridge": RIDGE,
            "fan_out_cap": FAN_OUT_CAP,
            "why_three_splits_and_not_twelve": "conditioning is a property of "
                                               "the bank rather than of the "
                                               "labels, and the spread across "
                                               "splits is reported so a reader "
                                               "can see whether three was "
                                               "enough",
        },
        "bank_construction": bank,
        "summary": summary,
        "orderings": {
            "by_auc_descending": by_auc,
            "by_condition_number_ascending": by_cond,
            "by_direction_cosine_descending": by_cos,
            "auc_agrees_with_conditioning": by_auc == by_cond,
            "auc_agrees_with_direction_cosine": by_auc == by_cos,
        },
        "per_split": per_split,
    }

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    print(f"\n  by AUC        : {by_auc}")
    print(f"  by conditioning: {by_cond}")
    print(f"  by cos(sol,mu) : {by_cos}")
    print(f"  AUC agrees with conditioning: "
          f"{doc['orderings']['auc_agrees_with_conditioning']}")
    print(f"  AUC agrees with direction cosine: "
          f"{doc['orderings']['auc_agrees_with_direction_cosine']}")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
