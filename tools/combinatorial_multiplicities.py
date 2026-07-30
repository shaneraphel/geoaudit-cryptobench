#!/usr/bin/env python3.12
"""Can the last fitted linear object be replaced by a counting rule?

What is still fitted
--------------------
Everything in the inference path is integer arithmetic over table cells except
one step. ``integer_fanout`` forms the within-class scatter ``S`` over the K
per-table values, adds a ridge, solves ``S w = mu1 - mu0``, and rounds the
direction onto ``[-32, 32]``. That solve is a regularised Fisher discriminant. It
is the only place where the detector fits a real-valued linear object, and the
repository's stated direction is combinatorial rather than fitted, so it is worth
knowing what the solve is buying.

The precise question
--------------------
A per-table rule can see a table's own separation and its own variance. It cannot
see how two tables covary. The solve's entire additional content is the
off-diagonal of ``S``, which decorrelates tables that are counting the same thing
-- and 5,152 tables drawn from random partitions of 645 wires certainly do repeat
themselves, which is why the ridge is not cosmetic: without it the direction
chased the null space and the score fell from 0.7844 to 0.6846 as the pool grew.
So the question is not "is the solve good" but "is the off-diagonal load-bearing",
and it is answered by comparing arms that differ in nothing else.

The arms, from most fitted to least
-----------------------------------
  deployed ridge solve   the full K x K solve, rounded. Reproduction gate.
  diagonal solve         w = delta / (S_kk + lambda). Same formula with the
                         off-diagonal deleted, so the difference between this and
                         the arm above is exactly the correlation term.
  standardised delta     w = delta / sqrt(S_kk + lambda), an effect size rather
                         than a solve.
  rank bands             order tables by |standardised delta|, cut at quartiles,
                         assign magnitudes 4, 8, 16, 32 with the sign of delta.
                         No real number survives into inference: a table's weight
                         is decided by which quarter of the ordering it falls in.
  sign only              w = 32 * sign(delta). Every table counts equally and
                         only its direction is read.
  random signs           control. If this is not far worse than sign only, the
                         signs are not carrying what they appear to.

All six round to integers in [-32, 32] and all six leave inference as an integer
weighted sum over cells, so this is a question about how the weights are chosen
and not about what inference costs.

Falsification
-------------
If every rule loses to the deployed solve by more than the reseed noise this
repository has already measured -- a different pairing seed costs 0.0026 -- then
the off-diagonal is load-bearing and the linear solve stays, with a measurement
saying why rather than an assumption. If the diagonal arm matches, the solve is
carrying nothing that a per-table statistic cannot, and the last fitted object
can go.

Reading the arms in order matters more than any single one. The four rules step
down from a solve to an ordering, so where the accuracy falls off says which
ingredient was doing the work: the correlation, the variance, the magnitude, or
only the sign.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from expand_invariant_bank import SEED  # noqa: E402
from quantisation_ladder import blocks_at, compile_at, offsets_at, score_at  # noqa: E402
from select_architecture_on_train import cluster_half_split, per_unit_auc  # noqa: E402

from pocket_bench.methods.table_bank import (
    N_LEVELS,
    chain_digits,
    partition_tables,
)
from pocket_bench.methods.table_field import (
    FAN_OUT_CAP,
    GATE_RADIUS,
    GATE_WEIGHT,
    PARTITION_ROUNDS,
    PARTITION_SEED,
    RIDGE,
    TABLE_WIDTH,
    apply_gate,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.combinatorial_multiplicities.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
PAIRSEL = ROOT / "results/architecture_sweep/SELECTED_PAIRINGS.json"
OUT = ROOT / "results/architecture_sweep/COMBINATORIAL_MULTIPLICITIES.json"
SIGN_SEED = 20260803
# Magnitudes for the banded rule, one per quartile of |standardised delta|.
# Powers of two so a weight is a shift, and the top band is the deployed cap.
BANDS = (4, 8, 16, 32)
ARMS = ("deployed ridge solve", "diagonal solve", "standardised delta",
        "rank bands", "sign only", "random signs")


def scatter_delta(D, y, tables, offsets, frac):
    """``S`` before the ridge, and ``mu1 - mu0``.

    Split out of ``fanout_at`` so every arm reads the same two objects and the
    arms differ only in what they do with them. The ridge is added by the callers
    that want it, at the deployed rule, so no arm gets a quietly different one.
    """
    K = len(tables)
    s1 = np.zeros(K)
    s0 = np.zeros(K)
    pos = y == 1
    n1 = int(pos.sum())
    n0 = int(len(y) - n1)
    for a, b, v in blocks_at(D, tables, offsets, frac, N_LEVELS):
        p = pos[a:b]
        s1 += v[p].sum(0)
        s0 += v[~p].sum(0)
    mu1, mu0 = s1 / max(n1, 1), s0 / max(n0, 1)
    S = np.zeros((K, K))
    for a, b, v in blocks_at(D, tables, offsets, frac, N_LEVELS):
        p = pos[a:b]
        c = np.where(p[:, None], v - mu1, v - mu0)
        S += c.T @ c
    S /= max(len(y) - 2, 1)
    return S, mu1 - mu0


def to_integer(w: np.ndarray) -> np.ndarray:
    """Scale so the largest magnitude is the cap, then round. The deployed rule."""
    peak = float(np.abs(w).max())
    if peak <= 0:
        return np.zeros(len(w), dtype=np.int64)
    return np.round(w / peak * FAN_OUT_CAP).astype(np.int64)


def multiplicities(S: np.ndarray, delta: np.ndarray, rng) -> dict[str, np.ndarray]:
    """Every arm's integer weights, from one scatter and one mean difference."""
    K = S.shape[0]
    lam = RIDGE * float(np.trace(S)) / K + 1e-12
    diag = np.diag(S) + lam

    Sr = S.copy()
    Sr.flat[::K + 1] += lam
    out = {"deployed ridge solve": to_integer(np.linalg.solve(Sr, delta))}
    out["diagonal solve"] = to_integer(delta / diag)
    z = delta / np.sqrt(diag)
    out["standardised delta"] = to_integer(z)

    # Quartiles of |z|, so the assignment depends on the ordering and not on the
    # values. Ties go to the lower band by argsort position, which is arbitrary
    # but fixed by a stable sort rather than by the data.
    order = np.argsort(np.abs(z), kind="stable")
    band = np.empty(K, dtype=np.int64)
    edges = np.linspace(0, K, len(BANDS) + 1).astype(int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        band[order[lo:hi]] = BANDS[i]
    out["rank bands"] = (band * np.sign(delta)).astype(np.int64)

    out["sign only"] = (FAN_OUT_CAP * np.sign(delta)).astype(np.int64)
    out["random signs"] = (FAN_OUT_CAP
                           * rng.choice([-1, 1], size=K)).astype(np.int64)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args(argv)

    cdoc = json.loads(COUNTING.read_text())
    frozen = {int(k.split()[-2]): np.asarray(v, dtype=float)
              for k, v in cdoc["per_split"].items()}
    n_splits = a.splits or cdoc["protocol"]["n_splits"]

    # Read only from minus_deployed. Both that block and "arms" carry a row named
    # "another seed", and scanning both would leave the arm's AUC of 0.787 in a
    # field labelled as a difference of 0.0026 -- two orders out, and it would
    # have made every rule below look like it was inside the noise.
    reseed_noise = None
    if PAIRSEL.is_file():
        for name, v in (json.loads(PAIRSEL.read_text()).get("minus_deployed")
                        or {}).items():
            if "seed" in name.lower() and isinstance(v, dict) and "mean" in v:
                reseed_noise = abs(float(v["mean"]))

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    n_wires = int(W.shape[1])
    if n_wires not in frozen:
        raise SystemExit(f"frozen artifact has widths {sorted(frozen)}")
    base = frozen[n_wires][:n_splits]

    tables = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                              PARTITION_SEED)
    offsets = offsets_at(tables, N_LEVELS)
    t0 = time.perf_counter()
    D = chain_digits(np.asarray(W, dtype=np.float64), n_res)
    print(f"banded {n_wires} wires in {time.perf_counter() - t0:.0f}s; "
          f"{len(tables)} tables, cap {FAN_OUT_CAP}, bands {BANDS}", flush=True)

    row = np.repeat(np.arange(len(n_res)), n_res)
    rng = np.random.default_rng(SIGN_SEED)
    got: dict[str, list[float]] = {k: [] for k in ARMS}
    agree: dict[str, list[float]] = {k: [] for k in ARMS}

    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        t1 = time.perf_counter()

        frac, _tot = compile_at(D[fit], y[fit], tables, offsets, N_LEVELS)
        S, delta = scatter_delta(D[fit], y[fit], tables, offsets, frac)
        mults = multiplicities(S, delta, rng)
        ref = mults["deployed ridge solve"].astype(float)
        nref = float(np.linalg.norm(ref)) or 1.0

        for k in ARMS:
            m = mults[k]
            raw = score_at(D[pick], tables, offsets, frac, m, N_LEVELS)
            got[k].append(float(per_unit_auc(apply_gate(raw, ctr[pick], n_pick),
                                             y[pick], n_pick)))
            mf = m.astype(float)
            agree[k].append(float(mf @ ref / (np.linalg.norm(mf) or 1.0) / nref))

        print(f"  split {s + 1}/{n_splits}  deployed {got[ARMS[0]][-1]:.4f}  "
              f"frozen {base[s]:.4f}  {time.perf_counter() - t1:.0f}s",
              flush=True)
        for k in ARMS[1:]:
            print(f"      {k:22s} {got[k][-1]:.4f}  "
                  f"{got[k][-1] - got[ARMS[0]][-1]:+.4f}  "
                  f"cos {agree[k][-1]:+.3f}", flush=True)

    def summarise(v):
        v = np.asarray(v, dtype=float)
        return {"mean": round(float(v.mean()), 6),
                "min": round(float(v.min()), 6),
                "max": round(float(v.max()), 6)}

    def compare(v, ref_):
        d = np.asarray(v, dtype=float) - np.asarray(ref_, dtype=float)
        return {"mean": round(float(d.mean()), 6),
                "n_splits_positive": int((d > 0).sum()),
                "n_splits": int(len(d))}

    dep = np.asarray(got[ARMS[0]], dtype=float)
    repro = {
        "arm": ARMS[0],
        "frozen_source": str(COUNTING.relative_to(ROOT)),
        "max_absolute_difference": float(np.max(np.abs(dep - base))),
        "reproduces_the_deployed_arm": bool(np.allclose(dep, base, atol=2e-6)),
    }
    losses = {k: compare(got[k], got[ARMS[0]])["mean"] for k in ARMS[1:]}
    best_rule = max((k for k in ARMS[1:] if k != "random signs"),
                    key=lambda k: losses[k])

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "whether the one fitted linear object left in the inference "
                    "path -- the ridge solve integer_fanout rounds -- can be "
                    "replaced by a rule computed from each table on its own",
        "what_the_solve_uniquely_has": "the off-diagonal of the within-class "
                                       "scatter, which decorrelates tables that "
                                       "count the same thing. No per-table "
                                       "statistic can see it, and 5,152 tables "
                                       "drawn from random partitions of 645 "
                                       "wires do repeat themselves -- without "
                                       "the ridge the direction chased the null "
                                       "space and the score fell from 0.7844 to "
                                       "0.6846 as the pool grew",
        "why_the_arms_are_ordered": "each arm deletes one more ingredient: the "
                                    "correlation, then the variance, then the "
                                    "magnitude, then everything but the sign. "
                                    "Where the accuracy falls off says which "
                                    "ingredient was doing the work, which no "
                                    "single comparison would",
        "what_would_falsify_it": "the diagonal arm matching the deployed solve "
                                 "within reseed noise. That would say the "
                                 "off-diagonal carries nothing and the last "
                                 "fitted object can be removed",
        "reseed_noise_for_scale": reseed_noise,
        "reseed_noise_source": (str(PAIRSEL.relative_to(ROOT))
                                if reseed_noise is not None else None),
        "every_arm_is_integer_at_inference": True,
        "why_that_matters_here": "all six arms round onto [-32, 32] and inference "
                                 "stays an integer weighted sum over cells. This "
                                 "measures how the weights are chosen, not what "
                                 "inference costs",
        "held_fixed": {
            "n_wires": n_wires, "n_levels": N_LEVELS,
            "table_width": TABLE_WIDTH, "partition_rounds": PARTITION_ROUNDS,
            "partition_seed": PARTITION_SEED, "n_tables": len(tables),
            "ridge": RIDGE, "fan_out_cap": FAN_OUT_CAP,
            "gate_radius": GATE_RADIUS, "gate_weight": GATE_WEIGHT,
            "cell_counts": "identical across arms; only the multiplicities move",
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "compile_and_solve_on": "the fit half only",
            "evaluate_on": "the pick half",
            "metric": "mean per-unit ROC-AUC, gate applied as deployed",
            "bands": list(BANDS),
            "sign_seed": SIGN_SEED,
        },
        "arms": {k: summarise(v) for k, v in got.items()},
        "minus_deployed": {k: compare(v, got[ARMS[0]]) for k, v in got.items()},
        "cosine_with_the_deployed_direction": {
            k: round(float(np.mean(v)), 4) for k, v in agree.items()},
        "closest_rule": best_rule,
        "reproduction_check": repro,
        "per_split": {k: [round(x, 6) for x in v] for k, v in got.items()},
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

    print(f"\n  deployed ridge solve: {dep.mean():.6f}")
    for k in ARMS[1:]:
        c = doc["minus_deployed"][k]
        print(f"  {k:22s} {doc['arms'][k]['mean']:.6f}  {c['mean']:+.6f}  on "
              f"{c['n_splits_positive']}/{c['n_splits']}   "
              f"cos {doc['cosine_with_the_deployed_direction'][k]:+.3f}")
    if reseed_noise is not None:
        print(f"\n  reseed noise for scale: {reseed_noise:.4f}")
    print(f"\n  deployed arm reproduces the frozen numbers: "
          f"{repro['reproduces_the_deployed_arm']} "
          f"(max |diff| {repro['max_absolute_difference']:.2e})")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
