#!/usr/bin/env python3.12
"""A global fit plus a damped per-region correction. Training folds only.

What the previous attempt got wrong, and what its own controls said
-------------------------------------------------------------------
ROUTED_MULTIPLICITIES.json gave each region its own independent solve and lost
badly: -0.0650 with an informed router, -0.0342 with a random one, -0.0188 with a
chain-size router, none positive on any of twelve splits. The ordering is the
useful part and it is about the metric rather than about routing.

The metric is per-unit ROC-AUC, so it reads only the ranking of residues within a
chain. A wire's digit is a within-chain rank quartile, so routing on one cuts
every chain into four regions and hands the residues of a single chain four
different weight vectors; their scores are then not comparable, which is exactly
what the metric measures. Random routing scrambles a chain the same way. Chain-size
routing does not -- it assigns whole chains to regions -- and it was the least
damaged of the three, at -0.0188.

That residue is not a routing failure. It is what a solve costs when it sees a
quarter of the rows. Every region in that design refits all 5,152 multiplicities
from N/4 rows, and BANK_TRUNCATION.json already established that this bank needs
its size.

The construction here
---------------------
Keep the global solve on all the rows, and let each region contribute only a
correction to it:

    w        = (S + lambda I)^-1 (mu_1 - mu_0)          over the whole fit half
    delta_g  = (S_g + lambda I)^-1 (d_g - S_g w)        over region g's rows
    m_g      = round( (w + delta_g) / peak * cap )

with ``lambda`` scaled to the *global* trace in both solves. That single choice is
what makes the correction damp itself: a region with few rows has a small S_g, so
the global-scale ridge dominates its system and delta_g goes to zero. There is no
shrinkage coefficient to choose and nothing is tuned. By construction the arm
degrades to the global solve when no regional structure exists, which is the
property the flat per-region design lacked and the reason it could lose 0.065.

``d_g - S_g w`` is the residual of the global direction inside region g. If the
global weights already separate that region, the residual is near zero and so is
the correction.

Routers, all constant within a chain
------------------------------------
Every router here assigns whole chains, so a chain's residues share one weight
vector and their scores stay comparable. That is not a preference, it is forced by
the metric.

``chain size``       length quartile. The one per-chain router already measured.
``chain polarity``   quartile of the fraction of residues in the charged and
                     polar classes, a composition property of the whole receptor.
``random chains``    whole chains assigned at a fixed seed. The control: R regions
                     is also R times the parameters, and if random per-chain
                     routing gains as much then the gain is capacity.
``global``           R = 1, which is the deployed detector and must reproduce it.

Two region counts are run, four and two, because the data cost per region falls
with R and the previous design could not separate the cost from the idea.

What would falsify it
---------------------
Any informed router failing to beat both the global arm and the random per-chain
router. The five earlier sweeps then stand as a complete saturation result: every
parameter of this construction, including whether the multiplicity assignment is
global, has been measured at its optimum.

Training folds only. No test residue and no external unit is read.

Usage: PYTHONPATH=src:tools python3.12 tools/hierarchical_multiplicities.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bank_truncation import table_outputs  # noqa: E402
from composition_wires import CLASSES, class_of_code  # noqa: E402
from expand_invariant_bank import SEED  # noqa: E402
from quantisation_ladder import compile_at, offsets_at  # noqa: E402
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

SCHEMA = "geoaudit.hierarchical_multiplicities.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
CODES = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
OUT = ROOT / "results/architecture_sweep/HIERARCHICAL_MULTIPLICITIES.json"
ROUTER_SEED = 20260801
ROUTERS = ("chain size", "chain polarity", "random chains")


def scatter_and_delta(D, y, tables, offsets, n_levels, frac, rows):
    """Row-normalised ``S`` and ``mu_1 - mu_0`` over a row subset.

    Normalised, as the deployed fan-out does it, and the damping is applied to the
    ridge instead. Two earlier attempts got this wrong in opposite directions and
    the ``|correction| / |global|`` diagnostic caught both.

    Attempt one used the normalised scatter with a global-scale ridge and expected
    a data-poor region to be ridge-dominated. It is not: normalising by the row
    count removes precisely the row-count dependence that would have made it so,
    and the correction came out at 1.6 times the norm of the global vector.

    Attempt two removed the normalisation. That is worse, because the right-hand
    side ``mu_1 - mu_0`` is a mean difference and does not scale with rows while a
    raw scatter does. The global solution then scales as 1/N and a region's as
    1/n_g, so the correction is larger than the global vector by exactly the
    region count: measured 3.1 at R = 4.

    The units have to match, so the scatter stays normalised and the damping moves
    to the ridge, inflated by N / n_g at the call site.
    """
    K = len(tables)
    Dr, yr = D[rows], y[rows]
    pos = yr == 1
    n1, n0 = int(pos.sum()), int(len(yr) - int(pos.sum()))
    if n1 == 0 or n0 == 0:
        return np.zeros((K, K)), np.zeros(K), 0
    s1 = np.zeros(K)
    s0 = np.zeros(K)
    for a, b, v in table_outputs(Dr, tables, offsets, frac, n_levels):
        p = pos[a:b]
        s1 += v[p].sum(0)
        s0 += v[~p].sum(0)
    mu1, mu0 = s1 / n1, s0 / n0
    S = np.zeros((K, K))
    for a, b, v in table_outputs(Dr, tables, offsets, frac, n_levels):
        p = pos[a:b]
        c = np.where(p[:, None], v - mu1, v - mu0)
        S += c.T @ c
    return S / max(len(yr) - 2, 1), mu1 - mu0, len(yr)


def integerise(w: np.ndarray) -> np.ndarray:
    peak = float(np.abs(w).max())
    if peak <= 0:
        return np.zeros(len(w), dtype=np.int64)
    return np.round(w / peak * FAN_OUT_CAP).astype(np.int64)


def score_routed(D, tables, offsets, frac, mults, region, n_levels):
    out = np.empty(D.shape[0], dtype=np.float64)
    M = np.stack([m.astype(np.float64) for m in mults])
    for a, b, v in table_outputs(D, tables, offsets, frac, n_levels):
        out[a:b] = np.einsum("rk,rk->r", v, M[region[a:b]])
    return out


def quartile_of(values: np.ndarray, n_regions: int) -> np.ndarray:
    """Rank-based bands of a per-chain quantity, equal in count."""
    order = np.argsort(values, kind="stable")
    band = np.empty(len(values), dtype=np.int64)
    edges = np.linspace(0, len(values), n_regions + 1).astype(int)
    for r in range(n_regions):
        band[order[edges[r]:edges[r + 1]]] = r
    return band


def chain_routers(n_res, codes, n_regions: int, rng) -> dict[str, np.ndarray]:
    """Per-chain region labels, expanded to residues.

    Every router assigns a whole chain, so a chain's residues share one weight
    vector. Forced by the metric, which ranks within a chain.
    """
    lens = np.asarray([int(n) for n in n_res], dtype=np.int64)
    cls = class_of_code()[codes]
    polar_idx = [i for i, k in enumerate(CLASSES)
                 if k in ("polar", "positive", "negative")]
    frac_polar = np.empty(len(lens))
    off = 0
    for i, n in enumerate(lens):
        blk = cls[off:off + int(n)]
        frac_polar[i] = float(np.isin(blk, polar_idx).mean())
        off += int(n)
    return {
        "chain size": np.repeat(quartile_of(lens.astype(float), n_regions),
                                lens),
        "chain polarity": np.repeat(quartile_of(frac_polar, n_regions), lens),
        "random chains": np.repeat(
            rng.integers(0, n_regions, size=len(lens)), lens),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=0)
    ap.add_argument("--regions", type=str, default="4,2")
    ap.add_argument("--routers", type=str, default="")
    ap.add_argument("--damping", type=str, default="1,4,16,64",
                    help="multipliers on the N/n_g ridge inflation. The whole "
                         "ladder is reported, not the best of it: the arm "
                         "interpolates between the global solve at infinite "
                         "damping and an independent per-region solve at none, "
                         "and the question is whether any interior point gains. "
                         "Reporting one chosen multiplier would be a fishing "
                         "expedition")
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args(argv)

    cdoc = json.loads(COUNTING.read_text())
    frozen = {int(k.split()[-2]): np.asarray(v, dtype=float)
              for k, v in cdoc["per_split"].items()}
    n_splits = a.splits or cdoc["protocol"]["n_splits"]
    region_counts = [int(s) for s in a.regions.split(",") if s.strip()]
    routers = ([s.strip() for s in a.routers.split(",") if s.strip()]
               if a.routers else list(ROUTERS))
    unknown = [r for r in routers if r not in ROUTERS]
    if unknown:
        raise SystemExit(f"unknown routers {unknown}; known {list(ROUTERS)}")

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    e = np.load(CODES, allow_pickle=False)
    if not np.array_equal(z["units"], e["units"]):
        raise SystemExit("the wide and expanded caches disagree about units")
    codes = e["codes"]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{x['pdb']}_{x['chain']}": x["cluster_id"] for x in entries}
    n_wires = int(W.shape[1])
    if n_wires not in frozen:
        raise SystemExit(f"frozen artifact has widths {sorted(frozen)}")

    tables = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                              PARTITION_SEED)
    offsets = offsets_at(tables, N_LEVELS)
    K = len(tables)
    t0 = time.perf_counter()
    D = chain_digits(np.asarray(W, dtype=np.float64), n_res)
    print(f"banded {n_wires} wires in {time.perf_counter() - t0:.0f}s; "
          f"{K} tables; regions {region_counts}; routers {routers}",
          flush=True)
    row = np.repeat(np.arange(len(n_res)), n_res)

    dampings = [float(s) for s in a.damping.split(",") if s.strip()]
    arms = ["global"] + [f"{r}, R={n}, d={d:g}" for n in region_counts
                         for r in routers for d in dampings]
    got: dict[str, list[float]] = {k: [] for k in arms}
    corrections: dict[str, list[dict]] = {}
    corr_ratio: dict[str, list[float]] = {k: [] for k in arms if k != "global"}

    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit, pick = is_fit[row], ~is_fit[row]
        fit_idx = np.flatnonzero(fit)
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])

        t1 = time.perf_counter()
        frac, _tc = compile_at(D[fit], y[fit], tables, offsets, N_LEVELS)
        S, delta, _n = scatter_and_delta(D, y, tables, offsets, N_LEVELS, frac,
                                         fit_idx)
        lam = RIDGE * float(np.trace(S)) / K + 1e-12
        Sr = S.copy()
        Sr.flat[::K + 1] += lam
        w_global = np.linalg.solve(Sr, delta)
        m_global = integerise(w_global)
        sc = apply_gate(score_routed(D[pick], tables, offsets, frac,
                                     [m_global],
                                     np.zeros(int(pick.sum()), dtype=np.int64),
                                     N_LEVELS), ctr[pick], n_pick)
        got["global"].append(float(per_unit_auc(sc, y[pick], n_pick)))
        print(f"  split {s + 1}/{n_splits}  global {got['global'][-1]:.4f}  "
              f"{got['global'][-1] - frozen[n_wires][s]:+.4f}  "
              f"{time.perf_counter() - t1:.0f}s", flush=True)

        for n_regions in region_counts:
            rng = np.random.default_rng(ROUTER_SEED + s)
            regs = chain_routers(n_res, codes, n_regions, rng)
            for rname in routers:
                reg = regs[rname]
                n_fit = int(fit.sum())
                # The per-region systems do not depend on the damping, only
                # their ridge does, so the scatters are formed once per router
                # and reused across the ladder. That is what makes a four-point
                # ladder cost what one point used to.
                per_region = []
                for g in range(n_regions):
                    rows = fit_idx[reg[fit_idx] == g]
                    Sg, dg, ng = scatter_and_delta(D, y, tables, offsets,
                                                   N_LEVELS, frac, rows)
                    per_region.append((Sg, dg - Sg @ w_global, int(ng)))
                for damp in dampings:
                    arm = f"{rname}, R={n_regions}, d={damp:g}"
                    t2 = time.perf_counter()
                    mults, sizes, norms = [], [], []
                    for g, (Sg, resid, ng) in enumerate(per_region):
                        Sgr = Sg.copy()
                        Sgr.flat[::K + 1] += lam * damp * (n_fit / max(ng, 1))
                        corr = np.linalg.solve(Sgr, resid)
                        mults.append(integerise(w_global + corr))
                        sizes.append(ng)
                        denom = float(np.linalg.norm(w_global))
                        norms.append(float(np.linalg.norm(corr) / denom)
                                     if denom > 0 else 0.0)
                    sc = apply_gate(score_routed(D[pick], tables, offsets, frac,
                                                 mults, reg[pick], N_LEVELS),
                                    ctr[pick], n_pick)
                    got[arm].append(float(per_unit_auc(sc, y[pick], n_pick)))
                    corr_ratio[arm].append(float(np.mean(norms)))
                    if s == 0:
                        corrections[arm] = [
                            {"region": g, "n_fit_rows": sizes[g],
                             "correction_norm_over_global_norm":
                                 round(norms[g], 6)}
                            for g in range(n_regions)]
                    print(f"      {arm:30s} {got[arm][-1]:.4f}  "
                          f"{got[arm][-1] - frozen[n_wires][s]:+.4f}  "
                          f"|corr|/|w| {np.mean(norms):.3f}  "
                          f"{time.perf_counter() - t2:.0f}s", flush=True)
                del per_region

    base = frozen[n_wires][:n_splits]

    def summarise(v):
        v = np.asarray(v)
        return {"mean": round(float(v.mean()), 6),
                "min": round(float(v.min()), 6),
                "max": round(float(v.max()), 6)}

    def compare(v, ref):
        d = np.asarray(v) - np.asarray(ref)
        return {"mean": round(float(d.mean()), 6),
                "min": round(float(d.min()), 6),
                "max": round(float(d.max()), 6),
                "n_splits_positive": int((d > 0).sum()),
                "n_splits": int(len(d)),
                "positive_on_every_split": bool((d > 0).all())}

    repro = {
        "arm": "global",
        "max_absolute_difference": round(
            float(np.abs(np.asarray(got["global"]) - base).max()), 9),
        "why_this_is_here": "the global arm is one region and is therefore the "
                            "deployed detector; it must return the frozen "
                            "numbers before any corrected arm is interpreted",
    }
    repro["reproduces_the_deployed_arm"] = bool(
        repro["max_absolute_difference"] < 1e-6)

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "the flat per-region solve lost 0.019 to 0.065 because each "
                    "region refit all 5,152 multiplicities from a fraction of "
                    "the rows, and because a within-chain router makes a chain's "
                    "scores incomparable under a within-chain metric. Does a "
                    "global fit with a damped per-region correction, routed by "
                    "whole chains, recover anything",
        "construction": {
            "global": "w = (S + lambda I)^-1 (mu_1 - mu_0) over all fit rows",
            "correction": "delta_g = (S_g + lambda (N/n_g) I)^-1 (d_g - S_g w) "
                          "over region g's rows, with both scatters row-"
                          "normalised as the deployed fan-out normalises them",
            "integerisation": "m_g = round((w + delta_g) / peak * cap)",
            "the_one_choice_that_matters": "the region's ridge is the global one "
                                           "inflated by N / n_g, so a region "
                                           "holding a quarter of the rows is "
                                           "regularised four times as hard. "
                                           "Monotone in how little data the "
                                           "region has, stated, with no "
                                           "coefficient to tune",
            "two_earlier_forms_that_were_wrong": (
                "a global-scale ridge on a row-normalised scatter does not damp "
                "at all, because normalising removes the row-count dependence "
                "the damping needed; measured correction 1.6x the global norm. "
                "Removing the normalisation is worse, because mu_1 - mu_0 does "
                "not scale with rows while a raw scatter does, so the correction "
                "exceeds the global vector by exactly the region count; measured "
                "3.1x at R=4. Both were caught by reporting "
                "|correction|/|global| rather than by inspection"),
            "why_it_cannot_lose_much": "with no regional structure the residual "
                                       "d_g - S_g w is near zero and the arm "
                                       "degrades to the global solve. That is "
                                       "the property the flat per-region design "
                                       "lacked, and the reason it could lose "
                                       "0.065",
            "still_integer_at_inference": True,
            "inference_cost": f"R integer vectors of {K} entries instead of one",
        },
        "why_every_router_is_per_chain": "the metric is per-unit ROC-AUC and "
                                         "reads only the ranking within a chain. "
                                         "A router that varies inside a chain "
                                         "gives its residues different weight "
                                         "vectors and their scores stop being "
                                         "comparable, which "
                                         "ROUTED_MULTIPLICITIES.json measured as "
                                         "-0.065 for the most within-chain "
                                         "router and -0.019 for the only "
                                         "per-chain one",
        "routers": {
            "chain size": "length quartile",
            "chain polarity": "quartile of the fraction of residues in the "
                              "polar, cationic and anionic classes; a "
                              "composition property of the whole receptor",
            "random chains": f"whole chains assigned at seed {ROUTER_SEED}+split. "
                             f"R regions is R times the parameters, so if random "
                             f"per-chain routing gains as much the gain is "
                             f"capacity and not structure",
        },
        "what_would_falsify_it": "any informed router failing to beat both the "
                                 "global arm and the random per-chain router. "
                                 "The earlier sweeps would then stand as a "
                                 "complete saturation result: every parameter of "
                                 "this construction, including whether the "
                                 "assignment is global, measured at its optimum",
        "held_fixed": {
            "n_wires": n_wires, "n_levels": N_LEVELS,
            "table_width": TABLE_WIDTH, "partition_rounds": PARTITION_ROUNDS,
            "partition_seed": PARTITION_SEED, "n_tables": K,
            "ridge": RIDGE, "fan_out_cap": FAN_OUT_CAP,
            "gate_radius": GATE_RADIUS, "gate_weight": GATE_WEIGHT,
            "cell_frequencies": "compiled once on the whole fit half and shared "
                                "by every region, so truncating the data per "
                                "region cannot thin a cell",
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "route_and_solve_on": "the fit half only",
            "evaluate_on": "the pick half",
            "metric": "mean per-unit ROC-AUC",
            "baseline_was_not_recomputed": str(COUNTING.relative_to(ROOT)),
        },
        "arms": {k: summarise(v) for k, v in got.items()},
        "deployed_arm_frozen": summarise(base),
        "minus_deployed": {k: compare(v, base) for k, v in got.items()},
        "minus_random_per_chain": {
            f"{r}, R={n}, d={d:g}": compare(
                got[f"{r}, R={n}, d={d:g}"],
                got[f"random chains, R={n}, d={d:g}"])
            for n in region_counts for r in routers for d in dampings
            if r != "random chains"
        },
        "damping_ladder": {
            "multipliers": dampings,
            "what_it_interpolates": "the arm becomes the global solve as the "
                                    "damping grows and an independent "
                                    "per-region solve as it falls, so the whole "
                                    "ladder is reported and not a chosen point. "
                                    "If the curve rises monotonically toward "
                                    "zero without crossing it, no interior "
                                    "point gains and the globality of the "
                                    "assignment is settled",
            "mean_correction_norm_over_global": {
                k: round(float(np.mean(v)), 4) for k, v in corr_ratio.items()},
            "crosses_zero_anywhere": bool(any(
                compare(got[k], base)["mean"] > 0 for k in got
                if k != "global")),
        },
        "correction_size_on_split_1": corrections,
        "reproduction_check": repro,
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
    for k in arms:
        c = doc["minus_deployed"][k]
        print(f"  {k:24s} {doc['arms'][k]['mean']:.6f}  {c['mean']:+.6f}  on "
              f"{c['n_splits_positive']}/{c['n_splits']}")
    print("\n  against the random per-chain router:")
    for k, c in doc["minus_random_per_chain"].items():
        print(f"    {k:24s} {c['mean']:+.6f}  on "
              f"{c['n_splits_positive']}/{c['n_splits']}")
    print(f"\n  global reproduces deployed: "
          f"{repro['reproduces_the_deployed_arm']} "
          f"(max |diff| {repro['max_absolute_difference']:.2e})")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
