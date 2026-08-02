#!/usr/bin/env python3.12
"""Score every deployed and development detector against all three published
baselines on the official CryptoBench fold, in one pass, on one residue universe.

Why this tool exists
--------------------
Three baselines are reproduced in this repository — P2Rank 2.5.1, the CryptoBench
pLM-NN readout, and PocketMiner — and every comparison so far has been written
against exactly one of them, by a tool that scored one detector. The consequence
is that ``seam_geometry_field`` has a published number against pLM-NN and *no*
number against P2Rank or PocketMiner, while ``table_field`` has the reverse. A
reader cannot assemble the missing cells because the per-unit values were never
persisted together on one universe.

This scores everything once, caches the per-residue vectors, and computes every
paired difference on the intersection where both sides are defined.

What is paired, and on what
---------------------------
``AGENTS.md`` §4: two means over different subsets are not a difference. Every
paired statistic here is computed only on units where *both* sides produced a
finite metric, and ``n_paired`` is recorded beside each difference. Per-unit
ROC-AUC is undefined when a chain has no positive or no negative residue; PR-AUC
is defined wherever ROC-AUC is.

A second, differently-powered statistic is reported beside the per-unit mean:
the **pooled residue read**, one ranking over every residue of every unit. Raw
scores are not commensurable across chains — P2Rank emits ligandability in
[0,1], the counting field emits an integer-weighted table sum whose spread
depends on chain length — so the pooled read ranks each chain's residues within
that chain first and pools the rank fractions. That is stated here because a
pooled AUC over raw scores would silently measure score calibration instead of
ranking quality.

Both statistics answer different questions. The per-unit mean asks "on a typical
structure, which method ranks better"; the pooled read asks "over all residues a
reader might triage, which ranking is better". A method can win one and lose the
other, and when that happens it is the finding, not an error.

``clinical_grade`` is false. The official fold has been read many times; this is
a development read and inherits that status. It re-reads already-committed
baseline scores and re-scores our own detectors; it fits nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.grand_baseline_read.v1"
N_JOBS = min(9, os.cpu_count() or 4)
CACHE_DIR = ROOT / "results/official_fold/_method_scores"
OUT = ROOT / "results/official_fold/GRAND_BASELINE_READ.json"

# Detectors scored by this repository's own code.
#
# The `_r14` variants apply the spatial gate at 14 A instead of the deployed
# 18 A. That radius is **not** chosen here: `GATE_BY_STRATUM.json` selected it on
# twelve cluster-disjoint halves of the *training* fold, with a both-directions
# replication check in which each half of the splits independently chose the
# same arm, at a paired +0.0025 (p = 1.15e-04) uniform across all four
# pocket-size strata. Scoring it here is one read of an already-made decision,
# not a sweep over the test fold, and no other radius is scored.
#
# It was never deployed in the frozen sibling for a reason that no longer
# applies: changing the gate would have made that repository's confirmatory
# external read a result about a detector that no longer existed. That paper is
# frozen and this line continues in the transfer atlas, so the change is now
# free to make and is recorded in docs/DECISIONS.md.
OURS = ("table_field", "geometry_field", "seam_geometry_field",
        "geometry_field_r14", "seam_geometry_field_r14")

# variant name -> (module to score with, gate radius in angstrom)
GATE_VARIANTS = {
    "geometry_field_r14": ("geometry_field", 14.0),
    "seam_geometry_field_r14": ("seam_geometry_field", 14.0),
}

# Published baselines, each already reproduced and pinned elsewhere in the repo.
BASELINES = ("p2rank", "plmnn", "pocketminer")

# Fusions are formed from cached per-residue vectors, not re-scored.
# equal-z: within-chain standardisation of each side, then an unweighted sum.
# No coefficient is fitted; "equal" is the whole rule.
FUSIONS = {
    "geo_seam_equalz": ("geometry_field", "seam_geometry_field"),
    "geo_seam_equalz_r14": ("geometry_field_r14", "seam_geometry_field_r14"),
}


def _positives(lab: dict) -> set[int]:
    if "cryptic_residues" in lab:
        return {int(x) for x in lab["cryptic_residues"]}
    if "labels" in lab and isinstance(lab["labels"], dict):
        return {int(k) for k, v in lab["labels"].items() if v}
    for key in ("positive_resseq", "binding_residues", "pocket_residues"):
        if key in lab:
            return {int(x) for x in lab[key]}
    return set()


def _score_one(args: tuple[str, dict]) -> tuple[str, str, dict]:
    """Run one detector on one unit and return its per-residue vector."""
    method, e = args
    from pocket_bench.methods import (  # noqa: PLC0415
        geometry_field,
        seam_geometry_field,
        table_field,
    )

    base, radius = GATE_VARIANTS.get(method, (method, None))
    mod = {"table_field": table_field, "geometry_field": geometry_field,
           "seam_geometry_field": seam_geometry_field}[base]
    if radius is not None:
        # Patch the constant the gate reads, in this worker process only. The
        # module that owns it is digest-pinned by compiled artifacts, so it is
        # not edited on disk; minimal blast radius near a pinned file is worth
        # more than tidiness. `apply_gate` reads the module global at call time.
        table_field.GATE_RADIUS = radius
    unit = f"{e['pdb']}_{e['chain']}"
    try:
        pred = mod.predict(ROOT / e["receptor_path"], pdb_id=e["pdb"],
                           chain=e["chain"])
        rs = pred.get("residue_scores")
        if not isinstance(rs, dict) or not rs:
            return method, unit, {"error": f"no scores; status={pred.get('status')} "
                                           f"err={pred.get('error')}"}
        return method, unit, {"residue_scores": {str(k): float(v)
                                                 for k, v in rs.items()}}
    except Exception as exc:  # noqa: BLE001
        return method, unit, {"error": f"{type(exc).__name__}: {exc}"}


def _load_or_score(entries: list[dict], methods: tuple[str, ...],
                   refresh: bool) -> dict[str, dict[str, dict]]:
    """Per-residue vectors for our detectors, cached to disk between runs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, dict]] = {}
    todo: list[tuple[str, dict]] = []
    for m in methods:
        p = CACHE_DIR / f"{m}.json"
        if p.exists() and not refresh:
            out[m] = json.loads(p.read_text())["units"]
            print(f"cache hit {m}: {len(out[m])} units", flush=True)
        else:
            out[m] = {}
            todo.extend((m, e) for e in entries)
    if not todo:
        return out

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futs = [ex.submit(_score_one, a) for a in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            m, unit, rec = fut.result()
            out[m][unit] = rec
            if i % 50 == 0:
                print(f"  scored {i}/{len(todo)} "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
    for m in methods:
        if out[m] and not (CACHE_DIR / f"{m}.json").exists() or refresh:
            (CACHE_DIR / f"{m}.json").write_text(json.dumps(
                {"schema": "geoaudit.method_residue_scores.v1",
                 "clinical_grade": False, "method": m,
                 "n_units": len(out[m]), "units": out[m]}) + "\n")
    return out


def _baseline_vectors(entries: list[dict]) -> dict[str, dict[str, dict]]:
    """Per-residue vectors for the three published baselines, as committed."""
    out: dict[str, dict[str, dict]] = {b: {} for b in BASELINES}

    p2 = json.loads((ROOT / "results/cryptobench_official/predictions/"
                     "p2rank.json").read_text())["units"]
    for unit, rec in p2.items():
        if rec.get("status") == "OK" and rec.get("residue_scores"):
            out["p2rank"][unit] = {"residue_scores": rec["residue_scores"]}

    plm = json.loads((ROOT / "results/baselines/PLMNN_SCORES.json").read_text())
    for u in plm["units"]:
        if u.get("scores"):
            out["plmnn"][u["unit_id"]] = {"residue_scores": u["scores"]}

    pm_dir = ROOT / "data/baselines/pocketminer"
    for e in entries:
        unit = f"{e['pdb']}_{e['chain']}"
        p = pm_dir / f"{unit}.json"
        if p.exists():
            d = json.loads(p.read_text())
            if d.get("residue_scores"):
                out["pocketminer"][unit] = {"residue_scores": d["residue_scores"]}
    return out


def _rankfrac(v: np.ndarray) -> np.ndarray:
    """Within-chain rank fraction in [0,1]; ties broken by position, stably."""
    n = len(v)
    if n <= 1:
        return np.zeros(n)
    return np.argsort(np.argsort(v, kind="stable"), kind="stable") / (n - 1)


def _zwithin(v: np.ndarray) -> np.ndarray:
    sd = float(np.std(v))
    return (v - float(np.mean(v))) / sd if sd > 0 else np.zeros_like(v)


def _paired(a: dict[str, float], b: dict[str, float], seed: int,
            n_boot: int) -> dict:
    """Bootstrap the paired difference on the intersection of two coverages."""
    units = sorted(set(a) & set(b))
    d = np.asarray([a[u] - b[u] for u in units], dtype=float)
    if len(d) < 8:
        return {"n_paired": len(d), "mean_delta": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
    # Exact two-sided sign test over units where the two differ at all.
    from math import comb  # noqa: PLC0415
    m = n_pos + n_neg
    k = min(n_pos, n_neg)
    p_sign = (min(1.0, 2.0 * sum(comb(m, i) for i in range(k + 1)) / (2 ** m))
              if 0 < m <= 1000 else None)
    return {
        "n_paired": len(d),
        "n_a_only": len(set(a) - set(b)),
        "n_b_only": len(set(b) - set(a)),
        "mean_delta": float(d.mean()),
        "median_delta": float(np.median(d)),
        "ci95": [float(np.percentile(boots, 2.5)),
                 float(np.percentile(boots, 97.5))],
        "excludes_zero": bool(np.percentile(boots, 2.5) > 0
                              or np.percentile(boots, 97.5) < 0),
        "n_ahead": n_pos,
        "n_behind": n_neg,
        "sign_test_p": p_sign,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--refresh", action="store_true",
                    help="re-score our detectors instead of using the cache")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    man = json.loads((ROOT / "data/cryptobench_apo/"
                      "official_manifest.json").read_text())
    entries = man["entries"]

    labels: dict[str, set[int]] = {}
    for e in entries:
        unit = f"{e['pdb']}_{e['chain']}"
        labels[unit] = _positives(json.loads((ROOT / e["label_path"]).read_text()))

    vec = _load_or_score(entries, OURS, args.refresh)
    vec.update(_baseline_vectors(entries))

    # Fusions, formed on the intersection of their parents' residue universes.
    for name, (l, r) in FUSIONS.items():
        fused: dict[str, dict] = {}
        for unit in set(vec[l]) & set(vec[r]):
            L, R = vec[l][unit], vec[r][unit]
            if "residue_scores" not in L or "residue_scores" not in R:
                continue
            keys = sorted(set(L["residue_scores"]) & set(R["residue_scores"]),
                          key=int)
            if len(keys) < 4:
                continue
            a = _zwithin(np.asarray([L["residue_scores"][k] for k in keys]))
            b = _zwithin(np.asarray([R["residue_scores"][k] for k in keys]))
            fused[unit] = {"residue_scores": {k: float(x)
                                              for k, x in zip(keys, a + b)}}
        vec[name] = fused

    all_methods = list(OURS) + list(FUSIONS) + list(BASELINES)

    # ---- per-unit metrics -------------------------------------------------
    roc: dict[str, dict[str, float]] = {m: {} for m in all_methods}
    prc: dict[str, dict[str, float]] = {m: {} for m in all_methods}
    pooled: dict[str, list[np.ndarray]] = {m: [] for m in all_methods}
    pooled_y: dict[str, list[np.ndarray]] = {m: [] for m in all_methods}
    errors: dict[str, list[str]] = {m: [] for m in all_methods}

    for e in entries:
        unit = f"{e['pdb']}_{e['chain']}"
        pos = labels[unit]
        for m in all_methods:
            rec = vec[m].get(unit)
            if not rec or "residue_scores" not in rec:
                errors[m].append(unit)
                continue
            rs = rec["residue_scores"]
            keys = sorted(rs, key=int)
            s = np.asarray([float(rs[k]) for k in keys], dtype=float)
            y = np.asarray([1 if int(k) in pos else 0 for k in keys], dtype=int)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            roc[m][unit] = float(roc_auc_score(y, s))
            prc[m][unit] = float(average_precision_score(y, s))
            pooled[m].append(_rankfrac(s))
            pooled_y[m].append(y)

    summary = {}
    for m in all_methods:
        r = np.asarray(list(roc[m].values()))
        p = np.asarray(list(prc[m].values()))
        py = np.concatenate(pooled_y[m]) if pooled_y[m] else np.zeros(0)
        ps = np.concatenate(pooled[m]) if pooled[m] else np.zeros(0)
        summary[m] = {
            "n_units_scored": len(r),
            "n_units_missing": len(errors[m]),
            "mean_per_unit_roc_auc": float(r.mean()) if len(r) else None,
            "mean_per_unit_pr_auc": float(p.mean()) if len(p) else None,
            "pooled_residue_roc_auc_on_rank_fractions":
                float(roc_auc_score(py, ps)) if py.sum() else None,
            "n_residues_pooled": int(len(py)),
        }

    # ---- paired differences ----------------------------------------------
    pairs: dict[str, dict] = {}
    for i, m in enumerate(list(OURS) + list(FUSIONS)):
        for j, b in enumerate(BASELINES):
            key = f"{m}_minus_{b}"
            pairs[key] = {
                "per_unit_roc_auc": _paired(roc[m], roc[b],
                                            args.seed + 7 * i + j, args.n_boot),
                "per_unit_pr_auc": _paired(prc[m], prc[b],
                                           args.seed + 101 + 7 * i + j,
                                           args.n_boot),
            }
    # Head-to-head among the baselines themselves, so a reader can calibrate
    # how large a resolvable difference is on this fold at all.
    for i, (a, b) in enumerate((("plmnn", "p2rank"), ("plmnn", "pocketminer"),
                                ("p2rank", "pocketminer"))):
        pairs[f"{a}_minus_{b}"] = {
            "per_unit_roc_auc": _paired(roc[a], roc[b], args.seed + 500 + i,
                                        args.n_boot),
        }

    out = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": True,
        "why_not_confirmatory": (
            "the official fold has been read many times during development; "
            "this tool fits nothing and re-reads committed baseline scores, "
            "but it is a development read and is labelled as one"
        ),
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        "n_units_in_manifest": len(entries),
        "n_units_in_manifest_definition": (
            "rows of data/cryptobench_apo/official_manifest.json, one per "
            "(pdb, chain) single-chain apo receptor in CryptoBench's own "
            "mmseqs2 10% test fold; not a count of PDB entries, since one "
            "entry can contribute more than one chain"
        ),
        "n_units_scored_definition": (
            "units where the method returned a finite per-residue vector AND "
            "the chain has at least one positive and one negative residue, so "
            "that ROC-AUC is defined; units failing either are excluded from "
            "that method's mean and from every pairing that involves it"
        ),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_jobs": N_JOBS,
        "pooled_read_definition": (
            "each chain's residues are replaced by their within-chain rank "
            "fraction in [0,1] before pooling, because raw scores from "
            "different methods and different chain lengths are not "
            "commensurable; the pooled number therefore measures ranking "
            "quality, not calibration"
        ),
        "fusion_definition": {
            name: (f"within-chain z({l}) + within-chain z({r}), unweighted; "
                   "no coefficient is fitted")
            for name, (l, r) in FUSIONS.items()
        },
        "summary": summary,
        "paired": pairs,
        "per_unit_roc_auc": {m: roc[m] for m in all_methods},
        "per_unit_pr_auc": {m: prc[m] for m in all_methods},
        "units_missing_per_method": {m: errors[m] for m in all_methods
                                     if errors[m]},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", args.out)

    print(f"\n{'method':28s} {'perUnitROC':>11s} {'perUnitPR':>10s} "
          f"{'pooledROC':>10s}  n")
    for m in all_methods:
        s = summary[m]
        print(f"{m:28s} {s['mean_per_unit_roc_auc']:11.4f} "
              f"{s['mean_per_unit_pr_auc']:10.4f} "
              f"{s['pooled_residue_roc_auc_on_rank_fractions']:10.4f} "
              f" {s['n_units_scored']}")
    print(f"\n{'comparison':46s} {'delta':>9s} {'CI95':>22s} {'ahead/behind':>13s}"
          f" {'resolved':>9s}")
    for k, v in pairs.items():
        d = v["per_unit_roc_auc"]
        if d.get("mean_delta") is None:
            continue
        ci = d["ci95"]
        print(f"{k:46s} {d['mean_delta']:+9.4f} "
              f"[{ci[0]:+.4f},{ci[1]:+.4f}] "
              f"{d['n_ahead']:6d}/{d['n_behind']:<6d} "
              f"{'YES' if d['excludes_zero'] else 'no':>9s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
